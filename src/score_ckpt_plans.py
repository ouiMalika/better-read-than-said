"""Replay the method-1 plans stored in a readout_ladder checkpoint (Table 1)."""
import re
import json
import argparse
import torch
from vsp_maze import load_level
from readout_ladder import MODELS, MOVES, optimal_set, ckpt_key
from gen_plans import episode

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--model', required=True, choices=list(MODELS))
    ap.add_argument('--levels', type=int, nargs='+', default=[3, 4, 5, 6, 7, 8])
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    cache = torch.load(a.ckpt, weights_only=False)
    items = [(p, lay) for lv in a.levels for _, lay, p in load_level(lv) if optimal_set(lay)]
    missing = [p for p, _ in items if ckpt_key(p) not in cache]
    if missing:
        print(f'  {len(missing)} instances missing from the checkpoint; scoring the rest')
        items = [(p, l) for p, l in items if ckpt_key(p) in cache]
    fm, ep, deaths, per = ([], [], {}, {})
    for p, lay in items:
        txt = cache[ckpt_key(p)].get('txt', '') or ''
        m = re.search('ANSWER:\\s*([UDLR]+)', txt, re.I) or re.search('\\b([UDLR]{2,})\\b', txt)
        plan = m.group(1).upper() if m else ''
        first = MOVES.index(plan[0]) if plan else -1
        won, death = episode(lay, plan) if plan else (False, 'unparsed')
        if death:
            deaths[death] = deaths.get(death, 0) + 1
        fm.append(first in optimal_set(lay))
        ep.append(won)
        d = per.setdefault(str(lay.level), {'n': 0, 'first_move': 0, 'episode': 0})
        d['n'] += 1
        d['first_move'] += int(fm[-1])
        d['episode'] += int(won)
    for d in per.values():
        d['first_move'] /= d['n']
        d['episode'] /= d['n']
    res = {'model': MODELS[a.model][0], 'position': 'hidden', 'n': len(items), 'prompt_mode': 'plan', 'max_new': 512, 'source': a.ckpt, 'incomplete': len(missing), 'first_move_agreement': sum(fm) / len(fm), 'episode_success': sum(ep) / len(ep), 'failure_modes': deaths, 'by_level': dict(sorted(per.items()))}
    with open(a.out, 'w') as fh:
        json.dump(res, fh, indent=2)
    print(f"{a.model}: first_move {res['first_move_agreement']:.3f}   EPISODE {res['episode_success']:.3f}   {deaths}   n={len(items)}")
if __name__ == '__main__':
    main()
