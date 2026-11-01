"""Generate method-1 plans, keep the text, and score first-move accuracy and episode success."""
import os
import re
import json
import argparse
import torch
from PIL import Image
from vsp_maze import load_level
from env import step
from readout_ladder import MODELS, MOVES, VSP2ENV, load_model, optimal_set, generate_text_batch, _prompts, ckpt_key
from vlm_remedy import INSTR
SERIAL_SUFFIX = {'serial': "Work through it in this order, writing each step on its own line.\n1. Row and column of the player.\n2. Row and column of the goal.\n3. Which of up/down/left/right leave the grid or step on a hole.\n4. Of the moves that remain, which one reduces the distance to the goal.\nThen give the full plan as a string of U/D/L/R after 'ANSWER:'.", 'geom': "Work through it in this order, writing each step on its own line.\n1. Row and column of the player.\n2. Row and column of the goal.\n3. The difference in rows and the difference in columns.\n4. The move that reduces the larger of those two differences.\nThen give the full plan as a string of U/D/L/R after 'ANSWER:'."}

def serial_prompt(lay, position, mode):
    ctx = f'The grid is {lay.level}x{lay.level}.'
    if position == 'given':
        ctx += f' You are at row {lay.start[0] + 1}, column {lay.start[1] + 1}.'
    base = INSTR.replace('Answer with EXACTLY ONE letter, one of U/D/L/R, and nothing else.', '')
    return base.strip() + '\n' + ctx + '\n' + SERIAL_SUFFIX[mode]

def episode(lay, plan):
    """Execute the plan under corrected VSP/FrozenLake semantics: an off-grid move is a no-op that still consumes a step, a hole is a terminal loss, the goal wins."""
    pos = lay.start
    for mv in plan:
        pos, ev = step(lay, pos, VSP2ENV[mv])
        if ev == 'hole':
            return (False, 'hole')
        if ev == 'goal':
            return (True, None)
    return (False, 'ran_out')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, choices=list(MODELS))
    ap.add_argument('--levels', type=int, nargs='+', default=[3, 4, 5, 6, 7, 8])
    ap.add_argument('--position', choices=['hidden', 'given'], default='hidden')
    ap.add_argument('--max-new', type=int, default=512)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--prompt-mode', choices=['plan', 'serial', 'geom'], default='plan', help="plan = the ladder's matched rung-1 prompt; serial = forced step-by-step decomposition (an intervention, not a matched control)")
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    model, processor, family, _ = load_model(a.model)
    items = [(p, lay) for lv in a.levels for _, lay, p in load_level(lv) if optimal_set(lay)]
    print(f'{len(items)} instances, batch={a.batch}', flush=True)
    ck = os.path.join(os.path.dirname(os.path.abspath(a.out)) or '.', f'.ckpt_plans_{a.model}_{a.position}_{a.prompt_mode}_mx{a.max_new}.pt')
    try:
        cache = torch.load(ck, weights_only=False) if os.path.exists(ck) else {}
    except Exception:
        cache = {}
    if cache:
        print(f'  resuming: {len(cache)} cached', flush=True)
    todo = [(p, l) for p, l in items if ckpt_key(p) not in cache]
    for c0 in range(0, len(todo), a.batch):
        chunk = todo[c0:c0 + a.batch]
        imgs = [Image.open(p).convert('RGB') for p, _ in chunk]
        qs = [serial_prompt(l, a.position, a.prompt_mode) if a.prompt_mode != 'plan' else _prompts(l, a.position)[1] for _, l in chunk]
        try:
            txts = generate_text_batch(model, processor, family, qs, imgs, max_new=a.max_new)
        except Exception as e:
            print(f'  WARNING generation failed: {type(e).__name__}: {e}', flush=True)
            continue
        for (p, _), t in zip(chunk, txts):
            cache[ckpt_key(p)] = t
        tmp = ck + '.tmp'
        torch.save(cache, tmp)
        os.replace(tmp, ck)
        print(f'  {min(c0 + a.batch, len(todo))}/{len(todo)}', flush=True)
    missing = [p for p, _ in items if ckpt_key(p) not in cache]
    if missing:
        print(f'  {len(missing)} instances missing after failed chunks; scoring the rest', flush=True)
        items = [(p, l) for p, l in items if ckpt_key(p) in cache]
    fm, ep, deaths, per = ([], [], {}, {})
    for p, lay in items:
        txt = cache.get(ckpt_key(p), '')
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
    res = {'model': MODELS[a.model][0], 'position': a.position, 'n': len(items), 'prompt_mode': a.prompt_mode, 'max_new': a.max_new, 'incomplete': len(missing), 'first_move_agreement': sum(fm) / len(fm), 'episode_success': sum(ep) / len(ep), 'failure_modes': deaths, 'by_level': dict(sorted(per.items())), 'plans': {ckpt_key(p): cache.get(ckpt_key(p), '') or '' for p, _ in items}}
    with open(a.out, 'w') as fh:
        json.dump(res, fh, indent=2)
    print(f"first_move {res['first_move_agreement']:.3f}   EPISODE {res['episode_success']:.3f}   {deaths}")
    print('by level:', {k: round(v['episode'], 3) for k, v in res['by_level'].items()})
    print(f'saved -> {a.out}')
if __name__ == '__main__':
    main()
