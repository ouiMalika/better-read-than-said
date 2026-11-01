"""Re-score an unexpected_step.py result offline."""
import os
import sys
import json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vsp_maze import load_level
from env import step, compute_distance_map
MOVES = ['U', 'D', 'L', 'R']
VSP2ENV = {'U': 'up', 'D': 'down', 'L': 'left', 'R': 'right'}

def optimal_set(lay, pos):
    dm = compute_distance_map(lay)
    if pos not in dm:
        return frozenset()
    d0 = dm[pos]
    return frozenset((i for i, m in enumerate(MOVES) if (lambda t: t[1] not in ('invalid', 'hole') and dm.get(t[0], 10 ** 9) == d0 - 1)(step(lay, pos, VSP2ENV[m]))))

def n_hazard_aware(lay, pos):
    """Moves that stay on the grid and avoid a hole: the paper's hazard-aware uniform support."""
    return sum((1 for m in MOVES if step(lay, pos, VSP2ENV[m])[1] not in ('invalid', 'hole'))) or 4

def layouts(levels):
    out = {}
    for lv in levels:
        for _, lay, p in load_level(lv):
            out[lv, os.path.basename(p)] = lay
    return out
MOVE_RE = __import__('re').compile('\\b([UDLR])\\b')

def first_move(txt):
    m = MOVE_RE.search((txt or '').strip())
    return m.group(1).upper() if m else None

def main(path):
    d = json.load(open(path))
    lays = layouts(d['levels'])
    sc = [r for r in d['per_instance'] if 'unx' in r]
    dropped = 0
    keep = []
    for r in sc:
        lay = lays[r['level'], r['id']]
        if tuple(r['planned_land']) == lay.goal or tuple(r['unexpected_land']) == lay.goal:
            dropped += 1
            continue
        keep.append(r)
    sc = keep
    d['dropped_landed_on_goal'] = dropped
    for r in sc:
        lay = lays[r['level'], r['id']]
        for arm, gk, mk, cell in (('ctrl', 'ctrl_gen', 'ctrl_move', 'planned_land'), ('unx', 'unx_gen', 'unx_move', 'unexpected_land')):
            if gk in r:
                r[mk] = first_move(r[gk])
            mv = r[mk]
            if mv is None:
                r[arm] = {}
                continue
            i = MOVES.index(mv)
            r[arm] = {'optimal_actual': i in optimal_set(lay, tuple(r[cell])), 'optimal_planned': i in optimal_set(lay, tuple(r['planned_land'])), 'equals_plan_next': mv == r['plan_next'] if r.get('plan_next') else None}
    rate = lambda arm, k: (lambda v: [sum(v) / len(v), len(v)] if v else None)([r[arm][k] for r in sc if r.get(arm, {}).get(k) is not None])
    for arm, pre in (('ctrl', 'ctrl'), ('unx', 'unx')):
        for k in ('optimal_actual', 'optimal_planned', 'equals_plan_next'):
            d[f'{pre}_{k}'] = rate(arm, k)
    both = [r for r in sc if r.get('ctrl_move') and r.get('unx_move')]
    same = sum((1 for r in both if r['ctrl_move'] == r['unx_move']))
    d['same_reply_in_both_arms'] = [same / len(both), len(both)] if both else None
    d['scored'] = len(sc)
    d['needs_gpu_if_plan_parser_changes'] = len(d['per_instance'])
    acc = {'ctrl': [[], []], 'unx': [[], []]}
    for r in sc:
        lay = lays[r['level'], r['id']]
        for arm, cell in (('ctrl', r['planned_land']), ('unx', r['unexpected_land'])):
            pos = tuple(cell)
            k = len(optimal_set(lay, pos))
            acc[arm][0].append(k / 4.0)
            acc[arm][1].append(k / n_hazard_aware(lay, pos))
    mean = lambda v: sum(v) / len(v) if v else None
    for arm in ('ctrl', 'unx'):
        d[f'chance_uniform4_{arm}'] = mean(acc[arm][0])
        d[f'chance_hazard_aware_{arm}'] = mean(acc[arm][1])
    for arm, key in (('ctrl', 'ctrl_optimal_actual'), ('unx', 'unx_optimal_actual')):
        obs = d[key][0] if d.get(key) else None
        d[f'margin_over_uniform4_{arm}'] = None if obs is None else obs - d[f'chance_uniform4_{arm}']
    json.dump(d, open(path, 'w'), indent=2)
    print(f"{d['model']}  scored={d['scored']}")
    for k in ('scored', 'dropped_landed_on_goal', 'same_reply_in_both_arms', 'ctrl_optimal_actual', 'chance_uniform4_ctrl', 'margin_over_uniform4_ctrl', 'unx_optimal_actual', 'chance_uniform4_unx', 'margin_over_uniform4_unx', 'unx_optimal_planned', 'ctrl_equals_plan_next', 'unx_equals_plan_next'):
        v = d.get(k)
        v = v[0] if isinstance(v, list) else v
        print(f'  {k:28s} {v:.3f}' if isinstance(v, float) else f'  {k:28s} {v}')
if __name__ == '__main__':
    for p in sys.argv[1:]:
        main(p)
