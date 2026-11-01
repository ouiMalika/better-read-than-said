"""Classify how replayed plans fail."""
import json
import argparse
from collections import Counter
from vsp_maze import load_level
from env import step, compute_distance_map, DELTAS
VSP2ENV = {'U': 'up', 'D': 'down', 'L': 'left', 'R': 'right'}

def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def classify(lay, letters):
    pos = lay.start
    if not letters:
        return {'mode': 'malformed', 'step': 0}
    for i, a in enumerate(letters):
        if a not in VSP2ENV:
            return {'mode': 'malformed', 'step': i}
        dr, dc = DELTAS[VSP2ENV[a]]
        target = (pos[0] + dr, pos[1] + dc)
        nxt, ev = step(lay, pos, VSP2ENV[a])
        if ev == 'invalid':
            return {'mode': 'illegal_offgrid', 'step': i}
        if ev == 'hole':
            greedy = manhattan(target, lay.goal) < manhattan(pos, lay.goal)
            return {'mode': 'illegal_hole', 'step': i, 'greedy': greedy}
        pos = nxt
        if ev == 'goal':
            return {'mode': 'goal', 'step': i}
    return {'mode': 'incomplete', 'step': len(letters)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--file', default='results_vlm_decorr.json')
    args = ap.parse_args()
    d = json.load(open(args.file))
    conds = ['image', 'text', 'table']
    modes = {c: Counter() for c in conds}
    greedy_hole = {c: [0, 0] for c in conds}
    first_break_step = {c: Counter() for c in conds}
    n_fail = {c: 0 for c in conds}
    n_total = {c: 0 for c in conds}
    for lv, r in d['levels'].items():
        by_id = {iid: lay for iid, lay, _ in load_level(int(lv))}
        for c in conds:
            for rec in r['raw'][c]:
                lay = by_id.get(rec['id'])
                if lay is None:
                    continue
                n_total[c] += 1
                res = classify(lay, list(rec['plan']))
                if res['mode'] == 'goal':
                    continue
                n_fail[c] += 1
                modes[c][res['mode']] += 1
                first_break_step[c][min(res['step'], 8)] += 1
                if res['mode'] == 'illegal_hole':
                    greedy_hole[c][1] += 1
                    greedy_hole[c][0] += int(res['greedy'])
    for c in conds:
        f = n_fail[c]
        print(f'\n=== {c.upper()}  ({f}/{n_total[c]} failed) ===')
        for mode, cnt in modes[c].most_common():
            print(f'  {mode:16s}: {cnt:4d}  ({cnt / f * 100:.0f}% of failures)')
        gh, tot = greedy_hole[c]
        if tot:
            print(f'  of illegal_hole failures, {gh}/{tot} = {gh / tot * 100:.0f}% were GREEDY (model headed toward goal and clipped a hole)')
        print(f'  first-break step distribution: {dict(sorted(first_break_step[c].items()))}')
    all_hole_greedy = [sum((greedy_hole[c][0] for c in conds)), sum((greedy_hole[c][1] for c in conds))]
    print('\n=== HEADLINE ===')
    if all_hole_greedy[1]:
        print(f'greedy-into-hole: {all_hole_greedy[0]}/{all_hole_greedy[1]} = {all_hole_greedy[0] / all_hole_greedy[1] * 100:.0f}% of hole-failures were the model moving toward the goal into a hole (not random) -> goal-seeking that ignores hazards')
if __name__ == '__main__':
    main()
