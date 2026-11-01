"""Where replayed VSP plans fail."""
import os
import sys, json, collections
from env import Layout, step, compute_distance_map, DELTAS
from vsp_maze import load_level
V = {'U': 'up', 'D': 'down', 'L': 'left', 'R': 'right'}
M = ['U', 'D', 'L', 'R']

def opt(lay, pos):
    dm = compute_distance_map(lay)
    if pos not in dm:
        return frozenset()
    d0 = dm[pos]
    return frozenset((m for m in M if (lambda t: t[1] not in ('invalid', 'hole') and dm.get(t[0], 10 ** 9) == d0 - 1)(step(lay, pos, V[m]))))

def goal_only(pos, goal):
    dr, dc = (goal[0] - pos[0], goal[1] - pos[1])
    return ('U' if dr < 0 else 'D') if abs(dr) >= abs(dc) else 'L' if dc < 0 else 'R'

def empty_path(lay):
    """Shortest path ignoring ALL holes (what a hazard-blind planner produces)."""
    blind = Layout(lay.level, frozenset(), lay.goal, lay.start)
    dm = compute_distance_map(blind)
    pos = lay.start
    out = []
    while pos != lay.goal and len(out) < 60:
        best = None
        for m in M:
            nxt, ev = step(blind, pos, V[m])
            if ev != 'invalid' and dm.get(nxt, 10 ** 9) == dm[pos] - 1:
                best = (m, nxt)
                break
        if not best:
            break
        out.append(best[0])
        pos = best[1]
    return out

def replay(lay, plan):
    pos = lay.start
    for i, m in enumerate(plan):
        if m not in V:
            return ('badtoken', i, pos)
        nxt, ev = step(lay, pos, V[m])
        if ev == 'hole':
            return ('hole', i, nxt)
        if ev == 'goal':
            return ('goal', i, nxt)
        if ev == 'invalid':
            continue
        pos = nxt
    return ('ran_out', len(plan), pos)
d = json.load(open(os.path.join(os.environ.get('RESULTS_DIR', 'results'), 'results_vlm_decorr.json')))
print(f"{'lvl':>3} {'mod':>6} {'n':>4} {'succ':>5} | {'1st=opt':>8} {'1st=goalonly':>13} {'plan~emptyBFS':>14} | {'die:hole':>9} {'die:ranout':>11} {'offgrid in plan':>16} | {'hole on empty path':>19}")
for lvl in ['3', '5', '8']:
    lays = {i: lay for i, lay, _ in load_level(int(lvl))}
    for mod in ['image', 'text']:
        recs = d['levels'][lvl]['raw'][mod]
        n = fa_opt = fa_go = emptymatch = hole = ranout = offgrid = holeonpath = succ = 0
        for r in recs:
            lay = lays.get(r['id'])
            if lay is None or not r.get('plan'):
                continue
            plan = [c for c in r['plan'] if c in V]
            if not plan:
                continue
            n += 1
            if r['ok']:
                succ += 1
            o = opt(lay, lay.start)
            fa_opt += plan[0] in o
            fa_go += plan[0] == goal_only(lay.start, lay.goal)
            ep = empty_path(lay)
            emptymatch += plan[:len(ep)] == ep
            pos = lay.start
            has_og = False
            for m in plan:
                nxt, ev = step(lay, pos, V[m])
                if ev == 'invalid':
                    has_og = True
                    continue
                if ev in ('hole', 'goal'):
                    break
                pos = nxt
            offgrid += has_og
            why, idx, at = replay(lay, plan)
            if why == 'hole':
                hole += 1
                p = lay.start
                cells = [p]
                for m in ep:
                    nxt, ev = step(Layout(lay.level, frozenset(), lay.goal, lay.start), p, V[m])
                    p = nxt
                    cells.append(p)
                holeonpath += at in cells
            elif why == 'ran_out':
                ranout += 1
        f = lambda x: x / n if n else 0
        print(f'{lvl:>3} {mod:>6} {n:>4} {f(succ):>5.2f} | {f(fa_opt):>8.2f} {f(fa_go):>13.2f} {f(emptymatch):>14.2f} | {f(hole):>9.2f} {f(ranout):>11.2f} {f(offgrid):>16.2f} | {holeonpath / max(hole, 1):>19.2f}')
