"""Execute a different legal move in place of the model's own and read its next move."""
import os
import re
import json
import random
import argparse
from env import step, compute_distance_map
from vsp_maze import load_level
from readout_ladder import MODELS, MOVES, VSP2ENV, load_model, build_inputs, generate_text, optimal_set, _prompts
PLAN_RE = re.compile('ANSWER:\\s*([UDLR]+)', re.I)
MOVE_RE = re.compile('\\b([UDLR])\\b')

def parse_plan(txt, level):
    """The paper's parser, with a length cap."""
    cap = 2 * level * level
    m = PLAN_RE.search(txt)
    if m:
        p = m.group(1).upper()
        return (p[:cap], 'answer') if len(p) <= cap else (p[:cap], 'answer_truncated')
    m = re.search('\\b([UDLR]{2,})\\b', txt)
    if not m:
        return ('', 'none')
    p = m.group(1).upper()
    return (p, 'fallback') if len(p) <= cap else ('', 'fallback_too_long')

def legal_moves(lay, pos):
    """Moves that neither leave the grid nor enter a hole."""
    out = []
    for mv in MOVES:
        nxt, ev = step(lay, pos, VSP2ENV[mv])
        if ev not in ('invalid', 'hole') and nxt != pos:
            out.append((mv, nxt))
    return out

def followup_prompt(lay, pos):
    """Exactly the closed-loop state text of `vlm_remedy.run_episode`: grid size, current position, goal position."""
    from vlm_remedy import INSTR
    return INSTR + '\n' + f'The grid is {lay.level}x{lay.level}. ' + f'You are at row {pos[0] + 1}, column {pos[1] + 1}. ' + f'The goal is at row {lay.goal[0] + 1}, column {lay.goal[1] + 1}. ' + 'What is your next move?'

def first_move(txt):
    m = MOVE_RE.search(txt.strip())
    return m.group(1).upper() if m else None

def score(lay, mv, pos_actual, pos_planned, plan_next):
    if mv is None:
        return {}
    i = MOVES.index(mv)
    return {'optimal_actual': i in optimal_set(lay, pos_actual), 'optimal_planned': i in optimal_set(lay, pos_planned), 'equals_plan_next': mv == plan_next if plan_next else None}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='qwen25_7b', choices=list(MODELS))
    ap.add_argument('--levels', type=int, nargs='+', default=[3, 4, 5, 6, 7, 8])
    ap.add_argument('--n', type=int, default=150)
    ap.add_argument('--max-new', type=int, default=640, help='budget for the PLAN generation. 192 was too small: 48 of 150 Qwen2.5-VL-7B generations had no ANSWER: line and were skipped, biasing the sample toward instances with very short plans.')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='results/results_unexpected_step.json')
    ap.add_argument('--pilot', type=int, default=0, help='generate this many instances, dump every raw generation, and stop. Run it on EVERY model before the full job: the first version of this experiment was smoke-tested on the 7B only and then run blind on the 3B, whose plans are degenerate loops, which cost two full runs.')
    a = ap.parse_args()
    if a.pilot:
        a.n = a.pilot
    rng = random.Random(a.seed)
    from PIL import Image
    model, processor, family, _ = load_model(a.model)
    if family == 'dsvl2':
        raise SystemExit('DeepSeek-VL2 drives its own embedding path; not supported here')
    items = [(p, l) for lv in a.levels for _, l, p in load_level(lv) if optimal_set(l)]
    rng.shuffle(items)
    items = items[:a.n]
    per = []
    for idx, (path, lay) in enumerate(items):
        img = Image.open(path).convert('RGB')
        plan_txt = generate_text(model, processor, family, _prompts(lay, 'hidden')[1], img, max_new=a.max_new)
        plan, plan_src = parse_plan(plan_txt, lay.level)
        rec = {'id': os.path.basename(path), 'level': lay.level, 'plan': plan, 'plan_source': plan_src, 'plan_gen': plan_txt}
        dm = compute_distance_map(lay)
        pos, cuts = (lay.start, [])
        for k in range(len(plan)):
            if pos in dm and len(legal_moves(lay, pos)) >= 2:
                cuts.append((k, pos))
            nxt, ev = step(lay, pos, VSP2ENV[plan[k]])
            if ev in ('hole', 'goal'):
                break
            if ev != 'invalid':
                pos = nxt
        if not cuts:
            rec['skipped'] = 'no usable cut'
            per.append(rec)
            continue
        k, pk = rng.choice(cuts)
        planned_mv = plan[k]
        nxt, ev = step(lay, pk, VSP2ENV[planned_mv])
        if ev == 'hole':
            rec['skipped'] = 'planned move enters a hole'
            per.append(rec)
            continue
        planned_land = pk if ev == 'invalid' else nxt
        alts = [(m, c) for m, c in legal_moves(lay, pk) if c != planned_land]
        if not alts:
            rec['skipped'] = 'no alternative displacement'
            per.append(rec)
            continue
        unx_mv, unx_land = rng.choice(alts)
        plan_next = plan[k + 1] if k + 1 < len(plan) else None
        gc = generate_text(model, processor, family, followup_prompt(lay, planned_land), img, max_new=16)
        gu = generate_text(model, processor, family, followup_prompt(lay, unx_land), img, max_new=16)
        mc, mu = (first_move(gc), first_move(gu))
        rec.update({'k': k, 'cut_pos': list(pk), 'planned_move': planned_mv, 'planned_land': list(planned_land), 'unexpected_move': unx_mv, 'unexpected_land': list(unx_land), 'plan_next': plan_next, 'ctrl_move': mc, 'unx_move': mu, 'ctrl_gen': gc, 'unx_gen': gu, 'ctrl': score(lay, mc, planned_land, planned_land, plan_next), 'unx': score(lay, mu, unx_land, planned_land, plan_next)})
        per.append(rec)
        if a.pilot:
            print(f"\n===== PILOT {idx + 1}/{a.n}  id={rec['id']} level={rec['level']} source={rec['plan_source']} plan={rec['plan']!r} skipped={rec.get('skipped')}\n--- plan generation ---\n{plan_txt}\n--- ctrl reply ---\n{rec.get('ctrl_gen')!r}\n--- unx reply ---\n{rec.get('unx_gen')!r}", flush=True)
        if idx == 0 and (not a.pilot):
            print(f'--- first instance ---\nplan={plan} k={k} planned={planned_mv}->{planned_land} unexpected={unx_mv}->{unx_land}\nctrl reply={gc!r}\nunx  reply={gu!r}\n--- end ---', flush=True)
        if (idx + 1) % 10 == 0:
            print(f'  {idx + 1}/{len(items)}', flush=True)

    def rate(arm, key):
        v = [r[arm][key] for r in per if r.get(arm, {}).get(key) is not None]
        return [sum(v) / len(v), len(v)] if v else None
    import collections
    sc = [r for r in per if 'unx' in r]
    dist = lambda key: dict(collections.Counter((r[key] for r in sc)).most_common())
    nxt = [r['plan_next'] for r in sc if r.get('plan_next')]

    def arm_chance(cellkey):
        v = [len(optimal_set(lay_of[r['id']], tuple(r[cellkey]))) / 4.0 for r in sc]
        return sum(v) / len(v) if v else None
    lay_of = {os.path.basename(p_): l for p_, l in items}
    res = {'model': MODELS[a.model][0], 'levels': a.levels, 'n': len(per), 'seed': a.seed, 'scored': len(sc), 'chance_uniform4_ctrl': arm_chance('planned_land'), 'chance_uniform4_unx': arm_chance('unexpected_land'), 'plan_source': dict(collections.Counter((r['plan_source'] for r in per)).most_common()), 'ctrl_reply_dist': dist('ctrl_move'), 'unx_reply_dist': dist('unx_move'), 'plan_next_dist': dict(collections.Counter(nxt).most_common()), 'reply_equals_stated_last_move': {'ctrl': sum((1 for r in sc if r['ctrl_move'] == r['planned_move'])) / max(len(sc), 1), 'unx': sum((1 for r in sc if r['unx_move'] == r['unexpected_move'])) / max(len(sc), 1)}, 'cut_at_start_fraction': sum((1 for r in sc if r['k'] == 0)) / max(len(sc), 1), 'skipped': {s: sum((1 for r in per if r.get('skipped') == s)) for s in {r['skipped'] for r in per if 'skipped' in r}}, 'ctrl_optimal_actual': rate('ctrl', 'optimal_actual'), 'unx_optimal_actual': rate('unx', 'optimal_actual'), 'unx_optimal_planned': rate('unx', 'optimal_planned'), 'ctrl_equals_plan_next': rate('ctrl', 'equals_plan_next'), 'unx_equals_plan_next': rate('unx', 'equals_plan_next'), 'reading': 'Re-plans: unx_optimal_actual near ctrl_optimal_actual, unx_equals_plan_next low. Carries on regardless: unx_optimal_actual collapses while unx_optimal_planned and unx_equals_plan_next hold up.', 'per_instance': per}
    with open(a.out, 'w') as fh:
        json.dump(res, fh, indent=2)
    print(json.dumps({k: v for k, v in res.items() if k != 'per_instance'}, indent=2))
    print('saved ->', a.out)
if __name__ == '__main__':
    main()
