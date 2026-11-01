"""Multi-step rollout of the position-only policy: cycles, self-trapping and decoded-geometry agreement."""
import json
import argparse
from collections import defaultdict
from env import Layout, step, compute_distance_map
MOVES = ['U', 'D', 'L', 'R']
VSP2ENV = {'U': 'up', 'D': 'down', 'L': 'left', 'R': 'right'}
LEVEL = 5

def optimal_at(lay, pos):
    dm = compute_distance_map(lay)
    if pos not in dm:
        return frozenset()
    d0 = dm[pos]
    out = set()
    for i, mv in enumerate(MOVES):
        nxt, ev = step(lay, pos, VSP2ENV[mv])
        if ev not in ('invalid', 'hole') and dm.get(nxt, 10 ** 9) == d0 - 1:
            out.add(i)
    return frozenset(out)

def ongrid_at(lay, pos):
    return frozenset((i for i, mv in enumerate(MOVES) if step(lay, pos, VSP2ENV[mv])[1] != 'invalid'))

def hazard_aware_at(lay, pos):
    return frozenset((i for i, mv in enumerate(MOVES) if step(lay, pos, VSP2ENV[mv])[1] not in ('invalid', 'hole')))

def goal_only_at(lay, pos):
    dr, dc = (lay.goal[0] - pos[0], lay.goal[1] - pos[1])
    if abs(dr) >= abs(dc):
        return 0 if dr < 0 else 1
    return 2 if dc < 0 else 3

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gen_dir', default='vsp_train6k')
    ap.add_argument('--n', type=int, default=300)
    ap.add_argument('--layer', type=int, default=21)
    ap.add_argument('--budget', type=int, default=12)
    ap.add_argument('--out', default='results_agreement_multistep.json')
    args = ap.parse_args()
    import torch
    from PIL import Image
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    from probe_spatial import load_instances, build_labels, MODEL_ID
    from probe_spatial_v2 import feat_from_image, train_binary_cols, _predict
    from vlm_remedy import move_logits, move_token_ids
    print(f'loading {MODEL_ID} ...', flush=True)
    model = Qwen2VLForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map='auto')
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model.eval()
    tok_ids = move_token_ids(processor.tokenizer)
    insts = load_instances(args.gen_dir, args.n)
    print(f'{len(insts)} mazes', flush=True)
    X = torch.stack([feat_from_image(model, processor, Image.open(p).convert('RGB'), args.layer) for p, _ in insts])
    cat, holes = build_labels(insts)
    ntr = int(0.8 * len(insts))
    hole_pred = train_binary_cols(X[:ntr], holes[:ntr], X[ntr:])
    geo = {k: _predict(X[:ntr], cat[k][:ntr], X[ntr:], LEVEL) for k in ('player_row', 'player_col', 'goal_row', 'goal_col')}
    test = insts[ntr:]
    print(f'probe trained on {ntr}, rolling out {len(test)}', flush=True)
    import random as _random
    rng = _random.Random(0)
    per_step = defaultdict(lambda: defaultdict(float))
    counts = defaultdict(int)
    fv = defaultdict(float)
    fv_n = 0
    fresh = defaultdict(float)
    fresh_n = defaultdict(int)
    cycles = []
    outcomes = {'goal': 0, 'hole': 0, 'budget': 0}
    for i, (img_path, lay) in enumerate(test):
        dec = Layout(LEVEL, frozenset(((r, c) for r in range(LEVEL) for c in range(LEVEL) if hole_pred[i, r * LEVEL + c])), lay.goal, lay.start)
        pos = lay.start
        visited = {}
        cand = [(r, c) for r in range(LEVEL) for c in range(LEVEL) if (r, c) not in lay.holes and (r, c) != lay.goal and optimal_at(lay, (r, c))]
        for t in range(args.budget):
            if cand:
                fp = rng.choice(cand)
                fa = int(move_logits(model, processor, tok_ids, _state_text(lay, fp), _render(lay, fp)).argmax())
                fresh[t] += int(fa in optimal_at(lay, fp))
                fresh_n[t] += 1
        for t in range(args.budget):
            img = _render(lay, pos)
            a = int(move_logits(model, processor, tok_ids, _state_text(lay, pos), img).argmax())
            ot, od = (optimal_at(lay, pos), optimal_at(dec, pos))
            og, ha = (ongrid_at(lay, pos), hazard_aware_at(lay, pos))
            per_step[t]['agree_true'] += int(a in ot)
            per_step[t]['agree_decoded'] += int(a in od)
            per_step[t]['decoded_solvable'] += int(bool(od))
            per_step[t]['decoded_current_cell_blocked'] += int(pos in dec.holes)
            per_step[t]['decoded_goal_blocked'] += int(dec.goal in dec.holes)
            per_step[t]['decoded_uniform4'] += len(od) / 4
            per_step[t]['uniform_4'] += len(ot) / 4
            per_step[t]['uniform_on_grid'] += len(ot) / max(len(og), 1)
            per_step[t]['hazard_aware'] += len(ot) / max(len(ha), 1)
            per_step[t]['goal_only'] += int(goal_only_at(lay, pos) in ot)
            if t == 0:
                dec_lay = Layout(LEVEL, frozenset(), (int(geo['goal_row'][i]), int(geo['goal_col'][i])), (int(geo['player_row'][i]), int(geo['player_col'][i])))
                per_step[t]['decoded_geometry_goal_only'] += int(goal_only_at(dec_lay, dec_lay.start) in ot)
            counts[t] += 1
            if pos not in visited:
                visited[pos] = t
                fv['agree_true'] += int(a in ot)
                fv['goal_only'] += int(goal_only_at(lay, pos) in ot)
                fv['hazard_aware'] += len(ot) / max(len(ha), 1)
                fv_n += 1
            elif not cycles or cycles[-1][0] != i:
                cycles.append((i, t, t - visited[pos]))
            nxt, ev = step(lay, pos, VSP2ENV[MOVES[a]])
            if ev == 'hole':
                outcomes['hole'] += 1
                break
            if ev == 'invalid':
                continue
            pos = nxt
            if ev == 'goal':
                outcomes['goal'] += 1
                break
        else:
            outcomes['budget'] += 1
        if (i + 1) % 25 == 0:
            print(f'  {i + 1}/{len(test)}', flush=True)
    res = {'model': MODEL_ID, 'layer': args.layer, 'n_episodes': len(test), 'outcomes': outcomes, 'per_step': {}}
    for t in sorted(counts):
        n = counts[t]
        res['per_step'][t] = {k: v / n for k, v in per_step[t].items()} | {'n': n}
    res['fresh_state_control'] = {t: {'agree_true': fresh[t] / fresh_n[t], 'n': fresh_n[t]} for t in sorted(fresh_n) if fresh_n[t]}
    res['first_visit'] = {k: v / fv_n for k, v in fv.items()} | {'n_states': fv_n}
    res['cycles'] = {'n_episodes_cycling': len(cycles), 'mean_close_step': sum((c[1] for c in cycles)) / len(cycles) if cycles else None, 'cycle_lengths': sorted(set((c[2] for c in cycles)))}
    first, later = (res['per_step'].get(0, {}), [res['per_step'][t] for t in sorted(counts) if t >= 3])
    res['summary'] = {'step0_agree_true': first.get('agree_true'), 'step3plus_agree_true': sum((d['agree_true'] * d['n'] for d in later)) / sum((d['n'] for d in later)) if later else None, 'step3plus_macro_step_agree_true': sum((d['agree_true'] for d in later)) / len(later) if later else None, 'step0_decoded_geometry_goal_only': first.get('decoded_geometry_goal_only'), 'reading': 'Compare step0 to step3+. A drop is a rollout failure invisible to the first-action results the audit is built on. Note n falls with t (episodes terminate), so late steps are a survivor-biased subsample -- read n.', 'decay_reading': 'pi is a pure function of position, so it cannot degrade with time. Read the decay against fresh_state_control, which draws states from a t-INDEPENDENT distribution and must therefore be flat; and against first_visit, which counts each state once instead of letting a short cycle donate the same wrong action to every later step.', 'dissociation_reading': "decoded_geometry_goal_only is a rule driven ENTIRELY by linear readouts of the model's own activations. If it exceeds step0_agree_true, then information sufficient for better action is present and linearly accessible in the representation while the policy fails to use it -- a representation-behaviour dissociation over GEOMETRY (not the full map). Handoff section 1.2 declared this direction dead by testing the full-map version instead."}
    with open(args.out, 'w') as f:
        json.dump(res, f, indent=2, default=str)
    print(json.dumps(res, indent=2, default=str), flush=True)
    print(f'\nsaved -> {args.out}', flush=True)

def _state_text(lay, pos):
    return f'The grid is {lay.level}x{lay.level}. You are at row {pos[0] + 1}, column {pos[1] + 1}. What is your next move?'

def _render(lay, pos):
    import os
    os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
    import numpy as np
    from PIL import Image
    import gym
    g = [['F'] * lay.level for _ in range(lay.level)]
    for r, c in lay.holes:
        g[r][c] = 'H'
    g[lay.goal[0]][lay.goal[1]] = 'G'
    g[pos[0]][pos[1]] = 'S'
    env = gym.make('FrozenLake-v1', desc=[''.join(r) for r in g], is_slippery=False, render_mode='rgb_array')
    env.reset()
    arr = env.render()
    env.close()
    return Image.fromarray(np.asarray(arr, dtype=np.uint8))
if __name__ == '__main__':
    main()
