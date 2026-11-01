"""Matched counterfactual pairs that move one hazard so the optimal first move changes."""
import os
import json
import math
import random
import argparse
from collections import defaultdict
from env import Layout, step, compute_distance_map, generate_instance
MOVES = ['U', 'D', 'L', 'R']
VSP2ENV = {'U': 'up', 'D': 'down', 'L': 'left', 'R': 'right'}

def optimal_set(lay):
    """Indices into MOVES that are on a shortest safe path from lay.start."""
    dm = compute_distance_map(lay)
    if lay.start not in dm:
        return frozenset()
    d0 = dm[lay.start]
    out = set()
    for i, mv in enumerate(MOVES):
        nxt, ev = step(lay, lay.start, VSP2ENV[mv])
        if ev not in ('invalid', 'hole') and dm.get(nxt, 10 ** 9) == d0 - 1:
            out.add(i)
    return frozenset(out)

def ongrid_set(lay):
    return frozenset((i for i, mv in enumerate(MOVES) if step(lay, lay.start, VSP2ENV[mv])[1] != 'invalid'))

def hazard_aware_set(lay):
    """What vipq_gate.py called 'legal': excludes off-grid AND holes."""
    return frozenset((i for i, mv in enumerate(MOVES) if step(lay, lay.start, VSP2ENV[mv])[1] not in ('invalid', 'hole')))

def goal_only_action(lay):
    """Hazard-BLIND heuristic: step along the larger axis gap toward the goal."""
    dr = lay.goal[0] - lay.start[0]
    dc = lay.goal[1] - lay.start[1]
    if abs(dr) >= abs(dc):
        return 0 if dr < 0 else 1
    return 2 if dc < 0 else 3

def _relocations(lay):
    """All single-hole relocations of `lay`, as (new_layout, moved_hole, manhattan_distance)."""
    free = [(r, c) for r in range(lay.level) for c in range(lay.level) if (r, c) not in lay.holes and (r, c) != lay.start and ((r, c) != lay.goal)]
    for h in sorted(lay.holes):
        rest = lay.holes - {h}
        for dst in free:
            new = Layout(lay.level, frozenset(rest | {dst}), lay.goal, lay.start)
            if not optimal_set(new):
                continue
            yield (new, h, abs(dst[0] - h[0]) + abs(dst[1] - h[1]))

def make_pair_group(lay, rng):
    """One base + a FLIP edit + a PRESERVE edit whose relocation distances are EQUAL."""
    base_opt = optimal_set(lay)
    if not base_opt:
        return None
    by_dist = defaultdict(lambda: {'flip': [], 'preserve': []})
    for new, _h, d in _relocations(lay):
        o = optimal_set(new)
        if not o & base_opt:
            by_dist[d]['flip'].append(new)
        elif o == base_opt:
            by_dist[d]['preserve'].append(new)
    usable = [d for d, v in by_dist.items() if v['flip'] and v['preserve']]
    if not usable:
        return None
    d = rng.choice(sorted(usable))
    return {'base': lay, 'flip': rng.choice(by_dist[d]['flip']), 'preserve': rng.choice(by_dist[d]['preserve']), 'edit_dist': d}

def build_pairs(n_groups, level, hole_frac, seed, max_tries_mult=40):
    rng = random.Random(seed)
    groups, i = ([], 0)
    while len(groups) < n_groups and i < n_groups * max_tries_mult:
        i += 1
        try:
            lay = generate_instance(level=level, hole_frac=hole_frac, min_len=3, seed=seed * 1000000 + i)
        except RuntimeError:
            continue
        g = make_pair_group(lay, rng)
        if g:
            groups.append(g)
    return groups

def _lay_to_json(l):
    return {'level': l.level, 'holes': sorted(map(list, l.holes)), 'goal': list(l.goal), 'start': list(l.start)}

def _lay_from_json(d):
    return Layout(d['level'], frozenset(map(tuple, d['holes'])), tuple(d['goal']), tuple(d['start']))

def save_pairs(groups, path):
    with open(path, 'w') as f:
        json.dump([{'edit_dist': g['edit_dist'], **{k: _lay_to_json(g[k]) for k in ('base', 'flip', 'preserve')}} for g in groups], f)

def load_pairs(path):
    with open(path) as f:
        raw = json.load(f)
    return [{'edit_dist': g['edit_dist'], **{k: _lay_from_json(g[k]) for k in ('base', 'flip', 'preserve')}} for g in raw]

def pair_diagnostics(groups):
    """Sanity checks that must pass before the pairs are worth any GPU time."""
    n = len(groups)
    bad = []
    dists = defaultdict(lambda: [0, 0])
    for g in groups:
        b, f, p = (optimal_set(g['base']), optimal_set(g['flip']), optimal_set(g['preserve']))
        if b & f:
            bad.append('flip shares an optimal action with base')
        if p != b:
            bad.append('preserve changed the optimal set')
        if not (b and f and p):
            bad.append('unsolvable member')
        for k, lay in (('flip', g['flip']), ('preserve', g['preserve'])):
            if len(lay.holes) != len(g['base'].holes):
                bad.append(f'{k} changed the hole count')
        dists[g['edit_dist']][0] += 1
        dists[g['edit_dist']][1] += 1
    print(f'groups: {n}')
    print(f'integrity violations: {len(bad)}' + (f'  e.g. {bad[:3]}' if bad else '  (clean)'))
    print('edit-distance match (flip vs preserve counts per distance):')
    for d in sorted(dists):
        print(f'  dist {d}: flip {dists[d][0]}  preserve {dists[d][1]}')
    gz = sum((1 for g in groups if goal_only_action(g['base']) in optimal_set(g['base']))) / max(n, 1)
    gf = sum((1 for g in groups if goal_only_action(g['flip']) in optimal_set(g['flip']))) / max(n, 1)
    print(f'goal-only heuristic on base : {gz:.3f}')
    print(f'goal-only heuristic on flip : {gf:.3f}   <- FLIP edits are hazard-DEPENDENT by construction, so this should fall well below the base value')
    return len(bad) == 0

def baselines(items):
    """items: list of Layout."""
    n = len(items)
    if not n:
        return {}
    u4 = sum((len(optimal_set(l)) / 4 for l in items)) / n
    og = sum((len(optimal_set(l)) / max(len(ongrid_set(l)), 1) for l in items)) / n
    ha = sum((len(optimal_set(l)) / max(len(hazard_aware_set(l)), 1) for l in items)) / n
    go = sum((1 for l in items if goal_only_action(l) in optimal_set(l))) / n
    const = max((sum((1 for l in items if i in optimal_set(l))) / n for i in range(4)))
    return {'uniform_4': u4, 'uniform_on_grid': og, 'uniform_hazard_aware': ha, 'goal_only_heuristic': go, 'best_constant': const}

def stage_render(groups, out_dir):
    os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
    import numpy as np
    from PIL import Image
    import gym

    def to_desc(lay):
        g = [['F'] * lay.level for _ in range(lay.level)]
        for r, c in lay.holes:
            g[r][c] = 'H'
        g[lay.goal[0]][lay.goal[1]] = 'G'
        g[lay.start[0]][lay.start[1]] = 'S'
        return [''.join(r) for r in g]
    os.makedirs(out_dir, exist_ok=True)
    for gi, g in enumerate(groups):
        for key in ('base', 'flip', 'preserve'):
            env = gym.make('FrozenLake-v1', desc=to_desc(g[key]), is_slippery=False, render_mode='rgb_array')
            env.reset()
            arr = env.render()
            env.close()
            Image.fromarray(np.asarray(arr, dtype=np.uint8)).save(os.path.join(out_dir, f'{gi}_{key}.png'))
        if (gi + 1) % 50 == 0:
            print(f'  rendered {gi + 1}/{len(groups)}', flush=True)
    print(f'done -> {out_dir}', flush=True)

def _torch():
    import torch
    return torch

def act_forward(model, processor, tok_ids, lay, img, layers):
    """ONE forward pass under the EXACT action prompt, returning both the U/D/L/R logits and the hidden states."""
    import torch
    from vlm_remedy import INSTR
    state = f'The grid is {lay.level}x{lay.level}. The player is at row {lay.start[0] + 1}, column {lay.start[1] + 1}. What is your next move?'
    content = ([{'type': 'image'}] if img is not None else []) + [{'type': 'text', 'text': INSTR + '\n' + state}]
    chat = processor.apply_chat_template([{'role': 'user', 'content': content}], tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[chat], images=[img] if img is not None else None, return_tensors='pt').to(model.device)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    lg = out.logits[0, -1, :].float().cpu()
    move = torch.tensor([max((lg[i].item() for i in ids)) for ids in tok_ids])
    feats = {l: out.hidden_states[l][0, -1, :].float().cpu() for l in layers}
    return (move, feats)

def set_ce(logits, opt_sets):
    """Set-valued cross-entropy: -log sum_{a in A*} p(a)."""
    import torch
    lp = torch.log_softmax(logits, dim=1)
    losses = []
    for i, o in enumerate(opt_sets):
        losses.append(-torch.logsumexp(lp[i, sorted(o)], dim=0))
    return torch.stack(losses).mean()

def fit_linear(Xtr, opt_tr, Xte, steps=1500, lr=0.01, wd=0.001, n_cls=4, seed=0):
    """Standardized features (fit on train only), set-valued loss, reports train fit so underfitting cannot masquerade as 'the signal is absent'."""
    import torch
    torch.manual_seed(seed)
    mu, sd = (Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True).clamp_min(1e-06))
    Xtr, Xte = ((Xtr - mu) / sd, (Xte - mu) / sd)
    W = torch.zeros(Xtr.shape[1], n_cls, requires_grad=True)
    b = torch.zeros(n_cls, requires_grad=True)
    opt = torch.optim.Adam([W, b], lr=lr, weight_decay=wd)
    for _ in range(steps):
        loss = set_ce(Xtr @ W + b, opt_tr)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        tr_pred = (Xtr @ W + b).argmax(1)
        te_pred = (Xte @ W + b).argmax(1)
    tr_acc = sum((int(int(tr_pred[i]) in opt_tr[i]) for i in range(len(opt_tr)))) / len(opt_tr)
    return (te_pred, tr_acc, float(loss.detach()))

def fit_binary(Xtr, ytr, Xte, steps=1200, lr=0.01, wd=0.01, seed=0):
    import torch
    torch.manual_seed(seed)
    mu, sd = (Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True).clamp_min(1e-06))
    Xtr, Xte = ((Xtr - mu) / sd, (Xte - mu) / sd)
    w = torch.zeros(Xtr.shape[1], 1, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=lr, weight_decay=wd)
    for _ in range(steps):
        loss = torch.nn.functional.binary_cross_entropy_with_logits(Xtr @ w + b, ytr)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        return (Xte @ w + b > 0).float().squeeze(1)

def mean_ci(vals):
    n = len(vals)
    m = sum(vals) / n
    if n < 2:
        return (m, 0.0)
    sd = math.sqrt(sum(((v - m) ** 2 for v in vals)) / (n - 1))
    return (m, 1.96 * sd / math.sqrt(n))

def stage_run(groups, img_dir, layers, n_folds, out_path, blank_control):
    import torch
    from PIL import Image
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    from vlm_remedy import move_token_ids, MODEL_ID
    print(f'loading {MODEL_ID} ...', flush=True)
    model = Qwen2VLForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map='auto')
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model.eval()
    tok_ids = move_token_ids(processor.tokenizer)
    keys = ('base', 'flip', 'preserve')
    feats = {l: {k: [] for k in keys} for l in layers}
    blank = {l: {k: [] for k in keys} for l in layers}
    policy = {k: [] for k in keys}
    for gi, g in enumerate(groups):
        for k in keys:
            img = Image.open(os.path.join(img_dir, f'{gi}_{k}.png')).convert('RGB')
            mv, fs = act_forward(model, processor, tok_ids, g[k], img, layers)
            policy[k].append(mv)
            for l in layers:
                feats[l][k].append(fs[l])
            if blank_control:
                bimg = Image.new('RGB', img.size, (255, 255, 255))
                _, bf = act_forward(model, processor, tok_ids, g[k], bimg, layers)
                for l in layers:
                    blank[l][k].append(bf[l])
        if (gi + 1) % 25 == 0:
            print(f'  {gi + 1}/{len(groups)} groups', flush=True)
    for l in layers:
        for k in keys:
            feats[l][k] = torch.stack(feats[l][k])
            if blank_control:
                blank[l][k] = torch.stack(blank[l][k])
    for k in keys:
        policy[k] = torch.stack(policy[k])
    n = len(groups)
    rng = random.Random(0)
    order = list(range(n))
    rng.shuffle(order)
    folds = [order[i::n_folds] for i in range(n_folds)]
    res = {'n_groups': n, 'layers': layers, 'n_folds': n_folds}
    all_items = [g[k] for g in groups for k in keys]
    flip_items = [g['flip'] for g in groups]
    res['baselines_all'] = baselines(all_items)
    res['baselines_flip_only'] = baselines(flip_items)
    res['policy_flip_fraction_note'] = "FLIP items are hazard-dependent by construction; goal_only_heuristic on them is the metric's real null"
    res['action_readout'] = {}
    for l in layers:
        X = torch.cat([feats[l][k] for k in keys])
        lays = all_items
        gidx = [gi for gi in range(n) for _ in keys]
        opts = [optimal_set(x) for x in lays]
        is_flip = [k == 'flip' for _ in range(n) for k in keys]
        accs, faccs, tracc = ([], [], [])
        shuf, blk = ([], [])
        for f in range(n_folds):
            te_groups = set(folds[f])
            te = [i for i in range(len(lays)) if gidx[i] in te_groups]
            tr = [i for i in range(len(lays)) if gidx[i] not in te_groups]
            pred, ta, _ = fit_linear(X[tr], [opts[i] for i in tr], X[te])
            tracc.append(ta)
            accs.append(sum((int(int(pred[j]) in opts[i]) for j, i in enumerate(te))) / len(te))
            fj = [j for j, i in enumerate(te) if is_flip[i]]
            if fj:
                faccs.append(sum((int(int(pred[j]) in opts[te[j]]) for j in fj)) / len(fj))
            sh = [opts[i] for i in tr]
            random.Random(f).shuffle(sh)
            spred, _, _ = fit_linear(X[tr], sh, X[te])
            shuf.append(sum((int(int(spred[j]) in opts[i]) for j, i in enumerate(te))) / len(te))
            if blank_control:
                Xb = torch.cat([blank[l][k] for k in keys])
                bpred, _, _ = fit_linear(Xb[tr], [opts[i] for i in tr], Xb[te])
                blk.append(sum((int(int(bpred[j]) in opts[i]) for j, i in enumerate(te))) / len(te))
        e = {'test_acc': mean_ci(accs), 'test_acc_FLIP_subset': mean_ci(faccs) if faccs else None, 'train_acc': mean_ci(tracc), 'shuffled_label_control': mean_ci(shuf)}
        if blk:
            e['blank_image_control'] = mean_ci(blk)
        res['action_readout'][l] = e
    res['representational_sensitivity'] = {}
    for l in layers:
        d_flip = feats[l]['flip'] - feats[l]['base']
        d_pres = feats[l]['preserve'] - feats[l]['base']
        X = torch.cat([d_flip, d_pres])
        y = torch.cat([torch.ones(n, 1), torch.zeros(n, 1)])
        gidx = list(range(n)) * 2
        accs, shuf, blk = ([], [], [])
        for f in range(n_folds):
            te_groups = set(folds[f])
            te = [i for i in range(2 * n) if gidx[i] in te_groups]
            tr = [i for i in range(2 * n) if gidx[i] not in te_groups]
            p = fit_binary(X[tr], y[tr], X[te])
            accs.append(float((p == y[te].squeeze(1)).float().mean()))
            ys = y[tr].clone()[torch.randperm(len(tr), generator=torch.Generator().manual_seed(f))]
            ps = fit_binary(X[tr], ys, X[te])
            shuf.append(float((ps == y[te].squeeze(1)).float().mean()))
            if blank_control:
                Xb = torch.cat([blank[l]['flip'] - blank[l]['base'], blank[l]['preserve'] - blank[l]['base']])
                pb = fit_binary(Xb[tr], y[tr], Xb[te])
                blk.append(float((pb == y[te].squeeze(1)).float().mean()))
        e = {'flip_vs_preserve_acc': mean_ci(accs), 'shuffled_label_control': mean_ci(shuf), 'chance': 0.5}
        if blk:
            e['blank_image_control'] = mean_ci(blk)
        res['representational_sensitivity'][l] = e
    lsm = torch.log_softmax

    def mass(lp, s):
        return float(torch.logsumexp(lp[sorted(s)], dim=0))
    dd, flipped, shift_f, shift_p = ([], [], [], [])
    for gi, g in enumerate(groups):
        old, new = (optimal_set(g['base']), optimal_set(g['flip']))
        lb = lsm(policy['base'][gi], 0)
        lf = lsm(policy['flip'][gi], 0)
        lp_ = lsm(policy['preserve'][gi], 0)
        dd.append(mass(lf, new) - mass(lb, new) - (mass(lf, old) - mass(lb, old)))
        flipped.append(int(int(policy['base'][gi].argmax()) in old and int(policy['flip'][gi].argmax()) in new))
        shift_f.append(float((lf.exp() - lb.exp()).abs().sum()))
        shift_p.append(float((lp_.exp() - lb.exp()).abs().sum()))
    res['behavioural_sensitivity'] = {'diff_in_diff_logprob_FLIP': mean_ci(dd), 'frac_positive_dd': sum((1 for v in dd if v > 0)) / len(dd), 'argmax_tracks_flip': sum(flipped) / len(flipped), 'L1_policy_shift_FLIP': mean_ci(shift_f), 'L1_policy_shift_PRESERVE': mean_ci(shift_p), 'note': 'dd > 0 means the policy moved toward the NEW optimum relative to the OLD. An agent that tracks control should show L1_shift_FLIP >> L1_shift_PRESERVE.'}
    with open(out_path, 'w') as f:
        json.dump(res, f, indent=2, default=str)
    print(json.dumps(res, indent=2, default=str), flush=True)
    print(f'\nsaved -> {out_path}', flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=['pairs', 'render', 'run'], required=True)
    ap.add_argument('--pairs_path', default='vipq_pairs.json')
    ap.add_argument('--img_dir', default='vipq_pairs_img')
    ap.add_argument('--n_groups', type=int, default=300)
    ap.add_argument('--level', type=int, default=5)
    ap.add_argument('--hole_frac', type=float, default=0.15)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--layers', type=int, nargs='+', default=[14, 21, 28])
    ap.add_argument('--n_folds', type=int, default=5)
    ap.add_argument('--no_blank_control', action='store_true')
    ap.add_argument('--out', default='results_vipq_gate_v2.json')
    args = ap.parse_args()
    if args.stage == 'pairs':
        g = build_pairs(args.n_groups, args.level, args.hole_frac, args.seed)
        ok = pair_diagnostics(g)
        save_pairs(g, args.pairs_path)
        print(f"saved {len(g)} groups -> {args.pairs_path}  (integrity {('OK' if ok else 'FAILED')})")
        print('\nbaselines on BASE items:', json.dumps(baselines([x['base'] for x in g]), indent=2))
        print('baselines on FLIP items:', json.dumps(baselines([x['flip'] for x in g]), indent=2))
    elif args.stage == 'render':
        stage_render(load_pairs(args.pairs_path), args.img_dir)
    else:
        stage_run(load_pairs(args.pairs_path), args.img_dir, args.layers, args.n_folds, args.out, not args.no_blank_control)
if __name__ == '__main__':
    main()
