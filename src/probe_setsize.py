"""Per-hazard recall and exact-set recovery as the number of hazards grows."""
import os
import json
import argparse
from env import Layout, generate_instance, compute_distance_map, step, DELTAS
LEVEL = 5

def gen_for_k(k, n, seed):
    """n solvable level-5 mazes with EXACTLY k holes."""
    out, i = ([], 0)
    while len(out) < n and i < n * 80:
        i += 1
        try:
            lay = generate_instance(level=LEVEL, hole_frac=k / (LEVEL * LEVEL), min_len=3, seed=seed * 1000000 + k * 10000 + i)
        except RuntimeError:
            continue
        if len(lay.holes) == k:
            out.append(lay)
    if len(out) < n:
        raise RuntimeError(f'k={k}: only {len(out)}/{n}')
    return out

def _j(l):
    return {'level': l.level, 'holes': sorted(map(list, l.holes)), 'goal': list(l.goal), 'start': list(l.start)}

def _u(d):
    return Layout(d['level'], frozenset(map(tuple, d['holes'])), tuple(d['goal']), tuple(d['start']))

def greedy_path(lay):
    dm = compute_distance_map(lay)
    if lay.start not in dm:
        return None
    pos, path = (lay.start, [])
    while pos != lay.goal and len(path) < 4 * lay.level * lay.level:
        best = None
        for a, (dr, dc) in DELTAS.items():
            nb = (pos[0] + dr, pos[1] + dc)
            if nb in dm and (best is None or dm[nb] < dm[best[1]]):
                best = (a, nb)
        if best is None:
            return None
        path.append(best[0])
        pos = best[1]
    return path

def executes_in_true(true_lay, path):
    if not path:
        return False
    pos = true_lay.start
    for a in path:
        nxt, ev = step(true_lay, pos, a)
        if ev == 'hole':
            return False
        if ev == 'invalid':
            continue
        pos = nxt
        if ev == 'goal':
            return True
    return pos == true_lay.goal

def stage_gen(ks, n, seed, path):
    data = {k: [_j(l) for l in gen_for_k(k, n, seed)] for k in ks}
    with open(path, 'w') as f:
        json.dump(data, f)
    for k in ks:
        assert all((len(_u(d).holes) == k for d in data[k]))
        print(f'  k={k}: {len(data[k])} mazes, all exactly {k} holes  (chance recall@k = {k / 25:.3f})')
    print(f'saved -> {path}')

def stage_render(data, out_dir):
    os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
    import numpy as np
    from PIL import Image
    import gym

    def desc(l):
        g = [['F'] * l.level for _ in range(l.level)]
        for r, c in l.holes:
            g[r][c] = 'H'
        g[l.goal[0]][l.goal[1]] = 'G'
        g[l.start[0]][l.start[1]] = 'S'
        return [''.join(r) for r in g]
    os.makedirs(out_dir, exist_ok=True)
    for k, items in data.items():
        for i, d in enumerate(items):
            env = gym.make('FrozenLake-v1', desc=desc(_u(d)), is_slippery=False, render_mode='rgb_array')
            env.reset()
            arr = env.render()
            env.close()
            Image.fromarray(np.asarray(arr, dtype=np.uint8)).save(os.path.join(out_dir, f'k{k}_{i}.png'))
        print(f'  rendered k={k} ({len(items)})', flush=True)

def stage_run(data, img_dir, layer, out_path):
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    from probe_spatial import MODEL_ID
    from probe_spatial_v2 import feat_from_image
    print(f'loading {MODEL_ID} ...', flush=True)
    model = Qwen2VLForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map='auto')
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model.eval()

    def train_scores(Xtr, Ytr, Xte, steps=600, lr=0.01, wd=0.001):
        mu, sd = (Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True).clamp_min(1e-06))
        Xtr, Xte = ((Xtr - mu) / sd, (Xte - mu) / sd)
        W = torch.zeros(Xtr.shape[1], LEVEL * LEVEL, requires_grad=True)
        b = torch.zeros(LEVEL * LEVEL, requires_grad=True)
        opt = torch.optim.Adam([W, b], lr=lr, weight_decay=wd)
        for _ in range(steps):
            loss = F.binary_cross_entropy_with_logits(Xtr @ W + b, Ytr)
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            return Xte @ W + b
    res = {'layer': layer, 'metric': 'recall@k = |top-k cells INTERSECT true holes| / k'}
    for k in sorted(data, key=int):
        lays = [_u(d) for d in data[k]]
        kk = int(k)
        for tag in ('image', 'blank'):
            feats = []
            for i, l in enumerate(lays):
                img = Image.open(os.path.join(img_dir, f'k{k}_{i}.png')).convert('RGB')
                if tag == 'blank':
                    img = Image.new('RGB', img.size, (255, 255, 255))
                feats.append(feat_from_image(model, processor, img, layer))
            X = torch.stack(feats)
            Y = torch.zeros(len(lays), LEVEL * LEVEL)
            for i, l in enumerate(lays):
                for r, c in l.holes:
                    Y[i, r * LEVEL + c] = 1.0
            ntr = int(0.8 * len(lays))
            S = train_scores(X[:ntr], Y[:ntr], X[ntr:])
            te = lays[ntr:]
            rec = anyh = exact = suff = 0
            for i, l in enumerate(te):
                topk = set((int(j) for j in S[i].topk(kk).indices))
                true = {r * LEVEL + c for r, c in l.holes}
                rec += len(topk & true) / kk
                anyh += int(int(S[i].argmax()) in true)
                exact += int(topk == true)
                dec = Layout(LEVEL, frozenset(((j // LEVEL, j % LEVEL) for j in topk)), l.goal, l.start)
                suff += int(executes_in_true(l, greedy_path(dec)))
            n = len(te)
            res.setdefault(tag, {})[kk] = {'recall_at_k': rec / n, 'any_hole_top1': anyh / n, 'exact_set': exact / n, 'planning_sufficiency': suff / n, 'chance_recall_at_k': kk / (LEVEL * LEVEL), 'n_test': n}
            print(f'  k={kk} [{tag}] recall@k {rec / n:.3f} (chance {kk / 25:.3f})  any_hole {anyh / n:.3f}  exact {exact / n:.3f}  plan {suff / n:.3f}', flush=True)
    im = res['image']
    ks = sorted(im)
    res['verdict'] = {'recall_at_k_curve': {k: im[k]['recall_at_k'] for k in ks}, 'any_hole_curve': {k: im[k]['any_hole_top1'] for k in ks}, 'reading': "any_hole flat + recall@k falling => the gap is SET COMPLETION, not hazard perception (the handoff's revised spine). recall@k flat => no set gap; the spine is wrong again and the 0.76-vs-0.33 contrast was pure metric mismatch."}
    with open(out_path, 'w') as f:
        json.dump(res, f, indent=2, default=str)
    print(f'\nsaved -> {out_path}', flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=['gen', 'render', 'run'], required=True)
    ap.add_argument('--data', default='setsize_data.json')
    ap.add_argument('--img_dir', default='setsize_img')
    ap.add_argument('--ks', type=int, nargs='+', default=[1, 2, 3, 4, 5, 6])
    ap.add_argument('--n', type=int, default=400)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--layer', type=int, default=21)
    ap.add_argument('--out', default='results_probe_setsize.json')
    a = ap.parse_args()
    if a.stage == 'gen':
        stage_gen(a.ks, a.n, a.seed, a.data)
    else:
        data = json.load(open(a.data))
        stage_render(data, a.img_dir) if a.stage == 'render' else stage_run(data, a.img_dir, a.layer, a.out)
if __name__ == '__main__':
    main()
