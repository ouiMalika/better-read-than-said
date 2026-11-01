"""Matched 25-way localization of agent, goal and hazard on single-hole mazes."""
import json
import random
import argparse
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from env import Layout, compute_distance_map
from gen_vsp_data import render_img, to_desc
from probe_spatial import MODEL_ID, LEVEL
from probe_spatial_v2 import feat_from_image, _predict
C = LEVEL * LEVEL

def gen_single_hole(rng):
    cells = [(r, c) for r in range(LEVEL) for c in range(LEVEL)]
    while True:
        start, goal, hole = rng.sample(cells, 3)
        lay = Layout(LEVEL, frozenset([hole]), goal, start)
        if start in compute_distance_map(lay):
            return lay

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=500)
    ap.add_argument('--layer', type=int, default=21)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='results_probe_singlehole.json')
    args = ap.parse_args()
    rng = random.Random(args.seed)
    print(f'loading {MODEL_ID} ...', flush=True)
    model = Qwen2VLForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map='auto')
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model.eval()
    lays = [gen_single_hole(rng) for _ in range(args.n)]
    print(f'generated {len(lays)} single-hole solvable level-{LEVEL} mazes', flush=True)
    X, ya, yg, yh = ([], [], [], [])
    for k, lay in enumerate(lays):
        img = render_img(to_desc(lay)).convert('RGB')
        X.append(feat_from_image(model, processor, img, args.layer))
        ya.append(lay.start[0] * LEVEL + lay.start[1])
        yg.append(lay.goal[0] * LEVEL + lay.goal[1])
        (hr, hc), = tuple(lay.holes)
        yh.append(hr * LEVEL + hc)
        if (k + 1) % 100 == 0:
            print(f'  extract {k + 1}/{len(lays)}', flush=True)
    X = torch.stack(X)
    Y = {'agent': torch.tensor(ya), 'goal': torch.tensor(yg), 'hazard': torch.tensor(yh)}
    ntr = int(0.8 * len(lays))
    out = {'model': MODEL_ID, 'layer': args.layer, 'n': len(lays), 'chance': 1 / C, 'top1': {}}
    print('\n=== matched 25-way top-1 localization (single-hole, all roles singleton) ===', flush=True)
    for role, y in Y.items():
        acc = (_predict(X[:ntr], y[:ntr], X[ntr:], C) == y[ntr:]).float().mean().item()
        out['top1'][role] = acc
        print(f'  {role:7s}: top-1 = {acc:.2f}  (chance {1 / C:.3f})', flush=True)
    a, g, h = (out['top1']['agent'], out['top1']['goal'], out['top1']['hazard'])
    actors = 0.5 * (a + g)
    out['verdict'] = 'REAL asymmetry: hazard localizes far below actors when perfectly matched' if h < actors - 0.15 else 'ARTIFACT: hazard ~= actors once matched -> earlier gap was cardinality/metric'
    print(f'\nactors(mean)={actors:.2f}  hazard={h:.2f}', flush=True)
    print(f"VERDICT: {out['verdict']}", flush=True)
    with open(args.out, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'saved -> {args.out}', flush=True)
if __name__ == '__main__':
    main()
