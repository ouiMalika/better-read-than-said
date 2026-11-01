"""Fit methods 3-5 from a stored checkpoint of hidden states."""
import argparse, json, os, re, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from readout_ladder import optimal_set, baselines, MOVES, ckpt_key
from vsp_maze import load_level

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--model', required=True, help='hf id to record in the output')
    ap.add_argument('--layer', type=int, required=True)
    ap.add_argument('--position', default='hidden')
    ap.add_argument('--levels', type=int, nargs='+', default=[3, 4, 5, 6, 7, 8])
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--max-new', type=int, default=512)
    ap.add_argument('--dataset', choices=['vsp'], default='vsp')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    cache = torch.load(a.ckpt, weights_only=False)
    ck = {ckpt_key(p): v for p, v in cache.items()}
    items = [(p, lay) for lv in a.levels for _, lay, p in load_level(lv) if optimal_set(lay)]
    kept = [(ckpt_key(p), lay) for p, lay in items if ckpt_key(p) in ck]
    cache = ck
    print(f'checkpoint holds {len(cache)} entries; {len(kept)} of {len(items)} instances usable', flush=True)
    if len(kept) < 100:
        sys.exit('too few cached instances to fit')
    lays = [l for _, l in kept]
    opts = [optimal_set(l) for l in lays]
    feats = torch.stack([cache[p]['h'] for p, _ in kept])
    forced = [cache[p]['mv'] for p, _ in kept]
    free = [cache[p]['free'] for p, _ in kept]
    trunc = [cache[p]['trunc'] for p, _ in kept]
    import readout_ladder as RL
    res = RL.fit_rungs(feats, lays, opts, forced, free, trunc, a)
    tag = 'pooled'
    out = {'model': a.model, 'layer': a.layer, 'levels': a.levels, 'pooled': True, 'position': a.position, 'max_new': a.max_new, 'dataset': a.dataset, 'seed': a.seed, 'note': f'fitted from checkpoint on {len(kept)} of {len(items)} instances', 'rungs': {tag: res}}
    json.dump(out, open(a.out, 'w'), indent=2)
    print('saved ->', a.out)
    print(json.dumps(res, indent=2))
if __name__ == '__main__':
    main()
