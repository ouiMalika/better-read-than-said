"""Join per-level checkpoints of one VSP run into a single pooled checkpoint."""
import argparse, glob, os, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from readout_ladder import ckpt_key
ap = argparse.ArgumentParser()
ap.add_argument('--parts', nargs='+', required=True, help='checkpoint files to join')
ap.add_argument('--out', required=True)
a = ap.parse_args()
merged = {}
for p in a.parts:
    c = {ckpt_key(k): v for k, v in torch.load(p, weights_only=False).items()}
    dup = set(c) & set(merged)
    if dup:
        sys.exit(f'{p} repeats {len(dup)} instances already joined, e.g. {sorted(dup)[:3]}')
    print(f'{p}: {len(c)} instances', flush=True)
    merged.update(c)
torch.save(merged, a.out)
print(f'{len(merged)} instances -> {a.out}')
