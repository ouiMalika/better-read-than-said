"""Linear probes for grid contents on frozen VLM activations."""
import os
import glob
import json
import argparse
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from vsp_maze import parse_pure_text
MODEL_ID = 'Qwen/Qwen2-VL-7B-Instruct'
LEVEL = 5
PROMPT = 'Here is a grid navigation map with a player, a goal, and holes.'

def load_instances(gen_dir, n):
    out = []
    for tf in sorted(glob.glob(os.path.join(gen_dir, 'text', '*.txt')), key=lambda p: int(os.path.basename(p)[:-4]))[:n]:
        i = os.path.basename(tf)[:-4]
        lay = parse_pure_text(open(tf).read())
        img = os.path.join(gen_dir, 'img', f'{i}.png')
        if lay.level == LEVEL and os.path.exists(img):
            out.append((img, lay))
    return out

@torch.no_grad()
def features(model, processor, img_path, layers, use_image):
    content = ([{'type': 'image'}] if use_image else []) + [{'type': 'text', 'text': PROMPT}]
    chat = processor.apply_chat_template([{'role': 'user', 'content': content}], tokenize=False, add_generation_prompt=True)
    imgs = [Image.open(img_path).convert('RGB')] if use_image else None
    inputs = processor(text=[chat], images=imgs, return_tensors='pt').to(model.device)
    out = model(**inputs, output_hidden_states=True)
    hs = out.hidden_states
    return torch.stack([hs[l][0, -1, :].float().cpu() for l in layers])

def train_probe(Xtr, ytr, Xte, yte, n_cls, steps=400, lr=0.01):
    """Linear multiclass probe; returns test accuracy."""
    H = Xtr.shape[1]
    W = torch.zeros(H, n_cls, requires_grad=True)
    b = torch.zeros(n_cls, requires_grad=True)
    opt = torch.optim.Adam([W, b], lr=lr, weight_decay=0.001)
    for _ in range(steps):
        logits = Xtr @ W + b
        loss = F.cross_entropy(logits, ytr)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        acc = ((Xte @ W + b).argmax(1) == yte).float().mean().item()
    return acc

def train_multilabel(Xtr, Ytr, Xte, Yte, steps=400, lr=0.01):
    """Independent binary probes (hole map); returns per-cell accuracy."""
    H, K = (Xtr.shape[1], Ytr.shape[1])
    W = torch.zeros(H, K, requires_grad=True)
    b = torch.zeros(K, requires_grad=True)
    opt = torch.optim.Adam([W, b], lr=lr, weight_decay=0.001)
    for _ in range(steps):
        loss = F.binary_cross_entropy_with_logits(Xtr @ W + b, Ytr)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred = Xte @ W + b > 0
        return (pred == (Yte > 0.5)).float().mean().item()

def build_labels(insts):
    pr = torch.tensor([lay.start[0] for _, lay in insts])
    pc = torch.tensor([lay.start[1] for _, lay in insts])
    gr = torch.tensor([lay.goal[0] for _, lay in insts])
    gc = torch.tensor([lay.goal[1] for _, lay in insts])
    holes = torch.zeros(len(insts), LEVEL * LEVEL)
    for i, (_, lay) in enumerate(insts):
        for hr, hc in lay.holes:
            holes[i, hr * LEVEL + hc] = 1.0
    return ({'player_row': pr, 'player_col': pc, 'goal_row': gr, 'goal_col': gc}, holes)

def run_modality(model, processor, insts, layers, use_image, tag):
    print(f'\n=== extracting {tag} features (n={len(insts)}) ===', flush=True)
    feats = []
    for k, (img, _) in enumerate(insts):
        feats.append(features(model, processor, img, layers, use_image))
        if (k + 1) % 200 == 0:
            print(f'  {k + 1}/{len(insts)}', flush=True)
    X = torch.stack(feats)
    cat_labels, holes = build_labels(insts)
    N = X.shape[0]
    ntr = int(0.8 * N)
    hole_base = max(holes.mean().item(), 1 - holes.mean().item())
    res = {'n': N, 'layers': layers, 'hole_base_rate': hole_base, 'per_layer': {}}
    for li, L in enumerate(layers):
        XL = X[:, li, :]
        Xtr, Xte = (XL[:ntr], XL[ntr:])
        row = {}
        for name, y in cat_labels.items():
            row[name] = train_probe(Xtr, y[:ntr], Xte, y[ntr:], LEVEL)
        row['holes'] = train_multilabel(Xtr, holes[:ntr], Xte, holes[ntr:])
        res['per_layer'][L] = row
        print(f"  layer {L}: player r/c={row['player_row']:.2f}/{row['player_col']:.2f} goal r/c={row['goal_row']:.2f}/{row['goal_col']:.2f} holes={row['holes']:.2f} (chance: pos={1 / LEVEL:.2f}, holes={hole_base:.2f})", flush=True)
    return res

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gen_dir', default='vsp_train6k')
    ap.add_argument('--n', type=int, default=1500)
    ap.add_argument('--layers', default='7,14,21,28')
    ap.add_argument('--out', default='results_probe_spatial.json')
    args = ap.parse_args()
    layers = [int(x) for x in args.layers.split(',')]
    print(f'loading {MODEL_ID} ...', flush=True)
    model = Qwen2VLForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map='auto')
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model.eval()
    insts = load_instances(args.gen_dir, args.n)
    print(f'loaded {len(insts)} level-{LEVEL} instances', flush=True)
    out = {'model': MODEL_ID, 'chance_pos': 1 / LEVEL}
    out['image'] = run_modality(model, processor, insts, layers, True, 'IMAGE')
    out['text'] = run_modality(model, processor, insts, layers, False, 'TEXT (control)')
    with open(args.out, 'w') as f:
        json.dump(out, f, indent=2)
    print('\n=== VERDICT ===', flush=True)
    best_img = max(out['image']['per_layer'].values(), key=lambda r: r['player_row'])
    print(f"IMAGE best-layer player-row probe = {best_img['player_row']:.2f} vs chance {1 / LEVEL:.2f}", flush=True)
    print('If IMAGE probes >> chance: the 2D map is linearly present in activations (present-but-unused = dissociation).', flush=True)
    print(f'saved -> {args.out}', flush=True)
if __name__ == '__main__':
    main()
