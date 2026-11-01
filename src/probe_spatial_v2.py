"""Feature extraction and probe fitting with blank-image and shuffled-label controls."""
import json
import argparse
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from probe_spatial import load_instances, build_labels, train_probe, MODEL_ID, LEVEL, PROMPT
from env import Layout, compute_distance_map, step, DELTAS

@torch.no_grad()
def feat_from_image(model, processor, img, layer):
    content = ([{'type': 'image'}] if img is not None else []) + [{'type': 'text', 'text': PROMPT}]
    chat = processor.apply_chat_template([{'role': 'user', 'content': content}], tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[chat], images=[img] if img is not None else None, return_tensors='pt').to(model.device)
    hs = model(**inputs, output_hidden_states=True).hidden_states
    return hs[layer][0, -1, :].float().cpu().clone()

def train_binary_cols(Xtr, Ytr, Xte, steps=500, lr=0.01):
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
        return Xte @ W + b > 0

def hole_f1(pred, true):
    tp = (pred & (true > 0.5)).sum().item()
    fp = (pred & (true < 0.5)).sum().item()
    fn = (~pred & (true > 0.5)).sum().item()
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return (prec, rec, f1)

def greedy_path(lay):
    dm = compute_distance_map(lay)
    if lay.start not in dm:
        return None
    pos, path = (lay.start, [])
    while pos != lay.goal and len(path) < 4 * lay.level * lay.level:
        best = None
        for a, (dr, dc) in DELTAS.items():
            npos = (pos[0] + dr, pos[1] + dc)
            if npos in dm and (best is None or dm[npos] < dm[best[1]]):
                best = (a, npos)
        if best is None:
            return None
        path.append(best[0])
        pos = best[1]
    return path

def executes_in_true(true_lay, path):
    if path is None:
        return False
    pos = true_lay.start
    if pos == true_lay.goal:
        return True
    for a in path:
        nxt, ev = step(true_lay, pos, a)
        if ev == 'hole':
            return False
        if ev == 'invalid':
            continue
        pos = nxt
        if ev == 'goal':
            return True
    return False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gen_dir', default='vsp_train6k')
    ap.add_argument('--n', type=int, default=800)
    ap.add_argument('--layer', type=int, default=21)
    ap.add_argument('--out', default='results_probe_v2.json')
    args = ap.parse_args()
    print(f'loading {MODEL_ID} ...', flush=True)
    model = Qwen2VLForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map='auto')
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model.eval()
    insts = load_instances(args.gen_dir, args.n)
    print(f'loaded {len(insts)} level-{LEVEL} instances', flush=True)

    def extract(blank):
        feats = []
        for k, (img_path, _) in enumerate(insts):
            img = Image.new('RGB', (320, 320), (0, 0, 0)) if blank else Image.open(img_path).convert('RGB')
            feats.append(feat_from_image(model, processor, img, args.layer))
            if (k + 1) % 200 == 0:
                print(f"  {('blank' if blank else 'image')} {k + 1}/{len(insts)}", flush=True)
        return torch.stack(feats)
    cat, holes = build_labels(insts)
    ntr = int(0.8 * len(insts))
    out = {'model': MODEL_ID, 'layer': args.layer, 'n': len(insts)}
    for cond, blank in [('image', False), ('blank_control', True)]:
        X = extract(blank)
        Xtr, Xte = (X[:ntr], X[ntr:])
        pos_acc = {name: train_probe(Xtr, cat[name][:ntr], Xte, cat[name][ntr:], LEVEL) for name in cat}
        hole_pred = train_binary_cols(Xtr, holes[:ntr], Xte)
        prec, rec, f1 = hole_f1(hole_pred, holes[ntr:])
        preds = {name: _predict(Xtr, cat[name][:ntr], Xte, LEVEL) for name in cat}
        test_insts = insts[ntr:]
        suff = 0
        for i, (_, true_lay) in enumerate(test_insts):
            dl = Layout(LEVEL, frozenset(((r, c) for r in range(LEVEL) for c in range(LEVEL) if hole_pred[i, r * LEVEL + c])), (int(preds['goal_row'][i]), int(preds['goal_col'][i])), (int(preds['player_row'][i]), int(preds['player_col'][i])))
            if executes_in_true(true_lay, greedy_path(dl)):
                suff += 1
        suff_rate = suff / len(test_insts)
        out[cond] = {'pos_acc': pos_acc, 'hole_prec': prec, 'hole_rec': rec, 'hole_f1': f1, 'planning_sufficiency': suff_rate}
        print(f'[{cond}] pos {pos_acc}  hole P/R/F1={prec:.2f}/{rec:.2f}/{f1:.2f}  planning-suff={suff_rate:.2f} (chance pos={1 / LEVEL:.2f})', flush=True)
    with open(args.out, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'saved -> {args.out}', flush=True)

def _predict(Xtr, ytr, Xte, n_cls, steps=400, lr=0.01):
    H = Xtr.shape[1]
    W = torch.zeros(H, n_cls, requires_grad=True)
    b = torch.zeros(n_cls, requires_grad=True)
    opt = torch.optim.Adam([W, b], lr=lr, weight_decay=0.001)
    for _ in range(steps):
        loss = F.cross_entropy(Xtr @ W + b, ytr)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        return (Xte @ W + b).argmax(1)
if __name__ == '__main__':
    main()
