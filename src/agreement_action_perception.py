"""Compare the model's first move with the optimal move for the decoded map and for the true map."""
import json
import argparse
import torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from env import Layout, compute_distance_map, step
from probe_spatial import load_instances, build_labels, MODEL_ID, LEVEL
from probe_spatial_v2 import feat_from_image, _predict, train_binary_cols
from vlm_remedy import move_logits, move_token_ids, MOVES, VSP2ENV
ACT_PROMPT = 'You are the elf. Reach the gift without stepping on an ice hole. What is your first move?'

def optimal_actions(lay):
    dm = compute_distance_map(lay)
    if lay.start not in dm:
        return set()
    d0 = dm[lay.start]
    opt = set()
    for mv in MOVES:
        nxt, ev = step(lay, lay.start, VSP2ENV[mv])
        if ev not in ('invalid', 'hole') and dm.get(nxt, 10 ** 9) == d0 - 1:
            opt.add(mv)
    return opt

def legal_actions(lay):
    out = set()
    for mv in MOVES:
        _, ev = step(lay, lay.start, VSP2ENV[mv])
        if ev not in ('invalid', 'hole'):
            out.add(mv)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gen_dir', default='vsp_train6k')
    ap.add_argument('--n', type=int, default=1200)
    ap.add_argument('--layer', type=int, default=21)
    args = ap.parse_args()
    print(f'loading {MODEL_ID} ...', flush=True)
    model = Qwen2VLForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map='auto')
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model.eval()
    tok_ids = move_token_ids(processor.tokenizer)
    insts = load_instances(args.gen_dir, args.n)
    print(f'loaded {len(insts)} level-{LEVEL} instances', flush=True)
    X, actual = ([], [])
    for k, (img_path, lay) in enumerate(insts):
        img = Image.open(img_path).convert('RGB')
        X.append(feat_from_image(model, processor, img, args.layer))
        ml = move_logits(model, processor, tok_ids, ACT_PROMPT, img)
        actual.append(MOVES[int(ml.argmax())])
        if (k + 1) % 100 == 0:
            print(f'  {k + 1}/{len(insts)}', flush=True)
    X = torch.stack(X)
    cat, holes = build_labels(insts)
    ntr = int(0.8 * len(insts))
    pr = _predict(X[:ntr], cat['player_row'][:ntr], X[ntr:], LEVEL)
    pc = _predict(X[:ntr], cat['player_col'][:ntr], X[ntr:], LEVEL)
    gr = _predict(X[:ntr], cat['goal_row'][:ntr], X[ntr:], LEVEL)
    gc = _predict(X[:ntr], cat['goal_col'][:ntr], X[ntr:], LEVEL)
    hole_pred = train_binary_cols(X[:ntr], holes[:ntr], X[ntr:])
    test = insts[ntr:]
    agree_true = agree_dec = chance = solv = both = 0
    decoded_chance = decoded_blocked_start = decoded_blocked_goal = 0
    n = len(test)
    for i, (_, true_lay) in enumerate(test):
        a = actual[ntr + i]
        dec = Layout(LEVEL, frozenset(((r, c) for r in range(LEVEL) for c in range(LEVEL) if hole_pred[i, r * LEVEL + c])), (int(gr[i]), int(gc[i])), (int(pr[i]), int(pc[i])))
        opt_t = optimal_actions(true_lay)
        opt_d = optimal_actions(dec)
        decoded_chance += len(opt_d) / 4
        decoded_blocked_start += int(dec.start in dec.holes)
        decoded_blocked_goal += int(dec.goal in dec.holes)
        leg = legal_actions(true_lay)
        agree_true += int(a in opt_t)
        agree_dec += int(a in opt_d)
        chance += len(opt_t) / max(len(leg), 1)
        solv += int(len(opt_d) > 0)
        if opt_t and opt_d:
            both += int((a in opt_t) == (a in opt_d))
    out = {'model': MODEL_ID, 'layer': args.layer, 'n_test': n, 'action_agreement_true_map': agree_true / n, 'action_agreement_decoded_map': agree_dec / n, 'random_legal_baseline': chance / n, 'decoded_uniform4_baseline': decoded_chance / n, 'decoded_agreement_given_solvable': agree_dec / solv if solv else None, 'decoded_start_in_hole': decoded_blocked_start, 'decoded_goal_in_hole': decoded_blocked_goal, 'decoded_map_solvable_frac': solv / n}
    print('\n=== ACTION vs PERCEPTION (image-only actions) ===', flush=True)
    print(f"agreement with TRUE-map optimal policy    : {out['action_agreement_true_map']:.2f}", flush=True)
    print(f"agreement with DECODED-map optimal policy : {out['action_agreement_decoded_map']:.2f}", flush=True)
    print(f"random-legal baseline                     : {out['random_legal_baseline']:.2f}", flush=True)
    print(f"decoded map solvable                      : {out['decoded_map_solvable_frac']:.2f}", flush=True)
    print('Compare decoded agreement to its decoded-map baseline and solvability ceiling. The true-map random-legal baseline does not test decoded-map agreement.', flush=True)
    with open('results_agreement.json', 'w') as f:
        json.dump(out, f, indent=2)
    print('saved -> results_agreement.json', flush=True)
if __name__ == '__main__':
    main()
