"""Move selection masked by a verifier (remedy battery)."""
import re
import json
import argparse
import torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from vsp_maze import load_level
from env import step, compute_distance_map
MOVES = ['U', 'D', 'L', 'R']
VSP2ENV = {'U': 'up', 'D': 'down', 'L': 'left', 'R': 'right'}
MODEL_ID = 'Qwen/Qwen2-VL-7B-Instruct'
INSTR = 'You control a player on a square grid. Reach the goal without stepping on a hole. Moves: U=up (row-1), D=down (row+1), L=left (col-1), R=right (col+1); row 1 is the top. Answer with EXACTLY ONE letter, one of U/D/L/R, and nothing else.'

def legal_mask(lay, pos):
    m = []
    for mv in MOVES:
        _, ev = step(lay, pos, VSP2ENV[mv])
        m.append(ev not in ('invalid', 'hole'))
    return torch.tensor(m)

def move_token_ids(tok):
    ids = []
    for mv in MOVES:
        cands = {tok.encode(mv, add_special_tokens=False)[0], tok.encode(' ' + mv, add_special_tokens=False)[0]}
        ids.append(list(cands))
    return ids

@torch.no_grad()
def move_logits(model, processor, tok_ids, state_text, img):
    content = ([{'type': 'image'}] if img is not None else []) + [{'type': 'text', 'text': INSTR + '\n' + state_text}]
    chat = processor.apply_chat_template([{'role': 'user', 'content': content}], tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[chat], images=[img] if img is not None else None, return_tensors='pt').to(model.device)
    logits = model(**inputs).logits[0, -1, :].float().cpu()
    return torch.tensor([max((logits[i].item() for i in ids)) for ids in tok_ids])

def run_episode(model, processor, tok_ids, lay, img_path, use_image, budget, constrained):
    dm = compute_distance_map(lay)
    d0 = dm.get(lay.start, 1)
    img = Image.open(img_path).convert('RGB') if use_image else None
    pos, reached, illegal_top = (lay.start, False, 0)
    for _ in range(budget):
        state = f'The grid is {lay.level}x{lay.level}. You are at row {pos[0] + 1}, column {pos[1] + 1}. The goal is at row {lay.goal[0] + 1}, column {lay.goal[1] + 1}. Your move?'
        ml = move_logits(model, processor, tok_ids, state, img)
        mask = legal_mask(lay, pos)
        if not mask.any():
            break
        top_illegal = not mask[int(ml.argmax())]
        illegal_top += int(top_illegal)
        if constrained:
            ml = ml.masked_fill(~mask, float('-inf'))
        mv = MOVES[int(ml.argmax())]
        nxt, ev = step(lay, pos, VSP2ENV[mv])
        if ev == 'hole':
            break
        if ev == 'invalid':
            continue
        pos = nxt
        if ev == 'goal':
            reached = True
            break
    prog = 1.0 if reached else max(0.0, (d0 - dm.get(pos, d0)) / d0) if d0 else 0.0
    return {'reached': reached, 'illegal_top': illegal_top, 'progress': prog}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--levels', default='3,4,5,6,7,8')
    ap.add_argument('--n', type=int, default=50)
    ap.add_argument('--modality', default='image', choices=['image', 'text'])
    ap.add_argument('--mode', default='constrained', choices=['free', 'constrained'])
    ap.add_argument('--budget_cap', type=int, default=25)
    ap.add_argument('--out', default='results_vlm_remedy.json')
    args = ap.parse_args()
    levels = [int(x) for x in args.levels.split(',')]
    print(f'loading {MODEL_ID} ...', flush=True)
    model = Qwen2VLForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map='auto')
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model.eval()
    tok_ids = move_token_ids(processor.tokenizer)
    res = {}
    for lv in levels:
        inst = load_level(lv)[:args.n]
        eps = []
        for iid, lay, img_path in inst:
            opt = compute_distance_map(lay).get(lay.start, 4)
            budget = min(args.budget_cap, max(8, 4 * opt))
            eps.append(run_episode(model, processor, tok_ids, lay, img_path, args.modality == 'image', budget, args.mode == 'constrained'))
        succ = sum((e['reached'] for e in eps)) / len(eps)
        ill = sum((e['illegal_top'] for e in eps))
        res[lv] = {'n': len(eps), 'modality': args.modality, 'mode': args.mode, 'success_rate': succ, 'mean_progress': sum((e['progress'] for e in eps)) / len(eps), 'illegal_top_total': ill, 'illegal_top_note': "times the model's unmasked top choice was illegal; under mode=constrained those moves were masked before execution, so this counts the preference, not a move that ran"}
        print(f'L{lv} [{args.modality}/{args.mode}]: success {succ * 100:.0f}%  illegal-top preference {ill}', flush=True)
        with open(args.out, 'w') as f:
            json.dump({'model': MODEL_ID, 'levels': res}, f, indent=2)
    print('saved ->', args.out, flush=True)
if __name__ == '__main__':
    main()
