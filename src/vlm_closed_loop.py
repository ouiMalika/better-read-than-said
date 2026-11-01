"""Closed-loop planning with execution feedback."""
import os
import re
import json
import random
import argparse
import torch
from PIL import Image
from transformers import AutoProcessor
from vsp_maze import load_level, VSP_ROOT
from env import step, compute_distance_map
VSP2ENV = {'U': 'up', 'D': 'down', 'L': 'left', 'R': 'right'}
MODEL_IDS = {'qwen': 'Qwen/Qwen2-VL-7B-Instruct', 'llava': 'llava-hf/llava-1.5-7b-hf'}

def load_model(name):
    mid = MODEL_IDS[name]
    if name == 'qwen':
        from transformers import Qwen2VLForConditionalGeneration
        m = Qwen2VLForConditionalGeneration.from_pretrained(mid, torch_dtype=torch.bfloat16, device_map='cuda')
    else:
        from transformers import LlavaForConditionalGeneration
        m = LlavaForConditionalGeneration.from_pretrained(mid, torch_dtype=torch.bfloat16, device_map='cuda')
    return (m, AutoProcessor.from_pretrained(mid), mid)
SYS = "You are controlling a player on a grid. Reach the goal without stepping on a hole. Moves: U=up (row-1), D=down (row+1), L=left (col-1), R=right (col+1). Row 1 is the top. I will execute each move and tell you the result. Propose exactly ONE move at a time. Reason in one short sentence, then end with a line 'MOVE: X' where X is one of U/D/L/R."

def wrong_cell(level, true_pos):
    """A random grid cell != true position, for size-matched FALSE state feedback."""
    while True:
        rc = (random.randrange(level), random.randrange(level))
        if rc != true_pos:
            return rc

def cell_state(lay, pos):
    if pos == lay.goal:
        return 'the goal'
    if pos in lay.holes:
        return 'a hole'
    return 'safe ground'

def parse_move(text):
    up = text.upper()
    m = re.search('MOVE[:\\s]*([UDLR])', up)
    if m:
        return m.group(1)
    m2 = re.findall('\\b([UDLR])\\b', up)
    return m2[-1] if m2 else None

def gen(model, processor, messages, images):
    chat = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[chat], images=images if images else None, return_tensors='pt').to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=128, do_sample=False)
    return processor.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()

def run_episode(model, processor, lay, img_path, use_image, budget, feedback='informative'):
    dm = compute_distance_map(lay)
    d0 = dm.get(lay.start, 1)
    ptext = f'The grid is {lay.level} by {lay.level}. The player starts at row {lay.start[0] + 1}, column {lay.start[1] + 1}. The goal is at row {lay.goal[0] + 1}, column {lay.goal[1] + 1}.'
    first_content = [{'type': 'text', 'text': SYS + '\n\n' + ptext + ' Propose your first move.'}]
    images = None
    if use_image:
        first_content = [{'type': 'image'}] + first_content
        images = [Image.open(img_path).convert('RGB')]
    messages = [{'role': 'user', 'content': first_content}]
    pos = lay.start
    illegal = 0
    attempts = 0
    repeat_illegal = 0
    last_illegal_from = {}
    reached = False
    for t in range(budget):
        out = gen(model, processor, messages, images)
        mv = parse_move(out)
        messages.append({'role': 'assistant', 'content': [{'type': 'text', 'text': out}]})
        attempts += 1
        if mv is None:
            fb = "I couldn't read a move. Reply with 'MOVE: X' (U/D/L/R)."
            messages.append({'role': 'user', 'content': [{'type': 'text', 'text': fb}]})
            continue
        nxt, ev = step(lay, pos, VSP2ENV[mv])
        if ev == 'hole':
            reached = False
            break
        if ev == 'invalid':
            illegal += 1
            if mv in last_illegal_from.get(pos, set()):
                repeat_illegal += 1
            last_illegal_from.setdefault(pos, set()).add(mv)
            if feedback == 'false':
                fr, fc = wrong_cell(lay.level, pos)
                fb = f'That move has no effect (edge of grid). You are still at row {fr + 1}, column {fc + 1}. Propose a different move.'
            elif feedback == 'informative':
                fb = f"'{mv}' has no effect: it would leave the grid. You are still at row {pos[0] + 1}, column {pos[1] + 1}. Propose a different move."
            else:
                fb = f'That move had no effect. You are still at row {pos[0] + 1}, column {pos[1] + 1}. Try again.'
        else:
            pos = nxt
            last_illegal_from[pos] = set()
            if ev == 'goal':
                reached = True
                break
            if feedback == 'false':
                fr, fc = wrong_cell(lay.level, pos)
                fb = f'You moved {mv}. Now at row {fr + 1}, column {fc + 1} ({cell_state(lay, (fr, fc))}). Next move?'
            else:
                fb = f'You moved {mv}. Now at row {pos[0] + 1}, column {pos[1] + 1} ({cell_state(lay, pos)}). Next move?'
        messages.append({'role': 'user', 'content': [{'type': 'text', 'text': fb}]})
    df = dm.get(pos, d0)
    progress = 1.0 if reached else max(0.0, (d0 - df) / d0) if d0 else 0.0
    return {'reached': reached, 'attempts': attempts, 'illegal': illegal, 'repeat_illegal': repeat_illegal, 'progress': progress}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--levels', default='3,5')
    ap.add_argument('--n', type=int, default=25)
    ap.add_argument('--modality', default='text', choices=['text', 'image'])
    ap.add_argument('--budget_mult', type=int, default=4, help='step budget = budget_mult * optimal, min 8')
    ap.add_argument('--budget_cap', type=int, default=30, help='hard cap on step budget per episode')
    ap.add_argument('--feedback', default='informative', choices=['informative', 'minimal', 'false'])
    ap.add_argument('--model', default='qwen', choices=list(MODEL_IDS))
    ap.add_argument('--out', default='results_vlm_closed_loop.json')
    args = ap.parse_args()
    levels = [int(x) for x in args.levels.split(',')]
    print(f'loading {MODEL_IDS[args.model]} ...', flush=True)
    model, processor, MODEL_ID = load_model(args.model)
    model.eval()
    res = {}
    for lv in levels:
        inst = load_level(lv)[:args.n]
        eps = []
        for k, (iid, lay, img_path) in enumerate(inst):
            opt = compute_distance_map(lay).get(lay.start, 4)
            budget = min(args.budget_cap, max(8, args.budget_mult * opt))
            e = run_episode(model, processor, lay, img_path, args.modality == 'image', budget, args.feedback)
            eps.append(e)
            if k < 3:
                print(f"[L{lv} {iid}] reached={e['reached']} attempts={e['attempts']} illegal={e['illegal']} repeat_illegal={e['repeat_illegal']} prog={e['progress']:.2f}", flush=True)
        succ = sum((e['reached'] for e in eps)) / len(eps)
        tot_ill = sum((e['illegal'] for e in eps))
        tot_rep = sum((e['repeat_illegal'] for e in eps))
        res[lv] = {'n': len(eps), 'modality': args.modality, 'feedback': args.feedback, 'success_rate': succ, 'mean_attempts': sum((e['attempts'] for e in eps)) / len(eps), 'mean_progress': sum((e['progress'] for e in eps)) / len(eps), 'illegal_total': tot_ill, 'repeat_illegal_total': tot_rep, 'repeat_illegal_frac': tot_rep / tot_ill if tot_ill else 0.0, 'episodes': eps}
        print(f"\n=== L{lv} closed-loop ({args.modality}): success {succ * 100:.0f}%  repeat-illegal {tot_rep}/{tot_ill} = {res[lv]['repeat_illegal_frac'] * 100:.0f}% ===", flush=True)
        with open(args.out, 'w') as f:
            json.dump({'model': MODEL_ID, 'levels': res}, f, indent=2)
    print('saved ->', args.out, flush=True)
if __name__ == '__main__':
    main()
