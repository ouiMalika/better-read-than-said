"""Method 1 with the answer forced at the token budget, from stored generations."""
import argparse
import json
import os
import torch
from PIL import Image
import readout_ladder as rl
from vlm_remedy import move_token_ids

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, choices=list(rl.MODELS))
    ap.add_argument('--dataset', choices=['vsp'], default='vsp')
    ap.add_argument('--levels', type=int, nargs='+', default=[3, 4, 5, 6, 7, 8], help='vsp: the pooled levels of the run')
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    cache = torch.load(a.ckpt, weights_only=False)
    sha = cache.pop('__pairs_sha__', None)
    meta = {}
    items = [(rl.ckpt_key(p), p, lay) for lv in a.levels for _, lay, p in rl.load_level(lv) if rl.optimal_set(lay)]
    cache = {rl.ckpt_key(k): v for k, v in cache.items()}
    missing = [key for key, _, _ in items if key not in cache]
    if missing:
        raise SystemExit(f'{len(missing)} instances not in {a.ckpt}, e.g. {missing[0]}')
    model, processor, family, _ = rl.load_model(a.model)
    if family == 'dsvl2':
        raise SystemExit('DeepSeek-VL2 is not supported here')
    tok_ids = move_token_ids(processor.tokenizer)
    out, n_forced = ([], 0)
    for key, img_path, lay in items:
        c = cache[key]
        if c['free'] >= 0:
            out.append(int(c['free']))
            continue
        _, plan = rl._prompts(lay, 'hidden')
        msg = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': plan}]}]
        chat = processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True, **rl._CHAT_KW)
        text = chat + c['txt'].rstrip() + '\n\nANSWER:'
        img = Image.open(img_path).convert('RGB')
        inp = processor(text=[text], images=[img], return_tensors='pt').to(model.device)
        with torch.no_grad():
            lg = model(**inp).logits[0, -1, :].float().cpu()
        mv = torch.tensor([max((lg[i].item() for i in ids)) for ids in tok_ids])
        out.append(int(mv.argmax()))
        n_forced += 1
        if n_forced % 50 == 0:
            print(f'  forced {n_forced}', flush=True)
    opts = [rl.optimal_set(lay) for _, _, lay in items]
    meta['forced_at_budget_accuracy'] = sum((int(m in o) for m, o in zip(out, opts))) / len(out)
    print(f"VSP method 1 with the answer forced at the budget: {meta['forced_at_budget_accuracy']:.3f} on {len(out)}", flush=True)
    with open(a.out, 'w') as f:
        json.dump({'model': rl.MODELS[a.model][0], 'dataset': a.dataset, 'forced_at_budget_accuracy': meta.get('forced_at_budget_accuracy'), 'ckpt': a.ckpt, 'n_forced': n_forced, 'per_instance': {'1b_answer_forced_at_budget': out}}, f)
    print(f'forced {n_forced} of {len(items)} -> {a.out}', flush=True)
if __name__ == '__main__':
    main()
