"""Method 1 regenerated with vLLM at a longer token budget."""
import argparse
import base64
import io
import json
import os
import re
import readout_ladder as rl
MOVES = ['U', 'D', 'L', 'R']

def parse(txt):
    """readout_ladder's method-1 parser, verbatim."""
    m = re.search('ANSWER:\\s*([UDLR]+)', txt, re.I) or re.search('\\b([UDLR]{2,})\\b', txt)
    return MOVES.index(m.group(1)[0].upper()) if m else -1

def data_url(path):
    with open(path, 'rb') as f:
        return 'data:image/png;base64,' + base64.b64encode(f.read()).decode()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, choices=list(rl.MODELS))
    ap.add_argument('--dataset', choices=['vsp'], default='vsp')
    ap.add_argument('--max-tokens', type=int, default=16384)
    ap.add_argument('--limit', type=int, default=0, help='first N instances only (smoke)')
    ap.add_argument('--tp', type=int, default=2, help='tensor-parallel GPUs')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    meta = {}
    items = [(p, lay) for lv in (3, 4, 5, 6, 7, 8) for _, lay, p in rl.load_level(lv) if rl.optimal_set(lay)]
    if a.limit:
        items = items[:a.limit]
    from vllm import LLM, SamplingParams
    mid = rl.MODELS[a.model][0]
    llm = LLM(model=mid, tensor_parallel_size=a.tp, max_model_len=a.max_tokens + 4096, limit_mm_per_prompt={'image': 1}, seed=0)
    sp = SamplingParams(temperature=0.0, max_tokens=a.max_tokens)
    msgs = []
    for path, lay in items:
        _, plan = rl._prompts(lay, 'hidden')
        msgs.append([{'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': data_url(path)}}, {'type': 'text', 'text': plan}]}])
    kw = {'chat_template_kwargs': {'enable_thinking': False}} if 'Qwen3.' in mid else {}
    outs = llm.chat(msgs, sp, **kw)
    texts = [o.outputs[0].text for o in outs]
    ntok = [len(o.outputs[0].token_ids) for o in outs]
    free = [parse(t) for t in texts]
    trunc = [n >= a.max_tokens for n in ntok]
    opts = [rl.optimal_set(lay) for _, lay in items]
    acc = sum((int(f in o) for f, o in zip(free, opts))) / len(items)
    rec = {'model': mid, 'engine': 'vllm', 'dataset': a.dataset, 'max_tokens': a.max_tokens, 'n': len(items), 'unparsed': sum((f < 0 for f in free)), 'truncated': sum(trunc), 'tokens_median': sorted(ntok)[len(ntok) // 2], 'tokens_max': max(ntok), 'accuracy_per_instance': acc, 'per_instance': {'1c_written_plan_long': free}, 'truncated_per_instance': trunc, 'texts': texts}
    with open(a.out, 'w') as f:
        json.dump(rec, f)
    print(f"{mid} {a.dataset}: n {len(items)}  unparsed {rec['unparsed']}  truncated {rec['truncated']}  tokens median {rec['tokens_median']} max {rec['tokens_max']}  accuracy {acc:.3f}", flush=True)
if __name__ == '__main__':
    main()
