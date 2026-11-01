"""The five methods for getting a next move from one forward pass, on the VSP mazes."""
import re
import json
import argparse
from env import Layout, step, compute_distance_map
from vsp_maze import load_level
MOVES = ['U', 'D', 'L', 'R']
VSP2ENV = {'U': 'up', 'D': 'down', 'L': 'left', 'R': 'right'}
MODELS = {'qwen': ('Qwen/Qwen2-VL-7B-Instruct', 'hf', 'Qwen2VLForConditionalGeneration', 21), 'llava': ('llava-hf/llava-1.5-7b-hf', 'hf', 'LlavaForConditionalGeneration', 21), 'deepseek': ('deepseek-ai/deepseek-vl2', 'dsvl2', None, None), 'qwen25_3b': ('Qwen/Qwen2.5-VL-3B-Instruct', 'hf', 'Qwen2_5_VLForConditionalGeneration', None), 'qwen25_7b': ('Qwen/Qwen2.5-VL-7B-Instruct', 'hf', 'Qwen2_5_VLForConditionalGeneration', None), 'qwen32': ('Qwen/Qwen2.5-VL-32B-Instruct', 'hf', 'Qwen2_5_VLForConditionalGeneration', None), 'qwen38': ('Qwen/Qwen3.8-27B-FP8', 'hf', 'AutoModelForImageTextToText', None), 'qwen3vl32': ('Qwen/Qwen3-VL-32B-Instruct-FP8', 'hf', 'AutoModelForImageTextToText', None)}
_CHAT_KW = {}

def load_model(tag):
    import torch
    mid, family, cls, layer = MODELS[tag]
    if family == 'dsvl2':
        import sys
        try:
            import attrdict
        except Exception:
            import attrdict3
            sys.modules['attrdict'] = attrdict3
        from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
        proc = DeepseekVLV2Processor.from_pretrained(mid)
        m = DeepseekVLV2ForCausalLM.from_pretrained(mid, torch_dtype=torch.bfloat16, device_map='auto', trust_remote_code=True)
        m.eval()
        n = m.language.config.num_hidden_layers
        return (m, proc, family, layer or int(round(0.75 * n)))
    import transformers, os
    Klass = getattr(transformers, cls)
    dev = os.environ.get('VVP_DEVICE', 'auto')
    kw = {}
    if 'FP8' in mid:
        from transformers import FineGrainedFP8Config
        kw['quantization_config'] = FineGrainedFP8Config(dequantize=True)
    if 'Qwen3.' in mid:
        _CHAT_KW['enable_thinking'] = False
    if dev == 'cpu':
        m = Klass.from_pretrained(mid, dtype=torch.float32, device_map='cpu', **kw)
    else:
        m = Klass.from_pretrained(mid, torch_dtype=torch.bfloat16, device_map='auto', **kw)
    m.eval()
    proc = transformers.AutoProcessor.from_pretrained(mid)
    if layer is None:
        tc = getattr(m.config, 'text_config', m.config)
        layer = int(round(0.75 * tc.num_hidden_layers))
    return (m, proc, family, layer)

def _ds_inputs(model, processor, text, img):
    """DeepSeek-VL2's own conversation format + embedding path."""
    conv = [{'role': '<|User|>', 'content': f'<image>\n{text}', 'images': [img]}, {'role': '<|Assistant|>', 'content': ''}]
    inp = processor(conversations=conv, images=[img], force_batchify=True, system_prompt='').to(model.device)
    return (inp, model.prepare_inputs_embeds(**inp))

def build_inputs(processor, text, img):
    """One prompt-construction path for both families."""
    msg = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': text}]}]
    chat = processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True, **_CHAT_KW)
    return processor(text=[chat], images=[img], return_tensors='pt')

def feats_and_logits(model, processor, family, text, img, layer, tok_ids):
    """Hidden state at `layer` AND the four move logits, from ONE forward pass."""
    import torch
    with torch.no_grad():
        if family == 'dsvl2':
            inp, emb = _ds_inputs(model, processor, text, img)
            out = model.language(inputs_embeds=emb, attention_mask=inp.attention_mask, output_hidden_states=True)
        else:
            inp = build_inputs(processor, text, img).to(model.device)
            out = model(**inp, output_hidden_states=True)
    lg = out.logits[0, -1, :].float().cpu()
    move = torch.tensor([max((lg[i].item() for i in ids)) for ids in tok_ids])
    return (out.hidden_states[layer][0, -1, :].float().cpu().clone(), move)

def _ds_greedy(model, tok, emb, mask, max_new):
    """Greedy decode for DeepSeek-VL2 without generate(): step the LM directly, append each new token's embedding, and carry past_key_values so cost stays linear in the number of new tokens."""
    import torch
    lm = model.language
    embed = lm.get_input_embeddings()
    toks, past = ([], None)
    cur, cur_mask = (emb, mask)
    for _ in range(max_new):
        out = lm(inputs_embeds=cur, attention_mask=cur_mask, past_key_values=past, use_cache=True)
        past = out.past_key_values
        nxt = int(out.logits[0, -1].argmax())
        if nxt == tok.eos_token_id:
            break
        toks.append(nxt)
        if len(toks) % 4 == 0 and _answer_complete(tok.decode(toks, skip_special_tokens=True)):
            break
        cur = embed(torch.tensor([[nxt]], device=emb.device)).to(emb.dtype)
        cur_mask = torch.cat([cur_mask, torch.ones((1, 1), dtype=cur_mask.dtype, device=cur_mask.device)], dim=1)
    return tok.decode(toks, skip_special_tokens=True)
_ANSWER_DONE = re.compile('ANSWER:[\\s*`_:]*[UDLR]+[^UDLR]', re.I)

def _answer_complete(txt):
    """True once an ANSWER: run of moves exists AND has been terminated."""
    return bool(_ANSWER_DONE.search(txt))

def _stopper(tok, prompt_len):
    """StoppingCriteria that ends generation as soon as the answer line is complete."""
    try:
        from transformers import StoppingCriteria, StoppingCriteriaList
    except Exception:
        return None

    class _S(StoppingCriteria):

        def __call__(self, input_ids, scores, **kw):
            tail = input_ids[0][prompt_len:]
            if len(tail) < 8 or len(tail) % 4:
                return False
            return _answer_complete(tok.decode(tail, skip_special_tokens=True))
    return StoppingCriteriaList([_S()])

def _stopper_batch(tok, prompt_len):
    """Per-row version of _stopper."""
    try:
        from transformers import StoppingCriteria, StoppingCriteriaList
    except Exception:
        return None
    import torch

    class _S(StoppingCriteria):

        def __call__(self, input_ids, scores, **kw):
            tail = input_ids[:, prompt_len:]
            done = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
            if tail.shape[1] < 8 or tail.shape[1] % 4:
                return done
            for i in range(tail.shape[0]):
                done[i] = _answer_complete(tok.decode(tail[i], skip_special_tokens=True))
            return done
    return StoppingCriteriaList([_S()])

def generate_text_batch(model, processor, family, texts, imgs, max_new=128):
    """Batched rung 1."""
    import torch
    if len(texts) == 1:
        return [generate_text(model, processor, family, texts[0], imgs[0], max_new)]
    if family == 'dsvl2':
        return [generate_text(model, processor, family, t, im, max_new) for t, im in zip(texts, imgs)]
    tok = processor.tokenizer
    old_side = tok.padding_side
    tok.padding_side = 'left'
    try:
        chats = []
        for t in texts:
            msg = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': t}]}]
            chats.append(processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True, **_CHAT_KW))
        inp = processor(text=chats, images=list(imgs), return_tensors='pt', padding=True).to(model.device)
        plen = inp['input_ids'].shape[1]
        sc = _stopper_batch(tok, plen)
        kw = {'stopping_criteria': sc} if sc is not None else {}
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id or tok.eos_token_id, **kw)
        return [processor.decode(out[i][plen:], skip_special_tokens=True) for i in range(len(texts))]
    finally:
        tok.padding_side = old_side

def _prompts(lay, position):
    """The ONE shared world-description, and the two ways the answer is requested from it."""
    from vlm_remedy import INSTR
    ctx = f'The grid is {lay.level}x{lay.level}.'
    if position == 'given':
        ctx += f' You are at row {lay.start[0] + 1}, column {lay.start[1] + 1}.'
    state = INSTR + '\n' + ctx + ' What is your next move?'
    q = INSTR.replace('Answer with EXACTLY ONE letter, one of U/D/L/R, and nothing else.', "Give the full plan as a string of U/D/L/R after 'ANSWER:'.") + '\n' + ctx
    return (state, q)

def generate_text(model, processor, family, text, img, max_new=128):
    import torch
    tok = processor.tokenizer
    with torch.no_grad():
        if family == 'dsvl2':
            inp, emb = _ds_inputs(model, processor, text, img)
            try:
                out = model.language.generate(inputs_embeds=emb, attention_mask=inp.attention_mask, max_new_tokens=max_new, do_sample=False, use_cache=True, pad_token_id=tok.eos_token_id, eos_token_id=tok.eos_token_id)
                return tok.decode(out[0].cpu().tolist(), skip_special_tokens=True)
            except Exception:
                return _ds_greedy(model, tok, emb, inp.attention_mask, max_new)
        inp = build_inputs(processor, text, img).to(model.device)
        plen = inp['input_ids'].shape[1]
        sc = _stopper(tok, plen)
        kw = {'stopping_criteria': sc} if sc is not None else {}
        out = model.generate(**inp, max_new_tokens=max_new, do_sample=False, **kw)
        return processor.decode(out[0][plen:], skip_special_tokens=True)

def ckpt_key(path):
    """Checkpoint key that survives moving between machines."""
    m = re.search('level(\\d+)[/\\\\]img[/\\\\](\\d+)\\.png$', str(path))
    if m:
        return f'L{m.group(1)}#{m.group(2)}'
    m = re.search('(\\d+)_([AB])\\.png$', str(path))
    return f'E2#{m.group(1)}{m.group(2)}' if m else str(path)

def optimal_set(lay, pos=None):
    pos = pos or lay.start
    dm = compute_distance_map(lay)
    if pos not in dm:
        return frozenset()
    d0 = dm[pos]
    return frozenset((i for i, m in enumerate(MOVES) if (lambda t: t[1] not in ('invalid', 'hole') and dm.get(t[0], 10 ** 9) == d0 - 1)(step(lay, pos, VSP2ENV[m]))))

def goal_dir(start, goal):
    dr, dc = (goal[0] - start[0], goal[1] - start[1])
    if abs(dr) >= abs(dc):
        return 0 if dr < 0 else 1
    return 2 if dc < 0 else 3

def baselines(lays):
    n = len(lays)

    def ongrid(l):
        return [i for i, m in enumerate(MOVES) if step(l, l.start, VSP2ENV[m])[1] != 'invalid']

    def safe(l):
        return [i for i, m in enumerate(MOVES) if step(l, l.start, VSP2ENV[m])[1] not in ('invalid', 'hole')]
    return {'uniform_4': sum((len(optimal_set(l)) / 4 for l in lays)) / n, 'uniform_on_grid': sum((len(optimal_set(l)) / max(len(ongrid(l)), 1) for l in lays)) / n, 'uniform_hazard_aware': sum((len(optimal_set(l)) / max(len(safe(l)), 1) for l in lays)) / n, 'goal_direction_TRUE_coords': sum((1 for l in lays if goal_dir(l.start, l.goal) in optimal_set(l))) / n, 'best_constant': max((sum((1 for l in lays if i in optimal_set(l))) / n for i in range(4)))}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--levels', type=int, nargs='+', default=[5])
    ap.add_argument('--pool', action='store_true', help='pool all --levels into ONE fit (n=600 across 3-8) instead of per-level')
    ap.add_argument('--model', choices=list(MODELS), default='qwen')
    ap.add_argument('--layer', type=int, default=None)
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--seed', type=int, default=0, help='seed for probe initialization; recorded for reproducibility')
    ap.add_argument('--max-new', type=int, default=512, help="rung-1 generation budget; must exceed the model's answer length")
    ap.add_argument('--batch', type=int, default=1, help='instances per rung-1 generation batch (CPU runs want 4-8)')
    ap.add_argument('--rung1-only', action='store_true', help='re-measure free generation alone; skip the probe/head rungs')
    ap.add_argument('--position', choices=['hidden', 'given'], default='hidden', help="whether the agent's row/column is stated in the prompt (ALL rungs)")
    ap.add_argument('--dataset', choices=['vsp'], default='vsp')
    ap.add_argument('--out', default='results_readout_ladder.json')
    args = ap.parse_args()
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    from vlm_remedy import move_token_ids, INSTR
    MODEL_ID = MODELS[args.model][0]
    print(f'loading {MODEL_ID} ...', flush=True)
    model, processor, family, deflayer = load_model(args.model)
    layer = args.layer if args.layer is not None else deflayer
    tok_ids = move_token_ids(processor.tokenizer)
    res = {'model': MODEL_ID, 'layer': layer, 'levels': args.levels, 'seed': args.seed, 'pooled': bool(args.pool), 'position': args.position, 'max_new': args.max_new, 'dataset': args.dataset, 'rungs': {}}
    groups = [('pooled', args.levels)] if args.pool else [(lv, [lv]) for lv in args.levels]
    for tag, lvs in groups:
        items = [(p, lay) for lv in lvs for _, lay, p in load_level(lv) if optimal_set(lay)]
        print(f'{tag}: {len(items)} solvable instances', flush=True)
        lays = [l for _, l in items]
        opts = [optimal_set(l) for l in lays]
        import os as _os
        ck = _os.path.join(_os.path.dirname(_os.path.abspath(args.out)) or '.', f'.ckpt_{args.model}_{args.position}_{tag}_mx{args.max_new}_L{layer}.pt')
        try:
            try:
                cache = torch.load(ck, weights_only=False) if _os.path.exists(ck) else {}
            except TypeError:
                cache = torch.load(ck) if _os.path.exists(ck) else {}
            cache = {ckpt_key(k): v for k, v in cache.items()}
            if cache:
                print(f'  resuming: {len(cache)}/{len(items)} instances already cached', flush=True)
        except Exception as _e:
            print(f'  WARNING unreadable checkpoint ({type(_e).__name__}); starting fresh', flush=True)
            cache = {}

        def _flush():
            tmp = ck + '.tmp'
            torch.save(cache, tmp)
            _os.replace(tmp, ck)
        todo = [(p, l) for p, l in items if ckpt_key(p) not in cache]
        if todo:
            print(f'  {len(todo)} to compute, batch={args.batch}', flush=True)
        failed_chunks = []
        for c0 in range(0, len(todo), args.batch):
            chunk = todo[c0:c0 + args.batch]
            imgs = [Image.open(p).convert('RGB') for p, _ in chunk]
            pr = [_prompts(l, args.position) for _, l in chunk]
            hm = [feats_and_logits(model, processor, family, s, im, layer, tok_ids) for (s, _), im in zip(pr, imgs)]
            try:
                txts = generate_text_batch(model, processor, family, [q for _, q in pr], imgs, max_new=args.max_new)
            except Exception as e:
                print(f'  WARNING rung-1 generation failed on chunk at {c0}: {type(e).__name__}: {e}', flush=True)
                failed_chunks.append(c0)
                continue
            for (path, _), (h, mv), txt in zip(chunk, hm, txts):
                m = re.search('ANSWER:\\s*([UDLR]+)', txt, re.I) or re.search('\\b([UDLR]{2,})\\b', txt)
                cache[ckpt_key(path)] = {'h': h, 'mvlg': mv.clone(), 'mv': int(mv.argmax()), 'free': MOVES.index(m.group(1)[0].upper()) if m else -1, 'trunc': len(processor.tokenizer(txt).input_ids) >= args.max_new - 2, 'txt': txt}
            _flush()
            print(f'  {min(c0 + args.batch, len(todo))}/{len(todo)}', flush=True)
        _flush()
        have = [i for i, (p, _) in enumerate(items) if ckpt_key(p) in cache]
        missing = len(items) - len(have)
        if missing:
            raise RuntimeError(f'{missing} instances missing after failed chunks {failed_chunks}; checkpoint retained. Resume before scoring a matched cohort.')
        feats = [cache[ckpt_key(p)]['h'] for p, _ in items]
        forced = [cache[ckpt_key(p)]['mv'] for p, _ in items]
        free = [cache[ckpt_key(p)]['free'] for p, _ in items]
        trunc = [cache[ckpt_key(p)]['trunc'] for p, _ in items]
        N = len(lays)
        res.setdefault('incomplete', {})[str(tag)] = missing
        if args.rung1_only:
            res['rungs'][tag] = {'n': N, '1_free_generation': sum((int(free[i] in opts[i]) for i in range(N))) / N, '1_free_unparsed': sum((1 for f_ in free if f_ < 0)) / N, '1_free_truncated': sum((1 for t_ in trunc if t_)) / N, '2_forced_choice': sum((int(forced[i] in opts[i]) for i in range(N))) / N, 'max_new': args.max_new, 'baselines': baselines(lays), 'by_level': _by_level(lays, opts, free, forced, {})}
            print(json.dumps(res['rungs'][tag], indent=2), flush=True)
            continue
        res['rungs'][tag] = fit_rungs(torch.stack(feats), lays, opts, forced, free, trunc, args)
        print(json.dumps(res['rungs'][tag], indent=2), flush=True)
    with open(args.out, 'w') as fh:
        json.dump(res, fh, indent=2)
    print(f'saved -> {args.out}', flush=True)

def fit_rungs(X, lays, opts, forced, free, trunc, args):
    """Rungs 1-5 from cached features."""
    import torch
    import torch.nn.functional as F
    import random, statistics as st
    torch.manual_seed(getattr(args, 'seed', 0))
    N = len(lays)
    if not 2 <= args.folds <= N:
        raise ValueError('folds must be between 2 and the number of instances')
    gid = list(range(N))
    ngrp = max(gid) + 1 if N else 0
    if args.folds > ngrp:
        raise ValueError('fold count exceeds the number of independent groups')
    gfold = {g: g % args.folds for g in range(ngrp)}
    folds = [[i for i in range(N) if gfold[gid[i]] == f] for f in range(args.folds)]
    assert sum((len(f) for f in folds)) == N, 'fold assignment lost instances'

    def std(a, b):
        mu, sd = (a.mean(0, keepdim=True), a.std(0, keepdim=True).clamp_min(1e-06))
        return ((a - mu) / sd, (b - mu) / sd)

    def train_pos(Xtr, ytr, Xte, ncls):
        Xtr, Xte = std(Xtr, Xte)
        W = torch.zeros(Xtr.shape[1], ncls, requires_grad=True)
        b = torch.zeros(ncls, requires_grad=True)
        o = torch.optim.Adam([W, b], lr=0.01, weight_decay=0.001)
        for _ in range(600):
            l = F.cross_entropy(Xtr @ W + b, ytr)
            o.zero_grad()
            l.backward()
            o.step()
        with torch.no_grad():
            return (Xte @ W + b).argmax(1)

    def set_ce(lg, os_):
        lp = torch.log_softmax(lg, 1)
        return torch.stack([-torch.logsumexp(lp[i, sorted(s)], 0) for i, s in enumerate(os_)]).mean()

    def train_act(Xtr, otr, Xte, hidden=0, steps=1500):
        Xtr, Xte = std(Xtr, Xte)
        H = Xtr.shape[1]
        if hidden:
            W1 = (torch.randn(H, hidden) * (1 / H ** 0.5)).requires_grad_()
            b1 = torch.zeros(hidden, requires_grad=True)
            W2 = (torch.randn(hidden, 4) * (1 / hidden ** 0.5)).requires_grad_()
            b2 = torch.zeros(4, requires_grad=True)
            ps = [W1, b1, W2, b2]
            fwd = lambda x: torch.relu(x @ W1 + b1) @ W2 + b2
        else:
            W = torch.zeros(H, 4, requires_grad=True)
            b = torch.zeros(4, requires_grad=True)
            ps = [W, b]
            fwd = lambda x: x @ W + b
        o = torch.optim.Adam(ps, lr=0.001 if hidden else 0.01, weight_decay=0.001)
        for _ in range(steps):
            l = set_ce(fwd(Xtr), otr)
            o.zero_grad()
            l.backward()
            o.step()
        with torch.no_grad():
            return fwd(Xte).argmax(1)
    acc = {k: [] for k in ['probe_then_rule', 'linear_action', 'mlp_action', 'linear_action_shuffled', 'mlp_action_shuffled']}
    per_inst = {k: {} for k in ['probe_then_rule', 'linear_action', 'mlp_action']}
    pred_inst = {k: {} for k in ['probe_then_rule', 'linear_action', 'mlp_action']}
    import random
    for f in range(args.folds):
        te = folds[f]
        tr = [i for i in range(N) if i not in set(te)]
        ncls = max((l.level for l in lays))
        pr = train_pos(X[tr], torch.tensor([lays[i].start[0] for i in tr]), X[te], ncls)
        pc = train_pos(X[tr], torch.tensor([lays[i].start[1] for i in tr]), X[te], ncls)
        gr = train_pos(X[tr], torch.tensor([lays[i].goal[0] for i in tr]), X[te], ncls)
        gc = train_pos(X[tr], torch.tensor([lays[i].goal[1] for i in tr]), X[te], ncls)
        for j, i in enumerate(te):
            _m = goal_dir((int(pr[j]), int(pc[j])), (int(gr[j]), int(gc[j])))
            pred_inst['probe_then_rule'][i] = int(_m)
            per_inst['probe_then_rule'][i] = int(_m in opts[i])
        acc['probe_then_rule'].append(sum((int(goal_dir((int(pr[j]), int(pc[j])), (int(gr[j]), int(gc[j]))) in opts[i]) for j, i in enumerate(te))) / len(te))
        for name, hid in [('linear_action', 0), ('mlp_action', 512)]:
            p = train_act(X[tr], [opts[i] for i in tr], X[te], hidden=hid)
            for j, i in enumerate(te):
                pred_inst[name][i] = int(p[j])
                per_inst[name][i] = int(int(p[j]) in opts[i])
            acc[name].append(sum((int(int(p[j]) in opts[i]) for j, i in enumerate(te))) / len(te))
            sh = [opts[i] for i in tr]
            random.Random(f).shuffle(sh)
            p2 = train_act(X[tr], sh, X[te], hidden=hid)
            acc[name + '_shuffled'].append(sum((int(int(p2[j]) in opts[i]) for j, i in enumerate(te))) / len(te))
    import statistics as st

    def ci(v):
        m = st.mean(v)
        return [m, 1.96 * st.stdev(v) / len(v) ** 0.5 if len(v) > 1 else 0.0]
    return {'n': N, 'uncertainty_note': '1.96 * fold-score SD / sqrt(folds); descriptive cross-fold variation, not an independent-sample confidence guarantee', '1_free_generation': sum((int(free[i] in opts[i]) for i in range(N))) / N, '1_free_unparsed': sum((1 for f_ in free if f_ < 0)) / N, '1_free_truncated': sum((1 for t_ in trunc if t_)) / N, '2_forced_choice': sum((int(forced[i] in opts[i]) for i in range(N))) / N, **({'3_probe_then_rule': ci(acc['probe_then_rule'])} if args.position == 'hidden' else {'3_probe_then_rule': 'NOT MEASURED under --position given: this rung decodes the agent row/column, which that condition prints in the prompt, so it would read its own input rather than perceive'}), '4_linear_action': ci(acc['linear_action']), '4_linear_action_shuffled': ci(acc['linear_action_shuffled']), '5_mlp_action': ci(acc['mlp_action']), '5_mlp_action_shuffled': ci(acc['mlp_action_shuffled']), 'baselines': baselines(lays), 'per_instance': {'1_free_generation': [int(x) for x in free], '2_forced_choice': [int(x) for x in forced], **{{'probe_then_rule': '3_probe_then_rule', 'linear_action': '4_linear_action', 'mlp_action': '5_mlp_action'}[k]: [int(pred_inst[k].get(i, -1)) for i in range(N)] for k in pred_inst}}, 'by_level': _by_level(lays, opts, free, forced, per_inst)}

def _by_level(lays, opts, free, forced, per_inst):
    """Every rung split by VSP difficulty level, with that level's own chance rate."""
    name = {'probe_then_rule': '3_probe_then_rule', 'linear_action': '4_linear_action', 'mlp_action': '5_mlp_action'}
    out = {}
    for lv in sorted({l.level for l in lays}):
        idx = [i for i, l in enumerate(lays) if l.level == lv]
        n = len(idx)
        d = {'n': n, '1_free_generation': sum((int(free[i] in opts[i]) for i in idx)) / n, '1_free_unparsed': sum((1 for i in idx if free[i] < 0)) / n, '2_forced_choice': sum((int(forced[i] in opts[i]) for i in idx)) / n, 'chance_uniform_4': sum((len(opts[i]) for i in idx)) / 4 / n}
        for k, m in per_inst.items():
            have = [m[i] for i in idx if i in m]
            if have:
                d[name[k]] = sum(have) / len(have)
        out[str(lv)] = d
    return out
if __name__ == '__main__':
    main()
