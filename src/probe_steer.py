"""Add or remove a probe direction in the hidden state and read the emitted move."""
import argparse
import json
import os
import sys
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from readout_ladder import MOVES, MODELS, load_model, build_inputs, optimal_set, _prompts, step, VSP2ENV
from vsp_maze import load_level

def layer_list(model):
    """The decoder layers, whichever attribute path this transformers version uses."""
    for path in (('model', 'language_model', 'layers'), ('model', 'layers'), ('language_model', 'model', 'layers'), ('model', 'model', 'layers')):
        o = model
        for a in path:
            o = getattr(o, a, None)
            if o is None:
                break
        if o is not None and hasattr(o, '__len__'):
            return o
    raise RuntimeError('cannot locate decoder layers')

def move_logits(out, tok_ids):
    lg = out.logits[0, -1, :].float().cpu()
    return torch.tensor([max((lg[i].item() for i in ids)) for ids in tok_ids])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='qwen25_7b')
    ap.add_argument('--levels', type=int, nargs='+', default=[3, 4, 5, 6, 7, 8])
    ap.add_argument('--stride', type=int, default=2, help='take every k-th instance for the edits')
    ap.add_argument('--alphas', type=float, nargs='+', default=[1.0, 2.0])
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--steps', type=int, default=1500)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--mode', choices=['steer', 'ablate'], default='steer', help='steer: add a probe direction. ablate: remove the probe subspace, which tests whether the model needs it')
    ap.add_argument('--cache', default=None, help='pass-1 hidden states; reused when present so a wall-clock kill does not cost them again')
    ap.add_argument('--dataset', choices=['vsp'], default='vsp')
    ap.add_argument('--ladder-ckpt', default=None, help='readout_ladder checkpoint whose cached states and forced-choice moves are reused')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    model, proc, family, layer = load_model(a.model)
    assert family == 'hf', 'this experiment is written for the hf path'
    layers = layer_list(model)
    print(f'{MODELS[a.model][0]}: {len(layers)} decoder layers, probe layer {layer}', flush=True)
    tok = proc.tokenizer
    tok_ids = [[i for i in tok.convert_tokens_to_ids([m, f'Ġ{m}']) if i is not None] for m in MOVES]
    unk = getattr(tok, 'unk_token_id', None)
    tok_ids = [[i for i in ids if i != unk] for ids in tok_ids]
    print('move token ids:', {m: [(i, tok.convert_ids_to_tokens(i)) for i in ids] for m, ids in zip(MOVES, tok_ids)}, flush=True)
    assert all(tok_ids), 'a move letter has no token id in this tokenizer'
    items = [(p, lay) for lv in a.levels for _, lay, p in load_level(lv) if optimal_set(lay)]
    print(f'{len(items)} instances', flush=True)
    from PIL import Image
    opts = [optimal_set(lay) for _p, lay in items]
    H, base_move = ([], [])
    if a.cache and os.path.exists(a.cache):
        blob = torch.load(a.cache, weights_only=False)
        H, base_move = (list(blob['H']), list(blob['base_move']))
        print(f'pass1 resumed at {len(H)}/{len(items)} from {a.cache}', flush=True)
    if len(H) < len(items):
        for k, (p, lay) in enumerate(items):
            if k < len(H):
                continue
            action, _plan = _prompts(lay, 'hidden')
            img = Image.open(p).convert('RGB')
            with torch.no_grad():
                inp = build_inputs(proc, action, img).to(model.device)
                out = model(**inp, output_hidden_states=True)
            H.append(out.hidden_states[layer][0, -1, :].float().cpu().clone())
            base_move.append(int(move_logits(out, tok_ids).argmax()))
            if (k + 1) % 50 == 0:
                print(f'  pass1 {k + 1}/{len(items)}', flush=True)
                if a.cache:
                    torch.save({'H': H, 'base_move': base_move}, a.cache + '.tmp')
                    os.replace(a.cache + '.tmp', a.cache)
    if a.cache:
        torch.save({'H': H, 'base_move': base_move}, a.cache + '.tmp')
        os.replace(a.cache + '.tmp', a.cache)
    X = torch.stack(H)
    N = len(items)
    grp = lambda i: i
    folds = [[i for i in range(N) if grp(i) % a.folds == f] for f in range(a.folds)]

    def fit(tr):
        mu, sd = (X[tr].mean(0, keepdim=True), X[tr].std(0, keepdim=True).clamp_min(1e-06))
        Z = (X[tr] - mu) / sd
        W = torch.zeros(X.shape[1], 4, requires_grad=True)
        b = torch.zeros(4, requires_grad=True)
        opt = torch.optim.Adam([W, b], lr=0.01, weight_decay=0.001)
        for _ in range(a.steps):
            lp = torch.log_softmax(Z @ W + b, 1)
            loss = torch.stack([-torch.logsumexp(lp[j, sorted(opts[i])], 0) for j, i in enumerate(tr)]).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        return (W.detach(), b.detach(), mu, sd)
    probe = {}
    for f in range(a.folds):
        tr = [i for i in range(N) if i not in set(folds[f])]
        probe[f] = fit(tr)
        print(f'  fold {f} fit', flush=True)
    delta = {'vec': None}

    def hook(_mod, _inp, output):
        if delta['vec'] is None:
            return output
        hs = output[0] if isinstance(output, tuple) else output
        hs[0, -1, :] += delta['vec'].to(hs.dtype).to(hs.device)
        return output
    handle = layers[layer - 1].register_forward_hook(hook)
    sel = list(range(0, N, a.stride))
    emitted = {}
    res = {'model': MODELS[a.model][0], 'layer': layer, 'n_all': N, 'n_edited': len(sel), 'alphas': a.alphas, 'stride': a.stride, 'baseline_correct': sum((int(base_move[i] in opts[i]) for i in sel)) / len(sel), 'conditions': {}}
    counts = {(d, al): {'target': 0, 'correct': 0, 'changed': 0, 'n': 0} for d in ('probe_opt', 'probe_dist', 'random') for al in a.alphas}
    selftest_done = False
    for c, i in enumerate(sel):
        p, lay = items[i]
        action, _plan = _prompts(lay, 'hidden')
        img = Image.open(p).convert('RGB')
        inp = build_inputs(proc, action, img).to(model.device)
        W, b, mu, sd = probe[grp(i) % a.folds]
        centre = W.mean(1, keepdim=True)
        h = X[i]
        legal = [m for m in range(4) if step(lay, lay.start, VSP2ENV[MOVES[m]])[1] != 'invalid']
        t_opt = max(sorted(opts[i]), key=lambda t: float((W[:, t] - centre[:, 0]) @ ((h - mu[0]) / sd[0])))
        wrong = [m for m in legal if m not in opts[i]] or [m for m in range(4) if m not in opts[i]]
        t_dist = min(wrong, key=lambda t: float((W[:, t] - centre[:, 0]) @ ((h - mu[0]) / sd[0])))

        def direction(t):
            u = (W[:, t] - centre[:, 0]) / sd[0]
            return u / u.norm().clamp_min(1e-09)
        dirs = {'probe_opt': (direction(t_opt), t_opt), 'probe_dist': (direction(t_dist), t_dist)}
        g = torch.randn(X.shape[1], generator=torch.Generator().manual_seed(a.seed + i))
        dirs['random'] = (g / g.norm(), t_opt)
        if not selftest_done:
            delta['vec'] = torch.zeros_like(h)
            with torch.no_grad():
                chk = move_logits(model(**inp, output_hidden_states=False), tok_ids)
            delta['vec'] = None
            with torch.no_grad():
                ref = move_logits(model(**inp, output_hidden_states=False), tok_ids)
            assert torch.equal(chk, ref), f'alpha=0 hook changed the logits: {chk} vs {ref}'
            print('  self-test passed: a zero edit leaves the logits unchanged', flush=True)
            selftest_done = True
        if a.mode == 'ablate':
            U = torch.stack([(W[:, t] - centre[:, 0]) / sd[0] for t in range(4)], 1)
            Q, _ = torch.linalg.qr(U)
            k = Q.shape[1]
            g2 = torch.randn(X.shape[1], k, generator=torch.Generator().manual_seed(a.seed + 10000 + i))
            Qr, _ = torch.linalg.qr(g2)
            for name, B in (('ablate_probe', Q), ('ablate_random', Qr)):
                delta['vec'] = -(B @ (B.T @ h))
                with torch.no_grad():
                    mv = int(move_logits(model(**inp, output_hidden_states=False), tok_ids).argmax())
                delta['vec'] = None
                emitted.setdefault((name, 0.0), {})[i] = mv
                d = counts.setdefault((name, 0.0), {'target': 0, 'correct': 0, 'changed': 0, 'n': 0})
                d['n'] += 1
                d['correct'] += int(mv in opts[i])
                d['changed'] += int(mv != base_move[i])
                hz = (h - B @ (B.T @ h) - mu[0]) / sd[0]
                d['target'] += int(int((hz @ W + b).argmax()) in opts[i])
            if (c + 1) % 50 == 0:
                print(f'  pass2 {c + 1}/{len(sel)}', flush=True)
            continue
        for name, (u, t) in dirs.items():
            for al in a.alphas:
                delta['vec'] = al * h.norm() * u
                with torch.no_grad():
                    mv = int(move_logits(model(**inp, output_hidden_states=False), tok_ids).argmax())
                delta['vec'] = None
                emitted.setdefault((name, al), {})[i] = mv
                d = counts[name, al]
                d['n'] += 1
                d['target'] += int(mv == t)
                d['correct'] += int(mv in opts[i])
                d['changed'] += int(mv != base_move[i])
        if (c + 1) % 50 == 0:
            print(f'  pass2 {c + 1}/{len(sel)}', flush=True)
    handle.remove()
    for (name, al), d in counts.items():
        if d['n'] == 0:
            continue
        key = 'probe_still_reads_optimal' if name.startswith('ablate') else 'emitted_equals_target'
        res['conditions'][f'{name}@{al}'] = {'n': d['n'], key: d['target'] / d['n'], 'emitted_is_optimal': d['correct'] / d['n'], 'emitted_changed_from_baseline': d['changed'] / d['n']}
    json.dump(res, open(a.out, 'w'), indent=2)
    print(json.dumps(res, indent=2), flush=True)
    print('saved ->', a.out, flush=True)
if __name__ == '__main__':
    main()
