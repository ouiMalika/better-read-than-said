"""Convert position sequences to actions."""
import torch
from env import DELTAS, step, compute_distance_map
from data import optimal_positions, random_positions
_NAMES = ['BOS', 'GOAL', 'START', 'HOLE', 'PLAN', 'up', 'down', 'left', 'right', 'EOS', 'PAD']

def build_vocab(level):
    vocab = {str(i): i for i in range(level)}
    for k, name in enumerate(_NAMES):
        vocab[name] = level + k
    return (vocab, level + len(_NAMES))

def positions_to_actions(positions):
    acts = []
    for a, b in zip(positions[:-1], positions[1:]):
        d = (b[0] - a[0], b[1] - a[1])
        for name, delta in DELTAS.items():
            if delta == d:
                acts.append(name)
                break
    return acts

def encode_instance(layout, vocab):
    """Symbolic prompt: goal coord, start coord, hole coords, then PLAN marker."""
    t = [vocab['BOS'], vocab['GOAL'], vocab[str(layout.goal[0])], vocab[str(layout.goal[1])], vocab['START'], vocab[str(layout.start[0])], vocab[str(layout.start[1])]]
    for hr, hc in sorted(layout.holes):
        t += [vocab['HOLE'], vocab[str(hr)], vocab[str(hc)]]
    t.append(vocab['PLAN'])
    return t

def encode_plan(actions, vocab):
    return [vocab[a] for a in actions] + [vocab['EOS']]

def build_streams(instances, mode, vocab, rng=None, max_len=8):
    streams = []
    for lay in instances:
        pos = optimal_positions(lay) if mode == 'optimal' else random_positions(lay, max_len, rng)
        if len(pos) < 2:
            continue
        streams.append(encode_instance(lay, vocab) + encode_plan(positions_to_actions(pos), vocab))
    return streams

def make_planner(model, vocab, level, device, block_size, max_actions=None):
    """Wrap the LM as a planner: prompt -> greedy action sequence."""
    inv = {v: k for k, v in vocab.items()}
    action_ids = {vocab[a] for a in ('up', 'down', 'left', 'right')}
    eos = vocab['EOS']
    if max_actions is None:
        max_actions = 4 * level * level

    def plan(layout):
        idx = torch.tensor([encode_instance(layout, vocab)], dtype=torch.long, device=device)
        acts = []
        model.eval()
        for _ in range(max_actions):
            if idx.shape[1] > block_size:
                idx = idx[:, -block_size:]
            with torch.no_grad():
                nt = int(model(idx)[:, -1, :].argmax(dim=-1))
            if nt == eos:
                break
            idx = torch.cat([idx, torch.tensor([[nt]], device=device)], dim=1)
            if nt in action_ids:
                acts.append(inv[nt])
            else:
                break
        return acts
    return plan

def eval_planner(planner, instances):
    """Per-instance success / exact-match by executing the generated action sequence."""
    per = []
    for lay in instances:
        acts = planner(lay)
        pos, ok, nsteps = (lay.start, False, 0)
        for a in acts:
            nxt, ev = step(lay, pos, a)
            nsteps += 1
            if ev in ('invalid', 'hole'):
                break
            pos = nxt
            if ev == 'goal':
                ok = True
                break
        opt = compute_distance_map(lay).get(lay.start)
        per.append({'success': ok, 'em': bool(ok and opt is not None and (nsteps == opt))})
    return per
