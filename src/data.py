"""Optimal and random trajectories on a layout."""
import torch
from env import ACTIONS, step, compute_distance_map, encode_frame
PAD = 4
VOCAB = 5

def optimal_positions(layout):
    dmap = compute_distance_map(layout)
    pos = layout.start
    out = [pos]
    guard = 4 * layout.level * layout.level
    while pos != layout.goal and len(out) <= guard:
        best = None
        for a in ACTIONS:
            nxt, ev = step(layout, pos, a)
            if ev == 'invalid':
                continue
            d = dmap.get(nxt)
            if d is None:
                continue
            if best is None or d < best[0]:
                best = (d, nxt)
        if best is None:
            break
        pos = best[1]
        out.append(pos)
    return out

def random_positions(layout, max_len, rng):
    pos = layout.start
    out = [pos]
    for _ in range(max_len):
        cands = []
        for a in ACTIONS:
            nxt, ev = step(layout, pos, a)
            if ev in ('move', 'goal'):
                cands.append((nxt, ev))
        if not cands:
            break
        nxt, ev = rng.choice(cands)
        out.append(nxt)
        pos = nxt
        if ev == 'goal':
            break
    return out

def stream_from_positions(layout, positions):
    s = []
    for p in positions:
        s.extend(encode_frame(layout, p))
    return s

def build_streams(instances, mode, rng=None, max_len=8):
    streams = []
    for lay in instances:
        pos = optimal_positions(lay) if mode == 'optimal' else random_positions(lay, max_len, rng)
        if len(pos) < 2:
            continue
        streams.append(stream_from_positions(lay, pos))
    return streams

def make_model_policy(model, level, device, block_size, temperature=0.0):
    """Wrap the LM as a policy: flatten prefix frames -> generate the next level*level tokens."""
    L = level * level

    def pol(frames):
        ctx = [t for fr in frames for t in fr]
        idx = torch.tensor([ctx[-block_size:]], dtype=torch.long, device=device)
        gen = []
        model.eval()
        for _ in range(L):
            if idx.shape[1] > block_size:
                idx = idx[:, -block_size:]
            with torch.no_grad():
                logits = model(idx)[:, -1, :].clone()
            logits[:, PAD] = float('-inf')
            if temperature and temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                nt = torch.multinomial(probs, 1)
            else:
                nt = logits.argmax(dim=-1, keepdim=True)
            gen.append(int(nt))
            idx = torch.cat([idx, nt], dim=1)
        return tuple(gen)
    return pol
