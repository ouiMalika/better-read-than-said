"""Per-cell hazard perception feeding a planner, with hard walls or a calibrated traversal cost."""
import os
import json
import math
import heapq
import argparse
from env import Layout, step, compute_distance_map, DELTAS
LEVEL = 5
ACTIONS = list(DELTAS.keys())

def dijkstra_actions(level, p_hole, start, goal, lam, blocked=frozenset()):
    """Least-cost path where entering cell x costs 1 + lam * p_hole[x]."""
    dist = {start: 0.0}
    prev = {}
    pq = [(0.0, start)]
    seen = set()
    while pq:
        d, cur = heapq.heappop(pq)
        if cur in seen:
            continue
        seen.add(cur)
        if cur == goal:
            break
        for a in ACTIONS:
            dr, dc = DELTAS[a]
            nb = (cur[0] + dr, cur[1] + dc)
            if not (0 <= nb[0] < level and 0 <= nb[1] < level) or nb in blocked:
                continue
            nd = d + 1.0 + lam * p_hole.get(nb, 0.0)
            if nd < dist.get(nb, float('inf')):
                dist[nb] = nd
                prev[nb] = (cur, a)
                heapq.heappush(pq, (nd, nb))
    if goal not in prev and goal != start:
        return None
    path, cur = ([], goal)
    while cur != start:
        cur, a = prev[cur]
        path.append(a)
    return path[::-1]

def executes_in_true(true_lay, path):
    """Correct FrozenLake semantics: off-grid = no-op & continue, hole = terminal loss."""
    if path is None:
        return False
    pos = true_lay.start
    for a in path:
        nxt, ev = step(true_lay, pos, a)
        if ev == 'hole':
            return False
        if ev == 'invalid':
            continue
        pos = nxt
        if ev == 'goal':
            return True
    return pos == true_lay.goal

def pr_at(scores, truth, tau):
    """scores/truth: dict cell -> float / bool."""
    tp = sum((1 for c in scores if scores[c] >= tau and truth.get(c)))
    fp = sum((1 for c in scores if scores[c] >= tau and (not truth.get(c))))
    fn = sum((1 for c in scores if scores[c] < tau and truth.get(c)))
    p = tp / (tp + fp) if tp + fp else float('nan')
    r = tp / (tp + fn) if tp + fn else float('nan')
    f = 2 * p * r / (p + r) if p == p and r == r and p + r else 0.0
    return (p, r, f)

def auprc(pairs):
    """pairs: list of (score, is_hole)."""
    pairs = sorted(pairs, key=lambda t: -t[0])
    n_pos = sum((1 for _, y in pairs if y))
    if not n_pos:
        return float('nan')
    tp = 0
    ap = 0.0
    i = 0
    while i < len(pairs):
        j = i + 1
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        positives = sum((bool(y) for _, y in pairs[i:j]))
        tp += positives
        ap += positives * tp / j
        i = j
    return ap / n_pos

def platt(scores, labels, iters=400, lr=0.05):
    """1-D logistic calibration p = sigmoid(a*s + b), fit by plain gradient descent (no torch)."""
    a, b = (1.0, 0.0)
    n = len(scores)
    for _ in range(iters):
        ga = gb = 0.0
        for s, y in zip(scores, labels):
            p = 1 / (1 + math.exp(-max(-30, min(30, a * s + b))))
            ga += (p - y) * s
            gb += p - y
        a -= lr * ga / n
        b -= lr * gb / n
    return (a, b)

def cell_score(model, processor, tok, img, r, c):
    """v1 returned `yes > no`, throwing away the margin."""
    import torch
    q = f'This is a {LEVEL}x{LEVEL} grid navigation map. Is there an ice hole (a cracked-ice tile) at row {r + 1}, column {c + 1}? Answer with exactly one word: Yes or No.'
    chat = processor.apply_chat_template([{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': q}]}], tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[chat], images=[img], return_tensors='pt').to(model.device)
    with torch.no_grad():
        lg = model(**inputs).logits[0, -1, :].float().cpu()
    yes = max((lg[tok.encode(w, add_special_tokens=False)[0]] for w in ['Yes', ' Yes', 'yes']))
    no = max((lg[tok.encode(w, add_special_tokens=False)[0]] for w in ['No', ' No', 'no']))
    return float(yes - no)

def save_scores(path, per_maze):
    """Retain expensive cell logits so all planner analyses can be replayed on CPU."""
    from pathlib import Path
    import tempfile
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [{'layout': {'level': m['lay'].level, 'holes': sorted(m['lay'].holes), 'goal': m['lay'].goal, 'start': m['lay'].start}, 'image_path': m.get('image_path'), 'cells': [[r, c, s] for (r, c), s in sorted(m['scores'].items())]} for m in per_maze]
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + '.')
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump({'schema': 1, 'rows': rows}, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)

def load_scores(path):
    with open(path) as f:
        rows = json.load(f)['rows']
    out = []
    for r in rows:
        l = r['layout']
        lay = Layout(l['level'], frozenset(map(tuple, l['holes'])), tuple(l['goal']), tuple(l['start']))
        if lay.level != LEVEL:
            raise ValueError('this experiment requires level 5')
        scores = {(x, y): float(s) for x, y, s in r['cells']}
        if set(scores) != {(x, y) for x in range(LEVEL) for y in range(LEVEL)}:
            raise ValueError('incomplete cell scores')
        out.append({'lay': lay, 'scores': scores, 'image_path': r.get('image_path'), 'truth': {c: c in lay.holes for c in scores}})
    return out

def evaluate(per_maze):
    if len(per_maze) < 2:
        raise ValueError('need both calibration and evaluation mazes')
    flat = [(m['scores'][c], m['truth'][c]) for m in per_maze for c in m['scores']]
    base_rate = sum((1 for _, y in flat if y)) / len(flat)
    res = {'n': len(per_maze), 'n_calibration': len(per_maze) // 2, 'n_evaluation': len(per_maze) - len(per_maze) // 2, 'selection_protocol': 'exploratory sweep on evaluation mazes; no independent parameter-selection test', 'auprc': auprc(flat), 'base_rate': base_rate, 'auprc_note': "AUPRC vs base_rate is the threshold-free 'is it there' measure"}
    half = len(per_maze) // 2
    tr = [(m['scores'][c], 1.0 if m['truth'][c] else 0.0) for m in per_maze[:half] for c in m['scores']]
    a, b = platt([s for s, _ in tr], [y for _, y in tr])
    res['platt'] = {'a': a, 'b': b}
    te = per_maze[half:]
    for m in te:
        m['p'] = {c: 1 / (1 + math.exp(-max(-30, min(30, a * s + b)))) for c, s in m['scores'].items()}

    def run(fn):
        return sum((fn(m) for m in te)) / len(te)
    res['controls'] = {'oracle_map': run(lambda m: executes_in_true(m['lay'], dijkstra_actions(LEVEL, {}, m['lay'].start, m['lay'].goal, 0.0, blocked=frozenset(m['lay'].holes)))), 'empty_map': run(lambda m: executes_in_true(m['lay'], dijkstra_actions(LEVEL, {}, m['lay'].start, m['lay'].goal, 0.0)))}
    res['hard_threshold_sweep'] = {}
    for tau in [-4, -2, -1, 0, 1, 2, 4]:
        P = R = F = 0.0
        nP = nR = 0
        succ = 0
        for m in te:
            p, r, f = pr_at(m['scores'], m['truth'], tau)
            if p == p:
                P += p
                nP += 1
            if r == r:
                R += r
                nR += 1
            F += f
            blocked = frozenset((c for c in m['scores'] if m['scores'][c] >= tau)) - {m['lay'].goal}
            succ += executes_in_true(m['lay'], dijkstra_actions(LEVEL, {}, m['lay'].start, m['lay'].goal, 0.0, blocked=blocked))
        n = len(te)
        res['hard_threshold_sweep'][tau] = {'precision': P / nP if nP else None, 'n_mazes_precision_defined': nP, 'recall': R / nR if nR else None, 'n_mazes_recall_defined': nR, 'f1': F / n, 'success': succ / n}
    res['risk_sensitive_sweep'] = {}
    for lam in [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]:
        res['risk_sensitive_sweep'][lam] = run(lambda m: executes_in_true(m['lay'], dijkstra_actions(LEVEL, m['p'], m['lay'].start, m['lay'].goal, lam)))
    positive = {k: v for k, v in res['risk_sensitive_sweep'].items() if float(k) > 0}
    if positive:
        vals = sorted(positive.values())
        res['risk_sensitive_positive_lambda'] = {'min': vals[0], 'max': vals[-1], 'median': vals[len(vals) // 2] if len(vals) % 2 else (vals[len(vals) // 2 - 1] + vals[len(vals) // 2]) / 2, 'n_weights': len(vals), 'note': 'the maximum is selected on the same mazes it is scored on; the median and the minimum over positive weights do not depend on that selection'}
    best_lam = max(res['risk_sensitive_sweep'], key=lambda k: res['risk_sensitive_sweep'][k])
    best_tau = max(res['hard_threshold_sweep'], key=lambda k: res['hard_threshold_sweep'][k]['success'])
    emp = res['controls']['empty_map']
    res['verdict'] = {'best_lambda': best_lam, 'best_lambda_success': res['risk_sensitive_sweep'][best_lam], 'best_tau': best_tau, 'best_tau_success': res['hard_threshold_sweep'][best_tau]['success'], 'empty_map_success': emp, 'reading': 'Exploratory best-of-sweep result; confirm selected parameters on independent mazes.'}
    return res

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gen_dir', default='vsp_train6k')
    ap.add_argument('--n', type=int, default=200)
    ap.add_argument('--out', default='results_perception_solver_v2.json')
    ap.add_argument('--scores-json', help='replay saved per-maze logits without a GPU')
    ap.add_argument('--save-scores', help='checkpoint logits here; defaults to OUT.scores.json')
    args = ap.parse_args()
    if args.scores_json:
        res = evaluate(load_scores(args.scores_json))
        with open(args.out, 'w') as f:
            json.dump(res, f, indent=2)
        print(json.dumps(res, indent=2))
        return
    import torch
    from PIL import Image
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    from probe_spatial import load_instances, MODEL_ID
    print(f'loading {MODEL_ID} ...', flush=True)
    model = Qwen2VLForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map='auto')
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model.eval()
    tok = processor.tokenizer
    insts = load_instances(args.gen_dir, args.n)
    print(f'{len(insts)} mazes', flush=True)
    per_maze = []
    for k, (img_path, lay) in enumerate(insts):
        img = Image.open(img_path).convert('RGB')
        cells = [(r, c) for r in range(LEVEL) for c in range(LEVEL)]
        scores = {rc: cell_score(model, processor, tok, img, *rc) for rc in cells}
        per_maze.append({'lay': lay, 'scores': scores, 'image_path': str(img_path), 'truth': {rc: rc in lay.holes for rc in cells}})
        save_scores(args.save_scores or args.out + '.scores.json', per_maze)
        if (k + 1) % 20 == 0:
            print(f'  {k + 1}/{len(insts)}', flush=True)
    res = evaluate(per_maze)
    with open(args.out, 'w') as f:
        json.dump(res, f, indent=2, default=str)
    print(json.dumps(res, indent=2, default=str), flush=True)
    print(f'\nsaved -> {args.out}', flush=True)
if __name__ == '__main__':
    main()
