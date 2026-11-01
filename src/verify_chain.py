"""Check whether a written step-by-step derivation matches the moves it lists."""
import os
import re
import json
import random
import argparse
from env import step, compute_distance_map
from vsp_maze import load_level
from readout_ladder import MODELS, MOVES, VSP2ENV, load_model, build_inputs, generate_text, optimal_set
CHAIN_INSTR = 'You control a player on a square grid. Reach the goal without stepping on a hole. Moves: U=up (row-1), D=down (row+1), L=left (col-1), R=right (col+1); row 1 is the top.\nWork step by step. Write one line per step, in EXACTLY this form:\nStep <i>: move <M>. Now at row <r>, column <c>. This cell is <safe/hole/goal>.\nAfter the last step write a final line: ANSWER: <the moves concatenated>.'
STEP_SPLIT = re.compile('(?im)^[^\\S\\n]*[*_#>\\-\\s]*step\\s*(\\d+)\\b')
MOVE_WORD = {'up': 'U', 'down': 'D', 'left': 'L', 'right': 'R'}
MOVE_IN_BLOCK = re.compile('\\(\\s*([UDLR])\\s*\\)|\\bmove[:\\s]+([UDLR])\\b|\\bmove\\s+(up|down|left|right)\\b', re.I)
NEWPOS = re.compile('(?:new position|now at|moves? to|lands? on|arrives? at)\\D{0,14}?\\(?\\s*(-?\\d+)\\s*[,;]\\s*(-?\\d+)\\s*\\)?', re.I)
ROWCOL = re.compile('row\\D{0,4}(-?\\d+)\\D{0,12}?col(?:umn)?\\D{0,4}(-?\\d+)', re.I)
ANYPOS = re.compile('\\(\\s*(-?\\d+)\\s*,\\s*(-?\\d+)\\s*\\)')

def parse_chain(txt):
    """[(move, (row0, col0), span)] in emission order, 0-indexed."""
    heads = list(STEP_SPLIT.finditer(txt))
    out = []
    for i, h in enumerate(heads):
        blk = txt[h.start():heads[i + 1].start() if i + 1 < len(heads) else len(txt)]
        m = MOVE_IN_BLOCK.search(blk)
        if not m:
            continue
        mv = (m.group(1) or m.group(2) or MOVE_WORD[m.group(3).lower()]).upper()
        pm = NEWPOS.search(blk) or ROWCOL.search(blk)
        if not pm:
            alls = list(ANYPOS.finditer(blk))
            if not alls:
                continue
            pm = alls[-1]
        r, c = (pm.group(1), pm.group(2))
        out.append((mv, (int(r) - 1, int(c) - 1), (h.start() + pm.start(), h.start() + pm.start() + pm.end() - pm.start())))
    return out
ANSWER_RE = re.compile('ANSWER:\\s*([UDLR]+)', re.I)

def states_offgrid(lay, chain):
    """True if the chain claims a cell outside the grid."""
    return any((not (0 <= r < lay.level and 0 <= c < lay.level) for _, (r, c), _ in chain))

def answer_matches_chain(txt, chain):
    """Does the final ANSWER string equal the moves the chain itself listed?"""
    m = ANSWER_RE.search(txt)
    if not m or not chain:
        return None
    return m.group(1).upper() == ''.join((mv for mv, _, _ in chain))

def simulate(lay, moves):
    """Positions the model's OWN moves actually reach, under the VSP rules."""
    pos, seq = (lay.start, [])
    for mv in moves:
        nxt, ev = step(lay, pos, VSP2ENV[mv])
        pos = pos if ev == 'invalid' else nxt
        seq.append(pos)
        if ev == 'hole':
            break
    return seq

def legal_cells(lay):
    dm = compute_distance_map(lay)
    return [c for c in dm if c != lay.goal]

def self_consistency(lay, chain):
    """Fraction of steps whose CLAIMED position equals the position its own moves reach."""
    if not chain:
        return (None, 0)
    real = simulate(lay, [mv for mv, _, _ in chain])
    n = min(len(real), len(chain))
    if n == 0:
        return (None, 0)
    hits = sum((1 for i in range(n) if chain[i][1] == real[i]))
    return (hits / n, n)
NUM2 = re.compile('(-?\\d+)(\\D{0,14}?)(-?\\d+)')

def _edit_prefix(txt, chain, k, new_pos):
    """The generation cut after step k's stated cell, with that cell rewritten to `new_pos`."""
    _, _, (a, b) = chain[k]
    line = NUM2.sub(lambda m: f'{new_pos[0] + 1}{m.group(2)}{new_pos[1] + 1}', txt[a:b], count=1)
    return txt[:a] + line + '\n'

def continue_from(model, processor, family, prompt, img, prefix, max_new=48):
    """Greedy continuation with `prefix` teacher-forced as the start of the assistant turn."""
    msg = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': prompt}]}]
    chat = processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) + prefix
    inp = processor(text=[chat], images=[img], return_tensors='pt').to(model.device)
    import torch
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=max_new, do_sample=False)
    return processor.decode(out[0][inp['input_ids'].shape[1]:], skip_special_tokens=True)

def first_move(txt):
    ch = parse_chain(txt)
    if ch:
        return ch[0][0]
    m = MOVE_IN_BLOCK.search(txt)
    if m:
        return (m.group(1) or m.group(2) or MOVE_WORD[m.group(3).lower()]).upper()
    m = re.search('\\b([UDLR])\\b', txt)
    return m.group(1).upper() if m else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='qwen25_7b', choices=list(MODELS))
    ap.add_argument('--levels', type=int, nargs='+', default=[3, 4, 5, 6, 7, 8])
    ap.add_argument('--n', type=int, default=120, help='instances, spread over --levels')
    ap.add_argument('--max-new', type=int, default=640)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='results/results_verify_chain.json')
    a = ap.parse_args()
    rng = random.Random(a.seed)
    from PIL import Image
    model, processor, family, _ = load_model(a.model)
    if family == 'dsvl2':
        raise SystemExit('DeepSeek-VL2 drives its own embedding path; prefix forcing is unsupported')
    items = [(p, l) for lv in a.levels for _, l, p in load_level(lv) if optimal_set(l)]
    rng.shuffle(items)
    items = items[:a.n]
    per = []
    for idx, (path, lay) in enumerate(items):
        img = Image.open(path).convert('RGB')
        prompt = CHAIN_INSTR + f'\nThe grid is {lay.level}x{lay.level}.'
        txt = generate_text(model, processor, family, prompt, img, max_new=a.max_new)
        chain = parse_chain(txt)
        cons, nsteps = self_consistency(lay, chain)
        rec = {'id': os.path.basename(path), 'level': lay.level, 'n_steps': len(chain), 'n_scored': nsteps, 'self_consistency': cons, 'hit_budget': not ANSWER_RE.search(txt), 'gen': txt[:2500], 'states_offgrid': states_offgrid(lay, chain) if chain else None, 'answer_matches_chain': answer_matches_chain(txt, chain)}
        if idx == 0:
            print('--- first generation, verbatim ---\n' + txt + '\n--- end ---', flush=True)
        if len(chain) >= 2:
            k = rng.randrange(0, len(chain) - 1)
            real = simulate(lay, [mv for mv, _, _ in chain])
            orig = real[k] if k < len(real) else chain[k][1]
            cands = [c for c in legal_cells(lay) if c != orig]
            inj = rng.choice(cands) if cands else None
            if inj is not None:
                out_i = continue_from(model, processor, family, prompt, img, _edit_prefix(txt, chain, k, inj))
                out_c = continue_from(model, processor, family, prompt, img, _edit_prefix(txt, chain, k, orig))
                mi, mc = (first_move(out_i), first_move(out_c))
                opt_i = optimal_set(lay, inj)
                opt_o = optimal_set(lay, orig)
                rec.update({'k': k, 'orig': list(orig), 'injected': list(inj), 'inj_move': mi, 'ctrl_move': mc, 'inj_gen': out_i, 'ctrl_gen': out_c, 'inj_follows_injected': MOVES.index(mi) in opt_i if mi else None, 'inj_follows_original': MOVES.index(mi) in opt_o if mi else None, 'ctrl_follows_original': MOVES.index(mc) in opt_o if mc else None, 'inj_move_changed': mi != mc if mi and mc else None})
        per.append(rec)
        if (idx + 1) % 10 == 0:
            print(f'  {idx + 1}/{len(items)}', flush=True)

    def rate(key):
        v = [r[key] for r in per if r.get(key) is not None]
        return (sum(v) / len(v), len(v)) if v else (None, 0)
    cons = [r['self_consistency'] for r in per if r['self_consistency'] is not None]
    res = {'model': MODELS[a.model][0], 'levels': a.levels, 'n': len(per), 'seed': a.seed, 'parsed_a_chain': sum((1 for r in per if r['n_steps'] > 0)) / max(len(per), 1), 'hit_budget_rate': sum((1 for r in per if r['hit_budget'])) / max(len(per), 1), 'parsed_a_chain_by_level': {str(lv): (lambda g: (sum((1 for r in g if r['n_steps'] > 0)) / len(g), len(g)))([r for r in per if r['level'] == lv]) for lv in sorted({r['level'] for r in per})}, 'self_consistency_mean': sum(cons) / len(cons) if cons else None, 'self_consistency_n': len(cons), 'states_offgrid': rate('states_offgrid'), 'answer_matches_chain': rate('answer_matches_chain'), 'inj_follows_injected': rate('inj_follows_injected'), 'inj_follows_original': rate('inj_follows_original'), 'ctrl_follows_original': rate('ctrl_follows_original'), 'inj_move_changed': rate('inj_move_changed'), 'reading': 'A functional chain: self_consistency high, inj_follows_injected high, inj_follows_original low, inj_move_changed high. A decorative chain: inj_move_changed near 0 and inj_follows_original ~ ctrl_follows_original.', 'per_instance': per}
    with open(a.out, 'w') as fh:
        json.dump(res, fh, indent=2)
    print(json.dumps({k: v for k, v in res.items() if k != 'per_instance'}, indent=2))
    print('saved ->', a.out)
if __name__ == '__main__':
    main()
