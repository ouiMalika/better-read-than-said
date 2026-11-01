"""Method 1 (written plan) on hosted models, with the same prompt and parser as the open models."""
import os
import re
import sys
import json
import base64
import argparse
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vsp_maze import load_level
from readout_ladder import optimal_set, MOVES
INSTR = 'You control a player on a square grid. Reach the goal without stepping on a hole. Moves: U=up (row-1), D=down (row+1), L=left (col-1), R=right (col+1); row 1 is the top. Answer with EXACTLY ONE letter, one of U/D/L/R, and nothing else.'

def _check_instr():
    """Fail loudly if the copy of INSTR above drifts from the one the open models used."""
    src = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vlm_remedy.py')
    txt = open(src).read()
    m = re.search('INSTR = \\((.*?)\\)\\n\\n', txt, re.S)
    literal = ''.join(re.findall('"([^"]*)"', m.group(1)))
    if literal != INSTR:
        sys.exit('INSTR has drifted from vlm_remedy.py; the comparison would not be matched.')
SERIAL_GEOM = "Work through it in this order, writing each step on its own line.\n1. Row and column of the player.\n2. Row and column of the goal.\n3. The difference in rows and the difference in columns.\n4. The move that reduces the larger of those two differences.\nThen give the full plan as a string of U/D/L/R after 'ANSWER:'."

def prompts(lay, position, mode='plan'):
    """Same construction as readout_ladder._prompts, which cannot be imported here because it pulls in vlm_remedy -> torch."""
    ctx = f'The grid is {lay.level}x{lay.level}.'
    if position == 'given':
        ctx += f' You are at row {lay.start[0] + 1}, column {lay.start[1] + 1}.'
    if mode == 'geom':
        base = INSTR.replace('Answer with EXACTLY ONE letter, one of U/D/L/R, and nothing else.', '').strip()
        return base + '\n' + ctx + '\n' + SERIAL_GEOM
    return INSTR.replace('Answer with EXACTLY ONE letter, one of U/D/L/R, and nothing else.', "Give the full plan as a string of U/D/L/R after 'ANSWER:'.") + '\n' + ctx

def parse_move(txt):
    """Byte-identical to the rung-1 parser in readout_ladder.main()."""
    m = re.search('ANSWER:\\s*([UDLR]+)', txt, re.I) or re.search('\\b([UDLR]{2,})\\b', txt)
    return MOVES.index(m.group(1)[0].upper()) if m else -1

def b64(path):
    return base64.standard_b64encode(open(path, 'rb').read()).decode('utf-8')

def call_anthropic(model, prompt, img_path, max_tokens, effort, thinking=None):
    """thinking: None keeps the model default; "disabled" turns extended thinking off, which is the closest available analogue to the open models' single greedy forward pass."""
    import anthropic
    c = anthropic.Anthropic()
    kw = {}
    if effort:
        kw['output_config'] = {'effort': effort}
    if thinking == 'disabled':
        kw['thinking'] = {'type': 'disabled'}
    r = c.messages.create(model=model, max_tokens=max_tokens, **kw, messages=[{'role': 'user', 'content': [{'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/png', 'data': b64(img_path)}}, {'type': 'text', 'text': prompt}]}])
    txt = ''.join((b.text for b in r.content if b.type == 'text'))
    return (txt, r.usage.input_tokens, r.usage.output_tokens)

def call_openai(model, prompt, img_path, max_tokens, effort, thinking=None):
    from openai import OpenAI
    c = OpenAI()
    kw = {}
    if effort:
        kw['reasoning'] = {'effort': effort}
    r = c.responses.create(model=model, max_output_tokens=max_tokens, **kw, input=[{'role': 'user', 'content': [{'type': 'input_image', 'image_url': f'data:image/png;base64,{b64(img_path)}'}, {'type': 'input_text', 'text': prompt}]}])
    u = r.usage
    return (r.output_text, getattr(u, 'input_tokens', 0), getattr(u, 'output_tokens', 0))
PROVIDERS = {'anthropic': call_anthropic, 'openai': call_openai}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--provider', choices=list(PROVIDERS), required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--levels', type=int, nargs='+', default=[3, 4, 5, 6, 7, 8])
    ap.add_argument('--n-per-level', type=int, default=20, help='instances per level; 100 uses the full VSP set')
    ap.add_argument('--position', choices=['hidden', 'given'], default='hidden')
    ap.add_argument('--max-tokens', type=int, default=2000)
    ap.add_argument('--effort', default='low', help="reasoning effort; low keeps the regime closer to the open models' single greedy pass. Pass an empty string to use the provider default.")
    ap.add_argument('--prompt-mode', choices=['plan', 'geom'], default='plan')
    ap.add_argument('--thinking', choices=['default', 'disabled'], default='default', help="disabled removes extended thinking, the closest analogue to the open models' single greedy pass")
    ap.add_argument('--workers', type=int, default=4, help='keep low; new orgs are rate limited')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    _check_instr()
    fn = PROVIDERS[a.provider]
    items = []
    for lv in a.levels:
        got = [(p, l) for _, l, p in load_level(lv) if optimal_set(l)][:a.n_per_level]
        items += got
    print(f"{len(items)} instances, {a.provider}/{a.model}, effort={a.effort or 'default'}, thinking={a.thinking}", flush=True)

    def one(it):
        path, lay = it
        try:
            txt, ti, to = fn(a.model, prompts(lay, a.position, a.prompt_mode), path, a.max_tokens, a.effort, None if a.thinking == 'default' else a.thinking)
        except Exception as e:
            return {'path': path, 'level': lay.level, 'error': f'{type(e).__name__}: {e}', 'move': -1, 'optimal': sorted(optimal_set(lay)), 'in': 0, 'out': 0}
        mv = parse_move(txt)
        return {'path': path, 'level': lay.level, 'move': mv, 'optimal': sorted(optimal_set(lay)), 'correct': bool(mv in optimal_set(lay)), 'text': txt[-400:], 'full': txt, 'start': list(lay.start), 'in': ti, 'out': to}
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        rows = list(ex.map(one, items))
    ok = [r for r in rows if 'error' not in r]
    errs = [r for r in rows if 'error' in r]
    by_lvl, moves = ({}, {})
    for r in ok:
        by_lvl.setdefault(r['level'], []).append(r['correct'])
        moves[r['move']] = moves.get(r['move'], 0) + 1
    res = {'provider': a.provider, 'model': a.model, 'position': a.position, 'effort': a.effort, 'thinking': a.thinking, 'prompt_mode': a.prompt_mode, 'n': len(ok), 'n_error': len(errs), 'rung': '1_free_generation ONLY (hosted API exposes no hidden states or logits)', 'accuracy': sum((r['correct'] for r in ok)) / len(ok) if ok else None, 'unparsed': sum((1 for r in ok if r['move'] < 0)) / len(ok) if ok else None, 'first_move_counts': {MOVES[k] if k >= 0 else 'unparsed': v for k, v in sorted(moves.items(), key=lambda x: -x[1])}, 'by_level': {str(k): {'n': len(v), '1_free_generation': sum(v) / len(v)} for k, v in sorted(by_lvl.items())}, 'tokens_in': sum((r['in'] for r in rows)), 'tokens_out': sum((r['out'] for r in rows)), 'rows': rows}
    with open(a.out, 'w') as fh:
        json.dump(res, fh, indent=2)
    print(f"\naccuracy {res['accuracy']}  unparsed {res['unparsed']}  errors {len(errs)}")
    print('first-move distribution (a single dominant entry is constant emission):')
    for k, v in res['first_move_counts'].items():
        print(f'    {k}: {v}/{len(ok)}')
    print('by level:', {k: round(v['1_free_generation'], 3) for k, v in res['by_level'].items()})
    print(f"tokens in={res['tokens_in']} out={res['tokens_out']}  -> {a.out}")
    if errs:
        print('first error:', errs[0]['error'][:200])
if __name__ == '__main__':
    main()
