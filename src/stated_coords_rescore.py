"""Re-score the stated-coordinate measurements with coord_parse."""
import json
import os
import re
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from coord_parse import stated_start, correct, correct_either
from vsp_maze import load_level
from readout_ladder import optimal_set
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
OPEN = {'qwen25_3b': 'Qwen2.5-VL-3B', 'qwen25_7b': 'Qwen2.5-VL-7B', 'qwen32': 'Qwen2.5-VL-32B'}
HOSTED = {'haiku': ('api_haiku_geom.json', 'Claude Haiku 4.5'), 'claude': ('api_claude_geom.json', 'Claude Opus 5')}

def layouts():
    out = {}
    for lv in range(3, 9):
        for i, lay, _p in load_level(lv):
            if optimal_set(lay):
                out[lv, str(i)] = lay
    return out

def score(pairs):
    """pairs: [(text, true start)] -> counts."""
    n = len(pairs)
    stated = hit = hit_e = 0
    for txt, start in pairs:
        got = stated_start(txt or '')
        if got is None:
            continue
        stated += 1
        hit += correct(got, start)
        hit_e += correct_either(got, start)
    return {'n': n, 'stated_any': stated / n, 'stated_correct': hit / n, 'stated_correct_either_indexing': hit_e / n}

def key_from_path(p):
    m = re.search('level(\\d+)/img/(\\d+)\\.png', p)
    return (int(m.group(1)), m.group(2)) if m else None

def main():
    lay = layouts()
    res = {'parser': 'coord_parse.py (rewritten 2026-09-04)', 'models': {}}
    for tag, name in OPEN.items():
        d = json.load(open(os.path.join(R, f'geom_{tag}_full.json')))
        pairs = []
        for k, txt in d['plans'].items():
            lv, i = k[1:].split('#')
            pairs.append((txt, lay[int(lv), i].start))
        res['models'][tag] = {'name': name, 'source': f'geom_{tag}_full.json', **score(pairs)}
    for tag, (f, name) in HOSTED.items():
        d = json.load(open(os.path.join(R, f)))
        pairs = []
        for r in d['rows']:
            k = key_from_path(r['path'])
            if k is None or k not in lay:
                continue
            pairs.append((r.get('full') or r.get('text', ''), lay[k].start))
        res['models'][tag] = {'name': name, 'source': f, **score(pairs)}
    out = os.path.join(R, 'stated_coords_rescored.json')
    json.dump(res, open(out, 'w'), indent=2)
    for tag, v in res['models'].items():
        print(f"{v['name']:<18} n={v['n']:<4} correct={v['stated_correct']:.4f} either={v['stated_correct_either_indexing']:.4f} stated_any={v['stated_any']:.3f}")
    print('saved ->', out)
if __name__ == '__main__':
    main()
