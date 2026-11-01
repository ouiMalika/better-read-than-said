"""Stated-coordinate accuracy for a hosted model."""
import re
import sys
import json
from coord_parse import stated_start, correct
d = json.load(open(sys.argv[1]))
tot = hit = stated = 0
for r in d['rows']:
    txt = r.get('full') or r.get('text', '')
    start = tuple(r.get('start', ()))
    if not start:
        continue
    tot += 1
    got = stated_start(txt)
    if got is None:
        continue
    stated += 1
    if correct(got, start):
        hit += 1
res = {'model': d['model'], 'prompt_mode': d.get('prompt_mode'), 'n': tot, 'stated_any': stated / max(tot, 1), 'stated_correct': hit / max(tot, 1)}
print('%s  n=%d  stated %.2f  CORRECT %.3f' % (d['model'], tot, res['stated_any'], res['stated_correct']))
if len(sys.argv) > 2:
    json.dump(res, open(sys.argv[2], 'w'), indent=2)
    print('saved ->', sys.argv[2])
