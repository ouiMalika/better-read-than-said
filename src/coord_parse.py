"""Coordinate parser shared by the open-model and hosted-model measurements."""
import re
PLAYER = "(?:player|you\\b|you're|your position|current position|agent)"
OTHER = '(?:goal|target|destination|hole|chest|gift)'
ROWCOL = re.compile('row[^\\S\\n]*[:=]?[^\\d\\n]{0,4}(\\d+)[^\\d\\n]{0,12}?col(?:umn)?[^\\S\\n]*[:=]?[^\\d\\n]{0,4}(\\d+)', re.I)
PAREN = re.compile('\\(?\\s*(\\d+)\\s*[,;]\\s*(\\d+)\\s*\\)?')

def _match_in(line):
    """The first (row, col) on this line, or None."""
    for m in (ROWCOL.search(line), PAREN.search(line)):
        if not m:
            continue
        before = line[:m.start()].lower()
        p = max((before.rfind(w) for w in ('player', 'you', 'agent', 'current position')), default=-1)
        o = max((before.rfind(w) for w in ('goal', 'target', 'destination', 'hole', 'chest', 'gift')), default=-1)
        if o > p:
            continue
        if p < 0 and (not re.match('\\s*1[.)]', line)):
            continue
        return (int(m.group(1)), int(m.group(2)))
    return None

def stated_start(txt):
    """First stated (row, col) for the PLAYER, or None."""
    for line in txt.splitlines():
        if not re.search(f'{PLAYER}|^\\s*1[.)]', line, re.I):
            continue
        got = _match_in(line)
        if got:
            return got
    return None

def correct(stated, start):
    """Score against the 1-indexed convention the prompt declares ("row 1 is the top")."""
    if stated is None:
        return False
    r, c = stated
    return (r - 1, c - 1) == tuple(start)

def correct_either(stated, start):
    """Also accept the 0-indexed reading."""
    if stated is None:
        return False
    r, c = stated
    return (r, c) == tuple(start) or (r - 1, c - 1) == tuple(start)
