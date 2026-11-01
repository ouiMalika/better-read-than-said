"""Load the VSP maze benchmark into the Layout format."""
import os
import re
import glob
from env import Layout, compute_distance_map
VSP_ROOT = os.environ.get('VSP_MAZE_ROOT', 'data/VSP/maze/task-main/maps')

def _coord(s):
    """'row 4, column 3' (1-indexed) -> (3, 2) 0-indexed."""
    m = re.search('row\\s+(\\d+)\\s*,\\s*column\\s+(\\d+)', s, re.I)
    return (int(m.group(1)) - 1, int(m.group(2)) - 1)

def parse_pure_text(text):
    level = int(re.search('(\\d+)x(\\d+)\\s+map', text).group(1))
    player = _coord(re.search('player is at:\\s*([^;]+)', text, re.I).group(1))
    goal = _coord(re.search('goal is at:\\s*([^;.\\n]+)', text, re.I).group(1))
    holes = set()
    hm = re.search('hole\\(s\\) are at:\\s*([^.\\n]+)', text, re.I)
    if hm:
        for part in hm.group(1).split(';'):
            if re.search('row', part, re.I):
                holes.add(_coord(part))
    return Layout(level, frozenset(holes), goal, player)

def load_level(level, root=VSP_ROOT):
    """Return [(id, Layout, img_path)] for a VSP maze difficulty level."""
    tdir = os.path.join(root, f'level{level}', 'pure_text')
    idir = os.path.join(root, f'level{level}', 'img')
    out = []
    for f in sorted(glob.glob(os.path.join(tdir, '*.txt')), key=lambda p: int(os.path.basename(p)[:-4])):
        i = os.path.basename(f)[:-4]
        with open(f) as fh:
            lay = parse_pure_text(fh.read())
        out.append((i, lay, os.path.join(idir, f'{i}.png')))
    return out
if __name__ == '__main__':
    for level in range(3, 9):
        try:
            inst = load_level(level)
        except FileNotFoundError:
            continue
        solvable = sum((1 for _, lay, _ in inst if lay.start in compute_distance_map(lay)))
        imgs = sum((1 for _, _, p in inst if os.path.exists(p)))
        print(f'level {level}: {len(inst):3d} instances | {solvable} solvable by our BFS | {imgs} images present')
