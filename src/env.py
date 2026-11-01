"""FrozenLake environment, breadth-first distances and maze generation."""
from collections import deque, namedtuple
import random
ACTIONS = ('up', 'down', 'left', 'right')
DELTAS = {'up': (-1, 0), 'down': (1, 0), 'left': (0, -1), 'right': (0, 1)}
DELTA_TO_ACTION = {v: k for k, v in DELTAS.items()}
FROZEN, HOLE, GOAL, AGENT = (0, 1, 2, 3)
Layout = namedtuple('Layout', 'level holes goal start')

def in_bounds(level, r, c):
    return 0 <= r < level and 0 <= c < level

def step(layout, pos, action):
    """One transition."""
    dr, dc = DELTAS[action]
    r, c = pos
    tr, tc = (r + dr, c + dc)
    if not in_bounds(layout.level, tr, tc):
        return (pos, 'invalid')
    if (tr, tc) in layout.holes:
        return ((tr, tc), 'hole')
    if (tr, tc) == layout.goal:
        return ((tr, tc), 'goal')
    return ((tr, tc), 'move')

def compute_distance_map(layout):
    """BFS shortest-path (steps) from every safe cell to the goal, avoiding holes."""
    if layout.goal in layout.holes or not in_bounds(layout.level, *layout.goal):
        return {}
    dist = {layout.goal: 0}
    q = deque([layout.goal])
    while q:
        cur = q.popleft()
        for a in ACTIONS:
            dr, dc = DELTAS[a]
            nb = (cur[0] + dr, cur[1] + dc)
            if not in_bounds(layout.level, *nb) or nb in layout.holes or nb in dist:
                continue
            dist[nb] = dist[cur] + 1
            q.append(nb)
    return dist

def state_index(pos, level):
    return pos[0] * level + pos[1]

def encode_frame(layout, pos):
    """Lossless grid -> token tuple (row-major, level*level tokens)."""
    toks = []
    for r in range(layout.level):
        for c in range(layout.level):
            cell = (r, c)
            if cell == pos:
                toks.append(AGENT)
            elif cell == layout.goal:
                toks.append(GOAL)
            elif cell in layout.holes:
                toks.append(HOLE)
            else:
                toks.append(FROZEN)
    return tuple(toks)

def decode_agent(frame, level):
    """Agent (r,c) from a token frame, or None if the frame is malformed (a half-trained model may emit zero or several AGENT tokens)."""
    if frame.count(AGENT) != 1:
        return None
    i = frame.index(AGENT)
    return (i // level, i % level)

def infer_action(frame_a, frame_b, level):
    """Dynamics interpreter D: the action taken between two frames, or 'invalid' (malformed frame, or a non-unit / hallucinated teleporting move)."""
    ra = decode_agent(frame_a, level)
    rb = decode_agent(frame_b, level)
    if ra is None or rb is None:
        return 'invalid'
    delta = (rb[0] - ra[0], rb[1] - ra[1])
    return DELTA_TO_ACTION.get(delta, 'invalid')

def generate_instance(level=5, hole_frac=0.15, min_len=4, seed=0, max_attempts=800):
    """A solvable, non-trivial FrozenLake instance (deterministic given seed)."""
    rng = random.Random(seed)
    cells = [(r, c) for r in range(level) for c in range(level)]
    n_holes = int(round(hole_frac * level * level))
    for _ in range(max_attempts):
        holes = set(rng.sample(cells, n_holes)) if n_holes else set()
        remaining = [c for c in cells if c not in holes]
        if len(remaining) < 2:
            continue
        goal, start = rng.sample(remaining, 2)
        layout = Layout(level, frozenset(holes), goal, start)
        d = compute_distance_map(layout)
        if start in d and d[start] >= min_len:
            return layout
    raise RuntimeError('could not generate instance; loosen params')

def generate_dataset(n, **kwargs):
    base = kwargs.pop('seed', 0)
    out, i = ([], 0)
    while len(out) < n and i < n * 60:
        try:
            out.append(generate_instance(seed=base * 100000 + i, **kwargs))
        except RuntimeError:
            pass
        i += 1
    if len(out) < n:
        raise RuntimeError(f'only generated {len(out)}/{n} instances')
    return out
