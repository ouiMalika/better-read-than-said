"""Generate 5x5 mazes in the VSP format."""
import os
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
import argparse
import numpy as np
from PIL import Image
import gym
from env import Layout, generate_instance
from data import optimal_positions
from symbolic import positions_to_actions
_A2VSP = {'up': 'U', 'down': 'D', 'left': 'L', 'right': 'R'}

def to_desc(lay):
    g = [['F'] * lay.level for _ in range(lay.level)]
    for r, c in lay.holes:
        g[r][c] = 'H'
    g[lay.goal[0]][lay.goal[1]] = 'G'
    g[lay.start[0]][lay.start[1]] = 'S'
    return [''.join(row) for row in g]

def to_pure_text(lay):

    def rc(p):
        return f'row {p[0] + 1}, column {p[1] + 1}'
    holes = '; '.join((f'Row {r + 1}, Column {c + 1}' for r, c in sorted(lay.holes)))
    return f'This is a {lay.level}x{lay.level} map.\nThe player is at: {rc(lay.start)};\nThe hole(s) are at: {holes};\nThe goal is at: {rc(lay.goal)}.'

def render_img(desc):
    env = gym.make('FrozenLake-v1', desc=desc, is_slippery=False, render_mode='rgb_array')
    env.reset()
    arr = env.render()
    env.close()
    return Image.fromarray(np.asarray(arr, dtype=np.uint8))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=2000)
    ap.add_argument('--level', type=int, default=5)
    ap.add_argument('--out', default='vsp_train')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    for sub in ('img', 'text', 'plan'):
        os.makedirs(os.path.join(args.out, sub), exist_ok=True)
    i = kept = 0
    while kept < args.n:
        try:
            lay = generate_instance(level=args.level, hole_frac=0.15, min_len=3, seed=args.seed * 1000000 + i)
        except RuntimeError:
            i += 1
            continue
        pos = optimal_positions(lay)
        plan = ','.join((_A2VSP[a] for a in positions_to_actions(pos)))
        render_img(to_desc(lay)).save(os.path.join(args.out, 'img', f'{kept}.png'))
        with open(os.path.join(args.out, 'text', f'{kept}.txt'), 'w') as f:
            f.write(to_pure_text(lay))
        with open(os.path.join(args.out, 'plan', f'{kept}.txt'), 'w') as f:
            f.write(plan)
        kept += 1
        i += 1
        if kept % 200 == 0:
            print(f'  {kept}/{args.n}', flush=True)
    print(f'done: {kept} instances -> {args.out}', flush=True)
if __name__ == '__main__':
    main()
