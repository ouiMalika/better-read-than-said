"""Print Tables 1 and 2 of the paper from the result files."""
import json
import os

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')
MODELS = [('qwen25_3b', 'Qwen2.5-VL-3B'), ('qwen', 'Qwen2-VL-7B'), ('qwen25_7b', 'Qwen2.5-VL-7B'),
          ('llava', 'LLaVA-1.5-7B'), ('deepseek', 'DeepSeek-VL2-27B'), ('qwen32', 'Qwen2.5-VL-32B'),
          ('qwen3vl32', 'Qwen3-VL-32B'), ('qwen38', 'Qwen3.8-27B')]
ROWS = [('1_free_generation', '1. free generation'), ('2_forced_choice', '2. forced choice'),
        ('3_probe_then_rule', '3. probe + hand rule'), ('4_linear_action', '4. linear head'),
        ('4_linear_action_shuffled', '   shuffled'), ('5_mlp_action', '5. MLP head')]


def load(name):
    with open(os.path.join(R, name)) as f:
        return json.load(f)


def cell(v):
    return f'{v[0]:.3f} ±{v[1]:.3f}' if isinstance(v, list) else f'{v:.3f}'


def main():
    print('Table 2: first-move accuracy, 600 VSP instances, levels 3-8 pooled')
    print(f"{'method':<22}" + ''.join(f'{n:>18}' for _, n in MODELS))
    ladder = {t: load(f'results_ladderP_{t}_hidden.json')['rungs']['pooled'] for t, _ in MODELS}
    for key, label in ROWS:
        print(f'{label:<22}' + ''.join(f'{cell(ladder[t][key]):>18}' for t, _ in MODELS))
    print()
    print('Table 1: how the written plans end')
    print(f"{'outcome':<22}" + ''.join(f'{n:>18}' for _, n in MODELS))
    plans = {t: load(f'plans_{t}.json') for t, _ in MODELS}
    print(f"{'episode success':<22}" + ''.join(f"{plans[t]['episode_success']:>18.3f}" for t, _ in MODELS))
    for key, label in (('hole', 'enters a hole'), ('ran_out', 'runs out of moves'), ('unparsed', 'parser finds no plan')):
        print(f'{label:<22}' + ''.join(f"{plans[t]['failure_modes'].get(key, 0):>18}" for t, _ in MODELS))


if __name__ == '__main__':
    main()
