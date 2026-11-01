"""Mean optimal path length per VSP level."""
import json
import statistics
from vsp_maze import load_level
from readout_ladder import optimal_set
from env import compute_distance_map
out = {'note': 'no model involved; computed from the VSP maze set alone', 'p_per_move': 0.35, 'by_level': {}}
for lv in [3, 4, 5, 6, 7, 8]:
    L = [compute_distance_map(lay)[lay.start] for _, lay, p in load_level(lv) if optimal_set(lay)]
    m = statistics.mean(L)
    out['by_level'][str(lv)] = {'n': len(L), 'mean_optimal_path_len': round(m, 1), 'implied_episode_success_at_p0.35': round(0.35 ** m, 3)}
with open('results/results_pathlen_by_level.json', 'w') as fh:
    json.dump(out, fh, indent=2)
print(json.dumps(out, indent=2))
