# Better Read Than Said

Code and results for *Better Read Than Said: Decoding Spatial-Planning Moves from the Hidden States
of Vision–Language Models* (VLM4RWD workshop, NeurIPS 2026).

Malika Buribayeva, Qiuyu Tang, Cristian-Ioan Vasile, Aparna Bharati

## Setup

```
pip install -r requirements.txt
```

The VSP mazes are not redistributed here. Download the VSP benchmark
(https://github.com/UCSB-NLP-Chang/Visual-Spatial-Planning) and point `VSP_MAZE_ROOT` at its
`maze/task-main/maps` directory (default: `data/VSP/maze/task-main/maps`). All runs use the
published images unmodified, levels 3 to 8, 100 instances each.

Model identifiers and Hugging Face revisions are listed in `src/readout_ladder.py` (`MODELS`) and in
Appendix J of the paper. Decoding is greedy with a 512-token budget.

## Reproducing the paper

Run every command from `src/`. Each script writes a JSON file; the files behind the paper's numbers
are in `results/`.

| Paper | Command |
|---|---|
| Table 2, methods 1–5 | `python readout_ladder.py --levels 3 4 5 6 7 8 --pool --model qwen32 --folds 5 --out ../results/results_ladderP_qwen32_hidden.json` (model keys in `MODELS`) |
| Table 2, refit and seeds (App. J) | `python fit_from_ckpt.py --ckpt <checkpoint> --model <key> --seed <s> --out <file>` |
| Table 1 (how plans end) | `python gen_plans.py --model <key> --out ../results/plans_<key>.json`; for Qwen3-VL-32B and Qwen3.8-27B, `python score_ckpt_plans.py --ckpt <checkpoint> --model <key> --out <file>` |
| Qwen3.8-27B, long budget and forced answer (Sec. 5) | `python vllm_plans.py --model qwen38 --dataset vsp --max-tokens 16384 --out <file>`; `python force_answer.py --model qwen38 --dataset vsp --ckpt <checkpoint> --out <file>` |
| Sec. 3, stated coordinates | `api_stated_coords.py`, `stated_coords_rescore.py`, `coord_parse.py` |
| Sec. 3, hosted models | `api_rung1.py` (needs `ANTHROPIC_API_KEY`) |
| Sec. 3 / App. F, written derivation | `verify_chain.py` |
| Sec. 4, agreement and self-trapping (App. H) | `gen_vsp_data.py`, `agreement_action_perception.py`, `agreement_multistep.py` |
| Sec. 6 / App. K, editing the hidden state | `probe_steer.py` |
| Sec. 7 / App. D and G, feedback and unexpected step | `vlm_closed_loop.py`, `unexpected_step.py`, `analyze_unexpected.py` |
| Sec. 8 / App. B, failures and remedies | `vsp_failure_analysis.py`, `failure_taxonomy.py`, `perception_solver_v2.py`, `vlm_remedy.py` |
| App. A, perception and set size | `probe_singlehole.py`, `probe_setsize.py`, `probe_spatial_v2.py` |
| App. C, per-level scores | `pathlen_by_level.py` |
| App. E, counterfactual pairs | `vipq_gate_v2.py` |

Each script prints its options with `--help`.

## Classifiers

Methods 3–5 read the hidden state of the last prompt token at 0.75 of depth (layer per model in
`MODELS` and Sec. 2). They are fit with 5-fold cross-validation over mazes (480 train / 120 test),
standardized features, Adam for 1500 full-batch steps, learning rate 1e-2 (linear) and 1e-3
(one hidden layer, 512 units), weight decay 1e-3. Seeds 0–4 are in `results/seeds_vsp/`.

## Citation

```
@inproceedings{buribayeva2026better,
  title     = {Better Read Than Said: Decoding Spatial-Planning Moves from the Hidden States of Vision--Language Models},
  author    = {Buribayeva, Malika and Tang, Qiuyu and Vasile, Cristian-Ioan and Bharati, Aparna},
  booktitle = {NeurIPS 2026 Workshop on Grounded and Faithful Vision-Language Models for Real-World Deployment},
  year      = {2026}
}
```
