# Reproducing "Merging Reinforced Specialists vs. Joint Multi-Task RL"

This directory holds the analysis/figure code, the small per-task eval logs, and the
LaTeX source for the paper.

## Quick reproduction — every statistic and figure, no GPU, seconds on a laptop

The five per-task result logs (`results_*.json`, 100 KB) and the derived geometry
summaries (`paper/data/*.csv`, `paper/calibration.json`, few KB) are committed, so all
reported numbers and figures regenerate on a laptop with no GPU, no model weights, and no
AppWorld install:

```bash
pip install -r ../requirements.txt      # numpy, scipy, matplotlib (+ torch only for Step 5b)
python build_stats.py                   # Tables 1-2 (per-difficulty TGC/SGC, CIs, McNemar/Fisher)
python mechanism_stats.py               # Table 3 (metric reversal, Wilcoxon, emergence, averaging)
python fig_forest.py fig_trajectory.py fig_module_cosine.py fig_calibration.py fig_jgrid.py
# ^ or: for f in fig_*.py; do python "$f"; done      -> paper/figures/*.pdf
```

- **Tables 1–3 and Fig 3** are *recomputed from the per-task logs* — the fully
  from-scratch, editor-verifiable path.
- **Figs 1, 2, 4, 5** (task-vector geometry) are *replotted from the committed summary
  files* (`paper/data/*.csv`, `calibration.json`). Re-deriving those summaries from the
  raw 15 GB LoRA weights is the checkpoint/GPU-gated path in Step 5b below.

Per-difficulty tables use a committed `paper/data/task_difficulty.json`, so no AppWorld
install is needed; scripts fall back to `$APPWORLD_ROOT` metadata if it is absent.

## Full reproduction from scratch

The steps below regenerate everything from raw training. The large or regenerable
artifacts (LoRA checkpoints, merged models) are **not** committed.

## Reproduction chain (overview)

```
train (main.py) ──▶ LoRA checkpoints ──▶ merge ──▶ merged_models/ ──▶ eval ──▶ results_*.json
                          │                                                          │
                          └────────────── geometry scripts ──▶ paper/data/*.csv      │
                                                                     │               │
                                                     figure/stats scripts ──▶ figures + tables
```

If you already have `results_*.json` and the LoRA checkpoints, skip to
**Step 5 (analysis)** — everything downstream is CPU-only and needs no GPU.

## Environments

Three separate virtualenvs are required (torch/vLLM and the AppWorld fork segfault if
co-located; mergekit needs a pinned `transformers`):

| env | purpose | key pin |
|-----|---------|---------|
| `appworld-env` | AppWorld fork, rollouts, evaluation | AppWorld fork |
| `vllm_env`     | vLLM serving + the trainer subprocess | vLLM, torch |
| `mergekit_env` | TIES / RAM merges | `transformers>=4.51,<4.58` (Qwen3) |

The analysis/figure scripts (Steps 5–6) run in any env with `torch`, `numpy`, `scipy`,
`matplotlib`, `safetensors`, `transformers`.

## External inputs not in git

- **LoRA checkpoints** `checkpoints/<arm>/lora_iter_<N>/` (~15 GB) — regenerate in Step 1,
  or restore from the Expansion drive. Needed by the geometry scripts.
- **Merged full models** `merged_models/<name>/` (~92 GB, symlink to Expansion) —
  regenerate in Step 3. Needed only for eval.
- **AppWorld benchmark data** at `$APPWORLD_ROOT` — the benchmark install; the
  per-difficulty analysis reads `$APPWORLD_ROOT/data/tasks/<id>/ground_truth/metadata.json`.
- **`results_*.json`** — regenerate in Step 4.

---

## Step 1 — Train the specialists and the joint model

Base model: `Qwen/Qwen3-8B`. Trainer: LOOP (RLOO advantage in a PPO clipped objective)
with LoRA rank 16 on all attention + MLP projections. Difficulty is selected in
`ppo_baseline/main.py` via the `difficulties` list (e.g. `[1]`, `[2]`, `[1, 2]`), which
filters tasks by their AppWorld difficulty label.

```bash
# from repo root, in appworld-env (spawns the vllm_env trainer subprocess)
python ppo_baseline/main.py --iterations 10        # set difficulties=[1] -> diff-1 specialist
python ppo_baseline/main.py --iterations 10        # set difficulties=[2] -> diff-2 specialist
python ppo_baseline/main.py --iterations 10        # set difficulties=[1,2] -> joint
```

Key hyperparameters (per the LOOP paper): K=6 rollouts/task, sampling temperature 1.0,
LR 5e-5, one update epoch/batch, advantage filter |Â|<0.01, interaction budget 40.
Adapters are saved to `checkpoints/<arm>/lora_iter_<N>/`. Mean training reward per
iteration is printed to the run log (`Average Reward:` lines).

## Step 2 — Select peak checkpoints

Select each arm's checkpoint of **maximum mean training reward** (from the run logs), not
on `test_normal`. In our run: **diff-1 = iter5, diff-2 = iter5, joint = iter9**. RL was
unstable past the peak (reward collapsed after iter5), so later checkpoints are unused.

## Step 3 — Build the merged full models

```bash
# specialists / joint: LoRA -> full model
python ppo_baseline/merge_lora_to_base.py   # lora_path, base_model, output_path

# TIES merge of the two specialists (mergekit; run with mergekit_env)
python ppo_baseline/merge_specialists.py    # method=ties, density 0.5

# RAM+ merge: RAM reference impl, mode arm-r-v2, rescale r=1.2 (mergekit_env)
```

Outputs go to `merged_models/{diff_1_iter5,diff_2_iter5,joint_iter9,merged_ties,merged_ram}`.

## Step 4 — Evaluate on test_normal (168 tasks)

Deterministic: greedy decoding (temperature 0), seed 100, vLLM tensor-parallel 2.

```bash
APPPY=/path/to/appworld-env/bin/python \
VLLM_BIN=/path/to/vllm_env/bin/vllm \
CUDA_VISIBLE_DEVICES=3,4 \
  bash run_full_eval.sh diff_1_iter5 diff_2_iter5 joint_iter9 merged_ties merged_ram
```

Writes `results_<model>.json` (per-task `success` + partial-credit `score`) to the repo
root — the inputs to all tables and the performance figure.

## Step 5a — Performance analysis (CPU only, from `results_*.json`)

```bash
# performance tables + bootstrap CIs + McNemar/Fisher  -> paper/stats.json
python paper/build_stats.py
# continuous-metric + mechanism analysis               -> paper/mechanism.json
python paper/mechanism_stats.py
# per-difficulty TGC/SGC (uses committed task_difficulty.json; $APPWORLD_ROOT optional)
python analyze_results.py results_*.json
```

## Step 5b — Geometry analysis (needs the LoRA checkpoints from Step 1)

Regenerates the committed summaries; skip if you only need to replot from them.

```bash
python cosine_analysis.py       # snapshot cosine    -> paper/data/module_cosine.csv
python cosine_trajectory.py     # per-iter cosine    -> paper/data/trajectory.csv
python paper/floor_ceiling.py   # floor/ceiling calib -> paper/calibration.json
python paper/j_objective.py     # interference grid   -> paper/data/interference_grid.csv
```

## Step 6 — Figures

```bash
python paper/fig_trajectory.py      # Fig 1
python paper/fig_module_cosine.py   # Fig 2
python paper/fig_forest.py          # Fig 3  (reads paper/stats.json)
python paper/fig_calibration.py     # Fig 4  (reads paper/calibration.json)
python paper/fig_jgrid.py           # Fig 5
```

Figures are written to `paper/figures/*.{pdf,png}`.

## Step 7 — Compile the paper

Upload `paper/main.tex` + `paper/figures/*.pdf` to Overleaf (or run `pdflatex` twice
locally). The bibliography is inline; no `.bib` needed. Remaining `\TODO` markers
(authors, LOOP arXiv id, a few merge hyperparameters) must be filled before submission.
