#!/bin/bash
# Portable eval driver: evaluates each model name (passed as args) on test_normal,
# sequentially, 8 parallel task-workers each. Caller sets APPPY, VLLM_BIN,
# CUDA_VISIBLE_DEVICES (and optional PORT). Continues on error. Writes
# results_<model>.json per model. Usage:
#   APPPY=.../env/bin/python VLLM_BIN=.../vllm CUDA_VISIBLE_DEVICES=3,4 \
#     bash run_full_eval.sh diff_1_iter5 joint_iter9
cd "$(dirname "$0")"
: "${APPPY:?set APPPY to the appworld-env python}"
: "${VLLM_BIN:?set VLLM_BIN to the vllm binary}"; export VLLM_BIN
: "${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES (GPU pair)}"; export CUDA_VISIBLE_DEVICES
PORT="${PORT:-8000}"

for m in "$@"; do
  echo "==================== EVAL $m START $(date '+%F %T') ===================="
  "$APPPY" ppo_baseline/evaluate_model.py \
    --model "merged_models/$m" \
    --dataset test_normal \
    --workers 8 --port "$PORT" \
    --out "results_${m}.json" \
    2>&1 | sed -u -E "s/\x1b\[[0-9;]*m//g"
  echo "==================== EVAL $m END $(date '+%F %T') ===================="
  pkill -9 -f 'vllm_env/bin/vllm' 2>/dev/null || true
  sleep 10
done
echo "==================== ALL EVALS COMPLETE $(date '+%F %T') ===================="
