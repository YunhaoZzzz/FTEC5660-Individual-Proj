#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi
: "${DEEPSEEK_API_KEY:?Set DEEPSEEK_API_KEY in .env or the environment}"

PYTHON="${PYTHON:-$SCRIPT_DIR/.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="python3"
fi
MODEL="${MODEL:-deepseek-flash}"
BASE_URL="${DEEPSEEK_BASE_URL:-https://api.deepseek.com}"
OUT_DIR="${OUT_DIR:-./outputs/course_reproduction/deepseek-flash}"
BASE_SEED="${BASE_SEED:-42}"
N_RUNS="${N_RUNS:-3}"
WORKERS="${API_WORKERS:-4}"
TEMPERATURE="${TEMPERATURE:-0.7}"

mkdir -p "$OUT_DIR"
export HF_HOME="${HF_HOME:-$SCRIPT_DIR/.cache/huggingface}"
mkdir -p "$HF_HOME"
if [[ ! -f "$OUT_DIR/default_metrics.json" ]]; then
  "$PYTHON" src/run_eval.py \
    --model_path "$MODEL" \
    --backend deepseek \
    --api_base_url "$BASE_URL" \
    --api_workers "$WORKERS" \
    --data_path ./data/tasks_v1_test.json \
    --envs CalculatorEnv \
    --reasoning_mode no_reasoning \
    --prompt_mode current \
    --temperature "$TEMPERATURE" \
    --max_new_tokens 512 \
    --n_runs "$N_RUNS" \
    --seed "$BASE_SEED" \
    --record_mode lite \
    --output_path "$OUT_DIR/default.json"
else
  echo "Default metrics already exist; skipping completed group."
fi

if [[ ! -f "$OUT_DIR/sparse_tool_metrics.json" ]]; then
  "$PYTHON" src/run_eval.py \
    --model_path "$MODEL" \
    --backend deepseek \
    --api_base_url "$BASE_URL" \
    --api_workers "$WORKERS" \
    --data_path ./data/tasks_v1_test.json \
    --envs CalculatorEnv \
    --reasoning_mode no_reasoning \
    --prompt_mode sparse_tool \
    --temperature "$TEMPERATURE" \
    --max_new_tokens 512 \
    --n_runs "$N_RUNS" \
    --seed "$BASE_SEED" \
    --record_mode lite \
    --output_path "$OUT_DIR/sparse_tool.json"
else
  echo "Sparse-policy metrics already exist; skipping completed group."
fi

"$PYTHON" src/summarize_course_results.py \
  --root "$OUT_DIR" \
  --output "$OUT_DIR/summary.json"

echo "Results saved under $OUT_DIR"
