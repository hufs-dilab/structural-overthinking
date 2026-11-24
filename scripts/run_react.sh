#!/usr/bin/env bash
# ReAct (smolagents) experiments on AIME 2025 / GPQA-Diamond / MMLU-Pro / HumanEval.
#
#   TOOL_MODE=tool  -> ordinary ToolRet tools only (baseline)
#   TOOL_MODE=mcp   -> cycle-inducing MCP tools only
#   TOOL_MODE=all   -> ordinary + cycle-inducing MCP tools
#
# Example:
#   MODEL_ID=Qwen/Qwen3-235B-A22B BASE_URL=http://localhost:8000/v1 \
#     DATASET=gpqa_diamond TOOL_MODE=all ./scripts/run_react.sh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
DATASET="${DATASET:-all}"
AGENT_TYPE="${AGENT_TYPE:-tool_call}"
TOOL_MODE="${TOOL_MODE:-all}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MAX_TOKENS="${MAX_TOKENS:-16384}"
NOWAIT="${NOWAIT:-0}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-react_${TOOL_MODE}}"

args=(
  "$PYTHON_BIN" -m src.run_react
  --dataset_name "$DATASET"
  --agent_type "$AGENT_TYPE"
  --tool_mode "$TOOL_MODE"
  --num_workers "$NUM_WORKERS"
  --max_tokens "$MAX_TOKENS"
  --experiment_name "$EXPERIMENT_NAME"
)

[[ -n "${MODEL_ID:-}" ]] && args+=(--model_id "$MODEL_ID")
[[ -n "${BASE_URL:-}" ]] && args+=(--api_base "$BASE_URL")
[[ -n "${API_KEY:-}" ]] && args+=(--api_key "$API_KEY")
[[ -n "${TOOL_PARSER:-}" ]] && args+=(--tool_parser "$TOOL_PARSER")
[[ "$NOWAIT" == "1" ]] && args+=(--nowait)

echo "Running: ${args[*]}"
"${args[@]}"
