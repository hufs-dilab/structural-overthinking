#!/usr/bin/env bash
set -euo pipefail

detect_python() {
  if [[ -n "${PYTHON_BIN:-}" ]]; then
    printf "%s\n" "$PYTHON_BIN"
  elif [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python3.12" ]]; then
    printf "%s\n" "${VIRTUAL_ENV}/bin/python3.12"
  elif [[ -x ".venv/bin/python3.12" ]]; then
    printf "%s\n" "$(pwd)/.venv/bin/python3.12"
  elif command -v python3.12 >/dev/null 2>&1; then
    command -v python3.12
  else
    command -v python3
  fi
}

PYTHON_BIN="$(detect_python)"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-Coder-480B-A35B-Instruct}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LIMIT="${LIMIT-full}"
if [[ "$LIMIT" == "full" ]]; then
  LIMIT=""
fi
REGISTRY_MODE="${REGISTRY_MODE:-mixed}"
MAX_TURNS="${MAX_TURNS:-1000}"
MAX_TOOL_CALL_DEPTH="${MAX_TOOL_CALL_DEPTH:-20}"
TIMEOUT="${TIMEOUT:-3600}"
DATASET_PATH="${DATASET_PATH:-dataset/human_eval.jsonl}"
LOG_BASE_DIR="${LOG_BASE_DIR:-qwen_code_logs}"
BASE_URL="${BASE_URL:-}"
MCP_PYTHON="${MCP_PYTHON:-}"
SERENA_COMMAND="${SERENA_COMMAND:-}"
SERENA_PYTHON="${SERENA_PYTHON:-$PYTHON_BIN}"
SERENA_CONTEXT="${SERENA_CONTEXT:-ide}"
CONTEXT7_NPX="${CONTEXT7_NPX:-}"
QWEN_BINARY="${QWEN_BINARY:-qwen}"
APPROVAL_MODE="${APPROVAL_MODE:-yolo}"
SKIP_LOOP_DETECTION="${SKIP_LOOP_DETECTION:-0}"
UNATTENDED_RETRY="${UNATTENDED_RETRY:-0}"
LIMIT_LABEL="${LIMIT:-full}"
LOG_DIR_NAME="${LOG_DIR_NAME:-qwen_code_${MODEL_ID//[^A-Za-z0-9_.-]/_}_humaneval_${REGISTRY_MODE}_limit${LIMIT_LABEL}}"

args=(
  "$PYTHON_BIN" -m src.run_qwen_code
  --dataset_path "$DATASET_PATH"
  --model_id "$MODEL_ID"
  --qwen_binary "$QWEN_BINARY"
  --num_workers "$NUM_WORKERS"
  --timeout "$TIMEOUT"
  --registry_mode "$REGISTRY_MODE"
  --max_turns "$MAX_TURNS"
  --max_tool_call_depth "$MAX_TOOL_CALL_DEPTH"
  --approval_mode "$APPROVAL_MODE"
  --serena_context "$SERENA_CONTEXT"
  --log_base_dir "$LOG_BASE_DIR"
  --log_dir_name "$LOG_DIR_NAME"
)

if [[ -n "$LIMIT" ]]; then
  args+=(--limit "$LIMIT")
fi

if [[ -n "$BASE_URL" ]]; then
  args+=(--base_url "$BASE_URL")
fi

if [[ -n "$MCP_PYTHON" ]]; then
  args+=(--mcp_python "$MCP_PYTHON")
fi

if [[ -n "$SERENA_COMMAND" ]]; then
  args+=(--serena_command "$SERENA_COMMAND")
fi

if [[ -n "$SERENA_PYTHON" ]]; then
  args+=(--serena_python "$SERENA_PYTHON")
fi

if [[ -n "$CONTEXT7_NPX" ]]; then
  args+=(--context7_npx "$CONTEXT7_NPX")
fi

if [[ "$UNATTENDED_RETRY" == "1" ]]; then
  args+=(--unattended_retry true)
fi

if [[ "$SKIP_LOOP_DETECTION" == "1" ]]; then
  args+=(--skip_loop_detection true)
fi

echo "Running: ${args[*]}"
"${args[@]}"
