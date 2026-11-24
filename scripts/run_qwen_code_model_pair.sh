#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

detect_python() {
  if [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python3.12" ]]; then
    printf "%s\n" "${VIRTUAL_ENV}/bin/python3.12"
  elif [[ -x ".venv/bin/python3.12" ]]; then
    printf "%s\n" "$(pwd)/.venv/bin/python3.12"
  elif command -v python3.12 >/dev/null 2>&1; then
    command -v python3.12
  else
    command -v python3
  fi
}

if [[ -z "${MODEL_ID:-}" ]]; then
  echo "MODEL_ID is required" >&2
  exit 2
fi

MODEL_SLUG="${MODEL_SLUG:-$(printf "%s" "$MODEL_ID" | sed 's/[^A-Za-z0-9_.-]/_/g')}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
RUNNER="${RUNNER:-./scripts/run_qwen_code.sh}"
LOG_BASE_DIR="${LOG_BASE_DIR:-qwen_code_logs}"
DATASET_PATH="${DATASET_PATH:-dataset/human_eval.jsonl}"

LIMIT="${LIMIT-}"
if [[ "$LIMIT" == "full" ]]; then
  LIMIT=""
fi

NUM_WORKERS="${NUM_WORKERS:-16}"
TIMEOUT="${TIMEOUT:-3600}"
MAX_TURNS="${MAX_TURNS:-1000}"
MAX_TOOL_CALL_DEPTH="${MAX_TOOL_CALL_DEPTH:-20}"
MCP_PYTHON="${MCP_PYTHON:-$(detect_python)}"
SERENA_PYTHON="${SERENA_PYTHON:-$MCP_PYTHON}"
SERENA_COMMAND="${SERENA_COMMAND:-}"
SERENA_CONTEXT="${SERENA_CONTEXT:-ide}"
CONTEXT7_NPX="${CONTEXT7_NPX:-}"
BASE_URL="${BASE_URL:-}"
QWEN_BINARY="${QWEN_BINARY:-qwen}"
APPROVAL_MODE="${APPROVAL_MODE:-yolo}"
UNATTENDED_RETRY="${UNATTENDED_RETRY:-0}"
STOP_ON_FAILURE="${STOP_ON_FAILURE:-1}"
DRY_RUN="${DRY_RUN:-0}"

REGISTRY_MODES=(${REGISTRY_MODES:-normal mixed})

STATUS_DIR="${LOG_BASE_DIR}/model_pair_status"
mkdir -p "$STATUS_DIR"
STATUS_FILE="${STATUS_DIR}/${MODEL_SLUG}_${RUN_TAG}.tsv"
printf "run_tag\tmodel_id\tregistry_mode\tlog_dir\tstatus\texit_code\tstarted_at\tended_at\tskip_loop_detection\n" > "$STATUS_FILE"

echo "Run tag: ${RUN_TAG}"
echo "Model: ${MODEL_ID}"
echo "Registries: ${REGISTRY_MODES[*]}"
echo "Dataset: ${DATASET_PATH}"
echo "Limit: ${LIMIT:-full}"
echo "Workers per run: ${NUM_WORKERS}"
echo "Timeout per sample: ${TIMEOUT}s"
echo "Approval mode: ${APPROVAL_MODE}"
echo "Qwen binary: ${QWEN_BINARY}"
echo "MCP Python: ${MCP_PYTHON}"
echo "Serena Python: ${SERENA_PYTHON}"
echo "Status file: ${STATUS_FILE}"
echo

failures=0
for registry_mode in "${REGISTRY_MODES[@]}"; do
  log_dir_name="full_humaneval_${MODEL_SLUG}_${registry_mode}_${RUN_TAG}"
  started_at="$(date -Iseconds)"
  skip_loop_detection="0"
  if [[ "$registry_mode" == "mixed" ]]; then
    skip_loop_detection="1"
  fi

  echo "================================================================================"
  echo "Running model=${MODEL_ID} registry=${registry_mode} log=${log_dir_name}"
  echo "================================================================================"

  cmd=(
    env
    "MODEL_ID=${MODEL_ID}"
    "REGISTRY_MODE=${registry_mode}"
    "LIMIT=${LIMIT}"
    "NUM_WORKERS=${NUM_WORKERS}"
    "TIMEOUT=${TIMEOUT}"
    "MAX_TURNS=${MAX_TURNS}"
    "MAX_TOOL_CALL_DEPTH=${MAX_TOOL_CALL_DEPTH}"
    "DATASET_PATH=${DATASET_PATH}"
    "LOG_BASE_DIR=${LOG_BASE_DIR}"
    "LOG_DIR_NAME=${log_dir_name}"
    "MCP_PYTHON=${MCP_PYTHON}"
    "SERENA_PYTHON=${SERENA_PYTHON}"
    "SERENA_CONTEXT=${SERENA_CONTEXT}"
    "QWEN_BINARY=${QWEN_BINARY}"
    "APPROVAL_MODE=${APPROVAL_MODE}"
    "SKIP_LOOP_DETECTION=${skip_loop_detection}"
    "UNATTENDED_RETRY=${UNATTENDED_RETRY}"
  )
if [[ -n "$SERENA_COMMAND" ]]; then
    cmd+=("SERENA_COMMAND=${SERENA_COMMAND}")
fi
  if [[ -n "$CONTEXT7_NPX" ]]; then
    cmd+=("CONTEXT7_NPX=${CONTEXT7_NPX}")
  fi
  if [[ -n "$BASE_URL" ]]; then
    cmd+=("BASE_URL=${BASE_URL}")
  fi
  cmd+=("$RUNNER")

  exit_code=0
  status="success"
  if [[ "$DRY_RUN" == "1" ]]; then
    printf "DRY_RUN:"
    printf " %q" "${cmd[@]}"
    printf "\n"
    status="dry_run"
  elif "${cmd[@]}"; then
    status="success"
  else
    exit_code=$?
    status="failed"
    failures=$((failures + 1))
  fi

  ended_at="$(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$RUN_TAG" "$MODEL_ID" "$registry_mode" "${LOG_BASE_DIR}/${log_dir_name}" \
    "$status" "$exit_code" "$started_at" "$ended_at" "$skip_loop_detection" >> "$STATUS_FILE"

  if [[ "$status" == "failed" && "$STOP_ON_FAILURE" == "1" ]]; then
    break
  fi
done

if [[ "$failures" -gt 0 ]]; then
  echo "Completed with ${failures} failed run(s). See ${STATUS_FILE}."
  exit 1
fi

echo "Completed Qwen Code model pair. See ${STATUS_FILE}."
