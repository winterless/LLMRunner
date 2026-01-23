#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:?}"
DRY_RUN="${DRY_RUN:?}"
DATAPOOL_ROOT="${DATAPOOL_ROOT:?}"
RUN_DIR="${RUN_DIR:?}"
ALLOW_EXTERNAL_PATHS="${ALLOW_EXTERNAL_PATHS:-0}"

CFG="${ROOT_DIR}/configs/steps/tokenize.env"
source "${CFG}"

set -euo pipefail

INPUT_GLOB="${INPUT_GLOB:?}"
TOKENIZER_MODEL="${TOKENIZER_MODEL:?}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:?}"
MEGATRON_DIR="${MEGATRON_DIR:?}"

WORKERS="${WORKERS:-16}"
PARTITIONS="${PARTITIONS:-16}"
LOG_INTERVAL="${LOG_INTERVAL:-100000}"
JSON_KEYS="${JSON_KEYS:-text}"
TOKENIZER_TYPE="${TOKENIZER_TYPE:-HuggingFaceTokenizer}"

cmd=(python tools/preprocess_data.py
  --input "${INPUT_GLOB}"
  --output-prefix "${OUTPUT_PREFIX}"
  --json-keys "${JSON_KEYS}"
  --tokenizer-type "${TOKENIZER_TYPE}"
  --tokenizer-model "${TOKENIZER_MODEL}"
  --append-eod
  --workers "${WORKERS}"
  --partitions "${PARTITIONS}"
  --log-interval "${LOG_INTERVAL}"
)

echo "tokenize: megatron_dir=${MEGATRON_DIR}"
echo "tokenize: output_prefix=${OUTPUT_PREFIX}"

# Enforce datapool-only I/O by default.
if [[ "${ALLOW_EXTERNAL_PATHS}" != "1" ]]; then
  # best-effort: strip glob tail for prefix check
  input_prefix="${INPUT_GLOB%%[*?]*}"
  if [[ "${input_prefix}" != "${DATAPOOL_ROOT}/"* ]]; then
    echo "tokenize: INPUT_GLOB must be under DATAPOOL_ROOT (${DATAPOOL_ROOT}) but got: ${INPUT_GLOB}" >&2
    echo "tokenize: set ALLOW_EXTERNAL_PATHS=1 in configs/pipeline.env to override (not recommended)" >&2
    exit 2
  fi
  if [[ "${OUTPUT_PREFIX}" != "${DATAPOOL_ROOT}/"* && "${OUTPUT_PREFIX}" != "${RUN_DIR}/"* ]]; then
    echo "tokenize: OUTPUT_PREFIX must be under DATAPOOL_ROOT (${DATAPOOL_ROOT}) but got: ${OUTPUT_PREFIX}" >&2
    echo "tokenize: set ALLOW_EXTERNAL_PATHS=1 in configs/pipeline.env to override (not recommended)" >&2
    exit 2
  fi
fi

mkdir -p "$(dirname "${OUTPUT_PREFIX}")"

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "[dry-run] (cd ${MEGATRON_DIR} && ${cmd[*]})"
  exit 0
fi

cd "${MEGATRON_DIR}"
"${cmd[@]}"

