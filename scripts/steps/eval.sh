#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:?}"
DRY_RUN="${DRY_RUN:?}"

CFG="${ROOT_DIR}/configs/steps/eval.env"
if [[ ! -f "${CFG}" ]]; then
  echo "Missing config: ${CFG}" >&2
  exit 2
fi
# shellcheck disable=SC1090
source "${CFG}"

echo "eval: suite=${EVAL_SUITE:-bfcl_v3}"

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "[dry-run] eval: ${EVAL_CMD:-<no cmd configured>}"
  exit 0
fi

if [[ -z "${EVAL_CMD:-}" ]]; then
  echo "eval: missing EVAL_CMD in configs/steps/eval.env" >&2
  exit 2
fi

bash -lc "${EVAL_CMD}"

