#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:?}"
DRY_RUN="${DRY_RUN:?}"

CFG="${ROOT_DIR}/configs/steps/train_cpt.env"
if [[ ! -f "${CFG}" ]]; then
  echo "Missing config: ${CFG}" >&2
  exit 2
fi
# shellcheck disable=SC1090
source "${CFG}"

MINDSPEED_DIR="${MINDSPEED_DIR:?}"
ENTRYPOINT="${ENTRYPOINT:-pretrain_gpt.py}"

echo "train_cpt: mindspeed_dir=${MINDSPEED_DIR}"

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "[dry-run] (cd ${MINDSPEED_DIR} && python ${ENTRYPOINT} ${ARGS:-})"
  exit 0
fi

cd "${MINDSPEED_DIR}"
if [[ -n "${ARGS:-}" ]]; then
  bash -lc "python ${ENTRYPOINT} ${ARGS}"
else
  python "${ENTRYPOINT}"
fi

