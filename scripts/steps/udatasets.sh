#!/usr/bin/env bash
set -euo pipefail

# Optional step1: UDatasets
# This runner only *calls a shell command* you configure.
# If you already prepared jsonl on OBS manually, disable this step.

ROOT_DIR="${ROOT_DIR:?}"
WORKDIR="${WORKDIR:?}"
DRY_RUN="${DRY_RUN:?}"

CFG="${ROOT_DIR}/configs/steps/udatasets.env"
if [[ -f "${CFG}" ]]; then
  # shellcheck disable=SC1090
  source "${CFG}"
fi

echo "udatasets: enabled=${STEP_UDATASETS_ENABLED:-0}"
echo "udatasets: output_uri=${UDATASETS_OUTPUT_URI:-}"

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "[dry-run] udatasets: ${UDATASETS_CMD:-<no cmd configured>}"
  exit 0
fi

if [[ -n "${UDATASETS_CMD:-}" ]]; then
  bash -lc "${UDATASETS_CMD}"
else
  echo "udatasets: no UDATASETS_CMD configured; nothing to do"
fi

