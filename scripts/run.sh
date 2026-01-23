#!/usr/bin/env bash
set -euo pipefail

# Wrapper: pipeline is now managed by Python (stdlib only).
#
# Usage:
#   bash scripts/run.sh -c configs/pipeline.env
exec python3 "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run.py" "$@"

