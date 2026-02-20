#!/usr/bin/env python3
"""
HF → MG conversion (atomic). Runs EXTERN_SCRIPT only.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))
from config import load_config_module, resolve_config_vars
from step_utils import apply_pipeline_context, run_extern_script


def main() -> int:
    root_dir = Path(os.environ["ROOT_DIR"])
    step_env_path = Path(os.environ.get("STEP_ENV_PATH", ""))
    datapool_root = Path(os.environ.get("DATAPOOL_ROOT", str(root_dir / "datapool")))
    dry_run = os.environ.get("DRY_RUN", "0") == "1"

    if not step_env_path or not step_env_path.exists():
        print(f"Missing config: STEP_ENV_PATH not set or file not found: {step_env_path}", file=sys.stderr)
        return 2
    config = load_config_module(step_env_path)
    context = {"ROOT_DIR": str(root_dir), "DATAPOOL_ROOT": str(datapool_root)}
    apply_pipeline_context(context, os.environ)
    config = resolve_config_vars(config, context)

    extern_result = run_extern_script(config, root_dir=root_dir, dry_run=dry_run, step_name="hf2mg")
    if extern_result is not None:
        return extern_result

    print("hf2mg: set EXTERN_SCRIPT in step config", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
