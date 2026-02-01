#!/usr/bin/env python3
"""
Step 6: HF -> MG conversion (size adapter)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))
from config import load_config_module, resolve_config_vars
from step_utils import apply_pipeline_context, run_extern_script


def main() -> int:
    # Get environment variables
    root_dir = Path(os.environ["ROOT_DIR"])
    step_env_path = Path(os.environ.get("STEP_ENV_PATH", ""))
    datapool_root = Path(os.environ.get("DATAPOOL_ROOT", str(root_dir / "datapool")))
    dry_run = os.environ.get("DRY_RUN", "0") == "1"

    # Load config - run.py already found the config file and passed it via STEP_ENV_PATH
    if not step_env_path or not step_env_path.exists():
        print(f"Missing config: STEP_ENV_PATH not set or file not found: {step_env_path}", file=sys.stderr)
        return 2

    # If it's a .env file, error - user should migrate to .py
    if step_env_path.suffix == ".env":
        print(f"hf2mg: .env files are deprecated, please migrate to .py config: {step_env_path}", file=sys.stderr)
        return 2

    # Load and resolve config
    config = load_config_module(step_env_path)
    context = {
        "ROOT_DIR": str(root_dir),
        "DATAPOOL_ROOT": str(datapool_root),
    }
    # Add pipeline config variables (BASE_MODEL_NAME, BASE_MODEL_SRC, BASE_MODEL_PATH, MODEL_PREFIX, MEGATRON, MINDSPEED)
    apply_pipeline_context(context, os.environ)
    config = resolve_config_vars(config, context)

    # Extern script shortcut (run entire conversion outside this step)
    extern_result = run_extern_script(
        config,
        root_dir=root_dir,
        dry_run=dry_run,
        step_name="hf2mg",
    )
    if extern_result is not None:
        return extern_result

    print("hf2mg: set EXTERN_SCRIPT in step config", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
