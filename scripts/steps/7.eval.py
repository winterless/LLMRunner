#!/usr/bin/env python3
"""
Step 6: Model evaluation
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))
from config import load_config_module, resolve_config_vars, require_config


def main() -> int:
    # Get environment variables
    root_dir = Path(os.environ["ROOT_DIR"])
    step_env_path = Path(os.environ.get("STEP_ENV_PATH", ""))
    dry_run = os.environ.get("DRY_RUN", "0") == "1"
    
    # Load config - run.py already found the config file and passed it via STEP_ENV_PATH
    if not step_env_path or not step_env_path.exists():
        print(f"Missing config: STEP_ENV_PATH not set or file not found: {step_env_path}", file=sys.stderr)
        return 2
    
    # If it's a .env file, error - user should migrate to .py
    if step_env_path.suffix == ".env":
        print(f"eval: .env files are deprecated, please migrate to .py config: {step_env_path}", file=sys.stderr)
        return 2
    
    # Load and resolve config
    config = load_config_module(step_env_path)
    context = {
        "ROOT_DIR": str(root_dir),
    }
    config = resolve_config_vars(config, context)
    
    # Extract config
    eval_suite = config.get("EVAL_SUITE", "bfcl_v3")
    eval_cmd = config.get("EVAL_CMD")
    
    print(f"eval: suite={eval_suite}")
    
    if dry_run:
        print(f"[dry-run] eval: {eval_cmd if eval_cmd else '<no cmd configured>'}")
        return 0
    
    if not eval_cmd:
        print(f"eval: missing EVAL_CMD in config", file=sys.stderr)
        return 2
    
    # Execute command
    try:
        subprocess.run(eval_cmd, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"eval: failed with exit code {e.returncode}", file=sys.stderr)
        return e.returncode
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
