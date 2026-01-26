#!/usr/bin/env python3
"""
Step 1: UDatasets - Optional step to prepare JSONL data
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
        # UDatasets is optional, so config might not exist
        print("udatasets: enabled=0 (no config found)")
        return 0
    
    # If it's a .env file, error - user should migrate to .py
    if step_env_path.suffix == ".env":
        print(f"udatasets: .env files are deprecated, please migrate to .py config: {step_env_path}", file=sys.stderr)
        return 2
    
    # Load and resolve config
    config = load_config_module(step_env_path)
    context = {
        "ROOT_DIR": str(root_dir),
    }
    config = resolve_config_vars(config, context)
    
    # Optional config
    udatasets_cmd = config.get("UDATASETS_CMD")
    output_uri = config.get("UDATASETS_OUTPUT_URI", "")
    
    print(f"udatasets: enabled=1")
    print(f"udatasets: output_uri={output_uri}")
    
    if dry_run:
        print(f"[dry-run] udatasets: {udatasets_cmd if udatasets_cmd else '<no cmd configured>'}")
        return 0
    
    if not udatasets_cmd:
        print("udatasets: no UDATASETS_CMD configured; nothing to do")
        return 0
    
    # Execute command
    try:
        subprocess.run(udatasets_cmd, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"udatasets: failed with exit code {e.returncode}", file=sys.stderr)
        return e.returncode
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
