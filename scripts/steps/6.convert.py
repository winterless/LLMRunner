#!/usr/bin/env python3
"""
Step 5: Model conversion
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))
from config import load_config_module, resolve_config_vars, require_config, require_path_exists


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
        print(f"convert: .env files are deprecated, please migrate to .py config: {step_env_path}", file=sys.stderr)
        return 2
    
    # Load and resolve config
    config = load_config_module(step_env_path)
    context = {
        "ROOT_DIR": str(root_dir),
    }
    config = resolve_config_vars(config, context)
    
    # Extract required config
    run_with = config.get("RUN_WITH")
    if run_with not in ("cmd", "entrypoint"):
        print("convert: set RUN_WITH=cmd (and CONVERT_CMD) or RUN_WITH=entrypoint (and ENTRYPOINT, ARGS) in step config", file=sys.stderr)
        return 2
    
    # TRAINER_DIR or MINDSPEED_DIR
    trainer_dir_str = config.get("TRAINER_DIR") or config.get("MINDSPEED_DIR")
    if not trainer_dir_str:
        print("convert: set TRAINER_DIR or MINDSPEED_DIR in step config", file=sys.stderr)
        return 2
    
    trainer_dir = require_path_exists(trainer_dir_str, root_dir, "convert")
    
    if run_with == "cmd":
        convert_cmd = config.get("CONVERT_CMD")
        if not convert_cmd:
            if dry_run:
                print("convert: CONVERT_CMD not set (dry-run only, skip)")
                return 0
            else:
                print("convert: RUN_WITH=cmd requires CONVERT_CMD in step config", file=sys.stderr)
                return 2
    else:  # entrypoint
        entrypoint = require_config(config, "ENTRYPOINT", "convert")
        args = config.get("ARGS", "")
    
    print(f"convert: trainer_dir={trainer_dir} RUN_WITH={run_with}")
    
    if dry_run:
        if run_with == "cmd":
            if convert_cmd:
                print(f"[dry-run] (cd {trainer_dir} && {convert_cmd})")
        else:
            print(f"[dry-run] (cd {trainer_dir} && python {entrypoint} {args})")
        return 0
    
    # Execute
    try:
        if run_with == "cmd":
            subprocess.run(convert_cmd, shell=True, cwd=trainer_dir, check=True)
        else:
            cmd = ["python", entrypoint]
            if args:
                cmd.extend(args.split())
            subprocess.run(cmd, cwd=trainer_dir, check=True)
    except subprocess.CalledProcessError as e:
        print(f"convert: failed with exit code {e.returncode}", file=sys.stderr)
        return e.returncode
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
