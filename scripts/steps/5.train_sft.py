#!/usr/bin/env python3
"""
Step 4: SFT training
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
    config_dir = Path(os.environ.get("CONFIG_DIR", root_dir / "configs"))
    step_env_path = Path(os.environ.get("STEP_ENV_PATH", ""))
    datapool_root = Path(os.environ["DATAPOOL_ROOT"])
    dry_run = os.environ.get("DRY_RUN", "0") == "1"
    
    # Load config - run.py already found the config file and passed it via STEP_ENV_PATH
    if not step_env_path or not step_env_path.exists():
        print(f"Missing config: STEP_ENV_PATH not set or file not found: {step_env_path}", file=sys.stderr)
        return 2
    
    # If it's a .env file, error - user should migrate to .py
    if step_env_path.suffix == ".env":
        print(f"train_sft: .env files are deprecated, please migrate to .py config: {step_env_path}", file=sys.stderr)
        return 2
    
    # Load and resolve config
    config = load_config_module(step_env_path)
    context = {
        "DATAPOOL_ROOT": str(datapool_root),
        "ROOT_DIR": str(root_dir),
    }
    # Add pipeline config variables (BASE_MODEL_NAME, BASE_MODEL_SRC, BASE_MODEL_PATH, MODEL_PREFIX)
    for key in ["BASE_MODEL_NAME", "BASE_MODEL_SRC", "BASE_MODEL_PATH", "MODEL_PREFIX"]:
        if key in os.environ:
            context[key] = os.environ[key]
    config = resolve_config_vars(config, context)
    
    # Get SFT_RAW_COPY_SRC from tokenize_sft config if needed
    sft_raw_copy_src = config.get("SFT_RAW_COPY_SRC")
    if not sft_raw_copy_src:
        # Try to load from tokenize_sft config
        tokenize_sft_config = config_dir / "steps" / "3.tokenize_sft.py"
        if not tokenize_sft_config.exists():
            tokenize_sft_config = config_dir / "steps" / "2.tokenize_sft.py"
        if tokenize_sft_config.exists():
            tokenize_config = load_config_module(tokenize_sft_config)
            tokenize_config = resolve_config_vars(tokenize_config, context)
            sft_raw_copy_src = tokenize_config.get("SFT_RAW_COPY_SRC")
    
    # If optional copy source set, copy *.jsonl into datapool raw/sft once
    if sft_raw_copy_src:
        print(f"train_sft: copying raw SFT from {sft_raw_copy_src} -> {datapool_root}/data/raw/sft")
        if not dry_run:
            # Import prepare_exp utilities
            sys.path.insert(0, str(root_dir / "scripts"))
            from prepare_exp import copy_jsonl_flat
            copy_jsonl_flat(Path(sft_raw_copy_src), datapool_root / "data" / "raw" / "sft")
    
    # Extract required config
    run_with = config.get("RUN_WITH")
    if run_with not in ("cmd", "entrypoint"):
        print("train_sft: set RUN_WITH=cmd (and TRAIN_CMD) or RUN_WITH=entrypoint (and ENTRYPOINT, ARGS) in step config", file=sys.stderr)
        return 2
    
    # TRAINER_DIR or MINDSPEED_DIR
    trainer_dir_str = config.get("TRAINER_DIR") or config.get("MINDSPEED_DIR")
    if not trainer_dir_str:
        print("train_sft: set TRAINER_DIR or MINDSPEED_DIR in step config", file=sys.stderr)
        return 2
    
    trainer_dir = require_path_exists(trainer_dir_str, root_dir, "train_sft")
    
    if run_with == "cmd":
        train_cmd = config.get("TRAIN_CMD")
        if not train_cmd:
            if dry_run:
                print("train_sft: TRAIN_CMD not set (dry-run only, skip)")
                return 0
            else:
                print("train_sft: RUN_WITH=cmd requires TRAIN_CMD in step config", file=sys.stderr)
                return 2
    else:  # entrypoint
        entrypoint = require_config(config, "ENTRYPOINT", "train_sft")
        args = config.get("ARGS", "")
    
    print(f"train_sft: trainer_dir={trainer_dir} RUN_WITH={run_with}")
    
    if dry_run:
        if run_with == "cmd":
            if train_cmd:
                print(f"[dry-run] (cd {trainer_dir} && {train_cmd})")
        else:
            print(f"[dry-run] (cd {trainer_dir} && python {entrypoint} {args})")
        return 0
    
    # Execute
    try:
        if run_with == "cmd":
            subprocess.run(train_cmd, shell=True, cwd=trainer_dir, check=True)
        else:
            cmd = ["python", entrypoint]
            if args:
                cmd.extend(args.split())
            subprocess.run(cmd, cwd=trainer_dir, check=True)
    except subprocess.CalledProcessError as e:
        print(f"train_sft: failed with exit code {e.returncode}", file=sys.stderr)
        return e.returncode
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
