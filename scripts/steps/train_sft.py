#!/usr/bin/env python3
"""
Step 7: SFT training
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))
from config import load_config_module, resolve_config_vars, require_config, require_path_exists
from step_utils import apply_pipeline_context, run_extern_script


def main() -> int:
    # Get environment variables
    root_dir = Path(os.environ["ROOT_DIR"])
    step_env_path = Path(os.environ.get("STEP_ENV_PATH", ""))
    datapool_root = Path(os.environ["DATAPOOL_ROOT"])
    dry_run = os.environ.get("DRY_RUN", "0") == "1"

    # Load config - run.py already found the config file and passed it via STEP_ENV_PATH
    if not step_env_path or not step_env_path.exists():
        print(f"Missing config: STEP_ENV_PATH not set or file not found: {step_env_path}", file=sys.stderr)
        return 2

    # Load and resolve config
    config = load_config_module(step_env_path)
    context = {
        "DATAPOOL_ROOT": str(datapool_root),
        "ROOT_DIR": str(root_dir),
    }
    # Add pipeline config variables (BASE_MODEL_NAME, BASE_MODEL_SRC, BASE_MODEL_PATH, MODEL_PREFIX, MEGATRON, MINDSPEED)
    apply_pipeline_context(context, os.environ)
    config = resolve_config_vars(config, context)

    # Export resolved config values for child scripts (e.g., bash wrappers)
    env = os.environ.copy()
    for key, value in config.items():
        if isinstance(key, str):
            env[key] = str(value)

    # Extern script shortcut (run entire training outside this step)
    extern_result = run_extern_script(
        config,
        root_dir=root_dir,
        dry_run=dry_run,
        step_name="train_sft",
    )
    if extern_result is not None:
        return extern_result

    # Extract required config
    run_with = config.get("RUN_WITH")
    if run_with not in ("cmd", "entrypoint"):
        print("train_sft: set RUN_WITH=cmd (and TRAIN_CMD) or RUN_WITH=entrypoint (and ENTRYPOINT, ARGS) in step config", file=sys.stderr)
        return 2

    # MEGATRON or MINDSPEED
    trainer_dir_str = config.get("MEGATRON") or config.get("MINDSPEED")
    if not trainer_dir_str:
        print("train_sft: set MEGATRON or MINDSPEED in step config", file=sys.stderr)
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
        # Replace variables in TRAIN_CMD using all config values
        for key, value in config.items():
            if isinstance(value, str):
                train_cmd = train_cmd.replace(f"${{{key}}}", value)
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

    # Execute with real-time output
    try:
        if run_with == "cmd":
            # Use Popen to get real-time output (unbuffered)
            proc = subprocess.Popen(
                train_cmd,
                shell=True,
                cwd=trainer_dir,
                env=env,
                stdout=sys.stdout,
                stderr=sys.stderr,
                bufsize=0,  # Unbuffered
            )
            return_code = proc.wait()
            if return_code != 0:
                raise subprocess.CalledProcessError(return_code, train_cmd)
        else:
            cmd = ["python", "-u", entrypoint]  # -u for unbuffered output
            if args:
                cmd.extend(args.split())
            proc = subprocess.Popen(
                cmd,
                cwd=trainer_dir,
                env=env,
                stdout=sys.stdout,
                stderr=sys.stderr,
                bufsize=0,  # Unbuffered
            )
            return_code = proc.wait()
            if return_code != 0:
                raise subprocess.CalledProcessError(return_code, cmd)
    except subprocess.CalledProcessError as e:
        print(f"train_sft: failed with exit code {e.returncode}", file=sys.stderr)
        return e.returncode

    return 0


if __name__ == "__main__":
    sys.exit(main())
