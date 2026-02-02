#!/usr/bin/env python3
"""
Step 8: Model conversion
"""
from __future__ import annotations

import os
import shutil
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
    datapool_root = Path(os.environ.get("DATAPOOL_ROOT", "datapool"))
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
    if not datapool_root.is_absolute():
        datapool_root = (root_dir / datapool_root).resolve()
    else:
        datapool_root = datapool_root.resolve()

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
        step_name="convert",
    )
    if extern_result is not None:
        return extern_result

    # Extract required config
    run_with = config.get("RUN_WITH")
    if run_with not in ("cmd", "entrypoint"):
        print("convert: set RUN_WITH=cmd (and CONVERT_CMD) or RUN_WITH=entrypoint (and ENTRYPOINT, ARGS) in step config", file=sys.stderr)
        return 2

    # MEGATRON or MINDSPEED
    trainer_dir_str = config.get("MEGATRON") or config.get("MINDSPEED")
    if not trainer_dir_str:
        print("convert: set MEGATRON or MINDSPEED in step config", file=sys.stderr)
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

    def copy_hf_files(when: str) -> None:
        out_hf_dir = config.get("OUT_HF_DIR")
        copy_from = config.get("COPY_HF_FROM")
        copy_files = config.get("COPY_HF_FILES")
        copy_overwrite = str(config.get("COPY_HF_OVERWRITE", "0")) == "1"
        subdir_key = "COPY_HF_BEFORE_SUBDIR" if when == "before" else "COPY_HF_SUBDIR"
        subdir = config.get(subdir_key, "")
        if not out_hf_dir or not copy_from or not copy_files:
            return
        out_hf_dir = Path(out_hf_dir)
        if subdir:
            out_hf_dir = out_hf_dir / subdir
        copy_from = Path(copy_from)
        if copy_files and isinstance(copy_files, str):
            copy_files = [x.strip() for x in copy_files.split(",") if x.strip()]
        if copy_from.exists():
            out_hf_dir.mkdir(parents=True, exist_ok=True)
            copied = 0
            for name in copy_files:
                src = copy_from / name
                dst = out_hf_dir / name
                if not src.exists():
                    continue
                if dst.exists() and not copy_overwrite:
                    continue
                shutil.copy2(src, dst)
                copied += 1
            print(f"convert: copied {copied} hf files from {copy_from} -> {out_hf_dir} ({when})")

    # Optional: copy tokenizer/config files into HF output dir before conversion
    if str(config.get("COPY_HF_BEFORE", "0")) == "1":
        copy_hf_files("before")

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

    # Optional: copy tokenizer/config files into HF output dir after conversion
    copy_hf_files("after")

    return 0


if __name__ == "__main__":
    sys.exit(main())
