#!/usr/bin/env python3
"""
MG → HF: atomic (EXTERN_SCRIPT) or full export (CONVERT_CMD + COPY_HF_*).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))
from config import load_config_module, resolve_config_vars, require_config, require_path_exists
from step_utils import apply_pipeline_context, run_extern_script


def main() -> int:
    root_dir = Path(os.environ["ROOT_DIR"])
    datapool_root = Path(os.environ.get("DATAPOOL_ROOT", "datapool"))
    step_env_path = Path(os.environ.get("STEP_ENV_PATH", ""))
    dry_run = os.environ.get("DRY_RUN", "0") == "1"

    if not step_env_path or not step_env_path.exists():
        print(f"Missing config: STEP_ENV_PATH not set or file not found: {step_env_path}", file=sys.stderr)
        return 2
    if step_env_path.suffix == ".env":
        print(f"mg2hf: .env files are deprecated, please migrate to .py config: {step_env_path}", file=sys.stderr)
        return 2

    config = load_config_module(step_env_path)
    if not datapool_root.is_absolute():
        datapool_root = (root_dir / datapool_root).resolve()
    else:
        datapool_root = Path(datapool_root).resolve()
    context = {"ROOT_DIR": str(root_dir), "DATAPOOL_ROOT": str(datapool_root)}
    apply_pipeline_context(context, os.environ)
    config = resolve_config_vars(config, context)

    extern_result = run_extern_script(config, root_dir=root_dir, dry_run=dry_run, step_name="mg2hf")
    if extern_result is not None:
        return extern_result

    # Full export path: RUN_WITH + CONVERT_CMD / entrypoint + COPY_HF_*
    run_with = config.get("RUN_WITH")
    if run_with not in ("cmd", "entrypoint"):
        print("mg2hf: set RUN_WITH=cmd (and CONVERT_CMD) or RUN_WITH=entrypoint (and ENTRYPOINT, ARGS) in step config", file=sys.stderr)
        return 2

    trainer_dir_str = config.get("MEGATRON") or config.get("MINDSPEED")
    if not trainer_dir_str:
        print("mg2hf: set MEGATRON or MINDSPEED in step config", file=sys.stderr)
        return 2
    trainer_dir = require_path_exists(trainer_dir_str, root_dir, "mg2hf")

    if run_with == "cmd":
        convert_cmd = config.get("CONVERT_CMD")
        if not convert_cmd:
            if dry_run:
                print("mg2hf: CONVERT_CMD not set (dry-run only, skip)")
                return 0
            print("mg2hf: RUN_WITH=cmd requires CONVERT_CMD in step config", file=sys.stderr)
            return 2
    else:
        entrypoint = require_config(config, "ENTRYPOINT", "mg2hf")
        args = config.get("ARGS", "")

    print(f"mg2hf: trainer_dir={trainer_dir} RUN_WITH={run_with}")

    if dry_run:
        if run_with == "cmd" and convert_cmd:
            print(f"[dry-run] (cd {trainer_dir} && {convert_cmd})")
        elif run_with == "entrypoint":
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
        if isinstance(copy_files, str):
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
            print(f"mg2hf: copied {copied} hf files from {copy_from} -> {out_hf_dir} ({when})")

    if str(config.get("COPY_HF_BEFORE", "0")) == "1":
        copy_hf_files("before")

    try:
        if run_with == "cmd":
            subprocess.run(convert_cmd, shell=True, cwd=trainer_dir, check=True)
        else:
            cmd = ["python", entrypoint]
            if args:
                cmd.extend(args.split())
            subprocess.run(cmd, cwd=trainer_dir, check=True)
    except subprocess.CalledProcessError as e:
        print(f"mg2hf: failed with exit code {e.returncode}", file=sys.stderr)
        return e.returncode

    copy_hf_files("after")
    return 0


if __name__ == "__main__":
    sys.exit(main())
