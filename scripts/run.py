#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional


STEP_ORDER: List[str] = [
    "udatasets",
    "tokenize_cpt",
    "tokenize_sft",
    "train_cpt",
    "mg2hf",
    "hf2mg",
    "train_sft",
    "convert",
    "eval",
]
STEP_NUM: Dict[str, int] = {step: i + 1 for i, step in enumerate(STEP_ORDER)}


def now_run_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def tee_process(proc: subprocess.Popen, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            f.write(line)
        return proc.wait()


def clear_output_directory(output_dir: Path, step_name: str, dry_run: bool = False) -> None:
    """
    Clear output directory before running a step.
    
    Args:
        output_dir: Directory to clear
        step_name: Step name for logging
        dry_run: If True, only print what would be cleared
    """
    if not output_dir.exists():
        return
    
    if not output_dir.is_dir():
        # If it's a file, remove it
        if not dry_run:
            output_dir.unlink()
        return
    
    # Count files before clearing
    files_before = list(output_dir.rglob("*"))
    file_count = len([f for f in files_before if f.is_file()])
    
    if file_count == 0:
        return
    
    if dry_run:
        print(f"[dry-run] {step_name}: would clear {file_count} files from {output_dir}")
        return
    
    # Clear directory contents (but keep the directory itself)
    for item in output_dir.iterdir():
        if item.is_file():
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)
    
    print(f"[{time.strftime('%F %T')}] {step_name}: cleared {file_count} files from {output_dir}")


def get_step_output_dir(
    step: str,
    config_dir: Path,
    root_dir: Path,
    datapool_root: Path,
    pipeline_env: Optional[Dict[str, str]] = None,
) -> Optional[Path]:
    """
    Get the output directory for a step by loading its config.
    
    Returns:
        Output directory path, or None if cannot be determined
    """
    # Import config utilities using importlib to avoid name conflicts
    import importlib.util
    utils_dir = root_dir / "scripts" / "utils"
    config_utils_path = utils_dir / "config.py"
    
    if not config_utils_path.exists():
        return None
    
    spec = importlib.util.spec_from_file_location("config_utils", config_utils_path)
    if spec is None or spec.loader is None:
        return None
    
    config_utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_utils)
    
    load_config_module = config_utils.load_config_module
    resolve_config_vars = config_utils.resolve_config_vars
    
    # Load step config
    step_config_py = config_dir / "steps" / f"{STEP_NUM[step]}.{step}.py"
    step_config_unnumbered = config_dir / "steps" / f"{step}.py"
    
    if step_config_py.exists():
        step_config_path = step_config_py
    elif step_config_unnumbered.exists():
        step_config_path = step_config_unnumbered
    else:
        return None
    
    try:
        config = load_config_module(step_config_path)
        context = {
            "DATAPOOL_ROOT": str(datapool_root),
            "ROOT_DIR": str(root_dir),
        }
        if pipeline_env and "MODEL_PREFIX" in pipeline_env:
            context["MODEL_PREFIX"] = pipeline_env["MODEL_PREFIX"]
        config = resolve_config_vars(config, context)
        
        # Determine output directory based on step type
        if step == "tokenize_cpt":
            output_prefix = config.get("OUTPUT_PREFIX")
            if output_prefix:
                return Path(output_prefix).parent
        elif step == "tokenize_sft":
            output_prefix = config.get("OUTPUT_PREFIX") or config.get("SFT_OUTPUT_PREFIX")
            if output_prefix:
                return Path(output_prefix).parent
        elif step == "convert":
            output_dir = config.get("OUTPUT_DIR") or config.get("HF_OUTPUT_DIR")
            if output_dir:
                return Path(output_dir)
            # Default: datapool/model/hf
            return datapool_root / "model" / "hf"
        elif step == "eval":
            output_dir = config.get("OUTPUT_DIR") or config.get("REPORT_DIR")
            if output_dir:
                return Path(output_dir)
            # Default: datapool/reports
            return datapool_root / "reports"
        elif step == "udatasets":
            output_dir = config.get("OUTPUT_DIR")
            if output_dir:
                return Path(output_dir)
            # Default: datapool/data/processed
            return datapool_root / "data" / "processed"
    except Exception as e:
        # If config loading fails, return None (will skip clearing)
        return None
    
    return None


def run_step(
    *,
    root_dir: Path,
    config_dir: Path,
    step: str,
    pipeline_env: Dict[str, str],
    run_id: str,
    workdir: Path,
    log_dir: Path,
) -> None:
    enabled_key = f"STEP_{step.upper()}_ENABLED"
    enabled = pipeline_env.get(enabled_key, "0")
    dry_run = pipeline_env.get("DRY_RUN", "0")

    def log(msg: str) -> None:
        ts = time.strftime("%F %T")
        print(f"[{ts}] {msg}")

    if enabled != "1":
        log(f"skip step={step} ({enabled_key}={enabled})")
        return

    if step not in STEP_NUM:
        raise SystemExit(f"Unknown step: {step}")
    
    # All steps now use Python scripts only
    step_script = root_dir / "scripts" / "steps" / f"{STEP_NUM[step]}.{step}.py"
    
    if not step_script.exists():
        raise SystemExit(f"Step script not found: {step_script}")

    # Default: numbered step config (.py); fallback to unnumbered
    # All steps now use Python config files (.py) only
    step_config_py = config_dir / "steps" / f"{STEP_NUM[step]}.{step}.py"
    step_config_unnumbered = config_dir / "steps" / f"{step}.py"
    
    if step_config_py.exists():
        step_config_path = step_config_py
    elif step_config_unnumbered.exists():
        step_config_path = step_config_unnumbered
    else:
        step_config_path = step_config_py  # Will be passed but may not exist
    
    # Python scripts load config themselves
    step_env = {}  # Python scripts load config themselves

    env = os.environ.copy()
    datapool_root = pipeline_env.get("DATAPOOL_ROOT", str(root_dir / "datapool"))
    # Basic safety check: require datapool to be inside this repo by default.
    # (User can override, but the default should keep data co-located.)
    try:
        repo_real = root_dir.resolve()
        dp_real = Path(datapool_root).expanduser().resolve()
        if repo_real not in dp_real.parents and dp_real != repo_real:
            print(f"[warn] DATAPOOL_ROOT is outside repo: {dp_real}", file=sys.stderr)
    except Exception:
        pass

    env.update(
        {
            "ROOT_DIR": str(root_dir),
            "CONFIG_DIR": str(config_dir),
            "STEP_ENV_PATH": str(step_config_path),
            "RUN_ID": run_id,
            "WORKDIR": str(workdir),
            "LOG_DIR": str(log_dir),
            "DRY_RUN": dry_run,
            "DATAPOOL_ROOT": datapool_root,
        }
    )
    # Pass pipeline config variables to steps (for BASE_MODEL_PATH, etc.)
    for key in ["BASE_MODEL_NAME", "BASE_MODEL_SRC", "BASE_MODEL_PATH", "MODEL_PREFIX", "MEGATRON", "MINDSPEED", "ROOT"]:
        if key in pipeline_env:
            env[key] = pipeline_env[key]
    # Python scripts load config themselves, no need to update env

    # Clear output directory before running the step
    output_dir = get_step_output_dir(step, config_dir, root_dir, Path(datapool_root), pipeline_env)
    if output_dir:
        clear_output_directory(output_dir, step, dry_run=(dry_run == "1"))

    log(f"run step={step}")
    if dry_run == "1":
        print(f"[dry-run] would run: {step_script}")
        # still invoke the script in dry-run mode so it can print the planned command
    
    cmd = ["python3", str(step_script)]
    
    proc = subprocess.Popen(
        cmd,
        cwd=str(root_dir),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    code = tee_process(proc, log_dir / f"{step}.log")
    if code != 0:
        raise SystemExit(f"step failed: {step} (exit={code}), see log: {log_dir / (step + '.log')}")
    log(f"done step={step}")


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python scripts/run.py")
    ap.add_argument("-c", "--config", required=True, help="Path to pipeline.py or pipeline.env")
    ap.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only create datapool/workdir structure, do not execute any steps",
    )
    args = ap.parse_args(argv)

    root_dir = Path(__file__).resolve().parent.parent
    if not os.environ.get("DATAPOOL"):
        os.environ["DATAPOOL"] = str(root_dir / "datapool")
    pipeline_config_path = Path(args.config).expanduser().resolve()
    if not pipeline_config_path.exists():
        raise SystemExit(f"Config not found: {pipeline_config_path}")

    # Convention: a config root is the directory containing pipeline.env or pipeline.py.
    # It must contain steps/<step>.py alongside pipeline config.
    config_dir = pipeline_config_path.parent

    # Load pipeline config (.py only)
    if pipeline_config_path.suffix != ".py":
        raise SystemExit("Only .py pipeline configs are supported")
    sys.path.insert(0, str(root_dir / "scripts" / "utils"))
    from config import apply_env_imports, load_config_module, merge_env_defaults, resolve_config_vars
    pipeline_config = load_config_module(pipeline_config_path)
    merge_env_defaults(pipeline_config, os.environ)
    # First resolve DATAPOOL_ROOT to get the actual path
    temp_context = {}
    apply_env_imports(temp_context, os.environ)
    temp_resolved = resolve_config_vars(pipeline_config, temp_context)
    datapool_root_temp = Path(temp_resolved.get("DATAPOOL_ROOT", str(root_dir / "datapool"))).expanduser().resolve()
    # Now resolve all variables with full context
    pipeline_context = {
        "DATAPOOL_ROOT": str(datapool_root_temp),
        "ROOT_DIR": str(root_dir),
    }
    apply_env_imports(pipeline_context, os.environ)
    pipeline_resolved = resolve_config_vars(pipeline_config, pipeline_context)
    pipeline_env = {k: str(v) for k, v in pipeline_resolved.items()}

    # Prepare experiment (datapool structure, base model, raw data)
    import prepare_exp
    prepare_exp.prepare_from_env(
        pipeline_env=pipeline_env,
        config_dir=config_dir,
        root_dir=root_dir,
        mode="copy",
    )

    # 产出路径按实验唯一，不再用 RUN_ID 分层；run_id 仅用于日志目录（默认 "run"）
    run_id = pipeline_env.get("RUN_ID", "").strip() or "run"
    workdir = Path(pipeline_env.get("WORKDIR") or (root_dir / ".llmrunner")).expanduser().resolve()
    log_dir = (workdir / "logs" / run_id).resolve()
    datapool_root = Path(pipeline_env.get("DATAPOOL_ROOT") or (root_dir / "datapool")).expanduser().resolve()

    workdir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[{time.strftime('%F %T')}] run_id={run_id} config_dir={config_dir} workdir={workdir} datapool_root={datapool_root} dry_run={pipeline_env.get('DRY_RUN','0')}"
    )

    if args.prepare_only:
        print(f"[{time.strftime('%F %T')}] prepare-only: done (no steps executed)")
        return 0

    for step in STEP_ORDER:
        run_step(
            root_dir=root_dir,
            config_dir=config_dir,
            step=step,
            pipeline_env=pipeline_env,
            run_id=run_id,
            workdir=workdir,
            log_dir=log_dir,
        )

    print(f"[{time.strftime('%F %T')}] pipeline finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

