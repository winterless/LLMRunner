#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Allow "from utils.step_registry import ..." when run from scripts/
_scripts_dir = Path(__file__).resolve().parent
if str(_scripts_dir) not in sys.path:
    sys.path.insert(0, str(_scripts_dir))
from utils.step_registry import Step, get_step, all_step_names


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


def _load_step_config(
    step_config_path: Path,
    root_dir: Path,
    datapool_root: Path,
    pipeline_env: Dict[str, str],
) -> Dict[str, Any]:
    """Load and resolve step config for output_dir / context."""
    from config import load_config_module, merge_env_defaults, resolve_config_vars
    config = load_config_module(step_config_path)
    merge_env_defaults(config, os.environ)
    context = {
        "DATAPOOL_ROOT": str(datapool_root),
        "ROOT_DIR": str(root_dir),
    }
    for key in ["MODEL_PREFIX", "BASE_MODEL_PATH", "MEGATRON", "MINDSPEED"]:
        if key in pipeline_env:
            context[key] = pipeline_env[key]
    return resolve_config_vars(config, context)


def get_step_output_dir(
    step_obj: Step,
    config_dir: Path,
    root_dir: Path,
    datapool_root: Path,
    pipeline_env: Dict[str, str],
    occurrence_index: int = 0,
) -> Optional[Path]:
    """Get the output directory for a step by loading its config. Returns None if not clearable."""
    step_config_path = step_obj.resolve_config_path(config_dir, occurrence_index)
    if not step_config_path.exists():
        return None
    try:
        config = _load_step_config(step_config_path, root_dir, datapool_root, pipeline_env)
        return step_obj.get_output_dir(config, datapool_root)
    except Exception:
        return None


def _resolve_steps(pipeline_config: Dict[str, Any]) -> List[str]:
    """
    Resolve list of step names to run.
    - If pipeline has STEPS (list or tuple), use it (order and repeats allowed).
    - Else fall back to legacy: all_step_names() in order, filter by STEP_*_ENABLED=1.
    """
    steps_raw = pipeline_config.get("STEPS")
    if steps_raw is not None and isinstance(steps_raw, (list, tuple)):
        return [str(s) for s in steps_raw]
    # Legacy: use default order, include only enabled
    result = []
    for name in all_step_names():
        enabled_key = f"STEP_{name.upper().replace('-', '_')}_ENABLED"
        if str(pipeline_config.get(enabled_key, "0")) == "1":
            result.append(name)
    return result


def run_step(
    *,
    root_dir: Path,
    config_dir: Path,
    step_obj: Step,
    step_index: int,
    occurrence_index: int,
    pipeline_env: Dict[str, str],
    run_id: str,
    workdir: Path,
    log_dir: Path,
) -> None:
    dry_run = pipeline_env.get("DRY_RUN", "0")
    step_name = step_obj.name

    def log(msg: str) -> None:
        ts = time.strftime("%F %T")
        print(f"[{ts}] {msg}")

    step_script = step_obj.script_path(root_dir)
    if not step_script.exists():
        raise SystemExit(f"Step script not found: {step_script}")

    step_config_path = step_obj.resolve_config_path(config_dir, occurrence_index)

    env = os.environ.copy()
    datapool_root = pipeline_env.get("DATAPOOL_ROOT", str(root_dir / "datapool"))
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
            "STEP_INDEX": str(step_index),
            "STEP_OCCURRENCE_INDEX": str(occurrence_index),
            "RUN_ID": run_id,
            "WORKDIR": str(workdir),
            "LOG_DIR": str(log_dir),
            "DRY_RUN": dry_run,
            "DATAPOOL_ROOT": datapool_root,
        }
    )
    for key in [
        "BASE_MODEL_NAME", "BASE_MODEL_SRC", "BASE_MODEL_PATH",
        "TOKENIZER_PATH", "SFT_TOKENIZER_PATH",
        "MODEL_PREFIX", "MEGATRON", "MINDSPEED", "ROOT",
    ]:
        if key in pipeline_env:
            env[key] = pipeline_env[key]

    output_dir = get_step_output_dir(step_obj, config_dir, root_dir, Path(datapool_root), pipeline_env, occurrence_index)
    if output_dir:
        clear_output_directory(output_dir, step_name, dry_run=(dry_run == "1"))

    log_name = f"{step_name}_{step_index}" if step_index > 0 else step_name
    log(f"run step[{step_index}] {step_name}")
    if dry_run == "1":
        print(f"[dry-run] would run: {step_script}")

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
    log_file = log_dir / f"{log_name}.log"
    code = tee_process(proc, log_file)
    if code != 0:
        raise SystemExit(f"step failed: {step_name} (exit={code}), see log: {log_file}")
    log(f"done step[{step_index}] {step_name}")


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
    # Resolve steps to run (from STEPS list or legacy STEP_*_ENABLED) before str() conversion
    steps_to_run = _resolve_steps(pipeline_config)
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

    for step_index, step_name in enumerate(steps_to_run):
        step_obj = get_step(step_name)
        # Occurrence index: 0 = first time this step type runs, 1 = second, ... (for config choice: convert_0.py, convert_1.py)
        occurrence_index = steps_to_run[:step_index].count(step_name)
        run_step(
            root_dir=root_dir,
            config_dir=config_dir,
            step_obj=step_obj,
            step_index=step_index,
            occurrence_index=occurrence_index,
            pipeline_env=pipeline_env,
            run_id=run_id,
            workdir=workdir,
            log_dir=log_dir,
        )

    print(f"[{time.strftime('%F %T')}] pipeline finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

