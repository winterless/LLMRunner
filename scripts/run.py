#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
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


@dataclass(frozen=True)
class StepInstance:
    """
    One concrete step run in a pipeline.

    - step_type: atomic capability type, e.g. "train_cpt"
    - instance_id: unique run id within current pipeline
    - config_ref: optional override config path (relative to config_dir/ or absolute)
    - position: index in full pipeline sequence
    - occurrence_index: 0-based index among same step_type occurrences
    """

    step_type: str
    instance_id: str
    config_ref: Optional[str]
    position: int
    occurrence_index: int


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
    from utils.config import load_config_module, merge_env_defaults, resolve_config_vars
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
    step_instance: StepInstance,
    config_dir: Path,
    root_dir: Path,
    datapool_root: Path,
    pipeline_env: Dict[str, str],
) -> Optional[Path]:
    """Get the output directory for a step by loading its config. Returns None if not clearable."""
    step_config_path = resolve_step_config_path(step_obj, step_instance, config_dir)
    if not step_config_path.exists():
        return None
    try:
        config = _load_step_config(step_config_path, root_dir, datapool_root, pipeline_env)
        return step_obj.get_output_dir(config, datapool_root)
    except Exception:
        return None


def _normalize_instance_dict(item: Dict[str, Any], idx: int) -> Dict[str, Any]:
    """
    Normalize dict-style step instance entry.

    Supported keys:
      - type (required)
      - id (optional)
      - config (optional)
      - enabled (optional, default true)
    """
    if "type" not in item and "step" in item:
        item["type"] = item["step"]
    if "type" not in item:
        raise SystemExit(f"STEPS[{idx}] object must include 'type' (or 'step')")
    return item


def _parse_enabled(value: Any) -> bool:
    """Parse enabled flag from bool/int/str safely."""
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value != 0
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"1", "true", "yes", "on"}:
            return True
        if v in {"0", "false", "no", "off", ""}:
            return False
    raise SystemExit(f"Invalid enabled value: {value!r}. Use true/false.")


def _canonical_instance_id(step_type: str, occurrence_index: int) -> str:
    """Build canonical instance id: type_idx."""
    return f"{step_type}_{occurrence_index}"


def _resolve_steps(pipeline_config: Dict[str, Any]) -> List[StepInstance]:
    """
    Resolve list of step instances to run.
    - If pipeline has STEPS (list/tuple), supports:
      1) string entries: ["train_cpt", "train_cpt", ...]
      2) object entries: [{"id":"cpt_stage1","type":"train_cpt","config":"steps/cpt_stage1.py"}, ...]
    - Else fall back to legacy: all_step_names() in order, filter by STEP_*_ENABLED=1.
    """
    steps_raw = pipeline_config.get("STEPS")
    if steps_raw is not None:
        if not isinstance(steps_raw, (list, tuple)):
            raise SystemExit("STEPS must be a list/tuple")

        seen_counts: Dict[str, int] = {}
        instances: List[StepInstance] = []
        used_ids: set[str] = set()

        for idx, raw in enumerate(steps_raw):
            enabled = True
            config_ref: Optional[str] = None

            if isinstance(raw, str):
                step_type = raw
                explicit_id: Optional[str] = None
            elif isinstance(raw, dict):
                item = _normalize_instance_dict(dict(raw), idx)
                step_type = str(item["type"])
                explicit_id = str(item["id"]) if item.get("id") is not None else None
                config_ref = str(item["config"]) if item.get("config") is not None else None
                if "enabled" in item:
                    enabled = _parse_enabled(item["enabled"])
            else:
                raise SystemExit(f"Unsupported STEPS[{idx}] entry type: {type(raw).__name__}")

            if not enabled:
                continue

            occurrence_index = seen_counts.get(step_type, 0)
            seen_counts[step_type] = occurrence_index + 1
            canonical_id = _canonical_instance_id(step_type, occurrence_index)
            if explicit_id is None:
                instance_id = canonical_id
            else:
                instance_id = explicit_id
                if instance_id != canonical_id:
                    raise SystemExit(
                        f"Invalid STEPS[{idx}].id={instance_id!r}; expected {canonical_id!r} "
                        f"for type={step_type!r} occurrence={occurrence_index}"
                    )
            if config_ref and Path(config_ref).stem != instance_id:
                raise SystemExit(
                    f"Invalid STEPS[{idx}].config={config_ref!r}; filename stem must equal id={instance_id!r}"
                )

            if instance_id in used_ids:
                raise SystemExit(f"Duplicate step instance id in STEPS: {instance_id!r}")
            used_ids.add(instance_id)

            instances.append(
                StepInstance(
                    step_type=step_type,
                    instance_id=instance_id,
                    config_ref=config_ref,
                    position=len(instances),
                    occurrence_index=occurrence_index,
                )
            )

        return instances
    # Legacy: use default order, include only enabled
    result: List[StepInstance] = []
    seen_counts: Dict[str, int] = {}
    for name in all_step_names():
        enabled_key = f"STEP_{name.upper().replace('-', '_')}_ENABLED"
        if str(pipeline_config.get(enabled_key, "0")) == "1":
            occurrence_index = seen_counts.get(name, 0)
            seen_counts[name] = occurrence_index + 1
            instance_id = f"{name}_{occurrence_index}"
            result.append(
                StepInstance(
                    step_type=name,
                    instance_id=instance_id,
                    config_ref=None,
                    position=len(result),
                    occurrence_index=occurrence_index,
                )
            )
    return result


def resolve_step_config_path(step_obj: Step, step_instance: StepInstance, config_dir: Path) -> Path:
    """
    Resolve config path for a step instance:
    1) instance.config_ref (absolute or relative to config_dir) if set
    2) fallback to steps/<instance_id>.py (instance-bound config)
    """
    if step_instance.config_ref:
        cfg = Path(step_instance.config_ref)
        if not cfg.is_absolute():
            cfg = (config_dir / cfg).resolve()
        return cfg
    return (config_dir / "steps" / f"{step_instance.instance_id}.py").resolve()


def run_step(
    *,
    root_dir: Path,
    config_dir: Path,
    step_obj: Step,
    step_instance: StepInstance,
    pipeline_env: Dict[str, str],
    run_id: str,
    workdir: Path,
    log_dir: Path,
) -> None:
    dry_run = pipeline_env.get("DRY_RUN", "0")
    step_name = step_obj.name
    step_index = step_instance.position

    def log(msg: str) -> None:
        ts = time.strftime("%F %T")
        print(f"[{ts}] {msg}")

    step_config_path = resolve_step_config_path(step_obj, step_instance, config_dir)

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
            "STEP_OCCURRENCE_INDEX": str(step_instance.occurrence_index),
            "STEP_ID": step_instance.instance_id,
            "STEP_TYPE": step_instance.step_type,
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

    output_dir = get_step_output_dir(step_obj, step_instance, config_dir, root_dir, Path(datapool_root), pipeline_env)
    if output_dir:
        clear_output_directory(output_dir, step_instance.instance_id, dry_run=(dry_run == "1"))

    # Load resolved step config once for script-mode execution and env export.
    resolved_step_config: Dict[str, Any] = {}
    if step_config_path.exists():
        try:
            resolved_step_config = _load_step_config(
                step_config_path=step_config_path,
                root_dir=root_dir,
                datapool_root=Path(datapool_root),
                pipeline_env=pipeline_env,
            )
        except Exception:
            resolved_step_config = {}

    # Execution mode: explicit SCRIPT from step config (required).
    script_cmd = str(resolved_step_config.get("SCRIPT", "")).strip()
    script_cwd_str = str(resolved_step_config.get("SCRIPT_CWD", "")).strip()
    script_cwd = root_dir
    if script_cwd_str:
        script_cwd = Path(script_cwd_str)
        if not script_cwd.is_absolute():
            script_cwd = (root_dir / script_cwd).resolve()
        else:
            script_cwd = script_cwd.resolve()

    # Export resolved config values as env for script mode.
    for key, value in resolved_step_config.items():
        if not isinstance(key, str):
            continue
        env[key] = str(value)

    log_name = step_instance.instance_id
    log(f"run step[{step_index}] id={step_instance.instance_id} type={step_name}")

    if not script_cmd:
        raise SystemExit(
            f"step config must set SCRIPT: id={step_instance.instance_id} type={step_name} config={step_config_path}"
        )

    if dry_run == "1":
        print(f"[dry-run] (cd {script_cwd} && {script_cmd})")
        return
    proc = subprocess.Popen(
        script_cmd,
        shell=True,
        cwd=str(script_cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    log_file = log_dir / f"{log_name}.log"
    code = tee_process(proc, log_file)
    if code != 0:
        raise SystemExit(f"step failed: id={step_instance.instance_id} type={step_name} (exit={code}), see log: {log_file}")
    log(f"done step[{step_index}] id={step_instance.instance_id} type={step_name}")


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
    from utils.config import apply_env_imports, load_config_module, merge_env_defaults, resolve_config_vars
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

    for step_instance in steps_to_run:
        step_obj = get_step(step_instance.step_type)
        run_step(
            root_dir=root_dir,
            config_dir=config_dir,
            step_obj=step_obj,
            step_instance=step_instance,
            pipeline_env=pipeline_env,
            run_id=run_id,
            workdir=workdir,
            log_dir=log_dir,
        )

    print(f"[{time.strftime('%F %T')}] pipeline finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

