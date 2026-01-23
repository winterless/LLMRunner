#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


STEP_ORDER: List[str] = ["udatasets", "tokenize", "train_cpt", "train_sft", "convert", "eval"]


def parse_env_file(path: Path) -> Dict[str, str]:
    """
    Minimal .env parser (stdlib only).
    Supports:
      KEY=value
      KEY="value"
      KEY='value'
    Ignores blank lines and lines starting with #.
    """
    env: Dict[str, str] = {}
    if not path.exists():
        return env
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        # Strip inline comments: everything after an unquoted '#'
        # (common .env style: KEY=value  # comment)
        in_quote: str | None = None
        cleaned_chars: List[str] = []
        for ch in line:
            if in_quote is None and ch in ("'", '"'):
                in_quote = ch
                cleaned_chars.append(ch)
                continue
            if in_quote is not None and ch == in_quote:
                in_quote = None
                cleaned_chars.append(ch)
                continue
            if in_quote is None and ch == "#":
                break
            cleaned_chars.append(ch)
        line = "".join(cleaned_chars).strip()
        if not line:
            continue
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip()
        if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
            v = v[1:-1]
        env[k] = v
    return env


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


def run_step(
    *,
    root_dir: Path,
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

    step_script = root_dir / "scripts" / "steps" / f"{step}.sh"
    if not step_script.exists():
        raise SystemExit(f"Step script not found: {step_script}")

    step_env_path = root_dir / "configs" / "steps" / f"{step}.env"
    step_env = parse_env_file(step_env_path)

    env = os.environ.copy()
    datapool_root = pipeline_env.get("DATAPOOL_ROOT", str(root_dir / "datapool"))
    run_dir = str((Path(datapool_root).expanduser().resolve() / "intermediates" / run_id))
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
            "RUN_ID": run_id,
            "WORKDIR": str(workdir),
            "LOG_DIR": str(log_dir),
            "DRY_RUN": dry_run,
            "DATAPOOL_ROOT": datapool_root,
            "RUN_DIR": run_dir,
            # escape hatch (default 0): allow steps to read/write outside DATAPOOL_ROOT
            "ALLOW_EXTERNAL_PATHS": pipeline_env.get("ALLOW_EXTERNAL_PATHS", "0"),
        }
    )
    env.update(step_env)

    log(f"run step={step}")
    if dry_run == "1":
        print(f"[dry-run] would run: {step_script}")
        # still invoke the script in dry-run mode so it can print the planned command
    proc = subprocess.Popen(
        ["bash", str(step_script)],
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
    ap.add_argument("-c", "--config", required=True, help="Path to pipeline.env")
    args = ap.parse_args(argv)

    root_dir = Path(__file__).resolve().parent.parent
    pipeline_env_path = Path(args.config).expanduser().resolve()
    if not pipeline_env_path.exists():
        raise SystemExit(f"Config not found: {pipeline_env_path}")

    pipeline_env = parse_env_file(pipeline_env_path)

    run_id = pipeline_env.get("RUN_ID") or now_run_id()
    workdir = Path(pipeline_env.get("WORKDIR") or (root_dir / ".llmrunner")).expanduser().resolve()
    log_dir = Path(pipeline_env.get("LOG_DIR") or (workdir / "logs" / run_id)).expanduser().resolve()
    datapool_root = Path(pipeline_env.get("DATAPOOL_ROOT") or (root_dir / "datapool")).expanduser().resolve()
    run_dir = datapool_root / "intermediates" / run_id

    workdir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    # Ensure datapool structure exists (single source of truth for data/artifacts)
    for p in [
        datapool_root / "data" / "raw",
        datapool_root / "data" / "processed",
        datapool_root / "data" / "tokenized",
        datapool_root / "model" / "cpt_checkpoints",
        datapool_root / "model" / "sft_checkpoints",
        datapool_root / "model" / "hf",
        datapool_root / "reports",
        datapool_root / "intermediates",
        run_dir,
    ]:
        p.mkdir(parents=True, exist_ok=True)

    print(
        f"[{time.strftime('%F %T')}] run_id={run_id} workdir={workdir} datapool_root={datapool_root} run_dir={run_dir} dry_run={pipeline_env.get('DRY_RUN','0')}"
    )

    for step in STEP_ORDER:
        run_step(
            root_dir=root_dir,
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

