#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

from config import apply_env_imports

PIPELINE_CONTEXT_KEYS = [
    "BASE_MODEL_NAME",
    "BASE_MODEL_SRC",
    "BASE_MODEL_PATH",
    "MODEL_PREFIX",
    "MEGATRON",
    "MINDSPEED",
]


def run_extern_script(
    config: dict[str, Any],
    *,
    root_dir: Path,
    dry_run: bool,
    step_name: str,
) -> Optional[int]:
    extern_script = (config.get("EXTERN_SCRIPT") or "").strip()
    if not extern_script:
        return None
    # EXTERN_SCRIPT is a standalone command; config vars are exported to env for it to read
    print(f"{step_name}: extern_script={extern_script}")
    env = os.environ.copy()
    for key, value in config.items():
        if not isinstance(key, str):
            continue
        env[key] = str(value)
    if dry_run:
        print(f"[dry-run] would run: {extern_script}")
        return 0
    try:
        subprocess.run(extern_script, shell=True, cwd=root_dir, env=env, check=True)
    except subprocess.CalledProcessError as e:
        print(f"{step_name}: extern_script failed with exit code {e.returncode}", file=sys.stderr)
        return e.returncode
    return 0


def apply_pipeline_context(context: dict[str, str], environ: dict[str, str]) -> None:
    for key in PIPELINE_CONTEXT_KEYS:
        if key in environ:
            context[key] = environ[key]
    apply_env_imports(context, environ)


def resolve_path(path_str: str, root_dir: Path) -> Path:
    """Resolve a path string to absolute Path."""
    if not path_str:
        raise ValueError("Empty path")
    path = Path(path_str)
    if path.is_absolute():
        return path.resolve()
    return (root_dir / path).resolve()
