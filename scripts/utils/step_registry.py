#!/usr/bin/env python3
"""
Step type registry and Step abstraction.

Step types: tokenize_cpt, tokenize_sft, train_cpt,
mg2hf, hf2mg, train_sft, eval. Run order is defined by pipeline STEPS
(with possible repeats, e.g. train_cpt, train_cpt, train_sft).

Script and config files are named by step type only: scripts/steps/<name>.py
and experiment steps/<name>.py.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

STEP_TYPES_IN_ORDER: List[str] = [
    "udataset",
    "tokenize_cpt",
    "tokenize_sft",
    "train_cpt",
    "mg2hf",   # MG→HF: EXTERN_SCRIPT (atomic) or CONVERT_CMD+copy (export)
    "hf2mg",   # HF→MG (atomic, EXTERN_SCRIPT)
    "train_sft",
    "eval",
]


class Step:
    """
    One step type in the pipeline. Script and config: scripts/steps/<name>.py,
    experiment steps/<name>.py. Same step type can appear multiple times in STEPS.
    """

    __slots__ = ("name",)

    def __init__(self, name: str) -> None:
        self.name = name

    @property
    def script_name(self) -> str:
        return f"{self.name}.py"

    def script_path(self, root_dir: Path) -> Path:
        return root_dir / "scripts" / "steps" / self.script_name

    def config_path(self, config_dir: Path, occurrence_index: int = 0) -> Path:
        """Config path: steps/<name>_<occurrence_index>.py (e.g. convert_0.py) else steps/<name>.py."""
        indexed = config_dir / "steps" / f"{self.name}_{occurrence_index}.py"
        if indexed.exists():
            return indexed
        return config_dir / "steps" / self.script_name

    def resolve_config_path(self, config_dir: Path, occurrence_index: int = 0) -> Path:
        """Path to use for config. First run of this step type → 0, second → 1, etc."""
        return self.config_path(config_dir, occurrence_index)

    def __repr__(self) -> str:
        return f"Step({self.name!r})"


def get_step(step_name: str) -> Step:
    if step_name not in STEP_TYPES_IN_ORDER:
        raise ValueError(
            f"Unknown step: {step_name!r}. Valid steps: {STEP_TYPES_IN_ORDER}"
        )
    return Step(step_name)


def all_step_names() -> List[str]:
    return list(STEP_TYPES_IN_ORDER)
