#!/usr/bin/env python3
"""
Step type registry and Step abstraction.

Step types: udatasets, tokenize_cpt, tokenize_sft, train_cpt,
mg2hf, hf2mg, train_sft, eval. Run order is defined by pipeline STEPS
(with possible repeats, e.g. train_cpt, train_cpt, train_sft).

Script and config files are named by step type only: scripts/steps/<name>.py
and experiment steps/<name>.py.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

STEP_TYPES_IN_ORDER: List[str] = [
    "udatasets",
    "tokenize_cpt",
    "tokenize_sft",
    "train_cpt",
    "mg2hf",   # MG→HF: EXTERN_SCRIPT (atomic) or CONVERT_CMD+copy (export)
    "hf2mg",   # HF→MG (atomic, EXTERN_SCRIPT)
    "train_sft",
    "eval",
]


def _output_dir_from_prefix(config: Dict[str, Any], key: str = "OUTPUT_PREFIX") -> Optional[Path]:
    p = config.get(key)
    return Path(p).parent if p else None


def _output_dir_from_dir(config: Dict[str, Any], *keys: str, default: Optional[Path] = None) -> Optional[Path]:
    for k in keys:
        p = config.get(k)
        if p:
            return Path(p)
    return default


# Per-step output-dir logic for clearing before run. Returns None if step has no clearable output.
def _get_output_dir_udatasets(config: Dict[str, Any], datapool_root: Path) -> Optional[Path]:
    return _output_dir_from_dir(config, "OUTPUT_DIR") or (datapool_root / "data" / "processed")


def _get_output_dir_tokenize_cpt(config: Dict[str, Any], datapool_root: Path) -> Optional[Path]:
    return _output_dir_from_prefix(config, "OUTPUT_PREFIX")


def _get_output_dir_tokenize_sft(config: Dict[str, Any], datapool_root: Path) -> Optional[Path]:
    return _output_dir_from_prefix(config, "OUTPUT_PREFIX") or _output_dir_from_prefix(config, "SFT_OUTPUT_PREFIX")


def _get_output_dir_mg2hf(config: Dict[str, Any], datapool_root: Path) -> Optional[Path]:
    return _output_dir_from_dir(config, "OUT_HF_DIR", "OUTPUT_DIR", "HF_OUTPUT_DIR") or (datapool_root / "model" / "hf")


def _get_output_dir_eval(config: Dict[str, Any], datapool_root: Path) -> Optional[Path]:
    return _output_dir_from_dir(config, "OUTPUT_DIR", "REPORT_DIR") or (datapool_root / "reports")


# Step type -> (output_dir getter or None)
_OUTPUT_DIR_GETTERS: Dict[str, Optional[Callable[[Dict[str, Any], Path], Optional[Path]]]] = {
    "udatasets": _get_output_dir_udatasets,
    "tokenize_cpt": _get_output_dir_tokenize_cpt,
    "tokenize_sft": _get_output_dir_tokenize_sft,
    "train_cpt": None,
    "mg2hf": _get_output_dir_mg2hf,
    "hf2mg": None,
    "train_sft": None,
    "eval": _get_output_dir_eval,
}


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

    def get_output_dir(
        self,
        config: Dict[str, Any],
        datapool_root: Path,
    ) -> Optional[Path]:
        getter = _OUTPUT_DIR_GETTERS.get(self.name)
        if getter is None:
            return None
        try:
            return getter(config, datapool_root)
        except Exception:
            return None

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
