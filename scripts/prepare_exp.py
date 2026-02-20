#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent
UTILS_DIR = ROOT_DIR / "scripts" / "utils"

# Load config utils without importing pipeline config modules.
import importlib.util

_spec = importlib.util.spec_from_file_location("config_utils", UTILS_DIR / "config.py")
if _spec is None or _spec.loader is None:
    raise SystemExit(f"Could not load config utils from {UTILS_DIR / 'config.py'}")
config_utils = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(config_utils)

_tu_spec = importlib.util.spec_from_file_location("tokenize_utils", UTILS_DIR / "tokenize_utils.py")
if _tu_spec is None or _tu_spec.loader is None:
    raise SystemExit(f"Could not load tokenize utils from {UTILS_DIR / 'tokenize_utils.py'}")
tokenize_utils = importlib.util.module_from_spec(_tu_spec)
_tu_spec.loader.exec_module(tokenize_utils)


def ensure_datapool_structure(datapool_root: Path) -> None:
    for p in [
        datapool_root / "data" / "raw",
        datapool_root / "data" / "tokenized",
        datapool_root / "model" / "cpt_checkpoints",
        datapool_root / "model" / "sft_checkpoints",
        datapool_root / "model" / "hf",
        datapool_root / "model" / "base",
        datapool_root / "reports",
    ]:
        p.mkdir(parents=True, exist_ok=True)
    return None


def _resolve_path(path_str: str, root_dir: Path) -> Path:
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (root_dir / path).resolve()


def _load_step_config(path: Path, *, root_dir: Path, datapool_root: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    if path.suffix != ".py":
        raise SystemExit("Only .py pipeline configs are supported")
    config = config_utils.load_config_module(path)
    config_utils.merge_env_defaults(config, os.environ)
    context = {
        "DATAPOOL_ROOT": str(datapool_root),
        "ROOT_DIR": str(root_dir),
    }
    config_utils.apply_env_imports(context, os.environ)
    return config_utils.resolve_config_vars(config, context)


def _iter_tokenize_step_configs(steps_dir: Path, step_type: str) -> List[Path]:
    """
    Return all tokenize step config files for a step type.
    Only instance-style naming is supported: <step_type>_<idx>.py
    Examples: tokenize_cpt_0.py, tokenize_sft_1.py
    """
    if not steps_dir.exists():
        return []

    indexed_prefix = f"{step_type}_"
    matched: List[Path] = []

    for p in steps_dir.glob(f"{step_type}_*.py"):
        stem = p.stem
        tail = stem[len(indexed_prefix):] if stem.startswith(indexed_prefix) else ""
        if tail.isdigit():
            matched.append(p)

    def sort_key(path: Path) -> tuple[int, str]:
        stem = path.stem
        tail = stem[len(indexed_prefix):]
        return (int(tail), stem)

    # Deduplicate while preserving deterministic order
    unique = {p.resolve(): p for p in matched}
    return sorted(unique.values(), key=sort_key)


def _iter_all_step_configs(steps_dir: Path) -> List[Path]:
    if not steps_dir.exists():
        return []
    return sorted([p for p in steps_dir.glob("*.py") if p.is_file()], key=lambda p: p.name)


def _ensure_data_path_dirs_from_config(
    config: Dict[str, str],
    *,
    root_dir: Path,
    source_config_name: str,
) -> None:
    for key, value in config.items():
        if not (
            isinstance(key, str)
            and (key.endswith("DATA_PATH") or key.endswith("MODEL_PATH"))
        ):
            continue
        if not isinstance(value, str):
            continue
        raw = value.strip()
        if not raw:
            continue

        # If path contains glob-like syntax, create parent directory only.
        has_glob_like = any(ch in raw for ch in ("*", "?", "[", "]", "{", "}"))
        resolved = _resolve_path(raw, root_dir)
        if has_glob_like:
            target_dir = resolved.parent
        elif key.endswith("INPUT_DATA_PATH") or raw.endswith("/"):
            target_dir = resolved
        else:
            # Most non-input DATA_PATH values are file/prefix style; ensure parent exists.
            target_dir = resolved.parent

        target_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"[{time.strftime('%F %T')}] ensure_dir[{source_config_name}]: "
            f"{key} -> {target_dir}"
        )


def prepare_from_env(
    *,
    pipeline_env: Dict[str, str],
    config_dir: Path,
    root_dir: Path,
) -> None:
    dp = pipeline_env.get("DATAPOOL_ROOT") or "datapool"
    datapool_root = _resolve_path(dp, root_dir)
    # Prepare (copy raw, merge jsonl) always runs; DRY_RUN only affects steps in run.py

    ensure_datapool_structure(datapool_root)
    print(f"[{time.strftime('%F %T')}] config_dir={config_dir}")
    print(f"[{time.strftime('%F %T')}] datapool_root={datapool_root}")

    # Copy/link base model
    base_model_src = pipeline_env.get("BASE_MODEL_SRC", "").strip()
    base_model_name = pipeline_env.get("BASE_MODEL_NAME", "base_model").strip() or "base_model"
    if base_model_src:
        src = _resolve_path(base_model_src, root_dir)
        dst = datapool_root / "model" / "base" / base_model_name
        print(f"[{time.strftime('%F %T')}] base_model: {src} -> {dst} (mode=copy)")
        if not src.exists():
            raise SystemExit(f"BASE_MODEL_SRC not found: {src}")
        if dst.exists():
            print(f"[{time.strftime('%F %T')}] base_model: exists, skip -> {dst}")
        else:
            shutil.copytree(src, dst)
    else:
        print(f"[{time.strftime('%F %T')}] base_model: skipped (BASE_MODEL_SRC not set)")

    # Optional: copy CPT/SFT raw data from all tokenize configs under steps/
    steps_dir = config_dir / "steps"
    cpt_config_paths = _iter_tokenize_step_configs(steps_dir, "tokenize_cpt")
    sft_config_paths = _iter_tokenize_step_configs(steps_dir, "tokenize_sft")
    all_step_config_paths = _iter_all_step_configs(steps_dir)

    # Ensure directories for all *_DATA_PATH config vars across all steps.
    for step_config_path in all_step_config_paths:
        step_config = _load_step_config(step_config_path, root_dir=root_dir, datapool_root=datapool_root)
        _ensure_data_path_dirs_from_config(
            step_config,
            root_dir=root_dir,
            source_config_name=step_config_path.name,
        )

    if not cpt_config_paths:
        print(f"[{time.strftime('%F %T')}] CPT_RAW_COPY_SRC: skipped (tokenize_cpt config not found)")
    for cpt_config_path in cpt_config_paths:
        cpt_config = _load_step_config(cpt_config_path, root_dir=root_dir, datapool_root=datapool_root)
        copy_src = cpt_config.get("CPT_RAW_COPY_SRC", "").strip()
        if copy_src:
            src_dir = _resolve_path(copy_src, root_dir)
            if not src_dir.exists():
                raise SystemExit(f"CPT_RAW_COPY_SRC not found: {src_dir}")
            dst_dir = datapool_root / "data" / "raw" / "cpt"
            print(
                f"[{time.strftime('%F %T')}] CPT_RAW_COPY_SRC[{cpt_config_path.name}]: "
                f"{src_dir} -> {dst_dir} (mode=copy)"
            )
            copied, clashes = copy_jsonl_flat(src_dir, dst_dir)
            print(
                f"[{time.strftime('%F %T')}] CPT_RAW_COPY_SRC[{cpt_config_path.name}]: "
                f"copied_jsonl={copied} clashes={len(clashes)}"
            )
            if clashes:
                for x in clashes[:20]:
                    print(f"  [warn] skip (exists): {x}", file=sys.stderr)
                if len(clashes) > 20:
                    print(f"  ... and {len(clashes) - 20} more", file=sys.stderr)
        else:
            print(f"[{time.strftime('%F %T')}] CPT_RAW_COPY_SRC: skipped (not set in {cpt_config_path.name})")

        merge_jsonl = str(cpt_config.get("MERGE_JSONL", "1")) == "1"
        if merge_jsonl:
            input_path = cpt_config.get("INPUT_DATA_PATH", "").strip()
            json_keys = cpt_config.get("JSON_KEYS", "text")
            shuffle_jsonl = str(cpt_config.get("SHUFFLE_JSONL", "0")) == "1"
            shuffle_seed = cpt_config.get("SHUFFLE_SEED")
            shuffle_buffer = int(cpt_config.get("SHUFFLE_BUFFER", "10000"))
            if input_path:
                input_abs = _resolve_path(input_path, root_dir)
                # Write merged input under raw/cpt so it is not cleared when tokenized/cpt is cleared
                merge_output = (input_abs / "merged_input.jsonl") if input_abs.is_dir() else (input_abs.parent / "merged_input.jsonl")
                if isinstance(json_keys, str):
                    required_keys = json_keys.split()
                else:
                    required_keys = json_keys if isinstance(json_keys, list) else None
                if merge_output.exists():
                    print(
                        f"[{time.strftime('%F %T')}] CPT merge_jsonl[{cpt_config_path.name}]: "
                        f"skipped (exists) output={merge_output}"
                    )
                else:
                    tokenize_utils.expand_input_pattern(
                        input_path,
                        root_dir,
                        merge_files=True,
                        merge_output=merge_output,
                        required_json_keys=required_keys,
                        shuffle=shuffle_jsonl,
                        shuffle_seed=int(shuffle_seed) if shuffle_seed else None,
                        shuffle_buffer=shuffle_buffer,
                    )
                    print(
                        f"[{time.strftime('%F %T')}] CPT merge_jsonl[{cpt_config_path.name}]: "
                        f"output={merge_output} shuffle={shuffle_jsonl}"
                    )
            else:
                print(f"[{time.strftime('%F %T')}] CPT merge_jsonl: skipped (missing INPUT_DATA_PATH in {cpt_config_path.name})")

    if not sft_config_paths:
        print(f"[{time.strftime('%F %T')}] SFT_RAW_COPY_SRC: skipped (tokenize_sft config not found)")
    for sft_config_path in sft_config_paths:
        sft_config = _load_step_config(sft_config_path, root_dir=root_dir, datapool_root=datapool_root)
        copy_src = sft_config.get("SFT_RAW_COPY_SRC", "").strip()
        if copy_src:
            src_dir = _resolve_path(copy_src, root_dir)
            if not src_dir.exists():
                raise SystemExit(f"SFT_RAW_COPY_SRC not found: {src_dir}")
            dst_dir = datapool_root / "data" / "raw" / "sft"
            print(
                f"[{time.strftime('%F %T')}] SFT_RAW_COPY_SRC[{sft_config_path.name}]: "
                f"{src_dir} -> {dst_dir} (mode=copy)"
            )
            copied, clashes = copy_jsonl_flat(src_dir, dst_dir)
            print(
                f"[{time.strftime('%F %T')}] SFT_RAW_COPY_SRC[{sft_config_path.name}]: "
                f"copied_jsonl={copied} clashes={len(clashes)}"
            )
            if clashes:
                for x in clashes[:20]:
                    print(f"  [warn] skip (exists): {x}", file=sys.stderr)
                if len(clashes) > 20:
                    print(f"  ... and {len(clashes) - 20} more", file=sys.stderr)
        else:
            print(f"[{time.strftime('%F %T')}] SFT_RAW_COPY_SRC: skipped (not set in {sft_config_path.name})")

        merge_jsonl = str(sft_config.get("MERGE_JSONL", "1")) == "1"
        if merge_jsonl:
            input_path = sft_config.get("INPUT_DATA_PATH") or sft_config.get("SFT_INPUT_DATA_PATH", "")
            json_keys = sft_config.get("JSON_KEYS") or sft_config.get("SFT_JSON_KEYS", "instruction input output")
            shuffle_jsonl = str(sft_config.get("SHUFFLE_JSONL", "0")) == "1"
            shuffle_seed = sft_config.get("SHUFFLE_SEED")
            shuffle_buffer = int(sft_config.get("SHUFFLE_BUFFER", "10000"))
            if input_path:
                input_abs = _resolve_path(input_path, root_dir)
                # Write merged input under raw/sft so it is not cleared when tokenized/sft is cleared
                merge_output = (input_abs / "merged_input.jsonl") if input_abs.is_dir() else (input_abs.parent / "merged_input.jsonl")
                if isinstance(json_keys, str):
                    required_keys = json_keys.split()
                else:
                    required_keys = json_keys if isinstance(json_keys, list) else None
                if merge_output.exists():
                    print(
                        f"[{time.strftime('%F %T')}] SFT merge_jsonl[{sft_config_path.name}]: "
                        f"skipped (exists) output={merge_output}"
                    )
                else:
                    tokenize_utils.expand_input_pattern(
                        input_path,
                        root_dir,
                        merge_files=True,
                        merge_output=merge_output,
                        required_json_keys=required_keys,
                        shuffle=shuffle_jsonl,
                        shuffle_seed=int(shuffle_seed) if shuffle_seed else None,
                        shuffle_buffer=shuffle_buffer,
                    )
                    print(
                        f"[{time.strftime('%F %T')}] SFT merge_jsonl[{sft_config_path.name}]: "
                        f"output={merge_output} shuffle={shuffle_jsonl}"
                    )
            else:
                print(f"[{time.strftime('%F %T')}] SFT merge_jsonl: skipped (missing INPUT_DATA_PATH in {sft_config_path.name})")

    print(f"[{time.strftime('%F %T')}] prepare_exp done")


def _copy_or_link_file(src: str, dst: str) -> None:
    """
    Try hardlink first (fast, saves space). If cross-device, fallback to copy2.
    """
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def iter_jsonl_files_recursive(src_dir: Path) -> Iterable[Path]:
    for p in src_dir.rglob("*.jsonl"):
        if p.is_file():
            yield p


def copy_jsonl_flat(src_dir: Path, dst_dir: Path) -> Tuple[int, List[str]]:
    dst_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    clashes: List[str] = []
    for p in iter_jsonl_files_recursive(src_dir):
        rel = p.relative_to(src_dir)
        # flatten: replace path separators with '__' to avoid subdirs
        flat_name = "__".join(rel.parts)
        out = dst_dir / flat_name
        if out.exists():
            clashes.append(str(out))
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        _copy_or_link_file(str(p), str(out))
        copied += 1
    return copied, clashes


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python scripts/prepare_exp.py")
    ap.add_argument("-c", "--config", required=True, help="Path to pipeline.py")
    args = ap.parse_args(argv)
    pipeline_config_path = Path(args.config).expanduser().resolve()
    if not pipeline_config_path.exists():
        raise SystemExit(f"Config not found: {pipeline_config_path}")

    root_dir = ROOT_DIR
    config_dir = pipeline_config_path.parent
    if not os.environ.get("DATAPOOL"):
        os.environ["DATAPOOL"] = str(root_dir / "datapool")
    
    # Load pipeline config (.py only)
    if pipeline_config_path.suffix != ".py":
        raise SystemExit("Only .py pipeline configs are supported")
    pipeline_config = config_utils.load_config_module(pipeline_config_path)
    config_utils.merge_env_defaults(pipeline_config, os.environ)
    # Resolve variables
    temp_context: Dict[str, str] = {}
    config_utils.apply_env_imports(temp_context, os.environ)
    temp_resolved = config_utils.resolve_config_vars(pipeline_config, temp_context)
    pipeline_context = {
        "DATAPOOL_ROOT": str(Path(temp_resolved.get("DATAPOOL_ROOT", "datapool")).expanduser().resolve()),
        "ROOT_DIR": str(root_dir),
    }
    config_utils.apply_env_imports(pipeline_context, os.environ)
    pipeline_resolved = config_utils.resolve_config_vars(pipeline_config, pipeline_context)
    env = {k: str(v) for k, v in pipeline_resolved.items()}
    prepare_from_env(
        pipeline_env=env,
        config_dir=config_dir,
        root_dir=root_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

