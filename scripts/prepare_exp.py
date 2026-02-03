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
        datapool_root / "data" / "processed",
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


def prepare_from_env(
    *,
    pipeline_env: Dict[str, str],
    config_dir: Path,
    root_dir: Path,
    mode: str = "copy",
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
        print(f"[{time.strftime('%F %T')}] base_model: {src} -> {dst} (mode={mode})")
        if not src.exists():
            raise SystemExit(f"BASE_MODEL_SRC not found: {src}")
        if dst.exists():
            print(f"[{time.strftime('%F %T')}] base_model: exists, skip -> {dst}")
        elif mode == "copy":
            shutil.copytree(src, dst)
        else:
            copytree_link_fallback(src, dst)
    else:
        print(f"[{time.strftime('%F %T')}] base_model: skipped (BASE_MODEL_SRC not set)")

    # Optional: copy CPT/SFT raw data from tokenize configs
    steps_dir = config_dir / "steps"
    cpt_config_path = steps_dir / "2.tokenize_cpt.py"
    if not cpt_config_path.exists():
        cpt_config_path = steps_dir / "tokenize_cpt.py"

    sft_config_path = steps_dir / "3.tokenize_sft.py"
    if not sft_config_path.exists():
        sft_config_path = steps_dir / "2.tokenize_sft.py"
        if not sft_config_path.exists():
            sft_config_path = steps_dir / "tokenize_sft.py"

    # Copy CPT raw data
    if cpt_config_path.exists():
        cpt_config = _load_step_config(cpt_config_path, root_dir=root_dir, datapool_root=datapool_root)
        copy_src = cpt_config.get("CPT_RAW_COPY_SRC", "").strip()
        if copy_src:
            src_dir = _resolve_path(copy_src, root_dir)
            if not src_dir.exists():
                raise SystemExit(f"CPT_RAW_COPY_SRC not found: {src_dir}")
            dst_dir = datapool_root / "data" / "raw" / "cpt"
            print(f"[{time.strftime('%F %T')}] CPT_RAW_COPY_SRC: {src_dir} -> {dst_dir} (mode={mode})")
            copied, clashes = copy_jsonl_flat(src_dir, dst_dir)
            print(f"[{time.strftime('%F %T')}] CPT_RAW_COPY_SRC: copied_jsonl={copied} clashes={len(clashes)}")
            if clashes:
                for x in clashes[:20]:
                    print(f"  [warn] skip (exists): {x}", file=sys.stderr)
                if len(clashes) > 20:
                    print(f"  ... and {len(clashes) - 20} more", file=sys.stderr)
        else:
            print(f"[{time.strftime('%F %T')}] CPT_RAW_COPY_SRC: skipped (not set in {cpt_config_path.name})")
    else:
        print(f"[{time.strftime('%F %T')}] CPT_RAW_COPY_SRC: skipped (config not found)")

    # Optional: merge + shuffle CPT jsonl
    if cpt_config_path.exists():
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
                    print(f"[{time.strftime('%F %T')}] CPT merge_jsonl: skipped (exists) output={merge_output}")
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
                    print(f"[{time.strftime('%F %T')}] CPT merge_jsonl: output={merge_output} shuffle={shuffle_jsonl}")
            else:
                print(f"[{time.strftime('%F %T')}] CPT merge_jsonl: skipped (missing INPUT_DATA_PATH)")

    # Copy SFT raw data
    if sft_config_path.exists():
        sft_config = _load_step_config(sft_config_path, root_dir=root_dir, datapool_root=datapool_root)
        copy_src = sft_config.get("SFT_RAW_COPY_SRC", "").strip()
        if copy_src:
            src_dir = _resolve_path(copy_src, root_dir)
            if not src_dir.exists():
                raise SystemExit(f"SFT_RAW_COPY_SRC not found: {src_dir}")
            dst_dir = datapool_root / "data" / "raw" / "sft"
            print(f"[{time.strftime('%F %T')}] SFT_RAW_COPY_SRC: {src_dir} -> {dst_dir} (mode={mode})")
            copied, clashes = copy_jsonl_flat(src_dir, dst_dir)
            print(f"[{time.strftime('%F %T')}] SFT_RAW_COPY_SRC: copied_jsonl={copied} clashes={len(clashes)}")
            if clashes:
                for x in clashes[:20]:
                    print(f"  [warn] skip (exists): {x}", file=sys.stderr)
                if len(clashes) > 20:
                    print(f"  ... and {len(clashes) - 20} more", file=sys.stderr)
        else:
            print(f"[{time.strftime('%F %T')}] SFT_RAW_COPY_SRC: skipped (not set in {sft_config_path.name})")
    else:
        print(f"[{time.strftime('%F %T')}] SFT_RAW_COPY_SRC: skipped (config not found)")

    # Optional: merge + shuffle SFT jsonl
    if sft_config_path.exists():
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
                    print(f"[{time.strftime('%F %T')}] SFT merge_jsonl: skipped (exists) output={merge_output}")
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
                    print(f"[{time.strftime('%F %T')}] SFT merge_jsonl: output={merge_output} shuffle={shuffle_jsonl}")
            else:
                print(f"[{time.strftime('%F %T')}] SFT merge_jsonl: skipped (missing INPUT_DATA_PATH)")

    print(f"[{time.strftime('%F %T')}] prepare_exp done")


def _copy_or_link_file(src: str, dst: str) -> None:
    """
    Try hardlink first (fast, saves space). If cross-device, fallback to copy2.
    """
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def copytree_link_fallback(src: Path, dst: Path) -> None:
    if dst.exists():
        # keep behavior simple & safe: refuse to clobber
        raise SystemExit(f"destination already exists: {dst}")
    shutil.copytree(src, dst, copy_function=_copy_or_link_file)


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


def clone_experiment(
    root_dir: Path,
    source_name: str,
    new_name: str,
    copy_datapool: bool,
) -> None:
    """复制实验：config、并可选复制 datapool（数据/模型等）。"""
    experiments = root_dir / "configs" / "experiments"
    source_dir = experiments / source_name
    new_dir = experiments / new_name
    if not source_dir.is_dir():
        raise SystemExit(f"Source experiment not found: {source_dir}")
    if new_dir.exists():
        raise SystemExit(f"Destination already exists: {new_dir}")

    # 复制整份 config（含 pipeline.env 与 steps/*.env）
    shutil.copytree(source_dir, new_dir)
    print(f"[{time.strftime('%F %T')}] cloned config: {source_dir.name} -> {new_name}")

    # 推断源/新 datapool 路径（与 pipeline.env 约定一致）
    source_dp = f"datapool/experiments/{source_name}"
    new_dp = f"datapool/experiments/{new_name}"
    source_dp_abs = (root_dir / source_dp).resolve()
    new_dp_abs = (root_dir / new_dp).resolve()

    # 把所有 .env 里的源 datapool 路径改成新的
    for env_path in new_dir.rglob("*.env"):
        text = env_path.read_text(encoding="utf-8")
        if source_dp in text:
            new_text = text.replace(source_dp, new_dp)
            env_path.write_text(new_text, encoding="utf-8")
            print(f"[{time.strftime('%F %T')}] updated {env_path.relative_to(new_dir)}")

    ensure_datapool_structure(new_dp_abs)

    if copy_datapool and source_dp_abs.is_dir():
        # 复制 data/、model/（含 base、cpt_checkpoints、sft_checkpoints、hf）、reports/
        for sub in ("data", "model", "reports"):
            src_sub = source_dp_abs / sub
            dst_sub = new_dp_abs / sub
            if not src_sub.is_dir():
                continue
            if dst_sub.exists():
                print(f"[{time.strftime('%F %T')}] skip (exists): {dst_sub}")
                continue
            shutil.copytree(src_sub, dst_sub, copy_function=_copy_or_link_file)
            print(f"[{time.strftime('%F %T')}] copied datapool: {sub}/ -> {new_dp}/{sub}/")
    elif copy_datapool:
        print(f"[{time.strftime('%F %T')}] source datapool missing, skipped copy: {source_dp_abs}")

    print(f"[{time.strftime('%F %T')}] clone-experiment done. New config: configs/experiments/{new_name}/pipeline.env")


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python scripts/prepare_exp.py")
    ap.add_argument("-c", "--config", help="Path to pipeline.env (required unless --copy-jsonl or --clone-experiment)")
    ap.add_argument(
        "--copy-jsonl",
        nargs=2,
        metavar=("SRC_DIR", "DST_DIR"),
        help="Copy *.jsonl from SRC_DIR to DST_DIR (flat); then exit. Use from step scripts when COPY_SRC is set.",
    )
    ap.add_argument(
        "--clone-experiment",
        nargs=2,
        metavar=("SOURCE_EXP", "NEW_EXP"),
        help="Create new experiment by copying SOURCE_EXP config (and optionally datapool). SOURCE_EXP/NEW_EXP are dir names under configs/experiments/.",
    )
    ap.add_argument(
        "--copy-datapool",
        action="store_true",
        help="With --clone-experiment: also copy source experiment datapool (data/, model/, reports/) to new experiment.",
    )
    ap.add_argument(
        "--mode",
        choices=["link", "copy"],
        default="link",
        help="For -c: base model mode; for --copy-jsonl: link or copy (default link)",
    )
    args = ap.parse_args(argv)

    if args.clone_experiment:
        root_dir = Path(__file__).resolve().parent.parent
        clone_experiment(root_dir, args.clone_experiment[0], args.clone_experiment[1], args.copy_datapool)
        return 0

    if args.copy_jsonl:
        src_dir = Path(args.copy_jsonl[0]).expanduser().resolve()
        dst_dir = Path(args.copy_jsonl[1]).expanduser().resolve()
        if not src_dir.exists():
            raise SystemExit(f"copy-jsonl: SRC_DIR not found: {src_dir}")
        dst_dir.mkdir(parents=True, exist_ok=True)
        copied, clashes = copy_jsonl_flat(src_dir, dst_dir)
        print(f"[{time.strftime('%F %T')}] copy-jsonl: {src_dir} -> {dst_dir} copied={copied} clashes={len(clashes)}")
        if clashes:
            for x in clashes[:20]:
                print(f"  [warn] skip (exists): {x}", file=sys.stderr)
            if len(clashes) > 20:
                print(f"  ... and {len(clashes) - 20} more", file=sys.stderr)
        return 0

    if not args.config:
        ap.error("-c/--config required when not using --copy-jsonl")
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
        mode=args.mode,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

