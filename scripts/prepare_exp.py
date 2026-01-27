#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


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
        in_quote: str | None = None
        cleaned: List[str] = []
        for ch in line:
            if in_quote is None and ch in ("'", '"'):
                in_quote = ch
                cleaned.append(ch)
                continue
            if in_quote is not None and ch == in_quote:
                in_quote = None
                cleaned.append(ch)
                continue
            if in_quote is None and ch == "#":
                break
            cleaned.append(ch)
        line = "".join(cleaned).strip()
        if not line or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip()
        if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
            v = v[1:-1]
        env[k] = v
    return env


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

    root_dir = Path(__file__).resolve().parent.parent
    config_dir = pipeline_config_path.parent
    
    # Load pipeline config (.py preferred, .env for backward compatibility)
    if pipeline_config_path.suffix == ".py":
        # Load Python config
        utils_dir = root_dir / "scripts" / "utils"
        if str(utils_dir) not in sys.path:
            sys.path.insert(0, str(utils_dir))
        import importlib.util
        spec = importlib.util.spec_from_file_location("config_utils", utils_dir / "config.py")
        if spec is None or spec.loader is None:
            raise SystemExit(f"Could not load config utils from {utils_dir / 'config.py'}")
        config_utils = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_utils)
        
        pipeline_config = config_utils.load_config_module(pipeline_config_path)
        # Resolve variables
        pipeline_context = {
            "DATAPOOL_ROOT": str(Path(pipeline_config.get("DATAPOOL_ROOT", "datapool")).expanduser().resolve()),
            "ROOT_DIR": str(root_dir),
        }
        pipeline_resolved = config_utils.resolve_config_vars(pipeline_config, pipeline_context)
        env = {k: str(v) for k, v in pipeline_resolved.items()}
    else:
        # Load .env config (backward compatibility)
        env = parse_env_file(pipeline_config_path)
    dp = env.get("DATAPOOL_ROOT") or "datapool"
    datapool_root = Path(dp).expanduser()
    if not datapool_root.is_absolute():
        datapool_root = (root_dir / datapool_root).resolve()
    else:
        datapool_root = datapool_root.resolve()

    ensure_datapool_structure(datapool_root)
    print(f"[{time.strftime('%F %T')}] config_dir={config_dir}")
    print(f"[{time.strftime('%F %T')}] datapool_root={datapool_root}")

    # Copy/link base model
    base_model_src = env.get("BASE_MODEL_SRC", "").strip()
    base_model_name = env.get("BASE_MODEL_NAME", "base_model").strip() or "base_model"
    if base_model_src:
        src = Path(base_model_src).expanduser().resolve()
        dst = datapool_root / "model" / "base" / base_model_name
        print(f"[{time.strftime('%F %T')}] base_model: {src} -> {dst} (mode={args.mode})")
        if not src.exists():
            raise SystemExit(f"BASE_MODEL_SRC not found: {src}")
        if dst.exists():
            print(f"[{time.strftime('%F %T')}] base_model: exists, skip -> {dst}")
        elif args.mode == "copy":
            shutil.copytree(src, dst)
        else:
            copytree_link_fallback(src, dst)
    else:
        print(f"[{time.strftime('%F %T')}] base_model: skipped (BASE_MODEL_SRC not set)")

    # Optional: copy CPT/SFT raw data from tokenize configs
    steps_dir = config_dir / "steps"
    # Try Python configs first (.py), then fallback to .env for backward compatibility
    cpt_config_path = steps_dir / "2.tokenize_cpt.py"
    if not cpt_config_path.exists():
        cpt_config_path = steps_dir / "tokenize_cpt.py"
    if not cpt_config_path.exists():
        # Fallback to .env
        cpt_config_path = steps_dir / "2.tokenize_cpt.env"
        if not cpt_config_path.exists():
            cpt_config_path = steps_dir / "tokenize_cpt.env"
    
    sft_config_path = steps_dir / "3.tokenize_sft.py"
    if not sft_config_path.exists():
        sft_config_path = steps_dir / "2.tokenize_sft.py"
        if not sft_config_path.exists():
            sft_config_path = steps_dir / "tokenize_sft.py"
    if not sft_config_path.exists():
        # Fallback to .env
        sft_config_path = steps_dir / "3.tokenize_sft.env"
        if not sft_config_path.exists():
            sft_config_path = steps_dir / "2.tokenize_sft.env"
            if not sft_config_path.exists():
                sft_config_path = steps_dir / "tokenize_sft.env"
    
    # Load config helper
    def load_config(path: Path) -> Dict[str, str]:
        if not path.exists():
            return {}
        if path.suffix == ".py":
            # Load Python config
            # Import from utils module to avoid conflicts with config files
            utils_dir = root_dir / "scripts" / "utils"
            if str(utils_dir) not in sys.path:
                sys.path.insert(0, str(utils_dir))
            import importlib.util
            spec = importlib.util.spec_from_file_location("config_utils", utils_dir / "config.py")
            if spec is None or spec.loader is None:
                raise SystemExit(f"Could not load config utils from {utils_dir / 'config.py'}")
            config_utils = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(config_utils)
            
            py_config = config_utils.load_config_module(path)
            context = {
                "DATAPOOL_ROOT": str(datapool_root),
                "ROOT_DIR": str(root_dir),
            }
            return config_utils.resolve_config_vars(py_config, context)
        else:
            # Load .env config (backward compatibility)
            return parse_env_file(path)
    
    # Copy CPT raw data
    if cpt_config_path.exists():
        cpt_config = load_config(cpt_config_path)
        copy_src = cpt_config.get("CPT_RAW_COPY_SRC", "").strip()
        if copy_src:
            src_dir = Path(copy_src).expanduser().resolve()
            if not src_dir.exists():
                raise SystemExit(f"CPT_RAW_COPY_SRC not found: {src_dir}")
            dst_dir = datapool_root / "data/raw/cpt"
            print(f"[{time.strftime('%F %T')}] CPT_RAW_COPY_SRC: {src_dir} -> {dst_dir} (mode={args.mode})")
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
    
    # Copy SFT raw data
    if sft_config_path.exists():
        sft_config = load_config(sft_config_path)
        copy_src = sft_config.get("SFT_RAW_COPY_SRC", "").strip()
        if copy_src:
            src_dir = Path(copy_src).expanduser().resolve()
            if not src_dir.exists():
                raise SystemExit(f"SFT_RAW_COPY_SRC not found: {src_dir}")
            dst_dir = datapool_root / "data/raw/sft"
            print(f"[{time.strftime('%F %T')}] SFT_RAW_COPY_SRC: {src_dir} -> {dst_dir} (mode={args.mode})")
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

    print(f"[{time.strftime('%F %T')}] prepare_exp done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

