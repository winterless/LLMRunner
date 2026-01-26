#!/usr/bin/env python3
"""
Step 2: CPT tokenization using Megatron-LM preprocess_data.py
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))
from config import load_config_module, resolve_config_vars, require_config, require_path_exists
from tokenize_utils import expand_input_pattern


def resolve_path(path_str: str, root_dir: Path) -> Path:
    """Resolve a path string to absolute Path."""
    if not path_str:
        raise ValueError("Empty path")
    path = Path(path_str)
    if path.is_absolute():
        return path.resolve()
    return (root_dir / path).resolve()


def main() -> int:
    # Get environment variables
    root_dir = Path(os.environ["ROOT_DIR"])
    step_env_path = Path(os.environ.get("STEP_ENV_PATH", ""))
    datapool_root = Path(os.environ["DATAPOOL_ROOT"])
    dry_run = os.environ.get("DRY_RUN", "0") == "1"
    allow_external_paths = os.environ.get("ALLOW_EXTERNAL_PATHS", "0") == "1"
    
    # Load config - run.py already found the config file and passed it via STEP_ENV_PATH
    if not step_env_path or not step_env_path.exists():
        print(f"Missing config: STEP_ENV_PATH not set or file not found: {step_env_path}", file=sys.stderr)
        return 2
    
    config_path = step_env_path
    
    # If it's a .env file, error - user should migrate to .py
    if config_path.suffix == ".env":
        print(f"tokenize_cpt: .env files are deprecated, please migrate to .py config: {config_path}", file=sys.stderr)
        return 2
    
    # Load and resolve config
    config = load_config_module(config_path)
    context = {
        "DATAPOOL_ROOT": str(datapool_root),
        "ROOT_DIR": str(root_dir),
    }
    config = resolve_config_vars(config, context)
    
    # Extract required config
    # INPUT_DIR can be: directory path or single file path (glob patterns are not supported)
    # It will be expanded and merged into a single file before processing
    input_path = require_config(config, "INPUT_DIR", "tokenize_cpt")
    tokenizer_model = require_config(config, "TOKENIZER_MODEL", "tokenize_cpt")
    output_prefix = require_config(config, "OUTPUT_PREFIX", "tokenize_cpt")
    megatron_dir = require_path_exists(require_config(config, "MEGATRON_DIR", "tokenize_cpt"), root_dir, "tokenize_cpt")
    
    # Optional config with defaults
    workers = int(config.get("WORKERS", "16"))
    partitions = int(config.get("PARTITIONS", "16"))
    log_interval = int(config.get("LOG_INTERVAL", "100000"))
    json_keys = config.get("JSON_KEYS", "text")
    tokenizer_type = config.get("TOKENIZER_TYPE", "HuggingFaceTokenizer")
    tokenizer_vocab_file = config.get("TOKENIZER_VOCAB_FILE")
    # MERGE_JSONL option is deprecated - we always merge now
    
    print("tokenize_cpt: starting")
    
    # Resolve paths
    tokenizer_model_abs = resolve_path(tokenizer_model, root_dir)
    output_prefix_abs = resolve_path(output_prefix, root_dir)
    
    # Always expand input path (directory/file) and merge into a single file before processing
    # This ensures Megatron always receives a single file path
    merge_output = output_prefix_abs.parent / "merged_input.jsonl"
    
    # Extract required keys from JSON_KEYS for filtering during merge
    # This ensures consistent behavior: always create merged_input.jsonl even for single file
    if isinstance(json_keys, str):
        required_keys = json_keys.split()
    else:
        required_keys = json_keys if isinstance(json_keys, list) else None
    
    try:
        input_file_path = expand_input_pattern(
            input_path,
            root_dir,
            merge_files=True,
            merge_output=merge_output,
            required_json_keys=required_keys,
        )
        input_abs = str(resolve_path(str(input_file_path), root_dir))
    except (FileNotFoundError, ValueError) as e:
        print(f"tokenize_cpt: {e}", file=sys.stderr)
        # Check if CPT_RAW_COPY_SRC is configured
        cpt_raw_copy_src = config.get("CPT_RAW_COPY_SRC")
        if cpt_raw_copy_src:
            print(f"tokenize_cpt: Hint: Run 'python scripts/prepare_exp.py -c {step_env_path.parent.parent}/pipeline.env' to copy data from {cpt_raw_copy_src}", file=sys.stderr)
        else:
            print(f"tokenize_cpt: Hint: Configure CPT_RAW_COPY_SRC in config and run 'prepare_exp' to copy data", file=sys.stderr)
        return 2
    
    # Validate paths are under datapool (unless allowed)
    if not allow_external_paths:
        datapool_abs = datapool_root.resolve()
        input_check = str(input_abs) if isinstance(input_abs, Path) else input_abs
        if not input_check.startswith(str(datapool_abs) + "/"):
            print(
                f"tokenize_cpt: INPUT_DIR must be under DATAPOOL_ROOT ({datapool_abs}) but got: {input_check}",
                file=sys.stderr,
            )
            print("tokenize_cpt: set ALLOW_EXTERNAL_PATHS=1 in pipeline.env to override", file=sys.stderr)
            return 2
        
        if not str(output_prefix_abs).startswith(str(datapool_abs) + "/"):
            print(
                f"tokenize_cpt: OUTPUT_PREFIX must be under DATAPOOL_ROOT ({datapool_abs}) but got: {output_prefix_abs}",
                file=sys.stderr,
            )
            print("tokenize_cpt: set ALLOW_EXTERNAL_PATHS=1 in pipeline.env to override", file=sys.stderr)
            return 2
    
    # Build command
    cmd = [
        "python",
        "tools/preprocess_data.py",
        "--input",
        str(input_abs),
        "--output-prefix",
        str(output_prefix_abs),
        "--json-keys",
        json_keys,
        "--tokenizer-type",
        tokenizer_type,
        "--tokenizer-model",
        str(tokenizer_model_abs),
        "--append-eod",
        "--workers",
        str(workers),
        "--partitions",
        str(partitions),
        "--log-interval",
        str(log_interval),
    ]
    
    if tokenizer_vocab_file:
        tokenizer_vocab_file_abs = resolve_path(tokenizer_vocab_file, root_dir)
        cmd.extend(["--vocab-file", str(tokenizer_vocab_file_abs)])
    
    print(f"tokenize_cpt: megatron_dir={megatron_dir}")
    print(f"tokenize_cpt: input={input_abs}")
    print(f"tokenize_cpt: output_prefix={output_prefix_abs}")
    print(f"tokenize_cpt: tokenizer_model={tokenizer_model_abs}")
    if tokenizer_vocab_file:
        print(f"tokenize_cpt: tokenizer_vocab_file={tokenizer_vocab_file_abs}")
    
    # Create output directory
    output_prefix_abs.parent.mkdir(parents=True, exist_ok=True)
    
    # Execute
    if dry_run:
        print(f"[dry-run] tokenize_cpt: (cd {megatron_dir} && {' '.join(cmd)})")
        return 0
    
    try:
        subprocess.run(cmd, cwd=megatron_dir, check=True)
    except subprocess.CalledProcessError as e:
        print(f"tokenize_cpt: failed with exit code {e.returncode}", file=sys.stderr)
        return e.returncode
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
