#!/usr/bin/env python3
"""
Step 3: SFT tokenization using Megatron-LM preprocess_data.py
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))
from config import load_config_module, merge_env_defaults, resolve_config_vars, require_config, require_path_exists
from step_utils import apply_pipeline_context, resolve_path, run_extern_script
from tokenize_utils import expand_input_pattern, rewrite_sft_jsonl_to_input_label


def main() -> int:
    # Get environment variables
    root_dir = Path(os.environ["ROOT_DIR"])
    config_dir = Path(os.environ.get("CONFIG_DIR", root_dir / "configs"))
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
        print(f"tokenize_sft: .env files are deprecated, please migrate to .py config: {config_path}", file=sys.stderr)
        return 2
    
    # Load and resolve config
    config = load_config_module(config_path)
    merge_env_defaults(config, os.environ)
    context = {
        "DATAPOOL_ROOT": str(datapool_root),
        "ROOT_DIR": str(root_dir),
    }
    # Add pipeline config variables (BASE_MODEL_NAME, BASE_MODEL_SRC, BASE_MODEL_PATH, MODEL_PREFIX, MEGATRON, MINDSPEED)
    apply_pipeline_context(context, os.environ)
    config = resolve_config_vars(config, context)
    
    # Extern script shortcut (run entire tokenize outside this step)
    extern_result = run_extern_script(
        config,
        root_dir=root_dir,
        dry_run=dry_run,
        step_name="tokenize_sft",
    )
    if extern_result is not None:
        return extern_result
    # Extract required config (support both naming conventions)
    # INPUT_DATA_PATH can be: directory path or single file path (glob patterns are not supported)
    # It will be expanded and merged into a single file before processing
    input_path = config.get("INPUT_DATA_PATH") or config.get("SFT_INPUT_DATA_PATH")
    if not input_path:
        print("tokenize_sft: INPUT_DATA_PATH or SFT_INPUT_DATA_PATH is required", file=sys.stderr)
        return 2

    # Optional: copy raw SFT data into datapool if source is configured
    sft_raw_copy_src = config.get("SFT_RAW_COPY_SRC")
    input_abs_path = resolve_path(input_path, root_dir)
    if sft_raw_copy_src:
        needs_copy = False
        if not input_abs_path.exists():
            needs_copy = True
        elif input_abs_path.is_dir():
            if not any(input_abs_path.glob("*.jsonl")):
                needs_copy = True
        if needs_copy:
            print(f"tokenize_sft: copying raw SFT from {sft_raw_copy_src} -> {input_abs_path}")
            if not dry_run:
                sys.path.insert(0, str(root_dir / "scripts"))
                from prepare_exp import copy_jsonl_flat
                copy_jsonl_flat(Path(sft_raw_copy_src), input_abs_path)
    
    tokenizer_path = require_config(config, "TOKENIZER_PATH", "tokenize_sft")
    output_prefix = config.get("OUTPUT_PREFIX") or config.get("SFT_OUTPUT_PREFIX")
    if not output_prefix:
        print("tokenize_sft: OUTPUT_PREFIX or SFT_OUTPUT_PREFIX is required", file=sys.stderr)
        return 2
    
    megatron_dir = config.get("MEGATRON")
    mindspeed_dir = config.get("MINDSPEED")
    preprocess_dir_str = megatron_dir or mindspeed_dir
    if not preprocess_dir_str:
        print("tokenize_sft: set MEGATRON or MINDSPEED in step config", file=sys.stderr)
        return 2
    preprocess_dir = require_path_exists(preprocess_dir_str, root_dir, "tokenize_sft")
    if megatron_dir:
        preprocess_script_abs = preprocess_dir / "tools" / "preprocess_data.py"
    else:
        preprocess_script_abs = preprocess_dir / "preprocess_data.py"
    if not preprocess_script_abs.exists():
        print(f"tokenize_sft: preprocess_data.py not found: {preprocess_script_abs}", file=sys.stderr)
        return 2
    preprocess_workdir_abs = preprocess_dir
    
    # Optional config with defaults
    workers = int(config.get("WORKERS", "16"))
    partitions = int(config.get("PARTITIONS", "16"))
    log_interval = int(config.get("LOG_INTERVAL", "100000"))
    json_keys = config.get("JSON_KEYS") or config.get("SFT_JSON_KEYS", "instruction input output")
    tokenizer_type = config.get("TOKENIZER_TYPE", "HuggingFaceTokenizer")
    tokenizer_vocab_file = config.get("TOKENIZER_VOCAB_FILE")
    merge_jsonl = str(config.get("MERGE_JSONL", "1")) == "1"
    shuffle_jsonl = str(config.get("SHUFFLE_JSONL", "0")) == "1"
    shuffle_seed = config.get("SHUFFLE_SEED")
    shuffle_buffer = int(config.get("SHUFFLE_BUFFER", "10000"))
    
    print("tokenize_sft: starting")
    
    # Resolve paths
    tokenizer_path_abs = resolve_path(tokenizer_path, root_dir)
    output_prefix_abs = resolve_path(output_prefix, root_dir)
    input_dir_abs = resolve_path(input_path, root_dir)
    # merged_input.jsonl lives under raw/sft (same as prepare_exp) so it is not cleared with tokenized/sft
    merge_output = (input_dir_abs / "merged_input.jsonl") if input_dir_abs.is_dir() else (input_dir_abs.parent / "merged_input.jsonl")
    
    # Extract required keys from JSON_KEYS for filtering during merge.
    # If we are rewriting to input/label/text, allow mixed raw formats and skip filtering here.
    rewrite_input_label = str(config.get("REWRITE_INPUT_LABEL", "0")) == "1"
    if rewrite_input_label:
        required_keys = None
    else:
        if isinstance(json_keys, str):
            required_keys = json_keys.split()
        else:
            required_keys = json_keys if isinstance(json_keys, list) else None
    
    # In dry-run, skip merge to avoid heavy I/O on huge datasets
    if dry_run:
        input_abs = str(resolve_path(input_path, root_dir))
    else:
        if merge_jsonl:
            merged_ready = merge_output.exists()
            if merged_ready:
                input_abs = str(merge_output)
            else:
                input_abs = ""
        else:
            input_abs = ""
        try:
            if not input_abs:
                input_file_path = expand_input_pattern(
                    input_path,
                    root_dir,
                    merge_files=merge_jsonl,
                    merge_output=merge_output,
                    required_json_keys=required_keys,
                    shuffle=shuffle_jsonl,
                    shuffle_seed=int(shuffle_seed) if shuffle_seed else None,
                    shuffle_buffer=shuffle_buffer,
                )
                input_abs = str(resolve_path(str(input_file_path), root_dir))
        except (FileNotFoundError, ValueError) as e:
            print(f"tokenize_sft: {e}", file=sys.stderr)
            # Check if SFT_RAW_COPY_SRC is configured
            sft_raw_copy_src = config.get("SFT_RAW_COPY_SRC")
            if sft_raw_copy_src:
                print(f"tokenize_sft: Hint: Run 'python scripts/prepare_exp.py -c {step_env_path.parent.parent}/pipeline.env' to copy data from {sft_raw_copy_src}", file=sys.stderr)
            else:
                print(f"tokenize_sft: Hint: Configure SFT_RAW_COPY_SRC in config and run 'prepare_exp' to copy data", file=sys.stderr)
            return 2

    # Optional: rewrite raw SFT into input/label format
    if rewrite_input_label:
        prompt_template = config.get("PROMPT_TEMPLATE", "### Instruction:\n{instruction}\n")
        input_template = config.get("PROMPT_INPUT_TEMPLATE", "### Input:\n{input}\n")
        response_prefix = config.get("PROMPT_RESPONSE_PREFIX", "### Response:\n")
        rewrite_output = config.get("REWRITE_OUTPUT_FILE")
        if rewrite_output:
            rewrite_output_abs = resolve_path(rewrite_output, root_dir)
        else:
            rewrite_output_abs = Path(input_abs).parent / "sft_input_label.jsonl"
        print(f"tokenize_sft: rewriting input/label -> {rewrite_output_abs}")
        if not dry_run:
            rewrite_sft_jsonl_to_input_label(
                Path(input_abs),
                rewrite_output_abs,
                prompt_template,
                input_template,
                response_prefix,
            )
        input_abs = str(rewrite_output_abs)
    
    # Validate paths are under datapool (unless allowed)
    if not allow_external_paths:
        datapool_abs = datapool_root.resolve()
        input_check = str(input_abs) if isinstance(input_abs, Path) else input_abs
        if not input_check.startswith(str(datapool_abs) + "/"):
            print(
                f"tokenize_sft: INPUT_DATA_PATH must be under DATAPOOL_ROOT ({datapool_abs}) but got: {input_check}",
                file=sys.stderr,
            )
            print("tokenize_sft: set ALLOW_EXTERNAL_PATHS=1 in pipeline.env to override", file=sys.stderr)
            return 2
        
        if not str(output_prefix_abs).startswith(str(datapool_abs) + "/"):
            print(
                f"tokenize_sft: OUTPUT_PREFIX must be under DATAPOOL_ROOT ({datapool_abs}) but got: {output_prefix_abs}",
                file=sys.stderr,
            )
            print("tokenize_sft: set ALLOW_EXTERNAL_PATHS=1 in pipeline.env to override", file=sys.stderr)
            return 2
    
    # Build command
    cmd = [
        "python",
        str(preprocess_script_abs),
        "--input",
        str(input_abs),
        "--output-prefix",
        str(output_prefix_abs),
        "--json-keys",
    ]
    # JSON_KEYS can be space-separated string or list
    if isinstance(json_keys, str):
        cmd.extend(json_keys.split())
    else:
        cmd.extend(json_keys)
    cmd.extend([
        "--tokenizer-type",
        tokenizer_type,
        "--tokenizer-model",
        str(tokenizer_path_abs),
        "--append-eod",
        "--workers",
        str(workers),
        "--partitions",
        str(partitions),
        "--log-interval",
        str(log_interval),
    ])
    
    if tokenizer_vocab_file:
        tokenizer_vocab_file_abs = resolve_path(tokenizer_vocab_file, root_dir)
        cmd.extend(["--vocab-file", str(tokenizer_vocab_file_abs)])
    
    print(f"tokenize_sft: preprocess_workdir={preprocess_workdir_abs}")
    print(f"tokenize_sft: preprocess_script={preprocess_script_abs}")
    print(f"tokenize_sft: input={input_abs}")
    print(f"tokenize_sft: output_prefix={output_prefix_abs}")
    print(f"tokenize_sft: json_keys={json_keys}")
    
    # Create output directory
    output_prefix_abs.parent.mkdir(parents=True, exist_ok=True)
    
    # Execute
    if dry_run:
        print(f"[dry-run] tokenize_sft: (cd {preprocess_workdir_abs} && {' '.join(cmd)})")
        return 0
    
    try:
        subprocess.run(cmd, cwd=preprocess_workdir_abs, check=True)
    except subprocess.CalledProcessError as e:
        print(f"tokenize_sft: failed with exit code {e.returncode}", file=sys.stderr)
        return e.returncode
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
