#!/usr/bin/env python3
"""
Utilities for tokenization steps, including JSONL merging.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import List


def merge_jsonl_files(input_files: List[Path], output_file: Path, required_keys: List[str] | None = None) -> int:
    """
    Merge multiple JSONL files into a single JSONL file.
    Validates JSON and optionally filters by required keys.
    
    Args:
        input_files: List of input JSONL file paths
        output_file: Output JSONL file path
        required_keys: Optional list of keys that must be present in each JSON object
        
    Returns:
        Total number of lines written
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    total_lines = 0
    skipped_lines = 0
    sys_module = __import__("sys")
    
    with open(output_file, "w", encoding="utf-8") as out_f:
        for input_file in sorted(input_files):
            if not input_file.exists():
                raise FileNotFoundError(f"Input file not found: {input_file}")
            
            with open(input_file, "r", encoding="utf-8") as in_f:
                for line_num, line in enumerate(in_f, start=1):
                    line = line.strip()
                    if not line:  # Skip empty lines
                        continue
                    
                    # Validate JSON
                    try:
                        json_obj = json.loads(line)
                        # Ensure it's a dict (not a list or other type)
                        if not isinstance(json_obj, dict):
                            print(f"Warning: Skipping non-dict JSON at {input_file}:{line_num}", file=sys_module.stderr)
                            skipped_lines += 1
                            continue
                        
                        # Check required keys if specified
                        if required_keys:
                            missing_keys = [key for key in required_keys if key not in json_obj]
                            if missing_keys:
                                print(f"Warning: Skipping line at {input_file}:{line_num} (missing keys: {missing_keys})", file=sys_module.stderr)
                                skipped_lines += 1
                                continue
                        
                        out_f.write(line + "\n")
                        total_lines += 1
                    except json.JSONDecodeError as e:
                        print(f"Warning: Invalid JSON at {input_file}:{line_num}: {e}, skipping", file=sys_module.stderr)
                        skipped_lines += 1
                        continue
    
    if skipped_lines > 0:
        print(f"merge_jsonl_files: Merged {total_lines} lines, skipped {skipped_lines} invalid/mismatched lines", file=sys_module.stderr)
    
    if total_lines == 0:
        raise ValueError(f"No valid lines found after merging {len(input_files)} files")
    
    return total_lines


def expand_input_pattern(
    input_path: str,
    root_dir: Path,
    merge_files: bool = True,
    merge_output: Path | None = None,
    required_json_keys: List[str] | None = None,
) -> Path:
    """
    Expand input path (directory or single file) and merge into a single file.
    
    Args:
        input_path: Directory path or single file path (glob patterns are not supported)
        root_dir: Root directory for resolving relative paths
        merge_files: If True (default), merge multiple files into one
        merge_output: Path to write merged file (if None, uses default location)
        required_json_keys: Optional list of keys that must be present in each JSON object
        
    Returns:
        Path to the input file (single file or merged file)
        
    Raises:
        ValueError: If input_path contains glob characters
        FileNotFoundError: If input path does not exist
    """
    # Check for glob characters and reject them
    if any(c in input_path for c in "*?["):
        raise ValueError(
            f"Glob patterns are not supported. Got: {input_path}\n"
            f"Please specify a directory path or a single file path instead."
        )
    
    # Resolve to absolute path if relative
    if not Path(input_path).is_absolute():
        input_path = str(root_dir / input_path)
    
    path = Path(input_path)
    
    if not path.exists():
        raise FileNotFoundError(
            f"Input path does not exist: {path}\n"
            f"Hint: You may need to run 'prepare_exp' first to copy data"
        )
    
    if path.is_dir():
        # Directory: find all .jsonl files, but exclude partition files (e.g., *_0.jsonl, *_1.jsonl)
        all_jsonl = sorted(path.glob("*.jsonl"))
        # Filter out partition files: files matching pattern *_<number>.jsonl (but not __<number>.jsonl)
        # Use negative lookbehind to ensure _ is not preceded by another _
        import re
        partition_pattern = re.compile(r'(?<!_)_\d+\.jsonl$')
        jsonl_files = [f for f in all_jsonl if not partition_pattern.search(f.name)]
        if not jsonl_files:
            raise FileNotFoundError(f"No .jsonl files found in directory: {path} (excluding partition files)")
    else:
        # Single file path
        if path.suffix != ".jsonl":
            raise ValueError(f"Input file must be a .jsonl file, got: {path}")
        jsonl_files = [path]
    
    if not jsonl_files:
        raise FileNotFoundError(f"No files match pattern: {input_pattern}")
    
    # If required_json_keys is specified, we need to filter/merge even for a single file
    # to ensure all lines have the required keys
    if required_json_keys is not None or len(jsonl_files) > 1:
        # Need to merge/filter
        if merge_files:
            if merge_output is None:
                # Default: merge to first file's directory with "merged.jsonl" name
                merge_output = jsonl_files[0].parent / "merged.jsonl"
            merge_jsonl_files(jsonl_files, merge_output, required_keys=required_json_keys)
            return merge_output
        else:
            # If merge_files=False but required_keys specified, still need to filter
            if required_json_keys is not None:
                if merge_output is None:
                    merge_output = jsonl_files[0].parent / "merged.jsonl"
                merge_jsonl_files(jsonl_files, merge_output, required_keys=required_json_keys)
                return merge_output
            # If merge_files=False and no required_keys, return first file
            return jsonl_files[0]
    else:
        # Single file, no required keys, no merge needed: return it directly
        return jsonl_files[0]
