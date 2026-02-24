#!/usr/bin/env python3
"""
Utilities for tokenization steps, including JSONL merging.
"""
from __future__ import annotations

import random
import shutil
import sys
from pathlib import Path
from typing import List, Tuple
import json

# Valid single-char escapes after \ in JSON strings (RFC 8259)
_JSON_ESCAPE_CHARS = frozenset('"\\/bfnrt')
_HEX_CHARS = frozenset('0123456789abcdefABCDEF')
# Control chars that must be escaped in JSON strings; when we emit after \\ we use this
_JSON_CONTROL_ESCAPES = {
    "\n": "\\n",
    "\r": "\\r",
    "\t": "\\t",
    "\b": "\\b",
    "\f": "\\f",
}


def _fix_invalid_json_escapes(line: str) -> str:
    """
    Fix invalid escape sequences inside JSON string values only.
    Valid escapes after \\ are: \" \\ \\/ \\b \\f \\n \\r \\t \\uXXXX.
    Any \\ followed by other char (e.g. \\s, \\1) is invalid; we escape the backslash (\\ -> \\\\).
    """
    result: List[str] = []
    i = 0
    n = len(line)
    in_string = False

    while i < n:
        c = line[i]
        if not in_string:
            result.append(c)
            if c == '"':
                in_string = True
            i += 1
            continue

        # Inside a double-quoted string
        if c != '\\':
            result.append(c)
            if c == '"':
                in_string = False
            i += 1
            continue

        # We saw \ inside a string (no next char: trailing backslash)
        if i + 1 >= n:
            result.append('\\')
            result.append('\\')
            i += 1
            continue
        next_c = line[i + 1]
        if next_c in _JSON_ESCAPE_CHARS:
            result.append(c)
            result.append(next_c)
            i += 2
            continue
        if next_c == 'u':
            # \uXXXX - need 4 hex digits; valid one: copy as-is
            if i + 5 <= n and all(line[i + 2 + k] in _HEX_CHARS for k in range(4)):
                result.append(line[i : i + 6])
                i += 6
                continue
            # invalid \u: pass through unchanged so parse still fails and user sees error
            num_tail = min(4, n - (i + 2))
            result.append(line[i : i + 2 + num_tail])
            i += 2 + num_tail
            continue
        # Invalid escape: \X -> \\X (control chars must be JSON-escaped in output)
        result.append('\\')
        result.append('\\')
        if next_c in _JSON_CONTROL_ESCAPES:
            result.append(_JSON_CONTROL_ESCAPES[next_c])
        elif ord(next_c) < 0x20:
            result.append(f"\\u{ord(next_c):04x}")
        else:
            result.append(next_c)
        i += 2

    return "".join(result)


def _normalize_jsonl_line(line: str) -> str:
    """
    Return a line that is valid JSON. If the line already parses, return as-is.
    If it fails due to invalid escape, fix and return; otherwise re-raise.
    """
    try:
        json.loads(line)
        return line
    except json.JSONDecodeError as e:
        if "escape" not in e.msg.lower():
            raise
    fixed = _fix_invalid_json_escapes(line)
    try:
        json.loads(fixed)
        return fixed
    except json.JSONDecodeError as e2:
        raise json.JSONDecodeError(
            f"Invalid escape fix failed: {e2.msg}",
            e2.doc,
            e2.pos,
        ) from e2


def merge_jsonl_files(
    input_files: List[Path],
    output_file: Path,
    required_keys: List[str] | None = None,
    *,
    shuffle: bool = False,
    shuffle_seed: int | None = None,
    shuffle_buffer: int = 10000,
) -> int:
    """
    Merge multiple JSONL files into a single JSONL file.
    Does not validate JSON or filter keys.
    
    Args:
        input_files: List of input JSONL file paths
        output_file: Output JSONL file path
        required_keys: Ignored (kept for compatibility)
        
    Returns:
        Total number of lines written
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    total_lines = 0
    skipped_lines = 0
    
    rng = random.Random(shuffle_seed) if shuffle else None
    buffer: List[str] = []

    def flush_buffer(out_f) -> None:
        if not buffer:
            return
        if rng is not None:
            rng.shuffle(buffer)
        out_f.write("".join(buffer))
        buffer.clear()

    ordered_files = sorted(input_files)
    if rng is not None:
        rng.shuffle(ordered_files)
    with open(output_file, "w", encoding="utf-8") as out_f:
        for input_file in ordered_files:
            if not input_file.exists():
                raise FileNotFoundError(f"Input file not found: {input_file}")
            
            with open(input_file, "r", encoding="utf-8") as in_f:
                for line_num, line in enumerate(in_f, start=1):
                    line = line.strip()
                    if not line:  # Skip empty lines
                        continue
                    try:
                        line = _normalize_jsonl_line(line)
                    except json.JSONDecodeError as e:
                        raise ValueError(
                            f"Invalid JSON in {input_file} line {line_num}: {e.msg}"
                        ) from e
                    if rng is None:
                        out_f.write(line + "\n")
                    else:
                        buffer.append(line + "\n")
                        if len(buffer) >= shuffle_buffer:
                            flush_buffer(out_f)
                    total_lines += 1
        if rng is not None:
            flush_buffer(out_f)

    if total_lines == 0:
        raise ValueError(f"No valid lines found after merging {len(input_files)} files")
    
    return total_lines


def rewrite_sft_jsonl_to_input_label(
    input_file: Path,
    output_file: Path,
    prompt_template: str,
    input_template: str,
    response_prefix: str,
) -> Tuple[int, int]:
    """
    Rewrite SFT jsonl into input/label (+text) format.

    Returns:
        (written_lines, skipped_lines)
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)

    def to_text(value) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=False)
        return str(value)

    def build_input_label(record: dict) -> Tuple[str, str] | None:
        # Already in input/label format
        if "input" in record and "label" in record:
            return to_text(record.get("input")), to_text(record.get("label"))

        # Instruction-style format
        if "instruction" in record and "output" in record:
            instruction = to_text(record.get("instruction")).strip()
            extra_input = to_text(record.get("input")).strip()
            prompt = prompt_template.format(instruction=instruction)
            if extra_input:
                prompt += input_template.format(input=extra_input)
            prompt += response_prefix
            return prompt, to_text(record.get("output"))

        # Prompt/response format
        if "prompt" in record and ("response" in record or "completion" in record):
            response = record.get("response")
            if response is None:
                response = record.get("completion")
            return to_text(record.get("prompt")), to_text(response)

        # Fallback: single text as label
        if "text" in record:
            return "", to_text(record.get("text"))

        return None

    written = 0
    skipped = 0
    with open(input_file, "r", encoding="utf-8") as in_f, open(
        output_file, "w", encoding="utf-8"
    ) as out_f:
        for line_num, line in enumerate(in_f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                print(
                    f"rewrite_sft_jsonl_to_input_label: invalid JSON at {input_file}:{line_num}: {exc}",
                    file=sys.stderr,
                )
                skipped += 1
                continue
            if not isinstance(record, dict):
                print(
                    f"rewrite_sft_jsonl_to_input_label: non-dict JSON at {input_file}:{line_num}",
                    file=sys.stderr,
                )
                skipped += 1
                continue
            pair = build_input_label(record)
            if not pair:
                skipped += 1
                continue
            prompt, label = pair
            if not label.strip():
                skipped += 1
                continue
            text = f"{prompt}{label}"
            out = {"input": prompt, "label": label, "text": text}
            out_f.write(json.dumps(out, ensure_ascii=False) + "\n")
            written += 1

    if skipped:
        print(
            f"rewrite_sft_jsonl_to_input_label: wrote {written} lines, skipped {skipped} lines",
            file=sys.stderr,
        )

    if written == 0:
        raise ValueError(f"No valid lines written to {output_file}")

    return written, skipped


def expand_input_pattern(
    input_path: str,
    root_dir: Path,
    merge_files: bool = True,
    merge_output: Path | None = None,
    required_json_keys: List[str] | None = None,
    *,
    shuffle: bool = False,
    shuffle_seed: int | None = None,
    shuffle_buffer: int = 10000,
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
        # Directory: find all .jsonl files
        jsonl_files = sorted(path.glob("*.jsonl"))
        if not jsonl_files:
            raise FileNotFoundError(f"No .jsonl files found in directory: {path}")
    else:
        # Single file path
        if path.suffix != ".jsonl":
            raise ValueError(f"Input file must be a .jsonl file, got: {path}")
        jsonl_files = [path]
    
    if not jsonl_files:
        raise FileNotFoundError(f"No files match pattern: {input_path}")
    
    # Merge when multiple files are present (or if caller explicitly wants merge)
    if len(jsonl_files) > 1 or required_json_keys is not None:
        # Need to merge
        if not merge_files:
            raise ValueError(
                "MERGE_JSONL=0 is incompatible with multiple JSONL files. "
                "Please enable MERGE_JSONL or provide a single .jsonl file."
            )
        if merge_output is None:
            # Default: merge to first file's directory with "merged.jsonl" name
            merge_output = jsonl_files[0].parent / "merged.jsonl"
        merge_jsonl_files(
            jsonl_files,
            merge_output,
            required_keys=None,
            shuffle=shuffle,
            shuffle_seed=shuffle_seed,
            shuffle_buffer=shuffle_buffer,
        )
        return merge_output
    else:
        # Single file, no required keys, no merge needed: return it directly
        return jsonl_files[0]
