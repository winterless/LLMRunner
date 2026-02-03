#!/usr/bin/env python3
"""
Configuration loading utilities for LLMRunner steps.
Replaces .env file parsing with Python config modules.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict


def load_config_module(config_path: Path) -> Dict[str, Any]:
    """
    Load a Python config file as a module and return its variables.
    
    The config file should define variables directly (not in a dict).
    Example:
        INPUT_DATA_PATH = "${DATAPOOL_ROOT}/data/raw/cpt"
        WORKERS = 32
    
    Variables are returned as a dict with string values (for compatibility
    with existing code that expects env-like dicts).
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    spec = importlib.util.spec_from_file_location("config", config_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Could not load config from: {config_path}")
    
    module = importlib.util.module_from_spec(spec)
    # Execute in a namespace that won't pollute globals
    sys.modules["config"] = module
    spec.loader.exec_module(module)
    
    # Extract all non-private variables (uppercase and lowercase)
    config: Dict[str, Any] = {}
    for name in dir(module):
        if name.startswith("_"):
            continue
        value = getattr(module, name)
        # Skip callables and modules
        if callable(value):
            continue
        if isinstance(value, type(importlib)):
            continue
        # Convert to string for compatibility with env-based code
        if isinstance(value, (str, int, float, bool)):
            config[name] = str(value)
        else:
            config[name] = value
    
    return config


def resolve_config_vars(config: Dict[str, Any], context: Dict[str, str]) -> Dict[str, str]:
    """
    Resolve ${VAR} substitutions in config values using context.
    
    Supports nested variable resolution (e.g., ${VAR1}/${VAR2}).
    Resolves in multiple passes to handle dependencies.
    
    Example:
        config = {"INPUT_DATA_PATH": "${DATAPOOL_ROOT}/data/raw"}
        context = {"DATAPOOL_ROOT": "/path/to/datapool"}
        result = {"INPUT_DATA_PATH": "/path/to/datapool/data/raw"}
    """
    resolved: Dict[str, str] = {}
    # First pass: copy all values
    for key, value in config.items():
        if isinstance(value, str):
            resolved[key] = value
        else:
            resolved[key] = str(value)
    
    # Multiple passes for nested variable resolution
    max_passes = 10
    for _ in range(max_passes):
        changed = False
        for key, value in resolved.items():
            if isinstance(value, str) and "${" in value:
                new_value = value
                for var_name, var_value in context.items():
                    if f"${{{var_name}}}" in new_value:
                        new_value = new_value.replace(f"${{{var_name}}}", var_value)
                        changed = True
                # Also resolve from already-resolved config values
                for var_name, var_value in resolved.items():
                    if var_name != key and f"${{{var_name}}}" in new_value:
                        new_value = new_value.replace(f"${{{var_name}}}", var_value)
                        changed = True
                resolved[key] = new_value
        if not changed:
            break
    
    return resolved


ENV_IMPORT_KEYS = [
    "DATAPOOL",
    "ROOT",
    "BASE_MODEL_SRC",
    "MINDSPEED",
    "MINDSPEED_LLM",
    "CPT_RAW_COPY_SRC",
    "SFT_RAW_COPY_SRC",
]


def apply_env_imports(context: Dict[str, str], environ: Dict[str, str]) -> None:
    """
    Import selected environment variables into context for ${VAR} expansion.
    """
    for key in ENV_IMPORT_KEYS:
        if key in environ:
            context[key] = environ[key]


def merge_env_defaults(config: Dict[str, Any], environ: Dict[str, str]) -> None:
    """
    Populate config with selected env vars (env overrides config).
    """
    for key in ENV_IMPORT_KEYS:
        if key in environ:
            config[key] = environ[key]


def require_config(config: Dict[str, Any], key: str, step_name: str = "") -> str:
    """
    Require a config key to exist and be non-empty.
    
    Args:
        config: Configuration dictionary
        key: Config key to check
        step_name: Optional step name for error message (e.g., "tokenize_cpt")
        
    Returns:
        The config value as string
        
    Raises:
        SystemExit: If key is missing or empty
    """
    value = config.get(key)
    if not value:
        prefix = f"{step_name}: " if step_name else ""
        print(f"{prefix}{key} is required", file=sys.stderr)
        raise SystemExit(2)
    return str(value)


def require_path_exists(path_str: str, root_dir: Path, step_name: str = "") -> Path:
    """
    Require a path to exist and resolve it to absolute path.
    
    Args:
        path_str: Path string (can be relative or absolute)
        root_dir: Root directory Path for resolving relative paths
        step_name: Optional step name for error message
        
    Returns:
        Resolved absolute Path
        
    Raises:
        SystemExit: If path doesn't exist
    """
    from pathlib import Path
    
    if not path_str:
        prefix = f"{step_name}: " if step_name else ""
        print(f"{prefix}Empty path provided", file=sys.stderr)
        raise SystemExit(2)
    
    path = Path(path_str)
    if path.is_absolute():
        resolved = path.resolve()
    else:
        resolved = (root_dir / path).resolve()
    
    if not resolved.exists():
        prefix = f"{step_name}: " if step_name else ""
        print(f"{prefix}Path does not exist: {resolved}", file=sys.stderr)
        raise SystemExit(2)
    
    return resolved
