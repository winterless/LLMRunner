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
        INPUT_DIR = "${DATAPOOL_ROOT}/data/raw/cpt"
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
    
    # Extract all uppercase variables (convention for config)
    config: Dict[str, Any] = {}
    for name in dir(module):
        if name.isupper() and not name.startswith("_"):
            value = getattr(module, name)
            # Convert to string for compatibility with env-based code
            if isinstance(value, (str, int, float, bool)):
                config[name] = str(value)
            else:
                config[name] = value
    
    return config


def resolve_config_vars(config: Dict[str, Any], context: Dict[str, str]) -> Dict[str, str]:
    """
    Resolve ${VAR} substitutions in config values using context.
    
    Example:
        config = {"INPUT_DIR": "${DATAPOOL_ROOT}/data/raw"}
        context = {"DATAPOOL_ROOT": "/path/to/datapool"}
        result = {"INPUT_DIR": "/path/to/datapool/data/raw"}
    """
    resolved: Dict[str, str] = {}
    for key, value in config.items():
        if isinstance(value, str):
            # Simple ${VAR} substitution
            resolved_value = value
            for var_name, var_value in context.items():
                resolved_value = resolved_value.replace(f"${{{var_name}}}", var_value)
            resolved[key] = resolved_value
        else:
            resolved[key] = str(value)
    return resolved


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
