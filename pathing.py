#!/usr/bin/env python3
"""
Simple path helper for environment-based directory resolution.
Centralizes DATA_DIR, CONFIG_DIR, STATIC_DIR handling.
"""
import os
from pathlib import Path

def get_path(path_type: str, *subpaths: str) -> str:
    """
    Get a path based on environment variables with sensible defaults.
    
    Args:
        path_type: One of 'data', 'config', 'static'
        *subpaths: Additional path components to join
        
    Returns:
        Full path as string
    """
    base_paths = {
        'data': os.getenv('DATA_DIR', './data'),
        'config': os.getenv('CONFIG_DIR', './config'), 
        'static': os.getenv('STATIC_DIR', './static')
    }
    
    if path_type not in base_paths:
        raise ValueError(f"Unknown path type: {path_type}. Must be one of: {list(base_paths.keys())}")
    
    base = Path(base_paths[path_type])
    return str(base.joinpath(*subpaths))

def ensure_dir(path_type: str, *subpaths: str) -> str:
    """
    Get a path and ensure the directory exists.
    
    Args:
        path_type: One of 'data', 'config', 'static'
        *subpaths: Additional path components to join
        
    Returns:
        Full path as string
    """
    path = get_path(path_type, *subpaths)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return path