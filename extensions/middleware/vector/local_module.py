"""
Helper module for setting up imports during testing.
This allows tests to run directly without needing to install the package.
"""
import os
import sys
from pathlib import Path

def add_parent_to_path():
    """Add the parent directory to sys.path if it's not already there"""
    parent_dir = str(Path(__file__).parent.absolute())
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
        print(f"Added {parent_dir} to sys.path")
    return parent_dir 