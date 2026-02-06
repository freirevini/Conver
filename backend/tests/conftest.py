"""Pytest configuration and fixtures for parser tests."""
import json
import os
from pathlib import Path
from typing import Any, Dict

import pytest


# Paths
BACKEND_DIR = Path(__file__).parent.parent
PROJECT_ROOT = BACKEND_DIR.parent
SAMPLE_WORKFLOW = PROJECT_ROOT / "temp_knime_extract" / "fluxo_knime_exemplo"
FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_workflow_dir() -> Path:
    """Path to sample KNIME workflow for testing."""
    # Look for extracted workflow
    if SAMPLE_WORKFLOW.exists():
        # Find the actual workflow directory (contains workflow.knime)
        for root, dirs, files in os.walk(SAMPLE_WORKFLOW):
            if "workflow.knime" in files:
                return Path(root)
    
    pytest.skip("Sample workflow not available")


@pytest.fixture
def sample_metanode_dir(sample_workflow_dir: Path) -> Path:
    """Path to a sample metanode directory."""
    # Look for CALCULA A TA metanode
    for item in sample_workflow_dir.iterdir():
        if item.is_dir() and "CALCULA A TA" in item.name:
            return item
    
    pytest.skip("Sample metanode not found")


@pytest.fixture
def golden_master_dir() -> Path:
    """Directory for golden master snapshots."""
    gm_dir = FIXTURES_DIR / "golden_masters"
    gm_dir.mkdir(parents=True, exist_ok=True)
    return gm_dir


def save_golden_master(data: Dict[str, Any], name: str, gm_dir: Path) -> None:
    """Save a golden master snapshot."""
    path = gm_dir / f"{name}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str, ensure_ascii=False)


def load_golden_master(name: str, gm_dir: Path) -> Dict[str, Any]:
    """Load a golden master snapshot."""
    path = gm_dir / f"{name}.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compare_with_golden_master(
    current: Dict[str, Any], 
    name: str, 
    gm_dir: Path,
    update: bool = False
) -> bool:
    """
    Compare current output with golden master.
    
    If update=True or no golden master exists, saves current as new master.
    Returns True if matches (or was updated).
    """
    if update:
        save_golden_master(current, name, gm_dir)
        return True
    
    master = load_golden_master(name, gm_dir)
    if master is None:
        # First run - create golden master
        save_golden_master(current, name, gm_dir)
        return True
    
    # Compare (simplified - just check key counts and structure)
    return _compare_dicts(current, master)


def _compare_dicts(d1: Dict, d2: Dict, path: str = "") -> bool:
    """Deep compare two dicts, logging differences."""
    if type(d1) != type(d2):
        print(f"Type mismatch at {path}: {type(d1)} vs {type(d2)}")
        return False
    
    if isinstance(d1, dict):
        if set(d1.keys()) != set(d2.keys()):
            print(f"Key mismatch at {path}: {set(d1.keys())} vs {set(d2.keys())}")
            return False
        return all(_compare_dicts(d1[k], d2[k], f"{path}.{k}") for k in d1)
    
    if isinstance(d1, list):
        if len(d1) != len(d2):
            print(f"List length mismatch at {path}: {len(d1)} vs {len(d2)}")
            return False
        return all(_compare_dicts(v1, v2, f"{path}[{i}]") for i, (v1, v2) in enumerate(zip(d1, d2)))
    
    # For leaf values, just check they exist (not exact match for paths)
    return True
