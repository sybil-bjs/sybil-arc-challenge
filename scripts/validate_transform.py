#!/usr/bin/env python3
"""
External validator for transform.py files.
Tests against all training examples — no agent self-reporting.

Usage: python validate_transform.py <task_id>
"""

import sys
import json
import importlib.util
import traceback
from pathlib import Path
import numpy as np

REPO = Path.home() / "sybil-arc-challenge"
DATA_DIR = REPO / "data" / "data"

def load_task(task_id: str) -> dict:
    """Load task from training or evaluation."""
    for subdir in ["training", "evaluation"]:
        task_file = DATA_DIR / subdir / f"{task_id}.json"
        if task_file.exists():
            with open(task_file) as f:
                return json.load(f)
    raise FileNotFoundError(f"Task {task_id} not found")

def load_transform(transform_path: Path):
    """Load transform function from file."""
    spec = importlib.util.spec_from_file_location("transform", str(transform_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {transform_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "transform"):
        raise AttributeError("No 'transform' function found")
    return mod.transform

def validate(task_id: str, transform_path: Path) -> dict:
    """Validate transform against all training examples."""
    task = load_task(task_id)
    
    try:
        transform_fn = load_transform(transform_path)
    except Exception as e:
        return {
            "status": "LOAD_ERROR",
            "error": str(e),
            "traceback": traceback.format_exc()
        }
    
    results = []
    all_pass = True
    
    for i, ex in enumerate(task["train"]):
        inp = np.array(ex["input"], dtype=int)
        expected = np.array(ex["output"], dtype=int)
        
        try:
            result = transform_fn(inp.copy())
            result = np.array(result, dtype=int)
            
            if np.array_equal(result, expected):
                results.append({"example": i, "status": "PASS"})
            else:
                all_pass = False
                # Find first difference
                diff_count = np.sum(result != expected)
                results.append({
                    "example": i,
                    "status": "WRONG",
                    "diff_pixels": int(diff_count),
                    "expected_shape": list(expected.shape),
                    "got_shape": list(result.shape)
                })
        except Exception as e:
            all_pass = False
            results.append({
                "example": i,
                "status": "ERROR",
                "error": str(e)
            })
    
    return {
        "status": "PASS" if all_pass else "FAIL",
        "examples": results,
        "pass_count": sum(1 for r in results if r["status"] == "PASS"),
        "total": len(results)
    }

def apply_to_test(task_id: str, transform_path: Path) -> list:
    """Apply transform to test input(s) and return predictions."""
    task = load_task(task_id)
    transform_fn = load_transform(transform_path)
    
    predictions = []
    for test_case in task["test"]:
        inp = np.array(test_case["input"], dtype=int)
        result = transform_fn(inp.copy())
        predictions.append(np.array(result, dtype=int).tolist())
    
    return predictions

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python validate_transform.py <task_id> [transform_path]")
        sys.exit(1)
    
    task_id = sys.argv[1]
    transform_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(f"transform_{task_id}.py")
    
    result = validate(task_id, transform_path)
    print(json.dumps(result, indent=2))
    
    if result["status"] == "PASS":
        print("\n✅ All training examples pass!")
        print("Applying to test...")
        predictions = apply_to_test(task_id, transform_path)
        pred_file = REPO / "results" / "predictions" / f"{task_id}.json"
        pred_file.parent.mkdir(parents=True, exist_ok=True)
        with open(pred_file, "w") as f:
            json.dump(predictions, f)
        print(f"Saved predictions to {pred_file}")
    else:
        print(f"\n❌ {result['pass_count']}/{result['total']} training examples pass")
        sys.exit(1)
