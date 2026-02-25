#!/usr/bin/env python3
"""
Objective scorer - compares predictions against ground truth.
No AI involved. Just array comparison.

Usage: python score.py
Output: results/scores.csv
"""

import json
import csv
import os
from pathlib import Path
from datetime import datetime

REPO = Path.home() / "sybil-arc-challenge"
EVAL_DIR = REPO / "data" / "data" / "evaluation"
PREDICTIONS_DIR = REPO / "results" / "predictions"
SCORES_FILE = REPO / "results" / "scores.csv"

def load_task_expected(task_id: str) -> list:
    """Load expected test outputs from evaluation data."""
    task_file = EVAL_DIR / f"{task_id}.json"
    if not task_file.exists():
        return None
    with open(task_file) as f:
        data = json.load(f)
    return [t["output"] for t in data["test"]]

def load_prediction(task_id: str) -> list:
    """Load our prediction for a task."""
    pred_file = PREDICTIONS_DIR / f"{task_id}.json"
    if not pred_file.exists():
        return None
    with open(pred_file) as f:
        return json.load(f)

def arrays_match(a, b) -> bool:
    """Exact pixel-by-pixel comparison."""
    if a is None or b is None:
        return False
    try:
        import numpy as np
        return np.array_equal(np.array(a), np.array(b))
    except:
        return a == b

def score_all():
    """Score all predictions against ground truth."""
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    SCORES_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    results = []
    total = 0
    correct = 0
    
    # Get all tasks we have predictions for
    pred_files = list(PREDICTIONS_DIR.glob("*.json"))
    
    for pred_file in sorted(pred_files):
        task_id = pred_file.stem
        expected = load_task_expected(task_id)
        predicted = load_prediction(task_id)
        
        if expected is None:
            status = "NO_GROUND_TRUTH"
        elif predicted is None:
            status = "NO_PREDICTION"
        else:
            # Handle different prediction formats:
            # - [grid] = list of test outputs (correct format)
            # - grid = single grid (needs wrapping)
            # - [[grid]] = nested (needs unwrapping)
            
            # Normalize predicted to list of grids
            if isinstance(predicted, list) and len(predicted) > 0:
                if isinstance(predicted[0], list) and len(predicted[0]) > 0:
                    if isinstance(predicted[0][0], list):
                        # [[grid]] format - already list of grids
                        pred_grids = predicted
                    else:
                        # [row, row, ...] format - single grid, wrap it
                        pred_grids = [predicted]
                else:
                    pred_grids = predicted
            else:
                pred_grids = [predicted]
            
            all_match = True
            for i, exp in enumerate(expected):
                pred = pred_grids[i] if i < len(pred_grids) else None
                if not arrays_match(exp, pred):
                    all_match = False
                    break
            
            status = "CORRECT" if all_match else "WRONG"
            total += 1
            if all_match:
                correct += 1
        
        results.append({
            "task_id": task_id,
            "status": status,
            "timestamp": datetime.now().isoformat()
        })
    
    # Write CSV
    with open(SCORES_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["task_id", "status", "timestamp"])
        writer.writeheader()
        writer.writerows(results)
    
    # Summary
    print(f"\n=== SCORING RESULTS ===")
    print(f"Predictions found: {len(pred_files)}")
    print(f"Scored: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {correct/total*100:.2f}%" if total > 0 else "N/A")
    print(f"\nResults saved to: {SCORES_FILE}")

if __name__ == "__main__":
    score_all()
