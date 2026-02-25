#!/usr/bin/env python3
"""
Analyze a task and suggest relevant patterns from the knowledge base.

Usage:
  python suggest_pattern.py <task_id>
  
This extracts features from the task and queries the pattern DB.
"""

import json
import sys
from pathlib import Path
import numpy as np

REPO = Path.home() / "sybil-arc-challenge"
DATA_DIR = REPO / "data" / "data"

# Import pattern_db functions
sys.path.insert(0, str(REPO / "scripts"))
from pattern_db import search_patterns, search_by_features, get_db

def load_task(task_id: str) -> dict:
    """Load task from training or evaluation."""
    for subdir in ["training", "evaluation"]:
        task_file = DATA_DIR / subdir / f"{task_id}.json"
        if task_file.exists():
            with open(task_file) as f:
                return json.load(f)
    raise FileNotFoundError(f"Task {task_id} not found")

def extract_features(task: dict) -> dict:
    """Extract visual features from a task for pattern matching."""
    features = {}
    
    # Analyze first training example
    inp = np.array(task['train'][0]['input'])
    out = np.array(task['train'][0]['output'])
    
    # Color analysis
    input_colors = set(np.unique(inp)) - {0}
    output_colors = set(np.unique(out)) - {0}
    
    # Check for markers (isolated single pixels of a color)
    for color in input_colors:
        mask = (inp == color)
        if np.sum(mask) <= 3:  # Few pixels of this color
            if color == 2:
                features["has_marker"] = "red"
            elif color == 7:
                features["has_marker"] = "orange"
    
    # Check for borders (color 2 forming rectangles)
    if 2 in input_colors:
        features["has_border"] = "yes"
    
    # Size change
    if inp.shape != out.shape:
        if out.shape[0] < inp.shape[0]:
            features["transform"] = "shrink"
        else:
            features["transform"] = "grow"
    
    # Check for quadrant patterns
    if len(output_colors) == 4:
        features["quadrant_based"] = "yes"
    
    # Check for reflection/symmetry in output
    if np.array_equal(out, np.fliplr(out)) or np.array_equal(out, np.flipud(out)):
        features["has_symmetry"] = "yes"
    
    return features

def suggest_patterns(task_id: str):
    """Analyze task and suggest matching patterns."""
    task = load_task(task_id)
    features = extract_features(task)
    
    print(f"\n=== Task Analysis: {task_id} ===")
    print(f"Detected features: {features}")
    
    # Search by features
    if features:
        print(f"\n--- Patterns matching features ---")
        results = search_by_features(features)
        if results:
            for r in results:
                rate = f"{r['successes']}/{r['total_uses']}" if r['total_uses'] > 0 else "untested"
                print(f"  • {r['name']} ({rate}) — matched {r['match_count']} features")
        else:
            print("  No feature matches found")
    
    # Also do keyword search based on features
    keywords = " ".join(features.keys()) + " " + " ".join(str(v) for v in features.values())
    print(f"\n--- Keyword search: '{keywords[:50]}...' ---")
    results = search_patterns(keywords)
    if results:
        for r in results:
            rate = f"{r['successes']}/{r['total_uses']}" if r['total_uses'] > 0 else "untested"
            print(f"  • {r['name']} ({rate})")
    else:
        print("  No keyword matches found")
    
    return features

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python suggest_pattern.py <task_id>")
        sys.exit(1)
    
    suggest_patterns(sys.argv[1])
