#!/usr/bin/env python3
"""
Cost tracking for ARC-AGI solving.

Pricing (per 1M tokens):
- Claude Sonnet 4: $3 input, $15 output
- Claude Opus 4: $15 input, $75 output  
- Gemini 2.5 Flash: $0.30 input, $2.50 output
- Gemini 2.5 Pro: $1.25 input, $10.00 output

Usage:
  python cost_tracker.py add <task_id> <model> <input_tokens> <output_tokens> [runtime_sec]
  python cost_tracker.py summary
  python cost_tracker.py export
"""

import sys
import json
import csv
from pathlib import Path
from datetime import datetime

REPO = Path.home() / "sybil-arc-challenge"
COSTS_FILE = REPO / "results" / "costs.json"

# Pricing per 1M tokens
PRICING = {
    "sonnet": {"input": 3.00, "output": 15.00},
    "opus": {"input": 15.00, "output": 75.00},
    "gemini-flash": {"input": 0.30, "output": 2.50},
    "gemini-pro": {"input": 1.25, "output": 10.00},
}

def load_costs() -> list:
    if COSTS_FILE.exists():
        with open(COSTS_FILE) as f:
            return json.load(f)
    return []

def save_costs(costs: list):
    COSTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(COSTS_FILE, "w") as f:
        json.dump(costs, f, indent=2)

def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    pricing = PRICING.get(model.lower(), PRICING["sonnet"])
    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]
    return input_cost + output_cost

def add_entry(task_id: str, model: str, input_tokens: int, output_tokens: int, runtime_sec: float = 0):
    costs = load_costs()
    cost = calculate_cost(model, input_tokens, output_tokens)
    
    entry = {
        "task_id": task_id,
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": round(cost, 4),
        "runtime_sec": runtime_sec,
        "timestamp": datetime.now().isoformat()
    }
    costs.append(entry)
    save_costs(costs)
    
    print(f"Added: {task_id} | {model} | {input_tokens:,} in / {output_tokens:,} out | ${cost:.4f}")
    return entry

def summary():
    costs = load_costs()
    if not costs:
        print("No cost data yet.")
        return
    
    total_cost = sum(c["cost_usd"] for c in costs)
    total_input = sum(c["input_tokens"] for c in costs)
    total_output = sum(c["output_tokens"] for c in costs)
    total_runtime = sum(c.get("runtime_sec", 0) for c in costs)
    
    print(f"\n=== COST SUMMARY ===")
    print(f"Tasks tracked: {len(costs)}")
    print(f"Total tokens: {total_input:,} input / {total_output:,} output")
    print(f"Total cost: ${total_cost:.4f}")
    print(f"Total runtime: {total_runtime:.0f}s ({total_runtime/60:.1f}m)")
    print(f"Avg cost/task: ${total_cost/len(costs):.4f}")
    
    # By model
    by_model = {}
    for c in costs:
        m = c["model"]
        if m not in by_model:
            by_model[m] = {"count": 0, "cost": 0}
        by_model[m]["count"] += 1
        by_model[m]["cost"] += c["cost_usd"]
    
    print(f"\nBy model:")
    for m, data in by_model.items():
        print(f"  {m}: {data['count']} tasks, ${data['cost']:.4f}")

def export_csv():
    costs = load_costs()
    csv_file = REPO / "results" / "costs.csv"
    
    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["task_id", "model", "input_tokens", "output_tokens", "cost_usd", "runtime_sec", "timestamp"])
        writer.writeheader()
        writer.writerows(costs)
    
    print(f"Exported to {csv_file}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python cost_tracker.py <add|summary|export> [args...]")
        sys.exit(1)
    
    cmd = sys.argv[1]
    
    if cmd == "add" and len(sys.argv) >= 5:
        task_id = sys.argv[2]
        model = sys.argv[3]
        input_tokens = int(sys.argv[4].replace(",", "").replace("k", "000"))
        output_tokens = int(sys.argv[5].replace(",", "").replace("k", "000"))
        runtime = float(sys.argv[6]) if len(sys.argv) > 6 else 0
        add_entry(task_id, model, input_tokens, output_tokens, runtime)
    elif cmd == "summary":
        summary()
    elif cmd == "export":
        export_csv()
    else:
        print("Usage: python cost_tracker.py <add|summary|export> [args...]")
