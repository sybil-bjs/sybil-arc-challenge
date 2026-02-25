import json
import subprocess
from pathlib import Path

TASKS_DIR = Path("arc_challenge/tasks")

def get_blind_task(task_id):
    task_file = TASKS_DIR / f"{task_id}.json"
    with open(task_file, 'r') as f:
        full_task = json.load(f)
    return {
        "train": full_task["train"],
        "test": [{"input": t["input"]} for t in full_task["test"]]
    }

def get_prompt(task_id):
    blind_task = get_blind_task(task_id)
    return f"""
I am an ARC-AGI solver agent. Your goal is to solve the following ARC puzzle.
Task Data: {json.dumps(blind_task)}

Instructions:
1. Analyze the 'train' input/output pairs to find the transformation logic.
2. Write a Python script 'transform.py' with a function 'transform(grid: list[list[int]]) -> list[list[int]]'.
3. Use the 'exec' tool to run a test script that validates your 'transform' function against the 'train' examples.
4. If it fails, refine your logic and try again (up to 5 times).
5. Once it passes all training examples, apply it to the 'test' input.
6. Return the final grid for the test input as a JSON array.

Important: You have access to a full Mac environment. Use python3, numpy, and any files you need.
    """

if __name__ == "__main__":
    # Just a helper for the main agent
    pass
