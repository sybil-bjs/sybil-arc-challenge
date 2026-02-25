import os
import json
import subprocess
import time
from pathlib import Path

# Use the actual tool name for internal consistency if possible, 
# but for background execution through the CLI is more robust for long runs.

# Configuration
TASKS_DIR = Path("arc_challenge/tasks")
RESULTS_DIR = Path("arc_challenge/results")
LOGS_DIR = Path("arc_challenge/logs")
GITHUB_RAW_BASE = "https://raw.githubusercontent.com/fchollet/ARC-AGI/master/data/evaluation/"

def get_task_list():
    # Fetch list of filenames from GitHub API
    cmd = "curl -s https://api.github.com/repos/fchollet/ARC-AGI/contents/data/evaluation"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        return []
    data = json.loads(result.stdout)
    return [item['name'] for item in data if item['name'].endswith('.json')]

def download_task(filename):
    path = TASKS_DIR / filename
    if path.exists():
        return True
    url = GITHUB_RAW_BASE + filename
    cmd = f"curl -s {url} -o {path}"
    subprocess.run(cmd, shell=True)
    return path.exists()

def spawn_solver(task_id):
    task_file = TASKS_DIR / f"{task_id}.json"
    with open(task_file, 'r') as f:
        full_task = json.load(f)
    
    # Strip test output to ensure blind solve
    blind_task = {
        "train": full_task["train"],
        "test": [{"input": t["input"]} for t in full_task["test"]]
    }
    
    prompt = f"""
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
    
    # Use the sessions_spawn tool via direct call
    print(f"Spawning solver for {task_id}...")
    # This script will now be run manually via the sessions_spawn tool from the main agent
    # to ensure the tool call is handled correctly by the system.

def main():
    tasks = get_task_list()
    print(f"Found {len(tasks)} tasks.")
    
    # Start with a batch of 5 to verify the pipeline
    batch = tasks[:5]
    for filename in batch:
        task_id = filename.replace('.json', '')
        if download_task(filename):
            spawn_solver(task_id)
            time.sleep(2) # Avoid hitting rate limits or overlapping spawns too fast

if __name__ == "__main__":
    main()
