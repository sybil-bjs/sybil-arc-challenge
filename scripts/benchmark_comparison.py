"""
Benchmark Comparison: Random vs Goal-Directed

Tests whether human priors + visual navigation beats random exploration.

Saber ⚔️ | Bridget 🎮 | Sybil 🧠
"""

import sys
import os
import numpy as np
from dataclasses import dataclass
from typing import List

sys.path.append(os.path.dirname(__file__))

try:
    import arc_agi
    from arcengine import GameAction
    HAS_ARC = True
except ImportError:
    print("arc_agi not installed")
    HAS_ARC = False
    sys.exit(1)

from goal_bayesian_agent import GoalBayesianAgent


@dataclass
class BenchmarkResult:
    agent_name: str
    game_id: str
    actions_taken: int
    levels_completed: int
    unique_states: int
    efficiency: float  # levels / actions


class RandomAgent:
    """Baseline: pure random action selection."""
    def __init__(self):
        self.actions_taken = 0
        self.levels_completed = 0
        self.visited_states = set()
    
    def choose_action(self, frame_data, available_actions):
        frame = np.array(frame_data.frame)
        state_hash = hash(frame.tobytes())
        self.visited_states.add(state_hash)
        
        # Track levels
        level = getattr(frame_data, 'levels_completed', 0)
        if level > self.levels_completed:
            self.levels_completed = level
        
        self.actions_taken += 1
        
        # Random action (exclude RESET)
        valid = [a for a in available_actions if a != GameAction.RESET]
        if not valid:
            valid = available_actions
        return np.random.choice(valid)
    
    def observe_result(self, frame_data):
        pass


def run_benchmark(game_id: str, max_actions: int, agent, agent_name: str) -> BenchmarkResult:
    """Run a single agent benchmark."""
    arc = arc_agi.Arcade()
    env = arc.make(game_id)
    
    frame = env.reset()
    levels_at_start = getattr(frame, 'levels_completed', 0)
    
    for i in range(max_actions):
        available = frame.available_actions or [1, 2, 3, 4]
        
        # Convert to GameAction if needed
        if isinstance(available[0], int):
            int_to_action = {a.value: a for a in GameAction}
            available = [int_to_action.get(a, GameAction.ACTION1) for a in available]
        
        action = agent.choose_action(frame, available)
        new_frame = env.step(action)
        
        if hasattr(agent, 'observe_result'):
            agent.observe_result(new_frame)
        
        frame = new_frame
        
        if str(frame.state) == 'WIN':
            break
    
    levels_completed = getattr(frame, 'levels_completed', 0) - levels_at_start
    if hasattr(agent, 'levels_completed'):
        levels_completed = agent.levels_completed
    
    actions = agent.actions_taken if hasattr(agent, 'actions_taken') else max_actions
    unique = len(agent.visited_states) if hasattr(agent, 'visited_states') else 0
    
    efficiency = levels_completed / max(actions, 1)
    
    return BenchmarkResult(
        agent_name=agent_name,
        game_id=game_id,
        actions_taken=actions,
        levels_completed=levels_completed,
        unique_states=unique,
        efficiency=efficiency,
    )


def main():
    game_id = sys.argv[1] if len(sys.argv) > 1 else "ls20"
    max_actions = int(sys.argv[2]) if len(sys.argv) > 2 else 500
    
    print(f"\n{'='*70}")
    print(f"🏁 BENCHMARK: {game_id} | {max_actions} actions per agent")
    print(f"{'='*70}\n")
    
    results = []
    
    # Test 1: Random Agent
    print("Testing Random Agent...")
    random_agent = RandomAgent()
    r1 = run_benchmark(game_id, max_actions, random_agent, "Random")
    results.append(r1)
    print(f"  ✓ Random: {r1.levels_completed} levels, {r1.actions_taken} actions, {r1.unique_states} states\n")
    
    # Test 2: Goal Agent WITHOUT visual navigation
    print("Testing Goal Agent (no visual)...")
    goal_agent_novis = GoalBayesianAgent(use_visual_navigation=False)
    r2 = run_benchmark(game_id, max_actions, goal_agent_novis, "Goal (no visual)")
    results.append(r2)
    print(f"  ✓ Goal (no visual): {r2.levels_completed} levels, {r2.actions_taken} actions\n")
    
    # Test 3: Goal Agent WITH visual navigation
    print("Testing Goal Agent (with visual)...")
    goal_agent_vis = GoalBayesianAgent(use_visual_navigation=True)
    r3 = run_benchmark(game_id, max_actions, goal_agent_vis, "Goal (visual)")
    results.append(r3)
    print(f"  ✓ Goal (visual): {r3.levels_completed} levels, {r3.actions_taken} actions\n")
    
    # Summary
    print(f"\n{'='*70}")
    print(f"📊 RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"{'Agent':<25} {'Levels':<10} {'Actions':<10} {'Efficiency':<15}")
    print(f"{'-'*70}")
    
    for r in results:
        eff_str = f"{r.efficiency*1000:.4f}/1k" if r.efficiency > 0 else "0"
        print(f"{r.agent_name:<25} {r.levels_completed:<10} {r.actions_taken:<10} {eff_str:<15}")
    
    print(f"\n{'='*70}")
    
    # Determine winner
    best = max(results, key=lambda x: (x.levels_completed, x.efficiency))
    print(f"🏆 WINNER: {best.agent_name}")
    
    if best.levels_completed > 0:
        print(f"   Completed {best.levels_completed} level(s) in {best.actions_taken} actions")
    else:
        print(f"   No levels completed (all agents need more actions or better detection)")
    
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
