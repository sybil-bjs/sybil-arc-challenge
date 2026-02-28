"""
Benchmark Script for ARC-AGI-3 Agents

Runs all prototype agents on available games and compares performance.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'agents'))

from world_model_agent import WorldModelAgent, run_game as run_world_model
from goal_inference_agent import GoalInferenceAgent, run_game as run_goal_inference
from llm_hybrid_agent import LLMHybridAgent, run_game as run_llm_hybrid

import arc_agi
from arcengine import GameAction
import numpy as np
from datetime import datetime


def run_benchmark(games: list = None, max_actions: int = 2000):
    """Run all agents on specified games and compare."""
    
    if games is None:
        games = ["ls20", "ft09", "vc33"]
    
    print("="*70)
    print("ARC-AGI-3 Agent Benchmark")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Max actions per game: {max_actions}")
    print("="*70)
    
    results = {}
    
    for game_id in games:
        print(f"\n\n{'#'*70}")
        print(f"# GAME: {game_id}")
        print(f"{'#'*70}")
        
        results[game_id] = {}
        
        # Test each agent
        agents = [
            ("World Model + Curiosity", run_world_model),
            ("Goal Inference", run_goal_inference),
            ("LLM Hybrid", run_llm_hybrid),
        ]
        
        for agent_name, run_fn in agents:
            print(f"\n--- {agent_name} ---")
            try:
                scorecard, stats = run_fn(game_id, max_actions=max_actions)
                
                # Extract results
                env_data = scorecard.environments[0] if scorecard.environments else {}
                run_data = env_data.get("runs", [{}])[0] if isinstance(env_data, dict) else {}
                
                results[game_id][agent_name] = {
                    "levels": run_data.get("levels_completed", stats.get("levels_completed", 0)),
                    "actions": run_data.get("actions", stats.get("total_actions", max_actions)),
                    "score": run_data.get("score", 0),
                    "stats": stats
                }
            except Exception as e:
                print(f"  ERROR: {e}")
                results[game_id][agent_name] = {
                    "levels": 0,
                    "actions": max_actions,
                    "score": 0,
                    "error": str(e)
                }
    
    # Summary
    print("\n\n" + "="*70)
    print("BENCHMARK SUMMARY")
    print("="*70)
    
    print(f"\n{'Game':<10} {'Agent':<25} {'Levels':<8} {'Actions':<10} {'Score':<8}")
    print("-"*70)
    
    for game_id in games:
        for agent_name, data in results[game_id].items():
            print(f"{game_id:<10} {agent_name:<25} {data['levels']:<8} {data['actions']:<10} {data['score']:<8.3f}")
        print()
    
    # Analysis
    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)
    
    # Find best agent per game
    for game_id in games:
        best_agent = max(results[game_id].items(), 
                        key=lambda x: (x[1]['levels'], -x[1]['actions']))
        print(f"\n{game_id}: Best = {best_agent[0]} ({best_agent[1]['levels']} levels)")
    
    # Overall best
    totals = {}
    for game_id in games:
        for agent_name, data in results[game_id].items():
            if agent_name not in totals:
                totals[agent_name] = {"levels": 0, "actions": 0}
            totals[agent_name]["levels"] += data["levels"]
            totals[agent_name]["actions"] += data["actions"]
    
    print(f"\n\nOVERALL (across {len(games)} games):")
    for agent_name, data in sorted(totals.items(), key=lambda x: -x[1]["levels"]):
        print(f"  {agent_name}: {data['levels']} total levels, {data['actions']} total actions")
    
    return results


def quick_test(game_id: str = "ls20", actions: int = 500):
    """Quick test of all agents on a single game."""
    print(f"Quick test on {game_id} with {actions} actions each\n")
    return run_benchmark([game_id], max_actions=actions)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "quick":
            game = sys.argv[2] if len(sys.argv) > 2 else "ls20"
            actions = int(sys.argv[3]) if len(sys.argv) > 3 else 500
            quick_test(game, actions)
        else:
            # Run on specified games
            games = sys.argv[1:]
            run_benchmark(games, max_actions=2000)
    else:
        # Full benchmark
        run_benchmark(max_actions=2000)
