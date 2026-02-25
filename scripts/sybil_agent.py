#!/usr/bin/env python3
"""
Sybil Agent — Integrated ARC-AGI-3 solver.

Combines:
- Systematic exploration (learn before winning)
- Visual analysis (understand game state)
- Failure memory (don't repeat mistakes)
- Pattern knowledge (transfer learning)
- Action efficiency tracking (optimize for score)

Usage:
  python sybil_agent.py --game=ls20 --explore-first
  python sybil_agent.py --game=ls20 --load-exploration
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass
import numpy as np

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent))

from visual_analyzer import (
    frame_to_emoji, 
    compare_frames, 
    analyze_frame, 
    describe_frame_for_llm
)
from exploration_agent import ExplorationAgent
from failure_memory import (
    record_failure, 
    add_lesson, 
    get_failure_context_prompt,
    get_lessons_for_game
)

REPO = Path.home() / "sybil-arc-challenge"

@dataclass
class AgentConfig:
    """Configuration for the Sybil agent."""
    game_id: str
    max_exploration_actions: int = 50
    max_execution_actions: int = 500
    explore_first: bool = True
    use_failure_memory: bool = True
    save_replays: bool = True
    verbose: bool = True

class SybilAgent:
    """
    Main agent class that orchestrates ARC-AGI-3 solving.
    """
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.explorer = ExplorationAgent(
            config.game_id, 
            max_exploration_actions=config.max_exploration_actions
        )
        self.action_history: List[Dict] = []
        self.frame_history: List[np.ndarray] = []
        self.current_level = 0
        self.levels_completed = 0
        self.total_actions = 0
        
    def log(self, msg: str):
        if self.config.verbose:
            print(f"[Sybil] {msg}")
    
    def get_failure_context(self) -> str:
        """Get context about past failures for this game."""
        if not self.config.use_failure_memory:
            return ""
        return get_failure_context_prompt(self.config.game_id)
    
    def decide_action(
        self, 
        current_frame: np.ndarray, 
        available_actions: List[str],
        game_state: dict
    ) -> str:
        """
        Main decision function — pick the next action.
        
        Strategy:
        1. If in exploration phase, follow exploration plan
        2. If exploration complete, use learned mechanics to plan
        3. Always check failure memory to avoid known mistakes
        """
        
        # Phase 1: Exploration
        if self.config.explore_first and self.explorer.should_explore():
            plan = self.explorer.get_exploration_plan(available_actions)
            if plan:
                action = plan[0]
                self.log(f"EXPLORE: Trying {action}")
                return action
            else:
                self.log("Exploration phase complete")
                self.log(self.explorer.get_learned_summary())
        
        # Phase 2: Strategic execution
        return self._strategic_action(current_frame, available_actions, game_state)
    
    def _strategic_action(
        self, 
        frame: np.ndarray, 
        actions: List[str], 
        state: dict
    ) -> str:
        """
        Choose action strategically based on learned mechanics.
        
        This is where we'd integrate with an LLM for complex reasoning.
        For now, we use heuristics based on exploration results.
        """
        
        # Get lessons from failures
        lessons = get_lessons_for_game(self.config.game_id)
        
        # Analyze current frame
        analysis = analyze_frame(frame)
        
        # Simple heuristic: prefer actions that caused changes during exploration
        best_action = None
        best_score = -1
        
        for action in actions:
            if action == "RESET":
                continue  # Don't reset unless necessary
            
            effects = self.explorer.state.action_effects.get(action, [])
            if effects:
                # Score based on how often this action caused meaningful changes
                change_rate = sum(1 for e in effects if e.get("changed")) / len(effects)
                score = change_rate
                
                # Bonus if this action caused movement (usually good for games)
                movement_rate = sum(1 for e in effects if e.get("movement")) / len(effects)
                score += movement_rate * 0.5
                
                if score > best_score:
                    best_score = score
                    best_action = action
        
        # Fallback to first available action
        if best_action is None:
            best_action = actions[0] if actions else "RESET"
        
        self.log(f"EXECUTE: {best_action} (score: {best_score:.2f})")
        return best_action
    
    def process_result(
        self, 
        action: str, 
        frame_before: np.ndarray, 
        frame_after: np.ndarray,
        result: dict
    ):
        """Process the result of an action."""
        
        self.total_actions += 1
        self.action_history.append({
            "action": action,
            "result": result
        })
        self.frame_history.append(frame_after)
        
        # Update exploration if still exploring
        if self.explorer.should_explore():
            self.explorer.explore_action(action, frame_before, frame_after, result)
        
        # Check for level completion
        if result.get("level_complete"):
            self.levels_completed += 1
            self.current_level += 1
            self.log(f"✅ Level {self.current_level} complete!")
        
        # Check for game over
        if result.get("game_over"):
            if result.get("won"):
                self.log(f"🎉 Game won in {self.total_actions} actions!")
            else:
                self.log(f"💀 Game lost after {self.total_actions} actions")
                self._record_failure(result)
    
    def _record_failure(self, result: dict):
        """Record failure for learning."""
        failure_id = record_failure(
            game_id=self.config.game_id,
            failure_mode=result.get("failure_reason", "unknown"),
            description=f"Failed at level {self.current_level}",
            strategy_tried=self.explorer.get_learned_summary()[:500],
            level=self.current_level,
            actions_taken=self.total_actions,
            context={"final_frame": self.frame_history[-1].tolist() if self.frame_history else None}
        )
        
        # Add automatic lesson
        add_lesson(
            failure_id,
            f"Strategy at level {self.current_level} needs improvement",
            applies_to="this_game",
            confidence=0.3
        )
    
    def save_state(self):
        """Save agent state for later resumption."""
        state_dir = REPO / "agent_states" / self.config.game_id
        state_dir.mkdir(parents=True, exist_ok=True)
        
        # Save exploration state
        self.explorer.save_state(str(state_dir / "exploration.json"))
        
        # Save action history
        with open(state_dir / "history.json", "w") as f:
            json.dump({
                "total_actions": self.total_actions,
                "levels_completed": self.levels_completed,
                "action_history": self.action_history[-100:]  # Keep last 100
            }, f, indent=2)
        
        self.log(f"State saved to {state_dir}")
    
    def load_state(self):
        """Load previous agent state."""
        state_dir = REPO / "agent_states" / self.config.game_id
        
        # Load exploration
        exploration_path = state_dir / "exploration.json"
        if exploration_path.exists():
            self.explorer = ExplorationAgent.load_state(str(exploration_path))
            self.log(f"Loaded exploration state: {self.explorer.state.total_exploration_actions} actions")
        
        # Load history
        history_path = state_dir / "history.json"
        if history_path.exists():
            with open(history_path) as f:
                data = json.load(f)
            self.total_actions = data["total_actions"]
            self.levels_completed = data["levels_completed"]
            self.log(f"Loaded history: {self.total_actions} actions, {self.levels_completed} levels")

def play_game(game_id: str, explore_first: bool = True, max_actions: int = 100):
    """
    Main entry point — play an ARC-AGI-3 game.
    """
    try:
        import arc_agi
        from arcengine import GameAction
    except ImportError:
        print("arc-agi package not installed. Run: pip install arc-agi")
        return
    
    config = AgentConfig(
        game_id=game_id,
        explore_first=explore_first,
        max_exploration_actions=30,
        max_execution_actions=max_actions
    )
    
    agent = SybilAgent(config)
    
    # Check for previous state
    state_dir = REPO / "agent_states" / game_id
    if (state_dir / "exploration.json").exists():
        agent.log("Found previous state. Loading...")
        agent.load_state()
    
    # Initialize game
    arc = arc_agi.Arcade()
    env = arc.make(game_id)
    
    state = env.reset()
    previous_frame = np.array(state.frame[0]) if state.frame else None
    
    agent.log(f"Starting game: {game_id}")
    agent.log(f"Win condition: {state.win_levels} levels")
    agent.log(f"Available actions: {state.available_actions}")
    
    # Game loop
    action_map = {
        1: GameAction.ACTION1,
        2: GameAction.ACTION2,
        3: GameAction.ACTION3,
        4: GameAction.ACTION4,
        5: GameAction.ACTION5,
        6: GameAction.ACTION6,
        7: GameAction.ACTION7,
    }
    
    action_names = [f"ACTION{a}" for a in state.available_actions]
    
    for i in range(max_actions):
        # Get current frame
        current_frame = np.array(state.frame[0]) if state.frame else previous_frame
        
        # Decide action
        game_state = {
            "levels_completed": state.levels_completed,
            "win_levels": state.win_levels
        }
        
        action_name = agent.decide_action(current_frame, action_names, game_state)
        
        # Parse action
        action_num = int(action_name.replace("ACTION", ""))
        game_action = action_map.get(action_num, GameAction.ACTION1)
        
        # Take action
        state = env.step(game_action)
        new_frame = np.array(state.frame[0]) if state.frame else current_frame
        
        # Process result
        result = {
            "levels_completed": state.levels_completed,
            "level_complete": state.levels_completed > agent.levels_completed,
            "game_over": state.levels_completed >= state.win_levels,
            "won": state.levels_completed >= state.win_levels
        }
        
        agent.process_result(action_name, current_frame, new_frame, result)
        
        # Check if done
        if result["game_over"]:
            break
        
        previous_frame = new_frame
    
    # Save state
    agent.save_state()
    
    # Print scorecard
    print("\n=== Final Scorecard ===")
    print(arc.get_scorecard())
    
    return agent

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sybil ARC-AGI-3 Agent")
    parser.add_argument("--game", "-g", default="ls20", help="Game ID to play")
    parser.add_argument("--explore-first", action="store_true", help="Do exploration before trying to win")
    parser.add_argument("--max-actions", "-n", type=int, default=100, help="Maximum actions to take")
    parser.add_argument("--load", action="store_true", help="Load previous state if available")
    
    args = parser.parse_args()
    
    play_game(
        game_id=args.game,
        explore_first=args.explore_first,
        max_actions=args.max_actions
    )
