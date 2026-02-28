"""
LLM + RL Hybrid Agent for ARC-AGI-3

Strategy:
1. Use LLM to observe frames and hypothesize rules
2. Test hypotheses with actions
3. Confirm or reject based on outcomes
4. Build a rule library over time
5. Use rules to guide action selection

This combines:
- LLM's ability to reason about visual patterns
- RL's ability to verify through interaction
"""

import arc_agi
from arcengine import GameAction
import numpy as np
from collections import deque
import hashlib
import json
import os

# Map integers to GameAction enum members
INT_TO_ACTION = {a.value: a for a in GameAction}


def int_to_action(i: int) -> GameAction:
    return INT_TO_ACTION.get(i, GameAction.ACTION1)


def frame_to_text(frame, max_size: int = 32) -> str:
    """
    Convert a frame to a text description.
    Simplified version - just describe basic statistics.
    """
    if frame is None:
        return "Empty frame"
    
    arr = np.array(frame)
    if len(arr.shape) == 3:
        arr = arr[0]  # Take first channel if 3D
    
    # Downsample if needed
    if arr.shape[0] > max_size:
        step = arr.shape[0] // max_size
        arr = arr[::step, ::step]
    
    # Get statistics
    unique_colors = np.unique(arr)
    num_colors = len(unique_colors)
    
    # Find regions of each color
    color_info = []
    for c in unique_colors[:5]:  # Limit to 5 colors
        mask = (arr == c)
        count = np.sum(mask)
        if count > 0:
            rows, cols = np.where(mask)
            center_r, center_c = np.mean(rows), np.mean(cols)
            color_info.append(f"Color {c}: {count} pixels, center ({center_r:.0f},{center_c:.0f})")
    
    return f"Frame {arr.shape[0]}x{arr.shape[1]}, {num_colors} colors. " + "; ".join(color_info)


class RuleLibrary:
    """
    Stores and retrieves learned rules about the game.
    Rules are simple condition-action mappings.
    """
    
    def __init__(self):
        # Rules: {rule_id: {condition, action, confidence, successes, attempts}}
        self.rules = {}
        self.rule_counter = 0
        
    def add_rule(self, condition: str, action: int, initial_confidence: float = 0.5):
        """Add a new hypothesized rule."""
        rule_id = f"rule_{self.rule_counter}"
        self.rule_counter += 1
        
        self.rules[rule_id] = {
            "condition": condition,
            "action": action,
            "confidence": initial_confidence,
            "successes": 0,
            "attempts": 0
        }
        return rule_id
    
    def update_rule(self, rule_id: str, success: bool):
        """Update rule confidence based on outcome."""
        if rule_id not in self.rules:
            return
        
        rule = self.rules[rule_id]
        rule["attempts"] += 1
        if success:
            rule["successes"] += 1
        
        # Update confidence with Bayesian-ish update
        rule["confidence"] = (rule["successes"] + 1) / (rule["attempts"] + 2)
    
    def find_matching_rules(self, state_description: str, threshold: float = 0.3):
        """Find rules whose conditions might match the current state."""
        matching = []
        
        for rule_id, rule in self.rules.items():
            if rule["confidence"] < threshold:
                continue
            
            # Simple keyword matching (would be semantic similarity in full version)
            condition_words = set(rule["condition"].lower().split())
            state_words = set(state_description.lower().split())
            
            overlap = len(condition_words & state_words) / max(len(condition_words), 1)
            if overlap > 0.3:
                matching.append((rule_id, rule, overlap))
        
        # Sort by confidence * overlap
        matching.sort(key=lambda x: x[1]["confidence"] * x[2], reverse=True)
        return matching
    
    def get_best_action(self, state_description: str) -> tuple:
        """Get the best action based on matching rules."""
        matches = self.find_matching_rules(state_description)
        if matches:
            rule_id, rule, _ = matches[0]
            return rule["action"], rule_id, rule["confidence"]
        return None, None, 0


class LLMHybridAgent:
    """
    Agent that combines LLM reasoning with RL-style learning.
    
    Note: This is a simplified version that uses heuristic rule generation
    instead of actual LLM calls (to avoid API costs during development).
    For production, replace generate_hypothesis() with actual LLM call.
    """
    
    def __init__(self, exploration_rate: float = 0.4):
        self.rule_library = RuleLibrary()
        self.exploration_rate = exploration_rate
        
        # State tracking
        self.last_state_desc = None
        self.last_action = None
        self.last_rule_id = None
        self.last_level = 0
        self.total_actions = 0
        
        # History for hypothesis generation
        self.action_history = deque(maxlen=20)
        self.state_history = deque(maxlen=20)
        self.reward_history = deque(maxlen=20)
        
    def generate_hypothesis(self, state_desc: str, action: int, reward: bool) -> str:
        """
        Generate a hypothesis about why an action worked/failed.
        
        In production, this would call an LLM like:
        "Given state: {state_desc}, action {action} resulted in {'success' if reward else 'failure'}.
         What rule might explain this?"
        
        For now, use simple heuristics.
        """
        # Extract key features from state description
        keywords = []
        
        if "Color 0:" in state_desc:
            keywords.append("empty")
        if "center" in state_desc.lower():
            # Extract center position
            keywords.append("centered")
        
        action_names = {1: "up", 2: "down", 3: "left", 4: "right", 5: "interact", 6: "click"}
        action_name = action_names.get(action, str(action))
        
        if reward:
            condition = f"When {' '.join(keywords) if keywords else 'any state'}, try {action_name}"
        else:
            condition = f"Avoid {action_name} when {' '.join(keywords) if keywords else 'similar state'}"
        
        return condition
    
    def choose_action(self, frame_data, available_actions) -> GameAction:
        # Get state description
        frame = frame_data.frame if hasattr(frame_data, 'frame') else None
        state_desc = frame_to_text(frame)
        current_level = frame_data.levels_completed if hasattr(frame_data, 'levels_completed') else 0
        
        # Check for reward
        got_reward = current_level > self.last_level
        
        # Update rule library based on last action's outcome
        if self.last_rule_id is not None:
            self.rule_library.update_rule(self.last_rule_id, got_reward)
        
        # If we got a reward, generate a hypothesis
        if got_reward and self.last_action is not None:
            hypothesis = self.generate_hypothesis(
                self.last_state_desc or "", 
                self.last_action.value,
                True
            )
            self.rule_library.add_rule(hypothesis, self.last_action.value, 0.6)
        
        # Decide action
        action = None
        used_rule_id = None
        
        # Try to use learned rules
        if np.random.random() > self.exploration_rate:
            best_action, rule_id, confidence = self.rule_library.get_best_action(state_desc)
            if best_action is not None and confidence > 0.4:
                action = int_to_action(best_action)
                used_rule_id = rule_id
        
        # Otherwise explore
        if action is None or action not in available_actions:
            valid = [a for a in available_actions if a != GameAction.RESET]
            action = np.random.choice(valid) if valid else GameAction.RESET
            used_rule_id = None
        
        # Handle ACTION6
        if action == GameAction.ACTION6:
            # Try to click on interesting regions
            if frame is not None:
                arr = np.array(frame)
                if len(arr.shape) == 3:
                    arr = arr[0]
                # Find non-zero regions
                nonzero = np.argwhere(arr > 0)
                if len(nonzero) > 0:
                    idx = np.random.randint(len(nonzero))
                    y, x = nonzero[idx]
                    action.set_data({"x": int(x), "y": int(y)})
                else:
                    action.set_data({"x": np.random.randint(0, 64), "y": np.random.randint(0, 64)})
            else:
                action.set_data({"x": np.random.randint(0, 64), "y": np.random.randint(0, 64)})
        
        # Update tracking
        self.action_history.append(action.value)
        self.state_history.append(state_desc)
        self.reward_history.append(got_reward)
        
        self.last_state_desc = state_desc
        self.last_action = action
        self.last_rule_id = used_rule_id
        self.last_level = current_level
        self.total_actions += 1
        
        return action
    
    def get_stats(self) -> dict:
        high_conf_rules = sum(1 for r in self.rule_library.rules.values() if r["confidence"] > 0.5)
        return {
            "total_actions": self.total_actions,
            "total_rules": len(self.rule_library.rules),
            "high_confidence_rules": high_conf_rules,
            "levels_completed": self.last_level
        }


def run_game(game_id: str, max_actions: int = 5000, exploration_rate: float = 0.4):
    """Run the LLM hybrid agent on a game."""
    arc = arc_agi.Arcade()
    env = arc.make(game_id)
    agent = LLMHybridAgent(exploration_rate=exploration_rate)
    
    print(f"\n{'='*60}")
    print(f"LLM Hybrid Agent on {game_id}")
    print(f"Exploration rate: {exploration_rate}")
    print(f"{'='*60}")
    
    frame = env.reset()
    current_level = 0
    level_start = 0
    
    for i in range(max_actions):
        available = frame.available_actions if hasattr(frame, 'available_actions') else [1, 2, 3, 4]
        if isinstance(available, list) and len(available) > 0 and isinstance(available[0], int):
            available = [int_to_action(a) for a in available]
        
        action = agent.choose_action(frame, available)
        frame = env.step(action)
        
        if hasattr(frame, 'levels_completed') and frame.levels_completed > current_level:
            print(f"  ✓ Level {current_level + 1} in {i + 1 - level_start} actions")
            current_level = frame.levels_completed
            level_start = i + 1
        
        if hasattr(frame, 'state') and str(frame.state) == 'WIN':
            print(f"\n🎉 Game WON in {agent.total_actions} actions!")
            break
        
        if i % 1000 == 0 and i > 0:
            stats = agent.get_stats()
            print(f"  [{i}] Levels: {current_level}, Rules: {stats['total_rules']}, "
                  f"High-conf: {stats['high_confidence_rules']}")
    
    stats = agent.get_stats()
    print(f"\nFinal Statistics:")
    print(f"  Total Actions: {stats['total_actions']}")
    print(f"  Total Rules: {stats['total_rules']}")
    print(f"  High-Confidence Rules: {stats['high_confidence_rules']}")
    print(f"  Levels Completed: {current_level}")
    
    return arc.get_scorecard(), stats


if __name__ == "__main__":
    import sys
    game = sys.argv[1] if len(sys.argv) > 1 else "ls20"
    max_actions = int(sys.argv[2]) if len(sys.argv) > 2 else 5000
    run_game(game, max_actions=max_actions)
