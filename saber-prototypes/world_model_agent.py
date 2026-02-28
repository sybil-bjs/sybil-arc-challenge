"""
World Model + Curiosity Agent for ARC-AGI-3

Strategy:
1. Learn a forward model that predicts next frame given (current frame, action)
2. Use prediction error as intrinsic reward (curiosity)
3. Explore actions where the model is uncertain/wrong
4. Gradually build understanding of game mechanics

Based on ideas from:
- Random Network Distillation (RND)
- Intrinsic Curiosity Module (ICM)
- BYOL-Explore
"""

import arc_agi
from arcengine import GameAction
import numpy as np
from collections import deque
import hashlib

# Map integers to GameAction enum members
INT_TO_ACTION = {a.value: a for a in GameAction}


def int_to_action(i: int) -> GameAction:
    return INT_TO_ACTION.get(i, GameAction.ACTION1)


class SimpleWorldModel:
    """
    Simplified world model using frame hashing and transition counting.
    Predicts: P(next_state | current_state, action)
    
    For a full implementation, this would be a neural network.
    Here we use a hash-based approximation for speed.
    """
    
    def __init__(self):
        # Transitions: (state_hash, action) -> {next_state_hash: count}
        self.transitions = {}
        # Total counts per (state, action) pair
        self.counts = {}
        
    def hash_frame(self, frame) -> str:
        """Hash a frame to a compact representation."""
        if frame is None:
            return "none"
        if isinstance(frame, list):
            data = str(frame).encode()
        elif hasattr(frame, 'tobytes'):
            data = frame.tobytes()
        else:
            data = str(frame).encode()
        return hashlib.md5(data).hexdigest()[:12]
    
    def observe(self, state_hash: str, action: int, next_state_hash: str):
        """Record a transition."""
        key = (state_hash, action)
        
        if key not in self.transitions:
            self.transitions[key] = {}
            self.counts[key] = 0
        
        if next_state_hash not in self.transitions[key]:
            self.transitions[key][next_state_hash] = 0
        
        self.transitions[key][next_state_hash] += 1
        self.counts[key] += 1
    
    def predict_entropy(self, state_hash: str, action: int) -> float:
        """
        Return uncertainty about the next state.
        High entropy = uncertain = interesting to explore.
        """
        key = (state_hash, action)
        
        if key not in self.transitions:
            return 1.0  # Never seen this (state, action) pair - max curiosity
        
        total = self.counts[key]
        if total == 0:
            return 1.0
        
        # Calculate entropy of next-state distribution
        entropy = 0.0
        for count in self.transitions[key].values():
            p = count / total
            if p > 0:
                entropy -= p * np.log2(p + 1e-10)
        
        # Normalize to [0, 1] roughly
        return min(1.0, entropy / 3.0)
    
    def prediction_error(self, state_hash: str, action: int, actual_next: str) -> float:
        """
        Return how surprising the actual next state was.
        1.0 = completely unexpected, 0.0 = perfectly predicted
        """
        key = (state_hash, action)
        
        if key not in self.transitions:
            return 1.0  # Never seen before
        
        total = self.counts[key]
        if total == 0:
            return 1.0
        
        # How often did we see this transition?
        seen_count = self.transitions[key].get(actual_next, 0)
        probability = seen_count / total
        
        # Surprise = 1 - probability
        return 1.0 - probability


class WorldModelAgent:
    """
    Agent that uses a world model to drive curiosity-based exploration.
    """
    
    def __init__(self, exploration_weight: float = 0.5):
        self.world_model = SimpleWorldModel()
        self.exploration_weight = exploration_weight
        
        # State tracking
        self.last_state_hash = None
        self.last_action = None
        self.total_actions = 0
        self.total_surprise = 0.0
        
        # Experience buffer for analysis
        self.surprise_history = deque(maxlen=100)
        
    def hash_frame(self, frame_data) -> str:
        """Get hash of current frame."""
        if hasattr(frame_data, 'frame') and frame_data.frame is not None:
            return self.world_model.hash_frame(frame_data.frame)
        return f"level_{frame_data.levels_completed}"
    
    def choose_action(self, frame_data, available_actions) -> GameAction:
        """Choose action based on curiosity (world model uncertainty)."""
        current_hash = self.hash_frame(frame_data)
        
        # Update world model with last transition
        if self.last_state_hash is not None and self.last_action is not None:
            self.world_model.observe(
                self.last_state_hash, 
                self.last_action.value,
                current_hash
            )
            
            # Track surprise
            surprise = self.world_model.prediction_error(
                self.last_state_hash,
                self.last_action.value, 
                current_hash
            )
            self.surprise_history.append(surprise)
            self.total_surprise += surprise
        
        # Score each action by curiosity (entropy) + exploration bonus
        action_scores = {}
        for action in available_actions:
            if action == GameAction.RESET:
                continue
            
            # Curiosity score = entropy of predictions
            entropy = self.world_model.predict_entropy(current_hash, action.value)
            
            # Add small random noise for tie-breaking
            noise = np.random.uniform(0, 0.1)
            
            action_scores[action] = entropy + noise
        
        if not action_scores:
            # Fallback
            valid = [a for a in available_actions if a != GameAction.RESET]
            if valid:
                action = np.random.choice(valid)
            else:
                action = GameAction.RESET
        else:
            # Select action with highest curiosity score
            # With some probability, pick randomly for exploration
            if np.random.random() < self.exploration_weight:
                action = np.random.choice(list(action_scores.keys()))
            else:
                action = max(action_scores, key=action_scores.get)
        
        # Handle ACTION6 coordinates
        if action == GameAction.ACTION6:
            action.set_data({
                "x": np.random.randint(0, 64),
                "y": np.random.randint(0, 64)
            })
        
        self.last_state_hash = current_hash
        self.last_action = action
        self.total_actions += 1
        
        return action
    
    def get_stats(self) -> dict:
        """Return agent statistics."""
        return {
            "total_actions": self.total_actions,
            "unique_states": len(set(
                k[0] for k in self.world_model.transitions.keys()
            )),
            "unique_transitions": len(self.world_model.transitions),
            "avg_surprise": np.mean(self.surprise_history) if self.surprise_history else 0,
            "total_surprise": self.total_surprise
        }


def run_game(game_id: str, max_actions: int = 5000, exploration_weight: float = 0.3):
    """Run the world model agent on a game."""
    arc = arc_agi.Arcade()
    env = arc.make(game_id)
    agent = WorldModelAgent(exploration_weight=exploration_weight)
    
    print(f"\n{'='*60}")
    print(f"World Model Agent on {game_id}")
    print(f"Exploration weight: {exploration_weight}")
    print(f"{'='*60}")
    
    frame = env.reset()
    current_level = 0
    level_start_action = 0
    
    for i in range(max_actions):
        # Get available actions
        available = frame.available_actions if hasattr(frame, 'available_actions') else [1, 2, 3, 4]
        if isinstance(available, list) and len(available) > 0 and isinstance(available[0], int):
            available = [int_to_action(a) for a in available]
        
        # Choose action
        action = agent.choose_action(frame, available)
        
        # Take action
        frame = env.step(action)
        
        # Check for level completion
        if hasattr(frame, 'levels_completed') and frame.levels_completed > current_level:
            actions_for_level = i + 1 - level_start_action
            print(f"  ✓ Level {current_level + 1} completed in {actions_for_level} actions")
            current_level = frame.levels_completed
            level_start_action = i + 1
        
        # Check for win
        if hasattr(frame, 'state') and str(frame.state) == 'WIN':
            print(f"\n🎉 Game WON in {agent.total_actions} actions!")
            break
        
        # Progress update
        if i % 1000 == 0 and i > 0:
            stats = agent.get_stats()
            print(f"  [{i}] Levels: {current_level}, States: {stats['unique_states']}, "
                  f"Transitions: {stats['unique_transitions']}, Surprise: {stats['avg_surprise']:.3f}")
    
    # Final stats
    stats = agent.get_stats()
    print(f"\nFinal Statistics:")
    print(f"  Total Actions: {stats['total_actions']}")
    print(f"  Unique States: {stats['unique_states']}")
    print(f"  Unique Transitions: {stats['unique_transitions']}")
    print(f"  Avg Surprise: {stats['avg_surprise']:.3f}")
    print(f"  Levels Completed: {current_level}")
    
    return arc.get_scorecard(), stats


if __name__ == "__main__":
    import sys
    game = sys.argv[1] if len(sys.argv) > 1 else "ls20"
    max_actions = int(sys.argv[2]) if len(sys.argv) > 2 else 5000
    run_game(game, max_actions=max_actions)
