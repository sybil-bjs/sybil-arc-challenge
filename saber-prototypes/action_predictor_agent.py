"""
Action Predictor Agent for ARC-AGI-3

This implements the core technique from StochasticGoose (1st place):
1. Train a model to predict which actions cause frame changes
2. Bias action selection toward "productive" actions
3. Track visited states to avoid loops

Key insight: Most random actions do nothing. Learning to predict
which actions cause changes dramatically improves exploration efficiency.
"""

import arc_agi
from arcengine import GameAction
import numpy as np
from collections import defaultdict, deque
import hashlib

# Map integers to GameAction
INT_TO_ACTION = {a.value: a for a in GameAction}


def int_to_action(i: int) -> GameAction:
    return INT_TO_ACTION.get(i, GameAction.ACTION1)


class ActionChangePredictor:
    """
    Simple tabular predictor for action change probability.
    
    For each (state_hash, action), tracks:
    - How many times we tried it
    - How many times it caused a frame change
    
    In the full StochasticGoose solution, this is a CNN.
    Here we use hash-based tables for simplicity.
    """
    
    def __init__(self, initial_optimism: float = 0.5):
        # (state_hash, action) -> {attempts, changes}
        self.stats = defaultdict(lambda: {"attempts": 0, "changes": 0})
        self.initial_optimism = initial_optimism
        
    def observe(self, state_hash: str, action: int, caused_change: bool):
        """Record an observation."""
        key = (state_hash, action)
        self.stats[key]["attempts"] += 1
        if caused_change:
            self.stats[key]["changes"] += 1
    
    def predict_change_prob(self, state_hash: str, action: int) -> float:
        """Predict probability that this action will cause a change."""
        key = (state_hash, action)
        data = self.stats[key]
        
        if data["attempts"] == 0:
            return self.initial_optimism  # Optimistic about unexplored
        
        # Beta posterior mean: (successes + 1) / (attempts + 2)
        return (data["changes"] + 1) / (data["attempts"] + 2)
    
    def get_action_scores(self, state_hash: str, available_actions: list) -> dict:
        """Get change probability scores for all actions."""
        scores = {}
        for action in available_actions:
            if isinstance(action, GameAction):
                action_id = action.value
            else:
                action_id = action
            scores[action_id] = self.predict_change_prob(state_hash, action_id)
        return scores


class StateTracker:
    """
    Tracks visited states to detect loops and encourage exploration.
    """
    
    def __init__(self):
        self.visit_counts = defaultdict(int)
        self.transitions = {}  # (from_state, action) -> to_state
        self.reverse_transitions = defaultdict(list)  # to_state -> [(from_state, action), ...]
        
    def hash_frame(self, frame) -> str:
        if frame is None:
            return "none"
        if isinstance(frame, list):
            data = str(frame).encode()
        elif hasattr(frame, 'tobytes'):
            data = frame.tobytes()
        else:
            data = str(frame).encode()
        return hashlib.md5(data).hexdigest()[:12]
    
    def visit(self, state_hash: str):
        self.visit_counts[state_hash] += 1
    
    def record_transition(self, from_state: str, action: int, to_state: str):
        key = (from_state, action)
        self.transitions[key] = to_state
        self.reverse_transitions[to_state].append((from_state, action))
    
    def get_novelty_bonus(self, state_hash: str) -> float:
        """Higher bonus for less-visited states."""
        visits = self.visit_counts[state_hash]
        return 1.0 / (visits + 1)
    
    def is_loop(self, from_state: str, action: int, to_state: str) -> bool:
        """Check if this transition creates a short loop."""
        key = (from_state, action)
        if key in self.transitions:
            return self.transitions[key] == to_state
        return False


class ActionPredictorAgent:
    """
    Agent that learns to predict which actions cause changes,
    then uses this to guide exploration.
    """
    
    def __init__(self, 
                 exploration_rate: float = 0.2,
                 novelty_weight: float = 0.3,
                 change_weight: float = 0.7):
        self.predictor = ActionChangePredictor()
        self.tracker = StateTracker()
        
        self.exploration_rate = exploration_rate
        self.novelty_weight = novelty_weight
        self.change_weight = change_weight
        
        # State
        self.last_state_hash = None
        self.last_action = None
        self.total_actions = 0
        self.actions_causing_change = 0
        self.levels_completed = 0
        
        # For ACTION6: track which coordinates caused changes
        self.coord_stats = defaultdict(lambda: {"attempts": 0, "changes": 0})
        
    def choose_action(self, frame_data, available_actions) -> GameAction:
        # Get current state
        frame = frame_data.frame if hasattr(frame_data, 'frame') else None
        current_hash = self.tracker.hash_frame(frame)
        
        # Check for level change
        current_level = frame_data.levels_completed if hasattr(frame_data, 'levels_completed') else 0
        if current_level > self.levels_completed:
            self.levels_completed = current_level
            # Clear predictor on new level (mechanics might change)
            # Actually, keep it - winning solution didn't reset
        
        # Update predictor with last action's result
        if self.last_state_hash is not None and self.last_action is not None:
            caused_change = (current_hash != self.last_state_hash)
            self.predictor.observe(self.last_state_hash, self.last_action.value, caused_change)
            self.tracker.record_transition(self.last_state_hash, self.last_action.value, current_hash)
            
            if caused_change:
                self.actions_causing_change += 1
        
        # Visit current state
        self.tracker.visit(current_hash)
        
        # Convert available actions
        if isinstance(available_actions[0], int):
            available_actions = [int_to_action(a) for a in available_actions]
        
        # Filter out RESET
        valid_actions = [a for a in available_actions if a != GameAction.RESET]
        if not valid_actions:
            valid_actions = [GameAction.ACTION1]  # Fallback
        
        # Get scores for each action
        change_scores = self.predictor.get_action_scores(
            current_hash, 
            [a.value for a in valid_actions]
        )
        
        # Combine change probability with novelty
        action_scores = {}
        for action in valid_actions:
            change_prob = change_scores.get(action.value, 0.5)
            
            # Predict next state and get its novelty
            key = (current_hash, action.value)
            if key in self.tracker.transitions:
                next_state = self.tracker.transitions[key]
                novelty = self.tracker.get_novelty_bonus(next_state)
            else:
                novelty = 1.0  # Unknown transition = novel
            
            score = (self.change_weight * change_prob + 
                    self.novelty_weight * novelty)
            action_scores[action] = score
        
        # Select action
        if np.random.random() < self.exploration_rate:
            # Pure exploration
            action = np.random.choice(valid_actions)
        else:
            # Exploit: pick highest scoring action
            action = max(action_scores, key=action_scores.get)
        
        # Handle ACTION6 (click coordinates)
        if action == GameAction.ACTION6:
            action = self._choose_coordinates(frame, action)
        
        self.last_state_hash = current_hash
        self.last_action = action
        self.total_actions += 1
        
        return action
    
    def _choose_coordinates(self, frame, action: GameAction) -> GameAction:
        """Choose coordinates for ACTION6, preferring areas that caused changes."""
        if frame is None:
            x, y = np.random.randint(0, 64), np.random.randint(0, 64)
        else:
            arr = np.array(frame)
            if len(arr.shape) == 3:
                arr = arr[0]
            
            # Find non-zero (interesting) regions
            nonzero = np.argwhere(arr > 0)
            
            if len(nonzero) > 0:
                # Prefer clicking on colored pixels
                # Could be smarter with coordinate prediction model
                idx = np.random.randint(len(nonzero))
                y, x = nonzero[idx]
            else:
                x, y = np.random.randint(0, 64), np.random.randint(0, 64)
        
        action.set_data({"x": int(x), "y": int(y)})
        return action
    
    def get_stats(self) -> dict:
        efficiency = self.actions_causing_change / max(1, self.total_actions)
        return {
            "total_actions": self.total_actions,
            "actions_causing_change": self.actions_causing_change,
            "change_efficiency": efficiency,
            "unique_states": len(self.tracker.visit_counts),
            "unique_transitions": len(self.tracker.transitions),
            "levels_completed": self.levels_completed
        }


def run_game(game_id: str, max_actions: int = 10000, 
             exploration_rate: float = 0.15,
             novelty_weight: float = 0.3):
    """Run the action predictor agent."""
    arc = arc_agi.Arcade()
    env = arc.make(game_id)
    agent = ActionPredictorAgent(
        exploration_rate=exploration_rate,
        novelty_weight=novelty_weight
    )
    
    print(f"\n{'='*60}")
    print(f"Action Predictor Agent on {game_id}")
    print(f"Exploration: {exploration_rate}, Novelty weight: {novelty_weight}")
    print(f"{'='*60}")
    
    frame = env.reset()
    current_level = 0
    level_start = 0
    
    for i in range(max_actions):
        available = frame.available_actions if hasattr(frame, 'available_actions') else [1, 2, 3, 4]
        
        action = agent.choose_action(frame, available)
        frame = env.step(action)
        
        if hasattr(frame, 'levels_completed') and frame.levels_completed > current_level:
            actions_for_level = i + 1 - level_start
            stats = agent.get_stats()
            print(f"  ✓ Level {current_level + 1} in {actions_for_level} actions "
                  f"(efficiency: {stats['change_efficiency']:.1%})")
            current_level = frame.levels_completed
            level_start = i + 1
        
        if hasattr(frame, 'state') and str(frame.state) == 'WIN':
            print(f"\n🎉 Game WON in {agent.total_actions} actions!")
            break
        
        if i % 2000 == 0 and i > 0:
            stats = agent.get_stats()
            print(f"  [{i}] States: {stats['unique_states']}, "
                  f"Transitions: {stats['unique_transitions']}, "
                  f"Efficiency: {stats['change_efficiency']:.1%}")
    
    stats = agent.get_stats()
    print(f"\nFinal Statistics:")
    print(f"  Total Actions: {stats['total_actions']}")
    print(f"  Actions Causing Change: {stats['actions_causing_change']}")
    print(f"  Change Efficiency: {stats['change_efficiency']:.1%}")
    print(f"  Unique States: {stats['unique_states']}")
    print(f"  Levels Completed: {stats['levels_completed']}")
    
    return arc.get_scorecard(), stats


if __name__ == "__main__":
    import sys
    game = sys.argv[1] if len(sys.argv) > 1 else "ls20"
    max_actions = int(sys.argv[2]) if len(sys.argv) > 2 else 10000
    run_game(game, max_actions=max_actions)
