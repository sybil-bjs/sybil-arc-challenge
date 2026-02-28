"""
Curiosity-Driven Baseline Agent for ARC-AGI-3

Strategy:
1. Track which actions cause frame changes (novelty)
2. Prioritize actions that have caused changes before
3. Avoid actions that consistently do nothing
4. Build a simple state-action map

This is a first baseline to understand the problem space.
"""

import arc_agi
from arcengine import GameAction
import numpy as np
from collections import defaultdict
import hashlib


# Map integers to GameAction enum members
INT_TO_ACTION = {a.value: a for a in GameAction}


def int_to_action(i: int) -> GameAction:
    """Convert integer to GameAction."""
    return INT_TO_ACTION.get(i, GameAction.ACTION1)


def hash_frame(frame_data) -> str:
    """Create a hash of the current frame for state tracking."""
    # Get the frame image if available
    if hasattr(frame_data, 'frame') and frame_data.frame is not None:
        frame = frame_data.frame
        if isinstance(frame, list):
            # Convert nested list to bytes
            flat = str(frame).encode()
        elif hasattr(frame, 'tobytes'):
            flat = frame.tobytes()
        else:
            flat = str(frame).encode()
        return hashlib.md5(flat).hexdigest()[:16]
    return str(frame_data.levels_completed)


class CuriosityAgent:
    def __init__(self):
        # Track action effectiveness per state
        self.action_changes = defaultdict(lambda: defaultdict(int))  # state -> action -> change_count
        self.action_attempts = defaultdict(lambda: defaultdict(int))  # state -> action -> attempt_count
        self.visited_states = set()
        self.last_state_hash = None
        self.last_action = None
        self.total_actions = 0
        self.level_actions = 0
        
    def reset_for_level(self):
        """Reset per-level tracking."""
        self.level_actions = 0
        # Keep cross-level knowledge
        
    def choose_action(self, frame_data, available_actions):
        """Choose action based on curiosity (preference for actions that cause changes)."""
        state_hash = hash_frame(frame_data)
        
        # Track if last action caused a change
        if self.last_state_hash is not None and self.last_action is not None:
            if state_hash != self.last_state_hash:
                self.action_changes[self.last_state_hash][self.last_action] += 1
            self.action_attempts[self.last_state_hash][self.last_action] += 1
        
        # Record state visit
        self.visited_states.add(state_hash)
        
        # Score actions by their change ratio
        action_scores = {}
        for action in available_actions:
            if action == GameAction.RESET:
                continue  # Skip reset unless stuck
                
            attempts = self.action_attempts[state_hash][action]
            changes = self.action_changes[state_hash][action]
            
            if attempts == 0:
                # Unexplored action - high curiosity score
                action_scores[action] = 1.0
            else:
                # Score based on change ratio with exploration bonus
                change_ratio = changes / attempts
                exploration_bonus = 1.0 / (attempts + 1)  # Decay with attempts
                action_scores[action] = change_ratio + 0.1 * exploration_bonus
        
        if not action_scores:
            # Fallback to random if no valid actions
            action = np.random.choice([a for a in available_actions if a != GameAction.RESET])
        else:
            # Softmax selection with temperature
            actions = list(action_scores.keys())
            scores = np.array([action_scores[a] for a in actions])
            
            # Add small noise for exploration
            scores = scores + np.random.uniform(0, 0.1, len(scores))
            
            # Select best
            action = actions[np.argmax(scores)]
        
        # Handle ACTION6 (click with coordinates)
        if action == GameAction.ACTION6:
            # For now, random coordinates - could be smarter
            action.set_data({
                "x": np.random.randint(0, 64),
                "y": np.random.randint(0, 64)
            })
        
        self.last_state_hash = state_hash
        self.last_action = action
        self.total_actions += 1
        self.level_actions += 1
        
        return action


def run_game(game_id: str, max_actions: int = 10000):
    """Run the curiosity agent on a game."""
    arc = arc_agi.Arcade()
    env = arc.make(game_id)
    agent = CuriosityAgent()
    
    print(f"\n{'='*50}")
    print(f"Running CuriosityAgent on {game_id}")
    print(f"{'='*50}")
    
    # Initial reset
    frame = env.reset()
    current_level = 0
    
    for i in range(max_actions):
        # Get available actions
        available = frame.available_actions if hasattr(frame, 'available_actions') else list(GameAction)
        if isinstance(available, list) and len(available) > 0 and isinstance(available[0], int):
            available = [int_to_action(a) for a in available]
        
        # Choose action
        action = agent.choose_action(frame, available)
        
        # Take action
        frame = env.step(action)
        
        # Check for level completion
        if hasattr(frame, 'levels_completed') and frame.levels_completed > current_level:
            print(f"  Level {current_level + 1} completed in {agent.level_actions} actions!")
            current_level = frame.levels_completed
            agent.reset_for_level()
        
        # Check for win
        if hasattr(frame, 'state') and str(frame.state) == 'WIN':
            print(f"\n🎉 Game completed in {agent.total_actions} total actions!")
            break
        
        # Progress update
        if i % 1000 == 0 and i > 0:
            print(f"  Actions: {i}, Levels: {current_level}, Unique states: {len(agent.visited_states)}")
    
    # Final stats
    scorecard = arc.get_scorecard()
    print(f"\nFinal Scorecard:")
    print(f"  Score: {scorecard.score if hasattr(scorecard, 'score') else 'N/A'}")
    print(f"  Total Actions: {agent.total_actions}")
    print(f"  Unique States Visited: {len(agent.visited_states)}")
    
    return scorecard


if __name__ == "__main__":
    import sys
    game = sys.argv[1] if len(sys.argv) > 1 else "ls20"
    run_game(game, max_actions=5000)
