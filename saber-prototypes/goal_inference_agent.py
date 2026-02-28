"""
Goal Inference + Planning Agent for ARC-AGI-3

Strategy:
1. Detect reward signals (level transitions, score changes)
2. When reward detected, backtrack to find what caused it
3. Build a model of "what leads to rewards"
4. Use planning to reach reward states more efficiently

Key insight: Most games have sparse rewards (level completion).
If we can identify *what* we did right, we can do more of it.
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


class RewardPredictor:
    """
    Learns to predict which (state, action) pairs lead to rewards.
    Uses a simple credit assignment: when reward is received,
    credit recent actions proportionally.
    """
    
    def __init__(self, credit_window: int = 50, decay: float = 0.9):
        self.credit_window = credit_window
        self.decay = decay
        
        # (state_hash, action) -> cumulative credit
        self.credits = {}
        
        # Recent history for credit assignment
        self.history = deque(maxlen=credit_window)
        
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
    
    def record_step(self, state_hash: str, action: int):
        """Record a step in history."""
        self.history.append((state_hash, action))
    
    def assign_credit(self, reward: float = 1.0):
        """
        Assign credit to recent (state, action) pairs.
        More recent actions get more credit.
        """
        for i, (state_hash, action) in enumerate(reversed(self.history)):
            # Exponential decay: most recent gets most credit
            credit = reward * (self.decay ** i)
            
            key = (state_hash, action)
            if key not in self.credits:
                self.credits[key] = 0.0
            self.credits[key] += credit
        
        # Clear history after credit assignment (new episode)
        self.history.clear()
    
    def get_action_value(self, state_hash: str, action: int) -> float:
        """Get learned value of (state, action) pair."""
        key = (state_hash, action)
        return self.credits.get(key, 0.0)


class StateGraph:
    """
    Builds a graph of state transitions.
    Used for planning paths to high-value states.
    """
    
    def __init__(self):
        # state -> [(action, next_state), ...]
        self.edges = {}
        # Reverse edges for backtracking
        self.reverse_edges = {}
        
    def add_transition(self, from_state: str, action: int, to_state: str):
        if from_state not in self.edges:
            self.edges[from_state] = []
        
        # Avoid duplicates
        edge = (action, to_state)
        if edge not in self.edges[from_state]:
            self.edges[from_state].append(edge)
        
        # Reverse edge
        if to_state not in self.reverse_edges:
            self.reverse_edges[to_state] = []
        rev_edge = (action, from_state)
        if rev_edge not in self.reverse_edges[to_state]:
            self.reverse_edges[to_state].append(rev_edge)
    
    def get_neighbors(self, state: str):
        """Get (action, next_state) pairs from this state."""
        return self.edges.get(state, [])
    
    def find_path_to_valuable(self, current: str, value_fn, max_depth: int = 10):
        """
        BFS to find a path to a valuable state.
        Returns list of actions to take, or None if no valuable state found.
        """
        from collections import deque as bfs_queue
        
        visited = {current}
        queue = bfs_queue([(current, [])])
        
        best_value = value_fn(current, None)
        best_path = []
        
        while queue and len(visited) < 1000:
            state, path = queue.popleft()
            
            if len(path) >= max_depth:
                continue
            
            for action, next_state in self.get_neighbors(state):
                if next_state in visited:
                    continue
                
                visited.add(next_state)
                new_path = path + [action]
                
                # Check value of this state
                value = value_fn(next_state, action)
                if value > best_value:
                    best_value = value
                    best_path = new_path
                
                queue.append((next_state, new_path))
        
        return best_path if best_path else None


class GoalInferenceAgent:
    """
    Agent that infers goals from rewards and plans toward them.
    """
    
    def __init__(self, exploration_rate: float = 0.3):
        self.reward_predictor = RewardPredictor()
        self.state_graph = StateGraph()
        self.exploration_rate = exploration_rate
        
        # State tracking
        self.last_state_hash = None
        self.last_action = None
        self.last_level = 0
        self.total_actions = 0
        self.total_rewards = 0
        
        # Current plan
        self.current_plan = []
        
    def hash_frame(self, frame_data) -> str:
        if hasattr(frame_data, 'frame') and frame_data.frame is not None:
            return self.reward_predictor.hash_frame(frame_data.frame)
        return f"level_{frame_data.levels_completed}"
    
    def choose_action(self, frame_data, available_actions) -> GameAction:
        current_hash = self.hash_frame(frame_data)
        current_level = frame_data.levels_completed if hasattr(frame_data, 'levels_completed') else 0
        
        # Check for reward (level completion)
        if current_level > self.last_level:
            self.reward_predictor.assign_credit(reward=1.0)
            self.total_rewards += 1
            self.current_plan = []  # Reset plan on new level
        
        # Update state graph
        if self.last_state_hash is not None and self.last_action is not None:
            self.state_graph.add_transition(
                self.last_state_hash,
                self.last_action.value,
                current_hash
            )
            self.reward_predictor.record_step(self.last_state_hash, self.last_action.value)
        
        # Decide: follow plan, explore, or plan
        action = None
        
        # Try to follow existing plan
        if self.current_plan:
            planned_action_id = self.current_plan.pop(0)
            planned_action = int_to_action(planned_action_id)
            if planned_action in available_actions:
                action = planned_action
        
        # If no plan or plan invalid, maybe make a new plan
        if action is None and np.random.random() > self.exploration_rate:
            # Try to plan toward valuable states
            def value_fn(state, act):
                if act is None:
                    return 0
                return self.reward_predictor.get_action_value(state, act)
            
            plan = self.state_graph.find_path_to_valuable(current_hash, value_fn)
            if plan:
                self.current_plan = plan
                planned_action_id = self.current_plan.pop(0)
                action = int_to_action(planned_action_id)
        
        # If still no action, explore
        if action is None or action not in available_actions:
            # Prefer actions with high learned value
            action_values = {}
            for a in available_actions:
                if a == GameAction.RESET:
                    continue
                value = self.reward_predictor.get_action_value(current_hash, a.value)
                # Add noise for exploration
                action_values[a] = value + np.random.uniform(0, 0.5)
            
            if action_values:
                # Softmax-ish selection
                if np.random.random() < self.exploration_rate:
                    action = np.random.choice(list(action_values.keys()))
                else:
                    action = max(action_values, key=action_values.get)
            else:
                valid = [a for a in available_actions if a != GameAction.RESET]
                action = np.random.choice(valid) if valid else GameAction.RESET
        
        # Handle ACTION6
        if action == GameAction.ACTION6:
            action.set_data({
                "x": np.random.randint(0, 64),
                "y": np.random.randint(0, 64)
            })
        
        self.last_state_hash = current_hash
        self.last_action = action
        self.last_level = current_level
        self.total_actions += 1
        
        return action
    
    def get_stats(self) -> dict:
        return {
            "total_actions": self.total_actions,
            "total_rewards": self.total_rewards,
            "unique_states": len(self.state_graph.edges),
            "learned_values": len(self.reward_predictor.credits),
            "plan_length": len(self.current_plan)
        }


def run_game(game_id: str, max_actions: int = 5000, exploration_rate: float = 0.3):
    """Run the goal inference agent on a game."""
    arc = arc_agi.Arcade()
    env = arc.make(game_id)
    agent = GoalInferenceAgent(exploration_rate=exploration_rate)
    
    print(f"\n{'='*60}")
    print(f"Goal Inference Agent on {game_id}")
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
            print(f"  [{i}] Levels: {current_level}, Values: {stats['learned_values']}, "
                  f"States: {stats['unique_states']}")
    
    stats = agent.get_stats()
    print(f"\nFinal Statistics:")
    print(f"  Total Actions: {stats['total_actions']}")
    print(f"  Total Rewards: {stats['total_rewards']}")
    print(f"  Unique States: {stats['unique_states']}")
    print(f"  Learned Values: {stats['learned_values']}")
    print(f"  Levels Completed: {current_level}")
    
    return arc.get_scorecard(), stats


if __name__ == "__main__":
    import sys
    game = sys.argv[1] if len(sys.argv) > 1 else "ls20"
    max_actions = int(sys.argv[2]) if len(sys.argv) > 2 else 5000
    run_game(game, max_actions=max_actions)
