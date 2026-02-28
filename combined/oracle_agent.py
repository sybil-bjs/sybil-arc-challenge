"""
Oracle Agent for ARC-AGI-3

Combines world knowledge (LLM) with learned action prediction.

Key insight from Bridget: Humans use ALL their world knowledge when
playing new games. They recognize game types ("this looks like a maze")
and apply known strategies before taking any actions.

Architecture:
1. CLASSIFY: Identify game archetype (maze, sokoban, pattern-match, etc.)
2. LOAD PRIORS: Get archetype-specific strategies and goals
3. PREDICT: Use action predictor to find productive actions
4. EXECUTE: Apply strategy with hypothesis testing
5. REFINE: Update beliefs based on outcomes

This agent leverages LLM reasoning to bootstrap understanding,
rather than learning everything from scratch.
"""

import arc_agi
from arcengine import GameAction
import numpy as np
from collections import defaultdict
import hashlib
import json

# Map integers to GameAction
INT_TO_ACTION = {a.value: a for a in GameAction}

def int_to_action(i: int) -> GameAction:
    return INT_TO_ACTION.get(i, GameAction.ACTION1)


# ============================================================================
# GAME ARCHETYPES AND PRIORS
# ============================================================================

GAME_ARCHETYPES = {
    "MAZE": {
        "description": "Navigate through a map to reach a goal",
        "visual_cues": ["distinct start/end colors", "paths and walls", "single controllable object"],
        "goal_heuristic": "Get the player object to the goal area",
        "action_priority": [1, 2, 3, 4],  # Directional movement
        "strategies": [
            "Move toward visually distinct goal area",
            "Avoid revisiting same positions",
            "Try all directions when stuck",
            "Look for keys/items that unlock paths"
        ],
        "failure_patterns": ["moving into walls", "going in circles"]
    },
    "SOKOBAN": {
        "description": "Push objects to target locations",
        "visual_cues": ["movable objects", "target markers", "player + boxes"],
        "goal_heuristic": "Place all pushable objects on target markers",
        "action_priority": [1, 2, 3, 4, 5],  # Movement + interact
        "strategies": [
            "Push objects toward targets",
            "Avoid pushing objects into corners (stuck)",
            "Plan pushes in reverse from goal",
            "Sometimes need to push objects out of the way first"
        ],
        "failure_patterns": ["object stuck in corner", "blocking own path"]
    },
    "PATTERN_MATCH": {
        "description": "Complete or match visual patterns",
        "visual_cues": ["partial symmetry", "matching colors/shapes", "grid of elements"],
        "goal_heuristic": "Make the pattern complete/symmetric",
        "action_priority": [6, 5, 1, 2, 3, 4],  # Click + interact often key
        "strategies": [
            "Look for incomplete symmetry",
            "Match similar colored elements",
            "Complete partial patterns",
            "Try clicking on highlighted elements"
        ],
        "failure_patterns": ["breaking existing symmetry", "random clicking"]
    },
    "ORCHESTRATION": {
        "description": "Manipulate multiple objects to achieve a state",
        "visual_cues": ["multiple controllable objects", "levels/bars", "no single player"],
        "goal_heuristic": "Get all objects to target states",
        "action_priority": [6, 5],  # Click-based manipulation
        "strategies": [
            "Click on objects to change their state",
            "Match target indicators",
            "Adjust levels/values to match goals",
            "Look for feedback on what's correct"
        ],
        "failure_patterns": ["overshooting targets", "ignoring feedback"]
    },
    "LOGIC_PUZZLE": {
        "description": "Satisfy constraints through trial and error",
        "visual_cues": ["abstract symbols", "constraint indicators", "toggle states"],
        "goal_heuristic": "Satisfy all visible constraints",
        "action_priority": [5, 6, 1, 2, 3, 4],
        "strategies": [
            "Try systematic exploration of options",
            "Remember which combinations failed",
            "Look for constraint satisfaction feedback",
            "Try reversing failed actions"
        ],
        "failure_patterns": ["repeating failed combinations", "random guessing"]
    },
    "UNKNOWN": {
        "description": "Game type not recognized",
        "visual_cues": [],
        "goal_heuristic": "Explore to discover the goal",
        "action_priority": [1, 2, 3, 4, 5, 6],
        "strategies": [
            "Try all actions to see effects",
            "Track what changes the frame",
            "Look for score/level indicators",
            "Build understanding through exploration"
        ],
        "failure_patterns": []
    }
}


# ============================================================================
# VISUAL ANALYZER (simplified - Sybil's will be more sophisticated)
# ============================================================================

def analyze_frame(frame) -> dict:
    """Analyze a frame to extract visual features."""
    if frame is None:
        return {"error": "No frame"}
    
    arr = np.array(frame)
    if len(arr.shape) == 3:
        arr = arr[0]
    
    unique_colors = np.unique(arr)
    color_counts = {int(c): int(np.sum(arr == c)) for c in unique_colors}
    
    # Find regions
    regions = []
    for c in unique_colors:
        if c == 0:  # Skip background
            continue
        mask = (arr == c)
        if np.sum(mask) > 0:
            rows, cols = np.where(mask)
            regions.append({
                "color": int(c),
                "count": int(np.sum(mask)),
                "center": (float(np.mean(rows)), float(np.mean(cols))),
                "bounds": (int(rows.min()), int(rows.max()), int(cols.min()), int(cols.max()))
            })
    
    # Detect potential features
    features = {
        "shape": arr.shape,
        "num_colors": len(unique_colors),
        "color_counts": color_counts,
        "regions": regions,
        "has_symmetry": detect_symmetry(arr),
        "has_isolated_object": any(r["count"] < 50 for r in regions),
        "has_large_regions": any(r["count"] > 500 for r in regions),
    }
    
    return features


def detect_symmetry(arr) -> bool:
    """Check if frame has approximate symmetry."""
    # Horizontal symmetry
    h_sym = np.mean(arr == np.fliplr(arr)) > 0.8
    # Vertical symmetry
    v_sym = np.mean(arr == np.flipud(arr)) > 0.8
    return h_sym or v_sym


def frame_to_description(frame) -> str:
    """Convert frame to text description for LLM."""
    features = analyze_frame(frame)
    
    if "error" in features:
        return "Empty or invalid frame"
    
    desc_parts = [
        f"Grid size: {features['shape'][0]}x{features['shape'][1]}",
        f"Colors used: {features['num_colors']}",
    ]
    
    if features["has_symmetry"]:
        desc_parts.append("Frame shows symmetry")
    
    if features["has_isolated_object"]:
        desc_parts.append("Contains small isolated objects (possible player/items)")
    
    if features["has_large_regions"]:
        desc_parts.append("Contains large colored regions (possible goals/areas)")
    
    # Describe regions
    for r in features["regions"][:5]:
        desc_parts.append(
            f"Color {r['color']}: {r['count']} pixels at center ({r['center'][0]:.0f}, {r['center'][1]:.0f})"
        )
    
    return ". ".join(desc_parts)


# ============================================================================
# GAME CLASSIFIER (simplified - Sybil's will use full LLM reasoning)
# ============================================================================

def classify_game(frame, available_actions: list) -> str:
    """
    Classify game into an archetype based on visual features.
    
    This is a heuristic version. Sybil's game_classifier.py will
    use actual LLM reasoning for analogical classification.
    """
    features = analyze_frame(frame)
    
    if "error" in features:
        return "UNKNOWN"
    
    # Heuristics for classification
    scores = defaultdict(float)
    
    # MAZE indicators
    if features["has_isolated_object"] and features["has_large_regions"]:
        scores["MAZE"] += 2
    if set(available_actions) == {1, 2, 3, 4}:
        scores["MAZE"] += 1
    
    # ORCHESTRATION indicators  
    if 6 in available_actions and len(available_actions) <= 2:
        scores["ORCHESTRATION"] += 3
    
    # PATTERN_MATCH indicators
    if features["has_symmetry"]:
        scores["PATTERN_MATCH"] += 2
    
    # LOGIC_PUZZLE indicators
    if features["num_colors"] > 5 and 5 in available_actions:
        scores["LOGIC_PUZZLE"] += 1
    
    if not scores:
        return "UNKNOWN"
    
    return max(scores, key=scores.get)


# ============================================================================
# ACTION PREDICTOR (from action_predictor_agent.py)
# ============================================================================

class ActionPredictor:
    """Predicts which actions cause frame changes."""
    
    def __init__(self):
        self.stats = defaultdict(lambda: {"attempts": 0, "changes": 0})
    
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
    
    def observe(self, state_hash: str, action: int, caused_change: bool):
        key = (state_hash, action)
        self.stats[key]["attempts"] += 1
        if caused_change:
            self.stats[key]["changes"] += 1
    
    def predict_change_prob(self, state_hash: str, action: int) -> float:
        key = (state_hash, action)
        data = self.stats[key]
        if data["attempts"] == 0:
            return 0.5  # Prior
        return (data["changes"] + 1) / (data["attempts"] + 2)


# ============================================================================
# ORACLE AGENT
# ============================================================================

class OracleAgent:
    """
    Agent that uses world knowledge to classify games and apply
    archetype-specific strategies.
    """
    
    def __init__(self, exploration_budget: int = 30):
        self.action_predictor = ActionPredictor()
        self.exploration_budget = exploration_budget
        
        # State
        self.game_type = None
        self.priors = None
        self.exploration_phase = True
        self.actions_taken = 0
        self.levels_completed = 0
        
        # Memory
        self.last_state_hash = None
        self.last_action = None
        self.visited_states = set()
        self.hypothesis = None
        
    def classify_and_load_priors(self, frame, available_actions):
        """Classify game type and load priors."""
        self.game_type = classify_game(frame, available_actions)
        self.priors = GAME_ARCHETYPES.get(self.game_type, GAME_ARCHETYPES["UNKNOWN"])
        
        print(f"  🎮 Classified as: {self.game_type}")
        print(f"  📋 Strategy: {self.priors['goal_heuristic']}")
    
    def choose_action(self, frame_data, available_actions) -> GameAction:
        frame = frame_data.frame if hasattr(frame_data, 'frame') else None
        current_hash = self.action_predictor.hash_frame(frame)
        current_level = frame_data.levels_completed if hasattr(frame_data, 'levels_completed') else 0
        
        # Check for level completion
        if current_level > self.levels_completed:
            self.levels_completed = current_level
            self.exploration_phase = True  # Re-explore on new level
            self.actions_taken = 0
        
        # First action: classify game
        if self.game_type is None:
            self.classify_and_load_priors(frame, available_actions)
        
        # Update action predictor
        if self.last_state_hash is not None and self.last_action is not None:
            caused_change = (current_hash != self.last_state_hash)
            self.action_predictor.observe(
                self.last_state_hash, 
                self.last_action.value, 
                caused_change
            )
        
        self.visited_states.add(current_hash)
        
        # Convert available actions
        if isinstance(available_actions[0], int):
            available_actions = [int_to_action(a) for a in available_actions]
        
        valid_actions = [a for a in available_actions if a != GameAction.RESET]
        
        # Choose action based on phase
        if self.exploration_phase and self.actions_taken < self.exploration_budget:
            action = self._explore_action(current_hash, valid_actions)
        else:
            self.exploration_phase = False
            action = self._exploit_action(current_hash, valid_actions)
        
        # Handle ACTION6 coordinates
        if action == GameAction.ACTION6:
            action = self._choose_coordinates(frame, action)
        
        self.last_state_hash = current_hash
        self.last_action = action
        self.actions_taken += 1
        
        return action
    
    def _explore_action(self, state_hash: str, valid_actions: list) -> GameAction:
        """Exploration phase: try actions to learn what works."""
        # Prioritize actions from archetype priors
        priority = self.priors.get("action_priority", [1, 2, 3, 4, 5, 6])
        
        # Try untested actions first
        for action_id in priority:
            action = int_to_action(action_id)
            if action in valid_actions:
                key = (state_hash, action_id)
                if self.action_predictor.stats[key]["attempts"] == 0:
                    return action
        
        # Otherwise, random from priority list
        for action_id in priority:
            action = int_to_action(action_id)
            if action in valid_actions:
                return action
        
        return np.random.choice(valid_actions)
    
    def _exploit_action(self, state_hash: str, valid_actions: list) -> GameAction:
        """Exploitation phase: use learned knowledge."""
        # Score actions by change probability + archetype priority
        action_scores = {}
        priority = self.priors.get("action_priority", [1, 2, 3, 4, 5, 6])
        
        for action in valid_actions:
            change_prob = self.action_predictor.predict_change_prob(state_hash, action.value)
            
            # Bonus for archetype-preferred actions
            priority_bonus = 0
            if action.value in priority:
                priority_bonus = 0.1 * (len(priority) - priority.index(action.value)) / len(priority)
            
            # Penalty for revisiting known states
            key = (state_hash, action.value)
            # (would need transition tracking for full implementation)
            
            action_scores[action] = change_prob + priority_bonus
        
        # Select best action with some exploration
        if np.random.random() < 0.1:
            return np.random.choice(valid_actions)
        
        return max(action_scores, key=action_scores.get)
    
    def _choose_coordinates(self, frame, action: GameAction) -> GameAction:
        """Choose coordinates for ACTION6 based on visual analysis."""
        if frame is None:
            x, y = np.random.randint(0, 64), np.random.randint(0, 64)
        else:
            arr = np.array(frame)
            if len(arr.shape) == 3:
                arr = arr[0]
            
            # Click on non-background pixels (more likely interactive)
            nonzero = np.argwhere(arr > 0)
            if len(nonzero) > 0:
                idx = np.random.randint(len(nonzero))
                y, x = nonzero[idx]
            else:
                x, y = np.random.randint(0, 64), np.random.randint(0, 64)
        
        action.set_data({"x": int(x), "y": int(y)})
        return action
    
    def get_stats(self) -> dict:
        return {
            "game_type": self.game_type,
            "actions_taken": self.actions_taken,
            "levels_completed": self.levels_completed,
            "unique_states": len(self.visited_states),
            "exploration_phase": self.exploration_phase
        }


# ============================================================================
# RUNNER
# ============================================================================

def run_game(game_id: str, max_actions: int = 5000, exploration_budget: int = 30):
    """Run the Oracle Agent on a game."""
    arc = arc_agi.Arcade()
    env = arc.make(game_id)
    agent = OracleAgent(exploration_budget=exploration_budget)
    
    print(f"\n{'='*60}")
    print(f"Oracle Agent on {game_id}")
    print(f"Exploration budget: {exploration_budget} actions")
    print(f"{'='*60}")
    
    frame = env.reset()
    current_level = 0
    level_start = 0
    
    for i in range(max_actions):
        available = frame.available_actions if hasattr(frame, 'available_actions') else [1, 2, 3, 4]
        
        action = agent.choose_action(frame, available)
        frame = env.step(action)
        
        if hasattr(frame, 'levels_completed') and frame.levels_completed > current_level:
            print(f"  ✓ Level {current_level + 1} in {i + 1 - level_start} actions")
            current_level = frame.levels_completed
            level_start = i + 1
        
        if hasattr(frame, 'state') and str(frame.state) == 'WIN':
            print(f"\n🎉 Game WON in {agent.actions_taken} actions!")
            break
        
        if i % 1000 == 0 and i > 0:
            stats = agent.get_stats()
            print(f"  [{i}] Type: {stats['game_type']}, States: {stats['unique_states']}")
    
    stats = agent.get_stats()
    print(f"\nFinal Statistics:")
    print(f"  Game Type: {stats['game_type']}")
    print(f"  Total Actions: {stats['actions_taken']}")
    print(f"  Unique States: {stats['unique_states']}")
    print(f"  Levels Completed: {stats['levels_completed']}")
    
    return arc.get_scorecard(), stats


if __name__ == "__main__":
    import sys
    game = sys.argv[1] if len(sys.argv) > 1 else "ls20"
    max_actions = int(sys.argv[2]) if len(sys.argv) > 2 else 5000
    run_game(game, max_actions=max_actions)
