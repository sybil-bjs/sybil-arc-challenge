"""
Oracle Agent for ARC-AGI-3 (Unified Version)

Combines world knowledge (LLM) with learned action prediction and adaptive learning.
Lead: Saber ⚔️ | ML Integration: Sybil 🧠

Architecture:
1. HUMAN KNOWLEDGE: Check for human-provided strategies first (knowledge/human_strategies/)
2. CLASSIFY: Identify game archetype (MAZE, SOKOBAN, etc.) via Sybil's AdaptiveClassifier.
3. ENRICH: Load learned priors and successful strategies from previous sessions.
4. PREDICT: Use action predictor to find productive actions (causes frame changes).
5. EXECUTE: Apply archetype-specific strategies with hypothesis testing.
6. RECORD: Feed learnings back into Sybil's Adaptive Learning System.
"""

import os
import sys
from pathlib import Path
import numpy as np
import hashlib
from collections import defaultdict
from dataclasses import asdict
import re

# Human strategy loader
KNOWLEDGE_DIR = Path(__file__).parent.parent / "knowledge" / "human_strategies"

def load_human_strategy(game_id: str) -> dict:
    """Load human-provided strategies for a game if available."""
    # Try exact match first, then prefix match
    game_base = game_id.split("-")[0]  # e.g., "ls20" from "ls20-cb3b57cc"
    
    for pattern in [f"{game_id}*.md", f"{game_base}*.md"]:
        matches = list(KNOWLEDGE_DIR.glob(pattern)) if KNOWLEDGE_DIR.exists() else []
        if matches:
            with open(matches[0], "r") as f:
                content = f.read()
            
            # Extract key mechanics
            strategy = {
                "has_human_knowledge": True,
                "source": str(matches[0].name),
                "elements": [],
                "win_condition": None,
                "priority_targets": [],
                "constraints": [],
            }
            
            # Parse elements table if present
            if "| Element |" in content:
                for line in content.split("\n"):
                    if "|" in line and not line.startswith("|--") and "Element" not in line:
                        parts = [p.strip() for p in line.split("|") if p.strip()]
                        if len(parts) >= 2:
                            strategy["elements"].append({"name": parts[0], "function": parts[-1]})
            
            # Extract priority targets
            if "plus sign" in content.lower() or "plus button" in content.lower():
                strategy["priority_targets"].append("plus_sign")
            if "rainbow" in content.lower():
                strategy["priority_targets"].append("rainbow_block")
            if "yellow" in content.lower() and "pickup" in content.lower():
                strategy["priority_targets"].append("yellow_pickup")
            
            # Extract constraints
            if "single-use" in content.lower():
                strategy["constraints"].append("plus_button_single_use")
            if "move budget" in content.lower() or "life bar" in content.lower():
                strategy["constraints"].append("limited_moves")
            
            # Win condition
            if "combine" in content.lower():
                strategy["win_condition"] = "combine_matching_patterns"
            
            return strategy
    
    return {"has_human_knowledge": False}

# Add scripts directory to path for imports
sys.path.append(os.path.dirname(__file__))

import arc_agi
from arcengine import GameAction
from scripts.adaptive_classifier import AdaptiveClassifier, GameLearning, record_learning
from scripts.game_classifier import GameArchetype

# Map integers to GameAction
INT_TO_ACTION = {a.value: a for a in GameAction}

def int_to_action(i: int) -> GameAction:
    return INT_TO_ACTION.get(i, GameAction.ACTION1)


class ActionPredictor:
    """Predicts which actions cause frame changes."""
    def __init__(self):
        self.stats = defaultdict(lambda: {"attempts": 0, "changes": 0})
    
    def hash_frame(self, frame) -> str:
        if frame is None: return "none"
        data = str(frame).encode() if isinstance(frame, list) else (frame.tobytes() if hasattr(frame, 'tobytes') else str(frame).encode())
        return hashlib.md5(data).hexdigest()[:12]
    
    def observe(self, state_hash: str, action: int, caused_change: bool):
        key = (state_hash, action)
        self.stats[key]["attempts"] += 1
        if caused_change: self.stats[key]["changes"] += 1
    
    def predict_change_prob(self, state_hash: str, action: int) -> float:
        key = (state_hash, action)
        data = self.stats[key]
        if data["attempts"] == 0: return 0.5
        return (data["changes"] + 1) / (data["attempts"] + 2)


class OracleAgent:
    def __init__(self, exploration_budget: int = 50):
        self.classifier = AdaptiveClassifier()
        self.action_predictor = ActionPredictor()
        self.exploration_budget = exploration_budget
        
        # Session State
        self.game_id = None
        self.classification = None
        self.human_strategy = None  # NEW: Human-provided knowledge
        self.actions_taken = 0
        self.levels_completed = 0
        self.total_actions = 0
        
        # Tracking
        self.last_state_hash = None
        self.last_action = None
        self.visited_states = set()
        
        # Learning Stats for record_learning
        self.successful_strategies = []
        self.failed_strategies = []
        self.discovered_mechanics = []

    def choose_action(self, frame_data, available_actions) -> GameAction:
        frame = np.array(frame_data.frame) if hasattr(frame_data, 'frame') else None
        current_hash = self.action_predictor.hash_frame(frame)
        current_level = frame_data.levels_completed if hasattr(frame_data, 'levels_completed') else 0
        
        # Identify Game ID if not set
        if self.game_id is None:
            self.game_id = getattr(frame_data, 'game_id', 'unknown')

        # 0. HUMAN KNOWLEDGE (highest priority!)
        if self.human_strategy is None:
            self.human_strategy = load_human_strategy(self.game_id)
            if self.human_strategy.get("has_human_knowledge"):
                print(f"  📚 HUMAN KNOWLEDGE LOADED: {self.human_strategy['source']}")
                print(f"     Priority targets: {self.human_strategy['priority_targets']}")
                print(f"     Constraints: {self.human_strategy['constraints']}")
                print(f"     Win condition: {self.human_strategy['win_condition']}")

        # 1. CLASSIFY (or retrieve from memory)
        if self.classification is None:
            print(f"  🔍 Oracle is classifying game: {self.game_id}...")
            self.classification = self.classifier.classify(frame, self.game_id)
            print(f"  🎮 Archetype: {self.classification.archetype.value} (Conf: {self.classification.confidence:.2f})")
            print(f"  📋 Reasoning: {self.classification.reasoning}")

        # 2. Update level progress
        if current_level > self.levels_completed:
            print(f"  ✨ Level {current_level} completed! Resetting exploration budget.")
            self.levels_completed = current_level
            self.actions_taken = 0
            # Track as successful strategy
            if self.last_action:
                self.successful_strategies.append(f"level_win_with_action_{self.last_action.value}")

        # 3. Update Predictor with result of last action
        if self.last_state_hash is not None and self.last_action is not None:
            caused_change = (current_hash != self.last_state_hash)
            self.action_predictor.observe(self.last_state_hash, self.last_action.value, caused_change)
            if not caused_change:
                self.failed_strategies.append(f"no_op_{self.last_action.value}_at_{self.last_state_hash}")
            else:
                # Discovered a mechanic: action X changes Y pixels
                if hasattr(self, 'last_frame') and frame is not None and self.last_frame is not None and frame.shape == self.last_frame.shape:
                    pixels_changed = np.sum(frame != self.last_frame)
                    self.discovered_mechanics.append(f"action_{self.last_action.value}_changes_{pixels_changed}_px")

        self.visited_states.add(current_hash)
        self.last_frame = frame

        # Convert available actions
        if isinstance(available_actions[0], int):
            available_actions = [int_to_action(a) for a in available_actions]
        valid_actions = [a for a in available_actions if a != GameAction.RESET]

        # 4. CHOOSE STRATEGY
        if self.actions_taken < self.exploration_budget:
            action = self._explore(current_hash, valid_actions)
        else:
            action = self._exploit(current_hash, valid_actions)

        # Handle ACTION6 coordinates
        if action == GameAction.ACTION6:
            action = self._choose_coordinates(frame, action)

        self.last_state_hash = current_hash
        self.last_action = action
        self.actions_taken += 1
        self.total_actions += 1
        
        return action

    def _explore(self, state_hash: str, valid_actions: list) -> GameAction:
        """Goal-directed exploration using human priors + Bayesian updates."""
        
        # HUMAN GOAL PRIORS (highest priority)
        # If we have human knowledge, use it to direct exploration
        if self.human_strategy and self.human_strategy.get("has_human_knowledge"):
            priority_targets = self.human_strategy.get("priority_targets", [])
            
            # Goal-directed action selection based on human priors
            # "What would a human try to do here?"
            if "plus_sign" in priority_targets:
                # Human goal: reach the action point
                # Directional actions (1-4) are likely movement
                movement_actions = [a for a in valid_actions if a.value in [1, 2, 3, 4]]
                if movement_actions:
                    # Try untested movements first (exploring toward goal)
                    for action in movement_actions:
                        if self.action_predictor.stats[(state_hash, action.value)]["attempts"] == 0:
                            return action
                    # If all tested, use highest change probability
                    return max(movement_actions, 
                              key=lambda a: self.action_predictor.predict_change_prob(state_hash, a.value))
            
            if "yellow_pickup" in priority_targets:
                # Human goal: collect resources when low
                # Could trigger different behavior based on observed life bar
                pass  # TODO: visual life bar detection
        
        # Sybil's classifier suggestions (next priority)
        suggestions = [int_to_action(int(a)) for a in self.classification.suggested_first_actions if str(a).isdigit()]
        
        # Try suggestions first
        for action in suggestions:
            if action in valid_actions:
                if self.action_predictor.stats[(state_hash, action.value)]["attempts"] == 0:
                    return action
        
        # Then try any unvisited action at this state
        for action in valid_actions:
            if self.action_predictor.stats[(state_hash, action.value)]["attempts"] == 0:
                return action
                
        return np.random.choice(valid_actions)

    def _exploit(self, state_hash: str, valid_actions: list) -> GameAction:
        """Use high-probability actions according to archetype priors."""
        scores = {}
        priors = self.classification.priors
        
        for action in valid_actions:
            change_prob = self.action_predictor.predict_change_prob(state_hash, action.value)
            
            # Archetype bias
            archetype_bias = 0.2 if str(action.value) in str(priors.action_priority) else 0.0
            
            scores[action] = change_prob + archetype_bias
            
        return max(scores, key=scores.get)

    def _choose_coordinates(self, frame, action: GameAction) -> GameAction:
        if frame is not None:
            # Oracle click: prefer non-background pixels
            nonzero = np.argwhere(frame > 0)
            if len(nonzero) > 0:
                idx = np.random.randint(len(nonzero))
                y, x = nonzero[idx][-2], nonzero[idx][-1] # Handle 2D or 3D
                action.set_data({"x": int(x), "y": int(y)})
                return action
        action.set_data({"x": 32, "y": 32})
        return action

    def finalize(self):
        """Record the session learnings to the database."""
        learning = GameLearning(
            game_id=self.game_id or "unknown",
            initial_archetype=self.classification.archetype.value if self.classification else "unknown",
            final_archetype=self.classification.archetype.value if self.classification else "unknown",
            confidence_change=0.0,
            successful_strategies=list(set(self.successful_strategies)),
            failed_strategies=list(set(self.failed_strategies))[:10],
            discovered_mechanics=list(set(self.discovered_mechanics))[:10],
            levels_completed=self.levels_completed,
            total_actions=self.total_actions,
            efficiency_vs_human=0.0, # TBD
            notes=f"Oracle run. Archetype: {self.classification.archetype.value if self.classification else 'None'}"
        )
        record_learning(learning)
        print(f"  💾 Session learnings recorded to the BJS Knowledge Base.")


def run_oracle(game_id: str, max_actions: int = 500):
    arc = arc_agi.Arcade()
    env = arc.make(game_id)
    agent = OracleAgent()
    
    print(f"\n{'='*60}")
    print(f"🔮 ORACLE AGENT: {game_id}")
    print(f"{'='*60}")
    
    frame = env.reset()
    try:
        for _ in range(max_actions):
            available = frame.available_actions or [1, 2, 3, 4]
            action = agent.choose_action(frame, available)
            frame = env.step(action)
            if str(frame.state) == 'WIN':
                print("  🎉 VICTORY!")
                break
    except KeyboardInterrupt:
        pass
    finally:
        agent.finalize()
        print(f"\nFinal Stats: {agent.levels_completed} levels, {agent.total_actions} actions.")

if __name__ == "__main__":
    game = sys.argv[1] if len(sys.argv) > 1 else "ls20"
    run_oracle(game)
