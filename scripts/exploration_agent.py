#!/usr/bin/env python3
"""
Systematic exploration agent for ARC-AGI-3 games.

Phase 1: EXPLORE — Learn game mechanics
Phase 2: PLAN — Build strategy from learned rules
Phase 3: EXECUTE — Win efficiently

This agent doesn't try to win immediately. It invests actions in understanding first.
"""

import json
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from pathlib import Path

@dataclass
class GameMechanic:
    """A learned rule about how the game works."""
    mechanic_type: str  # "movement", "collision", "interaction", "goal"
    description: str
    evidence: List[Dict]  # Frames/actions that demonstrate this
    confidence: float = 0.5
    
    def to_dict(self):
        return {
            "type": self.mechanic_type,
            "description": self.description,
            "confidence": self.confidence,
            "evidence_count": len(self.evidence)
        }

@dataclass 
class ExplorationState:
    """Track what we've learned about a game."""
    game_id: str
    mechanics: List[GameMechanic] = field(default_factory=list)
    action_effects: Dict[str, List[Dict]] = field(default_factory=dict)  # action -> observed effects
    explored_positions: set = field(default_factory=set)
    goals_identified: List[Dict] = field(default_factory=list)
    dangers_identified: List[Dict] = field(default_factory=list)
    exploration_complete: bool = False
    total_exploration_actions: int = 0

class ExplorationAgent:
    """
    Agent that systematically explores before trying to win.
    """
    
    def __init__(self, game_id: str, max_exploration_actions: int = 50):
        self.game_id = game_id
        self.state = ExplorationState(game_id=game_id)
        self.max_exploration = max_exploration_actions
        self.previous_frame = None
        self.frame_history: List = []
        
    def explore_action(self, action_name: str, frame_before, frame_after, result_info: dict):
        """Record what happened when we took an action."""
        from visual_analyzer import compare_frames, analyze_frame
        
        diff = compare_frames(frame_before, frame_after)
        
        effect = {
            "action": action_name,
            "changed": diff["changed"],
            "changed_pixels": diff.get("changed_pixels", 0),
            "movement": diff.get("possible_movement"),
            "result": result_info
        }
        
        # Store effect for this action
        if action_name not in self.state.action_effects:
            self.state.action_effects[action_name] = []
        self.state.action_effects[action_name].append(effect)
        
        self.state.total_exploration_actions += 1
        
        # Try to infer mechanics
        self._infer_mechanics(action_name, diff, frame_before, frame_after)
        
        return effect
    
    def _infer_mechanics(self, action: str, diff: dict, frame_before, frame_after):
        """Try to learn game rules from observations."""
        
        # Movement detection
        if diff.get("possible_movement"):
            mechanic = GameMechanic(
                mechanic_type="movement",
                description=f"{action} causes movement of colors {diff['possible_movement']}",
                evidence=[{"action": action, "diff": diff}],
                confidence=0.7
            )
            self._add_or_update_mechanic(mechanic)
        
        # No change = possible wall/boundary
        if not diff["changed"]:
            mechanic = GameMechanic(
                mechanic_type="collision",
                description=f"{action} had no effect - possible boundary or obstacle",
                evidence=[{"action": action}],
                confidence=0.5
            )
            self._add_or_update_mechanic(mechanic)
        
        # Check for level completion, score changes, etc.
        # (Would need result_info from the game)
    
    def _add_or_update_mechanic(self, new_mechanic: GameMechanic):
        """Add new mechanic or update confidence of existing one."""
        for existing in self.state.mechanics:
            if existing.mechanic_type == new_mechanic.mechanic_type and \
               existing.description == new_mechanic.description:
                # Increase confidence with more evidence
                existing.confidence = min(0.95, existing.confidence + 0.1)
                existing.evidence.extend(new_mechanic.evidence)
                return
        
        self.state.mechanics.append(new_mechanic)
    
    def get_exploration_plan(self, available_actions: List[str]) -> List[str]:
        """
        Generate a systematic exploration plan.
        
        Strategy:
        1. Try each action once in isolation
        2. Try each action multiple times to detect patterns
        3. Try action combinations
        """
        plan = []
        
        # Phase 1: Try each action once
        for action in available_actions:
            if action not in self.state.action_effects:
                plan.append(action)
        
        # Phase 2: Repeat actions to verify effects
        for action in available_actions:
            effects = self.state.action_effects.get(action, [])
            if len(effects) < 3:  # Need at least 3 samples
                plan.extend([action] * (3 - len(effects)))
        
        # Limit to max exploration
        remaining = self.max_exploration - self.state.total_exploration_actions
        return plan[:remaining]
    
    def should_explore(self) -> bool:
        """Decide if we should keep exploring or switch to execution."""
        if self.state.exploration_complete:
            return False
        
        if self.state.total_exploration_actions >= self.max_exploration:
            self.state.exploration_complete = True
            return False
        
        # Check if we have enough confidence in mechanics
        high_confidence = sum(1 for m in self.state.mechanics if m.confidence > 0.8)
        if high_confidence >= 3:  # Know at least 3 things well
            self.state.exploration_complete = True
            return False
        
        return True
    
    def get_learned_summary(self) -> str:
        """Generate a summary of what we learned for strategy planning."""
        summary = f"=== Exploration Summary for {self.game_id} ===\n"
        summary += f"Actions taken: {self.state.total_exploration_actions}\n\n"
        
        summary += "Learned Mechanics:\n"
        for m in sorted(self.state.mechanics, key=lambda x: -x.confidence):
            summary += f"  [{m.confidence:.0%}] {m.mechanic_type}: {m.description}\n"
        
        summary += "\nAction Effects:\n"
        for action, effects in self.state.action_effects.items():
            changes = sum(1 for e in effects if e.get("changed"))
            summary += f"  {action}: {changes}/{len(effects)} caused changes\n"
        
        return summary
    
    def save_state(self, path: str):
        """Save exploration state for future reference."""
        data = {
            "game_id": self.state.game_id,
            "mechanics": [m.to_dict() for m in self.state.mechanics],
            "action_effects": {k: v for k, v in self.state.action_effects.items()},
            "total_actions": self.state.total_exploration_actions,
            "exploration_complete": self.state.exploration_complete
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load_state(cls, path: str) -> 'ExplorationAgent':
        """Load previous exploration state."""
        with open(path) as f:
            data = json.load(f)
        
        agent = cls(data["game_id"])
        agent.state.total_exploration_actions = data["total_actions"]
        agent.state.exploration_complete = data["exploration_complete"]
        agent.state.action_effects = data["action_effects"]
        
        for m_data in data["mechanics"]:
            mechanic = GameMechanic(
                mechanic_type=m_data["type"],
                description=m_data["description"],
                evidence=[],
                confidence=m_data["confidence"]
            )
            agent.state.mechanics.append(mechanic)
        
        return agent

if __name__ == "__main__":
    # Demo
    agent = ExplorationAgent("ls20", max_exploration_actions=20)
    plan = agent.get_exploration_plan(["ACTION1", "ACTION2", "ACTION3", "ACTION4"])
    print(f"Exploration plan: {plan}")
