"""
Goal-Directed Bayesian Agent for ARC-AGI-3

Instead of asking "which action changes pixels?", this agent asks
"what goal should I be pursuing right now?"

Architecture:
1. GOAL HYPOTHESES: Maintain beliefs over possible goals
2. GOAL-DIRECTED ACTIONS: Select actions that advance the current goal
3. BAYESIAN UPDATE: After each observation, update goal probabilities
4. LEARNING DATABASE: Persist successful goal sequences across games

Lead: Saber ⚔️ | Theory: Bridget 🎮 | ML: Sybil 🧠
"""

import os
import sys
import sqlite3
import json
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple
from enum import Enum
import numpy as np
import hashlib

sys.path.append(os.path.dirname(__file__))

# Try to import ARC-AGI, but allow standalone testing
try:
    import arc_agi
    from arcengine import GameAction
    HAS_ARC = True
except ImportError:
    HAS_ARC = False
    class GameAction:
        """Mock for testing"""
        RESET = type('obj', (object,), {'value': 0})()
        ACTION1 = type('obj', (object,), {'value': 1})()
        ACTION2 = type('obj', (object,), {'value': 2})()
        ACTION3 = type('obj', (object,), {'value': 3})()
        ACTION4 = type('obj', (object,), {'value': 4})()

# ============================================================================
# GOAL DEFINITIONS
# ============================================================================

class GoalType(Enum):
    """
    Possible high-level goals a human might have.
    
    KEY INSIGHT (Bridget): Goals are RELATIONAL, not positional.
    Not "go to (x,y)" but "overlap with the thing that matches me."
    """
    # Relational goals (preferred)
    TRANSFORM_TO_MATCH = "transform_to_match"      # Become something that can combine
    OVERLAP_WITH_MATCH = "overlap_with_match"      # Find my match, converge with it
    COLLECT_RESOURCE = "collect_resource"          # Pick up items (color-coded)
    
    # Positional goals (fallback when relations unknown)
    REACH_ACTION_POINT = "reach_action_point"      # Go to plus sign, button, etc.
    EXPLORE_REGION = "explore_region"              # Directed exploration of area
    AVOID_HAZARD = "avoid_hazard"                  # Don't step on X
    UNKNOWN = "unknown"                            # Fallback
    
    # Legacy alias
    TRANSFORM = "transform"
    COMBINE = "combine"


@dataclass
class GoalHypothesis:
    """A hypothesis about what goal to pursue."""
    goal_type: GoalType
    target_region: Optional[Tuple[int, int, int, int]] = None  # (x1, y1, x2, y2)
    prior: float = 0.1                    # P(goal) before observations
    likelihood: float = 1.0               # P(observations | goal)
    posterior: float = 0.1                # P(goal | observations)
    evidence: List[str] = field(default_factory=list)  # Why we believe this
    
    def update(self, observation_likelihood: float):
        """Bayesian update: P(goal|obs) ∝ P(obs|goal) * P(goal)"""
        self.likelihood *= observation_likelihood
        # Posterior will be normalized across all hypotheses


@dataclass 
class GoalTransition:
    """Records a successful goal sequence."""
    game_id: str
    from_goal: str
    to_goal: str
    trigger: str          # What caused the transition
    success_count: int = 1


# ============================================================================
# GENERALIZABLE HUMAN INTUITION
# ============================================================================
# These are ABSTRACT patterns that transfer across games.
# Not "ls20 has cyan plus" but "action points tend to look like X"

HUMAN_INTUITION_PATTERNS = {
    # Visual → Meaning mappings (abstract, not game-specific)
    "visual_patterns": {
        "plus_cross_shape": {
            "likely_meaning": "action_point",
            "goal_type": "reach_action_point",
            "confidence": 0.8,
            "reasoning": "Plus/cross shapes universally indicate 'do something here'"
        },
        "depleting_bar": {
            "likely_meaning": "resource_constraint",
            "goal_type": "collect_resource",
            "confidence": 0.85,
            "reasoning": "Bars that shrink indicate limited budget - be efficient"
        },
        "color_matching": {
            "likely_meaning": "relationship_indicator",
            "goal_type": "collect_resource",
            "confidence": 0.75,
            "reasoning": "Same color often means 'these things go together'"
        },
        "rainbow_multicolor": {
            "likely_meaning": "transformation_point",
            "goal_type": "transform_to_match",
            "confidence": 0.7,
            "reasoning": "Rainbow/multicolor suggests changing properties"
        },
        "matching_patterns": {
            "likely_meaning": "combine_trigger",
            "goal_type": "overlap_with_match",
            "confidence": 0.9,
            "reasoning": "When you see two things that match, bring them together"
        },
        "distinct_single_element": {
            "likely_meaning": "player_avatar",
            "goal_type": None,
            "confidence": 0.7,
            "reasoning": "A unique element is often what you control"
        },
    },
    
    # Game structure patterns (universal)
    "structural_patterns": {
        "progressive_complexity": "Each level adds ONE new concept",
        "resource_before_action": "If path seems impossible, look for pickups first",
        "transformation_before_combination": "Usually: change yourself, then merge",
        "single_use_triggers": "Action points often can only be used once",
    },
    
    # Goal sequence patterns (common flows)
    "goal_sequences": [
        ["collect_resource", "reach_action_point", "transform_to_match", "overlap_with_match"],
        ["reach_action_point", "transform_to_match", "overlap_with_match"],
        ["explore_region", "reach_action_point", "transform_to_match"],
    ],
}
    

# ============================================================================
# GOAL LEARNING DATABASE
# ============================================================================

class GoalLearningDB:
    """Persistent storage for learned goal patterns."""
    
    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = Path(__file__).parent.parent / "knowledge" / "goal_learnings.db"
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Initialize database tables."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # Goal priors learned from experience
        c.execute("""
            CREATE TABLE IF NOT EXISTS goal_priors (
                id INTEGER PRIMARY KEY,
                game_pattern TEXT,          -- e.g., "ls*" for all ls games
                goal_type TEXT,
                prior_probability REAL,
                success_count INTEGER,
                failure_count INTEGER,
                last_updated TEXT
            )
        """)
        
        # Goal transitions (what goal comes after what)
        c.execute("""
            CREATE TABLE IF NOT EXISTS goal_transitions (
                id INTEGER PRIMARY KEY,
                game_pattern TEXT,
                from_goal TEXT,
                to_goal TEXT,
                trigger_description TEXT,
                transition_count INTEGER,
                last_updated TEXT
            )
        """)
        
        # Visual goal detectors (where to look for goals)
        c.execute("""
            CREATE TABLE IF NOT EXISTS goal_detectors (
                id INTEGER PRIMARY KEY,
                goal_type TEXT,
                visual_pattern TEXT,        -- JSON: color, shape, position hints
                confidence REAL,
                game_sources TEXT,          -- Which games taught this
                last_updated TEXT
            )
        """)
        
        # Human knowledge mappings
        c.execute("""
            CREATE TABLE IF NOT EXISTS human_mappings (
                id INTEGER PRIMARY KEY,
                visual_element TEXT,        -- "plus_sign", "yellow_box"
                human_meaning TEXT,         -- "action_point", "resource"
                goal_type TEXT,
                confidence REAL,
                source TEXT                 -- "bridget", "inferred"
            )
        """)
        
        conn.commit()
        conn.close()
        
        # Seed with Bridget's knowledge
        self._seed_human_knowledge()
    
    def _seed_human_knowledge(self):
        """Seed database with human-provided mappings."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # Check if already seeded
        c.execute("SELECT COUNT(*) FROM human_mappings")
        if c.fetchone()[0] > 0:
            conn.close()
            return
        
        # Bridget's insights from ls20
        mappings = [
            ("plus_sign", "action_point", "reach_action_point", 0.95, "bridget_ls20"),
            ("yellow_box", "move_resource", "collect_resource", 0.90, "bridget_ls20"),
            ("yellow_bar", "life_budget", "avoid_waste", 0.90, "bridget_ls20"),
            ("rainbow_block", "color_changer", "transform", 0.85, "bridget_ls20"),
            ("matching_pattern", "combine_trigger", "combine", 0.95, "bridget_ls20"),
        ]
        
        for visual, meaning, goal, conf, source in mappings:
            c.execute("""
                INSERT INTO human_mappings (visual_element, human_meaning, goal_type, confidence, source)
                VALUES (?, ?, ?, ?, ?)
            """, (visual, meaning, goal, conf, source))
        
        # Seed goal transitions from ls20
        transitions = [
            ("ls*", "reach_action_point", "transform", "stepped_on_plus", 1),
            ("ls*", "transform", "combine", "pattern_matched", 1),
            ("ls*", "collect_resource", "reach_action_point", "resource_collected", 1),
        ]
        
        for pattern, from_g, to_g, trigger, count in transitions:
            c.execute("""
                INSERT INTO goal_transitions (game_pattern, from_goal, to_goal, trigger_description, transition_count, last_updated)
                VALUES (?, ?, ?, ?, ?, datetime('now'))
            """, (pattern, from_g, to_g, trigger, count))
        
        conn.commit()
        conn.close()
        print("  💾 Seeded goal database with Bridget's human knowledge")
    
    def get_goal_priors(self, game_id: str) -> Dict[str, float]:
        """Get learned goal priors for a game pattern."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # Try exact match, then prefix match
        game_base = game_id.split("-")[0]
        
        c.execute("""
            SELECT goal_type, prior_probability 
            FROM goal_priors 
            WHERE game_pattern = ? OR game_pattern = ? OR game_pattern LIKE ?
            ORDER BY success_count DESC
        """, (game_id, game_base, f"{game_base[:2]}%"))
        
        priors = {row[0]: row[1] for row in c.fetchall()}
        conn.close()
        
        return priors if priors else self._default_priors()
    
    def _default_priors(self) -> Dict[str, float]:
        """Default goal priors based on human knowledge."""
        return {
            "reach_action_point": 0.35,  # Plus signs are common first goals
            "collect_resource": 0.20,    # Resources often needed
            "transform": 0.15,           # Transformation is mid-game
            "combine": 0.15,             # Combining is usually endgame
            "explore_region": 0.10,      # Fallback exploration
            "unknown": 0.05,             # True unknown
        }
    
    def get_likely_transitions(self, current_goal: str, game_id: str) -> List[Tuple[str, float]]:
        """What goal likely comes after the current one?"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        game_base = game_id.split("-")[0]
        
        c.execute("""
            SELECT to_goal, transition_count
            FROM goal_transitions
            WHERE from_goal = ? AND (game_pattern = ? OR game_pattern LIKE ?)
            ORDER BY transition_count DESC
        """, (current_goal, game_base, f"{game_base[:2]}%"))
        
        results = c.fetchall()
        conn.close()
        
        if not results:
            return [("unknown", 1.0)]
        
        total = sum(r[1] for r in results)
        return [(r[0], r[1]/total) for r in results]
    
    def record_goal_success(self, game_id: str, goal_type: str):
        """Record that a goal was successfully achieved."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        game_base = game_id.split("-")[0]
        
        c.execute("""
            INSERT INTO goal_priors (game_pattern, goal_type, prior_probability, success_count, failure_count, last_updated)
            VALUES (?, ?, 0.5, 1, 0, datetime('now'))
            ON CONFLICT(id) DO UPDATE SET 
                success_count = success_count + 1,
                prior_probability = (success_count + 1.0) / (success_count + failure_count + 2.0),
                last_updated = datetime('now')
        """, (game_base, goal_type))
        
        conn.commit()
        conn.close()
    
    def record_transition(self, game_id: str, from_goal: str, to_goal: str, trigger: str):
        """Record a goal transition."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        game_base = game_id.split("-")[0]
        
        # Check if exists
        c.execute("""
            SELECT id, transition_count FROM goal_transitions
            WHERE game_pattern = ? AND from_goal = ? AND to_goal = ?
        """, (game_base, from_goal, to_goal))
        
        row = c.fetchone()
        if row:
            c.execute("""
                UPDATE goal_transitions 
                SET transition_count = ?, trigger_description = ?, last_updated = datetime('now')
                WHERE id = ?
            """, (row[1] + 1, trigger, row[0]))
        else:
            c.execute("""
                INSERT INTO goal_transitions (game_pattern, from_goal, to_goal, trigger_description, transition_count, last_updated)
                VALUES (?, ?, ?, ?, 1, datetime('now'))
            """, (game_base, from_goal, to_goal, trigger))
        
        conn.commit()
        conn.close()


# ============================================================================
# GOAL BAYESIAN AGENT
# ============================================================================

class GoalBayesianAgent:
    """
    Agent that maintains beliefs over goals and selects actions to achieve them.
    
    Key insight (Bridget): The prior should always be high-level human goals,
    not random state exploration. Every move updates our belief about what
    goal we should be pursuing.
    """
    
    def __init__(self):
        self.db = GoalLearningDB()
        
        # Current belief state
        self.goal_hypotheses: List[GoalHypothesis] = []
        self.current_goal: Optional[GoalHypothesis] = None
        self.goal_history: List[str] = []
        
        # Game state
        self.game_id: Optional[str] = None
        self.actions_taken: int = 0
        self.levels_completed: int = 0
        
        # Frame tracking
        self.last_frame: Optional[np.ndarray] = None
        self.last_frame_hash: Optional[str] = None
        self.frame_history: List[str] = []
        
        # Transformation tracking
        self.transformation_detected: bool = False
        self.transformation_count: int = 0
    
    def _hash_frame(self, frame: np.ndarray) -> str:
        """Hash a frame for comparison."""
        return hashlib.md5(frame.tobytes()).hexdigest()[:12]
    
    def initialize_goals(self, game_id: str, initial_frame: np.ndarray):
        """Initialize goal hypotheses based on learned priors + human knowledge."""
        self.game_id = game_id
        self.last_frame = initial_frame
        self.last_frame_hash = self._hash_frame(initial_frame)
        
        # Get learned priors from database
        learned_priors = self.db.get_goal_priors(game_id)
        
        print(f"\n  🎯 Initializing goal hypotheses for {game_id}")
        print(f"     Learned priors: {learned_priors}")
        
        # Create hypothesis for each goal type
        self.goal_hypotheses = []
        for goal_type in GoalType:
            prior = learned_priors.get(goal_type.value, 0.05)
            hypothesis = GoalHypothesis(
                goal_type=goal_type,
                prior=prior,
                posterior=prior,  # Initially, posterior = prior
                evidence=[]
            )
            self.goal_hypotheses.append(hypothesis)
        
        # Set initial goal to highest prior
        self._select_current_goal()
        print(f"     Initial goal: {self.current_goal.goal_type.value} (p={self.current_goal.posterior:.2f})")
    
    def _select_current_goal(self):
        """Select the goal with highest posterior probability."""
        self.current_goal = max(self.goal_hypotheses, key=lambda h: h.posterior)
    
    def _normalize_posteriors(self):
        """Normalize posteriors to sum to 1."""
        total = sum(h.posterior for h in self.goal_hypotheses)
        if total > 0:
            for h in self.goal_hypotheses:
                h.posterior /= total
    
    def update_beliefs(self, new_frame: np.ndarray, action_taken: int, level_changed: bool):
        """
        Bayesian update of goal beliefs after observing action result.
        
        P(goal | observation) ∝ P(observation | goal) * P(goal)
        """
        new_hash = self._hash_frame(new_frame)
        frame_changed = (new_hash != self.last_frame_hash)
        
        # Calculate observation likelihoods for each goal
        for hypothesis in self.goal_hypotheses:
            likelihood = self._compute_likelihood(
                hypothesis.goal_type,
                action_taken,
                frame_changed,
                level_changed,
                new_frame
            )
            
            # Bayesian update
            hypothesis.posterior = hypothesis.posterior * likelihood
        
        # Normalize
        self._normalize_posteriors()
        
        # Check for goal transitions
        old_goal = self.current_goal.goal_type if self.current_goal else None
        self._select_current_goal()
        new_goal = self.current_goal.goal_type
        
        if old_goal != new_goal:
            # Determine trigger type
            if level_changed:
                trigger = "level_win"
                emoji = "🎉"
            elif getattr(self, 'transformation_detected', False) and self.transformation_count == 1:
                trigger = "transformation"
                emoji = "🔮"
            elif frame_changed:
                trigger = "frame_change"
                emoji = "🔄"
            else:
                trigger = "inference"
                emoji = "🔄"
            
            print(f"     {emoji} Goal shift: {old_goal.value} → {new_goal.value} (p={self.current_goal.posterior:.2f}) [{trigger}]")
            self.goal_history.append(new_goal.value)
            
            # Record transition to database
            if old_goal:
                self.db.record_transition(self.game_id, old_goal.value, new_goal.value, trigger)
        
        # Update frame tracking
        self.last_frame = new_frame
        self.last_frame_hash = new_hash
        self.frame_history.append(new_hash)
    
    def _compute_frame_change_magnitude(self, old_frame: np.ndarray, new_frame: np.ndarray) -> float:
        """Compute how dramatically the frame changed (0.0 to 1.0)."""
        if old_frame is None or new_frame is None:
            return 0.0
        if old_frame.size == 0 or new_frame.size == 0:
            return 0.0
        if old_frame.shape != new_frame.shape:
            return 0.5  # Different shapes = significant but not full change
        
        total_pixels = old_frame.size
        if total_pixels == 0:
            return 0.0
        changed_pixels = np.sum(old_frame != new_frame)
        return changed_pixels / total_pixels
    
    def _detect_transformation(self, change_magnitude: float) -> bool:
        """
        Detect if a transformation event occurred.
        
        Transformation = dramatic visual change (not just movement).
        Movement typically changes <5% of pixels.
        Transformation changes 15-50% of pixels.
        100% change = probably frame reset/error, not real transformation.
        """
        return 0.15 < change_magnitude < 0.80
    
    def _compute_likelihood(self, goal_type: GoalType, action: int, 
                           frame_changed: bool, level_changed: bool,
                           frame: np.ndarray) -> float:
        """
        Compute P(observation | goal).
        
        How likely is this observation if we're pursuing this goal?
        
        KEY: Detect transformation events (dramatic frame changes) to
        trigger goal shifts from reach_action_point → transform_to_match → overlap_with_match
        """
        
        # Calculate change magnitude
        change_magnitude = self._compute_frame_change_magnitude(self.last_frame, frame)
        is_transformation = self._detect_transformation(change_magnitude)
        
        if is_transformation:
            self.transformation_count += 1
            print(f"     🔮 TRANSFORMATION #{self.transformation_count} DETECTED! ({change_magnitude*100:.1f}% pixels changed)")
            self.transformation_detected = True
        
        if level_changed:
            # Level completion strongly supports OVERLAP_WITH_MATCH / COMBINE
            if goal_type in [GoalType.COMBINE, GoalType.OVERLAP_WITH_MATCH]:
                return 10.0  # Very strong evidence
            else:
                return 0.3  # Counter-evidence
        
        if is_transformation:
            # Dramatic change = transformation event
            # Strong evidence for TRANSFORM goals, triggers shift to OVERLAP
            if goal_type in [GoalType.TRANSFORM, GoalType.TRANSFORM_TO_MATCH]:
                return 8.0  # We just transformed!
            elif goal_type == GoalType.REACH_ACTION_POINT:
                return 0.5  # We were reaching, now we should shift
            elif goal_type in [GoalType.OVERLAP_WITH_MATCH, GoalType.COMBINE]:
                return 3.0  # Post-transformation, combining becomes likely
            else:
                return 0.8
        
        if frame_changed:
            # Normal movement (small change)
            if goal_type == GoalType.REACH_ACTION_POINT:
                return 1.5  # Moving toward something
            elif goal_type in [GoalType.TRANSFORM, GoalType.TRANSFORM_TO_MATCH]:
                return 1.1  # Might be approaching transform point
            elif goal_type == GoalType.COLLECT_RESOURCE:
                return 1.4
            elif goal_type in [GoalType.OVERLAP_WITH_MATCH, GoalType.COMBINE]:
                # If we've already transformed, movement toward match is good
                if getattr(self, 'transformation_detected', False):
                    return 2.0  # Moving toward our match!
                return 1.2
            elif goal_type == GoalType.EXPLORE_REGION:
                return 1.2
            else:
                return 1.0
        else:
            # No change - slight evidence against active goals
            if goal_type in [GoalType.REACH_ACTION_POINT, GoalType.COLLECT_RESOURCE]:
                return 0.9  # Stuck, maybe wrong direction
            elif goal_type == GoalType.UNKNOWN:
                return 1.1  # Unknown goals explain everything
            else:
                return 0.95
    
    def choose_action(self, frame_data, available_actions) -> 'GameAction':
        """Select action based on current goal."""
        frame = np.array(frame_data.frame) if hasattr(frame_data, 'frame') else frame_data
        if len(frame.shape) == 3:
            frame = frame[0]
        
        current_level = getattr(frame_data, 'levels_completed', 0)
        
        # Initialize on first call
        if self.game_id is None:
            game_id = getattr(frame_data, 'game_id', 'unknown')
            self.initialize_goals(game_id, frame)
        
        # Check for level completion
        level_changed = current_level > self.levels_completed
        if level_changed:
            print(f"     ✨ Level {current_level} completed!")
            self.levels_completed = current_level
            self.db.record_goal_success(self.game_id, self.current_goal.goal_type.value)
        
        # Convert available actions
        if HAS_ARC:
            from arcengine import GameAction as GA
            if isinstance(available_actions[0], int):
                int_to_action = {a.value: a for a in GA}
                available_actions = [int_to_action.get(a, GA.ACTION1) for a in available_actions]
            valid_actions = [a for a in available_actions if a != GA.RESET]
        else:
            valid_actions = available_actions
        
        # Select action based on current goal
        action = self._select_action_for_goal(self.current_goal.goal_type, valid_actions, frame)
        
        self.actions_taken += 1
        
        # Update beliefs based on observation (will happen next call)
        # For now, store what we need
        self._pending_action = action.value if hasattr(action, 'value') else action
        
        return action
    
    def _select_action_for_goal(self, goal: GoalType, valid_actions, frame: np.ndarray):
        """Select action that advances the current goal."""
        
        if HAS_ARC:
            from arcengine import GameAction as GA
        else:
            GA = GameAction
        
        # Goal-specific action selection
        if goal == GoalType.REACH_ACTION_POINT:
            # Prioritize directional movement (actions 1-4 are usually directions)
            movement = [a for a in valid_actions if hasattr(a, 'value') and a.value in [1, 2, 3, 4]]
            if movement:
                # Could add frame analysis here to detect direction of plus sign
                return np.random.choice(movement)
        
        elif goal == GoalType.COLLECT_RESOURCE:
            # Similar to reach, but might prefer different directions based on yellow detection
            movement = [a for a in valid_actions if hasattr(a, 'value') and a.value in [1, 2, 3, 4]]
            if movement:
                return np.random.choice(movement)
        
        elif goal == GoalType.TRANSFORM:
            # We're on or near the transformation point, try action 5 or special actions
            special = [a for a in valid_actions if hasattr(a, 'value') and a.value >= 5]
            if special:
                return special[0]
            movement = [a for a in valid_actions if hasattr(a, 'value') and a.value in [1, 2, 3, 4]]
            if movement:
                return np.random.choice(movement)
        
        elif goal == GoalType.COMBINE:
            # Move toward target, directional movement
            movement = [a for a in valid_actions if hasattr(a, 'value') and a.value in [1, 2, 3, 4]]
            if movement:
                return np.random.choice(movement)
        
        # Fallback: random valid action
        return np.random.choice(valid_actions)
    
    def observe_result(self, new_frame_data):
        """Called after action to update beliefs."""
        frame = np.array(new_frame_data.frame) if hasattr(new_frame_data, 'frame') else new_frame_data
        if len(frame.shape) == 3:
            frame = frame[0]
        
        level = getattr(new_frame_data, 'levels_completed', self.levels_completed)
        level_changed = level > self.levels_completed
        
        self.update_beliefs(frame, self._pending_action, level_changed)
        
        if level_changed:
            self.levels_completed = level
    
    def finalize(self):
        """Record session learnings."""
        print(f"\n  📊 Goal Bayesian Agent Session Summary")
        print(f"     Game: {self.game_id}")
        print(f"     Actions: {self.actions_taken}")
        print(f"     Levels: {self.levels_completed}")
        print(f"     Goal history: {' → '.join(self.goal_history) if self.goal_history else 'none'}")
        print(f"     Final beliefs:")
        for h in sorted(self.goal_hypotheses, key=lambda x: -x.posterior)[:3]:
            print(f"       {h.goal_type.value}: {h.posterior:.3f}")


# ============================================================================
# RUN LOOP
# ============================================================================

def run_goal_agent(game_id: str, max_actions: int = 500):
    """Run the Goal Bayesian Agent on a game."""
    if not HAS_ARC:
        print("ARC-AGI not installed, cannot run game.")
        return
    
    arc = arc_agi.Arcade()
    env = arc.make(game_id)
    agent = GoalBayesianAgent()
    
    print(f"\n{'='*60}")
    print(f"🎯 GOAL BAYESIAN AGENT: {game_id}")
    print(f"{'='*60}")
    
    frame = env.reset()
    try:
        for i in range(max_actions):
            available = frame.available_actions or [1, 2, 3, 4]
            action = agent.choose_action(frame, available)
            new_frame = env.step(action)
            agent.observe_result(new_frame)
            frame = new_frame
            
            if str(frame.state) == 'WIN':
                print("  🎉 VICTORY!")
                break
                
            # Progress indicator every 100 actions
            if (i + 1) % 100 == 0:
                print(f"     ... {i+1} actions, {agent.levels_completed} levels")
                
    except KeyboardInterrupt:
        pass
    finally:
        agent.finalize()


if __name__ == "__main__":
    game = sys.argv[1] if len(sys.argv) > 1 else "ls20"
    actions = int(sys.argv[2]) if len(sys.argv) > 2 else 500
    run_goal_agent(game, actions)
