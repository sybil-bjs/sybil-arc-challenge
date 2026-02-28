"""
Adaptive Game Classifier for ARC-AGI-3

DYNAMIC LEARNING SYSTEM - improves as we play more games.

Key principle: The 5 starting archetypes are just initial priors.
As we play, we:
1. DISCOVER new archetypes that don't fit existing categories
2. REFINE existing archetypes based on what works/fails
3. RECORD game-specific learnings to the knowledge base
4. PROMOTE validated insights to improve future classification

Integrates with:
- pattern_db.py (SQLite pattern storage)
- failure_memory.py (what went wrong)
- knowledge/corrections/ (documented mistakes)
- knowledge/insights/ (validated learnings)
"""

import os
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import numpy as np

from game_classifier import (
    GameArchetype, ArchetypePriors, ClassificationResult,
    GameClassifier, ARCHETYPE_PRIORS
)


# Path to knowledge base
KNOWLEDGE_DIR = Path(os.path.expanduser("~/sybil-arc-challenge/knowledge"))
LEARNED_ARCHETYPES_FILE = KNOWLEDGE_DIR / "learned_archetypes.json"
GAME_LEARNINGS_DB = KNOWLEDGE_DIR / "game_learnings.db"


@dataclass
class GameLearning:
    """What we learned from playing a specific game."""
    game_id: str
    initial_archetype: str
    final_archetype: str  # May differ if we discovered something new
    confidence_change: float  # Did classification confidence improve?
    successful_strategies: List[str]
    failed_strategies: List[str]
    discovered_mechanics: List[str]
    levels_completed: int
    total_actions: int
    efficiency_vs_human: float  # Our actions / human baseline
    notes: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass 
class LearnedArchetype:
    """A dynamically discovered archetype."""
    name: str
    description: str
    discovered_from_games: List[str]
    goal_heuristic: str
    action_priority: List[str]
    failure_patterns: List[str]
    visual_cues: List[str]
    human_analogy: str
    confidence: float  # How confident are we in this archetype?
    times_validated: int  # How many times has this worked?
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())


class LearningDatabase:
    """
    Persistent storage for game learnings and discovered archetypes.
    
    This is the MEMORY of our system - it accumulates knowledge over time.
    """
    
    def __init__(self, db_path: Path = GAME_LEARNINGS_DB):
        self.db_path = db_path
        self._ensure_tables()
    
    def _ensure_tables(self):
        """Create tables if they don't exist."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS game_learnings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    game_id TEXT NOT NULL,
                    initial_archetype TEXT,
                    final_archetype TEXT,
                    confidence_change REAL,
                    successful_strategies TEXT,
                    failed_strategies TEXT,
                    discovered_mechanics TEXT,
                    levels_completed INTEGER,
                    total_actions INTEGER,
                    efficiency_vs_human REAL,
                    notes TEXT,
                    timestamp TEXT,
                    UNIQUE(game_id, timestamp)
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS archetype_outcomes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    game_id TEXT NOT NULL,
                    archetype TEXT NOT NULL,
                    success INTEGER,
                    actions_taken INTEGER,
                    notes TEXT,
                    timestamp TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS strategy_effectiveness (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    archetype TEXT NOT NULL,
                    strategy TEXT NOT NULL,
                    times_tried INTEGER DEFAULT 0,
                    times_succeeded INTEGER DEFAULT 0,
                    avg_efficiency REAL,
                    notes TEXT
                )
            """)
    
    def record_game_learning(self, learning: GameLearning):
        """Record what we learned from a game session."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO game_learnings 
                (game_id, initial_archetype, final_archetype, confidence_change,
                 successful_strategies, failed_strategies, discovered_mechanics,
                 levels_completed, total_actions, efficiency_vs_human, notes, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                learning.game_id,
                learning.initial_archetype,
                learning.final_archetype,
                learning.confidence_change,
                json.dumps(learning.successful_strategies),
                json.dumps(learning.failed_strategies),
                json.dumps(learning.discovered_mechanics),
                learning.levels_completed,
                learning.total_actions,
                learning.efficiency_vs_human,
                learning.notes,
                learning.timestamp
            ))
    
    def record_archetype_outcome(self, game_id: str, archetype: str, 
                                  success: bool, actions: int, notes: str = ""):
        """Record whether an archetype classification led to success."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO archetype_outcomes 
                (game_id, archetype, success, actions_taken, notes, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (game_id, archetype, int(success), actions, notes, 
                  datetime.now().isoformat()))
    
    def get_archetype_success_rate(self, archetype: str) -> Tuple[int, int, float]:
        """Get success rate for an archetype: (successes, attempts, rate)."""
        with sqlite3.connect(self.db_path) as conn:
            result = conn.execute("""
                SELECT SUM(success), COUNT(*) FROM archetype_outcomes
                WHERE archetype = ?
            """, (archetype,)).fetchone()
            
            successes = result[0] or 0
            attempts = result[1] or 0
            rate = successes / attempts if attempts > 0 else 0.5
            return successes, attempts, rate
    
    def get_game_history(self, game_id: str) -> List[Dict]:
        """Get all learnings for a specific game."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT * FROM game_learnings WHERE game_id = ?
                ORDER BY timestamp DESC
            """, (game_id,)).fetchall()
            return [dict(row) for row in rows]
    
    def get_successful_strategies(self, archetype: str) -> List[str]:
        """Get strategies that have worked for an archetype."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("""
                SELECT successful_strategies FROM game_learnings
                WHERE final_archetype = ? AND levels_completed > 0
            """, (archetype,)).fetchall()
            
            strategies = []
            for row in rows:
                if row[0]:
                    strategies.extend(json.loads(row[0]))
            return list(set(strategies))
    
    def get_failed_strategies(self, archetype: str) -> List[str]:
        """Get strategies that have failed for an archetype."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("""
                SELECT failed_strategies FROM game_learnings
                WHERE final_archetype = ?
            """, (archetype,)).fetchall()
            
            strategies = []
            for row in rows:
                if row[0]:
                    strategies.extend(json.loads(row[0]))
            return list(set(strategies))


class LearnedArchetypeStore:
    """
    Manages dynamically discovered archetypes.
    
    When we encounter a game that doesn't fit existing archetypes,
    we can create a new one and refine it over time.
    """
    
    def __init__(self, file_path: Path = LEARNED_ARCHETYPES_FILE):
        self.file_path = file_path
        self.archetypes: Dict[str, LearnedArchetype] = {}
        self._load()
    
    def _load(self):
        """Load learned archetypes from file."""
        if self.file_path.exists():
            with open(self.file_path) as f:
                data = json.load(f)
                for name, arch_data in data.items():
                    self.archetypes[name] = LearnedArchetype(**arch_data)
    
    def _save(self):
        """Save learned archetypes to file."""
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.file_path, 'w') as f:
            data = {name: asdict(arch) for name, arch in self.archetypes.items()}
            json.dump(data, f, indent=2)
    
    def add_archetype(self, archetype: LearnedArchetype):
        """Add or update a learned archetype."""
        archetype.updated_at = datetime.now().isoformat()
        self.archetypes[archetype.name] = archetype
        self._save()
    
    def get_archetype(self, name: str) -> Optional[LearnedArchetype]:
        """Get a learned archetype by name."""
        return self.archetypes.get(name)
    
    def validate_archetype(self, name: str, success: bool):
        """Record a validation (success/failure) for an archetype."""
        if name in self.archetypes:
            arch = self.archetypes[name]
            if success:
                arch.times_validated += 1
                arch.confidence = min(1.0, arch.confidence + 0.05)
            else:
                arch.confidence = max(0.1, arch.confidence - 0.1)
            arch.updated_at = datetime.now().isoformat()
            self._save()
    
    def get_all(self) -> List[LearnedArchetype]:
        """Get all learned archetypes."""
        return list(self.archetypes.values())
    
    def suggest_new_archetype(self, game_id: str, mechanics: List[str], 
                               goal: str) -> LearnedArchetype:
        """Create a suggested new archetype from game observations."""
        name = f"DISCOVERED_{game_id.upper()}"
        return LearnedArchetype(
            name=name,
            description=f"Discovered from game {game_id}",
            discovered_from_games=[game_id],
            goal_heuristic=goal,
            action_priority=[],
            failure_patterns=[],
            visual_cues=[],
            human_analogy="Unknown - needs more observations",
            confidence=0.3,
            times_validated=0
        )


class AdaptiveClassifier:
    """
    Classifier that LEARNS and IMPROVES over time.
    
    Combines:
    1. Static priors (5 starting archetypes)
    2. Learned archetypes (discovered through play)
    3. Historical outcomes (what worked before)
    4. Game-specific learnings (patterns, mechanics)
    """
    
    def __init__(self):
        self.base_classifier = GameClassifier()
        self.learning_db = LearningDatabase()
        self.archetype_store = LearnedArchetypeStore()
    
    def classify(self, frame: np.ndarray, game_id: str = "unknown") -> ClassificationResult:
        """
        Classify with learning integration.
        
        Steps:
        1. Check if we've played this game before
        2. Get base classification from LLM
        3. Adjust based on historical outcomes
        4. Check for matching learned archetypes
        5. Return enriched classification
        """
        # Check game history
        history = self.learning_db.get_game_history(game_id)
        
        if history:
            # We've played this before! Use what we learned.
            last_session = history[0]
            known_archetype = last_session.get('final_archetype', 'unknown')
            
            # Get what worked and what didn't
            successful = self.learning_db.get_successful_strategies(known_archetype)
            failed = self.learning_db.get_failed_strategies(known_archetype)
            
            # Build enriched result
            base_result = self.base_classifier.classify(frame, game_id)
            
            # Override with learned knowledge
            if known_archetype != 'unknown':
                priors = self._get_enriched_priors(known_archetype, successful, failed)
                return ClassificationResult(
                    archetype=self._parse_archetype(known_archetype),
                    confidence=min(0.95, base_result.confidence + 0.2),  # Higher confidence
                    reasoning=f"Previously played. Known archetype: {known_archetype}. " + 
                              f"Successful strategies: {successful[:3]}",
                    priors=priors,
                    visual_features=base_result.visual_features,
                    suggested_first_actions=successful[:5] if successful else priors.action_priority
                )
        
        # First time seeing this game
        base_result = self.base_classifier.classify(frame, game_id)
        
        # Check if any learned archetype matches better
        learned_match = self._check_learned_archetypes(base_result.visual_features)
        if learned_match and learned_match.confidence > base_result.confidence:
            return ClassificationResult(
                archetype=GameArchetype.UNKNOWN,  # Custom archetype
                confidence=learned_match.confidence,
                reasoning=f"Matches learned archetype: {learned_match.name}. " +
                         f"Human analogy: {learned_match.human_analogy}",
                priors=self._learned_to_priors(learned_match),
                visual_features=base_result.visual_features,
                suggested_first_actions=learned_match.action_priority[:5]
            )
        
        # Adjust confidence based on archetype success rate
        success_rate = self.learning_db.get_archetype_success_rate(
            base_result.archetype.value
        )[2]
        adjusted_confidence = base_result.confidence * (0.5 + 0.5 * success_rate)
        
        return ClassificationResult(
            archetype=base_result.archetype,
            confidence=adjusted_confidence,
            reasoning=base_result.reasoning + 
                      f" [Archetype success rate: {success_rate:.0%}]",
            priors=base_result.priors,
            visual_features=base_result.visual_features,
            suggested_first_actions=base_result.suggested_first_actions
        )
    
    def _get_enriched_priors(self, archetype: str, 
                              successful: List[str], 
                              failed: List[str]) -> ArchetypePriors:
        """Get priors enriched with learned strategies."""
        base_archetype = self._parse_archetype(archetype)
        base_priors = ARCHETYPE_PRIORS.get(base_archetype, 
                                           ARCHETYPE_PRIORS[GameArchetype.UNKNOWN])
        
        # Combine base action_priority with successful strategies
        combined_actions = successful[:3] + [
            a for a in base_priors.action_priority if a not in successful
        ]
        
        # Add failed strategies to failure_patterns
        combined_failures = list(set(base_priors.failure_patterns + failed[:3]))
        
        return ArchetypePriors(
            archetype=base_archetype,
            goal_heuristic=base_priors.goal_heuristic,
            action_priority=combined_actions,
            failure_patterns=combined_failures,
            visual_cues=base_priors.visual_cues,
            human_analogy=base_priors.human_analogy
        )
    
    def _check_learned_archetypes(self, features: Dict) -> Optional[LearnedArchetype]:
        """Check if visual features match any learned archetype."""
        best_match = None
        best_score = 0.0
        
        for arch in self.archetype_store.get_all():
            # Simple matching based on visual cues
            score = 0.0
            for cue in arch.visual_cues:
                if cue in str(features):
                    score += 0.2
            
            # Weight by confidence and validations
            score *= arch.confidence * (1 + 0.1 * arch.times_validated)
            
            if score > best_score:
                best_score = score
                best_match = arch
        
        return best_match if best_score > 0.3 else None
    
    def _learned_to_priors(self, learned: LearnedArchetype) -> ArchetypePriors:
        """Convert learned archetype to priors format."""
        return ArchetypePriors(
            archetype=GameArchetype.UNKNOWN,
            goal_heuristic=learned.goal_heuristic,
            action_priority=learned.action_priority,
            failure_patterns=learned.failure_patterns,
            visual_cues=learned.visual_cues,
            human_analogy=learned.human_analogy
        )
    
    def _parse_archetype(self, archetype_str: str) -> GameArchetype:
        """Parse archetype string to enum."""
        mapping = {
            "maze": GameArchetype.MAZE,
            "sokoban": GameArchetype.SOKOBAN,
            "pattern_match": GameArchetype.PATTERN_MATCH,
            "orchestration": GameArchetype.ORCHESTRATION,
            "logic_puzzle": GameArchetype.LOGIC_PUZZLE,
        }
        return mapping.get(archetype_str.lower(), GameArchetype.UNKNOWN)
    
    # =========== LEARNING METHODS ===========
    
    def record_session(self, learning: GameLearning):
        """Record learnings from a game session."""
        self.learning_db.record_game_learning(learning)
        
        # Record archetype outcome
        success = learning.levels_completed > 0
        self.learning_db.record_archetype_outcome(
            learning.game_id,
            learning.final_archetype,
            success,
            learning.total_actions,
            learning.notes
        )
        
        # Validate any learned archetypes used
        if learning.final_archetype.startswith("DISCOVERED_"):
            self.archetype_store.validate_archetype(
                learning.final_archetype, success
            )
    
    def discover_archetype(self, game_id: str, mechanics: List[str],
                           goal: str, strategies: List[str]) -> LearnedArchetype:
        """
        Create a new archetype from game observations.
        
        Call this when a game doesn't fit existing archetypes.
        """
        new_arch = LearnedArchetype(
            name=f"DISCOVERED_{game_id.upper()}",
            description=f"Discovered from game {game_id}",
            discovered_from_games=[game_id],
            goal_heuristic=goal,
            action_priority=strategies,
            failure_patterns=[],
            visual_cues=[],
            human_analogy=f"Similar to {game_id}",
            confidence=0.3,
            times_validated=0
        )
        self.archetype_store.add_archetype(new_arch)
        return new_arch
    
    def merge_similar_archetypes(self, name1: str, name2: str, 
                                  merged_name: str) -> Optional[LearnedArchetype]:
        """
        Merge two similar learned archetypes into one.
        
        Use when we realize two discovered archetypes are actually the same.
        """
        arch1 = self.archetype_store.get_archetype(name1)
        arch2 = self.archetype_store.get_archetype(name2)
        
        if not arch1 or not arch2:
            return None
        
        merged = LearnedArchetype(
            name=merged_name,
            description=f"Merged from {name1} and {name2}",
            discovered_from_games=arch1.discovered_from_games + arch2.discovered_from_games,
            goal_heuristic=arch1.goal_heuristic,  # Use higher-confidence one
            action_priority=list(set(arch1.action_priority + arch2.action_priority)),
            failure_patterns=list(set(arch1.failure_patterns + arch2.failure_patterns)),
            visual_cues=list(set(arch1.visual_cues + arch2.visual_cues)),
            human_analogy=arch1.human_analogy if arch1.confidence > arch2.confidence else arch2.human_analogy,
            confidence=max(arch1.confidence, arch2.confidence),
            times_validated=arch1.times_validated + arch2.times_validated
        )
        
        self.archetype_store.add_archetype(merged)
        return merged
    
    def get_learning_summary(self) -> Dict:
        """Get a summary of what we've learned."""
        learned_archetypes = self.archetype_store.get_all()
        
        summary = {
            "base_archetypes": [a.value for a in GameArchetype if a != GameArchetype.UNKNOWN],
            "learned_archetypes": [
                {
                    "name": a.name,
                    "confidence": a.confidence,
                    "times_validated": a.times_validated,
                    "games": a.discovered_from_games
                }
                for a in learned_archetypes
            ],
            "archetype_success_rates": {},
            "total_games_played": 0,
            "total_learnings": 0
        }
        
        # Get success rates for base archetypes
        for arch in GameArchetype:
            if arch != GameArchetype.UNKNOWN:
                successes, attempts, rate = self.learning_db.get_archetype_success_rate(arch.value)
                summary["archetype_success_rates"][arch.value] = {
                    "successes": successes,
                    "attempts": attempts,
                    "rate": rate
                }
                summary["total_games_played"] += attempts
        
        return summary


# Convenience functions
def adaptive_classify(frame: np.ndarray, game_id: str = "unknown") -> ClassificationResult:
    """Classify with learning integration."""
    classifier = AdaptiveClassifier()
    return classifier.classify(frame, game_id)


def record_learning(learning: GameLearning):
    """Record learnings from a game session."""
    classifier = AdaptiveClassifier()
    classifier.record_session(learning)


if __name__ == "__main__":
    # Demo the adaptive classifier
    classifier = AdaptiveClassifier()
    
    print("=== Adaptive Classifier Demo ===\n")
    
    # Show learning summary
    summary = classifier.get_learning_summary()
    print("Learning Summary:")
    print(f"  Base archetypes: {summary['base_archetypes']}")
    print(f"  Learned archetypes: {len(summary['learned_archetypes'])}")
    print(f"  Total games played: {summary['total_games_played']}")
    
    print("\n  Archetype success rates:")
    for arch, data in summary['archetype_success_rates'].items():
        print(f"    {arch}: {data['successes']}/{data['attempts']} = {data['rate']:.0%}")
    
    # Test classification with a sample frame
    test_frame = np.array([
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 2, 0, 0, 3, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 4, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1],
    ])
    
    print("\n=== Test Classification ===")
    result = classifier.classify(test_frame, "test_game")
    print(f"Archetype: {result.archetype.value}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Reasoning: {result.reasoning}")
    print(f"Suggested actions: {result.suggested_first_actions}")
