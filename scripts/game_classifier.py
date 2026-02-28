"""
Game Classifier for ARC-AGI-3

Uses LLM world knowledge to classify game type BEFORE playing.
Core insight: Humans recognize "this looks like a maze" and apply known strategies.

This is the "analogical reasoning" layer - transferring prior knowledge to new games.
"""

import os
import json
import base64
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict, List, Any
import numpy as np

# Try to import LLM clients
try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

try:
    import google.generativeai as genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False


class GameArchetype(Enum):
    """Known game archetypes with transferable strategies."""
    MAZE = "maze"
    SOKOBAN = "sokoban"
    PATTERN_MATCH = "pattern_match"
    ORCHESTRATION = "orchestration"
    LOGIC_PUZZLE = "logic_puzzle"
    UNKNOWN = "unknown"


@dataclass
class ArchetypePriors:
    """Prior knowledge for a game archetype."""
    archetype: GameArchetype
    goal_heuristic: str
    action_priority: List[str]
    failure_patterns: List[str]
    visual_cues: List[str]
    human_analogy: str


# Prior knowledge bank - what we know about each archetype
ARCHETYPE_PRIORS: Dict[GameArchetype, ArchetypePriors] = {
    GameArchetype.MAZE: ArchetypePriors(
        archetype=GameArchetype.MAZE,
        goal_heuristic="Minimize distance to distinctly colored target area",
        action_priority=["move_toward_target", "explore_unseen", "backtrack_from_dead_end"],
        failure_patterns=["hitting_walls_repeatedly", "circular_movement", "moving_away_from_goal"],
        visual_cues=["corridors", "walls", "single_player_element", "distinct_goal_area"],
        human_analogy="Like navigating a maze - find the path to the exit"
    ),
    GameArchetype.SOKOBAN: ArchetypePriors(
        archetype=GameArchetype.SOKOBAN,
        goal_heuristic="Push movable objects onto target positions",
        action_priority=["push_toward_target", "avoid_corner_trap", "plan_push_sequence"],
        failure_patterns=["pushing_into_corner", "blocking_own_path", "wrong_push_order"],
        visual_cues=["movable_blocks", "target_positions", "player_avatar", "walls"],
        human_analogy="Like Sokoban - push crates to marked spots without getting stuck"
    ),
    GameArchetype.PATTERN_MATCH: ArchetypePriors(
        archetype=GameArchetype.PATTERN_MATCH,
        goal_heuristic="Create symmetry, alignment, or color matching",
        action_priority=["align_colors", "complete_symmetry", "fill_gaps"],
        failure_patterns=["breaking_existing_pattern", "random_changes", "ignoring_template"],
        visual_cues=["partial_symmetry", "color_groups", "template_area", "completion_target"],
        human_analogy="Like a jigsaw puzzle - complete the pattern or match colors"
    ),
    GameArchetype.ORCHESTRATION: ArchetypePriors(
        archetype=GameArchetype.ORCHESTRATION,
        goal_heuristic="Coordinate multiple elements to achieve combined state",
        action_priority=["identify_controllable_elements", "test_interactions", "sequence_actions"],
        failure_patterns=["focusing_on_one_element", "missing_dependencies", "wrong_sequence"],
        visual_cues=["multiple_distinct_elements", "interaction_zones", "state_indicators"],
        human_analogy="Like conducting an orchestra - coordinate multiple parts"
    ),
    GameArchetype.LOGIC_PUZZLE: ArchetypePriors(
        archetype=GameArchetype.LOGIC_PUZZLE,
        goal_heuristic="Satisfy constraints through logical deduction",
        action_priority=["identify_constraints", "try_reversible_actions", "backtrack_on_violation"],
        failure_patterns=["random_guessing", "ignoring_constraints", "no_backtracking"],
        visual_cues=["constraint_indicators", "binary_states", "rule_patterns"],
        human_analogy="Like Sudoku - satisfy all constraints simultaneously"
    ),
    GameArchetype.UNKNOWN: ArchetypePriors(
        archetype=GameArchetype.UNKNOWN,
        goal_heuristic="Explore systematically to discover mechanics",
        action_priority=["test_all_actions", "observe_changes", "form_hypotheses"],
        failure_patterns=["random_actions", "no_observation", "no_learning"],
        visual_cues=[],
        human_analogy="Unknown game type - explore carefully to learn mechanics"
    ),
}


@dataclass
class ClassificationResult:
    """Result of game classification."""
    archetype: GameArchetype
    confidence: float
    reasoning: str
    priors: ArchetypePriors
    visual_features: Dict[str, Any]
    suggested_first_actions: List[str]


class GameClassifier:
    """
    Classifies ARC-AGI-3 games using LLM world knowledge.
    
    The key insight: Humans don't start from scratch. They see a game
    and think "this reminds me of..." and apply relevant strategies.
    """
    
    def __init__(self, use_gemini: bool = True, use_anthropic: bool = False):
        self.use_gemini = use_gemini and HAS_GEMINI
        self.use_anthropic = use_anthropic and HAS_ANTHROPIC
        
        if self.use_gemini:
            api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            if api_key:
                genai.configure(api_key=api_key)
                self.gemini_model = genai.GenerativeModel("gemini-2.0-flash")
            else:
                self.use_gemini = False
                
        if self.use_anthropic:
            self.anthropic_client = anthropic.Anthropic()
    
    def extract_visual_features(self, frame: np.ndarray) -> Dict[str, Any]:
        """Extract visual features from a game frame."""
        features = {}

        # Ensure frame is 2D
        if len(frame.shape) == 3:
            frame = frame[0]
        
        # Color distribution
        unique_colors = np.unique(frame)
        features["num_colors"] = len(unique_colors)
        features["colors"] = unique_colors.tolist()
        
        # Color frequencies
        color_counts = {}
        for color in unique_colors:
            count = np.sum(frame == color)
            color_counts[int(color)] = int(count)
        features["color_distribution"] = color_counts
        
        # Dominant color (background)
        features["dominant_color"] = int(max(color_counts, key=color_counts.get))
        
        # Frame dimensions
        features["height"] = frame.shape[0]
        features["width"] = frame.shape[1]
        
        # Edge detection (simple - count color transitions)
        h_edges = np.sum(frame[:, 1:] != frame[:, :-1])
        v_edges = np.sum(frame[1:, :] != frame[:-1, :])
        features["edge_count"] = int(h_edges + v_edges)
        features["edge_density"] = features["edge_count"] / (frame.shape[0] * frame.shape[1])
        
        # Symmetry check
        h_sym = np.sum(frame == np.fliplr(frame)) / frame.size
        v_sym = np.sum(frame == np.flipud(frame)) / frame.size
        features["horizontal_symmetry"] = float(h_sym)
        features["vertical_symmetry"] = float(v_sym)
        
        # Detect potential walls (connected regions of same color on edges)
        features["has_border"] = self._detect_border(frame)
        
        # Count distinct regions
        features["num_regions"] = self._count_regions(frame)
        
        return features
    
    def _detect_border(self, frame: np.ndarray) -> bool:
        """Check if frame has a border/wall pattern."""
        top = frame[0, :]
        bottom = frame[-1, :]
        left = frame[:, 0]
        right = frame[:, -1]
        
        # Check if edges are mostly same color
        edges = np.concatenate([top, bottom, left, right])
        most_common = np.bincount(edges.astype(int)).argmax()
        edge_uniformity = np.sum(edges == most_common) / len(edges)
        
        return edge_uniformity > 0.7
    
    def _count_regions(self, frame: np.ndarray) -> int:
        """Count distinct color regions (simplified)."""
        return len(np.unique(frame))
    
    def _frame_to_ascii(self, frame: np.ndarray, max_size: int = 40) -> str:
        """Convert frame to ASCII representation for LLM."""
        # Ensure frame is 2D
        if len(frame.shape) == 3:
            frame = frame[0]

        # Downsample if needed
        h, w = frame.shape
        if h > max_size or w > max_size:
            scale = max_size / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            # Simple downsampling
            indices_h = np.linspace(0, h-1, new_h).astype(int)
            indices_w = np.linspace(0, w-1, new_w).astype(int)
            frame = frame[indices_h][:, indices_w]
        
        # Convert to characters
        chars = " .:-=+*#%@"
        max_val = frame.max() if frame.max() > 0 else 1
        normalized = (frame / max_val * (len(chars) - 1)).astype(int)
        
        lines = []
        for row in normalized:
            line = "".join(chars[min(v, len(chars)-1)] for v in row)
            lines.append(line)
        
        return "\n".join(lines)
    
    def classify(self, frame: np.ndarray, game_id: str = "unknown") -> ClassificationResult:
        """
        Classify a game based on its first frame.
        
        Uses LLM world knowledge to recognize game archetypes.
        """
        # Extract visual features
        visual_features = self.extract_visual_features(frame)
        
        # Create ASCII representation
        ascii_frame = self._frame_to_ascii(frame)
        
        # Build classification prompt
        prompt = self._build_classification_prompt(ascii_frame, visual_features, game_id)
        
        # Get LLM classification
        if self.use_gemini:
            result = self._classify_with_gemini(prompt)
        elif self.use_anthropic:
            result = self._classify_with_anthropic(prompt)
        else:
            # Fallback to heuristic classification
            result = self._classify_heuristic(visual_features)
        
        # Parse result and build response
        archetype = self._parse_archetype(result.get("archetype", "unknown"))
        priors = ARCHETYPE_PRIORS[archetype]
        
        return ClassificationResult(
            archetype=archetype,
            confidence=result.get("confidence", 0.5),
            reasoning=result.get("reasoning", ""),
            priors=priors,
            visual_features=visual_features,
            suggested_first_actions=result.get("suggested_actions", priors.action_priority[:3])
        )
    
    def _build_classification_prompt(self, ascii_frame: str, features: Dict, game_id: str) -> str:
        """Build the classification prompt for the LLM."""
        return f"""You are analyzing a video game screenshot to classify what TYPE of game it is.

GAME ID: {game_id}

VISUAL REPRESENTATION (ASCII):
```
{ascii_frame}
```

EXTRACTED FEATURES:
- Colors: {features['num_colors']} distinct colors
- Color distribution: {features['color_distribution']}
- Dimensions: {features['height']}x{features['width']}
- Edge density: {features['edge_density']:.3f}
- Horizontal symmetry: {features['horizontal_symmetry']:.2f}
- Vertical symmetry: {features['vertical_symmetry']:.2f}
- Has border/walls: {features['has_border']}

KNOWN GAME ARCHETYPES:
1. MAZE - Navigate corridors to reach a goal (like Pac-Man paths)
2. SOKOBAN - Push objects onto target positions (like the classic Sokoban)
3. PATTERN_MATCH - Complete patterns, create symmetry, match colors (like Tetris/puzzle games)
4. ORCHESTRATION - Coordinate multiple elements simultaneously
5. LOGIC_PUZZLE - Satisfy constraints through deduction (like Sudoku)

Based on your knowledge of video games and puzzles, classify this game:

1. What game archetype does this MOST resemble? Pick from: MAZE, SOKOBAN, PATTERN_MATCH, ORCHESTRATION, LOGIC_PUZZLE, or UNKNOWN
2. Confidence (0.0 to 1.0)?
3. What visual cues led to this classification?
4. What is the likely GOAL of this game?
5. What actions should the player try FIRST?

Respond in JSON format:
{{
    "archetype": "ARCHETYPE_NAME",
    "confidence": 0.X,
    "reasoning": "explanation of visual cues and analogies",
    "likely_goal": "what the player probably needs to do",
    "suggested_actions": ["action1", "action2", "action3"]
}}"""
    
    def _classify_with_gemini(self, prompt: str) -> Dict:
        """Classify using Gemini."""
        try:
            response = self.gemini_model.generate_content(prompt)
            text = response.text
            
            # Extract JSON from response
            import re
            json_match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            return {"archetype": "unknown", "confidence": 0.3, "reasoning": text}
        except Exception as e:
            return {"archetype": "unknown", "confidence": 0.0, "reasoning": f"Error: {e}"}
    
    def _classify_with_anthropic(self, prompt: str) -> Dict:
        """Classify using Claude."""
        try:
            response = self.anthropic_client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}]
            )
            text = response.content[0].text
            
            import re
            json_match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            return {"archetype": "unknown", "confidence": 0.3, "reasoning": text}
        except Exception as e:
            return {"archetype": "unknown", "confidence": 0.0, "reasoning": f"Error: {e}"}
    
    def _classify_heuristic(self, features: Dict) -> Dict:
        """Fallback heuristic classification when no LLM available."""
        # Simple heuristics based on visual features
        if features["has_border"] and features["edge_density"] > 0.1:
            return {
                "archetype": "maze",
                "confidence": 0.4,
                "reasoning": "Has borders and high edge density - likely maze or navigation"
            }
        elif features["horizontal_symmetry"] > 0.8 or features["vertical_symmetry"] > 0.8:
            return {
                "archetype": "pattern_match",
                "confidence": 0.4,
                "reasoning": "High symmetry detected - likely pattern matching"
            }
        elif features["num_colors"] > 5:
            return {
                "archetype": "orchestration",
                "confidence": 0.3,
                "reasoning": "Many colors suggest multiple elements to coordinate"
            }
        else:
            return {
                "archetype": "unknown",
                "confidence": 0.2,
                "reasoning": "No clear pattern detected"
            }
    
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
    
    def get_priors(self, archetype: GameArchetype) -> ArchetypePriors:
        """Get prior knowledge for an archetype."""
        return ARCHETYPE_PRIORS.get(archetype, ARCHETYPE_PRIORS[GameArchetype.UNKNOWN])


# Convenience function
def classify_game(frame: np.ndarray, game_id: str = "unknown") -> ClassificationResult:
    """Classify a game from its first frame."""
    classifier = GameClassifier()
    return classifier.classify(frame, game_id)


if __name__ == "__main__":
    # Test with a simple pattern
    test_frame = np.array([
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 2, 0, 0, 3, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 4, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1],
    ])
    
    classifier = GameClassifier()
    features = classifier.extract_visual_features(test_frame)
    print("Visual Features:")
    for k, v in features.items():
        print(f"  {k}: {v}")
    
    print("\nASCII representation:")
    print(classifier._frame_to_ascii(test_frame))
    
    # If LLM available, do full classification
    result = classifier.classify(test_frame, "test_game")
    print(f"\nClassification: {result.archetype.value}")
    print(f"Confidence: {result.confidence}")
    print(f"Reasoning: {result.reasoning}")
    print(f"Suggested actions: {result.suggested_first_actions}")
