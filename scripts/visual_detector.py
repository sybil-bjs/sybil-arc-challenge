"""
Visual Element Detector for ARC-AGI-3

Detects game elements based on human-provided visual descriptions.
Works with GoalBayesianAgent to enable directed navigation.

Grounded knowledge from Bridget + Sybil:
- Plus sign: WHITE CROSS shape
- Player: Orange top + blue bottom composite
- Goal: Pattern in dark square
- State indicator: Bottom-left HUD
- Power-ups: Yellow squares with dark centers

Lead: Saber ⚔️ | Grounding: Bridget 🎮 | ML: Sybil 🧠
"""

import numpy as np
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass
from enum import Enum


class ElementType(Enum):
    PLUS_SIGN = "plus_sign"
    PLAYER = "player"
    GOAL = "goal"
    STATE_INDICATOR = "state_indicator"
    POWERUP = "powerup"
    UNKNOWN = "unknown"


@dataclass
class DetectedElement:
    """A detected visual element."""
    element_type: ElementType
    position: Tuple[int, int]  # (x, y) center
    bounds: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float
    pattern: Optional[np.ndarray] = None  # Extracted pattern if applicable


class VisualDetector:
    """
    Detects game elements using human-grounded visual knowledge.
    
    This is NOT a learned CNN — it's pattern matching based on
    Bridget's descriptions of what elements look like.
    """
    
    # Color definitions (approximate, may need tuning per game)
    # ARC-AGI uses a 16-color palette typically
    WHITE = 255
    BLACK = 0
    
    def __init__(self):
        self.frame_shape = None
        self.last_detections: Dict[ElementType, DetectedElement] = {}
    
    def detect_all(self, frame: np.ndarray) -> Dict[ElementType, DetectedElement]:
        """Detect all known elements in frame."""
        # Ensure 2D
        if len(frame.shape) == 3:
            frame = frame[0]
        if len(frame.shape) == 1:
            # 1D array, can't detect anything
            return {}
        if len(frame.shape) != 2:
            return {}
        
        self.frame_shape = frame.shape
        detections = {}
        
        # Detect each element type
        plus = self.detect_plus_sign(frame)
        if plus:
            detections[ElementType.PLUS_SIGN] = plus
        
        goal = self.detect_goal_pattern(frame)
        if goal:
            detections[ElementType.GOAL] = goal
        
        state = self.detect_state_indicator(frame)
        if state:
            detections[ElementType.STATE_INDICATOR] = state
        
        player = self.detect_player(frame)
        if player:
            detections[ElementType.PLAYER] = player
        
        self.last_detections = detections
        return detections
    
    def detect_plus_sign(self, frame: np.ndarray) -> Optional[DetectedElement]:
        """
        Detect white cross (plus sign) in frame.
        
        Plus signs are transformation triggers.
        Visual: White pixels in a + shape.
        """
        h, w = frame.shape
        
        # Find white/bright pixels (high value in grayscale)
        # Threshold for "white" - adjust based on color palette
        white_threshold = np.max(frame) * 0.9  # Top 10% brightness
        white_mask = frame >= white_threshold
        
        if not np.any(white_mask):
            return None
        
        # Look for + shaped clusters
        # A plus sign has a center pixel with pixels above, below, left, right
        best_score = 0
        best_pos = None
        
        # Scan for plus pattern
        for y in range(2, h - 2):
            for x in range(2, w - 2):
                if white_mask[y, x]:
                    # Check for + shape (center + 4 directions)
                    score = 0
                    # Center
                    score += 1 if white_mask[y, x] else 0
                    # Up
                    score += 1 if white_mask[y-1, x] else 0
                    score += 0.5 if y > 1 and white_mask[y-2, x] else 0
                    # Down
                    score += 1 if white_mask[y+1, x] else 0
                    score += 0.5 if y < h-2 and white_mask[y+2, x] else 0
                    # Left
                    score += 1 if white_mask[y, x-1] else 0
                    score += 0.5 if x > 1 and white_mask[y, x-2] else 0
                    # Right
                    score += 1 if white_mask[y, x+1] else 0
                    score += 0.5 if x < w-2 and white_mask[y, x+2] else 0
                    
                    # Penalize if corners are filled (that's a square, not a plus)
                    if white_mask[y-1, x-1]: score -= 0.5
                    if white_mask[y-1, x+1]: score -= 0.5
                    if white_mask[y+1, x-1]: score -= 0.5
                    if white_mask[y+1, x+1]: score -= 0.5
                    
                    if score > best_score:
                        best_score = score
                        best_pos = (x, y)
        
        if best_pos and best_score >= 4:  # At least center + 4 arms
            x, y = best_pos
            return DetectedElement(
                element_type=ElementType.PLUS_SIGN,
                position=best_pos,
                bounds=(x-2, y-2, x+2, y+2),
                confidence=min(1.0, best_score / 7),
            )
        
        return None
    
    def detect_goal_pattern(self, frame: np.ndarray) -> Optional[DetectedElement]:
        """
        Detect goal pattern (pattern in dark square).
        
        The goal is typically a static pattern that the player
        needs to match and then overlap with.
        """
        h, w = frame.shape
        
        # Goal is often in a distinct region (dark background)
        # Look for dark rectangular regions with patterns inside
        
        # Strategy: Find dark regions, then check for patterns
        dark_threshold = np.max(frame) * 0.2  # Bottom 20% brightness
        dark_mask = frame <= dark_threshold
        
        # Find contiguous dark regions
        # Simple approach: look for rectangular dark areas
        best_region = None
        best_size = 0
        
        # Scan for dark rectangular regions (at least 5x5)
        for y in range(0, h - 5):
            for x in range(0, w - 5):
                # Check if this could be top-left of a dark region
                if dark_mask[y, x]:
                    # Expand to find region bounds
                    x2, y2 = x, y
                    while x2 < w - 1 and dark_mask[y, x2 + 1]:
                        x2 += 1
                    while y2 < h - 1 and dark_mask[y2 + 1, x]:
                        y2 += 1
                    
                    size = (x2 - x) * (y2 - y)
                    if size > best_size and size >= 25:  # At least 5x5
                        # Check it's actually mostly dark
                        region = frame[y:y2+1, x:x2+1]
                        if np.mean(region <= dark_threshold) > 0.6:
                            best_size = size
                            best_region = (x, y, x2, y2)
        
        if best_region:
            x1, y1, x2, y2 = best_region
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            pattern = frame[y1:y2+1, x1:x2+1].copy()
            
            return DetectedElement(
                element_type=ElementType.GOAL,
                position=(cx, cy),
                bounds=best_region,
                confidence=0.7,
                pattern=pattern,
            )
        
        return None
    
    def detect_state_indicator(self, frame: np.ndarray) -> Optional[DetectedElement]:
        """
        Detect state indicator (bottom-left HUD).
        
        Shows current player form/state.
        """
        h, w = frame.shape
        
        # State indicator is in bottom-left quadrant
        region_h = h // 4
        region_w = w // 4
        
        # Extract bottom-left region
        bl_region = frame[h - region_h:, :region_w]
        
        # Look for a distinct pattern (not all same color)
        unique_colors = len(np.unique(bl_region))
        
        if unique_colors >= 2:  # Has some pattern
            return DetectedElement(
                element_type=ElementType.STATE_INDICATOR,
                position=(region_w // 2, h - region_h // 2),
                bounds=(0, h - region_h, region_w, h),
                confidence=0.6,
                pattern=bl_region.copy(),
            )
        
        return None
    
    def detect_player(self, frame: np.ndarray) -> Optional[DetectedElement]:
        """
        Detect player token (orange top + blue bottom composite).
        
        The player is the controllable element.
        """
        h, w = frame.shape
        
        # Player has distinctive two-color vertical pattern
        # This is game-specific; for now use a heuristic
        
        # Look for vertically stacked different colors
        best_pos = None
        best_score = 0
        
        for y in range(1, h - 1):
            for x in range(1, w - 1):
                # Check for vertical color transition
                top = frame[y-1, x]
                mid = frame[y, x]
                bot = frame[y+1, x]
                
                # Different colors stacked vertically
                if top != mid and mid == bot:
                    # Might be player (top different from bottom)
                    score = 1
                    # Check horizontal consistency
                    if x > 0 and frame[y-1, x-1] == top and frame[y, x-1] == mid:
                        score += 0.5
                    if x < w-1 and frame[y-1, x+1] == top and frame[y, x+1] == mid:
                        score += 0.5
                    
                    if score > best_score:
                        best_score = score
                        best_pos = (x, y)
        
        if best_pos and best_score >= 1.5:
            x, y = best_pos
            return DetectedElement(
                element_type=ElementType.PLAYER,
                position=best_pos,
                bounds=(x-1, y-1, x+1, y+1),
                confidence=min(1.0, best_score / 2),
            )
        
        return None
    
    def calculate_direction_to(self, from_pos: Tuple[int, int], 
                                to_pos: Tuple[int, int]) -> int:
        """
        Calculate which action (1-4) moves from_pos toward to_pos.
        
        Assumes: 1=up, 2=down, 3=left, 4=right (common mapping)
        """
        dx = to_pos[0] - from_pos[0]
        dy = to_pos[1] - from_pos[1]
        
        # Move in the direction of largest difference
        if abs(dx) > abs(dy):
            return 4 if dx > 0 else 3  # right or left
        else:
            return 2 if dy > 0 else 1  # down or up
    
    def patterns_match(self, pattern1: np.ndarray, pattern2: np.ndarray, 
                       threshold: float = 0.9) -> bool:
        """Check if two patterns match (allowing for some noise)."""
        if pattern1 is None or pattern2 is None:
            return False
        if pattern1.shape != pattern2.shape:
            # Resize smaller to larger
            return False  # For now, require same shape
        
        match_ratio = np.mean(pattern1 == pattern2)
        return match_ratio >= threshold
    
    def count_transformations_needed(self, current_state: np.ndarray,
                                     goal_pattern: np.ndarray) -> int:
        """
        Estimate how many transformations needed to match goal.
        
        This is a heuristic — actual count depends on game mechanics.
        """
        if current_state is None or goal_pattern is None:
            return 0
        
        # Count pixel differences as proxy for transformations
        # This is simplified — real logic depends on transformation rules
        if current_state.shape != goal_pattern.shape:
            return 1  # At least one transformation needed
        
        diff_ratio = np.mean(current_state != goal_pattern)
        
        # Heuristic: more differences = more transformations
        # Assume each transformation changes ~25% of pattern
        estimated = int(np.ceil(diff_ratio / 0.25))
        return max(1, min(estimated, 10))  # Cap at reasonable range


class DirectedNavigator:
    """
    Uses VisualDetector to navigate toward goals.
    
    Integrates with GoalBayesianAgent to provide directed actions
    instead of random exploration.
    """
    
    def __init__(self):
        self.detector = VisualDetector()
        self.current_target: Optional[ElementType] = None
        self.transformations_done: int = 0
        self.transformations_needed: int = 0
    
    def analyze_frame(self, frame: np.ndarray) -> Dict:
        """Analyze frame and return navigation guidance."""
        detections = self.detector.detect_all(frame)
        
        analysis = {
            "detections": detections,
            "plus_sign_found": ElementType.PLUS_SIGN in detections,
            "goal_found": ElementType.GOAL in detections,
            "player_found": ElementType.PLAYER in detections,
            "state_found": ElementType.STATE_INDICATOR in detections,
            "recommended_action": None,
            "current_phase": "explore",
        }
        
        # Determine current phase and recommended action
        player = detections.get(ElementType.PLAYER)
        plus_sign = detections.get(ElementType.PLUS_SIGN)
        goal = detections.get(ElementType.GOAL)
        state = detections.get(ElementType.STATE_INDICATOR)
        
        if player and plus_sign:
            # Check if we're on the plus sign
            px, py = player.position
            plus_x, plus_y = plus_sign.position
            distance = abs(px - plus_x) + abs(py - plus_y)
            
            if distance <= 2:
                # On or near plus sign
                if state and goal:
                    if self.detector.patterns_match(state.pattern, goal.pattern):
                        # Transformed enough, go to goal
                        analysis["current_phase"] = "navigate_to_goal"
                        if goal:
                            analysis["recommended_action"] = self.detector.calculate_direction_to(
                                player.position, goal.position
                            )
                    else:
                        # Need more transformations, stay on plus
                        analysis["current_phase"] = "transforming"
                        analysis["recommended_action"] = 0  # Stay/hover
                else:
                    analysis["current_phase"] = "transforming"
            else:
                # Not at plus sign, navigate to it
                analysis["current_phase"] = "navigate_to_plus"
                analysis["recommended_action"] = self.detector.calculate_direction_to(
                    player.position, plus_sign.position
                )
        
        elif player and goal:
            # No plus sign visible, maybe already transformed
            # Try to go to goal
            analysis["current_phase"] = "navigate_to_goal"
            analysis["recommended_action"] = self.detector.calculate_direction_to(
                player.position, goal.position
            )
        
        else:
            # Can't find key elements, explore
            analysis["current_phase"] = "explore"
            analysis["recommended_action"] = np.random.randint(1, 5)
        
        return analysis


def test_detector(game_id: str = "ls20"):
    """Test the visual detector on a game."""
    try:
        import arc_agi
    except ImportError:
        print("arc_agi not installed, cannot test")
        return
    
    arc = arc_agi.Arcade()
    env = arc.make(game_id)
    
    print(f"\n{'='*60}")
    print(f"🔍 VISUAL DETECTOR TEST: {game_id}")
    print(f"{'='*60}")
    
    frame = env.reset()
    f = np.array(frame.frame)
    if len(f.shape) == 3:
        f = f[0]
    
    print(f"Frame shape: {f.shape}")
    print(f"Unique values: {np.unique(f)}")
    
    detector = VisualDetector()
    detections = detector.detect_all(f)
    
    print(f"\nDetected elements:")
    for elem_type, detection in detections.items():
        print(f"  {elem_type.value}:")
        print(f"    Position: {detection.position}")
        print(f"    Bounds: {detection.bounds}")
        print(f"    Confidence: {detection.confidence:.2f}")
        if detection.pattern is not None:
            print(f"    Pattern shape: {detection.pattern.shape}")
    
    # Test navigator
    print(f"\nDirected Navigator analysis:")
    navigator = DirectedNavigator()
    analysis = navigator.analyze_frame(f)
    print(f"  Phase: {analysis['current_phase']}")
    print(f"  Recommended action: {analysis['recommended_action']}")
    print(f"  Plus sign found: {analysis['plus_sign_found']}")
    print(f"  Goal found: {analysis['goal_found']}")
    print(f"  Player found: {analysis['player_found']}")


if __name__ == "__main__":
    import sys
    game = sys.argv[1] if len(sys.argv) > 1 else "ls20"
    test_detector(game)
