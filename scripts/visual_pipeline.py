"""
Visual Pipeline for ARC-AGI-3 (ls20)

Robust detection for all game elements based on Bridget's grounded knowledge.

Once these 4 detectors are solid, goal-directed logic is trivial:
- state != goal? -> navigate to plus, hover
- state == goal? -> navigate to goal, overlap
- done!

Saber ⚔️ | Bridget 🎮 | Sybil 🧠
"""

import numpy as np
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass
import sys
import os

sys.path.append(os.path.dirname(__file__))


@dataclass
class Detection:
    """A detected element with position and extracted data."""
    found: bool
    position: Optional[Tuple[int, int]] = None  # (x, y) center
    bounds: Optional[Tuple[int, int, int, int]] = None  # (x1, y1, x2, y2)
    pattern: Optional[np.ndarray] = None  # Extracted pattern
    confidence: float = 0.0


class VisualPipeline:
    """
    Robust visual detection for ls20 game elements.
    
    Elements (from Bridget's grounding):
    1. Plus Sign: White cross on gray background
    2. Goal Pattern: Pattern in dark square (static)
    3. State Indicator: Bottom-left HUD (fixed position)
    4. Player: Orange top + blue bottom composite block
    """
    
    # ls20 color palette (discovered empirically)
    # Values are indices in a 16-color palette
    WHITE = 0  # Actually the brightest color in frame
    
    def __init__(self):
        self.frame_shape = None
        self.color_stats = {}
        
        # Cache detections for efficiency
        self.last_frame_hash = None
        self.cached_detections = {}
        
        # Learn color mappings from first frame
        self.color_map = {}
        self.initialized = False
    
    def _hash_frame(self, frame: np.ndarray) -> str:
        return hash(frame.tobytes())
    
    def _init_from_frame(self, frame: np.ndarray):
        """Learn color mappings from the frame."""
        unique_colors = np.unique(frame)
        self.color_stats = {
            'unique': unique_colors,
            'min': frame.min(),
            'max': frame.max(),
            'mean': frame.mean(),
        }
        
        # Find "white" (brightest) and "black" (darkest)
        self.color_map['white'] = frame.max()
        self.color_map['black'] = frame.min()
        
        # Gray is somewhere in middle
        mid = (frame.max() + frame.min()) / 2
        grays = unique_colors[(unique_colors > frame.min()) & (unique_colors < frame.max())]
        if len(grays) > 0:
            self.color_map['gray'] = grays[len(grays)//2]
        
        self.initialized = True
    
    def process_frame(self, frame: np.ndarray) -> Dict[str, Detection]:
        """
        Process a frame and detect all elements.
        
        Returns dict with keys: 'plus_sign', 'goal', 'state', 'player'
        """
        # Ensure 2D
        if len(frame.shape) == 3:
            frame = frame[0]
        if len(frame.shape) != 2:
            return self._empty_detections()
        
        self.frame_shape = frame.shape
        
        # Initialize color mappings
        if not self.initialized:
            self._init_from_frame(frame)
        
        # Check cache
        frame_hash = self._hash_frame(frame)
        if frame_hash == self.last_frame_hash:
            return self.cached_detections
        
        # Run all detectors
        detections = {
            'plus_sign': self.detect_plus_sign(frame),
            'goal': self.detect_goal_pattern(frame),
            'state': self.detect_state_indicator(frame),
            'player': self.detect_player(frame),
        }
        
        # Cache
        self.last_frame_hash = frame_hash
        self.cached_detections = detections
        
        return detections
    
    def _empty_detections(self) -> Dict[str, Detection]:
        return {
            'plus_sign': Detection(found=False),
            'goal': Detection(found=False),
            'state': Detection(found=False),
            'player': Detection(found=False),
        }
    
    # =========================================================================
    # DETECTOR 1: PLUS SIGN (Rare color, cross pattern)
    # =========================================================================
    
    def detect_plus_sign(self, frame: np.ndarray) -> Detection:
        """
        Detect plus sign - the transformation trigger.
        
        Strategy: Look for rare colors that might be distinctive markers.
        In ls20, the plus sign is color value 1 (rare, distinct).
        Also look for cross/plus shaped patterns.
        """
        h, w = frame.shape
        
        # Find rare colors (< 2% of frame) - likely to be special markers
        unique, counts = np.unique(frame, return_counts=True)
        total_pixels = h * w
        
        rare_colors = unique[(counts > 2) & (counts < total_pixels * 0.02)]
        
        best_pos = None
        best_score = 0
        
        for target_color in rare_colors:
            color_mask = frame == target_color
            color_coords = np.argwhere(color_mask)
            
            if len(color_coords) == 0:
                continue
            
            # For each pixel of this color, check for cross pattern
            for (y, x) in color_coords:
                if y < 2 or y >= h - 2 or x < 2 or x >= w - 2:
                    continue
                
                score = 1  # Base score for being a rare color
                
                # Check if surrounded by different color (isolated)
                neighbors = [
                    frame[y-1, x], frame[y+1, x],
                    frame[y, x-1], frame[y, x+1]
                ]
                same_neighbors = sum(1 for n in neighbors if n == target_color)
                diff_neighbors = 4 - same_neighbors
                
                # Isolated rare pixels are interesting
                if diff_neighbors >= 3:
                    score += 2
                
                # Check for cross pattern with this color
                up = 1 if y > 0 and frame[y-1, x] == target_color else 0
                down = 1 if y < h-1 and frame[y+1, x] == target_color else 0
                left = 1 if x > 0 and frame[y, x-1] == target_color else 0
                right = 1 if x < w-1 and frame[y, x+1] == target_color else 0
                
                arms = up + down + left + right
                if arms >= 2:
                    score += arms
                
                if score > best_score:
                    best_score = score
                    best_pos = (x, y)
        
        if best_pos and best_score >= 2:
            x, y = best_pos
            return Detection(
                found=True,
                position=best_pos,
                bounds=(x - 2, y - 2, x + 2, y + 2),
                confidence=min(1.0, best_score / 5),
            )
        
        return Detection(found=False)
    
    # =========================================================================
    # DETECTOR 2: GOAL PATTERN (Dark square with pattern)
    # =========================================================================
    
    def detect_goal_pattern(self, frame: np.ndarray) -> Detection:
        """
        Detect goal pattern - the target to match and overlap.
        
        In ls20: Goal is a pattern with orange (9) pixels inside a bordered area,
        usually in upper portion of screen (rows 8-16 based on frame analysis).
        """
        h, w = frame.shape
        
        # From frame analysis, goal is around rows 8-16, cols 32-48
        # It's bordered by color 3 (-) with pattern inside
        
        # Look for bordered regions with patterns
        border_color = 3  # The '-' character in our visualization
        pattern_color = 9  # Orange 'O'
        
        # Find regions bordered by color 3
        best_region = None
        best_score = 0
        
        # Scan for rectangular bordered regions
        for y in range(4, h - 10):
            for x in range(4, w - 10):
                # Check if this could be top-left of a bordered region
                if frame[y, x] == border_color:
                    # Look for the extent of the border
                    # Find width (scan right)
                    x2 = x
                    while x2 < w - 1 and frame[y, x2] == border_color:
                        x2 += 1
                    
                    # Find height (scan down)
                    y2 = y
                    while y2 < h - 1 and frame[y2, x] == border_color:
                        y2 += 1
                    
                    width = x2 - x
                    height = y2 - y
                    
                    # Valid bordered region?
                    if width >= 6 and height >= 6 and width < 20 and height < 20:
                        # Extract interior
                        interior = frame[y+1:y2-1, x+1:x2-1]
                        
                        if interior.size > 0:
                            # Check for pattern colors inside
                            has_pattern = np.any(interior == pattern_color) or len(np.unique(interior)) >= 2
                            
                            if has_pattern:
                                score = width * height + len(np.unique(interior)) * 10
                                if score > best_score:
                                    best_score = score
                                    best_region = (x, y, x + width, y + height)
        
        if best_region:
            x1, y1, x2, y2 = best_region
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            pattern = frame[y1:y2, x1:x2].copy()
            
            return Detection(
                found=True,
                position=(cx, cy),
                bounds=best_region,
                pattern=pattern,
                confidence=min(1.0, best_score / 200),
            )
        
        return Detection(found=False)
    
    # =========================================================================
    # DETECTOR 3: STATE INDICATOR (Bottom-left HUD)
    # =========================================================================
    
    def detect_state_indicator(self, frame: np.ndarray) -> Detection:
        """
        Detect state indicator - shows current player form.
        
        Strategy: Fixed position in bottom-left corner. Just crop it.
        """
        h, w = frame.shape
        
        # State indicator is in bottom-left quadrant
        # Typically a small region showing current pattern
        region_h = min(16, h // 4)
        region_w = min(16, w // 4)
        
        # Extract bottom-left region
        state_region = frame[h - region_h:, :region_w].copy()
        
        # Check if it has a meaningful pattern (not all same color)
        unique_colors = len(np.unique(state_region))
        
        if unique_colors >= 2:
            return Detection(
                found=True,
                position=(region_w // 2, h - region_h // 2),
                bounds=(0, h - region_h, region_w, h),
                pattern=state_region,
                confidence=0.8,  # High confidence since it's fixed position
            )
        
        return Detection(found=False, confidence=0.3)
    
    # =========================================================================
    # DETECTOR 4: PLAYER (Orange + Blue composite)
    # =========================================================================
    
    def detect_player(self, frame: np.ndarray) -> Detection:
        """
        Detect player token - orange top + blue bottom composite.
        
        Strategy: Find vertically adjacent pixels with different colors
        that form a distinctive pattern.
        """
        h, w = frame.shape
        
        # Player is a composite block - look for vertical color transitions
        # that are consistent horizontally
        
        best_pos = None
        best_score = 0
        
        # Scan for vertical transitions
        for y in range(1, h - 2):
            for x in range(1, w - 1):
                top_color = frame[y - 1, x]
                mid_color = frame[y, x]
                bot_color = frame[y + 1, x]
                
                # Look for pattern: top != mid, mid == bot or similar structure
                if top_color != mid_color:
                    score = 1
                    
                    # Check horizontal consistency (player is usually 2-3 pixels wide)
                    if x > 0:
                        if frame[y - 1, x - 1] == top_color and frame[y, x - 1] == mid_color:
                            score += 1
                    if x < w - 1:
                        if frame[y - 1, x + 1] == top_color and frame[y, x + 1] == mid_color:
                            score += 1
                    
                    # Check that these colors are relatively rare (player is distinct)
                    top_count = np.sum(frame == top_color)
                    mid_count = np.sum(frame == mid_color)
                    total = h * w
                    
                    if top_count < total * 0.1 and mid_count < total * 0.1:
                        score += 2  # Rare colors = likely player
                    
                    if score > best_score:
                        best_score = score
                        best_pos = (x, y)
        
        if best_pos and best_score >= 3:
            x, y = best_pos
            return Detection(
                found=True,
                position=best_pos,
                bounds=(x - 1, y - 1, x + 1, y + 1),
                confidence=min(1.0, best_score / 5),
            )
        
        return Detection(found=False)
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def patterns_match(self, pattern1: np.ndarray, pattern2: np.ndarray, 
                       threshold: float = 0.85) -> bool:
        """Check if two patterns match (for state vs goal comparison)."""
        if pattern1 is None or pattern2 is None:
            return False
        
        # Resize to same shape if needed
        if pattern1.shape != pattern2.shape:
            # Simple resize: crop larger to smaller
            min_h = min(pattern1.shape[0], pattern2.shape[0])
            min_w = min(pattern1.shape[1], pattern2.shape[1])
            pattern1 = pattern1[:min_h, :min_w]
            pattern2 = pattern2[:min_h, :min_w]
        
        match_ratio = np.mean(pattern1 == pattern2)
        return match_ratio >= threshold
    
    def get_direction_to(self, from_pos: Tuple[int, int], 
                         to_pos: Tuple[int, int]) -> int:
        """
        Get action (1-4) to move from one position toward another.
        
        Assumes: 1=up, 2=down, 3=left, 4=right
        """
        dx = to_pos[0] - from_pos[0]
        dy = to_pos[1] - from_pos[1]
        
        # Move in direction of largest difference
        if abs(dx) > abs(dy):
            return 4 if dx > 0 else 3  # right or left
        else:
            return 2 if dy > 0 else 1  # down or up
    
    def is_overlapping(self, pos1: Tuple[int, int], pos2: Tuple[int, int],
                       threshold: int = 3) -> bool:
        """Check if two positions are overlapping (within threshold)."""
        if pos1 is None or pos2 is None:
            return False
        dx = abs(pos1[0] - pos2[0])
        dy = abs(pos1[1] - pos2[1])
        return dx <= threshold and dy <= threshold


class GoalDirectedController:
    """
    High-level controller using VisualPipeline for goal-directed play.
    
    Logic (from Bridget):
    - state != goal? -> navigate to plus, hover
    - state == goal? -> navigate to goal, overlap
    - done!
    """
    
    def __init__(self):
        self.pipeline = VisualPipeline()
        self.phase = "init"
        self.hover_count = 0
        self.max_hovers = 10  # Safety limit
    
    def get_action(self, frame: np.ndarray) -> Tuple[int, str]:
        """
        Get next action based on current frame.
        
        Returns: (action_int, phase_description)
        """
        detections = self.pipeline.process_frame(frame)
        
        plus = detections['plus_sign']
        goal = detections['goal']
        state = detections['state']
        player = detections['player']
        
        # Debug info
        found = [k for k, v in detections.items() if v.found]
        
        # Check if state matches goal
        state_matches_goal = False
        if state.found and goal.found and state.pattern is not None and goal.pattern is not None:
            state_matches_goal = self.pipeline.patterns_match(state.pattern, goal.pattern)
        
        # Decision logic
        if state_matches_goal:
            # STATE == GOAL: Navigate to goal and overlap
            self.phase = "navigate_to_goal"
            
            if player.found and goal.found:
                if self.pipeline.is_overlapping(player.position, goal.position):
                    self.phase = "overlapping_goal"
                    # Should win soon, keep moving toward it
                    return (self.pipeline.get_direction_to(player.position, goal.position), self.phase)
                else:
                    action = self.pipeline.get_direction_to(player.position, goal.position)
                    return (action, self.phase)
            else:
                # Can't find player or goal, explore
                return (np.random.randint(1, 5), "explore_for_goal")
        
        else:
            # STATE != GOAL: Navigate to plus sign and hover
            if player.found and plus.found:
                if self.pipeline.is_overlapping(player.position, plus.position):
                    # On plus sign - hover to transform
                    self.phase = "hovering_on_plus"
                    self.hover_count += 1
                    
                    if self.hover_count > self.max_hovers:
                        # Too many hovers, something's wrong
                        self.phase = "hover_timeout"
                        return (np.random.randint(1, 5), self.phase)
                    
                    # Stay on plus (small movements to trigger transformation)
                    return (np.random.choice([1, 2, 3, 4]), self.phase)
                else:
                    # Navigate to plus sign
                    self.phase = "navigate_to_plus"
                    self.hover_count = 0
                    action = self.pipeline.get_direction_to(player.position, plus.position)
                    return (action, self.phase)
            
            elif plus.found and not player.found:
                # Found plus but not player, try to find player
                self.phase = "searching_for_player"
                return (np.random.randint(1, 5), self.phase)
            
            else:
                # Can't find plus sign, explore
                self.phase = "searching_for_plus"
                return (np.random.randint(1, 5), self.phase)


def test_pipeline(game_id: str = "ls20"):
    """Test the visual pipeline on a game."""
    try:
        import arc_agi
        from arcengine import GameAction
    except ImportError:
        print("arc_agi not installed")
        return
    
    arc = arc_agi.Arcade()
    env = arc.make(game_id)
    
    print(f"\n{'='*70}")
    print(f"🔍 VISUAL PIPELINE TEST: {game_id}")
    print(f"{'='*70}")
    
    frame_data = env.reset()
    frame = np.array(frame_data.frame)
    if len(frame.shape) == 3:
        frame = frame[0]
    
    print(f"\nFrame shape: {frame.shape}")
    print(f"Unique colors: {np.unique(frame)}")
    
    pipeline = VisualPipeline()
    detections = pipeline.process_frame(frame)
    
    print(f"\n📊 DETECTIONS:")
    for name, det in detections.items():
        status = "✅" if det.found else "❌"
        print(f"  {status} {name}:")
        if det.found:
            print(f"      Position: {det.position}")
            print(f"      Bounds: {det.bounds}")
            print(f"      Confidence: {det.confidence:.2f}")
            if det.pattern is not None:
                print(f"      Pattern shape: {det.pattern.shape}")
    
    # Test controller
    print(f"\n🎮 CONTROLLER TEST (10 steps):")
    controller = GoalDirectedController()
    
    for i in range(10):
        action, phase = controller.get_action(frame)
        print(f"  Step {i+1}: action={action}, phase={phase}")
        
        # Take action
        action_map = {1: GameAction.ACTION1, 2: GameAction.ACTION2, 
                      3: GameAction.ACTION3, 4: GameAction.ACTION4}
        ga = action_map.get(action, GameAction.ACTION1)
        frame_data = env.step(ga)
        frame = np.array(frame_data.frame)
        if len(frame.shape) == 3:
            frame = frame[0]
    
    print(f"\n{'='*70}")


if __name__ == "__main__":
    game = sys.argv[1] if len(sys.argv) > 1 else "ls20"
    test_pipeline(game)
