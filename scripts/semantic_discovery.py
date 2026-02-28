"""
Semantic Discovery for ARC-AGI-3

Probes a new game to build a mental model BEFORE planning.
Colors and mechanics vary per game — we must discover them.

Phase 0: Discovery (~10-20 exploratory actions)
Phase 1: Planning (using discovered model)
Phase 2: Execution

Bridget's insight: "For each game these things are going to be different,
so there has to be a time in the beginning when you first create the
mental model — that's when you assign colors to features."

Saber ⚔️ | Bridget 🎮 | Sybil 🔬
2026-02-28
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import Counter
import sys
import os

sys.path.append(os.path.dirname(__file__))

try:
    import arc_agi
    from arcengine import GameAction
    HAS_ARC = True
except ImportError:
    HAS_ARC = False


@dataclass
class GameSemantics:
    """Discovered semantics for a game."""
    # Colors
    player_colors: Set[int] = field(default_factory=set)
    floor_color: int = -1
    wall_colors: Set[int] = field(default_factory=set)
    goal_colors: Set[int] = field(default_factory=set)
    special_colors: Set[int] = field(default_factory=set)  # Plus signs, power-ups, etc.
    
    # Positions
    player_position: Tuple[int, int] = (0, 0)
    goal_position: Tuple[int, int] = (0, 0)
    special_positions: List[Tuple[int, int]] = field(default_factory=list)
    
    # HUD regions (fixed positions)
    state_indicator_region: Tuple[int, int, int, int] = (0, 0, 0, 0)  # x1, y1, x2, y2
    resource_bar_region: Tuple[int, int, int, int] = (0, 0, 0, 0)
    
    # Movement
    cells_per_action: int = 1  # How many cells player moves per action
    
    # Patterns
    initial_state_pattern: Optional[np.ndarray] = None
    goal_pattern: Optional[np.ndarray] = None
    state_matches_goal: bool = False
    
    # Confidence
    discovery_complete: bool = False
    actions_used: int = 0


class SemanticDiscovery:
    """
    Discovers game semantics through careful probing.
    
    Strategy:
    1. Analyze initial frame (static analysis)
    2. Take a few exploratory actions (dynamic analysis)
    3. Build complete GameSemantics model
    """
    
    def __init__(self, max_discovery_actions: int = 15):
        self.max_actions = max_discovery_actions
        self.frames_seen: List[np.ndarray] = []
        self.actions_taken: List[int] = []
        self.semantics = GameSemantics()
    
    def analyze_initial_frame(self, frame: np.ndarray) -> Dict:
        """
        Static analysis of the first frame.
        
        Discovers:
        - Color distribution (floor = most common)
        - Rare colors (likely special elements)
        - HUD regions (bottom area typically)
        - Bordered regions (likely goals)
        """
        if len(frame.shape) == 3:
            frame = frame[0]
        
        self.frames_seen.append(frame.copy())
        height, width = frame.shape
        
        results = {
            'height': height,
            'width': width,
            'colors': {},
            'hud_detected': False,
            'bordered_regions': [],
            'rare_clusters': [],
        }
        
        # 1. Color frequency analysis
        unique, counts = np.unique(frame, return_counts=True)
        color_freq = dict(zip(unique.tolist(), counts.tolist()))
        results['colors'] = color_freq
        
        # Floor = most common color (usually >30% of pixels)
        total_pixels = height * width
        for color, count in sorted(color_freq.items(), key=lambda x: -x[1]):
            if count / total_pixels > 0.25:
                self.semantics.floor_color = color
                print(f"  🟩 Floor color: {color} ({count/total_pixels*100:.1f}% of pixels)")
                break
        
        # 2. Rare colors = special elements (< 1% of pixels)
        rare_threshold = total_pixels * 0.01
        for color, count in color_freq.items():
            if count < rare_threshold and count > 0:
                self.semantics.special_colors.add(color)
                # Find positions of this color
                positions = np.argwhere(frame == color)
                if len(positions) > 0:
                    center = positions.mean(axis=0)
                    results['rare_clusters'].append({
                        'color': color,
                        'count': count,
                        'center': (int(center[1]), int(center[0])),  # x, y
                        'positions': positions.tolist()[:10]  # First 10
                    })
        
        print(f"  ✨ Rare colors (specials): {self.semantics.special_colors}")
        
        # 3. Detect HUD region (bottom ~15% of screen, typically different pattern)
        hud_start = int(height * 0.85)
        hud_region = frame[hud_start:, :]
        main_region = frame[:hud_start, :]
        
        # HUD often has different color distribution
        hud_colors = set(np.unique(hud_region))
        main_colors = set(np.unique(main_region))
        hud_only = hud_colors - main_colors
        
        if len(hud_only) > 0:
            results['hud_detected'] = True
            self.semantics.resource_bar_region = (0, hud_start, width, height)
            print(f"  📊 HUD detected at y={hud_start}+, unique colors: {hud_only}")
        
        # 4. Detect state indicator (bottom-left corner, typically 8x8 to 12x12)
        state_region = frame[height-12:height-2, 2:14]  # Approximate position
        self.semantics.state_indicator_region = (2, height-12, 14, height-2)
        self.semantics.initial_state_pattern = state_region.copy()
        print(f"  📍 State indicator region: {self.semantics.state_indicator_region}")
        
        # 5. Find bordered regions (potential goals)
        bordered = self._find_bordered_regions(frame)
        results['bordered_regions'] = bordered
        if bordered:
            # First bordered region is likely the goal
            goal_region = bordered[0]
            self.semantics.goal_position = (goal_region['center_x'], goal_region['center_y'])
            self.semantics.goal_colors.add(goal_region.get('border_color', -1))
            print(f"  🎯 Potential goal at {self.semantics.goal_position}")
        
        return results
    
    def _find_bordered_regions(self, frame: np.ndarray) -> List[Dict]:
        """
        Find regions with distinct borders (likely interactive elements).
        """
        bordered = []
        height, width = frame.shape
        
        # Look for rectangular regions with consistent border color
        # Simple approach: find color 0 (often border) rectangles
        border_color = 0  # Common border color
        
        border_mask = (frame == border_color)
        if border_mask.sum() > 10:  # Has some border pixels
            # Find connected components (simplified)
            border_positions = np.argwhere(border_mask)
            if len(border_positions) > 0:
                # Cluster border pixels
                min_y, min_x = border_positions.min(axis=0)
                max_y, max_x = border_positions.max(axis=0)
                
                bordered.append({
                    'border_color': border_color,
                    'bounds': (min_x, min_y, max_x, max_y),
                    'center_x': (min_x + max_x) // 2,
                    'center_y': (min_y + max_y) // 2,
                })
        
        return bordered
    
    def probe_movement(self, game, action: int) -> Dict:
        """
        Take an action and analyze what changed.
        
        Discovers:
        - Player colors (what moved)
        - Movement distance (cells per action)
        - Wall colors (what blocked movement)
        """
        if not HAS_ARC:
            return {'error': 'arc_agi not available'}
        
        before = self.frames_seen[-1].copy()
        
        # Take action
        result = game.step(action)
        after = result.frame
        if len(after.shape) == 3:
            after = after[0]
        
        self.frames_seen.append(after.copy())
        self.actions_taken.append(action)
        self.semantics.actions_used += 1
        
        # Find what changed
        diff = (before != after)
        changed_positions = np.argwhere(diff)
        
        probe_result = {
            'action': action,
            'pixels_changed': len(changed_positions),
            'level_complete': result.is_complete,
            'player_moved': False,
            'movement_distance': 0,
        }
        
        if len(changed_positions) > 0:
            # Something changed — likely player movement
            probe_result['player_moved'] = True
            
            # Find the colors that changed
            before_colors = set(before[diff])
            after_colors = set(after[diff])
            
            # Colors that appeared = player's new position
            # Colors that disappeared = player's old position
            player_colors = before_colors | after_colors
            self.semantics.player_colors.update(player_colors)
            
            # Estimate movement distance
            if len(changed_positions) >= 2:
                # Distance between first and last changed position
                y_range = changed_positions[:, 0].max() - changed_positions[:, 0].min()
                x_range = changed_positions[:, 1].max() - changed_positions[:, 1].min()
                probe_result['movement_distance'] = max(y_range, x_range)
                
                # Update cells per action estimate
                if probe_result['movement_distance'] > self.semantics.cells_per_action:
                    self.semantics.cells_per_action = probe_result['movement_distance']
            
            print(f"  🎮 Action {action}: {len(changed_positions)} pixels changed, "
                  f"player colors: {player_colors}, distance: {probe_result['movement_distance']}")
        else:
            # Nothing changed — might have hit a wall or invalid action
            print(f"  🧱 Action {action}: no change (wall or invalid?)")
        
        return probe_result
    
    def discover(self, game) -> GameSemantics:
        """
        Full discovery process for a new game.
        
        1. Static analysis of initial frame
        2. Probe each direction once
        3. Identify player, walls, specials
        4. Compare state vs goal
        """
        print("\n🔍 SEMANTIC DISCOVERY PHASE")
        print("=" * 40)
        
        # Get initial frame
        initial_frame = game.get_frame()
        if len(initial_frame.shape) == 3:
            initial_frame = initial_frame[0]
        
        # 1. Static analysis
        print("\n📸 Analyzing initial frame...")
        static_results = self.analyze_initial_frame(initial_frame)
        
        # 2. Probe each direction
        print("\n🎮 Probing movement...")
        directions = {1: 'UP', 2: 'DOWN', 3: 'LEFT', 4: 'RIGHT'}
        
        for action, name in directions.items():
            if self.semantics.actions_used >= self.max_actions:
                break
            print(f"\n  Testing {name}...")
            probe = self.probe_movement(game, action)
            
            if probe.get('level_complete'):
                print("  🎉 Level completed during discovery!")
                break
        
        # 3. Identify walls (colors that are common but not floor or player)
        all_colors = set(static_results['colors'].keys())
        identified = {self.semantics.floor_color} | self.semantics.player_colors | self.semantics.special_colors
        self.semantics.wall_colors = all_colors - identified - {-1}
        print(f"\n🧱 Wall colors (inferred): {self.semantics.wall_colors}")
        
        # 4. Compare initial state vs goal
        if self.semantics.initial_state_pattern is not None:
            # Extract goal pattern from goal region
            if len(self.frames_seen) > 0:
                frame = self.frames_seen[0]
                gx, gy = self.semantics.goal_position
                if gx > 5 and gy > 5:
                    goal_pattern = frame[gy-5:gy+5, gx-5:gx+5]
                    self.semantics.goal_pattern = goal_pattern
                    
                    # Compare patterns
                    state = self.semantics.initial_state_pattern
                    goal = self.semantics.goal_pattern
                    
                    # Resize for comparison
                    min_h = min(state.shape[0], goal.shape[0])
                    min_w = min(state.shape[1], goal.shape[1])
                    
                    if min_h > 0 and min_w > 0:
                        state_crop = state[:min_h, :min_w]
                        goal_crop = goal[:min_h, :min_w]
                        match_ratio = np.mean(state_crop == goal_crop)
                        self.semantics.state_matches_goal = match_ratio > 0.8
                        print(f"\n🎯 State vs Goal match: {match_ratio*100:.1f}% "
                              f"({'MATCH!' if self.semantics.state_matches_goal else 'need transformation'})")
        
        # Mark discovery complete
        self.semantics.discovery_complete = True
        
        print("\n" + "=" * 40)
        print("📋 DISCOVERY COMPLETE")
        print(f"   Floor: {self.semantics.floor_color}")
        print(f"   Player: {self.semantics.player_colors}")
        print(f"   Walls: {self.semantics.wall_colors}")
        print(f"   Specials: {self.semantics.special_colors}")
        print(f"   Cells/action: {self.semantics.cells_per_action}")
        print(f"   State matches goal: {self.semantics.state_matches_goal}")
        print(f"   Actions used: {self.semantics.actions_used}")
        print("=" * 40 + "\n")
        
        return self.semantics


def test_discovery():
    """Test semantic discovery on a game."""
    if not HAS_ARC:
        print("arc_agi not available")
        return
    
    # Create game
    game = arc_agi.make_game("ls20")
    game.reset()
    
    # Run discovery
    discovery = SemanticDiscovery(max_discovery_actions=10)
    semantics = discovery.discover(game)
    
    print("\nReady for planning phase with discovered semantics!")
    return semantics


if __name__ == "__main__":
    test_discovery()
