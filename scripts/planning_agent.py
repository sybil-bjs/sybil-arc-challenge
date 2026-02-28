"""
Planning Agent for ARC-AGI-3 (ls20)

Mental model approach: UNDERSTAND first, then EXECUTE minimal actions.

Instead of 14K random actions, we:
1. Build spatial map from frame
2. Identify key positions (player, plus, goal)
3. Plan shortest path: player → plus → goal
4. Execute plan in 30-50 actions

Bridget's insight: Humans use mental models, not brute force.

Saber ⚔️ | Bridget 🎮 | Sybil 🧠
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from collections import deque
import sys
import os

sys.path.append(os.path.dirname(__file__))

try:
    import arc_agi
    from arcengine import GameAction
    HAS_ARC = True
except ImportError:
    HAS_ARC = False

from visual_pipeline import VisualPipeline


@dataclass
class Position:
    x: int
    y: int
    
    def __hash__(self):
        return hash((self.x, self.y))
    
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y
    
    def manhattan_distance(self, other: 'Position') -> int:
        return abs(self.x - other.x) + abs(self.y - other.y)


@dataclass
class GamePlan:
    """A complete plan to solve the level."""
    player_start: Position
    plus_position: Position
    goal_position: Position
    path_to_plus: List[int]  # Actions to reach plus
    hover_count: int         # How many times to hover on plus
    path_to_goal: List[int]  # Actions to reach goal after transformation
    total_actions: int


class SpatialMap:
    """
    Grid-based spatial representation of the game.
    
    Identifies walkable areas, obstacles, and key positions.
    """
    
    def __init__(self, frame: np.ndarray):
        self.frame = frame
        self.height, self.width = frame.shape
        
        # Identify walkable vs obstacle cells
        # Walkable = background colors (typically 3, 4, 5)
        # Obstacles = borders, other elements
        self.walkable = self._build_walkability_map()
    
    def _build_walkability_map(self) -> np.ndarray:
        """
        Determine which cells are walkable.
        
        Sybil's grid semantics (2026-02-28):
        - Floor = 3 (walkable)
        - Walls = 4 (blocks!)
        - Player = 9 + 12
        - Plus sign = 1
        - Goal = 8
        """
        # Floor is value 3 - this is the walkable path
        # Also allow walking on plus sign (1), goal (8), and player positions (9, 12)
        walkable_colors = {3, 1, 8, 9, 12}
        
        walkable = np.zeros_like(self.frame, dtype=bool)
        for color in walkable_colors:
            walkable |= (self.frame == color)
        
        return walkable
    
    def is_walkable(self, pos: Position) -> bool:
        """Check if a position is walkable."""
        if pos.x < 0 or pos.x >= self.width or pos.y < 0 or pos.y >= self.height:
            return False
        return self.walkable[pos.y, pos.x]
    
    def get_neighbors(self, pos: Position, movement_distance: int = 5) -> List[Tuple[Position, int]]:
        """
        Get reachable neighbors with the action to reach them.
        
        Sybil's insight: Player moves 5 CELLS per action, not 1!
        We simulate movement by checking if the path is clear.
        """
        # Actions: 1=up, 2=down, 3=left, 4=right
        directions = [
            (0, -1, 1),  # up
            (0, 1, 2),   # down
            (-1, 0, 3),  # left
            (1, 0, 4),   # right
        ]
        
        neighbors = []
        for dx, dy, action in directions:
            # Move up to 5 cells in this direction (or until blocked)
            final_pos = pos
            for step in range(1, movement_distance + 1):
                test_pos = Position(pos.x + dx * step, pos.y + dy * step)
                if self.is_walkable(test_pos):
                    final_pos = test_pos
                else:
                    break  # Hit a wall, stop here
            
            # Only add if we actually moved somewhere
            if final_pos != pos:
                neighbors.append((final_pos, action))
        
        return neighbors


class Pathfinder:
    """
    A* pathfinding on the spatial map.
    """
    
    def __init__(self, spatial_map: SpatialMap):
        self.map = spatial_map
    
    def find_path(self, start: Position, goal: Position) -> Optional[List[int]]:
        """
        Find shortest path from start to goal.
        
        Returns list of actions (1=up, 2=down, 3=left, 4=right).
        """
        if start == goal:
            return []
        
        # BFS for simplicity (A* would be faster for large maps)
        queue = deque([(start, [])])
        visited = {start}
        
        while queue:
            current, path = queue.popleft()
            
            for neighbor, action in self.map.get_neighbors(current):
                if neighbor in visited:
                    continue
                
                new_path = path + [action]
                
                if neighbor.manhattan_distance(goal) <= 2:
                    # Close enough - return path
                    return new_path
                
                visited.add(neighbor)
                queue.append((neighbor, new_path))
        
        # No path found - try direct movement
        return self._direct_path(start, goal)
    
    def _direct_path(self, start: Position, goal: Position) -> List[int]:
        """Generate direct path ignoring obstacles (fallback)."""
        path = []
        current = Position(start.x, start.y)
        
        max_steps = 100
        for _ in range(max_steps):
            if current.manhattan_distance(goal) <= 2:
                break
            
            dx = goal.x - current.x
            dy = goal.y - current.y
            
            if abs(dx) > abs(dy):
                action = 4 if dx > 0 else 3
                current = Position(current.x + (1 if dx > 0 else -1), current.y)
            else:
                action = 2 if dy > 0 else 1
                current = Position(current.x, current.y + (1 if dy > 0 else -1))
            
            path.append(action)
        
        return path


class PlanningAgent:
    """
    Agent that uses mental models and planning instead of exploration.
    
    Pipeline:
    1. Analyze frame → extract positions
    2. Build spatial map
    3. Plan path: player → plus → goal
    4. Execute plan
    """
    
    def __init__(self):
        self.pipeline = VisualPipeline()
        self.plan: Optional[GamePlan] = None
        self.plan_index: int = 0
        self.phase: str = "analyzing"
        self.actions_taken: int = 0
        self.hover_count: int = 0
    
    def analyze_and_plan(self, frame: np.ndarray) -> Optional[GamePlan]:
        """
        Analyze the frame and create a complete plan.
        
        KEY INSIGHT (Bridget 2026-02-28): Check if state ALREADY matches goal!
        If so, skip plus sign entirely — just navigate to goal.
        """
        # Get detections
        detections = self.pipeline.process_frame(frame)
        
        player = detections.get('player')
        plus_sign = detections.get('plus_sign')
        goal = detections.get('goal')
        state = detections.get('state')
        
        # Minimum requirements: player and goal
        if not (player and player.found and goal and goal.found):
            print("  ❌ Cannot build plan - missing player or goal")
            print(f"     Player: {player.found if player else False}")
            print(f"     Goal: {goal.found if goal else False}")
            return None
        
        # Extract positions
        player_pos = Position(int(player.position[0]), int(player.position[1]))
        goal_pos = Position(int(goal.position[0]), int(goal.position[1]))
        plus_pos = Position(int(plus_sign.position[0]), int(plus_sign.position[1])) if plus_sign and plus_sign.found else None
        
        print(f"  📍 Positions:")
        print(f"     Player: ({player_pos.x}, {player_pos.y})")
        print(f"     Goal:   ({goal_pos.x}, {goal_pos.y})")
        if plus_pos:
            print(f"     Plus:   ({plus_pos.x}, {plus_pos.y})")
        
        # Build spatial map
        spatial_map = SpatialMap(frame)
        pathfinder = Pathfinder(spatial_map)
        
        # KEY CHECK: Does state already match goal?
        hover_count = self._estimate_transformations(state, goal)
        state_matches_goal = (hover_count == 0)
        print(f"  🔍 State vs Goal: {'MATCH! ✅' if state_matches_goal else f'{hover_count} transformations needed'}")
        
        if state_matches_goal:
            # Skip plus sign — go directly to goal!
            print("  ⚡ STATE MATCHES GOAL — skipping plus sign!")
            path_to_goal = pathfinder.find_path(player_pos, goal_pos)
            print(f"  🗺️ Direct path to goal: {len(path_to_goal)} actions")
            
            return GamePlan(
                player_start=player_pos,
                plus_position=plus_pos or Position(0, 0),  # Unused but required
                goal_position=goal_pos,
                path_to_plus=[],  # Empty — skip plus sign
                hover_count=0,
                path_to_goal=path_to_goal,
                total_actions=len(path_to_goal)
            )
        
        # Need transformation — require plus sign
        if not plus_pos:
            print("  ❌ Need transformation but no plus sign found!")
            return None
        
        # Plan path to plus sign
        path_to_plus = pathfinder.find_path(player_pos, plus_pos)
        print(f"  🗺️ Path to plus: {len(path_to_plus)} actions")
        print(f"  🔄 Transformations needed: {hover_count}")
        
        # Plan path from plus to goal
        path_to_goal = pathfinder.find_path(plus_pos, goal_pos)
        print(f"  🗺️ Path to goal: {len(path_to_goal)} actions")
        
        total = len(path_to_plus) + hover_count + len(path_to_goal)
        print(f"  📊 Total planned actions: {total}")
        
        return GamePlan(
            player_start=player_pos,
            plus_position=plus_pos,
            goal_position=goal_pos,
            path_to_plus=path_to_plus,
            hover_count=hover_count,
            path_to_goal=path_to_goal,
            total_actions=total
        )
    
    def _estimate_transformations(self, state_det, goal_det) -> int:
        """
        Estimate how many transformations needed to match goal.
        
        Returns 0 if patterns already match (critical for skip-plus optimization).
        """
        if not state_det or not goal_det:
            return 5  # Default guess
        
        if state_det.pattern is None or goal_det.pattern is None:
            return 5
        
        # Compare patterns
        state = state_det.pattern
        goal = goal_det.pattern
        
        # Resize to compare
        min_h = min(state.shape[0], goal.shape[0])
        min_w = min(state.shape[1], goal.shape[1])
        
        state_crop = state[:min_h, :min_w]
        goal_crop = goal[:min_h, :min_w]
        
        # Count differences
        diff_ratio = np.mean(state_crop != goal_crop)
        
        # KEY: Return 0 if patterns match (or nearly match)
        if diff_ratio < 0.1:  # Less than 10% different = match
            return 0
        
        # Estimate: each transformation changes ~20% of pattern
        estimated = int(diff_ratio / 0.2)
        return min(max(1, estimated), 10)  # Between 1 and 10
    
    def get_action(self, frame: np.ndarray) -> Tuple[int, str]:
        """
        Get next action based on plan.
        """
        # Ensure 2D
        if len(frame.shape) == 3:
            frame = frame[0]
        
        # First call: analyze and plan
        if self.plan is None:
            print("\n  🧠 ANALYZING AND PLANNING...")
            self.plan = self.analyze_and_plan(frame)
            
            if self.plan is None:
                # Fallback to exploration
                self.phase = "exploring"
                return (np.random.randint(1, 5), self.phase)
            
            self.phase = "executing_path_to_plus"
            self.plan_index = 0
        
        # Execute plan phases
        if self.phase == "executing_path_to_plus":
            # Check if we should skip plus sign (state already matches goal)
            if len(self.plan.path_to_plus) == 0:
                print("  ⚡ Skipping plus phase — going directly to goal!")
                self.phase = "executing_path_to_goal"
                self.plan_index = 0
            elif self.plan_index < len(self.plan.path_to_plus):
                action = self.plan.path_to_plus[self.plan_index]
                self.plan_index += 1
                self.actions_taken += 1
                return (action, self.phase)
            else:
                # Reached plus sign, start hovering
                self.phase = "hovering"
                self.plan_index = 0
                self.hover_count = 0
        
        if self.phase == "hovering":
            # Skip if no hover needed
            if self.plan.hover_count == 0:
                self.phase = "executing_path_to_goal"
                self.plan_index = 0
            elif self.hover_count < self.plan.hover_count:
                self.hover_count += 1
                self.actions_taken += 1
                # Small movements to trigger transformation
                action = [1, 2, 3, 4][self.hover_count % 4]
                return (action, self.phase)
            else:
                # Done hovering, go to goal
                self.phase = "executing_path_to_goal"
                self.plan_index = 0
        
        if self.phase == "executing_path_to_goal":
            if self.plan_index < len(self.plan.path_to_goal):
                action = self.plan.path_to_goal[self.plan_index]
                self.plan_index += 1
                self.actions_taken += 1
                return (action, self.phase)
            else:
                # Should be at goal now
                self.phase = "at_goal"
        
        if self.phase == "at_goal":
            # Stay at goal / make small movements
            self.actions_taken += 1
            return (np.random.randint(1, 5), self.phase)
        
        # Fallback
        self.actions_taken += 1
        return (np.random.randint(1, 5), "fallback")


def run_planning_agent(game_id: str = "ls20", max_actions: int = 200):
    """Run the planning agent."""
    if not HAS_ARC:
        print("arc_agi not installed")
        return
    
    arc = arc_agi.Arcade()
    env = arc.make(game_id)
    
    print(f"\n{'='*70}")
    print(f"🧠 PLANNING AGENT: {game_id}")
    print(f"    Using mental models, not brute force!")
    print(f"{'='*70}")
    
    agent = PlanningAgent()
    
    frame_data = env.reset()
    frame = np.array(frame_data.frame)
    if len(frame.shape) == 3:
        frame = frame[0]
    
    action_map = {1: GameAction.ACTION1, 2: GameAction.ACTION2,
                  3: GameAction.ACTION3, 4: GameAction.ACTION4}
    
    levels_completed = 0
    last_phase = None
    
    for i in range(max_actions):
        action_int, phase = agent.get_action(frame)
        
        if phase != last_phase:
            print(f"\n  [{i:3d}] Phase: {phase}")
            last_phase = phase
        
        # Execute
        ga = action_map.get(action_int, GameAction.ACTION1)
        frame_data = env.step(ga)
        
        frame = np.array(frame_data.frame)
        if len(frame.shape) == 3:
            frame = frame[0]
        
        # Check level completion
        if frame_data.levels_completed > levels_completed:
            print(f"\n  🎉 LEVEL {frame_data.levels_completed} COMPLETED!")
            levels_completed = frame_data.levels_completed
        
        if str(frame_data.state) == 'WIN':
            print(f"\n  🏆 GAME WON!")
            break
    
    # Summary
    print(f"\n{'='*70}")
    print(f"📊 RESULTS")
    print(f"{'='*70}")
    print(f"  Actions taken: {agent.actions_taken}")
    print(f"  Levels completed: {levels_completed}")
    if agent.plan:
        print(f"  Planned actions: {agent.plan.total_actions}")
        print(f"  Efficiency: {levels_completed / max(agent.actions_taken, 1) * 100:.1f}%")
    print(f"{'='*70}\n")
    
    return levels_completed, agent.actions_taken


if __name__ == "__main__":
    game = sys.argv[1] if len(sys.argv) > 1 else "ls20"
    actions = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    run_planning_agent(game, actions)
