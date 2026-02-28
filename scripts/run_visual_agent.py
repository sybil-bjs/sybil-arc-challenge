"""
Run Visual Pipeline Agent on ls20

Tests whether the visual pipeline + goal-directed logic can complete levels.

Saber ⚔️ | Bridget 🎮 | Sybil 🧠
"""

import sys
import os
import numpy as np

sys.path.append(os.path.dirname(__file__))

try:
    import arc_agi
    from arcengine import GameAction
except ImportError:
    print("arc_agi not installed")
    sys.exit(1)

from visual_pipeline import VisualPipeline, GoalDirectedController


def run_visual_agent(game_id: str = "ls20", max_actions: int = 1000):
    """Run the visual pipeline agent."""
    
    arc = arc_agi.Arcade()
    env = arc.make(game_id)
    
    print(f"\n{'='*70}")
    print(f"🎯 VISUAL PIPELINE AGENT: {game_id}")
    print(f"    Max actions: {max_actions}")
    print(f"{'='*70}\n")
    
    controller = GoalDirectedController()
    
    frame_data = env.reset()
    frame = np.array(frame_data.frame)
    if len(frame.shape) == 3:
        frame = frame[0]
    
    action_map = {
        1: GameAction.ACTION1, 
        2: GameAction.ACTION2, 
        3: GameAction.ACTION3, 
        4: GameAction.ACTION4
    }
    
    levels_completed = 0
    actions_taken = 0
    phase_counts = {}
    last_phase = None
    
    for i in range(max_actions):
        # Get action from visual controller
        action_int, phase = controller.get_action(frame)
        
        # Track phases
        phase_counts[phase] = phase_counts.get(phase, 0) + 1
        
        # Log phase changes
        if phase != last_phase:
            print(f"  [{i:4d}] Phase: {phase}")
            last_phase = phase
        
        # Execute action
        ga = action_map.get(action_int, GameAction.ACTION1)
        frame_data = env.step(ga)
        
        frame = np.array(frame_data.frame)
        if len(frame.shape) == 3:
            frame = frame[0]
        
        actions_taken += 1
        
        # Check for level completion
        current_level = getattr(frame_data, 'levels_completed', 0)
        if current_level > levels_completed:
            print(f"\n  🎉 LEVEL {current_level} COMPLETED after {actions_taken} actions!")
            levels_completed = current_level
            controller.hover_count = 0  # Reset for next level
        
        # Check for win
        if str(frame_data.state) == 'WIN':
            print(f"\n  🏆 GAME WON!")
            break
        
        # Progress indicator
        if (i + 1) % 100 == 0:
            det = controller.pipeline.cached_detections
            found = [k for k, v in det.items() if v.found]
            print(f"  [{i+1:4d}] Actions: {actions_taken}, Levels: {levels_completed}, Detecting: {found}")
    
    # Summary
    print(f"\n{'='*70}")
    print(f"📊 RESULTS")
    print(f"{'='*70}")
    print(f"  Game: {game_id}")
    print(f"  Actions taken: {actions_taken}")
    print(f"  Levels completed: {levels_completed}")
    print(f"  Efficiency: {levels_completed / max(actions_taken, 1) * 1000:.2f} levels per 1000 actions")
    print(f"\n  Phase breakdown:")
    for phase, count in sorted(phase_counts.items(), key=lambda x: -x[1]):
        pct = count / actions_taken * 100
        print(f"    {phase}: {count} ({pct:.1f}%)")
    print(f"{'='*70}\n")
    
    return levels_completed, actions_taken


if __name__ == "__main__":
    game = sys.argv[1] if len(sys.argv) > 1 else "ls20"
    actions = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    run_visual_agent(game, actions)
