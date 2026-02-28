"""
Diagnostic script to understand ARC-AGI-3 game mechanics.
"""

import arc_agi
from arcengine import GameAction
import numpy as np
from collections import defaultdict
import hashlib


def hash_frame(frame) -> str:
    if frame is None:
        return "none"
    if isinstance(frame, list):
        data = str(frame).encode()
    elif hasattr(frame, 'tobytes'):
        data = frame.tobytes()
    else:
        data = str(frame).encode()
    return hashlib.md5(data).hexdigest()[:8]


def diagnose_game(game_id: str, num_actions: int = 200):
    """Analyze what actions actually do in a game."""
    
    print(f"\n{'='*60}")
    print(f"DIAGNOSING: {game_id}")
    print(f"{'='*60}")
    
    arc = arc_agi.Arcade()
    env = arc.make(game_id)
    frame = env.reset()
    
    print(f"\nGame info:")
    print(f"  Win levels: {frame.win_levels}")
    print(f"  Available actions: {frame.available_actions}")
    
    # Analyze frame
    if frame.frame is not None:
        arr = np.array(frame.frame)
        print(f"  Frame shape: {arr.shape}")
        print(f"  Unique colors: {np.unique(arr)}")
        print(f"  Non-zero pixels: {np.sum(arr > 0)}")
    
    # Track action effects
    action_effects = defaultdict(lambda: {"changes": 0, "no_change": 0})
    state_hashes = set()
    transitions = []
    
    INT_TO_ACTION = {a.value: a for a in GameAction}
    
    last_hash = hash_frame(frame.frame)
    state_hashes.add(last_hash)
    
    print(f"\nRunning {num_actions} random actions...")
    
    for i in range(num_actions):
        # Pick a random available action
        available = frame.available_actions or [1, 2, 3, 4]
        action_id = np.random.choice(available)
        action = INT_TO_ACTION.get(action_id, GameAction.ACTION1)
        
        # Handle ACTION6
        if action == GameAction.ACTION6:
            # Try clicking on non-zero pixels
            if frame.frame is not None:
                arr = np.array(frame.frame)
                if len(arr.shape) == 3:
                    arr = arr[0]
                nonzero = np.argwhere(arr > 0)
                if len(nonzero) > 0:
                    idx = np.random.randint(len(nonzero))
                    y, x = nonzero[idx]
                    action.set_data({"x": int(x), "y": int(y)})
                else:
                    action.set_data({"x": np.random.randint(0, 64), "y": np.random.randint(0, 64)})
            else:
                action.set_data({"x": np.random.randint(0, 64), "y": np.random.randint(0, 64)})
        
        # Take action
        frame = env.step(action)
        new_hash = hash_frame(frame.frame)
        
        # Track effect
        if new_hash != last_hash:
            action_effects[action_id]["changes"] += 1
            transitions.append((last_hash, action_id, new_hash))
        else:
            action_effects[action_id]["no_change"] += 1
        
        state_hashes.add(new_hash)
        last_hash = new_hash
        
        # Check for level completion
        if frame.levels_completed > 0:
            print(f"  🎉 Level completed at action {i+1}!")
            break
    
    # Report
    print(f"\nAction Effects:")
    for action_id in sorted(action_effects.keys()):
        data = action_effects[action_id]
        total = data["changes"] + data["no_change"]
        change_rate = data["changes"] / total if total > 0 else 0
        print(f"  ACTION{action_id}: {data['changes']}/{total} caused changes ({change_rate:.1%})")
    
    print(f"\nState Space:")
    print(f"  Unique states visited: {len(state_hashes)}")
    print(f"  Unique transitions: {len(set(transitions))}")
    
    # Analyze transition patterns
    if transitions:
        print(f"\nTransition patterns (first 10):")
        for t in transitions[:10]:
            print(f"  {t[0]} --[ACTION{t[1]}]--> {t[2]}")
    
    # Check if we're stuck in a loop
    if len(state_hashes) < 10:
        print(f"\n⚠️  Only {len(state_hashes)} unique states - agent may be stuck!")
        print("   The game likely requires specific action sequences to progress.")
    
    return {
        "game_id": game_id,
        "unique_states": len(state_hashes),
        "action_effects": dict(action_effects),
        "transitions": len(set(transitions))
    }


if __name__ == "__main__":
    import sys
    
    games = sys.argv[1:] if len(sys.argv) > 1 else ["ls20", "ft09", "vc33"]
    
    for game in games:
        try:
            diagnose_game(game, num_actions=500)
        except Exception as e:
            print(f"Error with {game}: {e}")
