# Game Notes — ARC-AGI-3

Per-game discoveries and mechanics learned through exploration.

---

## ls20

**Date First Explored:** 2026-02-25
**Level 1 Status:** NOT SOLVED
**Human Baselines:** 29, 41, 172, 49, 53, 62, 82 actions (levels 1-7)

### Visual Layout (64x64 grid)

```
┌─────────────────────────────────────────┐
│  Yellow (walls)      │   Yellow         │
│                      │                  │
│  ┌─────────────────┐ │                  │
│  │  Green          │ │                  │
│  │  (playable)     │ │                  │
│  │     🔵⬛         │ │                  │  ← L-shape (colors 0,1)
│  │     ⬛⬛         │ │                  │     rows 31-33, cols 20-22
│  │     🔵          │ │                  │     STATIONARY
│  └─────────────────┘ │                  │
│                      │                  │
│  ┌─────────────────┐ │                  │
│  │  Green          │ │                  │
│  │      ?????      │ │                  │  ← Color12 block (5x2)
│  │      ?????      │ │                  │     rows 45-46, cols 39-43
│  │      🟤🟤🟤🟤🟤  │ │                  │     THIS IS CONTROLLED
│  └─────────────────┘ │                  │
│                      │                  │
└─────────────────────────────────────────┘
```

### Elements Identified

| Element | Colors | Position | Behavior |
|---------|--------|----------|----------|
| L-shape | 0 (black), 1 (blue) | rows 31-33, cols 20-22 | STATIONARY — doesn't move with actions |
| Moveable block | 12 | rows 45-46, cols 39-43 | CONTROLLED by actions |
| Brown block | 9 | near moveable block | Unknown purpose |
| Playable area | 3 (green) | large regions | Traversable |
| Walls | 4 (yellow), 5 (gray) | borders | Block movement |
| Unknown | 8, 11 | bottom edge | UI elements? |

### Action Effects

| Action | Effect on Color12 Block | Pixels Changed |
|--------|------------------------|----------------|
| ACTION1 | Moves UP | 52 |
| ACTION2 | Minor effect (toggle?) | 2 |
| ACTION3 | Moves LEFT | 52 |
| ACTION4 | Moves RIGHT | 52 |

### Movement Tests

**Test 1:** 5x ACTION3 (LEFT)
- Block moved from col 39-43 to col 19
- Hit wall, stopped moving
- Level NOT complete

**Test 2:** 50x ACTION1 (UP)
- Level NOT complete
- Block position unclear

**Test 3:** 30x ACTION4 (RIGHT) + 20x ACTION2 (DOWN?)
- Level NOT complete
- L-shape position unchanged throughout

### Hypotheses to Test

1. **Shape Matching:** Does color12 need to match/overlap the L-shape?
2. **Target Zone:** Is there a hidden target area to reach?
3. **ACTION2 Mystery:** What does the 2-pixel change do?
4. **Multi-element Coordination:** Do multiple pieces need to interact?

### Next Session TODO

- [ ] Move color12 toward the L-shape (UP + LEFT from start)
- [ ] Test what ACTION2 does more carefully
- [ ] Watch for any "win condition" triggers
- [ ] Try reaching rows 31-33, cols 20-22 area

---

## ft09 (Not Yet Explored)

Available via API. Mentioned in docs as another starter game.

---

## Game Discovery Commands

```python
# List available games
import arc_agi
arc = arc_agi.Arcade()
envs = arc.get_environments()
for e in envs:
    print(e.game_id, e.title)

# Analyze a frame
from scripts.visual_analyzer import frame_to_emoji, analyze_frame
import numpy as np

state = env.reset()
frame = np.array(state.frame[0])
print(frame_to_emoji(frame[20:40, 20:40]))  # Crop view
print(analyze_frame(frame))

# Track changes
from scripts.visual_analyzer import compare_frames
diff = compare_frames(frame_before, frame_after)
print(diff)
```

---

*Update this file after each game session.*
