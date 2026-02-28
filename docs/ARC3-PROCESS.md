# ARC-AGI-3 Process — Master Reference

**Purpose:** Document the process for solving ARC-AGI-3 interactive games.
**Last Updated:** 2026-02-27
**Status:** Contest starting soon — this is our playbook.

---

## Quick Start

```bash
cd ~/sybil-arc-challenge
source .venv/bin/activate
# API key is in .env — ready to use
```

**Before each session:**
1. Read this file
2. Check `knowledge/corrections/` — don't repeat mistakes
3. Check `knowledge/insights/` — apply meta-learnings
4. Check `docs/GAME_NOTES.md` — game-specific mechanics

---

## ARC-3 vs ARC-2

| Aspect | ARC-2 (Old) | ARC-3 (Current) |
|--------|-------------|-----------------|
| Format | Static puzzles | Interactive games |
| Task | Input → Output transformation | Action sequences |
| Scoring | Correct/Incorrect | Efficiency vs human baseline |
| Levels | 1 per task | Multiple levels per game |
| Time | Unlimited | Real-time interaction |

**Key shift:** We're not finding transformations anymore. We're learning game mechanics and optimizing action sequences.

---

## The 6-Step Process for ARC-3

### Step 1: EXPLORE — Learn Before Winning

**Budget:** First 50 actions are for exploration, not winning.

```python
# Test each action in isolation
for action in available_actions:
    frame_before = current_frame.copy()
    env.step(action)
    frame_after = get_current_frame()
    diff = compare_frames(frame_before, frame_after)
    print(f"{action}: {diff['changed_pixels']} pixels changed")
```

**Document:**
- Which pixels change for each action
- What elements are player-controlled
- What elements are stationary
- Where are walls/boundaries

---

### Step 2: MAP — Identify Game Elements

**Checklist:**
- [ ] Player element (what I control)
- [ ] Goal/target (what I'm trying to reach/match)
- [ ] Obstacles (walls, hazards)
- [ ] Collectibles (if any)
- [ ] Interactive objects (doors, switches)

**Visual analysis:**
```python
from visual_analyzer import frame_to_emoji, analyze_frame
print(frame_to_emoji(current_frame[20:50, 15:45]))  # Crop to relevant area
print(analyze_frame(current_frame))
```

---

### Step 3: THEORIZE — 3 Hypotheses for Win Condition

**Before attempting to win, generate 3 theories:**

1. **Theory A:** "Move player to match/overlap the target shape"
2. **Theory B:** "Collect all items of color X"
3. **Theory C:** "Reach specific coordinates (exit zone)"

**Argue AGAINST each:**
- "Theory A fails because 50 actions of movement didn't trigger completion"
- "Theory B fails because I don't see any collectible items"
- "Theory C is possible — need to find the exit zone"

**Select the surviving theory and test it.**

---

### Step 4: PLAN — Sequence of Actions

**With the winning theory, plan the path:**

```
Current position: (45, 39)
Target position: (31, 20)
Path: 14 UP + 19 LEFT = 33 actions minimum

But human baseline is 29... so there must be a shorter path or mechanic.
```

**Consider:**
- Direct path vs shortcuts
- Special mechanics (teleports, phase-through)
- ACTION2 mystery (toggle? special ability?)

---

### Step 5: EXECUTE — Run and Observe

```python
plan = ["ACTION1"] * 14 + ["ACTION3"] * 19  # UP + LEFT
for action in plan:
    state = env.step(action)
    if state.levels_completed > current_level:
        print("LEVEL COMPLETE!")
        break
    if state.terminated:
        print("Game ended — check what happened")
        break
```

**If it fails:** Go back to THEORIZE with new information.

---

### Step 6: LEARN — Update Knowledge Base

**On success:**
```bash
# Add to insights
echo "Win condition for ls20: reach coordinates (X, Y)" >> knowledge/insights/ls20.md
```

**On failure:**
```bash
# Add correction
echo "Theory 'overlap shapes' was wrong because..." >> knowledge/corrections/ls20.md
```

**On pattern discovery:**
```python
# If mechanic applies to multiple games, add to promoted
from pattern_db import save_pattern
save_pattern(
    name="coordinate_target",
    description="Win by reaching specific coordinates",
    games=["ls20", "ft09"]
)
```

---

## Knowledge Structure

```
~/sybil-arc-challenge/knowledge/
├── patterns.db           # SQLite — successful mechanics
├── corrections/          # When my theory was wrong
│   ├── README.md
│   └── 2026-02-25-ls20.md
├── insights/             # Meta-patterns across games
│   ├── README.md
│   └── 2026-02-25-exploration.md
└── promoted/             # Validated, high-confidence learnings
    └── README.md
```

---

## Game-Specific Notes

See `docs/GAME_NOTES.md` for per-game discoveries:
- ls20: Movement mechanics, element identification
- ft09: (Not yet explored)
- Others: (Add as we play)

---

## Scripts Reference

| Script | Purpose |
|--------|---------|
| `visual_analyzer.py` | Render frames, compare changes |
| `sybil_agent.py` | Main agent orchestrator |
| `exploration_agent.py` | Systematic action discovery |
| `failure_memory.py` | Track what didn't work |
| `pattern_db.py` | Store successful mechanics |

---

## Human Baselines

We're scored against human efficiency. Know the baseline before attempting:

**ls20:**
| Level | Human Actions |
|-------|---------------|
| 1 | 29 |
| 2 | 41 |
| 3 | 172 |
| 4 | 49 |
| 5 | 53 |
| 6 | 62 |
| 7 | 82 |

**Goal:** Match or beat human action count.

---

## What We Learned from ARC-2

These principles from ARC-2 still apply:

1. **Meflex Method** — 3 hypotheses + argue against each (now for win conditions, not transformations)
2. **Visual Verification** — Render and look at the game state
3. **Failure Memory** — Don't repeat mistakes
4. **Pattern Reuse** — Similar games may have similar mechanics

---

## Daily Practice Checklist

- [ ] Read this document
- [ ] Pick a game to practice
- [ ] Follow the 6-step process
- [ ] Update corrections/ if theories were wrong
- [ ] Update insights/ if meta-patterns discovered
- [ ] Commit changes to git

---

*This is our ARC-3 playbook. Update after every session.*
