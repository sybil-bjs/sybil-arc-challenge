# ARC-AGI-2 Winning Process — Reference Guide

**Purpose:** Document the exact process that achieved 75% (3/4) on ARC-AGI-2 hard eval tasks, so we can apply it to ARC-AGI-3.

**Last Updated:** 2026-02-27
**Reference Repo:** https://github.com/confluence-labs/arc-agi-2 (97.92% on public eval)

---

## Quick Reference

When starting an ARC session:
1. Read this file
2. Follow the 6-step process below
3. Use the scripts in `scripts/`
4. Update `docs/GAME_NOTES.md` with discoveries

---

## The 6-Step Process

### Step 1: CAPTURE — Render the Puzzle as Image

```python
from visual_analyzer import frame_to_image, frame_to_emoji

# For quick view (terminal)
print(frame_to_emoji(puzzle_input))

# For visual analysis (save as image)
frame_to_image(puzzle_input, "/tmp/arc_input.png")
frame_to_image(puzzle_output, "/tmp/arc_output.png")
```

**Why:** Visual inspection catches patterns that raw numbers miss. Sybil can "see" the puzzle.

---

### Step 2: ANALYZE — Vision Model Examines the Puzzle

Use `describe_frame_for_llm()` or manually analyze:

**Checklist:**
- [ ] What objects are present? (shapes, colors, sizes)
- [ ] Where are they positioned? (corners, center, edges)
- [ ] What's the background color?
- [ ] Are there symmetries? (horizontal, vertical, rotational)
- [ ] Are there patterns? (repeating, gradients, sequences)
- [ ] What's the relationship between input and output?

**Template prompt:**
```
Looking at this ARC puzzle:

INPUT:
[emoji grid]

OUTPUT:
[emoji grid]

I observe:
1. Objects: ...
2. Positions: ...
3. Colors: ...
4. Transformation: The output appears to be the input with...
```

---

### Step 3: THEORIZE — Meflex Method (3 Hypotheses)

**Before writing ANY code:**

1. **Generate 3 different theories** for the transformation:
   - Theory A: "It's a reflection across the vertical axis"
   - Theory B: "Each object is duplicated to the right"
   - Theory C: "Colors are swapped according to a mapping"

2. **Argue AGAINST each theory:**
   - "Theory A fails because Example 2 shows objects moving DOWN, not flipping"
   - "Theory B fails because Example 3 has an odd number of objects"
   - "Theory C works for all examples!"

3. **Select the surviving theory** and proceed to code

**Why:** Prevents getting "married" to wrong ideas. Internal debate eliminates bad hypotheses early.

---

### Step 4: ITERATE — Sub-Agent Writes transform.py

**Architecture:**
- **Orchestrator (Opus):** Strategic thinking, theory selection, verification
- **Workers (Sonnet):** Code iteration, fast execution, 10 attempts max

**Sub-agent prompt template:**
```
You are solving an ARC-AGI puzzle.

TRAINING EXAMPLES:
[input1] → [output1]
[input2] → [output2]
[input3] → [output3]

THEORY: [the surviving theory from Step 3]

Write a Python function `transform(input_grid)` that:
1. Takes a 2D numpy array (values 0-9)
2. Returns the transformed output grid
3. Works on ALL training examples

Test your function on each training example before submitting.
```

**Iteration loop:**
```
attempt = 0
while attempt < 10:
    code = sub_agent.generate_transform()
    results = test_on_training(code)
    if all_correct(results):
        return code
    else:
        sub_agent.feedback(f"Failed on example {failed_idx}: expected {expected}, got {actual}")
        attempt += 1
```

---

### Step 5: VERIFY — Visual Verification

**After sub-agent claims success:**

1. **Render the sub-agent's output:**
   ```python
   predicted = transform(test_input)
   frame_to_image(predicted, "/tmp/predicted.png")
   ```

2. **Visually compare:**
   - Does it look right?
   - Are there obvious errors the code missed?
   - Does it match the pattern from training examples?

3. **Accept/Reject/Redirect:**
   - ✅ Accept: Solution looks correct → submit
   - ❌ Reject: Obvious visual error → send back with feedback
   - 🔄 Redirect: Partially right → adjust theory and retry

**Why:** Sub-agent self-assessment can be wrong (f560132c claimed PASS but failed test). Visual verification catches this.

---

### Step 6: LEARN — Save Patterns & Record Failures

**On success:**
```python
from pattern_db import save_pattern

save_pattern(
    name="axis_reflection",
    description="Reflect grid across vertical axis",
    code=successful_transform_code,
    task_ids=["4a21e3da"]
)
```

**On failure:**
```python
from failure_memory import record_failure, add_lesson

record_failure(
    task_id="f560132c",
    pattern_attempted="color_mapping_overlay",
    failure_mode="boundary_complexity",
    notes="Failed because boundary shapes are irregular"
)

add_lesson(
    failure_id=failure_id,
    lesson="Always check if boundaries are rectangular before assuming grid-based overlay"
)
```

**Why:** Builds cumulative knowledge. Next session starts with learned patterns.

---

## Confluence Labs Reference

**Repo:** https://github.com/confluence-labs/arc-agi-2
**Score:** 97.92% on ARC-AGI-2 public eval

### Their Architecture:
| Parameter | Value | Description |
|-----------|-------|-------------|
| GEMINI_CLI_AGENTS | 12 | Agents per test input |
| GEMINI_CLI_MAX_ITERATIONS | 10 | Max refinement loops per agent |
| GEMINI_CLI_CONCURRENCY | 132 | Max simultaneous sandboxes |
| WALL_CLOCK_LIMIT | 43200 (12h) | Total timeout |

### Key Insight:
They use **12 parallel agents** that each try different strategies, then vote on the best solution. External validation catches self-assessment errors.

### Our Enhancement:
- **Visual verification** (we can actually SEE the grids)
- **Persistent memory** (patterns learned across sessions)
- **Meflex method** (3 hypotheses + argue against)
- **Mac Tools execution** (local Python, no sandbox limits)

---

## Our ARC-AGI-2 Results

| Task | Status | Pattern | Notes |
|------|--------|---------|-------|
| 00dbd492 | ✅ CORRECT | size_based_fill | |
| 4a21e3da | ✅ CORRECT | axis_reflection | |
| d8e07eb2 | ✅ CORRECT | (unknown) | |
| f560132c | ❌ WRONG | color_mapping_overlay | Boundary complexity issue |

**Score:** 3/4 = 75%
**Cost:** $6.41 total

---

## Scripts Reference

| Script | Purpose |
|--------|---------|
| `visual_analyzer.py` | Render frames as emoji/images, compare frames |
| `sybil_agent.py` | Main agent orchestrator |
| `exploration_agent.py` | Systematic action discovery (for ARC-3 games) |
| `failure_memory.py` | Track failures and lessons |
| `pattern_db.py` | Store/retrieve successful patterns |
| `cost_tracker.py` | Track API spending |
| `score.py` | External validation against ground truth |

---

## Applying to ARC-AGI-3

### Key Differences:
- **ARC-2:** Static puzzles (input → output transformation)
- **ARC-3:** Interactive games (action sequences, real-time state)

### What Transfers:
1. ✅ Visual analysis (render game frames)
2. ✅ Meflex method (3 hypotheses for game mechanics)
3. ✅ Failure memory (don't repeat mistakes)
4. ✅ Pattern storage (save learned mechanics)

### What's New for ARC-3:
1. 🆕 **Exploration phase** — learn actions before trying to win
2. 🆕 **State tracking** — remember what changed between frames
3. 🆕 **Action efficiency** — scored against human baseline
4. 🆕 **Level progression** — multiple levels per game

### ARC-3 Process (Modified):
```
1. EXPLORE — Systematically test each action, learn effects
2. MAP — Identify objects, goals, obstacles
3. THEORIZE — 3 hypotheses for win condition
4. PLAN — Sequence of actions to reach goal
5. EXECUTE — Run the plan, observe results
6. LEARN — Save mechanics for this game type
```

---

## Daily Practice Checklist

- [ ] Read this document
- [ ] Pick a puzzle/game to practice
- [ ] Follow the 6-step process
- [ ] Update GAME_NOTES.md with discoveries
- [ ] Commit any new patterns to primitives/
- [ ] Log session in memory/YYYY-MM-DD.md

---

*This document is the master reference for ARC solving. Update after each session.*
