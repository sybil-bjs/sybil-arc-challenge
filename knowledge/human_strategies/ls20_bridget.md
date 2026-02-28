# ls20 Human Strategy — Bridget (Ground Truth)

**Source:** Direct explanation from Bridget, 2026-02-28
**Confidence:** HIGH (human demonstration)

## Game Elements Identified

1. **Controllable box** — One of two boxes, moves with controller input
2. **Target box** — The other box (top), has a pattern
3. **Plus sign** — Action point / power-up location
4. **Yellow bar (bottom)** — Move counter / life bar (decreases per action)
5. **Red indicators** — Part of life bar display

## Human Intuitions Applied

### 1. Life Bar Recognition
> "This little yellow bar with the little red things looks to me a lot like a video game bar, the amount of life I have left."

- Yellow bar = limited moves
- Every action decreases it
- **Implication:** Budget moves, don't explore wastefully

### 2. Plus Sign Semantics  
> "As a human, I realized that plus signs usually mean you're going to do something."

- Plus signs are universal "action points" in UI
- Prioritize moving toward plus sign over unknown elements

### 3. Pattern Matching Instinct
> "When I see two things that match, as a human I want to put them together."

- Matching patterns = completion goal
- When box changes to match target, bring them together

## Winning Strategy

```
1. Identify controllable element (test with actions)
2. Notice resource constraint (yellow bar)
3. Move toward plus sign (known action point)
4. Step on plus sign → observe pattern change
5. Pattern now matches top box → move to combine
6. WIN
```

## Key Mechanics Discovered

| Element | Function |
|---------|----------|
| Plus sign | Transforms controlled box pattern to match target |
| Yellow bar | Move counter (budget constraint) |
| Pattern match | Win condition trigger when boxes combine |

## Implications for Agent

1. **Don't explore randomly** — moves are limited
2. **Prioritize plus signs** — they're action points
3. **Watch for pattern changes** — signals next objective
4. **When patterns match** — combine the elements

---

*This is human-transferred knowledge, not learned from exploration.*
*Confidence: Maximum (direct human instruction)*

---

## Level 2+ Mechanics (Additional Human Insights)

### Yellow Boxes = Life/Move Pickups

**Observation:** Small yellow boxes appear on the field

**Human Reasoning Chain:**
1. **Color matching:** Yellow boxes match yellow life bar → probably related
2. **Resource calculation:** "I don't have enough moves to reach plus sign AND return to target"
3. **Game design logic:** "The puzzle must be solvable → there must be extra resources"
4. **Hypothesis:** Yellow boxes = life/move pickups
5. **Verification:** Step on yellow box → life bar increases ✅

### Key Insight: Forward Resource Planning

> "When I try completing without extra life, I realize I don't have enough. So there MUST be another resource given to me."

This is **constraint-based reasoning:**
- Observe: Not enough moves for the obvious path
- Deduce: Designer must have provided a solution
- Search: Look for resource pickups (color-coded yellow)
- Execute: Collect pickups, then complete puzzle

### Updated Strategy for Later Levels

```
1. Assess move budget vs required distance
2. If insufficient → look for yellow pickups
3. Plan route: pickups → plus sign → combine
4. Execute efficiently
```

### Mechanics Summary

| Element | Color | Function |
|---------|-------|----------|
| Life bar | Yellow | Current move budget |
| Pickups | Yellow | +moves when collected |
| Plus sign | ? | Transforms pattern |
| Target box | ? | Combine with transformed box to win |

---

*Human knowledge continues to accumulate. Each level teaches more.*

---

## Level 3+ Mechanics: Path Planning & State Preservation

### Plus Button is Single-Use
> "I cannot step over that X button again or I'll change my state into a state I can't use"

**Key constraint:** Once transformed, avoid re-triggering the plus button
- Plan route carefully
- Don't backtrack over action points

### Efficient Pathing
- Yellow pickups should be collected on the way, not detoured to
- Minimize total moves (life bar constraint)
- Plan: pickups → plus → target (without crossing plus again)

---

## Level 4+ Mechanics: Shape AND Color

### Two-Step Transformation

**Observation:** Plus button gives right SHAPE but wrong COLOR

**New Element: Rainbow Block**
> "Something that's rainbow colored generally tells me it might be a color changer"

- Rainbow = multiple colors = color transformation
- Human intuition: rainbow symbolizes color variety/change

### Full Process for Later Levels

```
1. Assess: Do I have enough moves?
2. Collect: Yellow pickups if needed
3. Shape: Go to plus button (get correct shape)
4. Color: Go to rainbow block (get correct color)  
5. Combine: Bring transformed box to target
6. WIN
```

### Element Summary (Complete)

| Element | Visual | Function |
|---------|--------|----------|
| Life bar | Yellow bar (bottom) | Move budget |
| Pickups | Yellow boxes | +moves |
| Plus button | + symbol | Shape transformation (single-use!) |
| Rainbow block | Rainbow colors | Color transformation |
| Target | Patterned box | Match shape+color, then combine |

---

## Meta-Insight: Progressive Complexity

The game teaches mechanics progressively:
- Level 1: Basic movement + transformation
- Level 2: Resource management (pickups)  
- Level 3: Path planning (avoid re-triggers)
- Level 4: Multi-step transformation (shape + color)

**Human learning pattern:** Each level adds ONE new concept. Build on prior knowledge.

*This mirrors how humans learn — scaffolded complexity.*

---

## GROUNDED VISUAL ELEMENTS (Sybil + Bridget, 2026-02-28)

### Visual Element Identification

| Element | Visual Description | Location |
|---------|-------------------|----------|
| Player token | Composite block (orange top + blue bottom) | Movable on grid |
| State indicator | Shows current player form | Bottom-left HUD |
| Goal pattern | Pattern in dark square | Static on grid |
| Plus sign | **WHITE CROSS** | Transformation trigger |
| Power-ups | Yellow squares with dark centers | Collectible |

### CRITICAL MECHANIC CORRECTION

**Plus sign is MULTI-USE, not single-use!**

- Each hover over plus sign = 1 appearance change
- Keep hovering until state indicator MATCHES goal pattern
- THEN navigate to overlap goal

### Optimal Algorithm

```
1. Detect goal pattern (dark square area)
2. Detect current state (bottom-left HUD)
3. Calculate: N = transformations_needed(state, goal)
4. Navigate to plus sign (white cross)
5. Hover N times (each hover = 1 transformation)
6. Verify: state == goal
7. Navigate to goal, overlap
8. WIN
```

### Visual Detection Targets

For automated detection:
- **Plus sign:** Scan for white pixels in + shape
- **Goal pattern:** Dark square region, extract pattern
- **State indicator:** Bottom-left HUD, extract current pattern  
- **Player:** Orange+blue composite block (unique)

---

## ANNOTATED SCREENSHOT ANALYSIS (Bridget, 2026-02-28 13:48 EST)

### Level 2 / 7 — Verified Element Positions

| Element | Description | Detection Notes |
|---------|-------------|-----------------|
| **Power up token** | Yellow box with black center | Top-left area of playable field |
| **Goal** | Blue L-shape in black-bordered box | "Place to put your final token" |
| **Player token** | Orange top + Blue bottom composite | Movable via arrow keys |
| **Plus Sign** | Small white cross | Right side of field |
| **Player state** | Blue L-shape indicator | Bottom-left corner HUD |
| **Yellow bar** | Move budget | Bottom of screen |
| **Red indicators** | End of bar | Danger zone / penalty |

### WIN CONDITION (Confirmed)

```
IF state_indicator MATCHES goal_pattern:
    → Navigate player to goal box
    → Overlap triggers WIN
ELSE:
    → Navigate to plus sign
    → Transform state
    → Check again
```

### KEY OPTIMIZATION

In the screenshot, state indicator (blue L) **already matches** goal (blue L).
Agent should CHECK FOR MATCH FIRST before assuming transformation is needed!

### Controls Confirmed
- Arrow keys: Movement
- Space bar: Action?
- Click: Point interaction
- Undo (Z): Undo last action
- RESET LEVEL: Full restart

### Agent Logic Update

```python
def plan_level(frame):
    state = detect_state_indicator(frame)
    goal = detect_goal_pattern(frame)
    
    if patterns_match(state, goal):
        # Skip plus sign entirely!
        return plan_path_to_goal()
    else:
        return plan_path_to_plus() + transform() + plan_path_to_goal()
```

---

*Visual ground truth from human annotation. Highest confidence.*
