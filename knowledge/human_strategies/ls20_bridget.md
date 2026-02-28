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
