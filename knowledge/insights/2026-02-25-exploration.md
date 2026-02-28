# ARC-3 Insights — Exploration Phase

---
id: ARC3-INS-001
timestamp: 2026-02-25
insight: "ARC-3 games often have multiple visual elements, but only ONE is player-controlled. The controlled element may not be the most visually prominent."
evidence: ["ls20 — L-shape is prominent but color12 block is controlled"]
confidence: medium
applications: "Always test each action and track WHICH pixels change. The changing pixels reveal the player element."
tags: [exploration, mechanics]

---
id: ARC3-INS-002
timestamp: 2026-02-25
insight: "Action effects can be measured by pixel-diff between frames. Large pixel changes (50+) indicate movement. Small changes (1-5) indicate toggles or state changes."
evidence: ["ls20 — ACTION1/3/4 change 52 pixels (movement), ACTION2 changes 2 pixels (toggle?)"]
confidence: high
applications: "Use compare_frames() to quantify action effects. Categorize actions by magnitude of change."
tags: [exploration, mechanics]

---
id: ARC3-INS-003
timestamp: 2026-02-25
insight: "Human baseline action counts reveal puzzle complexity. Level 1 of ls20 needs 29 human actions — this suggests a non-trivial path, not just 'walk to goal'."
evidence: ["ls20 human baselines: 29, 41, 172, 49, 53, 62, 82 for levels 1-7"]
confidence: high
applications: "Check human baseline before attempting. If baseline is high, expect complex mechanics or multi-step solutions."
tags: [efficiency, win-conditions]

---
id: ARC3-INS-004
timestamp: 2026-02-25
insight: "Exploration budget is essential. Spending 50 actions to understand mechanics is worth it if it saves 100 random actions later."
evidence: ["ls20 — systematic action testing revealed movement patterns, random play would have been inefficient"]
confidence: medium
applications: "Allocate first N actions purely for exploration. Map action→effect before attempting to win."
tags: [exploration, efficiency]
