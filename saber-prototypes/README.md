# ARC-AGI-3 Challenge

**Goal:** Build a winning agent for the ARC-AGI-3 Interactive Reasoning Benchmark  
**Launch Date:** March 25, 2026  
**Lead:** Saber ⚔️ (collaborating with Sybil)

---

## Overview

ARC-AGI-3 is the first **interactive reasoning benchmark** — video-game-like environments where agents must:
- Explore to discover rules (no instructions given)
- Infer goals from environment feedback
- Execute efficiently (scored on action count vs. human baseline)

**Scale:** 1,000+ levels across 150+ hand-crafted environments  
**Key Metric:** Action efficiency — not just "did you win" but "how fast"

---

## Current State

### Preview Competition Results (3 games, 30 days)
| Rank | Team | Score | Approach |
|------|------|-------|----------|
| 1st | StochasticGoose (Tufa Labs) | 12.58% | CNN action prediction |
| 2nd | Blind Squirrel | 6.71% | State graph + value model |
| Human | — | ~100% | — |

The gap is massive. AI burns hundreds of actions just discovering what's clickable.

---

## Research Synthesis

### What Works (ARC-AGI-3 Preview)

1. **Action Prediction CNN** (StochasticGoose)
   - Predict which actions cause frame changes
   - Hierarchical: pick action type → then coordinates for ACTION6
   - Reset experience buffer between levels
   - ~7M parameter network

2. **State Graph Building** (Blind Squirrel)
   - Hash frames to detect loops/duplicates
   - Prune no-op actions
   - Back-propagate value when score improves
   - ResNet18-based value model

### What Works (ARC-AGI-1/2 — may transfer)

1. **Test-Time Training (TTT)** — Fine-tune on augmented puzzle variants
2. **AIRV** — Augment → Inference → Reverse-augment → Vote
3. **Refinement Loops** — Evolutionary program synthesis with verification
4. **Tiny Recursive Models** — 7M params, recursive self-improvement

---

## Proposed Approaches

### Approach A: World Model + Curiosity
Build a predictive model of state transitions. Use prediction error as intrinsic reward — explore where uncertainty is high.

**Components:**
- State encoder (CNN or ViT)
- Transition model (predicts next state given action)
- Curiosity signal = prediction error
- Policy trained on curiosity + sparse rewards

### Approach B: Goal Inference + Planning
Detect reward signals (score changes, level transitions). Once you know the goal, use planning.

**Components:**
- Change detector (frame differencing)
- Reward classifier (did score improve?)
- Sub-goal extractor
- MCTS or beam search for planning

### Approach C: LLM + RL Hybrid
Use LLM to hypothesize rules from observations, RL to verify and optimize.

**Components:**
- Vision encoder → text description
- LLM generates rule hypotheses
- RL policy tests hypotheses
- Feedback loop to refine rules

### Approach D: Meta-Learning
Train on many games to learn *how to learn games*.

**Components:**
- MAML or similar meta-learning
- Rapid adaptation at test time
- Transfer primitive discovery

---

## Project Structure

```
arc-agi-3/
├── ARC-AGI-3-Agents/     # Official agent framework (cloned)
├── research/             # Papers, notes, analysis
├── agents/               # Our custom agents
│   ├── curiosity_agent.py
│   ├── goal_inference_agent.py
│   └── hybrid_agent.py
├── experiments/          # Experiment logs and results
├── test_arc.py           # Basic toolkit test
└── README.md
```

---

## Key Papers

1. **StochasticGoose** — [Blog](https://medium.com/@dries.epos/1st-place-in-the-arc-agi-3-agent-preview-competition-49263f6287db)
2. **Tiny Recursive Models** — [arXiv:2510.04871](https://arxiv.org/abs/2510.04871)
3. **Evolutionary Program Synthesis** — [OpenReview](https://openreview.net/pdf?id=z4IG090qt2)
4. **ARC Prize 2025 Technical Report** — [arXiv:2601.10904](https://arxiv.org/html/2601.10904v1)
5. **On the Measure of Intelligence** — [arXiv:1911.01547](https://arxiv.org/abs/1911.01547)

---

## Experimental Findings

### Game Mechanics Analysis

| Game | Available Actions | Change Rate | Unique States (500 actions) |
|------|-------------------|-------------|----------------------------|
| ls20 | 1-4 (directional) | 1-2% | 4 |
| ft09 | 1-6 (mixed) | 1-3% | 7 |
| vc33 | 6 only (click) | 1% | 6 |

**Key insight:** 97-99% of random actions do nothing. Games have extremely sparse dynamics.

### Why Random Exploration Fails

1. Most (state, action) pairs are no-ops
2. State space is large but reachable states are sparse
3. Progress requires specific action sequences
4. No generalization across states without learning

### What Winning Requires

1. **CNN-based action prediction** — Learn which actions cause frame changes
2. **State hashing** — Detect loops and avoid revisiting
3. **Long training** — ~100k actions per game (8+ hours)
4. **Generalization** — Transfer patterns across similar states

---

## Prototype Agents

| Agent | Approach | Status |
|-------|----------|--------|
| `world_model_agent.py` | Curiosity via prediction error | ✅ Implemented |
| `goal_inference_agent.py` | Credit assignment + planning | ✅ Implemented |
| `llm_hybrid_agent.py` | Rule hypothesis + verification | ✅ Implemented |
| `action_predictor_agent.py` | Action change prediction | ✅ Implemented |

All agents achieve 0 levels in short runs — matching expectation that these games need ~14,000 actions per level.

---

## Next Steps

1. [x] Set up ARC-AGI-3 toolkit
2. [x] Clone official agents repo
3. [x] Analyze game mechanics (sparse dynamics confirmed)
4. [x] Implement 4 prototype agents
5. [x] Run diagnostic benchmarks
6. [ ] Implement CNN-based action predictor (like StochasticGoose)
7. [ ] Run extended training (10k+ actions)
8. [ ] Compare with Sybil's approach
9. [ ] Combine best ideas for final submission

---

## Collaboration Notes

- Sybil is working on this in parallel (ML/Research perspective)
- Compare approaches after initial prototypes
- Combine best ideas for final submission
