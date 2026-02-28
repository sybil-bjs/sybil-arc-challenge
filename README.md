# ARC-AGI-3 Challenge - BJS Labs

**Goal:** Build a winning agent for the ARC-AGI-3 Interactive Reasoning Benchmark  
**Launch Date:** March 25, 2026  
**Team:** Sybil (ML/Research) + Saber (Sales/Marketing)

---

## Project Structure

```
├── saber-prototypes/     # Saber's prototype agents
│   ├── README.md         # Research findings & approach details
│   ├── action_predictor_agent.py   # Action change prediction (closest to winner)
│   ├── world_model_agent.py        # Curiosity via prediction error
│   ├── goal_inference_agent.py     # Credit assignment + planning
│   ├── llm_hybrid_agent.py         # Rule hypothesis testing
│   ├── benchmark.py                # Multi-agent comparison
│   └── diagnose_game.py            # Game mechanics analysis
│
├── sybil-approach/       # Sybil's approach (TBD)
│
└── combined/             # Best ideas from both (TBD)
```

---

## Quick Start

```bash
# Install toolkit
pip install arc-agi

# Run diagnostics
python saber-prototypes/diagnose_game.py ls20

# Run benchmark
python saber-prototypes/benchmark.py quick ls20 1000
```

---

## Key Findings (Saber)

### Game Mechanics
- **97-99% of random actions do nothing** — extremely sparse dynamics
- Games require learning which (state, action) pairs cause transitions
- Winning approach (StochasticGoose) used CNN to predict action effects

### Prototype Approaches
1. **World Model + Curiosity** — explore where predictions fail
2. **Goal Inference** — credit assignment when levels complete
3. **LLM Hybrid** — generate rule hypotheses, test them
4. **Action Predictor** — learn which actions cause changes

See `saber-prototypes/README.md` for detailed analysis.

---

## Collaboration

- Compare approaches from both contributors
- Combine best ideas
- Iterate toward winning submission
