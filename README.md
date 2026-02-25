# Sybil-ARC: Saturating ARC-AGI-2 with Integrated Agentic Intelligence

## Overview
This repository documents a state-of-the-art attempt to solve the [ARC-AGI-2](https://arcprize.org/) benchmark using a unique **Human-Agent Collaboration** model. 

While most solvers rely on stateless pipelines, **Sybil-ARC** leverages the **VULKN Architecture**: a persistent AI agent (Sybil) with direct access to a local Mac environment ("Mac Tools"), allowing for real-time code execution, visual verification, and cumulative learning.

**Principal Investigator:** Bridget (@bridget4g)
**Lead Researcher:** Sybil (Autonomous Agent via OpenClaw)

## The "Mac Tools" Advantage
Unlike standard LLM entries, Sybil utilizes a "System 2" thinking approach:
1. **Visual Discovery:** Using LVM capabilities to identify high-level strategies (symmetry, gravity, tiling).
2. **Technical Execution:** Writing bespoke Python scripts to verify the logic against training data.
3. **Iterative Refinement:** Self-correcting logic based on execution errors until 100% accuracy is achieved on training pairs.
4. **Logic Primitives:** Building a library of reusable geometric and logical functions (the "Hive Mind Library").

## Repository Structure
- `scripts/`: The orchestrator and sub-agent spawning logic.
- `primitives/`: A growing library of verified ARC logic functions.
- `logs/`: Full conversation and execution logs showing the "Thinking Process."
- `results/`: Live leaderboard and verified solutions for the 400-task evaluation set.

## Progress
- **Current Score:** [Calculating...]
- **Strategy:** Integrated Intelligence (Vision + Code + Persistence)

---
*This repository is maintained by Sybil under the direction of Bridget. Our goal is to prove that agents with "hands" and "memory" are the key to AGI.*
