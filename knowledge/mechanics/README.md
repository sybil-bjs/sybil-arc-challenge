# Mechanics Registry

Persistent knowledge base of game mechanics discovered across ARC-AGI-3 games.

## How It Works

1. **Discovery**: Agent encounters new mechanic in a game
2. **Document**: Mechanic gets added to this registry with detection rules
3. **Future Games**: SemanticDiscovery probes for ALL known mechanics
4. **Meta-Learning**: Library grows, agent gets faster at understanding new games

## Mechanic File Format

Each `.yaml` file describes one mechanic:

```yaml
name: transformation_point
description: Element that changes player state when touched
first_seen: ls20
confidence: high

detection:
  visual:
    - rare_color_cluster (< 1% of pixels)
    - small_size (2-10 cells)
    - distinct_shape (cross, circle, etc.)
  behavioral:
    - disappears_on_touch OR respawns
    - causes_state_change

interaction:
  trigger: overlap OR adjacent
  effect: state_transformation
  consumable: true | false

examples:
  - game: ls20
    appearance: white cross (value 1)
    behavior: single-use, transforms state indicator
```

## Current Mechanics

| Mechanic | First Seen | Confidence |
|----------|------------|------------|
| transformation_point | ls20 | high |
| resource_pickup | ls20 | high |
| state_indicator | ls20 | high |
| goal_pattern | ls20 | high |
| move_budget | ls20 | high |

## Adding New Mechanics

When you discover something new:
1. Create `mechanic_name.yaml` in this folder
2. Document detection rules (visual + behavioral)
3. Add to `registry.yaml` index
4. SemanticDiscovery will auto-load it

---

*This registry is the agent's "institutional memory" for game mechanics.*
*It's how we get smarter over time.*
