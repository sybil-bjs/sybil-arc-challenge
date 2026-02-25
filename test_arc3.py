import arc_agi
from arcengine import GameAction

arc = arc_agi.Arcade()
env = arc.make("ls20", render_mode="terminal")

# Check what step() returns
result = env.step(GameAction.ACTION1)
print(f"Step returns: {type(result)}")
print(f"Dir: {[x for x in dir(result) if not x.startswith('_')]}")

# Try accessing attributes
if hasattr(result, 'reward'):
    print(f"reward: {result.reward}")
if hasattr(result, 'done'):
    print(f"done: {result.done}")
if hasattr(result, 'terminated'):
    print(f"terminated: {result.terminated}")
if hasattr(result, 'truncated'):
    print(f"truncated: {result.truncated}")
if hasattr(result, 'state'):
    print(f"state: {result.state}")
if hasattr(result, 'info'):
    print(f"info: {result.info}")

print("\n--- Scorecard ---")
print(arc.get_scorecard())
