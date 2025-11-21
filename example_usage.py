"""
Example usage of Group 5 Custom Environment

This script demonstrates how to:
1. Load the custom environment
2. Run episodes with random actions
3. View statistics and environment features

Features demonstrated:
- 5-lane highway with realistic traffic
- Emergency vehicles (purple, fast, prioritized)
- Yielding traffic (moves aside for emergency vehicles)
- Lane closures (static obstacles)
- Stalled vehicles (stopped on road)
- Custom observation space (includes emergency vehicle detection)
"""

import gymnasium as gym
from group5_custom_env import register_group5_env

# Register the custom environment
register_group5_env()

print("=" * 70)
print("Group 5 Custom Environment - Example Usage")
print("=" * 70)

# Create environment
env = gym.make('group5-env-v0')

print(f"\nEnvironment Configuration:")
print(f"  Road: 5 lanes, 625 meters")
print(f"  Duration: 50 seconds")
print(f"  Traffic: 10 vehicles (yielding behavior)")
print(f"  Emergency vehicles: 1 (50% spawn probability)")
print(f"  Hazards: Lane closures + stalled vehicle")

print(f"\nEnvironment Details:")
print(f"  Observation space: {env.observation_space}")
print(f"  Action space: {env.action_space}")
print(f"  Actions: LANE_LEFT(0), IDLE(1), LANE_RIGHT(2), FASTER(3), SLOWER(4)")

# Run 3 episodes
num_episodes = 3
print(f"\nRunning {num_episodes} episodes with random actions...")
print("Watch for:")
print("  🟣 Purple vehicles = Emergency vehicles (fast, leftmost lane)")
print("  🚗 White vehicle = Agent (you)")
print("  🚧 Red/Orange obstacles = Lane closures")
print("  🛑 Stopped vehicles = Stalled vehicles")
print()

total_successes = 0
total_crashes = 0

for episode in range(num_episodes):
    obs, info = env.reset()
    episode_reward = 0
    steps = 0
    
    print(f"Episode {episode + 1}:")
    
    # Check if emergency vehicle spawned this episode
    emergency_spawned = any(
        hasattr(v, 'is_emergency') and v.is_emergency 
        for v in env.unwrapped.road.vehicles
    )
    if emergency_spawned:
        print(f"  🚨 Emergency vehicle active this episode!")
    else:
        print(f"  ✓ No emergency vehicle this episode")
    
    while True:
        # Random action
        action = env.action_space.sample()
        
        # Step
        obs, reward, terminated, truncated, info = env.step(action)
        
        episode_reward += reward
        steps += 1
        
        if terminated or truncated:
            # Get end reason
            ego = env.unwrapped.vehicle
            
            if ego.crashed:
                reason = "💥 Crashed"
                total_crashes += 1
            elif ego.position[0] >= env.unwrapped.config["road_length"]:
                reason = "🎉 Success"
                total_successes += 1
            elif not ego.on_road:
                reason = "🚧 Off-road"
                total_crashes += 1
            else:
                reason = "⏱️ Timeout"
            
            print(f"  Result: {reason}")
            print(f"  Steps: {steps}")
            print(f"  Total reward: {episode_reward:.2f}")
            print(f"  Avg reward/step: {episode_reward/steps:.3f}")
            print()
            break

env.close()

print("=" * 70)
print(f"Summary:")
print(f"  Successes: {total_successes}/{num_episodes} ({total_successes/num_episodes*100:.0f}%)")
print(f"  Crashes: {total_crashes}/{num_episodes} ({total_crashes/num_episodes*100:.0f}%)")
print("=" * 70)
print("\n✅ Done!")
