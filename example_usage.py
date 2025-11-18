"""
Example usage of Group 5 Custom Environment

This script demonstrates how to:
1. Load the environment
2. Run episodes
3. View statistics
"""

import gymnasium as gym
from group5_custom_env import register_group5_env

# Register the environment
register_group5_env()

print("=" * 70)
print("Group 5 Custom Environment - Example Usage")
print("=" * 70)

# Create environment
env = gym.make('group5-env-v0')

print(f"\nEnvironment Details:")
print(f"  Observation space: {env.observation_space}")
print(f"  Action space: {env.action_space}")
print(f"  Actions: LANE_LEFT(0), IDLE(1), LANE_RIGHT(2), FASTER(3), SLOWER(4)")

# Run 3 episodes
num_episodes = 3
print(f"\nRunning {num_episodes} episodes with random actions...\n")

for episode in range(num_episodes):
    obs, info = env.reset()
    episode_reward = 0
    steps = 0

    print(f"Episode {episode + 1}:")

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
            elif ego.position[0] >= env.unwrapped.config["road_length"]:
                reason = "🎉 Success"
            elif not ego.on_road:
                reason = "🚧 Off-road"
            else:
                reason = "⏱️ Timeout"

            print(f"  {reason}")
            print(f"  Steps: {steps}")
            print(f"  Total reward: {episode_reward:.2f}")
            print(f"  Avg reward/step: {episode_reward/steps:.3f}")
            print()
            break

env.close()
print("✅ Done!")
