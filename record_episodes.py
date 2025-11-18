"""
Simple script to record a few episodes for visualization.
"""
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='gymnasium.envs.registration')

import gymnasium as gym
import highway_env
import os

# Configuration
NUM_EPISODES = 3  # Record 3 episodes
MAX_STEPS = 500   # Max steps per episode

# Create videos directory
os.makedirs('./videos', exist_ok=True)

print("=" * 70)
print("🎥 Recording Group 5 Environment Episodes")
print("=" * 70)
print(f"\nSettings:")
print(f"  Episodes to record: {NUM_EPISODES}")
print(f"  Max steps per episode: {MAX_STEPS}")
print(f"  Output folder: ./videos/")
print("\n" + "=" * 70)

# Create environment with video recording
env = gym.make('group5-env-v0', render_mode='rgb_array')
env = gym.wrappers.RecordVideo(
    env,
    video_folder='./videos',
    episode_trigger=lambda _: True,  # Record all episodes
    name_prefix='group5-episode',
    disable_logger=True
)

total_rewards = []
episode_lengths = []

for episode in range(NUM_EPISODES):
    print(f"\n📹 Recording Episode {episode + 1}/{NUM_EPISODES}...")

    obs, info = env.reset()
    episode_reward = 0
    steps = 0

    for step in range(MAX_STEPS):
        # Random action
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        episode_reward += reward
        steps += 1

        # Print progress
        if step % 50 == 0:
            print(f"  Step {step}/{MAX_STEPS} - Reward: {episode_reward:.2f}", end='\r')

        if terminated or truncated:
            # Determine end reason
            ego = env.unwrapped.vehicle
            if ego.crashed:
                end_reason = "💥 CRASHED"
            elif ego.position[0] >= env.unwrapped.config["road_length"]:
                end_reason = "🎉 SUCCESS"
            elif not ego.on_road:
                end_reason = "🚧 OFF-ROAD"
            else:
                end_reason = "⏱️ TIMEOUT"

            print(f"  Episode ended at step {steps}: {end_reason}")
            print(f"  Total reward: {episode_reward:.2f}")
            break

    total_rewards.append(episode_reward)
    episode_lengths.append(steps)

env.close()

# Summary
print("\n" + "=" * 70)
print("✅ Recording Complete!")
print("=" * 70)
print(f"\nEpisode Statistics:")
for i, (reward, length) in enumerate(zip(total_rewards, episode_lengths)):
    print(f"  Episode {i+1}: {length} steps, reward = {reward:.2f}")

print(f"\nAverage:")
print(f"  Reward: {sum(total_rewards)/len(total_rewards):.2f}")
print(f"  Length: {sum(episode_lengths)/len(episode_lengths):.1f} steps")

print(f"\n🎥 Videos saved to: ./videos/")
print(f"   Files: group5-episode-episode-0.mp4, group5-episode-episode-1.mp4, ...")
print("\nYou can watch these videos to see:")
print("  - How the agent navigates around hazards")
print("  - Where collisions occur")
print("  - Traffic behavior")
print("  - Lane closure and stalled vehicle positions")
