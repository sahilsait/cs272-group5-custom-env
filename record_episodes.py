"""
Record video episodes of the Group 5 Custom Environment.

This script records episodes with random actions and saves them as MP4 videos.
Useful for:
- Visualizing environment features
- Debugging environment behavior
- Creating demonstrations

The videos will show:
- Emergency vehicles (purple, fast, leftmost lane)
- Yielding traffic (moves aside for emergency vehicles)
- Lane closures and stalled vehicles
- Agent navigation
"""

import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='gymnasium.envs.registration')

import gymnasium as gym
from group5_custom_env import register_group5_env
import os

# Register custom environment
register_group5_env()

# Configuration
NUM_EPISODES = 3  # Record 3 episodes
MAX_STEPS = 500   # Max steps per episode

# Create videos directory
os.makedirs('./videos', exist_ok=True)

print("=" * 70)
print("🎥 Recording Group 5 Custom Environment Episodes")
print("=" * 70)
print(f"\nSettings:")
print(f"  Episodes to record: {NUM_EPISODES}")
print(f"  Max steps per episode: {MAX_STEPS}")
print(f"  Output folder: ./videos/")
print(f"\nEnvironment Features:")
print(f"  🟣 Purple vehicles = Emergency vehicles")
print(f"  🚗 White vehicle = Agent")
print(f"  🚧 Red/Orange obstacles = Lane closures")
print(f"  🛑 Stopped vehicles = Stalled vehicles")
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
successes = 0
crashes = 0

for episode in range(NUM_EPISODES):
    print(f"\n📹 Recording Episode {episode + 1}/{NUM_EPISODES}...")
    
    obs, info = env.reset()
    episode_reward = 0
    steps = 0
    
    # Check if emergency vehicle spawned
    emergency_spawned = any(
        hasattr(v, 'is_emergency') and v.is_emergency 
        for v in env.unwrapped.road.vehicles
    )
    if emergency_spawned:
        print(f"   🚨 Emergency vehicle active!")
    else:
        print(f"   ✓ No emergency vehicle")
    
    for step in range(MAX_STEPS):
        # Random action
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        episode_reward += reward
        steps += 1
        
        # Print progress
        if step % 50 == 0:
            print(f"   Step {step}/{MAX_STEPS} - Reward: {episode_reward:.2f}", end='\r')
        
        if terminated or truncated:
            # Determine end reason
            ego = env.unwrapped.vehicle
            if ego.crashed:
                end_reason = "💥 CRASHED"
                crashes += 1
            elif ego.position[0] >= env.unwrapped.config["road_length"]:
                end_reason = "🎉 SUCCESS"
                successes += 1
            elif not ego.on_road:
                end_reason = "🚧 OFF-ROAD"
                crashes += 1
            else:
                end_reason = "⏱️ TIMEOUT"
            
            print(f"   Episode ended at step {steps}: {end_reason}")
            print(f"   Total reward: {episode_reward:.2f}")
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

print(f"\nAverages:")
print(f"  Reward: {sum(total_rewards)/len(total_rewards):.2f}")
print(f"  Length: {sum(episode_lengths)/len(episode_lengths):.1f} steps")
print(f"  Success rate: {successes}/{NUM_EPISODES} ({successes/NUM_EPISODES*100:.0f}%)")

print(f"\n🎥 Videos saved to: ./videos/")
print(f"   Files: group5-episode-episode-0.mp4, group5-episode-episode-1.mp4, ...")
print("\nWatch these videos to see:")
print("  ✓ Emergency vehicles seeking the leftmost lane")
print("  ✓ Traffic yielding to emergency vehicles")
print("  ✓ Agent navigating around hazards")
print("  ✓ Lane closures and stalled vehicle positions")
print("  ✓ Where crashes and successes occur")
