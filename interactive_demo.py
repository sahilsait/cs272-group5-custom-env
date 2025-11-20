"""
Interactive Demo for Group 5 Custom Environment

Control the vehicle manually with keyboard and see environment parameters in real-time.

Controls:
    Arrow Keys:
        ← LEFT  : Change lane left
        → RIGHT : Change lane right
        ↑ UP    : Accelerate
        ↓ DOWN  : Decelerate
    SPACE       : Do nothing (IDLE)
    Q or ESC    : Quit
    R           : Reset environment

Parameters are displayed on screen during gameplay.
"""

import gymnasium as gym
import pygame
import sys
from group5_custom_env import register_group5_env

# Initialize pygame for keyboard input
pygame.init()

# Register and create environment
register_group5_env()
env = gym.make('group5-env-v0', render_mode='human')

# Action mapping
ACTIONS = {
    pygame.K_LEFT: 0,    # LANE_LEFT
    pygame.K_SPACE: 1,   # IDLE
    pygame.K_RIGHT: 2,   # LANE_RIGHT
    pygame.K_UP: 3,      # FASTER
    pygame.K_DOWN: 4,    # SLOWER
}

ACTION_NAMES = {
    0: "LANE LEFT",
    1: "IDLE",
    2: "LANE RIGHT",
    3: "FASTER",
    4: "SLOWER"
}


def print_status(obs, reward, terminated, truncated, info, action, step, config):
    """Print current environment status with parameters."""
    ego = env.unwrapped.vehicle
    
    print("\n" + "="*80)
    print(f"STEP {step} | Action: {ACTION_NAMES.get(action, 'NONE')}")
    print("="*80)
    
    # Vehicle Status
    print("\n📊 EGO VEHICLE STATUS:")
    print(f"  Position:      {ego.position[0]:.1f} / {config['road_length']}m")
    print(f"  Speed:         {ego.speed:.1f} m/s (target: {config['reward_speed_range'][0]}-{config['reward_speed_range'][1]} m/s)")
    print(f"  Lane:          {ego.lane_index[2] if len(ego.lane_index) > 2 else 'N/A'} / {config['lanes_count']-1}")
    print(f"  Crashed:       {'❌ YES' if ego.crashed else '✅ NO'}")
    print(f"  On Road:       {'✅ YES' if ego.on_road else '❌ NO'}")
    
    # Time Status
    time_remaining = config['duration'] - env.unwrapped.time
    print(f"\n⏱️  TIME:")
    print(f"  Elapsed:       {env.unwrapped.time:.1f}s / {config['duration']}s")
    print(f"  Remaining:     {time_remaining:.1f}s")
    
    # Reward Breakdown
    print(f"\n🏆 REWARD: {reward:.3f}")
    if hasattr(env.unwrapped, '_last_rewards'):
        rewards = env.unwrapped._last_rewards
        print(f"  - Collision:   {rewards.get('collision_reward', 0):.3f} (weight: {config['collision_reward']})")
        print(f"  - Speed:       {rewards.get('high_speed_reward', 0):.3f} (weight: {config['high_speed_reward']})")
        print(f"  - Progress:    {rewards.get('progress_reward', 0):.3f} (weight: {config['progress_reward']})")
        print(f"  - Success:     {rewards.get('success_reward', 0):.3f} (weight: {config['success_reward']})")
        print(f"  - Lane Change: {rewards.get('lane_change_reward', 0):.3f} (weight: {config['lane_change_reward']})")
    
    # Hazards Info
    print(f"\n🚧 HAZARDS:")
    print(f"  Lane Closures:     {len(env.unwrapped.road.objects)} (size: {config['lane_closure_length']}m × {config['lane_closure_width']}m)")
    emergency_speed = config['ego_initial_speed'] + config['emergency_vehicle_speed_offset']
    print(f"  Emergency Vehicles: {config['emergency_vehicles_count']} purple @ {emergency_speed} m/s")
    print(f"  Traffic Vehicles:   {config['vehicles_count']} (yielding enabled)")
    
    # Count vehicles by type
    emergency_count = 0
    stalled_count = 0
    for vehicle in env.unwrapped.road.vehicles:
        if hasattr(vehicle, 'color') and vehicle.color == config['emergency_vehicle_color']:
            emergency_count += 1
        elif vehicle.speed == 0 and vehicle != ego:
            stalled_count += 1
    
    print(f"  Stalled Vehicles:   {stalled_count} (speed: {config['stalled_vehicle_speed']} m/s)")
    
    # Episode Status
    if terminated:
        if ego.crashed:
            print("\n❌ EPISODE TERMINATED: CRASHED")
        elif not ego.on_road:
            print("\n❌ EPISODE TERMINATED: OFF ROAD")
        elif ego.position[0] >= config['road_length']:
            print("\n✅ EPISODE TERMINATED: SUCCESS! Reached the end!")
    elif truncated:
        print("\n⏱️  EPISODE TRUNCATED: TIME LIMIT")
    
    print("\n" + "="*80)
    print("Controls: ←/→ Lane | ↑/↓ Speed | SPACE Idle | R Reset | Q Quit")
    print("="*80)


def main():
    """Main interactive loop."""
    obs, info = env.reset()
    config = env.unwrapped.config
    
    # Store rewards in environment for display
    env.unwrapped._last_rewards = {}
    
    action = 1  # Start with IDLE
    step = 0
    total_reward = 0
    
    # Print initial status
    print_status(obs, 0, False, False, info, action, step, config)
    
    running = True
    terminated = False
    truncated = False
    
    while running:
        # Wait for keyboard input
        action = 1  # Default to IDLE
        waiting = True
        
        while waiting and running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    waiting = False
                    break  # Exit event loop immediately
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                        running = False
                        waiting = False
                        break  # Exit event loop immediately
                    elif event.key == pygame.K_r:
                        # Reset environment
                        obs, info = env.reset()
                        env.unwrapped._last_rewards = {}
                        step = 0
                        total_reward = 0
                        terminated = False
                        truncated = False
                        print("\n🔄 ENVIRONMENT RESET\n")
                        print_status(obs, 0, False, False, info, 1, step, config)
                        waiting = False
                        break  # Exit event loop immediately
                    elif event.key in ACTIONS:
                        action = ACTIONS[event.key]
                        # Debug: Print which key was pressed
                        key_name = pygame.key.name(event.key)
                        print(f"🎮 Key pressed: {key_name} -> Action: {ACTION_NAMES[action]} ({action})")
                        waiting = False
                        break  # Exit event loop immediately after capturing action
            
            # Small delay to prevent CPU spinning
            pygame.time.wait(10)
        
        if not running:
            break
        
        # Don't step if episode is done
        if terminated or truncated:
            print("\n⚠️  Episode finished. Press R to reset or Q to quit.")
            continue
        
        # Store lane before action for debugging
        lane_before = env.unwrapped.vehicle.lane_index[2] if len(env.unwrapped.vehicle.lane_index) > 2 else 'N/A'
        
        # Take action
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Store lane after action for debugging
        lane_after = env.unwrapped.vehicle.lane_index[2] if len(env.unwrapped.vehicle.lane_index) > 2 else 'N/A'
        
        # Alert if lane changed on non-lane-change action
        if action not in [0, 2] and lane_before != lane_after:
            print(f"⚠️  WARNING: Lane changed from {lane_before} to {lane_after} on {ACTION_NAMES[action]} action!")
        
        # Store reward breakdown for display
        env.unwrapped._last_rewards = env.unwrapped._rewards(action)
        
        step += 1
        total_reward += reward
        
        # Render and print status
        env.render()
        print_status(obs, reward, terminated, truncated, info, action, step, config)
        print(f"\n📈 TOTAL REWARD: {total_reward:.3f}")
    
    env.close()
    pygame.quit()
    print("\n👋 Thanks for testing the environment!")


if __name__ == "__main__":
    main()
