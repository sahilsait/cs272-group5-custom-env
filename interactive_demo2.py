"""
Interactive Demo 2 - Group 5 Custom Environment with Detailed Reward Breakdown

This script provides real-time feedback on:
- Action taken
- Vehicle state (speed, position, lane, crashed status)
- Total step reward
- Detailed breakdown of all reward components
- Cumulative episode reward

Controls:
  LEFT ARROW  - Lane Left
  RIGHT ARROW - Lane Right
  UP ARROW    - Faster
  DOWN ARROW  - Slower
  SPACE       - Idle (maintain speed)
  R           - Reset episode
  ESC         - Exit
"""

import gymnasium as gym
import numpy as np
from group5_custom_env import register_group5_env
import pygame
import time
from datetime import datetime

# Register the environment
register_group5_env()

# Setup reward logging
REWARD_LOG_FILE = "rewards.log"


def pretty_print_rewards(rewards_dict, reward_weights):
    """
    Pretty-print the reward breakdown for better readability.
    
    Args:
        rewards_dict: Dictionary of reward components (0 or 1 values)
        reward_weights: Dictionary of reward weights from config
    """
    if not rewards_dict:
        print("  No rewards info found.")
        return
    
    print("\n  📊 Reward Breakdown:")
    print("  " + "=" * 60)
    
    total = 0.0
    for component_name, component_value in rewards_dict.items():
        weight = reward_weights.get(component_name, 0.0)
        weighted_value = component_value * weight
        total += weighted_value
        
        # Format component name nicely
        display_name = component_name.replace('_', ' ').title()
        
        # Add marker for negative rewards
        marker = " <== PENALTY" if weighted_value < 0 else (" <== BONUS!" if weighted_value > 0.5 else "")
        
        # Display format: component (raw * weight = result)
        if component_value != 0.0 or weighted_value != 0.0:
            print(f"    {display_name:22}: {component_value:.2f} × {weight:6.2f} = {weighted_value:7.3f}{marker}")
    
    print("  " + "-" * 60)
    print(f"    {'TOTAL STEP REWARD':22}:                  {total:7.3f}")
    print("  " + "=" * 60)


class InteractiveDemo2:
    def __init__(self):
        self.env = gym.make('group5-env-v0', render_mode='human')
        self.running = True
        self.step_count = 0
        self.episode_reward = 0.0
        self.cumulative_rewards = {
            'collision_reward': 0.0,
            'high_speed_reward': 0.0,
            'progress_reward': 0.0,
            'success_reward': 0.0,
            'lane_change_reward': 0.0,
            'yielding_reward': 0.0,
        }
        self.clock = pygame.time.Clock()
        self.log_file = None
        self.episode_number = 0
        
        # Action names
        self.action_names = {
            0: "LANE_LEFT",
            1: "IDLE",
            2: "LANE_RIGHT",
            3: "FASTER",
            4: "SLOWER"
        }
    
    def init_log_file(self):
        """Initialize the rewards log file."""
        self.log_file = open(REWARD_LOG_FILE, 'w')
        self.log_file.write("=" * 100 + "\n")
        self.log_file.write("GROUP 5 ENVIRONMENT - REWARDS LOG\n")
        self.log_file.write(f"Session started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.log_file.write("=" * 100 + "\n\n")
        self.log_file.flush()
    
    def log_episode_start(self):
        """Log episode start information."""
        self.episode_number += 1
        self.log_file.write("\n" + "=" * 100 + "\n")
        self.log_file.write(f"EPISODE {self.episode_number} STARTED\n")
        self.log_file.write(f"Time: {datetime.now().strftime('%H:%M:%S')}\n")
        self.log_file.write("=" * 100 + "\n\n")
        self.log_file.flush()
    
    def log_step_rewards(self, step, action, rewards_dict, reward_weights, step_reward, ego):
        """Log detailed reward information for each step."""
        self.log_file.write(f"\n--- STEP {step} ---\n")
        self.log_file.write(f"Action: {self.action_names[action]}\n")
        self.log_file.write(f"Position: {ego.position[0]:.2f}m | Speed: {ego.speed:.2f} m/s | Lane: {ego.lane_index[2] if len(ego.lane_index) > 2 else 0}\n")
        self.log_file.write(f"\nReward Components:\n")
        
        for component_name, component_value in rewards_dict.items():
            weight = reward_weights.get(component_name, 0.0)
            weighted_value = component_value * weight
            display_name = component_name.replace('_', ' ').title()
            self.log_file.write(f"  {display_name:22}: {component_value:.3f} × {weight:7.3f} = {weighted_value:8.3f}\n")
        
        self.log_file.write(f"\nStep Reward: {step_reward:.3f}\n")
        self.log_file.write(f"Cumulative Episode Reward: {self.episode_reward:.3f}\n")
        self.log_file.write("-" * 80 + "\n")
        self.log_file.flush()
    
    def log_episode_end(self, ego, result_text):
        """Log episode summary."""
        self.log_file.write("\n" + "=" * 100 + "\n")
        self.log_file.write(f"EPISODE {self.episode_number} COMPLETED\n")
        self.log_file.write(f"Result: {result_text}\n")
        self.log_file.write(f"Total Steps: {self.step_count}\n")
        self.log_file.write(f"Total Reward: {self.episode_reward:.3f}\n")
        self.log_file.write(f"Average Reward/Step: {self.episode_reward/max(self.step_count, 1):.3f}\n")
        self.log_file.write(f"Final Position: {ego.position[0]:.1f}m / 625m ({ego.position[0]/625*100:.1f}%)\n")
        self.log_file.write(f"\nCumulative Rewards:\n")
        
        for component_name, cumulative_value in self.cumulative_rewards.items():
            display_name = component_name.replace('_', ' ').title()
            self.log_file.write(f"  {display_name:22}: {cumulative_value:8.3f}\n")
        
        self.log_file.write("=" * 100 + "\n\n")
        self.log_file.flush()
    
    def close_log_file(self):
        """Close the log file."""
        if self.log_file:
            self.log_file.write("\n" + "=" * 100 + "\n")
            self.log_file.write(f"Session ended: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            self.log_file.write("=" * 100 + "\n")
            self.log_file.close()
    
    def get_action_from_keys(self):
        """Get action based on keyboard input."""
        keys = pygame.key.get_pressed()
        
        # Priority: Lane changes > Speed changes > Idle
        if keys[pygame.K_LEFT]:
            return 0
        elif keys[pygame.K_RIGHT]:
            return 2
        elif keys[pygame.K_UP]:
            return 3
        elif keys[pygame.K_DOWN]:
            return 4
        else:
            return 1
    
    def print_episode_header(self):
        """Print episode start information."""
        print("\n" + "=" * 80)
        print("🎬 NEW EPISODE STARTED")
        print("=" * 80)
        print("\n📍 Initial Setup:")
        print("  • Ego vehicle: Lane 2 (middle), Position 0m, Speed 25 m/s")
        print("  • Emergency vehicle: Lane 2 (middle), Position -50m, Speed 27 m/s")
        print("  • Lane closure: 80m long, random lane (1-3), position 250-400m")
        print("  • Stalled vehicles: 2 vehicles (one before, one after closure)")
        print("  • Traffic: 20 AI vehicles")
        print("\n🎯 Goal: Reach 625m without crashing in 40 seconds")
        print("\n⌨️  Controls: ← → (lanes) | ↑ ↓ (speed) | SPACE (idle) | R (reset) | ESC (exit)")
        print("=" * 80)
        print("\nPress any action key to begin...\n")
    
    def print_step_info(self, action, ego, rewards_dict, step_reward):
        """Print information after each step."""
        # Get emergency vehicle info
        emergency_info = "Not found"
        for vehicle in self.env.unwrapped.road.vehicles:
            if hasattr(vehicle, 'color') and vehicle.color == (0.5, 0.0, 0.5):
                distance = vehicle.position[0] - ego.position[0]
                emergency_info = f"{distance:+.1f}m ({'AHEAD' if distance > 0 else 'BEHIND'})"
                break
        
        # Header
        print("\n" + "-" * 80)
        print(f"⏱️  STEP {self.step_count}")
        print("-" * 80)
        
        # Action and vehicle state
        print(f"\n🎮 Action: {self.action_names[action]:12}  |  ", end="")
        print(f"🏁 Position: {ego.position[0]:6.1f}m / 625m ({ego.position[0]/625*100:5.1f}%)")
        
        print(f"🚗 Lane: {ego.lane_index[2] if len(ego.lane_index) > 2 else 0}             |  ", end="")
        print(f"⚡ Speed: {ego.speed:5.1f} m/s ({ego.speed * 3.6:5.1f} km/h)")
        
        print(f"🚨 Emergency: {emergency_info:15}  |  ", end="")
        print(f"💥 Crashed: {'YES ❌' if ego.crashed else 'NO ✅'}")
        
        # Reward information
        print(f"\n💰 Step Reward: {step_reward:7.3f}  |  Total Episode Reward: {self.episode_reward:8.3f}")
        
        # Detailed reward breakdown
        reward_weights = {
            'collision_reward': self.env.unwrapped.config['collision_reward'],
            'high_speed_reward': self.env.unwrapped.config['high_speed_reward'],
            'progress_reward': self.env.unwrapped.config['progress_reward'],
            'success_reward': self.env.unwrapped.config['success_reward'],
            'lane_change_reward': self.env.unwrapped.config['lane_change_reward'],
            'yielding_reward': self.env.unwrapped.config['yielding_reward'],
        }
        
        pretty_print_rewards(rewards_dict, reward_weights)
    
    def print_episode_summary(self, ego):
        """Print episode summary when it ends."""
        print("\n" + "=" * 80)
        print("🏁 EPISODE COMPLETED")
        print("=" * 80)
        
        # Determine end reason
        if ego.crashed:
            result = "💥 CRASHED"
            emoji = "❌"
        elif ego.position[0] >= self.env.unwrapped.config['road_length']:
            result = "🎉 SUCCESS!"
            emoji = "✅"
        elif not ego.on_road:
            result = "🚧 WENT OFF ROAD"
            emoji = "⚠️"
        else:
            result = "⏱️ TIME LIMIT REACHED"
            emoji = "⏰"
        
        print(f"\n{emoji} Result: {result}")
        print(f"\n📈 Episode Statistics:")
        print(f"  • Total Steps: {self.step_count}")
        print(f"  • Total Reward: {self.episode_reward:.3f}")
        print(f"  • Average Reward/Step: {self.episode_reward/max(self.step_count, 1):.3f}")
        print(f"  • Final Position: {ego.position[0]:.1f}m / 625m")
        print(f"  • Progress: {(ego.position[0] / self.env.unwrapped.config['road_length'] * 100):.1f}%")
        print(f"  • Final Speed: {ego.speed:.1f} m/s")
        
        # Cumulative rewards breakdown
        print(f"\n💰 Cumulative Rewards Breakdown:")
        print("  " + "-" * 60)
        for component_name, cumulative_value in self.cumulative_rewards.items():
            display_name = component_name.replace('_', ' ').title()
            marker = " 🔴" if cumulative_value < 0 else (" 🟢" if cumulative_value > 0 else "")
            print(f"    {display_name:22}: {cumulative_value:8.3f}{marker}")
        print("  " + "-" * 60)
        print(f"    {'TOTAL':22}: {self.episode_reward:8.3f}")
        print("=" * 80)
        
        print("\n⌨️  Press R to reset or ESC to exit")
    
    def run(self):
        """Main game loop."""
        pygame.init()
        
        # Initialize log file
        self.init_log_file()
        print(f"\n📝 Logging rewards to: {REWARD_LOG_FILE}\n")
        
        obs, info = self.env.reset()
        self.step_count = 0
        self.episode_reward = 0.0
        self.cumulative_rewards = {key: 0.0 for key in self.cumulative_rewards.keys()}
        
        self.log_episode_start()
        self.print_episode_header()
        
        # Initial render
        self.env.render()
        
        while self.running:
            # Wait for action key press
            waiting_for_action = True
            action = 1  # Default to IDLE
            
            while waiting_for_action and self.running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.running = False
                        waiting_for_action = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            self.running = False
                            waiting_for_action = False
                        elif event.key == pygame.K_r:
                            print("\n🔄 Resetting episode...")
                            obs, info = self.env.reset()
                            self.step_count = 0
                            self.episode_reward = 0.0
                            self.cumulative_rewards = {key: 0.0 for key in self.cumulative_rewards.keys()}
                            self.env.render()
                            self.print_episode_header()
                            waiting_for_action = True
                        elif event.key == pygame.K_LEFT:
                            action = 0
                            waiting_for_action = False
                        elif event.key == pygame.K_RIGHT:
                            action = 2
                            waiting_for_action = False
                        elif event.key == pygame.K_UP:
                            action = 3
                            waiting_for_action = False
                        elif event.key == pygame.K_DOWN:
                            action = 4
                            waiting_for_action = False
                        elif event.key == pygame.K_SPACE:
                            action = 1
                            waiting_for_action = False
                
                self.clock.tick(30)  # 30 FPS while waiting
            
            if not self.running:
                break
            
            # Take step with the chosen action
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.step_count += 1
            self.episode_reward += reward
            
            # Get reward breakdown (use cached rewards from step)
            ego = self.env.unwrapped.vehicle
            rewards_dict = self.env.unwrapped._last_rewards
            
            # Update cumulative rewards
            reward_weights = {
                'collision_reward': self.env.unwrapped.config['collision_reward'],
                'high_speed_reward': self.env.unwrapped.config['high_speed_reward'],
                'progress_reward': self.env.unwrapped.config['progress_reward'],
                'success_reward': self.env.unwrapped.config['success_reward'],
                'lane_change_reward': self.env.unwrapped.config['lane_change_reward'],
                'yielding_reward': self.env.unwrapped.config['yielding_reward'],
            }
            
            for component_name, component_value in rewards_dict.items():
                weight = reward_weights.get(component_name, 0.0)
                self.cumulative_rewards[component_name] += component_value * weight
            
            # Log step rewards to file
            self.log_step_rewards(self.step_count, action, rewards_dict, reward_weights, reward, ego)
            
            # Print step information
            self.print_step_info(action, ego, rewards_dict, reward)
            
            # Render
            self.env.render()
            
            # Check if episode ended
            if terminated or truncated:
                # Determine result text for logging
                if ego.crashed:
                    result_text = "CRASHED"
                elif ego.position[0] >= self.env.unwrapped.config['road_length']:
                    result_text = "SUCCESS"
                elif not ego.on_road:
                    result_text = "OFF ROAD"
                else:
                    result_text = "TIME LIMIT"
                
                self.log_episode_end(ego, result_text)
                self.print_episode_summary(ego)
                
                # Wait for reset or exit
                waiting = True
                while waiting and self.running:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            self.running = False
                            waiting = False
                        elif event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_ESCAPE:
                                self.running = False
                                waiting = False
                            elif event.key == pygame.K_r:
                                obs, info = self.env.reset()
                                self.step_count = 0
                                self.episode_reward = 0.0
                                self.cumulative_rewards = {key: 0.0 for key in self.cumulative_rewards.keys()}
                                waiting = False
                                self.env.render()
                                self.log_episode_start()
                                self.print_episode_header()
                    
                    self.clock.tick(30)
        
        self.close_log_file()
        self.env.close()
        pygame.quit()
        print(f"\n✅ Demo completed! Rewards logged to: {REWARD_LOG_FILE}")
        print("Thank you for testing Group 5 Environment!\n")


def main():
    """Run interactive demo with detailed reward breakdown."""
    try:
        demo = InteractiveDemo2()
        demo.run()
    except KeyboardInterrupt:
        print("\n\n👋 Demo interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
