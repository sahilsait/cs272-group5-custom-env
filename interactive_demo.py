"""
Interactive Demo - Group 5 Custom Environment

Control the ego vehicle using keyboard in the rendering window.

Controls:
  LEFT ARROW  - Lane Left
  RIGHT ARROW - Lane Right
  UP ARROW    - Faster
  DOWN ARROW  - Slower
  SPACE       - Idle (maintain speed)
  R           - Reset episode
  ESC         - Exit

The environment will render in a window and respond to your keypresses in real-time.
"""

import gymnasium as gym
import numpy as np
from group5_custom_env import register_group5_env
import pygame
import time

# Register the environment
register_group5_env()

class InteractiveDemo:
    def __init__(self):
        self.env = gym.make('group5-env-v0', render_mode='human')
        self.running = True
        self.paused = False
        self.step_count = 0
        self.episode_reward = 0
        self.clock = pygame.time.Clock()
        self.font = None
        
    def init_display(self):
        """Initialize pygame display for text overlay."""
        pygame.init()
        try:
            self.font = pygame.font.Font(None, 24)
            self.small_font = pygame.font.Font(None, 20)
        except:
            self.font = None
            
    def get_action_from_keys(self):
        """Get action based on keyboard input."""
        keys = pygame.key.get_pressed()
        
        # Priority: Lane changes > Speed changes > Idle
        if keys[pygame.K_LEFT]:
            return 0, "LANE_LEFT"
        elif keys[pygame.K_RIGHT]:
            return 2, "LANE_RIGHT"
        elif keys[pygame.K_UP]:
            return 3, "FASTER"
        elif keys[pygame.K_DOWN]:
            return 4, "SLOWER"
        else:
            return 1, "IDLE"
    
    def render_overlay(self, screen):
        """Render text overlay with vehicle information."""
        # No overlay - just render the environment
        pass
    
    def run(self):
        """Main game loop."""
        self.init_display()
        
        obs, info = self.env.reset()
        self.step_count = 0
        self.episode_reward = 0
        
        print("=" * 80)
        print("🚗 Interactive Demo Started!")
        print("=" * 80)
        print("\n📋 CONTROLS:")
        print("  ← → : Change Lane Left/Right")
        print("  ↑ ↓ : Speed Up/Slow Down")
        print("  SPACE : Idle (maintain current speed)")
        print("  R : Reset episode")
        print("  ESC : Exit demo")
        print("\n📌 INSTRUCTIONS:")
        print("  - The rendering window will open shortly")
        print("  - Click on the window to make sure it has focus")
        print("  - Press any action key to take a step")
        print("  - The environment pauses between actions (wait for your input)")
        print("  - Watch for the emergency vehicle (purple) approaching from behind")
        print("  - Avoid lane closures (obstacles) and stalled vehicles")
        print("=" * 80)
        
        # Initial render
        self.env.render()
        try:
            screen = pygame.display.get_surface()
            if screen:
                self.render_overlay(screen)
                pygame.display.flip()
        except:
            pass
        
        while self.running:
            # Wait for action key press
            waiting_for_action = True
            action = 1  # Default to IDLE
            action_name = "IDLE"
            
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
                            self.episode_reward = 0
                            waiting_for_action = False
                            # Render after reset
                            self.env.render()
                            try:
                                screen = pygame.display.get_surface()
                                if screen:
                                    self.render_overlay(screen)
                                    pygame.display.flip()
                            except:
                                pass
                            waiting_for_action = True  # Continue waiting
                        elif event.key == pygame.K_LEFT:
                            action, action_name = 0, "LANE_LEFT"
                            waiting_for_action = False
                        elif event.key == pygame.K_RIGHT:
                            action, action_name = 2, "LANE_RIGHT"
                            waiting_for_action = False
                        elif event.key == pygame.K_UP:
                            action, action_name = 3, "FASTER"
                            waiting_for_action = False
                        elif event.key == pygame.K_DOWN:
                            action, action_name = 4, "SLOWER"
                            waiting_for_action = False
                        elif event.key == pygame.K_SPACE:
                            action, action_name = 1, "IDLE"
                            waiting_for_action = False
                
                # Keep rendering while waiting
                try:
                    screen = pygame.display.get_surface()
                    if screen:
                        self.render_overlay(screen)
                        pygame.display.flip()
                except:
                    pass
                
                self.clock.tick(30)  # 30 FPS while waiting
            
            if not self.running:
                break
            
            # Take step with the chosen action
            print(f"Action: {action_name}")
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.step_count += 1
            self.episode_reward += reward
            
            # Render
            self.env.render()
            
            # Add overlay
            try:
                screen = pygame.display.get_surface()
                if screen:
                    self.render_overlay(screen)
                    pygame.display.flip()
            except:
                pass
            
            # Check if episode ended
            if terminated or truncated:
                ego = self.env.unwrapped.vehicle
                
                print("\n" + "=" * 80)
                print("📊 EPISODE ENDED")
                print("=" * 80)
                
                if ego.crashed:
                    print("   Result: 💥 CRASHED")
                elif ego.position[0] >= self.env.unwrapped.config['road_length']:
                    print("   Result: 🎉 SUCCESS!")
                elif not ego.on_road:
                    print("   Result: 🚧 WENT OFF ROAD")
                else:
                    print("   Result: ⏱️ TIME LIMIT")
                
                print(f"   Steps: {self.step_count}")
                print(f"   Total Reward: {self.episode_reward:.3f}")
                print(f"   Final Position: {ego.position[0]:.1f}m")
                print(f"   Progress: {(ego.position[0] / self.env.unwrapped.config['road_length'] * 100):.1f}%")
                print("=" * 80)
                print("\nPress R to reset or ESC to exit")
                
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
                                self.episode_reward = 0
                                waiting = False
                                print("\n🔄 New episode started!")
                                # Render after reset
                                self.env.render()
                                try:
                                    screen = pygame.display.get_surface()
                                    if screen:
                                        self.render_overlay(screen)
                                        pygame.display.flip()
                                except:
                                    pass
                    
                    self.clock.tick(30)  # Limit to 30 FPS while waiting
        
        self.env.close()
        pygame.quit()
        print("\n✅ Demo completed!")

def main():
    """Run interactive demo."""
    try:
        demo = InteractiveDemo()
        demo.run()
    except KeyboardInterrupt:
        print("\n\n👋 Demo interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
