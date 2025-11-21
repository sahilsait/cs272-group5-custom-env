# Group 5 Custom Highway Environment

A custom reinforcement learning environment built on top of [highway-env](https://github.com/Eleurent/highway-env) for autonomous vehicle navigation with **emergency vehicles, yielding traffic, and multiple hazards**.

## 🚀 Quick Start

```bash
# Clone repository
git clone https://github.com/sahilsait/cs272-group5-custom-env.git
cd cs272-group5-custom-env

# Install dependencies
pip install -r requirements.txt

# Run example
python example_usage.py

# Record demo videos
python record_episodes.py
```

---

## 🎯 Environment Features

### Road Layout
- **5-lane straight highway** (625 meters long)
- **Speed limit**: 25 m/s (~56 mph)
- **Time limit**: 50 seconds per episode

### 🚨 Emergency Vehicles
- **Bright purple** high-speed vehicles (35 m/s)
- Always seek the **leftmost (fast) lane**
- **50% spawn probability** per episode
- Agent must **detect and yield** to avoid blocking them

### 🚗 Intelligent Traffic
- **10 AI-controlled vehicles** using IDM/MOBIL models
- **Yielding behavior**: Traffic automatically moves right when emergency vehicles approach
- Realistic highway driving simulation

### 🚧 Hazards (Randomized Each Episode)
1. **Lane Closure**: 40-meter obstacle placed in lanes 1, 2, or 3 at position 250-400m
2. **Stalled Vehicle**: Stationary vehicle (speed=0) randomly positioned on the road
3. **Dynamic Traffic**: Vehicles that react to emergency vehicles

### 🎮 Custom Observation Space
- **5x5 kinematics matrix** includes:
  - Standard features: position, velocity
  - **`is_emergency` flag**: 1.0 for emergency vehicles, 0.0 for others
  - Allows agent to **detect and respond** to emergency vehicles

### Goal
Navigate from start to end (625m) while:
- ✅ Avoiding collisions with obstacles, stalled vehicles, and traffic
- ✅ **Yielding to emergency vehicles** (don't block them!)
- ✅ Maintaining reasonable speed (20-28 m/s)
- ✅ Staying on the road
- ✅ Completing within 50 seconds

---

## 📦 Installation

### Prerequisites
```bash
pip install gymnasium
pip install highway-env
pip install numpy
```

Or use the included requirements file:
```bash
pip install -r requirements.txt
```

---

## 🎮 Usage

### Basic Usage

```python
import gymnasium as gym
from group5_custom_env import register_group5_env

# Register the environment
register_group5_env()

# Create the environment
env = gym.make('group5-env-v0')

# Use it like any Gym environment
obs, info = env.reset()
for _ in range(1000):
    action = env.action_space.sample()  # Random action
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        obs, info = env.reset()

env.close()
```

### With Rendering

```python
import gymnasium as gym
from group5_custom_env import register_group5_env

register_group5_env()

# Create with rendering
env = gym.make('group5-env-v0', render_mode='human')

obs, info = env.reset()
for _ in range(1000):
    env.render()
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        obs, info = env.reset()

env.close()
```

### Recording Videos

Use the included `record_episodes.py` script:

```bash
python record_episodes.py
```

Or manually:

```python
import gymnasium as gym
from group5_custom_env import register_group5_env

register_group5_env()

env = gym.make('group5-env-v0', render_mode='rgb_array')
env = gym.wrappers.RecordVideo(
    env,
    video_folder='./videos',
    episode_trigger=lambda e: True,  # Record all episodes
    name_prefix='group5-episode'
)

# Run episodes - they'll be recorded automatically
obs, info = env.reset()
for _ in range(500):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        break

env.close()
# Videos saved to ./videos/
```

---

## 🎮 Action Space

**Type**: Discrete(5)

| Action | Description |
|--------|-------------|
| 0 | LANE_LEFT - Change to left lane |
| 1 | IDLE - Maintain current lane and speed |
| 2 | LANE_RIGHT - Change to right lane |
| 3 | FASTER - Increase target speed |
| 4 | SLOWER - Decrease target speed |

---

## 📊 Observation Space

**Type**: Kinematics (Enhanced)

Returns a **(5, 5)** array with information about nearby vehicles:
- `presence`: 1 if vehicle exists, 0 otherwise
- `x`: x-coordinate (normalized to [0, 625])
- `y`: y-coordinate (normalized to [-10, 20])
- `vx`: x-velocity (normalized to [0, 40])
- `vy`: y-velocity
- **`is_emergency`**: 🚨 **1.0 for emergency vehicles, 0.0 for others**

The observation includes the ego vehicle and 4 nearest vehicles.

### Using Emergency Vehicle Detection

```python
obs, info = env.reset()
# obs.shape = (5, 5)
# obs[:, 4] contains is_emergency flags for each observed vehicle
emergency_nearby = any(obs[:, 4] > 0.5)
if emergency_nearby:
    print("Emergency vehicle detected! Should yield!")
```

---

## 🎁 Reward Function

The reward is a weighted sum of multiple components:

### Reward Components

| Component | Weight | Description |
|-----------|--------|-------------|
| `collision_reward` | -1.5 | Heavy penalty for crashes |
| `high_speed_reward` | 0.3 | Reward for maintaining 20-28 m/s |
| `progress_reward` | 0.4 | Incremental reward for moving forward |
| `success_reward` | 1.0 | Bonus for reaching the end |
| `lane_change_reward` | -0.01 | Small penalty to discourage weaving |
| **`yielding_reward`** | **0.5** | **Reward for yielding to emergency vehicles** |
| `on_road_reward` | multiplier | 0 if off-road, 1 if on-road |

### Yielding Reward Details
- **Blocking emergency vehicle** (same lane, within 60m): **-0.25** penalty
- **Yielding properly** (emergency vehicle nearby but not blocked): **+0.05** reward

### Reward Calculation

```
reward = (collision * -1.5) + (speed * 0.3) + (progress * 0.4)
         + (success * 1.0) + (lane_change * -0.01) + (yielding * 0.5)
reward *= on_road  # Zero if off-road
```

### Example Rewards
- **Good driving** (300m, 25 m/s, yielding, no collision): ~0.45
- **Blocking emergency vehicle**: ~-0.25 per step
- **Crashed**: ~-1.5
- **Reached end successfully**: ~1.8

---

## 🏁 Termination Conditions

An episode terminates when:
1. **Collision**: Vehicle crashes into obstacle, stalled vehicle, or traffic
2. **Off-road**: Vehicle leaves the road
3. **Success**: Vehicle reaches 625m
4. **Timeout**: 50 seconds elapsed (truncation, not termination)

---

## ⚙️ Customization

You can modify the environment by changing config values:

```python
env = gym.make('group5-env-v0')

# Modify configuration
env.unwrapped.config.update({
    "collision_reward": -2.0,         # Harsher collision penalty
    "vehicles_count": 15,             # More/less traffic
    "duration": 60,                   # Longer episodes
    "emergency_spawn_probability": 0.8,  # More frequent emergency vehicles
})

obs, info = env.reset()
```

### Available Config Parameters

```python
{
    # Road configuration
    "lanes_count": 5,              # Number of lanes
    "road_length": 625,            # Road length in meters
    "speed_limit": 25,             # Speed limit in m/s
    "duration": 50,                # Episode timeout in seconds
    "vehicles_count": 10,          # Number of traffic vehicles
    "vehicles_density": 1.0,       # Traffic density

    # Emergency vehicle configuration
    "emergency_vehicles_count": 1,        # Number of emergency vehicles
    "emergency_vehicle_speed": 35.0,      # Target speed (m/s)
    "emergency_spawn_range": [100, 500],  # Where they can spawn
    "emergency_spawn_probability": 0.5,   # Chance to spawn (0.0-1.0)

    # Hazard positions
    "lane_closure_length": 40,
    "closure_lane_range": [1, 3],
    "closure_position_range": [250, 400],
    "stalled_position_before_range": [150, 250],
    "stalled_position_after_range": [450, 550],

    # Reward weights
    "collision_reward": -1.5,
    "high_speed_reward": 0.3,
    "progress_reward": 0.4,
    "success_reward": 1.0,
    "lane_change_reward": -0.01,
    "yielding_reward": 0.5,           # Yielding to emergency vehicles
    "reward_speed_range": [20, 28],
}
```

---

## 🎨 Visual Guide

When viewing the environment (with rendering):
- 🟣 **Purple vehicles** = Emergency vehicles (fast, aggressive)
- 🚗 **White vehicle** = Agent (you)
- 🚙 **Blue/Green vehicles** = Traffic (yields to emergency vehicles)
- 🚧 **Red/Orange obstacles** = Lane closures
- 🛑 **Stopped vehicles** = Stalled vehicles

---

## 📁 Repository Structure

```
.
├── README.md                  # This file
├── group5_custom_env.py       # Environment implementation
├── example_usage.py           # Basic usage example
├── record_episodes.py         # Video recording script
├── requirements.txt           # Python dependencies
└── .gitignore                 # Git ignore rules
```

---

## 🤖 Training an Agent

Use [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) for training:

```python
from stable_baselines3 import PPO
from group5_custom_env import register_group5_env
import gymnasium as gym

register_group5_env()

# Create environment
env = gym.make('group5-env-v0')

# Create PPO agent
model = PPO("MlpPolicy", env, verbose=1)

# Train for 100,000 steps
model.learn(total_timesteps=100_000)

# Save model
model.save("highway_agent")

# Test trained agent
obs, info = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
```

---

## 🐛 Troubleshooting

### "No module named 'highway_env'"
```bash
pip install highway-env
```

### "AttributeError: 'OrderEnforcing' object has no attribute 'vehicle'"
Use `env.unwrapped.vehicle` instead of `env.vehicle`

### Videos not recording
Make sure you have:
```bash
pip install opencv-python
# or
pip install moviepy
```

### Emergency vehicles going off-road
This shouldn't happen - the environment has explicit boundary checks. If you see this, please report it!

---

## 🌟 Key Features

1. ✅ **Emergency Vehicle System** - Realistic priority vehicle behavior
2. ✅ **Intelligent Traffic** - Vehicles yield to emergency vehicles
3. ✅ **Custom Observations** - Emergency vehicle detection in observation space
4. ✅ **Balanced Rewards** - Encourages yielding and safe navigation
5. ✅ **Configurable Difficulty** - Adjust traffic, hazards, and spawn rates
6. ✅ **Clean Code** - Human-readable variable names and clear documentation

---

## 👥 Team

Group 5 - CS272 Fall 2025

---

## 📄 License

This is a custom environment built for educational purposes using highway-env.

---

## 🔗 Resources

- [highway-env documentation](https://highway-env.readthedocs.io/)
- [Gymnasium documentation](https://gymnasium.farama.org/)
- [Original highway-env](https://github.com/Eleurent/highway-env)
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) (for training RL agents)
