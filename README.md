# Group 5 Custom Highway Environment

A custom reinforcement learning environment built on top of [highway-env](https://github.com/Eleurent/highway-env) for autonomous vehicle navigation with hazards.

## 🚀 Quick Start

```bash
# Clone repository
git clone <your-repo-url>
cd <repo-name>

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
- **Time limit**: 40 seconds per episode

### Hazards (Randomized Each Episode)
1. **Lane Closure**: 20-meter obstacle placed in lanes 1, 2, or 3 at position 250-400m
2. **Stalled Vehicle**: Stationary vehicle (speed=0) in a different lane, either before (150-250m) or after (450-550m) the closure
3. **Traffic**: 20 AI-controlled vehicles using IDM/MOBIL models

### Goal
Navigate from start to end (625m) while:
- Avoiding collisions with obstacles, stalled vehicle, and traffic
- Maintaining reasonable speed
- Staying on the road
- Completing within 40 seconds

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

**Type**: Kinematics

Returns a (5, 5) array with information about nearby vehicles:
- `presence`: 1 if vehicle exists, 0 otherwise
- `x`: x-coordinate (normalized)
- `y`: y-coordinate (normalized)
- `vx`: x-velocity (normalized)
- `vy`: y-velocity (normalized)

The observation includes the ego vehicle and 4 nearest vehicles.

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
| `on_road_reward` | multiplier | 0 if off-road, 1 if on-road |

### Reward Calculation

```
reward = (collision * -1.5) + (speed * 0.3) + (progress * 0.4)
         + (success * 1.0) + (lane_change * -0.01)
reward *= on_road  # Zero if off-road
```

### Example Rewards
- **Good driving** (300m, 25 m/s, no collision): ~0.38
- **Crashed**: ~-1.3
- **Reached end successfully**: ~1.67

---

## 🏁 Termination Conditions

An episode terminates when:
1. **Collision**: Vehicle crashes into obstacle, stalled vehicle, or traffic
2. **Off-road**: Vehicle leaves the road
3. **Success**: Vehicle reaches 625m
4. **Timeout**: 40 seconds elapsed (truncation, not termination)

---

## ⚙️ Customization

You can modify the environment by changing config values:

```python
env = gym.make('group5-env-v0')

# Modify configuration
env.unwrapped.config.update({
    "collision_reward": -2.0,  # Harsher collision penalty
    "vehicles_count": 30,      # More traffic
    "duration": 50,            # Longer episodes
})

obs, info = env.reset()
```

### Available Config Parameters

```python
{
    "lanes_count": 5,              # Number of lanes
    "road_length": 625,            # Road length in meters
    "speed_limit": 25,             # Speed limit in m/s
    "duration": 40,                # Episode timeout in seconds
    "vehicles_count": 20,          # Number of traffic vehicles
    "vehicles_density": 1.0,       # Traffic density

    # Hazard positions
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
    "reward_speed_range": [20, 28],
}
```

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
