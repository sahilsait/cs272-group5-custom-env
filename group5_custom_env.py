"""
Group 5 Custom Highway Environment

A custom highway-env environment featuring:
- 5-lane straight highway
- Randomized lane closures (40m obstacles)
- Randomized stalled vehicles
- Traffic (20 AI vehicles)
- Custom reward function for hazard navigation

Usage:
    import gymnasium as gym
    from group5_custom_env import register_group5_env

    register_group5_env()
    env = gym.make('group5-env-v0')
"""

import numpy as np
import gymnasium as gym

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.objects import Obstacle
from highway_env.vehicle.kinematics import Vehicle
from highway_env.vehicle.behavior import IDMVehicle

Observation = np.ndarray


class LaneClosureObstacle(Obstacle):
    """
    An obstacle representing a lane closure on the highway.
    """
    LENGTH = 20.0  # 20 meters long
    WIDTH = 4.0    # Full lane width


class Group5Env(AbstractEnv):
    """
    A custom highway driving environment with lane closures and stalled vehicles.

    The ego-vehicle must navigate a 5-lane highway with:
    - A lane closure (40m obstacle) in lanes 1-3
    - A stalled vehicle (speed=0) in a different lane
    - 20 traffic vehicles using IDM/MOBIL models

    Goal: Reach the end (625m) within 40 seconds without crashing.

    Reward Function:
        - Collision: -1.5 (heavy penalty)
        - Speed: 0.3 (target 20-28 m/s)
        - Progress: 0.4 (forward movement)
        - Success: 1.0 (reaching the end)
        - Lane change: -0.01 (discourage weaving)
    """

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                "observation": {"type": "Kinematics"},
                "action": {"type": "DiscreteMetaAction"},
                "lanes_count": 5,
                "road_length": 625,  # meters
                "speed_limit": 25,   # m/s
                "duration": 40,      # seconds
                "vehicles_count": 20,
                "vehicles_density": 1.0,

                # Lane closure config
                "lane_closure_length": 40,
                "closure_lane_range": [1, 3],  # lanes that can be closed
                "closure_position_range": [250, 400],  # mid-road

                # Stalled vehicle config
                "stalled_position_before_range": [150, 250],
                "stalled_position_after_range": [450, 550],

                # Reward weights - tuned for hazard navigation
                "collision_reward": -1.5,      # Heavy penalty for crashing
                "high_speed_reward": 0.3,      # Moderate reward for speed
                "progress_reward": 0.4,        # Good reward for moving forward
                "success_reward": 1.0,         # Big bonus for completion
                "lane_change_reward": -0.01,   # Small penalty for lane changes
                "yielding_reward": 0.0,        # For future emergency vehicle

                # Reward settings
                "normalize_reward": False,
                "reward_speed_range": [20, 28],  # Target speed range (m/s)
            }
        )
        return config

    def _reset(self) -> None:
        self._create_road()
        self._create_vehicles()
        self._spawn_custom_elements()

    def _spawn_custom_elements(self):
        """Spawn lane closure and stalled vehicle with random positions."""

        # Lane Closure
        closure_lane_min, closure_lane_max = self.config["closure_lane_range"]
        closure_lane = self.np_random.integers(closure_lane_min, closure_lane_max + 1)
        closure_pos_min, closure_pos_max = self.config["closure_position_range"]
        closure_position = self.np_random.uniform(closure_pos_min, closure_pos_max)

        lane_closure = LaneClosureObstacle.make_on_lane(
            road=self.road,
            lane_index=("0", "1", closure_lane),
            longitudinal=closure_position
        )
        self.road.objects.append(lane_closure)

        # Stalled Vehicle
        lanes = list(range(self.config["lanes_count"]))
        available_lanes = [lane for lane in lanes if lane != closure_lane]
        stalled_lane = self.np_random.choice(available_lanes)

        if self.np_random.random() < 0.5:
            pos_min, pos_max = self.config["stalled_position_before_range"]
        else:
            pos_min, pos_max = self.config["stalled_position_after_range"]

        stalled_position = self.np_random.uniform(pos_min, pos_max)
        stalled_vehicle = Vehicle.make_on_lane(
            self.road,
            lane_index=("0", "1", stalled_lane),
            longitudinal=stalled_position,
            speed=0.0
        )
        self.road.vehicles.append(stalled_vehicle)

    def _create_road(self) -> None:
        """Create a straight 5-lane highway."""
        self.road = Road(
            network=RoadNetwork.straight_road_network(
                self.config["lanes_count"],
                speed_limit=self.config["speed_limit"],
                length=self.config["road_length"]
            ),
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
        )

    def _create_vehicles(self) -> None:
        """Create ego vehicle and traffic."""
        # Ego vehicle
        ego_vehicle = Vehicle.create_random(self.road, speed=25.0, spacing=2.0)
        ego_vehicle = self.action_type.vehicle_class(
            self.road, ego_vehicle.position, ego_vehicle.heading, ego_vehicle.speed
        )
        self.controlled_vehicles = [ego_vehicle]
        self.road.vehicles.append(ego_vehicle)

        # Traffic vehicles
        vehicles_count = self.config["vehicles_count"]
        for _ in range(vehicles_count):
            traffic_vehicle = IDMVehicle.create_random(
                self.road,
                spacing=1 / self.config["vehicles_density"]
            )
            traffic_vehicle.randomize_behavior()
            self.road.vehicles.append(traffic_vehicle)

    def _rewards(self, action: Action) -> dict[str, float]:
        """
        Calculate individual reward components.
        All components are normalized to reasonable ranges.

        Returns:
            Dictionary of reward components with their values
        """
        ego = self.vehicle

        # === 1. COLLISION PENALTY ===
        collision = float(ego.crashed)

        # === 2. SPEED REWARD ===
        forward_speed = ego.speed * np.cos(ego.heading)
        scaled_speed = utils.lmap(
            forward_speed,
            self.config["reward_speed_range"],
            [0, 1]
        )
        high_speed = np.clip(scaled_speed, 0, 1)

        # === 3. PROGRESS REWARD ===
        progress = ego.position[0] / self.config["road_length"]

        # === 4. SUCCESS BONUS ===
        reached_end = (
            ego.position[0] >= self.config["road_length"]
            and not ego.crashed
        )
        success = float(reached_end)

        # === 5. ON-ROAD CHECK ===
        on_road = float(ego.on_road)

        # === 6. LANE CHANGE PENALTY ===
        lane_change = float(action in [0, 2])

        # === 7. YIELDING REWARD ===
        yielding = 0.0

        return {
            "collision_reward": collision,
            "high_speed_reward": high_speed,
            "progress_reward": progress,
            "success_reward": success,
            "on_road_reward": on_road,
            "lane_change_reward": lane_change,
            "yielding_reward": yielding,
        }

    def _reward(self, action: Action) -> float:
        """
        Aggregate weighted reward components into a single scalar.

        Returns:
            Scalar reward value
        """
        rewards = self._rewards(action)

        reward = sum(
            self.config.get(name, 0) * value
            for name, value in rewards.items()
            if name != "on_road_reward"
        )

        reward *= rewards["on_road_reward"]

        return reward

    def _is_terminated(self) -> bool:
        """Terminate if vehicle crashes, goes off-road, or reaches the end."""
        if self.vehicle.crashed:
            return True

        if not self.vehicle.on_road:
            return True

        if self.vehicle.position[0] >= self.config["road_length"]:
            return True

        return False

    def _is_truncated(self) -> bool:
        """Truncate episode if time limit exceeded."""
        return self.time >= self.config["duration"]


def register_group5_env():
    """Register the Group 5 custom environment with Gymnasium."""
    gym.register(
        id='group5-env-v0',
        entry_point='group5_custom_env:Group5Env',
    )
