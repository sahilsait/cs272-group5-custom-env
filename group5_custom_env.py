"""
Group 5 Custom Highway Environment

A custom highway-env environment featuring:
- 5-lane straight highway
- Randomized lane closures (40m obstacles)
- Randomized stalled vehicles
- Emergency vehicles (faster, purple, overtake all traffic)
- Traffic (20 AI vehicles)
- Custom reward function for hazard navigation

#TestComment

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


class YieldingIDMVehicle(IDMVehicle):
    """
    IDM vehicle that yields to emergency vehicles by changing lanes or slowing down.
    """
    
    def __init__(self, road, position, heading=0, speed=0):
        super().__init__(road, position, heading, speed)
        self._emergency_yielding = False  # Track if currently yielding
        self._original_target_speed = getattr(self, 'target_speed', None)  # Store original target speed
        self._speed_modified = False  # Track if we modified target speed
    
    def act(self, action=None):
        """
        Check for emergency vehicles and yield if one is approaching from behind.
        """
        # Check for emergency vehicles approaching from behind
        emergency_approaching = self._check_emergency_vehicle_approaching()
        
        if emergency_approaching:
            # Try to yield by changing lanes to the right or slowing down
            self._yield_to_emergency()
            self._emergency_yielding = True
        else:
            # Reset yielding behavior when no emergency vehicle is nearby
            if self._emergency_yielding:
                self._reset_yielding_behavior()
            self._emergency_yielding = False
        
        # Use parent's IDM behavior
        super().act(action)
    
    def _check_emergency_vehicle_approaching(self) -> bool:
        """
        Check if an emergency vehicle is approaching from behind.
        
        Returns:
            True if an emergency vehicle is detected approaching from behind
        """
        if not hasattr(self, 'road') or not self.road:
            return False
        
        current_lane = self.lane_index[2] if len(self.lane_index) > 2 else 0
        current_position = self.position[0]  # Longitudinal position
        
        for vehicle in self.road.vehicles:
            # Check if it's an emergency vehicle
            if isinstance(vehicle, EmergencyVehicle) and vehicle is not self:
                vehicle_lane = vehicle.lane_index[2] if len(vehicle.lane_index) > 2 else 0
                vehicle_position = vehicle.position[0]
                
                # Check if emergency vehicle is behind (lower longitudinal position)
                # and in the same lane or adjacent lanes (within 1 lane)
                distance_behind = current_position - vehicle_position
                if (distance_behind > 0 and 
                    distance_behind < 150 and  # Within 150m behind
                    abs(vehicle_lane - current_lane) <= 1):  # Same or adjacent lane
                    # Check if it's approaching (faster speed or catching up)
                    if vehicle.speed > self.speed or (vehicle.speed > 0 and distance_behind < 80):
                        return True
        
        return False
    
    def _yield_to_emergency(self):
        """
        Yield to emergency vehicle by attempting to change lanes to the right.
        If not possible, slow down.
        """
        current_lane = self.lane_index[2] if len(self.lane_index) > 2 else 0
        max_lane = len(self.road.network.graph["0"]["1"]) - 1
        
        # Try to change to the right lane (higher lane index = right side)
        if current_lane < max_lane:
            # Make right lane changes more favorable by adjusting MOBIL parameters
            # Lower delta makes lane changes easier (more likely)
            self.delta = -1.0  # Very low threshold - prioritize yielding
            # Increase politeness to allow yielding even if slightly slower
            self.politeness = 0.8
            
            # Check if right lane is relatively clear
            right_lane_index = ("0", "1", current_lane + 1)
            right_lane_clear = True
            min_distance = 50  # Minimum safe distance
            
            for vehicle in self.road.vehicles:
                if vehicle is not self and vehicle.lane_index == right_lane_index:
                    distance = abs(vehicle.position[0] - self.position[0])
                    if distance < min_distance:
                        right_lane_clear = False
                        break
            
            # If right lane is clear, the MOBIL model should favor the lane change
            # We've already adjusted delta and politeness to make it more likely
        else:
            # Already in rightmost lane, slow down to let emergency vehicle pass
            # Reduce target speed temporarily
            if not self._speed_modified:
                self._original_target_speed = self.target_speed
                self._speed_modified = True
            if self.speed > 18.0:
                self.target_speed = max(18.0, self.speed * 0.85)  # Slow down by 15%
    
    def _reset_yielding_behavior(self):
        """Reset behavior parameters to normal after yielding."""
        # Reset to default MOBIL parameters
        self.delta = 0.0  # Default threshold
        self.politeness = 0.5  # Default politeness
        # Restore original target speed if we modified it
        if self._speed_modified and self._original_target_speed is not None:
            self.target_speed = self._original_target_speed
            self._speed_modified = False


class EmergencyVehicle(IDMVehicle):
    """
    Emergency vehicle that goes faster than all traffic and overtakes vehicles.
    Purple in color.
    """
    
    def __init__(self, road, position, heading=0, speed=0):
        super().__init__(road, position, heading, speed)
        # Set purple color (RGB normalized 0-1)
        self.color = (0.5, 0.0, 0.5)  # Purple
        # Set higher desired speed than normal traffic
        self.target_speed = 35.0  # Faster than speed limit (25 m/s) and ego vehicle
        # Aggressive behavior for overtaking
        self.speed = min(speed, self.target_speed)
        # More aggressive lane changing parameters for MOBIL
        self.politeness = 0.0  # No politeness - always prioritize own speed (default is usually 0.5)
        self.delta = -0.5  # Lower threshold for lane changes (default is usually 0.0)
        # Lower time headway for more aggressive following
        self.T = 0.5  # Default is usually 1.0-1.5
        # Higher acceleration capability
        self.ACC_MAX = 6.0  # More aggressive acceleration (default is usually 3.0)
    
    def act(self, action=None):
        """
        Emergency vehicles always try to maintain high speed and overtake.
        """
        # Force high speed - emergency vehicles accelerate faster
        if self.speed < self.target_speed:
            self.speed = min(self.speed + 3.0, self.target_speed)
        
        # Use parent's IDM behavior but with aggressive parameters
        super().act(action)


class Group5Env(AbstractEnv):
    """
    A custom highway driving environment with lane closures, stalled vehicles, and emergency vehicles.

    The ego-vehicle must navigate a 5-lane highway with:
    - A lane closure (40m obstacle) in lanes 1-3
    - A stalled vehicle (speed=0) in a different lane
    - Emergency vehicles (purple, faster, overtake all traffic)
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

                # Emergency vehicle config
                "emergency_vehicles_count": 2,  # Number of emergency vehicles
                "emergency_vehicle_speed": 35.0,  # Target speed (m/s)
                "emergency_spawn_range": [100, 500],  # Where they can spawn

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
        """Spawn lane closure, stalled vehicle, and emergency vehicles with random positions."""

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

        # Emergency Vehicles
        emergency_count = self.config["emergency_vehicles_count"]
        spawn_min, spawn_max = self.config["emergency_spawn_range"]
        
        for _ in range(emergency_count):
            # Random lane for emergency vehicle
            emergency_lane = self.np_random.integers(0, self.config["lanes_count"])
            emergency_position = self.np_random.uniform(spawn_min, spawn_max)
            
            # Create emergency vehicle
            emergency_vehicle = EmergencyVehicle.make_on_lane(
                self.road,
                lane_index=("0", "1", emergency_lane),
                longitudinal=emergency_position,
                speed=self.config["emergency_vehicle_speed"] * 0.8  # Start at 80% of target speed
            )
            # Update target speed from config
            emergency_vehicle.target_speed = self.config["emergency_vehicle_speed"]
            self.road.vehicles.append(emergency_vehicle)

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

        # Traffic vehicles (using YieldingIDMVehicle to yield to emergency vehicles)
        vehicles_count = self.config["vehicles_count"]
        for _ in range(vehicles_count):
            traffic_vehicle = YieldingIDMVehicle.create_random(
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