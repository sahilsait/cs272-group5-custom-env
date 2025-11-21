"""
Group 5 Custom Highway Environment

A custom highway-env environment featuring:
- 5-lane straight highway
- Randomized lane closures (40m obstacles)
- Randomized stalled vehicles
- Emergency vehicles (faster, purple, traffic priority)
- Traffic (10 AI vehicles)
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

# === Monkey-patch to add emergency vehicle detection to observations ===
# The agent needs to know which vehicles are emergency vehicles, so we add
# an 'is_emergency' field to the observation data for all vehicles and obstacles.

original_vehicle_to_dict = Vehicle.to_dict
original_obstacle_to_dict = Obstacle.to_dict

def custom_vehicle_to_dict(self, origin_vehicle=None, observe_intentions=True):
    """Add emergency vehicle indicator to vehicle observations."""
    vehicle_data = original_vehicle_to_dict(self, origin_vehicle, observe_intentions)
    vehicle_data['is_emergency'] = float(getattr(self, 'is_emergency', False))
    return vehicle_data

def custom_obstacle_to_dict(self, origin_vehicle=None, observe_intentions=True):
    """Add emergency vehicle indicator to obstacle observations (always False)."""
    obstacle_data = original_obstacle_to_dict(self, origin_vehicle, observe_intentions)
    obstacle_data['is_emergency'] = 0.0  # Obstacles are never emergency vehicles
    return obstacle_data

# Apply the custom observation methods
Vehicle.to_dict = custom_vehicle_to_dict
Obstacle.to_dict = custom_obstacle_to_dict

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
        emergency_vehicle = self._check_emergency_vehicle_approaching()
        
        if emergency_vehicle:
            # Try to yield by changing lanes AWAY from the emergency vehicle
            self._yield_to_emergency(emergency_vehicle)
            self._emergency_yielding = True
        else:
            # Reset yielding behavior when no emergency vehicle is nearby
            if self._emergency_yielding:
                self._reset_yielding_behavior()
            self._emergency_yielding = False
        
        # Use parent's IDM behavior
        super().act(action)
    
    def _check_emergency_vehicle_approaching(self):
        """
        Look for emergency vehicles approaching from behind that we should yield to.
        
        Returns:
            EmergencyVehicle if one is close and approaching, otherwise None
        """
        if not hasattr(self, 'road') or not self.road:
            return None
        
        my_lane = self.lane_index[2] if len(self.lane_index) > 2 else 0
        my_position = self.position[0]
        
        # Check each vehicle on the road
        for vehicle in self.road.vehicles:
            if not isinstance(vehicle, EmergencyVehicle) or vehicle is self:
                continue
            
            emergency_lane = vehicle.lane_index[2] if len(vehicle.lane_index) > 2 else 0
            emergency_position = vehicle.position[0]
            
            # How far behind us is the emergency vehicle?
            distance_behind = my_position - emergency_position
            
            # Is it close enough to care about (behind us within 150m)?
            if distance_behind <= 0 or distance_behind > 150:
                continue
            
            # Is it in our lane or an adjacent lane?
            lane_difference = abs(emergency_lane - my_lane)
            if lane_difference > 1:
                continue
            
            # Is it actually approaching (going faster than us or very close)?
            is_faster = vehicle.speed > self.speed
            is_very_close = vehicle.speed > 0 and distance_behind < 80
            
            if is_faster or is_very_close:
                return vehicle  # Yes, we should yield!
        
        return None  # No emergency vehicle approaching
    
    def _yield_to_emergency(self, emergency_vehicle):
        """
        Get out of the way of the emergency vehicle!
        Try to move right. If we can't, slow down.
        """
        my_lane = self.lane_index[2] if len(self.lane_index) > 2 else 0
        rightmost_lane = len(self.road.network.graph["0"]["1"]) - 1
        
        # Can we move to the right?
        if my_lane < rightmost_lane:
            # Yes! Make lane change to the right very attractive
            self.delta = -1.0  # Lower threshold = more willing to change lanes
            self.politeness = 0.8  # Be cooperative, accept slower speed to help others
        else:
            # No, we're already in the rightmost lane
            # Slow down to let the emergency vehicle pass
            if not self._speed_modified:
                self._original_target_speed = self.target_speed
                self._speed_modified = True
            
            # Reduce speed by 15% (minimum 18 m/s)
            if self.speed > 18.0:
                self.target_speed = max(18.0, self.speed * 0.85)
    
    def _reset_yielding_behavior(self):
        """Return to normal driving behavior after the emergency vehicle has passed."""
        # Reset lane change parameters to normal
        self.delta = 0.0
        self.politeness = 0.5
        
        # Restore normal target speed if we had slowed down
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
        
        # Visual appearance
        self.color = (0.8, 0.0, 0.8)  # Bright purple/magenta for visibility
        
        # Speed settings - emergency vehicles are fast!
        self.target_speed = 35.0  # Faster than the 25 m/s speed limit
        self.speed = min(speed, self.target_speed)
        
        # Aggressive driving behavior
        self.politeness = 0.0  # Don't slow down for others
        self.delta = -0.5  # Change lanes easily
        self.T = 0.5  # Follow closer (smaller time headway)
        self.ACC_MAX = 6.0  # Accelerate faster
        
        # Mark this vehicle so the agent can detect it
        self.is_emergency = True
    
    def act(self, action=None):
        """
        Emergency vehicles prefer the leftmost (fast) lane and drive aggressively.
        """
        my_lane = self.lane_index[2] if len(self.lane_index) > 2 else 0
        
        if my_lane > 0:
            # We're not in the leftmost lane yet - try to move left
            self.delta = -0.8  # Make left lane changes attractive
        else:
            # We're already in the leftmost lane - stay here
            self.delta = 0.2  # Prefer staying in current lane
        
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
        - Yielding: Reward for not blocking emergency vehicle
    """

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                "observation": {
                    "type": "Kinematics",
                    # Add 'is_emergency' to features so agent can distinguish Emergency Vehicle
                    "features": ["x", "y", "vx", "vy", "is_emergency"], 
                    "features_range": {
                        "x": [0, 625],
                        "y": [-10, 20],
                        "vx": [0, 40],
                        "vy": [-5, 5],
                        "is_emergency": [0, 1],
                    },
                },
                "action": {"type": "DiscreteMetaAction"},
                "lanes_count": 5,
                "road_length": 625,  # meters
                "speed_limit": 25,   # m/s
                "duration": 50,      # seconds
                "vehicles_count": 10,
                "vehicles_density": 1.0,

                # Lane closure config
                "lane_closure_length": 40,
                "closure_lane_range": [1, 3],  # lanes that can be closed
                "closure_position_range": [250, 400],  # mid-road

                # Stalled vehicle config
                "stalled_position_before_range": [150, 250],
                "stalled_position_after_range": [450, 550],

                # Emergency vehicle config
                "emergency_vehicles_count": 1,  # Number of emergency vehicles
                "emergency_vehicle_speed": 35.0,  # Target speed (m/s)
                "emergency_spawn_range": [100, 500],  # Where they can spawn
                "emergency_spawn_probability": 0.5,  # 50% chance to spawn emergency vehicle

                # Reward weights - tuned for hazard navigation
                "collision_reward": -1.5,      # Heavy penalty for crashing
                "high_speed_reward": 0.3,      # Moderate reward for speed
                "progress_reward": 0.4,        # Good reward for moving forward
                "success_reward": 1.0,         # Big bonus for completion
                "lane_change_reward": -0.01,   # Small penalty for lane changes
                "yielding_reward": 0.5,        # Strong reward for yielding to emergency vehicles

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
        # Stalled vehicle is NOT an emergency vehicle
        stalled_vehicle.is_emergency = False
        self.road.vehicles.append(stalled_vehicle)

        # Emergency Vehicles (spawn with probability)
        if self.np_random.random() < self.config["emergency_spawn_probability"]:
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
        # Ego is NOT an emergency vehicle
        ego_vehicle.is_emergency = False
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
            # Traffic is NOT an emergency vehicle
            traffic_vehicle.is_emergency = False
            self.road.vehicles.append(traffic_vehicle)

    def _rewards(self, action: Action) -> dict[str, float]:
        """
        Calculate how well the agent is doing.
        Returns a dictionary of reward components.
        """
        ego = self.vehicle  # The agent's vehicle

        # 1. Did we crash? (Big penalty)
        collision_penalty = float(ego.crashed)

        # 2. Are we going fast enough?
        forward_speed = ego.speed * np.cos(ego.heading)
        speed_normalized = utils.lmap(
            forward_speed,
            self.config["reward_speed_range"],  # [20, 28] m/s
            [0, 1]
        )
        speed_reward = np.clip(speed_normalized, 0, 1)

        # 3. How far have we traveled?
        progress_reward = ego.position[0] / self.config["road_length"]

        # 4. Did we reach the end successfully?
        reached_end_safely = (
            ego.position[0] >= self.config["road_length"]
            and not ego.crashed
        )
        success_bonus = float(reached_end_safely)

        # 5. Are we still on the road?
        on_road_reward = float(ego.on_road)

        # 6. Did we change lanes? (Small penalty for unnecessary lane changes)
        changed_lanes = action in [0, 2]  # 0 = left, 2 = right
        lane_change_penalty = float(changed_lanes)

        # 7. Are we yielding properly to emergency vehicles?
        yielding_reward = self._calculate_yielding_reward()

        # Return all the reward components
        return {
            "collision_reward": collision_penalty,
            "high_speed_reward": speed_reward,
            "progress_reward": progress_reward,
            "success_reward": success_bonus,
            "on_road_reward": on_road_reward,
            "lane_change_reward": lane_change_penalty,
            "yielding_reward": yielding_reward,
        }
    
    def _calculate_yielding_reward(self) -> float:
        """
        Check if we're properly yielding to emergency vehicles.
        Returns positive reward for yielding, negative for blocking.
        """
        ego = self.vehicle
        my_lane = ego.lane_index[2]
        my_position = ego.position[0]
        
        # Look for emergency vehicles near us
        for emergency_vehicle in self.road.vehicles:
            if not isinstance(emergency_vehicle, EmergencyVehicle):
                continue
            if emergency_vehicle is ego:
                continue
            
            ev_lane = emergency_vehicle.lane_index[2]
            ev_position = emergency_vehicle.position[0]
            
            # How far behind us is the emergency vehicle?
            distance_behind = my_position - ev_position
            
            # Only care if it's behind us within 60 meters
            if not (0 < distance_behind < 60):
                continue
            
            # Are we blocking it?
            if ev_lane == my_lane:
                # BAD: We're in the emergency vehicle's way!
                return -0.5
            else:
                # GOOD: Emergency vehicle is nearby but we're not blocking it
                return 0.1
        
        # No emergency vehicle nearby
        return 0.0

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