"""
Group 5 Custom Highway Environment

A custom highway-env environment featuring:
- 5-lane straight highway
- Randomized lane closures (40m obstacles)
- Randomized stalled vehicles
- Emergency vehicles (faster, purple, overtake all traffic)
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
    LENGTH = 150.0  # 150 meters long
    WIDTH = 4.0     # Full lane width


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
        Also avoid obstacles and stalled vehicles ahead.
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
        
        # Check for obstacles ahead and avoid them
        self._avoid_obstacles_ahead()
        
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
    
    def _avoid_obstacles_ahead(self):
        """
        Detect obstacles, stalled vehicles, or lane closures ahead and take evasive action.
        Either change lanes if possible, or slow down/stop to avoid collision.
        """
        if not hasattr(self, 'road') or not self.road:
            return
        
        current_lane = self.lane_index[2] if len(self.lane_index) > 2 else 0
        current_position = self.position[0]
        look_ahead_distance = 60.0  # Look 60m ahead
        critical_distance = 30.0  # Critical zone - must take action
        
        obstacle_ahead = False
        min_obstacle_distance = float('inf')
        
        # Check for obstacles (lane closures)
        if hasattr(self.road, 'objects'):
            for obj in self.road.objects:
                if hasattr(obj, 'lane_index') and hasattr(obj, 'position'):
                    obj_lane = obj.lane_index[2] if len(obj.lane_index) > 2 else 0
                    if obj_lane == current_lane:
                        distance_to_obj = obj.position[0] - current_position
                        if 0 < distance_to_obj < look_ahead_distance:
                            obstacle_ahead = True
                            min_obstacle_distance = min(min_obstacle_distance, distance_to_obj)
        
        # Check for stalled or very slow vehicles ahead
        for vehicle in self.road.vehicles:
            if vehicle is not self:
                vehicle_lane = vehicle.lane_index[2] if len(vehicle.lane_index) > 2 else 0
                if vehicle_lane == current_lane:
                    distance_to_vehicle = vehicle.position[0] - current_position
                    # Consider vehicles that are stopped or very slow (< 5 m/s)
                    if 0 < distance_to_vehicle < look_ahead_distance and vehicle.speed < 5.0:
                        obstacle_ahead = True
                        min_obstacle_distance = min(min_obstacle_distance, distance_to_vehicle)
        
        if obstacle_ahead:
            # Try to change lanes first
            lane_changed = self._try_lane_change_for_obstacle(current_lane, min_obstacle_distance)
            
            # If can't change lanes, slow down or stop
            if not lane_changed:
                if min_obstacle_distance < critical_distance:
                    # Emergency braking
                    if not self._speed_modified:
                        self._original_target_speed = self.target_speed
                        self._speed_modified = True
                    # Reduce speed based on distance to obstacle
                    safe_speed = max(0.0, min_obstacle_distance / critical_distance * 10.0)
                    self.target_speed = safe_speed
                elif min_obstacle_distance < look_ahead_distance:
                    # Gradual slowdown
                    if not self._speed_modified:
                        self._original_target_speed = self.target_speed
                        self._speed_modified = True
                    self.target_speed = max(15.0, self.target_speed * 0.7)
    
    def _try_lane_change_for_obstacle(self, current_lane: int, obstacle_distance: float) -> bool:
        """
        Try to change lanes to avoid obstacle.
        Returns True if a lane change is feasible, False otherwise.
        """
        max_lane = len(self.road.network.graph["0"]["1"]) - 1
        
        # Prefer left lane (lower index) for overtaking
        lanes_to_try = []
        if current_lane > 0:
            lanes_to_try.append(current_lane - 1)  # Left lane
        if current_lane < max_lane:
            lanes_to_try.append(current_lane + 1)  # Right lane
        
        for target_lane in lanes_to_try:
            if self._is_lane_clear(target_lane, obstacle_distance):
                # Make lane change more favorable
                self.delta = -2.0  # Very aggressive lane change
                self.politeness = 0.3  # Low politeness for safety
                return True
        
        return False
    
    def _is_lane_clear(self, lane_index: int, distance_ahead: float) -> bool:
        """
        Check if a lane is clear for a safe lane change.
        """
        target_lane_index = ("0", "1", lane_index)
        min_safe_distance = 40.0  # Need 40m clearance
        
        # Check for vehicles in target lane
        for vehicle in self.road.vehicles:
            if vehicle is not self and vehicle.lane_index == target_lane_index:
                distance = vehicle.position[0] - self.position[0]
                # Check both ahead and behind
                if abs(distance) < min_safe_distance:
                    return False
        
        # Check for obstacles in target lane
        if hasattr(self.road, 'objects'):
            for obj in self.road.objects:
                if hasattr(obj, 'lane_index') and obj.lane_index == target_lane_index:
                    if hasattr(obj, 'position'):
                        distance = obj.position[0] - self.position[0]
                        if -20 < distance < distance_ahead + 20:
                            return False
        
        return True


class EmergencyVehicle(IDMVehicle):
    """
    Emergency vehicle that goes faster than all traffic and overtakes vehicles.
    Purple in color. Cannot crash - passes through obstacles.
    """
    
    def __init__(self, road, position, heading=0, speed=0):
        super().__init__(road, position, heading, speed)
        # Set purple color (RGB normalized 0-1)
        self.color = (0.5, 0.0, 0.5)  # Purple
        # Set higher desired speed than normal traffic
        self.target_speed = 27.0  # 2 m/s faster than ego
        # Aggressive but controlled behavior for overtaking
        self.speed = min(speed, self.target_speed)
        # Lane changing parameters for MOBIL - less aggressive to stay stable
        self.politeness = 0.1  # Low politeness but not zero
        self.delta = -0.3  # Easier lane changes but not too aggressive
        # Time headway
        self.T = 0.8  # Closer following but safe
        # Acceleration
        self.ACC_MAX = 4.0  # Moderate acceleration
        # Lane keeping - make sure it follows lanes properly
        self.LANE_CHANGE_MIN_ACC = 0.5  # Minimum acceleration advantage for lane change
        self.LANE_CHANGE_MAX_BRAKING_IMPOSED = 2.0  # Don't impose too much braking on others
        # Emergency vehicle can collide (physics enabled)
        self.collidable = True  # Emergency vehicle has physics enabled
    
    def act(self, action=None):
        """
        Emergency vehicles try to maintain high speed and overtake safely.
        They actively avoid collisions by changing lanes or slowing down.
        """
        # Check for obstacles and vehicles ahead
        self._avoid_collisions()
        
        # Ensure vehicle stays on road by checking lane boundaries
        if hasattr(self, 'lane') and self.lane:
            # Keep vehicle centered in lane to prevent going off-road
            lateral_position = self.position[1]
            lane_center = self.lane.position(self.lane.local_coordinates(self.position)[0], 0)[1]
            
            # If drifting too far from center, correct heading
            lateral_error = lateral_position - lane_center
            if abs(lateral_error) > 1.5:  # More than 1.5m from center
                # Gentle correction towards lane center
                correction = -0.1 * lateral_error
                self.heading += correction
        
        # Maintain target speed with smooth acceleration
        if self.speed < self.target_speed:
            acceleration = min(1.5, self.target_speed - self.speed)  # Smooth acceleration
            self.speed = min(self.speed + acceleration, self.target_speed)
        
        # Use parent's IDM behavior with lane keeping
        super().act(action)
    
    def _avoid_collisions(self):
        """
        Detect obstacles and vehicles ahead and take evasive action.
        Emergency vehicle will change lanes or slow down to avoid collisions.
        """
        if not hasattr(self, 'road') or not self.road:
            return
        
        current_lane = self.lane_index[2] if len(self.lane_index) > 2 else 0
        current_position = self.position[0]
        look_ahead_distance = 80.0  # Look 80m ahead
        critical_distance = 40.0  # Critical zone - must take action
        
        obstacle_ahead = False
        min_obstacle_distance = float('inf')
        
        # Check for obstacles (lane closures)
        if hasattr(self.road, 'objects'):
            for obj in self.road.objects:
                if hasattr(obj, 'lane_index') and hasattr(obj, 'position'):
                    obj_lane = obj.lane_index[2] if len(obj.lane_index) > 2 else 0
                    if obj_lane == current_lane:
                        distance_to_obj = obj.position[0] - current_position
                        if 0 < distance_to_obj < look_ahead_distance:
                            obstacle_ahead = True
                            min_obstacle_distance = min(min_obstacle_distance, distance_to_obj)
        
        # Check for vehicles ahead (including stalled ones)
        for vehicle in self.road.vehicles:
            if vehicle is not self:
                vehicle_lane = vehicle.lane_index[2] if len(vehicle.lane_index) > 2 else 0
                if vehicle_lane == current_lane:
                    distance_to_vehicle = vehicle.position[0] - current_position
                    if 0 < distance_to_vehicle < look_ahead_distance:
                        # Consider vehicles ahead
                        obstacle_ahead = True
                        min_obstacle_distance = min(min_obstacle_distance, distance_to_vehicle)
        
        if obstacle_ahead:
            # Try to change lanes to avoid obstacle
            lane_changed = self._try_emergency_lane_change(current_lane, min_obstacle_distance)
            
            # If can't change lanes safely, slow down
            if not lane_changed and min_obstacle_distance < critical_distance:
                # Gradual slowdown based on distance
                safe_speed = max(15.0, (min_obstacle_distance / critical_distance) * self.target_speed)
                if self.speed > safe_speed:
                    self.speed = max(safe_speed, self.speed - 2.0)  # Decelerate
    
    def _try_emergency_lane_change(self, current_lane: int, obstacle_distance: float) -> bool:
        """
        Try to change lanes to avoid obstacle ahead.
        Returns True if lane change is initiated, False otherwise.
        """
        max_lane = len(self.road.network.graph["0"]["1"]) - 1
        
        # Try left lane first (for passing), then right lane
        lanes_to_try = []
        if current_lane > 0:
            lanes_to_try.append(current_lane - 1)  # Left lane
        if current_lane < max_lane:
            lanes_to_try.append(current_lane + 1)  # Right lane
        
        for target_lane in lanes_to_try:
            if self._is_emergency_lane_clear(target_lane):
                # Make lane change very favorable
                self.delta = -3.0  # Very aggressive lane change for emergency
                self.politeness = 0.0  # No politeness in emergency
                return True
        
        return False
    
    def _is_emergency_lane_clear(self, lane_index: int) -> bool:
        """
        Check if a lane is clear enough for emergency lane change.
        Less strict than normal vehicles.
        """
        target_lane_index = ("0", "1", lane_index)
        min_safe_distance = 30.0  # Emergency vehicles need less clearance
        
        # Check for vehicles in target lane
        for vehicle in self.road.vehicles:
            if vehicle is not self and vehicle.lane_index == target_lane_index:
                distance = vehicle.position[0] - self.position[0]
                # Check both ahead and behind
                if abs(distance) < min_safe_distance:
                    return False
        
        # Check for obstacles in target lane (within reasonable range)
        if hasattr(self.road, 'objects'):
            for obj in self.road.objects:
                if hasattr(obj, 'lane_index') and obj.lane_index == target_lane_index:
                    if hasattr(obj, 'position'):
                        distance = obj.position[0] - self.position[0]
                        # Only avoid if very close
                        if -10 < distance < 60:
                            return False
        
        return True


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
                "emergency_vehicles_count": 1,  # Number of emergency vehicles
                "emergency_vehicle_speed": 27.0,  # Target speed (2 units faster than ego at 25 m/s)
                "emergency_spawn_behind_ego": 50.0,  # Spawns 50m behind ego

                # Reward weights - tuned for hazard navigation
                "collision_reward": -1.0,      # Collision penalty
                "high_speed_reward": 0.3,      # Moderate reward for speed
                "progress_reward": 0.4,        # Good reward for moving forward
                "success_reward": 1.0,         # Success bonus
                "lane_change_reward": -0.05,   # Penalty for lane changes (5x original)
                "yielding_reward": 1.0,        # Reward for yielding to emergency vehicle

                # Reward settings
                "normalize_reward": False,
                "reward_speed_range": [24, 27],  # Target speed range (24-27 m/s)
            }
        )
        return config

    def _reset(self) -> None:
        self._create_road()
        self._create_vehicles()
        self._spawn_custom_elements()
        # Reset tracking variables
        self._prev_position = 0.0
        self._emergency_passed = False
        self._emergency_was_behind = False

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

        # Stalled Vehicles - one before closure, one after closure
        lanes = list(range(self.config["lanes_count"]))
        available_lanes = [lane for lane in lanes if lane != closure_lane]
        
        # First stalled vehicle - before the closure
        stalled_lane_1 = self.np_random.choice(available_lanes)
        pos_min, pos_max = self.config["stalled_position_before_range"]
        stalled_position_1 = self.np_random.uniform(pos_min, pos_max)
        stalled_vehicle_1 = Vehicle.make_on_lane(
            self.road,
            lane_index=("0", "1", stalled_lane_1),
            longitudinal=stalled_position_1,
            speed=0.0
        )
        self.road.vehicles.append(stalled_vehicle_1)
        
        # Second stalled vehicle - after the closure
        stalled_lane_2 = self.np_random.choice(available_lanes)
        pos_min, pos_max = self.config["stalled_position_after_range"]
        stalled_position_2 = self.np_random.uniform(pos_min, pos_max)
        stalled_vehicle_2 = Vehicle.make_on_lane(
            self.road,
            lane_index=("0", "1", stalled_lane_2),
            longitudinal=stalled_position_2,
            speed=0.0
        )
        self.road.vehicles.append(stalled_vehicle_2)

        # Emergency Vehicles
        emergency_count = self.config["emergency_vehicles_count"]
        
        for _ in range(emergency_count):
            # Spawn in middle lane (lane 2 for 5-lane highway: 0,1,2,3,4)
            middle_lane = self.config["lanes_count"] // 2
            # Spawn 50m behind ego (ego starts at 0, so emergency at -50)
            emergency_position = -self.config["emergency_spawn_behind_ego"]
            
            # Create emergency vehicle
            emergency_vehicle = EmergencyVehicle.make_on_lane(
                self.road,
                lane_index=("0", "1", middle_lane),
                longitudinal=emergency_position,
                speed=self.config["emergency_vehicle_speed"] * 0.9  # Start at 90% of target speed
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
        # Ego vehicle - start at middle lane (lane 2) at position 0
        middle_lane = self.config["lanes_count"] // 2
        
        # Create a base vehicle at the desired position
        ego_lane_index = ("0", "1", middle_lane)
        ego_lane = self.road.network.get_lane(ego_lane_index)
        ego_position = ego_lane.position(0.0, 0)  # longitudinal=0, lateral=0
        
        # Create ego vehicle with action type wrapper
        ego_vehicle = self.action_type.vehicle_class(
            self.road,
            ego_position,
            heading=ego_lane.heading_at(0.0),
            speed=25.0
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

        # === 2. SPEED REWARD (only for 23-27 m/s range) ===
        forward_speed = ego.speed * np.cos(ego.heading)
        speed_min, speed_max = self.config["reward_speed_range"]
        # Reward only if within target range
        high_speed = 1.0 if speed_min <= forward_speed <= speed_max else 0.0

        # === 3. PROGRESS REWARD (per-step distance moved) ===
        # Track previous position to calculate distance moved this step
        if not hasattr(self, '_prev_position'):
            self._prev_position = 0.0
        
        distance_moved = ego.position[0] - self._prev_position
        progress = distance_moved / self.config["road_length"]
        self._prev_position = ego.position[0]

        # === 4. SUCCESS BONUS ===
        reached_end = (
            ego.position[0] >= self.config["road_length"]
            and not ego.crashed
        )
        success = float(reached_end)

        # === 5. LANE CHANGE PENALTY ===
        lane_change = float(action in [0, 2])

        # === 6. YIELDING REWARD (when emergency vehicle passes) ===
        yielding = 0.0
        
        # Track emergency vehicle position to detect passing
        if not hasattr(self, '_emergency_passed'):
            self._emergency_passed = False
            self._emergency_was_behind = False
        
        # Find emergency vehicle
        for vehicle in self.road.vehicles:
            if hasattr(vehicle, 'color') and vehicle.color == (0.5, 0.0, 0.5):
                emergency_distance = vehicle.position[0] - ego.position[0]
                emergency_is_ahead = emergency_distance > 0
                
                # Check if emergency vehicle just passed (was behind, now ahead)
                if self._emergency_was_behind and emergency_is_ahead and not self._emergency_passed:
                    yielding = 1.0
                    self._emergency_passed = True
                
                # Update tracking for next step (emergency is behind if distance < 0)
                self._emergency_was_behind = not emergency_is_ahead
                break

        return {
            "collision_reward": collision,
            "high_speed_reward": high_speed,
            "progress_reward": progress,
            "success_reward": success,
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
        
        # Store the last rewards for logging purposes
        self._last_rewards = rewards

        reward = sum(
            self.config.get(name, 0) * value
            for name, value in rewards.items()
        )

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