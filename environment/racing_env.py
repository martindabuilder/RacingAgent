import gymnasium as gym
import numpy as np
from gymnasium import spaces
from simulation.vehicle_controller import VehicleController
from track.track_geometry import TrackGeometry


class RacingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, dt=1.0 / 120.0, max_episode_steps=12000, start_speed=0.0, start_index=1):
        super().__init__()

        self.dt = float(dt)
        self.max_episode_steps = int(max_episode_steps)

        self.start_speed = float(start_speed)
        self.start_index = int(start_index)

        self.current_step = 0
        self.total_steps = 0

        # loading the Yas Marina track
        self.track = TrackGeometry()
        self.track_points = np.asarray(self.track.centerline, dtype=np.float64)
        self.track_length = float(self.track.get_track_length())
        self.num_track_points = len(self.track_points)

        if self.num_track_points < 2:
            raise ValueError("Track must contain at least two points.")

        self.start_index %= self.num_track_points

        # making sure that the environment is a closed circle, that connects all points together.
        closed_points = np.vstack([self.track_points, self.track_points[0]])

        segment_vectors = np.diff(closed_points, axis=0)
        self.centerline_segment_lengths = np.linalg.norm(segment_vectors, axis=1)
        self.centerline_cumulative = np.concatenate([[0.0], np.cumsum(self.centerline_segment_lengths)])
        self.centerline_length = float(self.centerline_cumulative[-1])

        if self.centerline_length <= 0.0:
            raise ValueError("Centerline length must be positive.")

        # creating the W11 and its controller with the built-in actions
        self.controller = VehicleController()
        self.state = None
        self.previous_action = np.zeros(3, dtype=np.float32)
        self.previous_progress = 0.0
        self.current_progress = 0.0

        # actions in this order: [steering, throttle, brake]
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

        # includes the 19 main parameters and details for the agent
        # 0 - speed
        # 1 - lateral speed
        # 2 - cornering speed
        # 3 - lateral offset
        # 4 - heading error
        # 5 - acceleration
        # 6 - lateral acceleration
        # 7 - steering angle for the front wheels
        # 8 to 15 include lookahead values: 5.0/10.0/20.0/40.0
        # 16 to 18 are track curviture, left and right width of the track
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(19,), dtype=np.float32)

    def _get_position(self):
        return np.array([self.state["x"], self.state["y"]], dtype=np.float64)

    # nearest position to the track
    def _get_nearest_track_point(self):
        position = self._get_position()
        return self.track.get_nearest_point(position)

    def _get_centerline_point_at_progress(self, progress):
        progress = float(progress)
        progress %= self.centerline_length

        index = (np.searchsorted(self.centerline_cumulative, progress, side="right") - 1)
        index = min(max(index, 0), self.num_track_points - 1)

        segment_start = self.centerline_cumulative[index]
        segment_length = self.centerline_segment_lengths[index]

        if segment_length <= 1e-9:
            return self.track_points[index].copy()

        fraction = (progress - segment_start) / segment_length
        next_index = (index + 1) % self.num_track_points
        point = (self.track_points[index]+ fraction * (self.track_points[next_index] - self.track_points[index]))

        return point

    def _get_relative_lookahead(self, nearest, lookahead_distance):
        target_progress = (float(nearest["progress"]) + float(lookahead_distance))
        target_point = self._get_centerline_point_at_progress(target_progress)
        vehicle_position = self._get_position()

        dx = (target_point[0] - vehicle_position[0])
        dy = (target_point[1] - vehicle_position[1])

        heading = float(self.state["heading"])
        forward = (np.cos(heading) * dx + np.sin(heading) * dy)
        lateral = (-np.sin(heading) * dx + np.cos(heading) * dy)

        return float(forward), float(lateral)

    # combines all the information from the 19 parameters 
    def _get_observation(self):
        nearest = self._get_nearest_track_point()
        speed = float(self.state["speed"])
        lateral_speed = float(self.state["lateral_speed"])
        yaw_rate = float(self.state["yaw_rate"])
        acceleration = float(self.state["acceleration"])
        lateral_acceleration = float(self.state["lateral_acceleration"])
        steering_angle = float(self.state["steering_angle"])
        lateral_offset = float(nearest["lateral_offset"])
        vehicle_heading = float(self.state["heading"])
        track_heading = float(nearest["heading_rad"])
        heading_error = (vehicle_heading - track_heading + np.pi) % (2.0 * np.pi) - np.pi
        width_left = float(nearest["width_left"])
        width_right = float(nearest["width_right"])

        lookahead_distances = [5.0, 10.0, 20.0, 40.0]
        lookahead_values = []

        for distance in lookahead_distances:
            forward, lateral = self._get_relative_lookahead(nearest, distance)
            normalized_forward = (forward / distance)

            normalized_lateral = (lateral / max(width_left, width_right, 1.0))
            lookahead_values.extend([normalized_forward, normalized_lateral])

        curvature = float(nearest["curvature"])

        observation = np.array(
            [
                speed / 100.0,
                lateral_speed / 30.0,
                yaw_rate / 3.0,
                lateral_offset / max(width_left, width_right, 1.0),
                heading_error / np.pi,
                acceleration / 30.0,
                lateral_acceleration / 50.0,
                steering_angle / 0.55,

                # lookahead positions
                *lookahead_values,

                # track information
                curvature / 0.2,
                width_left / 20.0,
                width_right / 20.0
            ],
            dtype=np.float32
        )

        observation = np.nan_to_num(observation, nan=0.0, posinf=1.0, neginf=-1.0)
        observation = np.clip(observation, -1.0, 1.0)

        return observation

    def _calculate_progress_delta(self, previous, current):
        delta = current - previous
        half_length = self.track_length / 2.0

        if delta > half_length:
            delta -= self.track_length

        elif delta < -half_length:
            delta += self.track_length

        return float(delta)

   
    def _is_within_track(self):
        position = self._get_position()
        return bool(self.track.is_within_track(position))

    # takes into account all the rules we have for the agent
    # i.e penalties and rewards for 
    def _calculate_reward(self, progress_delta, nearest, previous_action, current_action):
        reward = 0.0
        reward += (progress_delta * 0.1)
        reward -= (abs(float(self.state["lateral_speed"])) * 0.01)

        steering_change = abs(float(current_action[0]) - float(previous_action[0]))
        reward -= (steering_change * 0.05)

        lateral_offset = abs(float(nearest["lateral_offset"]))
        available_width = min(float(nearest["width_left"]), float(nearest["width_right"]))

        if available_width > 0.0:

            edge_ratio = (lateral_offset / available_width)

            if edge_ratio > 0.7:
                reward -= (edge_ratio - 0.7) * 0.5

        return float(reward)

   
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_step = 0
        self.previous_action = np.zeros(3, dtype=np.float32)

        start_position = self.track_points[self.start_index]

        next_index = (self.start_index + 1) % self.num_track_points
        direction = (self.track_points[next_index] - start_position)
        heading = float(np.arctan2(direction[1], direction[0]))

        self.state = self.controller.reset(
            x=float(start_position[0]),
            y=float(start_position[1]),
            heading=heading,
            speed=self.start_speed
        )

        nearest = self._get_nearest_track_point()

        self.current_progress = float(nearest["progress"])
        self.previous_progress = (self.current_progress)

        observation = self._get_observation()

        info = {
            "progress": self.current_progress,
            "speed": float(self.state["speed"]),
            "lateral_offset": float(nearest["lateral_offset"]),
            "observation_size": len(observation)
        }

        return observation, info

    # every time the agent takes an action (steering throttle brake) a new observation gets generated
    # based on the new observation the reward increases or decreases
    def step(self, action):
        self.current_step += 1
        self.total_steps += 1

        action = np.asarray(action, dtype=np.float32)

        if action.shape != (3,):
            raise ValueError(f"Expected action shape (3,), got {action.shape}")

        action = np.nan_to_num(action, nan=0.0, posinf=1.0, neginf=-1.0)

        action = np.clip(action, self.action_space.low, self.action_space.high)
        old_action = self.previous_action.copy()
        old_progress = self.previous_progress
        self.state = self.controller.step(action.tolist(), dt=self.dt)

        nearest = self._get_nearest_track_point()
        new_progress = float(nearest["progress"])
        progress_delta = self._calculate_progress_delta(old_progress, new_progress)

        self.current_progress = new_progress

        observation = self._get_observation()

        reward = self._calculate_reward(progress_delta, nearest, old_action, action)
        terminated = False
        truncated = False
        within_track = self._is_within_track()

        if not within_track:

            reward -= 100.0
            terminated = True

        if (not terminated and progress_delta > 0.0 and old_progress > self.track_length * 0.9 and new_progress < self.track_length * 0.1):

            reward += 200.0
            terminated = True

        if self.current_step >= self.max_episode_steps:
            truncated = True

        self.previous_progress = new_progress
        self.previous_action = action.copy()

        info = {
            "progress": self.current_progress,
            "progress_delta": progress_delta,
            "speed": float(self.state["speed"]),
            "lateral_speed": float(self.state["lateral_speed"]),
            "lateral_offset": float(nearest["lateral_offset"]),
            "within_track": within_track,
            "step": self.current_step,
            "episode_steps": self.current_step
        }

        return (observation, float(reward), terminated, truncated, info)

    def close(self):
        pass
