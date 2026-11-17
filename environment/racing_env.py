
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from simulation.vehicle_controller import VehicleController
from environment.track_loader import TrackLoader
from environment.reward_system import RewardSystem
from environment.track_limits import TrackLimits
from environment.lap_completion import LapCompletion


class RacingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        dt=1.0 / 120.0,
        max_episode_steps=12000,
        start_speed=0.0,
        start_index=1,
        track=None,
        track_name="yas_marina",
        track_names=None,
    ):
        super().__init__()

        self.dt = float(dt)
        self.max_episode_steps = int(max_episode_steps)

        self.start_speed = float(start_speed)
        self.requested_start_index = int(start_index)

        self.current_step = 0
        self.total_steps = 0

        # Load and cache the available tracks.
        self.track_cache = {}

        if track is not None:
            if track_names is not None:
                raise ValueError("Use either track or track_names, not both.")

            self.track_cache[track_name] = track
            self.track_names = [track_name]

        else:
            if track_names is None:
                track_names = [track_name]

            self.track_names = list(track_names)

            if not self.track_names:
                raise ValueError("At least one track must be provided.")

            if len(self.track_names) != len(set(self.track_names)):
                raise ValueError("Track names must be unique.")

            for name in self.track_names:
                self.track_cache[name] = TrackLoader.load(name)

        self.randomize_tracks = len(self.track_names) > 1

        self._set_track(self.track_names[0])
        self.controller = VehicleController()
        self.reward_system = RewardSystem(dt=self.dt)
        self.state = None
        self.previous_action = np.zeros(3, dtype=np.float32)
        self.previous_progress = 0.0
        self.current_progress = 0.0

        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(19,),
            dtype=np.float32
        )

    # Configure the selected track.
    def _set_track(self, track_name):
        if track_name not in self.track_cache:
            raise ValueError(
                f"Track '{track_name}' is not loaded. Available tracks: {self.track_names}"
            )

        self.track_name = track_name
        self.track = self.track_cache[track_name]

        self.track_points = np.asarray(
            self.track.centerline,
            dtype=np.float64
        )

        self.track_length = float(
            self.track.get_track_length()
        )

        self.num_track_points = len(self.track_points)

        if self.num_track_points < 2:
            raise ValueError(
                "Track must contain at least two points."
            )

        self.start_index = (
            self.requested_start_index % self.num_track_points
        )

        # Close the centerline by connecting the last point
        # to the first point.
        closed_points = np.vstack([
            self.track_points,
            self.track_points[0]
        ])

        segment_vectors = np.diff(
            closed_points,
            axis=0
        )

        self.centerline_segment_lengths = np.linalg.norm(
            segment_vectors,
            axis=1
        )

        self.centerline_cumulative = np.concatenate([
            [0.0],
            np.cumsum(self.centerline_segment_lengths)
        ])

        self.centerline_length = float(
            self.centerline_cumulative[-1]
        )

        if self.centerline_length <= 0.0:
            raise ValueError(
                "Centerline length must be positive."
            )

        # Initialize lap completion.
        start_position = self.track_points[self.start_index]

        start_nearest = self.track.get_nearest_point(
            start_position
        )

        self.lap_completion = LapCompletion(
            track_length=self.track_length,
            start_progress=float(start_nearest["progress"]),
        )

        # Initialize track limits.
        self.track_limits = TrackLimits(
            track=self.track,
            dt=self.dt,
            ground_margin=150.0
        )

    def _get_position(self):
        return np.array(
            [self.state["x"], self.state["y"]],
            dtype=np.float64
        )

    # Find the nearest point on the track.
    def _get_nearest_track_point(self):
        position = self._get_position()

        return self.track.get_nearest_point(position)

    def _get_centerline_point_at_progress(self, progress):
        progress = float(progress)
        progress %= self.centerline_length

        index = (
            np.searchsorted(
                self.centerline_cumulative,
                progress,
                side="right"
            ) - 1
        )

        index = min(
            max(index, 0),
            self.num_track_points - 1
        )

        segment_start = self.centerline_cumulative[index]
        segment_length = self.centerline_segment_lengths[index]

        if segment_length <= 1e-9:
            return self.track_points[index].copy()

        fraction = (
            (progress - segment_start) / segment_length
        )

        next_index = (
            index + 1
        ) % self.num_track_points

        point = (
            self.track_points[index]
            + fraction * (
                self.track_points[next_index]
                - self.track_points[index]
            )
        )

        return point

    def _get_curvature_at_progress(self, progress):
        progress = float(progress)
        progress %= self.track_length

        index = (
            np.searchsorted(
                self.track.distance,
                progress,
                side="right"
            ) - 1
        )

        index = min(
            max(index, 0),
            self.num_track_points - 1
        )

        next_index = (
            index + 1
        ) % self.num_track_points

        segment_start = float(
            self.track.distance[index]
        )

        segment_length = float(
            self.track.segment_lengths[index]
        )

        if segment_length <= 1e-9:
            return float(
                self.track.curvature[index]
            )

        fraction = (
            (progress - segment_start) / segment_length
        )

        fraction = float(
            np.clip(fraction, 0.0, 1.0)
        )

        curvature = (
            self.track.curvature[index] * (1.0 - fraction)
            + self.track.curvature[next_index] * fraction
        )

        return float(curvature)

    def _get_future_curvatures(self, nearest):
        lookahead_distances = [
            5.0,
            10.0,
            20.0,
            40.0
        ]

        future_curvatures = []

        for distance in lookahead_distances:
            target_progress = (
                float(nearest["progress"]) + distance
            )

            curvature = self._get_curvature_at_progress(
                target_progress
            )

            future_curvatures.append(
                float(curvature)
            )

        return future_curvatures

    def _get_relative_lookahead(
        self,
        nearest,
        lookahead_distance
    ):
        target_progress = (
            float(nearest["progress"])
            + float(lookahead_distance)
        )

        target_point = self._get_centerline_point_at_progress(
            target_progress
        )

        vehicle_position = self._get_position()

        dx = target_point[0] - vehicle_position[0]
        dy = target_point[1] - vehicle_position[1]

        heading = float(self.state["heading"])

        forward = (
            np.cos(heading) * dx
            + np.sin(heading) * dy
        )

        lateral = (
            -np.sin(heading) * dx
            + np.cos(heading) * dy
        )

        return float(forward), float(lateral)

    # Combine the 19 observation parameters.
    def _get_observation(self):
        nearest = self._get_nearest_track_point()

        speed = float(self.state["speed"])
        lateral_speed = float(self.state["lateral_speed"])
        yaw_rate = float(self.state["yaw_rate"])
        acceleration = float(self.state["acceleration"])
        lateral_acceleration = float(
            self.state["lateral_acceleration"]
        )
        steering_angle = float(self.state["steering_angle"])

        lateral_offset = float(nearest["lateral_offset"])

        vehicle_heading = float(self.state["heading"])
        track_heading = float(nearest["heading_rad"])

        heading_error = (
            vehicle_heading - track_heading + np.pi
        ) % (2.0 * np.pi) - np.pi

        width_left = float(nearest["width_left"])
        width_right = float(nearest["width_right"])

        lookahead_distances = [
            5.0,
            10.0,
            20.0,
            40.0
        ]

        lookahead_values = []

        for distance in lookahead_distances:
            forward, lateral = self._get_relative_lookahead(
                nearest,
                distance
            )

            normalized_forward = forward / distance

            normalized_lateral = (
                lateral / max(width_left, width_right, 1.0)
            )

            lookahead_values.extend([
                normalized_forward,
                normalized_lateral
            ])

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

                *lookahead_values,

                curvature / 0.2,
                width_left / 20.0,
                width_right / 20.0
            ],
            dtype=np.float32
        )

        observation = np.nan_to_num(
            observation,
            nan=0.0,
            posinf=1.0,
            neginf=-1.0
        )

        observation = np.clip(
            observation,
            -1.0,
            1.0
        )

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
        tire_positions = (
            self.track_limits._get_tire_positions(self.state)
        )

        tires_outside = (
            self.track_limits._check_track_limits(tire_positions)
        )

        return len(tires_outside) == 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        options = options or {}

        # Select a track for the new episode.
        requested_track = options.get("track_name")

        if requested_track is not None:
            if requested_track not in self.track_cache:
                raise ValueError(
                    f"Track '{requested_track}' is not available. Available tracks: {self.track_names}"
                )

            selected_track = requested_track

        elif self.randomize_tracks:
            selected_track = self.np_random.choice(
                self.track_names
            )

        else:
            selected_track = self.track_names[0]

        # Update track-dependent systems if necessary.
        if selected_track != self.track_name:
            self._set_track(selected_track)

        self.current_step = 0

        self.previous_action = np.zeros(
            3,
            dtype=np.float32
        )

        self.reward_system.reset()
        self.track_limits.reset()
        self.lap_completion.reset()

        start_position = self.track_points[self.start_index]

        next_index = (
            self.start_index + 1
        ) % self.num_track_points

        direction = (
            self.track_points[next_index]
            - start_position
        )

        heading = float(
            np.arctan2(direction[1], direction[0])
        )

        self.state = self.controller.reset(
            x=float(start_position[0]),
            y=float(start_position[1]),
            heading=heading,
            speed=self.start_speed
        )

        nearest = self._get_nearest_track_point()

        self.current_progress = float(
            nearest["progress"]
        )

        self.previous_progress = self.current_progress

        observation = self._get_observation()

        info = {
            "track_name": self.track_name,
            "progress": self.current_progress,
            "speed": float(self.state["speed"]),
            "lateral_offset": float(nearest["lateral_offset"]),
            "observation_size": len(observation)
        }

        return observation, info

    # Execute one agent action.
    def step(self, action):
        self.current_step += 1
        self.total_steps += 1

        action = np.asarray(
            action,
            dtype=np.float32
        )

        if action.shape != (3,):
            raise ValueError(
                f"Expected action shape (3,), got {action.shape}"
            )

        action = np.nan_to_num(
            action,
            nan=0.0,
            posinf=1.0,
            neginf=-1.0
        )

        action = np.clip(
            action,
            self.action_space.low,
            self.action_space.high
        )

        old_action = self.previous_action.copy()
        old_progress = self.previous_progress

        self.state = self.controller.step(
            action.tolist(),
            dt=self.dt
        )

        nearest = self._get_nearest_track_point()

        new_progress = float(
            nearest["progress"]
        )

        progress_delta = self._calculate_progress_delta(
            old_progress,
            new_progress
        )

        self.current_progress = new_progress

        observation = self._get_observation()

        # Calculate curvature ahead of the vehicle.
        future_curvatures = self._get_future_curvatures(
            nearest
        )

        terminated = False
        truncated = False

        # Check all four tires and the ground boundaries.
        limits_info = self.track_limits.update(
            self.state,
            dt=self.dt
        )

        within_track = limits_info["within_track"]

        reward, reward_info = self.reward_system.calculate(
            state=self.state,
            progress_delta=progress_delta,
            nearest=nearest,
            previous_action=old_action,
            current_action=action,
            within_track=within_track,
            future_curvatures=future_curvatures,
        )

        # Apply track limits penalty.
        track_limit_penalty = float(
            limits_info["penalty"]
        )

        reward -= track_limit_penalty

        # Leaving the ground plane terminates the episode.
        map_exit_penalty = 0.0

        if limits_info["map_exited"]:
            map_exit_penalty = 100.0
            reward -= map_exit_penalty
            terminated = True

        lap_completed = self.lap_completion.update(
            previous_progress=old_progress,
            current_progress=new_progress,
            progress_delta=progress_delta,
        )

        # Reward successful lap completion.
        lap_completion_reward = 0.0

        if not terminated and lap_completed:
            lap_completion_reward = 200.0
            reward += lap_completion_reward
            terminated = True

        if self.current_step >= self.max_episode_steps:
            truncated = True

        self.previous_progress = new_progress
        self.previous_action = action.copy()

        # Store the complete reward breakdown.
        reward_info["track_limit_penalty"] = float(
            -track_limit_penalty
        )

        reward_info["map_exit_penalty"] = float(
            -map_exit_penalty
        )

        reward_info["lap_completion_reward"] = float(
            lap_completion_reward
        )

        reward_info["total_reward"] = float(reward)

        info = {
            "track_name": self.track_name,
            "progress": self.current_progress,
            "progress_delta": progress_delta,
            "speed": float(self.state["speed"]),
            "lateral_speed": float(self.state["lateral_speed"]),
            "lateral_offset": float(nearest["lateral_offset"]),
            "within_track": within_track,
            "tires_outside": limits_info["tires_outside"],
            "tire_count_outside": limits_info["tire_count_outside"],
            "violation_duration": limits_info["violation_duration"],
            "violation_category": limits_info["violation_category"],
            "track_limit_penalty": limits_info["penalty"],
            "map_exited": limits_info["map_exited"],
            "step": self.current_step,
            "episode_steps": self.current_step,
            "reward_components": reward_info,
            "environment_total_reward": float(reward),
            "lap_completed": lap_completed,
            "lap_count": self.lap_completion.get_lap_count(),
            "future_curvatures": future_curvatures,
        }

        return (
            observation,
            float(reward),
            terminated,
            truncated,
            info
        )

    def close(self):
        pass
