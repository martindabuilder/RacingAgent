
import math


class RewardSystem:
    def __init__(
        self,
        dt=1.0 / 120.0,
        max_speed=90.0,
        max_lateral_acceleration=16.0,
        min_target_speed=15.0,
        braking_deceleration=12.0,
        stall_speed=0.5,
        stall_time=2.0,
    ):
        self.dt = float(dt)
        self.max_speed = float(max_speed)
        self.max_lateral_acceleration = float(max_lateral_acceleration)
        self.min_target_speed = float(min_target_speed)
        self.braking_deceleration = float(braking_deceleration)

        self.stall_speed = float(stall_speed)
        self.stall_time = float(stall_time)

        self.stopped_duration = 0.0
        self.previous_speed = None

        if self.dt <= 0.0:
            raise ValueError("dt must be positive.")

        if self.max_speed <= 0.0:
            raise ValueError("max_speed must be positive.")

        if self.max_lateral_acceleration <= 0.0:
            raise ValueError("max_lateral_acceleration must be positive.")

        if self.min_target_speed <= 0.0:
            raise ValueError("min_target_speed must be positive.")

        if self.braking_deceleration <= 0.0:
            raise ValueError("braking_deceleration must be positive.")

    def reset(self):
        self.stopped_duration = 0.0
        self.previous_speed = None

    def _calculate_target_speed(self, curvature):
        curvature = abs(float(curvature))
        epsilon = 1e-4

        target_speed = math.sqrt(
            self.max_lateral_acceleration
            / max(curvature, epsilon)
        )

        target_speed = min(target_speed, self.max_speed)
        target_speed = max(target_speed, self.min_target_speed)

        return target_speed

    def _calculate_safe_speed(
        self,
        curvature,
        future_curvatures=None,
    ):
        target_speed = self._calculate_target_speed(curvature)

        if future_curvatures is None:
            return target_speed

        lookahead_distances = [
            5.0,
            10.0,
            20.0,
            40.0
        ]

        for distance, future_curvature in zip(
            lookahead_distances,
            future_curvatures
        ):
            future_target_speed = self._calculate_target_speed(
                future_curvature
            )

            braking_speed = math.sqrt(
                future_target_speed ** 2
                + 2.0 * self.braking_deceleration * distance
            )

            target_speed = min(
                target_speed,
                braking_speed
            )

        return max(
            target_speed,
            self.min_target_speed
        )

    @staticmethod
    def _calculate_heading_error(state, nearest):
        vehicle_heading = float(state["heading"])
        track_heading = float(nearest["heading_rad"])

        difference = vehicle_heading - track_heading

        return math.atan2(
            math.sin(difference),
            math.cos(difference)
        )

    def calculate(
        self,
        state,
        progress_delta,
        nearest,
        previous_action,
        current_action,
        within_track,
        future_curvatures=None,
    ):
        speed = max(0.0, float(state["speed"]))
        lateral_speed = abs(float(state["lateral_speed"]))

        curvature = float(nearest["curvature"])

        # 1. Reward forward progress.
        progress_reward = float(progress_delta) * 0.12

        # 2. Calculate safe speed using current and future curvature.
        target_speed = self._calculate_safe_speed(
            curvature,
            future_curvatures
        )

        speed_ratio = min(
            speed / max(target_speed, 1e-6),
            1.0
        )

        speed_reward = 0.10 * speed_ratio

        # 3. Penalize driving faster than the safe speed.
        overspeed_ratio = max(
            0.0,
            (speed - target_speed) / max(target_speed, 1e-6)
        )

        overspeed_penalty = min(
            overspeed_ratio ** 2 * 0.08,
            0.25
        )

        # 4. Braking rewards and unnecessary braking penalty.
        brake_action = max(
            0.0,
            min(float(current_action[2]), 1.0)
        )

        braking_reward = 0.0
        corner_entry_reward = 0.0
        unnecessary_braking_penalty = 0.0

        safe_speed_limit = target_speed * 1.05

        if self.previous_speed is not None and within_track:
            speed_reduction = self.previous_speed - speed

            actual_deceleration = max(
                0.0,
                speed_reduction / self.dt
            )

            braking_needed = speed > safe_speed_limit

            if brake_action > 0.1:
                if braking_needed and speed_reduction > 0.005:

                    braking_effectiveness = min(
                        actual_deceleration
                        / self.braking_deceleration,
                        1.0
                    )

                    overspeed_factor = min(
                        overspeed_ratio / 0.25,
                        1.0
                    )

                    braking_reward = (
                        0.0015
                        * brake_action
                        * braking_effectiveness
                        * overspeed_factor
                    )

                    if (
                        self.previous_speed > safe_speed_limit
                        and speed <= safe_speed_limit
                    ):
                        corner_entry_reward = 0.01

                elif not braking_needed:
                    unnecessary_braking_penalty = (
                        0.0025 * brake_action
                    )

        # 5. Penalize excessive lateral movement.
        lateral_penalty = min(
            lateral_speed / 25.0,
            2.0
        ) * 0.04

        # 6. Penalize large heading errors.
        heading_error = abs(
            self._calculate_heading_error(state, nearest)
        )

        heading_penalty = (
            heading_error / math.pi
        ) * 0.015

        # 7. Penalize sudden steering changes.
        steering_change = abs(
            float(current_action[0])
            - float(previous_action[0])
        )

        steering_penalty = steering_change * 0.005

        # 8. Detect prolonged stopping.
        stall_penalty = 0.0

        if speed < self.stall_speed:
            self.stopped_duration += self.dt

            if self.stopped_duration >= self.stall_time:
                stall_penalty = 0.10
        else:
            self.stopped_duration = 0.0

        # 9. Do not reward driving outside the track.
        if not within_track:
            progress_reward = 0.0
            speed_reward = 0.0
            braking_reward = 0.0
            corner_entry_reward = 0.0
            unnecessary_braking_penalty = 0.0

        total_reward = (
            progress_reward
            + speed_reward
            + braking_reward
            + corner_entry_reward
            - overspeed_penalty
            - lateral_penalty
            - heading_penalty
            - steering_penalty
            - stall_penalty
            - unnecessary_braking_penalty
        )

        reward_info = {
            "progress_reward": float(progress_reward),
            "speed_reward": float(speed_reward),
            "target_speed": float(target_speed),
            "speed_ratio": float(speed_ratio),
            "overspeed_penalty": float(-overspeed_penalty),
            "braking_reward": float(braking_reward),
            "corner_entry_reward": float(corner_entry_reward),
            "unnecessary_braking_penalty": float(
                -unnecessary_braking_penalty
            ),
            "lateral_penalty": float(-lateral_penalty),
            "heading_penalty": float(-heading_penalty),
            "steering_penalty": float(-steering_penalty),
            "stall_penalty": float(-stall_penalty),
            "heading_error": float(heading_error),
            "stopped_duration": float(self.stopped_duration),
            "total_reward": float(total_reward),
        }

        self.previous_speed = speed

        return float(total_reward), reward_info
