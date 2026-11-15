import math
import numpy as np

class TrackLimits:
    def __init__(
        self,
        track,
        dt=1.0 / 120.0,
        ground_margin=150.0,
        grace_period_one_tire=2.5,
        grace_period_multiple_tires=2.0,
        tire_width=0.30,
        tire_length=0.35,
        front_axle_distance=1.65,
        rear_axle_distance=1.95,
        tire_track_width=1.60,
        one_tire_penalty_rate=0.15,
        multiple_tires_penalty_rate=0.50,
        four_tires_penalty_rate=1.00,
    ):
        self.track = track
        self.dt = float(dt)
        self.ground_margin = float(ground_margin)

        self.grace_period_one_tire = float(grace_period_one_tire)
        self.grace_period_multiple_tires = float(grace_period_multiple_tires)

        self.tire_width = float(tire_width)
        self.tire_length = float(tire_length)

        self.front_axle_distance = float(front_axle_distance)
        self.rear_axle_distance = float(rear_axle_distance)
        self.tire_track_width = float(tire_track_width)

        self.one_tire_penalty_rate = float(one_tire_penalty_rate)
        self.multiple_tires_penalty_rate = float(multiple_tires_penalty_rate)
        self.four_tires_penalty_rate = float(four_tires_penalty_rate)

        if self.dt <= 0.0:
            raise ValueError("dt must be positive.")

        if self.ground_margin < 0.0:
            raise ValueError("ground_margin cannot be negative.")

        if self.tire_width <= 0.0 or self.tire_length <= 0.0:
            raise ValueError("Tire dimensions must be positive.")

        if self.front_axle_distance <= 0.0:
            raise ValueError("front_axle_distance must be positive.")

        if self.rear_axle_distance <= 0.0:
            raise ValueError("rear_axle_distance must be positive.")

        if self.tire_track_width <= 0.0:
            raise ValueError("tire_track_width must be positive.")

        if self.grace_period_one_tire < 0.0:
            raise ValueError("One-tire grace period cannot be negative.")

        if self.grace_period_multiple_tires < 0.0:
            raise ValueError("Multiple-tire grace period cannot be negative.")

        if min(
            self.one_tire_penalty_rate,
            self.multiple_tires_penalty_rate,
            self.four_tires_penalty_rate,
        ) < 0.0:
            raise ValueError("Penalty rates cannot be negative.")

        self._calculate_world_bounds()

        self.reset()

    def _calculate_world_bounds(self):
        centerline = np.asarray(self.track.centerline, dtype=float)

        if centerline.ndim != 2 or centerline.shape[1] < 2:
            raise ValueError("track.centerline must have shape (N, 2) or (N, 3).")

        if len(centerline) == 0:
            raise ValueError("Track centerline cannot be empty.")

        xy = centerline[:, :2]

        if not np.all(np.isfinite(xy)):
            raise ValueError("Track centerline contains invalid coordinates.")

        min_x = float(np.min(xy[:, 0]))
        max_x = float(np.max(xy[:, 0]))

        min_y = float(np.min(xy[:, 1]))
        max_y = float(np.max(xy[:, 1]))

        margin = self.ground_margin

        self.world_min = np.array([min_x - margin, min_y - margin,], dtype=float)

        self.world_max = np.array([max_x + margin, max_y + margin,], dtype=float)

    def reset(self):
        self.violation_durations = {"one_tire": 0.0, "multiple_tires": 0.0, "four_tires": 0.0,}

        self.violation_duration = 0.0

    def _get_tire_positions(self, state):
        x = float(state["x"])
        y = float(state["y"])
        heading = float(state["heading"])

        steering_angle = float(state.get("steering_angle", 0.0))

        if not all(math.isfinite(value) for value in (x, y, heading, steering_angle)):
            raise ValueError("Vehicle state contains invalid coordinates.")

        center = np.array([x, y], dtype=float)
        body_forward = np.array([math.cos(heading), math.sin(heading),])
        body_lateral = np.array([-math.sin(heading), math.cos(heading),])

        axle_data = {
            "front_left": (self.front_axle_distance, 1.0, steering_angle,),
            "front_right": (self.front_axle_distance, -1.0, steering_angle,),
            "rear_left": (-self.rear_axle_distance, 1.0, 0.0,),
            "rear_right": ( -self.rear_axle_distance, -1.0, 0.0,),
        }

        positions = {}

        half_length = self.tire_length / 2.0
        half_width = self.tire_width / 2.0

        for name, (longitudinal_offset, lateral_direction, wheel_steering,) in axle_data.items():

            wheel_heading = heading + wheel_steering

            wheel_forward = np.array([
                math.cos(wheel_heading),
                math.sin(wheel_heading),
            ])

            wheel_lateral = np.array([
                -math.sin(wheel_heading),
                math.cos(wheel_heading),
            ])

            wheel_center = (
                center
                + body_forward * longitudinal_offset
                + body_lateral
                * lateral_direction
                * (self.tire_track_width / 2.0)
            )

            corners = [
                wheel_center
                + wheel_forward * half_length
                + wheel_lateral * half_width,

                wheel_center
                + wheel_forward * half_length
                - wheel_lateral * half_width,

                wheel_center
                - wheel_forward * half_length
                + wheel_lateral * half_width,

                wheel_center
                - wheel_forward * half_length
                - wheel_lateral * half_width,
            ]

            positions[name] = {"center": wheel_center, "corners": corners,}

        return positions

    def _is_point_inside_world(self, position):
        position = np.asarray(position, dtype=float)
        return bool(self.world_min[0] <= position[0] <= self.world_max[0] and self.world_min[1] <= position[1] <= self.world_max[1])

    def _is_point_on_track(self, position):
        return bool(self.track.is_within_track(position))

    def _check_world_boundaries(self, tire_positions):
        for tire in tire_positions.values():
            for corner in tire["corners"]:
                if not self._is_point_inside_world(corner):
                    return False

        return True

    def _check_track_limits(self, tire_positions):
        tires_outside = []

        for name, tire in tire_positions.items():

            tire_inside = all(
                self._is_point_on_track(corner)
                for corner in tire["corners"]
            )

            if not tire_inside:
                tires_outside.append(name)

        return tires_outside

    def _calculate_penalty(self, tires_outside, dt):
        tire_count = len(tires_outside)

        if tire_count == 0:
            self.reset()

            return 0.0, None

        if tire_count == 1:
            category = "one_tire"
            grace_period = self.grace_period_one_tire
            penalty_rate = self.one_tire_penalty_rate

        elif tire_count == 4:
            category = "four_tires"
            grace_period = self.grace_period_multiple_tires
            penalty_rate = self.four_tires_penalty_rate

        else:
            category = "multiple_tires"
            grace_period = self.grace_period_multiple_tires
            penalty_rate = self.multiple_tires_penalty_rate

        # Reset timers belonging to inactive categories.
        for name in self.violation_durations:
            if name != category:
                self.violation_durations[name] = 0.0

        # Accumulate continuous violation time.
        self.violation_durations[category] += dt

        self.violation_duration = (self.violation_durations[category])

        if self.violation_duration >= grace_period:
            penalty = penalty_rate * dt
        else:
            penalty = 0.0

        return float(penalty), category

    def update(self, state, dt=None):
        if dt is None:
            dt = self.dt

        dt = float(dt)

        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be a positive finite number.")

        tire_positions = self._get_tire_positions(state)

        # 1. Check whether the vehicle has left the entire world.
        on_world = self._check_world_boundaries(tire_positions)

        if not on_world:
            return {
                "tires_outside": [],
                "tire_count_outside": 0,
                "violation_duration": float(self.violation_duration),
                "violation_durations": dict(self.violation_durations),
                "violation_category": None,
                "penalty": 0.0,
                "map_exited": True,
                "within_track": False,
                "terminated": True,
            }

        # 2. Check the four tires against the track boundaries.
        tires_outside = self._check_track_limits(tire_positions)

        tire_count = len(tires_outside)

        # 3. Calculate the penalty.
        penalty, category = self._calculate_penalty(tires_outside, dt,)

        within_track = tire_count == 0

        return {
            "tires_outside": tires_outside,
            "tire_count_outside": tire_count,
            "violation_duration": float(self.violation_duration),
            "violation_durations": dict(self.violation_durations),
            "violation_category": category,
            "penalty": float(penalty),
            "map_exited": False,
            "within_track": within_track,
            "terminated": False,
        }
