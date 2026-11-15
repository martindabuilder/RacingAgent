import math
import numpy as np

class VehiclePhysics:
    def __init__(self):
        self.mass = 800.0
        self.yaw_inertia = 1400.0

        self.wheelbase = 3.60
        self.front_axle_distance = 1.65
        self.rear_axle_distance = (self.wheelbase - self.front_axle_distance)

        self.max_engine_force = 15000.0
        self.max_engine_power = 750000.0
        self.max_brake_force = 26000.0

        self.max_steering_angle = 0.55
        self.max_steering_rate = 2.5
        self.steering_time_constant = 0.08

        self.friction_coefficient = 1.8
        self.front_cornering_stiffness = 120000.0
        self.rear_cornering_stiffness = 140000.0

        self.air_density = 1.225
        self.drag_coefficient_area = 1.0
        self.downforce_coefficient = 4.0

        self.front_weight_distribution = 0.45
        self.rear_weight_distribution = 0.55

        self.rolling_resistance = 0.015

        self.maximum_speed = 100.0
        self.maximum_slip_angle = 0.5
        self.maximum_time_step = 0.01

        self.reset()

    def reset(self, x=0.0, y=0.0, heading=0.0, speed=0.0):
    
        self.x = float(x)
        self.y = float(y)

        self.heading = float(heading)

        self.vx = max(0.0, float(speed))
        self.vy = 0.0

        self.yaw_rate = 0.0

        self.steering_angle = 0.0

        self.acceleration = 0.0
        self.lateral_acceleration = 0.0

        self.last_steering_command = 0.0
        self.last_throttle = 0.0
        self.last_brake = 0.0

        self.time = 0.0

        return self.get_state()

    @staticmethod
    def _clip(value, minimum, maximum):
        return max(minimum, min(maximum, value))

    @staticmethod
    def _wrap_angle(angle):
        return math.atan2(math.sin(angle), math.cos(angle))

    def _calculate_aerodynamic_forces(self):
        speed = math.sqrt(self.vx ** 2 + self.vy ** 2)
        drag = (0.5 * self.air_density * self.drag_coefficient_area * speed ** 2)
        downforce = (self.downforce_coefficient * speed ** 2)

        return drag, downforce

    def _calculate_tire_forces(self, steering, downforce):
        speed_for_slip = max(abs(self.vx), 1.0)
        front_slip = math.atan2(self.vy + self.front_axle_distance * self.yaw_rate, speed_for_slip) - steering
        rear_slip = math.atan2(self.vy - self.rear_axle_distance * self.yaw_rate, speed_for_slip)
        front_slip = self._clip(front_slip, -self.maximum_slip_angle, self.maximum_slip_angle)
        rear_slip = self._clip(rear_slip, -self.maximum_slip_angle, self.maximum_slip_angle)
        normal_force = (self.mass * 9.81 + downforce)
        front_normal = (normal_force * self.front_weight_distribution)
        rear_normal = (normal_force * self.rear_weight_distribution)
        front_force = (-self.front_cornering_stiffness * front_slip)
        rear_force = (-self.rear_cornering_stiffness * rear_slip)
        front_limit = (self.friction_coefficient * front_normal)
        rear_limit = (self.friction_coefficient * rear_normal)
        front_force = self._clip(front_force, -front_limit, front_limit)
        rear_force = self._clip(rear_force, -rear_limit, rear_limit)

        return (front_force, rear_force, front_normal, rear_normal)

    def _update_steering(self, steering_command, dt):
        target_angle = (steering_command * self.max_steering_angle)
        difference = (target_angle - self.steering_angle)
        maximum_change = (self.max_steering_rate * dt)
        difference = self._clip(difference, -maximum_change, maximum_change)

        self.steering_angle += difference

    def _physics_step(self, steering, throttle, brake, dt):
        self._update_steering(steering, dt)
        drag, downforce = (self._calculate_aerodynamic_forces())

        (
            front_lateral_force,
            rear_lateral_force,
            front_normal,
            rear_normal
        ) = self._calculate_tire_forces(
            self.steering_angle,
            downforce
        )

        available_engine_force = min(self.max_engine_force, self.max_engine_power / max(self.vx, 5.0))
        engine_force = (throttle * available_engine_force)
        rear_traction_limit = (self.friction_coefficient * rear_normal)
        engine_force = min(engine_force, rear_traction_limit)
        braking_force = (brake * self.max_brake_force)
        rolling_force = (self.mass * 9.81 * self.rolling_resistance)

        if self.vx > 0.1:
            longitudinal_resistance = (rolling_force + drag)
        else:
            longitudinal_resistance = (min(rolling_force + drag, 0.0))

        longitudinal_acceleration = (
            engine_force
            - braking_force
            - longitudinal_resistance
            - front_lateral_force
            * math.sin(self.steering_angle)
            + self.mass * self.vy * self.yaw_rate
        ) / self.mass

        lateral_acceleration = (
            rear_lateral_force
            + front_lateral_force
            * math.cos(self.steering_angle)
            - self.mass * self.vx * self.yaw_rate
        ) / self.mass

        yaw_acceleration = (
            self.front_axle_distance
            * front_lateral_force
            * math.cos(self.steering_angle)
            - self.rear_axle_distance
            * rear_lateral_force
        ) / self.yaw_inertia

        self.vx += longitudinal_acceleration * dt
        self.vy += lateral_acceleration * dt
        self.yaw_rate += yaw_acceleration * dt

        if self.vx < 0.0:
            self.vx = 0.0

        self.vx = min(self.vx, self.maximum_speed)

        if self.vx < 0.5:
            self.vy *= math.exp(-4.0 * dt)
            self.yaw_rate *= math.exp(-3.0 * dt)

        world_velocity_x = (self.vx * math.cos(self.heading) - self.vy * math.sin(self.heading))
        world_velocity_y = (self.vx * math.sin(self.heading) + self.vy * math.cos(self.heading))

        self.x += world_velocity_x * dt
        self.y += world_velocity_y * dt

        self.heading = self._wrap_angle(self.heading + self.yaw_rate * dt)

        self.acceleration = longitudinal_acceleration
        self.lateral_acceleration = lateral_acceleration

        self.time += dt

    def step(self, steering, throttle, brake, dt=1.0 / 120.0):
        dt = float(dt)

        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError("Time step must be a positive finite number.")

        if dt > 0.5:
            raise ValueError("Time step is too large.")

        steering = float(steering)
        throttle = float(throttle)
        brake = float(brake)

        if not all(math.isfinite(value) for value in (steering, throttle, brake)):
            raise ValueError("Vehicle actions must be finite numbers.")

        steering = self._clip(steering, -1.0, 1.0)
        throttle = self._clip(throttle, 0.0, 1.0)
        brake = self._clip(brake, 0.0, 1.0)

        if brake > 0.0:
            throttle = 0.0

        self.last_steering_command = steering
        self.last_throttle = throttle
        self.last_brake = brake

        substeps = max(1, math.ceil(dt / self.maximum_time_step))
        substep_dt = dt / substeps

        for _ in range(substeps):
            self._physics_step(steering, throttle, brake, substep_dt)

        return self.get_state()

    def get_state(self):
        speed = math.sqrt(self.vx ** 2 + self.vy ** 2)

        return {
            "x": float(self.x),
            "y": float(self.y),
            "heading": float(self.heading),
            "heading_deg": float(math.degrees(self.heading)),
            "speed": float(speed),
            "longitudinal_speed": float(self.vx),
            "lateral_speed": float(self.vy),
            "yaw_rate": float(self.yaw_rate),
            "steering_angle": float(self.steering_angle),
            "acceleration": float(self.acceleration),
            "lateral_acceleration": float(self.lateral_acceleration),
            "time": float(self.time)
        }
