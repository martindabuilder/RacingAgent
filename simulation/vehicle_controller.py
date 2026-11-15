import math
from simulation.vehicle_physics import VehiclePhysics

class VehicleController:
    def __init__(self, physics=None):
        self.physics = physics if physics is not None else VehiclePhysics()

        self.last_action = {
            "steering": 0.0,
            "throttle": 0.0,
            "brake": 0.0,
        }

    @staticmethod
    def _clip(value, minimum, maximum, name):
        value = float(value)

        if not math.isfinite(value):
            raise ValueError(f"{name} must be a finite number.")

        return max(minimum, min(maximum, value))

    def _parse_action(self, action):
        if isinstance(action, dict):
            try:
                steering = action["steering"]
                throttle = action["throttle"]
                brake = action["brake"]
            except KeyError as error:
                raise ValueError(f"Missing action component: {error}") from error

        elif isinstance(action, (list, tuple)):
            if len(action) != 3:
                raise ValueError("Action must contain exactly 3 values: [steering, throttle, brake].")

            steering, throttle, brake = action

        else:
            raise TypeError("Action must be a list, tuple, or dictionary.")

        steering = self._clip(steering, -1.0, 1.0, "steering")
        throttle = self._clip(throttle, 0.0, 1.0, "throttle")
        brake = self._clip(brake, 0.0, 1.0, "brake")

        if brake > 0.0:
            throttle = 0.0

        return {
            "steering": steering,
            "throttle": throttle,
            "brake": brake,
        }

    def reset(self, x=0.0, y=0.0, heading=0.0, speed=0.0):
        self.last_action = {
            "steering": 0.0,
            "throttle": 0.0,
            "brake": 0.0,
        }

        return self.physics.reset(x=x, y=y, heading=heading, speed=speed,)

    def step(self, action, dt=1.0 / 120.0):
        dt = float(dt)

        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be a positive finite number.")

        controls = self._parse_action(action)

        self.last_action = controls.copy()

        return self.physics.step(steering=controls["steering"], throttle=controls["throttle"], brake=controls["brake"], dt=dt,)

    def get_state(self):
        return self.physics.get_state()

    def get_last_action(self):
        return self.last_action.copy()
