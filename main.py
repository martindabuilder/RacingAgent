import time
import numpy as np
import torch
from pathlib import Path
from direct.showbase.ShowBase import ShowBase
from panda3d.core import (WindowProperties, AmbientLight, DirectionalLight, Vec3, Vec4, ClockObject)
from simulation.vehicle import F1Vehicle
from environment.track_loader import TrackLoader
from environment.racing_env import RacingEnv
from tracks.yas_marina.track_mesh import TrackMesh
from tracks.yas_marina.track_environment import TrackEnvironment
from tracks.yas_marina.track_barriers import TrackBarriers
from tracks.yas_marina.track_grandstands import TrackGrandstands
from tracks.yas_marina.track_lighting import TrackLighting
from tracks.yas_marina.track_facilities import TrackFacilities
from tracks.yas_marina.track_markings import TrackMarkings

torch.set_num_threads(1)

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = (
    PROJECT_DIR
    / "training"
    / "models"
    / "ppo_racing_agent.zip"
)

SIMULATION_DT = 1.0 / 120.0
MAX_STEPS_PER_FRAME = 2
MAX_FRAME_DT = 0.05
MAX_ACCUMULATOR = SIMULATION_DT * MAX_STEPS_PER_FRAME

CAMERA_DISTANCE = 10.0
CAMERA_HEIGHT = 3.5
CAMERA_TARGET_HEIGHT = 8.0
CAMERA_LOOK_AHEAD = 6.0


class YasMarinaSimulator(ShowBase):
    def __init__(self):
        super().__init__()

        self.setFrameRateMeter(True)

        properties = WindowProperties()
        properties.setTitle("F1 Autonomous Racing Simulator")
        properties.setSize(960, 540)

        self.win.requestProperties(properties)

        self.setBackgroundColor(
            0.008,
            0.015,
            0.045,
            1.0
        )

        ambient_light = AmbientLight("night_ambient")
        ambient_light.setColor(
            Vec4(0.16, 0.19, 0.30, 1.0)
        )

        self.ambient_light_node = (
            self.render.attachNewNode(ambient_light)
        )

        self.render.setLight(self.ambient_light_node)

        moon_light = DirectionalLight("moon_light")
        moon_light.setColor(
            Vec4(0.30, 0.34, 0.48, 1.0)
        )

        self.moon_light_node = (
            self.render.attachNewNode(moon_light)
        )

        self.moon_light_node.setHpr(35, -55, 0)
        self.render.setLight(self.moon_light_node)

        self.track = TrackLoader.load("yas_marina")

        self.track_mesh = TrackMesh(
            self.render,
            self.track
        )
        self.track_mesh.create()

        self.environment = TrackEnvironment(
            self.render,
            self.track
        )
        self.environment.create()

        self.barriers = TrackBarriers(
            self.render,
            self.track
        )
        self.barriers.create()

        self.track_markings = TrackMarkings(
            self.render,
            self.track
        )
        self.track_markings.create()

        self.grandstands = TrackGrandstands(
            self.track,
            self.render
        )
        self.grandstands.create()

        self.track_lighting = TrackLighting(
            self.render,
            self.track
        )
        self.track_lighting.create()

        self.facilities = TrackFacilities(
            self.render,
            self.track
        )
        self.facilities.create()

        self.vehicle = F1Vehicle(
            self.render,
            self.loader,
            self.track
        )

        start_position = self.track.centerline[0]

        self.vehicle.set_position(
            float(start_position[0]),
            float(start_position[1]),
            0.2
        )

        start_heading = self.track.get_heading(0)

        self.vehicle.set_rotation(start_heading)

        self.camLens.setFov(50)

        from stable_baselines3 import PPO

        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Trained model not found: {MODEL_PATH}"
            )

        print("[PPO] Loading trained model...")

        self.model = PPO.load(
            str(MODEL_PATH),
            device="cpu"
        )

        print("[PPO] Trained model loaded successfully.")

        self.env = RacingEnv(
            track=self.track,
            max_episode_steps=12000,
            start_speed=0.0
        )

        self.observation, self.info = self.env.reset()

        self.simulation_accumulator = 0.0

        self.previous_position = None
        self.current_position = None

        self.previous_heading = 0.0
        self.current_heading = 0.0

        self.total_reward = 0.0
        self.total_steps = 0
        self.completed_laps = 0

        self.update_vehicle_visuals()
        self.update_camera(1.0)

        self.taskMgr.add(
            self.update_agent,
            "update_agent"
        )

        print("[PPO] Autonomous driving started.")

    def predict_action(self, observation):

        observation = np.asarray(
            observation,
            dtype=np.float32
        )

        observation_tensor = self.model.policy.obs_to_tensor(observation)[0]

        with torch.inference_mode():
            actions = self.model.policy(observation_tensor, deterministic=True)[0]

        actions = actions.cpu().numpy()

        if self.model.policy.squash_output:
            actions = self.model.policy.unscale_action(actions)
        else:
            actions = np.clip(actions, self.model.action_space.low, self.model.action_space.high)

        return actions.reshape(self.model.action_space.shape)

    def get_vehicle_state(self):

        state = self.env.controller.get_state()

        return (
            float(state["x"]),
            float(state["y"]),
            float(state["heading_deg"])
        )

    def update_interpolation_state(self):

        position_x, position_y, heading = self.get_vehicle_state()

        self.previous_position = (
            position_x,
            position_y
        )

        self.current_position = (
            position_x,
            position_y
        )

        self.previous_heading = heading
        self.current_heading = heading

    def update_vehicle_visuals(self, alpha=1.0):
        if self.previous_position is None:
            self.update_interpolation_state()

        alpha = float(np.clip(alpha, 0.0, 1.0))

        x = (
            self.previous_position[0]
            + (self.current_position[0] - self.previous_position[0]) * alpha
        )

        y = (
            self.previous_position[1]
            + (self.current_position[1] - self.previous_position[1]) * alpha
        )

        heading_difference = ((self.current_heading - self.previous_heading + 180.0) % 360.0) - 180.0

        heading = (self.previous_heading + heading_difference * alpha)
        self.vehicle.set_position(x, y, 0.2)

        self.vehicle.set_rotation(heading)
        state = self.env.controller.get_state()
        self.vehicle.set_speed(state["speed"])

    def update_camera(self, dt):

        car_position = self.vehicle.car.getPos(self.render)

        forward = self.vehicle.model.getQuat(
            self.render
        ).getForward()

        forward.setZ(0)
        forward = -forward

        if forward.lengthSquared() > 0.000001:
            forward.normalize()

        camera_position = (
            car_position
            - forward * CAMERA_DISTANCE
            + Vec3(0, 0, CAMERA_HEIGHT)
        )

        look_position = (
            car_position
            + forward * CAMERA_LOOK_AHEAD
            + Vec3(0, 0, 1.0)
        )

        self.camera.setPos(
            self.render,
            camera_position
        )

        self.camera.lookAt(
            self.render,
            look_position
        )

    def reset_agent(self):

        self.observation, self.info = self.env.reset()

        self.total_reward = 0.0

        self.update_interpolation_state()

    def update_agent(self, task):
        frame_dt = ClockObject.getGlobalClock().getDt()
        frame_dt = max(0.0, min(frame_dt, MAX_FRAME_DT))

        self.simulation_accumulator += frame_dt

        if self.simulation_accumulator > MAX_ACCUMULATOR:
            self.simulation_accumulator = MAX_ACCUMULATOR

        steps_this_frame = 0

        while (
            self.simulation_accumulator >= SIMULATION_DT
            and steps_this_frame < MAX_STEPS_PER_FRAME
        ):

            self.previous_position = self.current_position
            self.previous_heading = self.current_heading

            action = self.predict_action(
                self.observation
            )

            (
                self.observation,
                reward,
                terminated,
                truncated,
                info
            ) = self.env.step(action)

            self.total_reward += float(reward)
            self.total_steps += 1

            state = self.env.controller.get_state()

            self.current_position = (
                float(state["x"]),
                float(state["y"])
            )

            self.current_heading = float(
                state["heading_deg"]
            )

            self.simulation_accumulator -= SIMULATION_DT

            steps_this_frame += 1

            if self.total_steps % 600 == 0:

                state = self.env.controller.get_state()

                print(f"[PPO] Speed: {state['speed']:.2f} m/s | Position: ({state['x']:.2f}, {state['y']:.2f})")

            if terminated or truncated:
                if info.get("lap_completed", False):
                    self.completed_laps += 1
                    print(f"[PPO] Lap completed: {self.completed_laps}")

                print(f"[PPO] Episode finished | Steps: {self.total_steps} | Reward: {self.total_reward:.2f}")
                self.reset_agent()

        alpha = (self.simulation_accumulator / SIMULATION_DT)
        self.update_vehicle_visuals(alpha)
        self.update_camera(frame_dt)
        return task.cont

if __name__ == "__main__":
    simulator = YasMarinaSimulator()
    simulator.run()
