from pathlib import Path
from direct.showbase.ShowBase import ShowBase
from direct.filter.CommonFilters import CommonFilters
from panda3d.core import (WindowProperties, AmbientLight, DirectionalLight, Vec4, ClockObject)
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


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = (
    PROJECT_DIR
    / "training"
    / "models"
    / "ppo_racing_agent.zip"
)

SIMULATION_DT = 1.0 / 120.0
MAX_STEPS_PER_FRAME = 8


class YasMarinaSimulator(ShowBase):
    def __init__(self):
        super().__init__()

        self.render.setShaderAuto()

        properties = WindowProperties()
        properties.setTitle("F1 Autonomous Racing Simulator")
        properties.setSize(1600, 900)

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

        # Load track data and geometry.
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

        self.vehicle = F1Vehicle(
            self.render,
            self.loader,
            self.track
        )

        start_position = self.track.centerline[0]

        self.facilities = TrackFacilities(
            self.render,
            self.track
        )
        self.facilities.create()

        self.vehicle.set_position(
            float(start_position[0]),
            float(start_position[1]),
            0.2
        )

        start_heading = self.track.get_heading(0)

        self.vehicle.set_rotation(start_heading)

        centerline = self.track.centerline

        min_x = float(centerline[:, 0].min())
        max_x = float(centerline[:, 0].max())

        min_y = float(centerline[:, 1].min())
        max_y = float(centerline[:, 1].max())

        center_x = (min_x + max_x) / 2.0
        center_y = (min_y + max_y) / 2.0

        self.camera.setPos(
            center_x,
            center_y,
            1100
        )

        self.camera.lookAt(
            center_x,
            center_y,
            0
        )

        self.camLens.setFov(50)

        self.filters = CommonFilters(
            self.win,
            self.cam
        )

        try:
            bloom_enabled = self.filters.setBloom(
                blend=(0.3, 0.3, 0.3, 0.0),
                desat=0.0,
                intensity=1.5,
                size="medium",
                mintrigger=0.6,
                maxtrigger=1.0
            )

            if bloom_enabled:
                print("[Graphics] Bloom enabled successfully.")
            else:
                print(
                    "[Graphics] Bloom could not be enabled. "
                    "Graphics hardware or shaders may not support it."
                )

        except Exception as error:
            print(
                "[Graphics] Bloom initialization failed:",
                error
            )


        # Load trained PPO model.
        from stable_baselines3 import PPO
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Trained model not found: {MODEL_PATH}")

        self.model = PPO.load(str(MODEL_PATH), device="auto")
        print("[PPO] Trained model loaded successfully.")

        # Create racing environment.
        self.env = RacingEnv(track=self.track, max_episode_steps=12000, start_speed=0.0)
        self.observation, self.info = self.env.reset()

        # Initialize simulation variables.
        self.simulation_accumulator = 0.0

        self.total_reward = 0.0
        self.total_steps = 0
        self.completed_laps = 0

        self.update_vehicle_visuals()

        # Start autonomous driving.
        self.taskMgr.add(self.update_agent, "update_agent")

        print("[PPO] Autonomous driving started.")

    def update_vehicle_visuals(self):
        state = self.env.controller.get_state()
        self.vehicle.set_position(state["x"], state["y"], 0.2)
        self.vehicle.set_rotation(state["heading_deg"])
        self.vehicle.set_speed(state["speed"])

    def reset_agent(self):
        self.observation, self.info = self.env.reset()
        self.total_reward = 0.0
        self.update_vehicle_visuals()

    def update_agent(self, task):
        frame_dt = ClockObject.getGlobalClock().getDt()
        frame_dt = min(frame_dt, 0.1)

        self.simulation_accumulator += frame_dt

        steps_this_frame = 0

        while (self.simulation_accumulator >= SIMULATION_DT and steps_this_frame < MAX_STEPS_PER_FRAME):

            # Get action from PPO model.
            action, _ = self.model.predict(self.observation, deterministic=True)

            # Update racing environment.
            (self.observation, reward, terminated, truncated, info) = self.env.step(action)

            self.total_reward += float(reward)
            self.total_steps += 1
            self.simulation_accumulator -= SIMULATION_DT

            steps_this_frame += 1

            # Update 3D vehicle position and rotation.
            self.update_vehicle_visuals()
                if self.total_steps % 120 == 0:

                state = self.env.controller.get_state()

                print(f"[PPO] Speed: {state['speed']:.2f} m/s | Position: ({state['x']:.2f}, {state['y']:.2f})")

            # Check episode completion.
            if terminated or truncated:

                if info.get("lap_completed", False):
                    self.completed_laps += 1
                    print(f"[PPO] Lap completed: {self.completed_laps}")

                print(f"[PPO] Episode finished | Steps: {self.total_steps} | Reward: {self.total_reward:.2f}")
                self.reset_agent()

        return task.cont

if __name__ == "__main__":
    simulator = YasMarinaSimulator()
    simulator.run()
