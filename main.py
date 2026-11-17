import numpy as np
import torch
from pathlib import Path
import math
from direct.showbase.ShowBase import ShowBase
from direct.filter.CommonFilters import CommonFilters
from direct.gui.OnscreenText import OnscreenText
from direct.gui.OnscreenImage import OnscreenImage
from direct.gui.DirectGui import DirectFrame
from panda3d.core import (WindowProperties, AmbientLight, DirectionalLight, Vec3, Vec4, ClockObject, TextNode, PNMImage, Texture)
from simulation.vehicle import F1Vehicle
from environment.track_loader import TrackLoader
from environment.racing_env import RacingEnv
from tracks.yas_marina.track_mesh import TrackMesh
from tracks.yas_marina.track_environment_polished import PolishedEnvironment
from tracks.yas_marina.track_runoff import TrackRunoff
from tracks.yas_marina.track_barriers import TrackBarriers
from tracks.yas_marina.track_grandstands import TrackGrandstands
from tracks.yas_marina.track_lighting import TrackLighting
from tracks.yas_marina.track_facilities import TrackFacilities
from tracks.yas_marina.track_markings import TrackMarkings
from tracks.yas_marina.track_light_tools import bake_vertex_lighting, CarBlobShadow

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
CAMERA_LOOK_AHEAD = 6.0


class YasMarinaSimulator(ShowBase):
    def __init__(self):
        super().__init__()

        self.setFrameRateMeter(True)

        properties = WindowProperties()
        properties.setTitle("F1 Autonomous Racing Simulator")
        properties.setSize(960, 540)

        self.win.requestProperties(properties)
        self.setBackgroundColor(0.025, 0.035, 0.060, 1.0)
        self.hud_font = None

        try:
            self.hud_font = self.loader.loadFont("C:/Windows/Fonts/arial.ttf")

            if self.hud_font:
                self.hud_font.setPixelsPerUnit(100)
                print("[HUD] High-resolution font loaded.")

        except Exception as error:
            print(f"[HUD] Custom font unavailable: {error}")

        ambient_light = AmbientLight("night_ambient")
        ambient_light.setColor(Vec4(0.22, 0.24, 0.30, 1.0))

        self.ambient_light_node = (self.render.attachNewNode(ambient_light))
        self.render.setLight(self.ambient_light_node)

        moon_light = DirectionalLight("moon_light")
        moon_light.setColor(Vec4(0.22, 0.25, 0.32, 1.0))

        self.moon_light_node = (self.render.attachNewNode(moon_light))
        self.moon_light_node.setHpr(35, -55, 0)
        self.render.setLight(self.moon_light_node)
        self.filters = None
        self.bloom_enabled = False

        try:
            self.filters = CommonFilters(self.win, self.cam)
            self.bloom_enabled = self.filters.setBloom(blend=(0.10, 0.10, 0.10, 0.0), desat=0.0, intensity=0.5, size="medium")

            if self.bloom_enabled:
                print("[Graphics] Bloom enabled.")
            else:
                print("[Graphics] Bloom is not supported.")

        except Exception as error:
            self.filters = None
            self.bloom_enabled = False
            print(f"[Graphics] Bloom unavailable: {error}")

    
        self.track = TrackLoader.load("yas_marina")

        self.track_lighting = TrackLighting( self.render, self.track)
        self.track_lighting.create()
        self.light_field = self.track_lighting.make_light_field()

        self.track_mesh = TrackMesh(self.render, self.track, self.light_field)
        self.track_mesh.create()

        self.environment = PolishedEnvironment(self.render, self.track, self.light_field)
        self.environment.create()

        self.runoff = TrackRunoff(self.render, self.track, self.light_field)
        self.runoff.create()

        self.barriers = TrackBarriers(self.render, self.track)
        self.barriers.create()

        self.track_markings = TrackMarkings(self.render, self.track)
        self.track_markings.create()

        self.grandstands = TrackGrandstands(self.track, self.render)
        self.grandstands.create()

        self.facilities = TrackFacilities( self.render, self.track)
        self.facilities.create()

        for root in (self.track_markings.root, self.barriers.root, self.grandstands.root, self.facilities.root,):
            root.flattenStrong()

        bake_vertex_lighting(self.track_markings.root, self.light_field, up=True)
        bake_vertex_lighting(self.barriers.root, self.light_field, up=False)
        bake_vertex_lighting(self.grandstands.root, self.light_field, up=False)

    
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
        self.blob_shadow = CarBlobShadow(self.render, heading_offset=0.0)
        self.render.setShaderAuto()
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

        print(
            "[PPO] Trained model loaded successfully."
        )

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
        self.last_reward = 0.0

        self.last_action = np.zeros(
            self.model.action_space.shape,
            dtype=np.float32
        )

        initial_state = self.env.controller.get_state()
        self.vehicle.set_steering_angle(initial_state["steering_angle"])


        self.setup_reward_hud()
        self.setup_f1_hud()

        self.update_vehicle_visuals()
        self.update_camera(1.0)
        self.update_f1_hud()

        self.taskMgr.add(
            self.update_agent,
            "update_agent"
        )

        print("[PPO] Autonomous driving started.")


    def create_hud_text(self, text, pos,
        scale,
        color,
        align=TextNode.ACenter
    ):

        return OnscreenText(
            text=text,
            pos=pos,
            scale=scale,
            fg=color,
            align=align,
            font=self.hud_font,
            mayChange=True,
            shadow=(0, 0, 0, 0.45),
            shadowOffset=(0.015, 0.015)
        )

    def create_rounded_panel_texture(self):

        width = 512
        height = 160
        radius = 28

        image = PNMImage(
            width,
            height,
            4
        )

        image.fill(0.012, 0.020, 0.035)
        image.alphaFill(0.0)

        for y in range(height):
            for x in range(width):

                nearest_x = min(
                    max(x, radius),
                    width - radius - 1
                )

                nearest_y = min(
                    max(y, radius),
                    height - radius - 1
                )

                dx = x - nearest_x
                dy = y - nearest_y

                inside = (
                    dx * dx + dy * dy
                    <= radius * radius
                )

                if (
                    inside
                    or (
                        radius <= x < width - radius
                        or radius <= y < height - radius
                    )
                ):
                    image.setXelA(
                        x,
                        y,
                        0.012,
                        0.020,
                        0.035,
                        0.90
                    )

        texture = Texture("rounded_telemetry_panel")
        texture.load(image)

        texture.setMinfilter(Texture.FTLinear)
        texture.setMagfilter(Texture.FTLinear)

        return texture


    def setup_reward_hud(self):

        self.reward_last_hud = self.create_hud_text(
            text="LAST  +0.00",
            pos=(-1.68, 0.925),
            scale=0.032,
            color=(0.20, 1.0, 0.42, 1.0),
            align=TextNode.ALeft
        )

        self.reward_total_hud = self.create_hud_text(
            text="TOTAL  +0.00",
            pos=(-1.68, 0.865),
            scale=0.032,
            color=(0.0, 0.82, 0.95, 1.0),
            align=TextNode.ALeft
        )

        self.reward_last_hud.setBin("fixed", 20)
        self.reward_total_hud.setBin("fixed", 20)

        self.update_reward_hud()

    def update_reward_hud(self):

        reward = self.last_reward
        total = self.total_reward

        # Last step reward color

        if reward > 0.000001:
            last_color = (0.20, 1.0, 0.42, 1.0)
            last_text = f"+{reward:.2f}"

        elif reward < -0.000001:
            last_color = (1.0, 0.25, 0.25, 1.0)
            last_text = f"{reward:.2f}"

        else:
            last_color = (0.90, 0.96, 1.0, 1.0)
            last_text = "0.00"

        # Total episode reward

        if total >= 0:
            total_text = f"+{total:,.2f}"
        else:
            total_text = f"{total:,.2f}"

        self.reward_last_hud.setText(
            f"LAST  {last_text}"
        )

        self.reward_last_hud.setFg(last_color)

        self.reward_total_hud.setText(
            f"TOTAL  {total_text}"
        )

    def setup_f1_hud(self):

        # Rounded main panel

        panel_texture = self.create_rounded_panel_texture()

        self.telemetry_panel = OnscreenImage(
            image=panel_texture,
            parent=self.aspect2d,
            pos=(0, 0, -0.805),
            scale=(0.55, 1, 0.145)
        )

        self.telemetry_panel.setTransparency(True)
        self.telemetry_panel.setBin("fixed", 10)

        # Top accent

        self.telemetry_accent = DirectFrame(
            parent=self.aspect2d,
            frameColor=(0.0, 0.78, 0.90, 0.95),
            frameSize=(-0.49, 0.49, -0.002, 0.002),
            pos=(0, 0, -0.662)
        )

        # Bottom accent

        self.telemetry_bottom_accent = DirectFrame(
            parent=self.aspect2d,
            frameColor=(0.15, 0.20, 0.27, 0.75),
            frameSize=(-0.40, 0.40, -0.001, 0.001),
            pos=(0, 0, -0.948)
        )

        self.steering_hud = self.create_hud_text(
            text="0.0°",
            pos=(0.0, -0.680),
            scale=0.026,
            color=(0.72, 0.86, 0.96, 1.0)
        )

        self.throttle_hud = self.create_hud_text(
            text="0%",
            pos=(-0.31, -0.738),
            scale=0.040,
            color=(0.20, 1.0, 0.42, 1.0)
        )

        self.gear_hud = self.create_hud_text(
            text="N",
            pos=(0.0, -0.738),
            scale=0.054,
            color=(1.0, 0.85, 0.20, 1.0)
        )

        self.brake_hud = self.create_hud_text(
            text="0%",
            pos=(0.31, -0.738),
            scale=0.040,
            color=(1.0, 0.28, 0.28, 1.0)
        )

        # Separators

        self.left_separator = DirectFrame(
            parent=self.aspect2d,
            frameColor=(0.25, 0.34, 0.43, 0.75),
            frameSize=(-0.001, 0.001, -0.025, 0.025),
            pos=(-0.155, 0, -0.738)
        )

        self.right_separator = DirectFrame(
            parent=self.aspect2d,
            frameColor=(0.25, 0.34, 0.43, 0.75),
            frameSize=(-0.001, 0.001, -0.025, 0.025),
            pos=(0.155, 0, -0.738)
        )

        self.speed_hud = self.create_hud_text(
            text="0  KM/H",
            pos=(0.0, -0.845),
            scale=0.066,
            color=(1.0, 1.0, 1.0, 1.0),
            align=TextNode.ACenter
        )

   
        self.rpm_hud = self.create_hud_text(
            text="1000",
            pos=(0.0, -0.920),
            scale=0.024,
            color=(0.68, 0.78, 0.88, 1.0)
        )

    def get_control_values(self, state):

        action = np.asarray(
            self.last_action,
            dtype=np.float32
        ).flatten()

        if len(action) >= 3:

            throttle = float(
                np.clip(action[1], 0.0, 1.0)
            )

            brake = float(
                np.clip(action[2], 0.0, 1.0)
            )

        else:
            throttle = 0.0
            brake = 0.0

        return throttle, brake

    def update_f1_hud(self):

        state = self.env.controller.get_state()

        speed_ms = max(
            0.0,
            float(state["speed"])
        )

        speed_kmh = speed_ms * 3.6

        steering_deg = math.degrees(
            float(state["steering_angle"])
        )

        throttle, brake = self.get_control_values(state)


        if speed_kmh < 5:
            gear = "N"
            rpm = 1000

        else:
            gear_number = min(8, max(1, int(speed_kmh / 45) + 1))
            gear = str(gear_number)
            gear_min_speed = (gear_number - 1) * 45
            gear_max_speed = gear_number * 45

            gear_progress = (speed_kmh - gear_min_speed) / max(1, gear_max_speed - gear_min_speed)
            rpm = int(5000 + np.clip(gear_progress, 0, 1) * 7000)

        self.speed_hud.setText(f"{speed_kmh:.0f}  KM/H")
        self.gear_hud.setText(gear)
        self.rpm_hud.setText(f"{rpm:,}")
        self.throttle_hud.setText(f"{throttle * 100:.0f}%")
        self.brake_hud.setText(f"{brake * 100:.0f}%")
        self.steering_hud.setText(f"{steering_deg:+.1f}°")

    def predict_action(self, observation):

        observation = np.asarray(
            observation,
            dtype=np.float32
        )

        observation_tensor = (
            self.model.policy.obs_to_tensor(
                observation
            )[0]
        )

        with torch.inference_mode():

            actions = self.model.policy(
                observation_tensor,
                deterministic=True
            )[0]

        actions = actions.cpu().numpy()

        if self.model.policy.squash_output:

            actions = self.model.policy.unscale_action(
                actions
            )

        else:

            actions = np.clip(
                actions,
                self.model.action_space.low,
                self.model.action_space.high
            )

        return actions.reshape(
            self.model.action_space.shape
        )

    def get_vehicle_state(self):

        state = self.env.controller.get_state()

        return (
            float(state["x"]),
            float(state["y"]),
            float(state["heading_deg"])
        )

    def update_interpolation_state(self):

        position_x, position_y, heading = (
            self.get_vehicle_state()
        )

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

    def update_vehicle_visuals(self, alpha=1.0, dt=0.0):

        if self.previous_position is None:
            self.update_interpolation_state()

        alpha = float(np.clip(alpha, 0.0, 1.0))

        x = (
            self.previous_position[0]
            + (
                self.current_position[0]
                - self.previous_position[0]
            ) * alpha
        )

        y = (
            self.previous_position[1]
            + (
                self.current_position[1]
                - self.previous_position[1]
            ) * alpha
        )

        heading_difference = (
            (
                self.current_heading
                - self.previous_heading
                + 180.0
            ) % 360.0
        ) - 180.0

        heading = (
            self.previous_heading
            + heading_difference * alpha
        )

        self.vehicle.set_position(x, y, 0.2)
        self.vehicle.set_rotation(heading)

        state = self.env.controller.get_state()

        self.vehicle.set_speed(state["speed"])
        self.vehicle.update_wheel_rotation(dt)

        self.vehicle.set_steering_angle(
            state["steering_angle"]
        )

        self.track_lighting.update(x, y)
        self.blob_shadow.follow(self.vehicle.car)

    def update_camera(self, dt):

        car_position = self.vehicle.car.getPos(
            self.render
        )

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
        self.last_reward = 0.0

        self.last_action = np.zeros(
            self.model.action_space.shape,
            dtype=np.float32
        )

        self.update_interpolation_state()

        state = self.env.controller.get_state()

        self.vehicle.set_steering_angle(
            state["steering_angle"]
        )

    def update_agent(self, task):

        frame_dt = ClockObject.getGlobalClock().getDt()

        frame_dt = max(
            0.0,
            min(frame_dt, MAX_FRAME_DT)
        )

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

            self.last_action = np.asarray(action, dtype=np.float32).copy()

            (self.observation, reward, terminated, truncated, info) = self.env.step(action)

            self.last_reward = float(reward)
            self.total_reward += self.last_reward
            self.total_steps += 1

            state = self.env.controller.get_state()

            self.current_position = (float(state["x"]), float(state["y"]))
            self.current_heading = float(state["heading_deg"])
            self.simulation_accumulator -= SIMULATION_DT

            steps_this_frame += 1

            if self.total_steps % 600 == 0:
                print(f"[PPO] Speed: {state['speed']:.2f} m/s | Position: ({state['x']:.2f}, {state['y']:.2f}) | Steering: {math.degrees(state['steering_angle']):.2f} deg")

            if terminated or truncated:
                if info.get("lap_completed", False):
                    self.completed_laps += 1
                    print(f"[PPO] Lap completed: {self.completed_laps}")

                print(f"[PPO] Episode finished | Steps: {self.total_steps} | Reward: {self.total_reward:.2f}")
                self.reset_agent()

        alpha = (self.simulation_accumulator / SIMULATION_DT)
        self.update_vehicle_visuals(alpha, frame_dt)
        self.update_camera(frame_dt)
        self.update_reward_hud()
        self.update_f1_hud()
        return task.cont

if __name__ == "__main__":
    simulator = YasMarinaSimulator()
    simulator.run()