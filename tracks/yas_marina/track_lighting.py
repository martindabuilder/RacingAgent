
import numpy as np

from panda3d.core import (
    Spotlight,
    PerspectiveLens,
    Vec3,
    Vec4,
    LineSegs,
)


class TrackLighting:

    def __init__(self, render, track):

        self.render = render
        self.track = track

        self.lights = []
        self.poles = []
        self.beams = []
        self.glows = []

        self.pole_height = 32.0
        self.pole_offset = 17.0
        self.target_height = 0.5

        self.light_count = 64

        self.light_color = Vec4(
            0.38,
            0.42,
            0.50,
            1.0
        )

        self.light_fov = 95.0
        self.light_exponent = 8.0
        self.light_range = 280.0

        self.led_color = Vec4(
            1.0,
            0.98,
            0.90,
            1.0
        )

        self.glow_color = Vec4(
            1.0,
            0.88,
            0.65,
            0.45
        )

        self.pole_lines = None
        self.led_core_lines = None
        self.led_halo_lines = None

    def create(self):

        print("\nCREATING OPTIMIZED YAS MARINA LIGHTING")
        print("-" * 45)

        self.remove()

        centerline = self.track.centerline
        left = self.track.left_boundary_3d
        right = self.track.right_boundary_3d

        point_count = len(centerline)

        if point_count < 3:
            raise ValueError(
                "Track must contain at least 3 points."
            )

        distances = np.zeros(point_count)

        for i in range(1, point_count):

            distances[i] = (
                distances[i - 1]
                + np.linalg.norm(
                    centerline[i] - centerline[i - 1]
                )
            )

        track_length = float(distances[-1])

        if track_length <= 0:
            raise ValueError("Invalid track length.")

        target_distances = np.linspace(
            0,
            track_length,
            self.light_count,
            endpoint=False
        )

        self.pole_lines = LineSegs(
            "yas_marina_all_poles"
        )

        self.led_core_lines = LineSegs(
            "yas_marina_all_led_cores"
        )

        self.led_halo_lines = LineSegs(
            "yas_marina_all_led_halos"
        )

        for light_index, target_distance in enumerate(
            target_distances
        ):

            point_index = int(
                np.searchsorted(
                    distances,
                    target_distance
                )
            )

            point_index = min(
                point_index,
                point_count - 1
            )

            self.create_light(
                point_index,
                light_index,
                centerline,
                left,
                right
            )

        self._finish_geometry()

        print(
            f"[Lighting] Created {len(self.lights)} spotlights."
        )

        print(
            "[Lighting] Shadows disabled."
        )

        print(
            "[Lighting] Transparent light beams disabled."
        )

        print(
            "[Lighting] Poles and LEDs combined into shared geometry."
        )

    def create_light(
        self,
        point_index,
        light_index,
        centerline,
        left,
        right
    ):

        point_count = len(centerline)

        previous = centerline[
            (point_index - 1) % point_count
        ]

        following = centerline[
            (point_index + 1) % point_count
        ]

        tangent = following - previous
        tangent_length = np.linalg.norm(tangent)

        if tangent_length < 1e-8:
            return

        tangent = tangent / tangent_length

        center = centerline[point_index]

        if light_index % 2 == 0:
            boundary = left[point_index]
        else:
            boundary = right[point_index]

        outward = np.array([
            boundary[0] - center[0],
            boundary[1] - center[1]
        ], dtype=float)

        outward_length = np.linalg.norm(outward)

        if outward_length < 1e-8:

            outward = np.array([
                -tangent[1],
                tangent[0]
            ])

        else:
            outward = outward / outward_length

        inward = -outward

        pole_x = (
            float(boundary[0])
            + float(outward[0]) * self.pole_offset
        )

        pole_y = (
            float(boundary[1])
            + float(outward[1]) * self.pole_offset
        )

        ground_z = float(boundary[2])
        pole_top_z = ground_z + self.pole_height

        self.create_pole(
            pole_x,
            pole_y,
            ground_z,
            pole_top_z,
            light_index,
            inward
        )

        target = Vec3(
            float(center[0]),
            float(center[1]),
            self.target_height
        )

        spotlight = Spotlight(
            "yas_marina_spotlight_" + str(light_index)
        )

        lens = PerspectiveLens()

        lens.setFov(self.light_fov)
        lens.setNearFar(
            1.0,
            self.light_range
        )

        spotlight.setLens(lens)
        spotlight.setColor(self.light_color)
        spotlight.setExponent(self.light_exponent)

        spotlight.setAttenuation(
            Vec3(1.0, 0.0, 0.000008)
        )

        # No shadow maps: significantly lower GPU cost.

        light_node = self.render.attachNewNode(
            spotlight
        )

        light_node.setPos(
            pole_x,
            pole_y,
            pole_top_z
        )

        light_node.lookAt(target)

        self.render.setLight(light_node)

        self.lights.append(light_node)

        self.create_led(
            pole_x,
            pole_y,
            pole_top_z,
            light_index,
            tangent,
            inward
        )

    def create_led(
        self,
        x,
        y,
        top_z,
        index,
        tangent,
        inward
    ):

        arm_length = 2.5

        lamp_x = x + inward[0] * arm_length
        lamp_y = y + inward[1] * arm_length
        lamp_z = top_z

        bar_length = 0.9

        self.led_core_lines.setColor(
            self.led_color
        )

        self.led_core_lines.setThickness(9.0)

        self.led_core_lines.moveTo(
            lamp_x - tangent[0] * bar_length,
            lamp_y - tangent[1] * bar_length,
            lamp_z
        )

        self.led_core_lines.drawTo(
            lamp_x + tangent[0] * bar_length,
            lamp_y + tangent[1] * bar_length,
            lamp_z
        )

        self.led_halo_lines.setColor(
            self.glow_color
        )

        self.led_halo_lines.setThickness(20.0)

        self.led_halo_lines.moveTo(
            lamp_x - tangent[0] * 1.15,
            lamp_y - tangent[1] * 1.15,
            lamp_z
        )

        self.led_halo_lines.drawTo(
            lamp_x + tangent[0] * 1.15,
            lamp_y + tangent[1] * 1.15,
            lamp_z
        )

    def create_pole(
        self,
        x,
        y,
        ground_z,
        top_z,
        index,
        inward
    ):

        self.pole_lines.setThickness(3.0)

        self.pole_lines.setColor(
            0.10,
            0.12,
            0.15,
            1.0
        )

        self.pole_lines.moveTo(
            x,
            y,
            ground_z
        )

        self.pole_lines.drawTo(
            x,
            y,
            top_z
        )

        arm_length = 2.5

        self.pole_lines.moveTo(
            x,
            y,
            top_z
        )

        self.pole_lines.drawTo(
            x + inward[0] * arm_length,
            y + inward[1] * arm_length,
            top_z
        )

    def _finish_geometry(self):
        pole_node = self.render.attachNewNode(self.pole_lines.create())
        pole_node.setLightOff()

        self.poles.append(pole_node)

        core_node = self.render.attachNewNode(
            self.led_core_lines.create()
        )

        core_node.setLightOff()

        self.glows.append(core_node)

        halo_node = self.render.attachNewNode(
            self.led_halo_lines.create()
        )

        halo_node.setLightOff()
        halo_node.setTransparency(1)

        self.glows.append(halo_node)

        self.pole_lines = None
        self.led_core_lines = None
        self.led_halo_lines = None

    def remove(self):

        for light in self.lights:
            self.render.clearLight(light)
            light.removeNode()

        for node in self.poles:
            node.removeNode()

        for node in self.beams:
            node.removeNode()

        for node in self.glows:
            node.removeNode()

        self.lights.clear()
        self.poles.clear()
        self.beams.clear()
        self.glows.clear()

        self.pole_lines = None
        self.led_core_lines = None
        self.led_halo_lines = None
