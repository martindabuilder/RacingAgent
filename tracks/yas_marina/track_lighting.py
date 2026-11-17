import numpy as np
from panda3d.core import LineSegs
from .track_light_tools import LightField, LightPool


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
        self.pool_size = 6

        self.led_color = (1.0, 0.98, 0.90, 1.0)
        self.glow_color = (1.0, 0.88, 0.65, 0.45)

        self.pole_data = []
        self.pool = None

        self.pole_lines = None
        self.led_core_lines = None
        self.led_halo_lines = None

    def create(self):
        print("\n Creating lights.")

        self.remove()

        centerline = np.asarray(self.track.centerline, dtype=float)
        left = self.track.left_boundary_3d
        right = self.track.right_boundary_3d

        point_count = len(centerline)

        if point_count < 3:
            raise ValueError("Track must contain at least 3 points.")

        steps = np.linalg.norm(np.roll(centerline, -1, axis=0) - centerline, axis=1)
        distances = np.concatenate(([0.0], np.cumsum(steps)))[:-1]
        track_length = float(steps.sum())

        targets = np.linspace(0, track_length, self.light_count, endpoint=False)

        self.pole_lines = LineSegs("all_poles")
        self.led_core_lines = LineSegs("all_led_cores")
        self.led_halo_lines = LineSegs("all_led_halos")

        for light_index, target in enumerate(targets):
            point_index = min(int(np.searchsorted(distances, target)), point_count - 1,)

            self.create_light(point_index, light_index, centerline, left, right)

        self._finish_geometry()
        self.pool = LightPool(self.render, self.pole_data, count=self.pool_size)
        self.lights = list(self.pool.nodes)
        print(f"[Lighting] {len(self.pole_data)} poles, {self.pool.count} real pooled spotlights.")

    def make_light_field(self, **kwargs):
        return LightField(self.pole_data, **kwargs)

    def update(self, x, y):
        if self.pool is not None:
            self.pool.update(float(x), float(y))

    def create_light(self, point_index, light_index, centerline, left, right):
        point_count = len(centerline)

        previous = centerline[(point_index - 1) % point_count]
        following = centerline[(point_index + 1) % point_count]

        tangent = following - previous
        tangent_length = np.linalg.norm(tangent)

        if tangent_length < 1e-8:
            return

        tangent = tangent / tangent_length
        center = centerline[point_index]
        boundary = left[point_index] if light_index % 2 == 0 else right[point_index]
        outward = np.array([boundary[0] - center[0], boundary[1] - center[1]], dtype=float)
        outward_length = np.linalg.norm(outward)

        if outward_length < 1e-8:
            outward = np.array([-tangent[1], tangent[0]])
        else:
            outward = outward / outward_length

        inward = -outward
        pole_x = float(boundary[0]) + float(outward[0]) * self.pole_offset
        pole_y = float(boundary[1]) + float(outward[1]) * self.pole_offset
        ground_z = float(boundary[2])
        pole_top_z = ground_z + self.pole_height

        self.create_pole(pole_x, pole_y, ground_z, pole_top_z, inward)
        self.pole_data.append(((pole_x, pole_y, pole_top_z), (float(center[0]), float(center[1]), self.target_height),))
        self.create_led(pole_x, pole_y, pole_top_z, tangent, inward)

    def create_led(self, x, y, top_z, tangent, inward):
        for row, arm in enumerate((2.0, 3.1)):
            lamp_x = x + inward[0] * arm
            lamp_y = y + inward[1] * arm

            for k in (-1.0, 0.0, 1.0):
                shift = k * 0.9

                cx = lamp_x + tangent[0] * shift
                cy = lamp_y + tangent[1] * shift

                self.led_core_lines.setColor(*self.led_color)
                self.led_core_lines.setThickness(8.0)
                self.led_core_lines.moveTo(cx - tangent[0] * 0.3, cy - tangent[1] * 0.3, top_z)
                self.led_core_lines.drawTo(cx + tangent[0] * 0.3, cy + tangent[1] * 0.3, top_z)

                self.led_halo_lines.setColor(*self.glow_color)
                self.led_halo_lines.setThickness(20.0)
                self.led_halo_lines.moveTo(cx - tangent[0] * 0.3, cy - tangent[1] * 0.3, top_z)
                self.led_halo_lines.drawTo(cx + tangent[0] * 0.3, cy + tangent[1] * 0.3, top_z)

    def create_pole(self, x, y, ground_z, top_z, inward):
        self.pole_lines.setThickness(3.0)
        self.pole_lines.setColor(0.10, 0.12, 0.15, 1.0)

        self.pole_lines.moveTo(x, y, ground_z)
        self.pole_lines.drawTo(x, y, top_z)

        self.pole_lines.moveTo(x, y, top_z)
        self.pole_lines.drawTo(x + inward[0] * 3.6, y + inward[1] * 3.6, top_z)

    def _finish_geometry(self):
        pole_node = self.render.attachNewNode(self.pole_lines.create())
        pole_node.setLightOff()
        self.poles.append(pole_node)

        core_node = self.render.attachNewNode(self.led_core_lines.create())
        core_node.setLightOff()
        self.glows.append(core_node)

        halo_node = self.render.attachNewNode(self.led_halo_lines.create())
        halo_node.setLightOff()
        halo_node.setTransparency(1)
        self.glows.append(halo_node)

        self.pole_lines = None
        self.led_core_lines = None
        self.led_halo_lines = None

    def remove(self):
        if self.pool is not None:
            self.pool.remove()
            self.pool = None

        for node in self.poles + self.beams + self.glows:
            node.removeNode()

        self.lights.clear()
        self.poles.clear()
        self.beams.clear()
        self.glows.clear()
        self.pole_data.clear()