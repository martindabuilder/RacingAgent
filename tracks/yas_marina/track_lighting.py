import numpy as np
from panda3d.core import (Spotlight, PerspectiveLens, Vec3, Vec4, LineSegs, Geom, GeomNode, GeomTriangles, GeomVertexData, GeomVertexFormat, GeomVertexWriter, TransparencyAttrib, Material)

class TrackLighting:
    def __init__(self, render, track):

        self.render = render
        self.track = track

        self.lights = []
        self.poles = []
        self.beams = []
        self.glows = []

        # Pole configuration
        self.pole_height = 32.0
        self.pole_offset = 17.0
        self.target_height = 0.5

        # Projector configuration
        self.light_count = 60

        self.light_color = Vec4(
            1.0,
            0.97,
            0.88,
            1.0
        )

        self.light_fov = 105.0
        self.light_exponent = 8.0
        self.light_range = 260.0

        # Shadow configuration
        self.shadow_every = 4
        self.shadow_map_size = 2048

        # Beam configuration
        self.beam_radius = 42.0
        self.beam_segments = 40
        self.beam_rings = 10
        self.beam_alpha = 0.055

        # LED configuration
        self.led_color = Vec4(
            1.0,
            0.98,
            0.88,
            1.0
        )

        self.glow_color = Vec4(
            1.0,
            0.88,
            0.58,
            1.0
        )

    def create(self):

        print("\nCREATING YAS MARINA NIGHT LIGHTING")
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

        normal = np.array([
            -tangent[1],
            tangent[0]
        ])

        center = centerline[point_index]

        if light_index % 2 == 0:

            boundary = left[point_index]

            outward = np.array([
                boundary[0] - center[0],
                boundary[1] - center[1]
            ])

        else:

            boundary = right[point_index]

            outward = np.array([
                boundary[0] - center[0],
                boundary[1] - center[1]
            ])

        outward_length = np.linalg.norm(outward)

        if outward_length < 1e-8:
            outward = normal
        else:
            outward = outward / outward_length

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
            light_index
        )

        target = Vec3(
            float(center[0]),
            float(center[1]),
            self.target_height
        )

        # Real spotlight
        spotlight = Spotlight(
            "yas_marina_spotlight_" + str(light_index)
        )

        lens = PerspectiveLens()
        lens.setFov(self.light_fov)
        lens.setNearFar(1.0, self.light_range)

        spotlight.setLens(lens)
        spotlight.setColor(self.light_color)
        spotlight.setExponent(self.light_exponent)

        # Shadow mapping
        if light_index % self.shadow_every == 0:

            spotlight.setShadowCaster(
                True,
                self.shadow_map_size,
                self.shadow_map_size
            )

            print(
                f"[Shadows] Enabled for spotlight {light_index}"
            )

        spotlight.setAttenuation(
            Vec3(1.0, 0.0, 0.000025)
        )

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

        # Rounded atmospheric beam
        self.create_beam(
            pole_x,
            pole_y,
            pole_top_z,
            target,
            light_index
        )

        # Bright LED source
        self.create_led(
            pole_x,
            pole_y,
            pole_top_z,
            light_index
        )

    def create_beam(
        self,
        x,
        y,
        top_z,
        target,
        index
    ):

        origin = np.array([
            x,
            y,
            top_z
        ], dtype=float)

        destination = np.array([
            target.x,
            target.y,
            target.z
        ], dtype=float)

        direction = destination - origin
        direction_length = np.linalg.norm(direction)

        if direction_length < 1e-8:
            return

        direction = direction / direction_length

        reference = np.array([0.0, 0.0, 1.0])

        side = np.cross(direction, reference)

        side_length = np.linalg.norm(side)

        if side_length < 1e-8:
            reference = np.array([1.0, 0.0, 0.0])
            side = np.cross(direction, reference)
            side_length = np.linalg.norm(side)

        side = side / side_length

        up = np.cross(direction, side)
        up = up / np.linalg.norm(up)

        vertex_format = GeomVertexFormat.getV3c4()

        vertex_data = GeomVertexData(
            "rounded_beam_" + str(index),
            vertex_format,
            Geom.UHStatic
        )

        vertex = GeomVertexWriter(
            vertex_data,
            "vertex"
        )

        color = GeomVertexWriter(
            vertex_data,
            "color"
        )

        triangles = GeomTriangles(Geom.UHStatic)

        for ring in range(self.beam_rings + 1):

            t = ring / self.beam_rings

            smooth_t = t * t * (3.0 - 2.0 * t)

            radius = (
                0.8
                + self.beam_radius * smooth_t
            )

            fade = np.sin(np.pi * t) ** 2.5

            for segment in range(self.beam_segments):

                angle = (
                    2.0 * np.pi * segment
                    / self.beam_segments
                )

                px = (
                    destination
                    + direction * (
                        -direction_length * (1.0 - t)
                    )
                    + side * np.cos(angle) * radius
                    + up * np.sin(angle) * radius
                )

                vertex.addData3f(
                    float(px[0]),
                    float(px[1]),
                    float(px[2])
                )

                alpha = self.beam_alpha * fade

                color.addData4f(
                    1.0,
                    0.94,
                    0.78,
                    float(alpha)
                )

        for ring in range(self.beam_rings):

            for segment in range(self.beam_segments):

                next_segment = (
                    segment + 1
                ) % self.beam_segments

                a = (
                    ring * self.beam_segments
                    + segment
                )

                b = (
                    ring * self.beam_segments
                    + next_segment
                )

                c = (
                    (ring + 1) * self.beam_segments
                    + segment
                )

                d = (
                    (ring + 1) * self.beam_segments
                    + next_segment
                )

                triangles.addVertices(a, b, c)
                triangles.addVertices(b, d, c)

        geom = Geom(vertex_data)
        geom.addPrimitive(triangles)

        geom_node = GeomNode(
            "rounded_beam_node_" + str(index)
        )

        geom_node.addGeom(geom)

        beam_node = self.render.attachNewNode(
            geom_node
        )

        beam_node.setTransparency(
            TransparencyAttrib.MAlpha
        )

        beam_node.setDepthWrite(False)
        beam_node.setTwoSided(True)
        beam_node.setLightOff()
        beam_node.setBin("transparent", 10)

        self.beams.append(beam_node)

    def create_led(
        self,
        x,
        y,
        top_z,
        index
    ):

        arm_length = 2.5

        lamp_x = x + arm_length
        lamp_y = y
        lamp_z = top_z

        # Bright white LED core
        core = LineSegs(
            "led_core_" + str(index)
        )

        core.setThickness(13.0)
        core.setColor(
            1.0,
            1.0,
            0.94,
            1.0
        )

        core.moveTo(
            lamp_x - 0.9,
            lamp_y,
            lamp_z
        )

        core.drawTo(
            lamp_x + 0.9,
            lamp_y,
            lamp_z
        )

        core_node = self.render.attachNewNode(
            core.create()
        )

        core_node.setLightOff()

        core_material = Material()
        core_material.setEmission(
            Vec4(1.0, 1.0, 0.88, 1.0)
        )

        core_node.setMaterial(core_material)

        self.glows.append(core_node)

        # Strong warm halo
        halo = LineSegs(
            "led_halo_" + str(index)
        )

        halo.setThickness(25.0)
        halo.setColor(
            1.0,
            0.82,
            0.42,
            0.85
        )

        halo.moveTo(
            lamp_x - 1.15,
            lamp_y,
            lamp_z
        )

        halo.drawTo(
            lamp_x + 1.15,
            lamp_y,
            lamp_z
        )

        halo_node = self.render.attachNewNode(
            halo.create()
        )

        halo_node.setLightOff()

        self.glows.append(halo_node)

        # Outer soft glow
        outer = LineSegs(
            "led_outer_glow_" + str(index)
        )

        outer.setThickness(38.0)
        outer.setColor(
            1.0,
            0.72,
            0.30,
            0.28
        )

        outer.moveTo(
            lamp_x - 1.4,
            lamp_y,
            lamp_z
        )

        outer.drawTo(
            lamp_x + 1.4,
            lamp_y,
            lamp_z
        )

        outer_node = self.render.attachNewNode(
            outer.create()
        )

        outer_node.setLightOff()

        self.glows.append(outer_node)

    def create_pole(
        self,
        x,
        y,
        ground_z,
        top_z,
        index
    ):

        lines = LineSegs(
            "yas_marina_light_pole_" + str(index)
        )

        lines.setThickness(3.0)
        lines.setColor(0.10, 0.12, 0.15, 1.0)

        lines.moveTo(x, y, ground_z)
        lines.drawTo(x, y, top_z)

        arm_length = 2.5

        lines.moveTo(x, y, top_z)
        lines.drawTo(x + arm_length, y, top_z)

        pole_node = self.render.attachNewNode(
            lines.create()
        )

        self.poles.append(pole_node)

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
