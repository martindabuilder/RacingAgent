import numpy as np
from shapely.geometry import LineString, Polygon
from panda3d.core import (Geom, GeomNode, GeomTriangles, GeomVertexData, GeomVertexFormat, GeomVertexWriter,)

class TrackGrandstands:
    def __init__(self, track_geometry, render):

        self.track = track_geometry
        self.render = render
        self.root = None

        self.barrier_clearance = 15.0
        self.wall_width = 0.45
        self.safety_gap = 5.0

        self.stand_depth = 30.0

        self.rows = 15
        self.row_pitch = 1.25
        self.row_height = 0.30

        self.seat_width = 0.82
        self.seat_depth = 0.72
        self.seat_pitch = 1.0

        self.seat_height = 0.18
        self.backrest_height = 0.55

        self.concourse_height = 0.8

        self.roof_height = 13.0
        self.roof_thickness = 0.9
        self.roof_overhang = 5.0

        self.concrete_color = (0.42, 0.44, 0.47, 1.0)
        self.concrete_light = (0.57, 0.59, 0.62, 1.0)

        self.seat_blue = (0.035, 0.12, 0.24, 1.0)
        self.seat_cyan = (0.02, 0.48, 0.58, 1.0)
        self.seat_white = (0.82, 0.84, 0.86, 1.0)

        self.roof_color = (0.19, 0.22, 0.27, 1.0)
        self.roof_edge_color = (0.035, 0.07, 0.12, 1.0)

        self.support_color = (0.36, 0.39, 0.43, 1.0)
        self.vip_color = (0.12, 0.16, 0.21, 1.0)

        self.stand_specs = [
            (0.025, "main-grandstand", "left", 105.0),
            (0.225, "north-grandstand", "right", 115.0),
            (0.425, "west-grandstand", "left", 105.0),
            (0.625, "south-grandstand", "right", 100.0),
            (0.550, "marina-grandstand", "left", 120.0),
        ]

        self.minimum_stand_spacing = 180.0
        self.track_keepout = None

    def _get_track_data(self):
        centerline = np.asarray(self.track.centerline, dtype=float)[:, :2]
        left_boundary = np.asarray(self.track.left_boundary_3d, dtype=float)[:, :2]
        right_boundary = np.asarray(self.track.right_boundary_3d, dtype=float)[:, :2]

        if len(centerline) < 3:
            raise ValueError(
                "Track centerline contains too few points."
            )

        if (len(left_boundary) != len(centerline) or len(right_boundary) != len(centerline)):
            raise ValueError("Track boundary point counts do not match.")

        segments = (np.roll(centerline, -1, axis=0) - centerline)
        lengths = np.linalg.norm(segments, axis=1)
        cumulative = np.concatenate(([0.0], np.cumsum(lengths)))

        return (centerline, left_boundary, right_boundary, cumulative, cumulative[-1])

    def _build_track_keepout(self):
        (centerline, left_boundary, right_boundary, cumulative, total_length) = self._get_track_data()
        closed_centerline = np.vstack((centerline, centerline[0]))
        line = LineString(closed_centerline)

        left_widths = np.linalg.norm(left_boundary - centerline, axis=1)
        right_widths = np.linalg.norm(right_boundary - centerline, axis=1)

        max_width = max(float(np.max(left_widths)), float(np.max(right_widths)))
        safety_margin = 2.0

        self.track_keepout = line.buffer(
            max_width + safety_margin,
            quad_segs=8,
            cap_style="round",
            join_style="round"
        )

        print(
            "[GRANDSTANDS] Track safety zone created. "
            f"Radius: {max_width + safety_margin:.2f} m"
        )

    def _sample_track(self, fraction, side):

        (
            centerline,
            left_boundary,
            right_boundary,
            cumulative,
            total_length
        ) = self._get_track_data()

        target = (
            fraction % 1.0
        ) * total_length

        index = np.searchsorted(
            cumulative,
            target,
            side="right"
        ) - 1

        index = int(np.clip(
            index,
            0,
            len(centerline) - 1
        ))

        next_index = (
            index + 1
        ) % len(centerline)

        segment_length = (
            cumulative[index + 1]
            - cumulative[index]
        )

        if segment_length < 1e-8:
            raise ValueError("Invalid track segment.")

        t = (
            target - cumulative[index]
        ) / segment_length

        t = float(np.clip(t, 0.0, 1.0))

        center = (
            centerline[index] * (1.0 - t)
            + centerline[next_index] * t
        )

        tangent = (
            centerline[next_index]
            - centerline[index]
        )

        tangent_length = np.linalg.norm(tangent)

        if tangent_length < 1e-8:
            raise ValueError("Invalid track tangent.")

        tangent /= tangent_length

        if side == "left":
            boundary_array = left_boundary
        else:
            boundary_array = right_boundary

        boundary = (
            boundary_array[index] * (1.0 - t)
            + boundary_array[next_index] * t
        )

        outward = np.array([
            -tangent[1],
            tangent[0]
        ])

        if np.dot(
            outward,
            boundary - center
        ) < 0:
            outward *= -1.0

        return boundary, tangent, outward

 
    def _get_stand_footprint(
        self,
        anchor,
        tangent,
        outward,
        length
    ):

        # Включваме основата, VIP зоната и покрива.
        half_length = (
            length / 2.0
            + self.roof_overhang / 2.0
            + 2.0
        )

        min_depth = -self.stand_depth / 2.0 - 0.5
        max_depth = 18.0

        corners = [
            anchor
            - tangent * half_length
            + outward * min_depth,

            anchor
            + tangent * half_length
            + outward * min_depth,

            anchor
            + tangent * half_length
            + outward * max_depth,

            anchor
            - tangent * half_length
            + outward * max_depth,
        ]

        footprint = Polygon([
            (float(p[0]), float(p[1]))
            for p in corners
        ])

        return footprint

 
    def _find_safe_marina_placement(
        self,
        boundary,
        tangent,
        outward,
        requested_length
    ):

        base_offset = (
            self.barrier_clearance
            + self.wall_width / 2.0
            + self.safety_gap
            + self.stand_depth / 2.0
        )

        print(
            "\n[GRANDSTANDS] Checking Marina Grandstand "
            "against the track..."
        )

        # Първо опитваме с оригиналната дължина.
        # Ако не се побира, намаляваме я постепенно.
        candidate_lengths = range(
            int(requested_length),
            49,
            -5
        )

        # Проверяваме и постепенно изместване навън.
        candidate_offsets = range(
            0,
            121,
            5
        )

        for candidate_length in candidate_lengths:

            for extra_offset in candidate_offsets:

                offset = (
                    base_offset + extra_offset
                )

                anchor = (
                    boundary + outward * offset
                )

                footprint = self._get_stand_footprint(
                    anchor,
                    tangent,
                    outward,
                    candidate_length
                )

                # Проверка за пресичане с безопасната зона
                # около цялото трасе.
                collision = footprint.intersects(
                    self.track_keepout
                )

                if not collision:

                    print(
                        "[GRANDSTANDS] Marina Grandstand "
                        "SAFE placement found."
                    )

                    print(
                        f"[GRANDSTANDS] Final length: "
                        f"{candidate_length} m"
                    )

                    print(
                        f"[GRANDSTANDS] Outward offset: "
                        f"{offset:.1f} m"
                    )

                    if candidate_length < requested_length:
                        print(
                            "[GRANDSTANDS] Stand length "
                            "automatically reduced."
                        )

                    if extra_offset > 0:
                        print(
                            "[GRANDSTANDS] Stand moved "
                            "farther away from the track."
                        )

                    return (
                        anchor,
                        float(candidate_length),
                        float(offset)
                    )

        raise RuntimeError(
            "Could not find a safe position for "
            "Marina Grandstand. No stand was created."
        )


    def _local_point(
        self,
        anchor,
        tangent,
        outward,
        x,
        y
    ):

        return (
            anchor
            + tangent * x
            + outward * y
        )


    def _add_box(
        self,
        vertices,
        colors,
        triangles,
        anchor,
        tangent,
        outward,
        x,
        y,
        length,
        depth,
        base_z,
        height,
        color,
        slope=0.0
    ):

        half_length = length / 2.0
        half_depth = depth / 2.0

        corners = [
            (x - half_length, y - half_depth),
            (x + half_length, y - half_depth),
            (x + half_length, y + half_depth),
            (x - half_length, y + half_depth),
        ]

        start = len(vertices)

        for z_offset in (0.0, height):

            for lx, ly in corners:

                point = self._local_point(
                    anchor,
                    tangent,
                    outward,
                    lx,
                    ly
                )

                z = (
                    base_z
                    + z_offset
                    + slope * (ly - y)
                )

                vertices.append((
                    float(point[0]),
                    float(point[1]),
                    float(z)
                ))

                colors.append(color)

        faces = [
            (0, 1, 2), (0, 2, 3),
            (4, 6, 5), (4, 7, 6),
            (0, 4, 5), (0, 5, 1),
            (1, 5, 6), (1, 6, 2),
            (2, 6, 7), (2, 7, 3),
            (3, 7, 4), (3, 4, 0),
        ]

        for face in faces:

            triangles.append(tuple(
                start + i for i in face
            ))

    def _build_mesh(
        self,
        name,
        vertices,
        colors,
        triangles
    ):

        if not vertices or not triangles:
            return None

        data = GeomVertexData(
            name,
            GeomVertexFormat.getV3c4(),
            Geom.UHStatic
        )

        data.setNumRows(len(vertices))

        vertex_writer = GeomVertexWriter(
            data,
            "vertex"
        )

        color_writer = GeomVertexWriter(
            data,
            "color"
        )

        for vertex, color in zip(vertices, colors):

            vertex_writer.addData3f(
                float(vertex[0]),
                float(vertex[1]),
                float(vertex[2])
            )

            color_writer.addData4f(
                float(color[0]),
                float(color[1]),
                float(color[2]),
                float(color[3])
            )

        primitive = GeomTriangles(
            Geom.UHStatic
        )

        for triangle in triangles:
            primitive.addVertices(*triangle)

        primitive.closePrimitive()

        geometry = Geom(data)
        geometry.addPrimitive(primitive)

        node = GeomNode(name)
        node.addGeom(geometry)

        mesh = self.root.attachNewNode(node)

        mesh.setName(name)
        mesh.setTwoSided(True)

        return mesh


    def _add_seat(
        self,
        data,
        anchor,
        tangent,
        outward,
        x,
        y,
        z,
        color
    ):

        vertices, colors, triangles = data

        self._add_box(
            vertices,
            colors,
            triangles,
            anchor,
            tangent,
            outward,
            x,
            y,
            self.seat_width,
            self.seat_depth,
            z,
            self.seat_height,
            color
        )

        self._add_box(
            vertices,
            colors,
            triangles,
            anchor,
            tangent,
            outward,
            x,
            y + 0.27,
            self.seat_width,
            0.14,
            z + self.seat_height,
            self.backrest_height,
            color
        )


    def _create_base(
        self,
        data,
        anchor,
        tangent,
        outward,
        length
    ):

        vertices, colors, triangles = data

        self._add_box(
            vertices,
            colors,
            triangles,
            anchor,
            tangent,
            outward,
            0.0,
            0.0,
            length,
            self.stand_depth,
            0.0,
            0.8,
            self.concrete_color
        )

        self._add_box(
            vertices,
            colors,
            triangles,
            anchor,
            tangent,
            outward,
            0.0,
            -self.stand_depth / 2.0 + 0.25,
            length,
            0.5,
            0.0,
            1.1,
            self.concrete_light
        )

        self._add_box(
            vertices,
            colors,
            triangles,
            anchor,
            tangent,
            outward,
            0.0,
            self.stand_depth / 2.0 - 3.0,
            length - 8.0,
            5.0,
            0.8,
            1.2,
            self.concrete_light
        )

    def _create_seating(
        self,
        data,
        anchor,
        tangent,
        outward,
        length
    ):

        center_aisle = 4.5
        side_aisle = 2.0

        usable_length = length - 10.0

        sections = [
            (
                -usable_length / 2.0,
                -center_aisle / 2.0
            ),
            (
                center_aisle / 2.0,
                usable_length / 2.0
            )
        ]

        stair_positions = [
            -length / 3.0,
            length / 3.0
        ]

        for row in range(self.rows):

            row_y = (
                -self.stand_depth / 2.0
                + 4.0
                + row * self.row_pitch
            )

            row_z = (
                self.concourse_height
                + row * self.row_height
            )

            for start, end in sections:

                section_center = (
                    start + end
                ) / 2.0

                section_length = end - start

                self._add_box(
                    data[0],
                    data[1],
                    data[2],
                    anchor,
                    tangent,
                    outward,
                    section_center,
                    row_y,
                    section_length,
                    1.1,
                    row_z,
                    0.28,
                    self.concrete_light
                )

                x = (
                    start
                    + side_aisle
                    + self.seat_pitch / 2.0
                )

                while x < end - side_aisle:

                    blocked = any(
                        abs(x - stair) < 1.2
                        for stair in stair_positions
                    )

                    if not blocked:

                        if row % 7 == 0:
                            color = self.seat_cyan
                        elif row % 5 == 0:
                            color = self.seat_white
                        else:
                            color = self.seat_blue

                        self._add_seat(
                            data,
                            anchor,
                            tangent,
                            outward,
                            x,
                            row_y,
                            row_z + 0.28,
                            color
                        )

                    x += self.seat_pitch

        for stair_x in [
            0.0,
            -length / 3.0,
            length / 3.0
        ]:

            self._add_box(
                data[0],
                data[1],
                data[2],
                anchor,
                tangent,
                outward,
                stair_x,
                0.0,
                1.4,
                self.stand_depth - 5.0,
                self.concourse_height,
                0.10,
                self.concrete_color
            )

    def _create_supports(
        self,
        data,
        anchor,
        tangent,
        outward,
        length
    ):

        support_x = [
            -length / 2.0 + 5.0,
            -length / 4.0,
            0.0,
            length / 4.0,
            length / 2.0 - 5.0
        ]

        support_y = [7.0, 12.0]

        for x in support_x:

            for y in support_y:

                self._add_box(
                    data[0],
                    data[1],
                    data[2],
                    anchor,
                    tangent,
                    outward,
                    x,
                    y,
                    0.9,
                    0.9,
                    0.0,
                    self.roof_height - 0.5,
                    self.support_color
                )

        for y in support_y:

            self._add_box(
                data[0],
                data[1],
                data[2],
                anchor,
                tangent,
                outward,
                0.0,
                y,
                length + 5.0,
                1.0,
                self.roof_height - 1.0,
                1.0,
                self.support_color
            )

        for x in support_x:

            self._add_box(
                data[0],
                data[1],
                data[2],
                anchor,
                tangent,
                outward,
                x,
                9.5,
                0.8,
                13.0,
                self.roof_height - 0.5,
                0.7,
                self.support_color
            )

    def _create_roof(
        self,
        data,
        anchor,
        tangent,
        outward,
        length
    ):

        roof_length = (
            length + self.roof_overhang
        )

        roof_depth = 20.0

        self._add_box(
            data[0],
            data[1],
            data[2],
            anchor,
            tangent,
            outward,
            0.0,
            7.0,
            roof_length,
            roof_depth,
            self.roof_height,
            self.roof_thickness,
            self.roof_color,
            slope=0.025
        )

        self._add_box(
            data[0],
            data[1],
            data[2],
            anchor,
            tangent,
            outward,
            0.0,
            -3.0,
            roof_length + 2.0,
            0.8,
            self.roof_height,
            1.2,
            self.roof_edge_color
        )

        self._add_box(
            data[0],
            data[1],
            data[2],
            anchor,
            tangent,
            outward,
            0.0,
            17.0,
            roof_length,
            0.8,
            self.roof_height,
            1.0,
            self.roof_edge_color
        )

        for x in np.linspace(
            -length / 2.0,
            length / 2.0,
            7
        ):

            self._add_box(
                data[0],
                data[1],
                data[2],
                anchor,
                tangent,
                outward,
                float(x),
                7.0,
                0.55,
                19.0,
                self.roof_height - 0.1,
                0.4,
                self.support_color
            )

    def _create_vip_area(
        self,
        data,
        anchor,
        tangent,
        outward,
        length
    ):

        vip_length = length * 0.55

        self._add_box(
            data[0],
            data[1],
            data[2],
            anchor,
            tangent,
            outward,
            0.0,
            14.0,
            vip_length,
            3.0,
            1.0,
            2.0,
            self.vip_color
        )

        self._add_box(
            data[0],
            data[1],
            data[2],
            anchor,
            tangent,
            outward,
            0.0,
            14.0,
            vip_length - 5.0,
            2.5,
            3.0,
            2.0,
            self.concrete_light
        )

        self._add_box(
            data[0],
            data[1],
            data[2],
            anchor,
            tangent,
            outward,
            0.0,
            14.0,
            vip_length + 2.0,
            3.5,
            5.0,
            0.6,
            self.roof_edge_color
        )

        for x in np.linspace(
            -vip_length / 2.0 + 4.0,
            vip_length / 2.0 - 4.0,
            10
        ):

            self._add_box(
                data[0],
                data[1],
                data[2],
                anchor,
                tangent,
                outward,
                float(x),
                15.1,
                0.18,
                0.12,
                1.2,
                3.5,
                self.seat_white
            )

    def _check_spacing(self):

        (
            centerline,
            left_boundary,
            right_boundary,
            cumulative,
            total_length
        ) = self._get_track_data()

        positions = []

        for fraction, name, side, length in self.stand_specs:

            distance = (
                fraction % 1.0
            ) * total_length

            positions.append((
                distance,
                length,
                name
            ))

        positions.sort()

        for i in range(len(positions)):

            distance_a, length_a, name_a = positions[i]

            distance_b, length_b, name_b = positions[
                (i + 1) % len(positions)
            ]

            gap = (
                distance_b - distance_a
            ) % total_length

            required_gap = (
                length_a / 2.0
                + length_b / 2.0
                + 20.0
            )

            if gap < required_gap:

                print(
                    "[GRANDSTANDS] WARNING:",
                    name_a,
                    "and",
                    name_b,
                    "may overlap along track."
                )

            else:

                print(
                    "[GRANDSTANDS] Spacing OK:",
                    name_a,
                    "->",
                    name_b,
                    f"({gap:.1f} m)"
                )

    def _create_stand(
        self,
        fraction,
        name,
        side,
        length
    ):

        boundary, tangent, outward = self._sample_track(
            fraction,
            side
        )

        offset = (
            self.barrier_clearance
            + self.wall_width / 2.0
            + self.safety_gap
            + self.stand_depth / 2.0
        )

        anchor = (
            boundary + outward * offset
        )

        if name == "marina-grandstand":

            (
                anchor,
                length,
                offset
            ) = self._find_safe_marina_placement(
                boundary,
                tangent,
                outward,
                length
            )

        data = ([],[], [])

        self._create_base(
            data,
            anchor,
            tangent,
            outward,
            length
        )

        self._create_seating(
            data,
            anchor,
            tangent,
            outward,
            length
        )

        self._create_supports(
            data,
            anchor,
            tangent,
            outward,
            length
        )

        self._create_vip_area(
            data,
            anchor,
            tangent,
            outward,
            length
        )

        self._create_roof(
            data,
            anchor,
            tangent,
            outward,
            length
        )

        vertices, colors, triangles = data

        self._build_mesh(
            name,
            vertices,
            colors,
            triangles
        )


    def create(self):
        self.root = self.render.attachNewNode("yas-marina-grandstands")
        self._build_track_keepout()
        self._check_spacing()

        for fraction, name, side, length in self.stand_specs:

            self._create_stand(
                fraction,
                name,
                side,
                length
            )

        print("\n")
        print("GRANDSTANDS CREATED SUCCESSFULLY")
  
        return self.root
