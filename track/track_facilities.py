import math

import numpy as np

from shapely.geometry import Polygon, Point

from panda3d.core import (
    Geom,
    GeomNode,
    GeomTriangles,
    GeomVertexData,
    GeomVertexFormat,
    GeomVertexWriter,
    TransparencyAttrib,
    Material,
    LColor,
    PointLight,
    Vec3,
)


class TrackFacilities:

    def __init__(self, render, track):

        self.render = render
        self.track = track

        self.root = self.render.attachNewNode(
            "yas_marina_facilities"
        )

        self.pit_fraction = 0.025
        self.pit_side = 1

        self.hotel_fraction = 0.78
        self.hotel_side = -1

        self.occupied_sites = []

        self.track_polygon = self.build_track_polygon()

    # ---------------------------------------------------------
    # TRACK POLYGON AND SAFE POSITIONING
    # ---------------------------------------------------------

    def build_track_polygon(self):

        left = np.asarray(
            self.track.left_boundary_3d,
            dtype=float,
        )[:, :2]

        right = np.asarray(
            self.track.right_boundary_3d,
            dtype=float,
        )[:, :2]

        polygon = Polygon(
            list(map(tuple, left)) +
            list(map(tuple, right[::-1]))
        )

        if not polygon.is_valid:
            polygon = polygon.buffer(0)

        if polygon.is_empty:
            raise RuntimeError(
                "Could not construct the track polygon."
            )

        return polygon

    def get_track_frame(self, fraction, side, offset):

        points = np.asarray(
            self.track.centerline,
            dtype=float,
        )

        index = int(
            fraction * len(points)
        ) % len(points)

        center = points[index][:2]

        previous = points[
            (index - 1) % len(points)
        ][:2]

        following = points[
            (index + 1) % len(points)
        ][:2]

        tangent = following - previous
        length = np.linalg.norm(tangent)

        if length < 1e-8:
            tangent = np.array([1.0, 0.0])
        else:
            tangent /= length

        left = np.asarray(
            self.track.left_boundary_3d[index][:2]
        )

        right = np.asarray(
            self.track.right_boundary_3d[index][:2]
        )

        boundary = left if side == 1 else right

        outward = boundary - center
        length = np.linalg.norm(outward)

        if length < 1e-8:
            outward = np.array([
                -tangent[1],
                tangent[0],
            ]) * side
        else:
            outward /= length

        position = center + outward * offset

        heading = math.degrees(
            math.atan2(tangent[1], tangent[0])
        )

        return position, heading, tangent, outward

    def find_safe_site(
        self,
        fraction,
        preferred_side,
        clearance,
    ):

        offsets = range(100, 1801, 10)

        fraction_offsets = [
            0.0,
            0.01, -0.01,
            0.02, -0.02,
            0.03, -0.03,
            0.05, -0.05,
            0.075, -0.075,
            0.10, -0.10,
            0.15, -0.15,
            0.20, -0.20,
        ]

        best_distance = -1.0

        for fraction_offset in fraction_offsets:

            candidate_fraction = (
                fraction + fraction_offset
            ) % 1.0

            for side in (
                preferred_side,
                -preferred_side,
            ):

                for offset in offsets:

                    (
                        position,
                        heading,
                        tangent,
                        outward,
                    ) = self.get_track_frame(
                        candidate_fraction,
                        side,
                        offset,
                    )

                    point = Point(
                        float(position[0]),
                        float(position[1]),
                    )

                    distance = point.distance(
                        self.track_polygon
                    )

                    best_distance = max(
                        best_distance,
                        distance,
                    )

                    if distance >= clearance:

                        print(
                            "[Facilities] Hotel site found: "
                            f"fraction={candidate_fraction:.3f}, "
                            f"side={side}, "
                            f"offset={offset}, "
                            f"clearance={distance:.1f}m"
                        )

                        return (
                            position,
                            heading,
                            tangent,
                            outward,
                        )

        raise RuntimeError(
            "Could not find a safe hotel position. "
            f"Required clearance: {clearance}m; "
            f"best available: {best_distance:.1f}m."
        )

    # ---------------------------------------------------------
    # ADDITIONAL BUILDING POSITION SEARCH
    # ---------------------------------------------------------

    def find_additional_site(
        self,
        base_fraction,
        preferred_side,
        clearance=100.0,
        minimum_separation=190.0,
    ):

        fraction_offsets = [
            0.0,
            0.025, -0.025,
            0.05, -0.05,
            0.075, -0.075,
            0.10, -0.10,
            0.15, -0.15,
            0.20, -0.20,
            0.25, -0.25,
        ]

        sides = [
            preferred_side,
            -preferred_side,
        ]

        offsets = range(150, 1801, 20)

        for fraction_offset in fraction_offsets:

            candidate_fraction = (
                base_fraction + fraction_offset
            ) % 1.0

            for side in sides:

                for offset in offsets:

                    (
                        position,
                        heading,
                        tangent,
                        outward,
                    ) = self.get_track_frame(
                        candidate_fraction,
                        side,
                        offset,
                    )

                    candidate_point = Point(
                        float(position[0]),
                        float(position[1]),
                    )

                    track_clearance = (
                        candidate_point.distance(
                            self.track_polygon
                        )
                    )

                    if track_clearance < clearance:
                        continue

                    too_close = False

                    for occupied in self.occupied_sites:

                        distance = np.linalg.norm(
                            position - occupied
                        )

                        if distance < minimum_separation:
                            too_close = True
                            break

                    if too_close:
                        continue

                    print(
                        "[Facilities] Additional building site: "
                        f"fraction={candidate_fraction:.3f}, "
                        f"side={side}, "
                        f"offset={offset}, "
                        f"track clearance={track_clearance:.1f}m"
                    )

                    return (
                        position,
                        heading,
                        tangent,
                        outward,
                    )

        print(
            "[Facilities] Could not find a safe additional "
            f"building site near fraction {base_fraction:.3f}."
        )

        return None

    def make_anchor(self, name, position, heading):

        anchor = self.root.attachNewNode(name)

        anchor.setPos(
            float(position[0]),
            float(position[1]),
            0,
        )

        anchor.setH(float(heading))

        self.occupied_sites.append(
            np.asarray(position, dtype=float).copy()
        )

        return anchor

    # ---------------------------------------------------------
    # REAL LIGHT SOURCES
    # ---------------------------------------------------------

    def add_point_light(
        self,
        parent,
        name,
        position,
        color,
        attenuation=(1.0, 0.003, 0.00004),
    ):

        light = PointLight(name)

        light.setColor(
            LColor(
                color[0],
                color[1],
                color[2],
                1.0,
            )
        )

        light.setAttenuation(
            Vec3(
                attenuation[0],
                attenuation[1],
                attenuation[2],
            )
        )

        light_node = parent.attachNewNode(light)

        light_node.setPos(
            float(position[0]),
            float(position[1]),
            float(position[2]),
        )

        parent.setLight(light_node)

        return light_node

    def add_hotel_lighting(self, hotel):

        # FRONT - cool blue illumination.

        self.add_point_light(
            hotel,
            "hotel_front_blue_light_left",
            (-85, -58, 34),
            (0.30, 0.55, 1.0),
            attenuation=(1.0, 0.002, 0.000015),
        )

        self.add_point_light(
            hotel,
            "hotel_front_blue_light_center",
            (0, -58, 34),
            (0.30, 0.65, 1.0),
            attenuation=(1.0, 0.002, 0.000015),
        )

        self.add_point_light(
            hotel,
            "hotel_front_blue_light_right",
            (85, -58, 34),
            (0.30, 0.55, 1.0),
            attenuation=(1.0, 0.002, 0.000015),
        )

        # REAR - softer blue illumination.

        self.add_point_light(
            hotel,
            "hotel_rear_light_left",
            (-75, 55, 35),
            (0.20, 0.40, 0.85),
            attenuation=(1.0, 0.003, 0.00002),
        )

        self.add_point_light(
            hotel,
            "hotel_rear_light_center",
            (0, 55, 35),
            (0.25, 0.45, 0.90),
            attenuation=(1.0, 0.003, 0.00002),
        )

        self.add_point_light(
            hotel,
            "hotel_rear_light_right",
            (75, 55, 35),
            (0.20, 0.40, 0.85),
            attenuation=(1.0, 0.003, 0.00002),
        )

        # LEFT AND RIGHT SIDES.

        self.add_point_light(
            hotel,
            "hotel_left_side_light",
            (-140, 0, 32),
            (0.25, 0.50, 0.90),
            attenuation=(1.0, 0.002, 0.00002),
        )

        self.add_point_light(
            hotel,
            "hotel_right_side_light",
            (140, 0, 32),
            (0.25, 0.50, 0.90),
            attenuation=(1.0, 0.002, 0.00002),
        )

        # WINGS.

        self.add_point_light(
            hotel,
            "hotel_west_wing_light",
            (-65, -42, 48),
            (0.20, 0.50, 1.0),
            attenuation=(1.0, 0.002, 0.00002),
        )

        self.add_point_light(
            hotel,
            "hotel_east_wing_light",
            (65, -42, 48),
            (0.20, 0.50, 1.0),
            attenuation=(1.0, 0.002, 0.00002),
        )

        # WARM ENTRANCE LIGHT.

        self.add_point_light(
            hotel,
            "hotel_entrance_warm_light",
            (0, -67, 13),
            (1.0, 0.65, 0.30),
            attenuation=(1.0, 0.006, 0.00015),
        )

        # ROOF LIGHTING.

        self.add_point_light(
            hotel,
            "hotel_roof_light_left",
            (-75, 0, 68),
            (0.35, 0.55, 1.0),
            attenuation=(1.0, 0.003, 0.00002),
        )

        self.add_point_light(
            hotel,
            "hotel_roof_light_right",
            (75, 0, 68),
            (0.35, 0.55, 1.0),
            attenuation=(1.0, 0.003, 0.00002),
        )

    def add_building_perimeter_lighting(
        self,
        parent,
        name,
        x,
        y,
        width,
        depth,
        height,
    ):

        half_width = width / 2
        half_depth = depth / 2

        light_height = height * 0.60

        # Front.

        self.add_point_light(
            parent,
            f"{name}_front_light",
            (x, y - half_depth - 7, light_height),
            (0.40, 0.65, 1.0),
            attenuation=(1.0, 0.008, 0.00015),
        )

        # Rear.

        self.add_point_light(
            parent,
            f"{name}_rear_light",
            (x, y + half_depth + 7, light_height),
            (0.30, 0.50, 0.90),
            attenuation=(1.0, 0.008, 0.00015),
        )

        # Left side.

        self.add_point_light(
            parent,
            f"{name}_left_light",
            (x - half_width - 7, y, light_height),
            (0.35, 0.55, 1.0),
            attenuation=(1.0, 0.008, 0.00015),
        )

        # Right side.

        self.add_point_light(
            parent,
            f"{name}_right_light",
            (x + half_width + 7, y, light_height),
            (0.35, 0.55, 1.0),
            attenuation=(1.0, 0.008, 0.00015),
        )

        # Roof.

        self.add_point_light(
            parent,
            f"{name}_roof_light",
            (x, y, height + 5),
            (0.45, 0.60, 1.0),
            attenuation=(1.0, 0.006, 0.0001),
        )

    # ---------------------------------------------------------
    # EMISSIVE MATERIAL
    # ---------------------------------------------------------

    def apply_emission(self, node, color, intensity=2.0):

        material = Material()
        material.setName("facility_emissive_material")

        material.setEmission(
            LColor(
                color[0] * intensity,
                color[1] * intensity,
                color[2] * intensity,
                1.0,
            )
        )

        material.setAmbient(
            LColor(
                color[0] * 0.25,
                color[1] * 0.25,
                color[2] * 0.25,
                1.0,
            )
        )

        node.setMaterial(material, 1)
        node.setLightOff(1)

        return node

    # ---------------------------------------------------------
    # GEOMETRY HELPERS
    # ---------------------------------------------------------

    def make_box(
        self,
        parent,
        name,
        position,
        size,
        color,
        heading=0,
        alpha=1.0,
        emissive=False,
    ):

        sx, sy, sz = [
            float(v) / 2.0
            for v in size
        ]

        # A closed solid box with six faces.
        vertices = [
            (-sx, -sy, -sz),
            (sx, -sy, -sz),
            (sx, sy, -sz),
            (-sx, sy, -sz),
            (-sx, -sy, sz),
            (sx, -sy, sz),
            (sx, sy, sz),
            (-sx, sy, sz),
        ]

        faces = [
            (0, 1, 2, 3),
            (4, 7, 6, 5),
            (0, 4, 5, 1),
            (1, 5, 6, 2),
            (2, 6, 7, 3),
            (3, 7, 4, 0),
        ]

        fmt = GeomVertexFormat.getV3n3c4()

        data = GeomVertexData(
            name,
            fmt,
            Geom.UHStatic,
        )

        vertex_writer = GeomVertexWriter(
            data,
            "vertex",
        )

        normal_writer = GeomVertexWriter(
            data,
            "normal",
        )

        color_writer = GeomVertexWriter(
            data,
            "color",
        )

        triangles = GeomTriangles(
            Geom.UHStatic
        )

        for face in faces:

            a, b, c, d = face

            p0 = np.array(vertices[a])
            p1 = np.array(vertices[b])
            p2 = np.array(vertices[c])

            normal = np.cross(
                p1 - p0,
                p2 - p0,
            )

            length = np.linalg.norm(normal)

            if length > 0:
                normal /= length

            start = data.getNumRows()

            for index in face:

                vertex_writer.addData3(
                    *vertices[index]
                )

                normal_writer.addData3(
                    *normal
                )

                color_writer.addData4(
                    float(color[0]),
                    float(color[1]),
                    float(color[2]),
                    float(alpha),
                )

            triangles.addVertices(
                start,
                start + 1,
                start + 2,
            )

            triangles.addVertices(
                start,
                start + 2,
                start + 3,
            )

        geom = Geom(data)
        geom.addPrimitive(triangles)

        node = GeomNode(name)
        node.addGeom(geom)

        result = parent.attachNewNode(node)

        result.setPos(*position)
        result.setH(float(heading))

        if alpha < 1.0:

            result.setTransparency(
                TransparencyAttrib.MAlpha
            )

            result.setDepthWrite(False)

        if emissive:

            self.apply_emission(
                result,
                color,
                intensity=2.0,
            )

        return result

    def make_line(
        self,
        parent,
        name,
        points,
        color,
        thickness=0.25,
    ):

        from panda3d.core import LineSegs

        lines = LineSegs(name)

        lines.setThickness(thickness)
        lines.setColor(*color)

        if not points:
            return None

        lines.moveTo(*points[0])

        for point in points[1:]:
            lines.drawTo(*point)

        result = parent.attachNewNode(
            lines.create()
        )

        result.setLightOff(1)

        return result

    def make_ellipsoid(
        self,
        parent,
        name,
        position,
        radii,
        color,
        heading=0,
        segments=96,
    ):

        rx, ry, rz = radii

        rings = 20

        vertices = []
        faces = []

        for i in range(rings + 1):

            latitude = (
                -math.pi / 2
                + math.pi * i / rings
            )

            for j in range(segments):

                longitude = (
                    2 * math.pi * j / segments
                )

                x = (
                    rx
                    * math.cos(latitude)
                    * math.cos(longitude)
                )

                y = (
                    ry
                    * math.cos(latitude)
                    * math.sin(longitude)
                )

                z = rz * math.sin(latitude)

                vertices.append((x, y, z))

        for i in range(rings):

            for j in range(segments):

                a = i * segments + j

                b = (
                    i * segments
                    + (j + 1) % segments
                )

                c = (
                    (i + 1) * segments
                    + (j + 1) % segments
                )

                d = (
                    (i + 1) * segments
                    + j
                )

                faces.append((a, b, c))
                faces.append((a, c, d))

        fmt = GeomVertexFormat.getV3n3c4()

        data = GeomVertexData(
            name,
            fmt,
            Geom.UHStatic,
        )

        vertex_writer = GeomVertexWriter(
            data,
            "vertex",
        )

        normal_writer = GeomVertexWriter(
            data,
            "normal",
        )

        color_writer = GeomVertexWriter(
            data,
            "color",
        )

        triangles = GeomTriangles(
            Geom.UHStatic
        )

        for face in faces:

            start = data.getNumRows()

            for index in face:

                vertex = np.array(
                    vertices[index],
                    dtype=float,
                )

                normal = np.array([
                    vertex[0] / (rx * rx),
                    vertex[1] / (ry * ry),
                    vertex[2] / (rz * rz),
                ])

                length = np.linalg.norm(normal)

                if length > 0:
                    normal /= length

                vertex_writer.addData3(
                    *vertex
                )

                normal_writer.addData3(
                    *normal
                )

                color_writer.addData4(
                    *color,
                    1.0,
                )

            triangles.addVertices(
                start,
                start + 1,
                start + 2,
            )

        geom = Geom(data)
        geom.addPrimitive(triangles)

        node = GeomNode(name)
        node.addGeom(geom)

        result = parent.attachNewNode(node)

        result.setPos(*position)
        result.setH(float(heading))

        return result

    # ---------------------------------------------------------
    # HOTEL FACADE - DIAMOND GRID
    # ---------------------------------------------------------

    def create_hotel_lattice(
        self,
        parent,
        name,
        center_x,
        rx,
        ry,
        rz,
        center_z,
    ):

        palette = [
            (0.05, 0.35, 1.0, 1.0),
            (0.0, 0.85, 1.0, 1.0),
            (0.35, 0.15, 1.0, 1.0),
            (0.85, 0.08, 1.0, 1.0),
            (0.15, 0.55, 1.0, 1.0),
            (0.65, 0.25, 1.0, 1.0),
        ]

        u_min = -2.65
        u_max = 2.65

        v_min = -0.20
        v_max = 1.05

        # Small clearance so the lattice follows the shell.
        surface_offset = 0.25

        def surface(u, v):

            x = (
                center_x
                + (rx + surface_offset)
                * math.cos(v)
                * math.cos(u)
            )

            y = (
                (ry + surface_offset)
                * math.cos(v)
                * math.sin(u)
            )

            z = (
                center_z
                + (rz + surface_offset)
                * math.sin(v)
            )

            return (x, y, z)

        spacing = 0.19
        slope = 1.55
        samples = 90

        for direction in (-1, 1):

            intercept = -4.5
            line_index = 0

            while intercept <= 4.5:

                points = []

                for step in range(samples + 1):

                    v = (
                        v_min
                        + (v_max - v_min)
                        * step / samples
                    )

                    u = (
                        intercept
                        + direction * slope * v
                    )

                    if u_min <= u <= u_max:

                        points.append(
                            surface(u, v)
                        )

                if len(points) >= 2:

                    color = palette[
                        (line_index + direction * 2)
                        % len(palette)
                    ]

                    self.make_line(
                        parent,
                        (
                            f"{name}_diamond_"
                            f"{direction}_{line_index}"
                        ),
                        points,
                        color,
                        2.8,
                    )

                intercept += spacing
                line_index += 1

        for v, label in [
            (v_min, "lower"),
            (v_max, "upper"),
        ]:

            points = [
                surface(
                    u_min
                    + (u_max - u_min) * i / 120,
                    v,
                )
                for i in range(121)
            ]

            self.make_line(
                parent,
                f"{name}_outline_{label}",
                points,
                (0.15, 0.75, 1.0, 1.0),
                3.5,
            )

    # ---------------------------------------------------------
    # MAIN CREATION
    # ---------------------------------------------------------

    def create(self):

        self.create_pit_building()
        self.create_paddock()
        self.create_hotel()

        self.create_surrounding_buildings()

    # ---------------------------------------------------------
    # PIT BUILDING
    # ---------------------------------------------------------

    def create_pit_building(self):

        position, heading, _, _ = self.get_track_frame(
            self.pit_fraction,
            self.pit_side,
            48.0,
        )

        building = self.make_anchor(
            "yas_marina_pit_building",
            position,
            heading,
        )

        self.make_box(
            building,
            "pit_main",
            (0, 0, 12),
            (145, 30, 24),
            (0.80, 0.82, 0.85),
        )

        self.make_box(
            building,
            "pit_upper",
            (0, 0, 25),
            (145, 25, 4),
            (0.12, 0.14, 0.17),
        )

        self.make_box(
            building,
            "pit_roof",
            (0, 0, 28),
            (150, 33, 2),
            (0.08, 0.10, 0.13),
        )

        for i in range(12):

            x = -65 + i * 11.8

            self.make_box(
                building,
                f"garage_{i}",
                (x, -15.2, 7),
                (10, 0.5, 13),
                (0.08, 0.09, 0.11),
            )

            self.make_box(
                building,
                f"garage_light_{i}",
                (x, -15.6, 13),
                (7, 0.15, 0.3),
                (0.1, 0.8, 1.0),
                emissive=True,
            )

        for floor in range(3):

            self.make_box(
                building,
                f"pit_window_band_{floor}",
                (0, -15.3, 17 + floor * 3),
                (140, 0.3, 1.5),
                (0.15, 0.35, 0.48),
            )

    # ---------------------------------------------------------
    # PADDOCK
    # ---------------------------------------------------------

    def create_paddock(self):

        position, heading, _, _ = self.get_track_frame(
            self.pit_fraction,
            self.pit_side,
            105.0,
        )

        paddock = self.make_anchor(
            "yas_marina_paddock",
            position,
            heading,
        )

        self.make_box(
            paddock,
            "paddock_platform",
            (0, 0, 0.2),
            (170, 85, 0.4),
            (0.20, 0.22, 0.24),
        )

        team_colors = [
            (0.0, 0.65, 0.85),
            (0.9, 0.1, 0.1),
            (0.95, 0.75, 0.05),
            (0.1, 0.1, 0.12),
            (0.2, 0.55, 0.25),
        ]

        for i, color in enumerate(team_colors):

            x = -65 + i * 32

            self.make_box(
                paddock,
                f"team_hospitality_{i}",
                (x, 0, 5),
                (25, 35, 10),
                color,
            )

            self.make_box(
                paddock,
                f"team_roof_{i}",
                (x, 0, 10.5),
                (27, 37, 1),
                (0.12, 0.13, 0.15),
            )

    # ---------------------------------------------------------
    # W ABU DHABI HOTEL
    # ---------------------------------------------------------

    def create_hotel(self):

        position, heading, _, _ = self.find_safe_site(
            self.hotel_fraction,
            self.hotel_side,
            clearance=165.0,
        )

        hotel = self.make_anchor(
            "w_abu_dhabi_hotel",
            position,
            heading,
        )

        # Rotate the complete hotel 180 degrees.
        # Its world position remains unchanged.
        hotel.setH(float(heading + 180.0))

        self.make_box(
            hotel,
            "hotel_main_base",
            (0, 0, 10),
            (270, 100, 20),
            (0.09, 0.12, 0.17),
        )

        self.make_box(
            hotel,
            "hotel_front_glass",
            (0, -50.5, 12),
            (260, 1.0, 16),
            (0.08, 0.25, 0.38),
            alpha=0.95,
        )

        self.make_box(
            hotel,
            "hotel_rear_glass",
            (0, 50.5, 12),
            (260, 1.0, 16),
            (0.08, 0.25, 0.38),
            alpha=0.95,
        )

        self.make_box(
            hotel,
            "hotel_main_roof",
            (0, 0, 21),
            (275, 105, 2),
            (0.05, 0.07, 0.11),
        )

        self.make_box(
            hotel,
            "hotel_central_volume",
            (0, 0, 31),
            (145, 75, 20),
            (0.12, 0.16, 0.21),
        )

        self.make_box(
            hotel,
            "hotel_central_glass",
            (0, -38, 31),
            (135, 0.8, 17),
            (0.10, 0.32, 0.48),
            alpha=0.9,
        )

        self.make_box(
            hotel,
            "hotel_central_roof",
            (0, 0, 41),
            (150, 80, 2),
            (0.06, 0.08, 0.12),
        )

        wing_data = [
            {
                "name": "west",
                "center_x": -65,
            },
            {
                "name": "east",
                "center_x": 65,
            },
        ]

        for wing in wing_data:

            name = wing["name"]
            center_x = wing["center_x"]

            rx = 82
            ry = 54
            rz = 35
            center_z = 43

            self.make_ellipsoid(
                hotel,
                f"hotel_{name}_shell",
                (center_x, 0, center_z),
                (rx, ry, rz),
                (0.12, 0.16, 0.22),
                segments=96,
            )

            self.make_box(
                hotel,
                f"hotel_{name}_glass",
                (center_x, -47, 27),
                (130, 1.0, 17),
                (0.08, 0.30, 0.44),
                alpha=0.92,
            )

            self.create_hotel_lattice(
                hotel,
                f"hotel_{name}_lattice",
                center_x,
                rx,
                ry,
                rz,
                center_z,
            )

        led_colors = [
            (0.0, 0.55, 1.0),
            (0.35, 0.1, 1.0),
            (0.85, 0.05, 1.0),
            (0.0, 0.8, 1.0),
        ]

        for i, color in enumerate(led_colors):

            z = 19 + i * 1.2

            self.make_box(
                hotel,
                f"hotel_front_led_{i}",
                (0, -51.2, z),
                (250, 0.35, 0.45),
                color,
                emissive=True,
            )

        for i in range(9):

            z = 23 + i * 2

            color = led_colors[
                i % len(led_colors)
            ]

            self.make_box(
                hotel,
                f"hotel_central_led_{i}",
                (0, -38.7, z),
                (130, 0.3, 0.35),
                color,
                emissive=True,
            )

        self.make_box(
            hotel,
            "hotel_entrance",
            (0, -56, 9),
            (38, 12, 18),
            (0.06, 0.10, 0.15),
        )

        self.make_box(
            hotel,
            "hotel_entrance_glass",
            (0, -62.2, 9),
            (30, 0.4, 15),
            (0.10, 0.42, 0.58),
            alpha=0.95,
        )

        self.make_box(
            hotel,
            "hotel_entrance_canopy",
            (0, -58, 18.5),
            (46, 15, 1.5),
            (0.08, 0.12, 0.18),
        )

        self.make_box(
            hotel,
            "hotel_entrance_light",
            (0, -65.7, 18),
            (38, 0.35, 0.5),
            (0.05, 0.75, 1.0),
            emissive=True,
        )

        self.add_hotel_lighting(hotel)

        print(
            "[Facilities] Large illuminated "
            "W Abu Dhabi hotel created."
        )

    # ---------------------------------------------------------
    # BUILDING WINDOWS
    # ---------------------------------------------------------

    def add_building_windows(
        self,
        parent,
        prefix,
        width,
        depth,
        floors,
        window_color=(0.12, 0.42, 0.65),
        light_color=(0.75, 0.85, 1.0),
    ):

        spacing = 7.0

        count = max(
            2,
            int(width / spacing)
        )

        for floor in range(floors):

            z = 4.0 + floor * 4.5

            for i in range(count):

                x = (
                    -width / 2
                    + (i + 0.5) * width / count
                )

                self.make_box(
                    parent,
                    f"{prefix}_front_window_{floor}_{i}",
                    (x, -depth / 2 - 0.08, z),
                    (3.2, 0.16, 2.7),
                    window_color,
                )

                if (i + floor) % 3 == 0:

                    self.make_box(
                        parent,
                        f"{prefix}_front_light_{floor}_{i}",
                        (x, -depth / 2 - 0.18, z),
                        (2.2, 0.08, 1.9),
                        light_color,
                        emissive=True,
                    )

    # ---------------------------------------------------------
    # YAS PLAZA HOTEL COMPLEX
    # ---------------------------------------------------------

    def create_yas_plaza_complex(self):

        site = self.find_additional_site(
            base_fraction=0.36,
            preferred_side=1,
            clearance=105.0,
            minimum_separation=210.0,
        )

        if site is None:
            return

        position, heading, _, _ = site

        complex_root = self.make_anchor(
            "yas_plaza_hotel_complex",
            position,
            heading,
        )

        building_data = [
            {
                "name": "hotel_a",
                "x": -52,
                "y": 0,
                "width": 58,
                "depth": 43,
                "height": 35,
                "color": (0.16, 0.21, 0.28),
            },
            {
                "name": "hotel_b",
                "x": 18,
                "y": 9,
                "width": 65,
                "depth": 48,
                "height": 49,
                "color": (0.20, 0.24, 0.30),
            },
            {
                "name": "hotel_c",
                "x": 78,
                "y": -4,
                "width": 48,
                "depth": 40,
                "height": 29,
                "color": (0.13, 0.19, 0.26),
            },
        ]

        for data in building_data:

            name = data["name"]
            x = data["x"]
            y = data["y"]
            width = data["width"]
            depth = data["depth"]
            height = data["height"]

            self.make_box(
                complex_root,
                f"yas_plaza_{name}_main",
                (x, y, height / 2),
                (width, depth, height),
                data["color"],
            )

            self.make_box(
                complex_root,
                f"yas_plaza_{name}_roof",
                (x, y, height + 0.8),
                (width + 3, depth + 3, 1.6),
                (0.07, 0.10, 0.15),
            )

            self.make_box(
                complex_root,
                f"yas_plaza_{name}_glass",
                (x, y - depth / 2 - 0.12, height / 2),
                (width - 5, 0.25, height - 5),
                (0.08, 0.27, 0.42),
                alpha=0.95,
            )

            floors = max(
                2,
                int(height / 5.0)
            )

            self.add_building_windows(
                complex_root,
                f"yas_plaza_{name}",
                width - 5,
                depth,
                floors,
            )

            self.make_box(
                complex_root,
                f"yas_plaza_{name}_roof_light",
                (x, y - depth / 2 - 0.3, height + 1.8),
                (width - 8, 0.25, 0.35),
                (0.05, 0.65, 1.0),
                emissive=True,
            )

            self.add_building_perimeter_lighting(
                complex_root,
                f"yas_plaza_{name}",
                x,
                y,
                width,
                depth,
                height,
            )

        self.make_box(
            complex_root,
            "yas_plaza_entrance_platform",
            (15, -42, 0.35),
            (160, 30, 0.7),
            (0.17, 0.19, 0.22),
        )

        for i in range(8):

            x = -52 + i * 19

            self.make_box(
                complex_root,
                f"yas_plaza_entrance_light_{i}",
                (x, -56, 1.0),
                (5, 0.4, 0.25),
                (0.0, 0.65, 1.0),
                emissive=True,
            )

        print(
            "[Facilities] Yas Plaza-inspired hotel "
            "complex created."
        )

    # ---------------------------------------------------------
    # MODERN ARENA
    # ---------------------------------------------------------

    def create_modern_arena(self):

        site = self.find_additional_site(
            base_fraction=0.53,
            preferred_side=-1,
            clearance=110.0,
            minimum_separation=220.0,
        )

        if site is None:
            return

        position, heading, _, _ = site

        arena = self.make_anchor(
            "yas_island_modern_arena",
            position,
            heading,
        )

        self.make_ellipsoid(
            arena,
            "arena_main_shell",
            (0, 0, 17),
            (82, 58, 17),
            (0.18, 0.21, 0.25),
            segments=96,
        )

        self.make_box(
            arena,
            "arena_base",
            (0, 0, 5),
            (145, 95, 10),
            (0.12, 0.15, 0.19),
        )

        self.make_box(
            arena,
            "arena_front_glass",
            (0, -48, 13),
            (105, 1.0, 16),
            (0.08, 0.28, 0.42),
            alpha=0.95,
        )

        self.make_ellipsoid(
            arena,
            "arena_roof",
            (0, 0, 30),
            (86, 62, 5),
            (0.08, 0.11, 0.15),
            segments=96,
        )

        colors = [
            (0.0, 0.55, 1.0),
            (0.0, 0.8, 1.0),
            (0.25, 0.4, 1.0),
            (0.0, 0.65, 1.0),
            (0.0, 0.45, 0.9),
        ]

        for i, color in enumerate(colors):

            z = 8 + i * 4

            self.make_box(
                arena,
                f"arena_facade_light_{i}",
                (0, -49, z),
                (110, 0.35, 0.35),
                color,
                emissive=True,
            )

        self.make_box(
            arena,
            "arena_entrance_canopy",
            (0, -60, 20),
            (48, 20, 2),
            (0.08, 0.12, 0.17),
        )

        self.make_box(
            arena,
            "arena_entrance_led",
            (0, -70.2, 20),
            (42, 0.3, 0.5),
            (0.0, 0.8, 1.0),
            emissive=True,
        )

        self.make_box(
            arena,
            "arena_roof_front_light",
            (0, -55, 30),
            (125, 0.4, 0.45),
            (0.1, 0.65, 1.0),
            emissive=True,
        )

        # Illumination from all four sides.

        self.add_point_light(
            arena,
            "arena_front_illumination",
            (0, -58, 18),
            (0.20, 0.55, 1.0),
            attenuation=(1.0, 0.003, 0.00004),
        )

        self.add_point_light(
            arena,
            "arena_rear_illumination",
            (0, 58, 18),
            (0.20, 0.45, 0.90),
            attenuation=(1.0, 0.003, 0.00004),
        )

        self.add_point_light(
            arena,
            "arena_left_illumination",
            (-85, 0, 18),
            (0.25, 0.50, 1.0),
            attenuation=(1.0, 0.003, 0.00004),
        )

        self.add_point_light(
            arena,
            "arena_right_illumination",
            (85, 0, 18),
            (0.25, 0.50, 1.0),
            attenuation=(1.0, 0.003, 0.00004),
        )

        print(
            "[Facilities] Etihad Arena-inspired "
            "building created."
        )

    # ---------------------------------------------------------
    # YAS MALL
    # ---------------------------------------------------------

    def create_shopping_complex(self):

        site = self.find_additional_site(
            base_fraction=0.14,
            preferred_side=-1,
            clearance=115.0,
            minimum_separation=230.0,
        )

        if site is None:
            return

        position, heading, _, _ = site

        mall = self.make_anchor(
            "yas_island_shopping_complex",
            position,
            heading,
        )

        self.make_box(
            mall,
            "yas_mall_main",
            (0, 0, 12),
            (190, 115, 24),
            (0.16, 0.19, 0.23),
        )

        self.make_box(
            mall,
            "yas_mall_roof",
            (0, 0, 25),
            (198, 123, 2),
            (0.07, 0.09, 0.13),
        )

        self.make_box(
            mall,
            "yas_mall_front_glass",
            (0, -58, 12),
            (175, 0.7, 17),
            (0.08, 0.30, 0.45),
            alpha=0.95,
        )

        for side in (-1, 1):

            self.make_box(
                mall,
                f"yas_mall_side_glass_{side}",
                (side * 95.5, 0, 12),
                (0.7, 95, 17),
                (0.08, 0.26, 0.40),
                alpha=0.95,
            )

        for i in range(5):

            z = 5 + i * 4

            self.make_box(
                mall,
                f"yas_mall_front_led_{i}",
                (0, -58.6, z),
                (165, 0.25, 0.35),
                (0.0, 0.55, 1.0),
                emissive=True,
            )

        self.make_box(
            mall,
            "yas_mall_entrance",
            (0, -62, 8),
            (52, 8, 16),
            (0.06, 0.11, 0.16),
        )

        self.make_box(
            mall,
            "yas_mall_entrance_glass",
            (0, -66.2, 8),
            (42, 0.3, 13),
            (0.10, 0.38, 0.55),
            alpha=0.95,
        )

        self.make_box(
            mall,
            "yas_mall_entrance_light",
            (0, -66.5, 16),
            (48, 0.35, 0.45),
            (0.0, 0.75, 1.0),
            emissive=True,
        )

        self.make_box(
            mall,
            "yas_mall_forecourt",
            (0, -90, 0.25),
            (180, 42, 0.5),
            (0.20, 0.22, 0.24),
        )

        for i in range(9):

            x = -80 + i * 20

            self.make_box(
                mall,
                f"yas_mall_forecourt_light_{i}",
                (x, -90, 0.8),
                (0.4, 30, 0.15),
                (0.0, 0.55, 1.0),
                emissive=True,
            )

        # Four-sided mall illumination.

        self.add_point_light(
            mall,
            "yas_mall_front_illumination",
            (0, -68, 15),
            (0.25, 0.55, 1.0),
            attenuation=(1.0, 0.003, 0.00004),
        )

        self.add_point_light(
            mall,
            "yas_mall_rear_illumination",
            (0, 68, 15),
            (0.25, 0.45, 0.90),
            attenuation=(1.0, 0.003, 0.00004),
        )

        self.add_point_light(
            mall,
            "yas_mall_left_illumination",
            (-105, 0, 15),
            (0.25, 0.50, 1.0),
            attenuation=(1.0, 0.003, 0.00004),
        )

        self.add_point_light(
            mall,
            "yas_mall_right_illumination",
            (105, 0, 15),
            (0.25, 0.50, 1.0),
            attenuation=(1.0, 0.003, 0.00004),
        )

        self.add_point_light(
            mall,
            "yas_mall_roof_illumination",
            (0, 0, 35),
            (0.40, 0.55, 1.0),
            attenuation=(1.0, 0.004, 0.00006),
        )

        print(
            "[Facilities] Yas Mall-inspired "
            "shopping complex created."
        )

    # ---------------------------------------------------------
    # DISTANT CITY SKYLINE
    # ---------------------------------------------------------

    def create_city_skyline(self):

        site = self.find_additional_site(
            base_fraction=0.91,
            preferred_side=1,
            clearance=125.0,
            minimum_separation=240.0,
        )

        if site is None:
            return

        position, heading, _, _ = site

        skyline = self.make_anchor(
            "yas_island_city_skyline",
            position,
            heading,
        )

        buildings = [
            (-100, 0, 34, 30, 24),
            (-58, 8, 27, 27, 38),
            (-15, -4, 38, 31, 55),
            (32, 8, 31, 28, 43),
            (75, -5, 42, 32, 62),
            (120, 10, 29, 27, 35),
        ]

        colors = [
            (0.12, 0.17, 0.23),
            (0.16, 0.20, 0.27),
            (0.10, 0.15, 0.22),
            (0.17, 0.20, 0.26),
            (0.12, 0.16, 0.23),
            (0.15, 0.19, 0.25),
        ]

        for i, building_data in enumerate(buildings):

            x, y, width, depth, height = building_data

            self.make_box(
                skyline,
                f"skyline_building_{i}",
                (x, y, height / 2),
                (width, depth, height),
                colors[i],
            )

            self.make_box(
                skyline,
                f"skyline_glass_{i}",
                (x, y - depth / 2 - 0.12, height / 2),
                (width - 4, 0.2, height - 4),
                (0.08, 0.25, 0.38),
            )

            self.make_box(
                skyline,
                f"skyline_roof_{i}",
                (x, y, height + 0.6),
                (width + 1, depth + 1, 1.2),
                (0.06, 0.08, 0.12),
            )

            floors = max(
                2,
                int(height / 5.0)
            )

            window_count = max(
                3,
                int(width / 6.0)
            )

            for floor in range(floors):

                z = 4 + floor * 5

                for window in range(window_count):

                    window_x = (
                        x - width / 2
                        + (window + 0.5) * width / window_count
                    )

                    if (window + floor + i) % 3 != 0:
                        continue

                    self.make_box(
                        skyline,
                        (
                            f"skyline_window_{i}_"
                            f"{floor}_{window}"
                        ),
                        (
                            window_x,
                            y - depth / 2 - 0.25,
                            z,
                        ),
                        (1.8, 0.12, 2.4),
                        (0.75, 0.82, 1.0),
                        emissive=True,
                    )

            self.make_box(
                skyline,
                f"skyline_vertical_light_{i}",
                (
                    x - width / 2 + 1.5,
                    y - depth / 2 - 0.25,
                    height / 2,
                ),
                (0.25, 0.2, height - 4),
                (0.0, 0.55, 0.9),
                emissive=True,
            )

            self.add_building_perimeter_lighting(
                skyline,
                f"skyline_building_{i}",
                x,
                y,
                width,
                depth,
                height,
            )

        self.add_point_light(
            skyline,
            "skyline_blue_illumination",
            (0, -25, 30),
            (0.20, 0.40, 0.85),
            attenuation=(1.0, 0.003, 0.00005),
        )

        print(
            "[Facilities] Distant Yas Island-inspired "
            "city skyline created."
        )

    # ---------------------------------------------------------
    # CREATE ALL ADDITIONAL BUILDINGS
    # ---------------------------------------------------------

    def create_surrounding_buildings(self):

        print("\nCREATING YAS ISLAND SURROUNDING BUILDINGS")
        print("-" * 50)

        self.create_yas_plaza_complex()
        self.create_modern_arena()
        self.create_shopping_complex()
        self.create_city_skyline()

        print(
            "[Facilities] Surrounding architecture pass complete."
        )

    # ---------------------------------------------------------
    # CLEANUP
    # ---------------------------------------------------------

    def remove(self):

        self.root.removeNode()

        self.root = self.render.attachNewNode(
            "yas_marina_facilities"
        )

        self.occupied_sites.clear()