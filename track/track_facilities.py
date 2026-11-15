import math
import numpy as np
from shapely import affinity
from shapely.geometry import Polygon, Point, box
from shapely.ops import nearest_points
from shapely.prepared import prep
from panda3d.core import (ColorBlendAttrib, Geom, GeomNode, GeomTriangles, GeomVertexData, GeomVertexFormat, GeomVertexWriter, LineSegs, TransparencyAttrib, Material, LColor, PointLight, Vec3,)

class TrackFacilities:
    BOX_FACES = (
        ((0, 3, 2, 1), (0, 0, -1)),
        ((4, 5, 6, 7), (0, 0, 1)),
        ((0, 1, 5, 4), (0, -1, 0)),
        ((1, 2, 6, 5), (1, 0, 0)),
        ((2, 3, 7, 6), (0, 1, 0)),
        ((3, 0, 4, 7), (-1, 0, 0)),
    )

    HOTEL_PALETTE = [
        (0.10, 0.55, 1.00),
        (0.00, 0.90, 1.00),
        (0.45, 0.25, 1.00),
        (0.90, 0.20, 1.00),
    ]

    TOWER_PALETTE = [
        (0.10, 0.65, 1.00),
        (0.00, 0.85, 1.00),
        (0.35, 0.45, 1.00),
    ]

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
        self.occupied_footprints = []

        self.track_gap = 50.0
        self.building_gap = 40.0
        self._zone_cache = {}

        self.track_polygon = self.build_track_polygon()


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

    def footprint_polygon(
        self,
        position,
        heading,
        x_min,
        x_max,
        y_min,
        y_max,
    ):

        rect = box(x_min, y_min, x_max, y_max)

        rect = affinity.rotate(
            rect,
            float(heading),
            origin=(0, 0),
        )

        rect = affinity.translate(
            rect,
            float(position[0]),
            float(position[1]),
        )

        return rect

    def heading_toward_track(self, position):
        point = Point(float(position[0]), float(position[1]))

        _, nearest = nearest_points(point, self.track_polygon)

        to_track = np.array([
            nearest.x - point.x,
            nearest.y - point.y,
        ])

        length = np.linalg.norm(to_track)

        if length < 1e-6:
            return None

        to_track /= length

        return math.degrees(
            math.atan2(to_track[0], -to_track[1])
        )

    def _track_zone(self, gap):

        key = round(float(gap), 1)

        if key not in self._zone_cache:
            self._zone_cache[key] = prep(
                self.track_polygon.buffer(gap)
            )

        return self._zone_cache[key]

    def find_building_site(
        self,
        base_fraction,
        preferred_side,
        extents,
        track_gap=None,
        building_gap=None,
    ):
  
        track_gap = self.track_gap if track_gap is None else track_gap

        building_gap = (
            self.building_gap
            if building_gap is None
            else building_gap
        )

        x_min, x_max, y_min, y_max = extents

        fraction_offsets = [
            0.0,
            0.01, -0.01,
            0.025, -0.025,
            0.05, -0.05,
            0.075, -0.075,
            0.10, -0.10,
            0.15, -0.15,
            0.20, -0.20,
            0.25, -0.25,
        ]

        offsets = range(60, 2001, 20)

        # If no site is found, the track gap is reduced step by step.
        for gap in (track_gap, track_gap * 0.6, track_gap * 0.3):

            zone = self._track_zone(gap)

            for fraction_offset in fraction_offsets:

                fraction = (base_fraction + fraction_offset) % 1.0

                for side in (preferred_side, -preferred_side):

                    for offset in offsets:

                        position, _, _, _ = self.get_track_frame(
                            fraction,
                            side,
                            offset,
                        )

                        point = Point(
                            float(position[0]),
                            float(position[1]),
                        )

                        if zone.contains(point):
                            continue

                        heading = self.heading_toward_track(position)

                        if heading is None:
                            continue

                        footprint = self.footprint_polygon(
                            position,
                            heading,
                            x_min,
                            x_max,
                            y_min,
                            y_max,
                        )

                        # Whole footprint must stay away from the track.
                        if zone.intersects(footprint):
                            continue

                        # ...and away from other buildings.
                        if any(
                            footprint.distance(other) < building_gap
                            for other in self.occupied_footprints
                        ):
                            continue

                        print(
                            "[Facilities] Site found: "
                            f"fraction={fraction:.3f}, side={side}, "
                            f"offset={offset}, heading={heading:.1f}, "
                            f"track gap={gap:.0f}m"
                        )

                        return position, heading, footprint

        print(
            "[Facilities] No valid site near fraction "
            f"{base_fraction:.3f}."
        )

        return None

    def make_anchor(
        self,
        name,
        position,
        heading,
        footprint=None,
        check_facing=False,
    ):

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

        if footprint is not None:
            self.occupied_footprints.append(footprint)

        if check_facing:

            point = Point(float(position[0]), float(position[1]))

            _, nearest = nearest_points(point, self.track_polygon)

            wanted = np.array([
                nearest.x - point.x,
                nearest.y - point.y,
            ])

            wanted /= max(np.linalg.norm(wanted), 1e-8)

            front = self.root.getRelativeVector(
                anchor,
                Vec3(0, -1, 0),
            )

            dot = front.x * wanted[0] + front.y * wanted[1]

            print(
                f"[Facilities] {name}: facing-track check "
                f"= {dot:.2f} (1.0 = faces the track)"
            )

        return anchor

    def add_point_light(
        self,
        parent,
        name,
        position,
        color,
        attenuation=(1.0, 0.003, 0.00004),
        intensity=1.0,
    ):

        light = PointLight(name)

        light.setColor(
            LColor(
                color[0] * intensity,
                color[1] * intensity,
                color[2] * intensity,
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

        specs = [
            ("front", (0, -80, 36), (0.30, 0.60, 1.0)),
            ("rear", (0, 80, 36), (0.25, 0.45, 0.95)),
            ("left", (-165, 0, 36), (0.25, 0.50, 0.95)),
            ("right", (165, 0, 36), (0.25, 0.50, 0.95)),
            ("roof", (0, 0, 100), (0.35, 0.55, 1.0)),
        ]

        for label, position, color in specs:

            self.add_point_light(
                hotel,
                f"hotel_{label}_light",
                position,
                color,
                attenuation=(1.0, 0.002, 0.000015),
                intensity=1.8,
            )

        # Warm entrance light.

        self.add_point_light(
            hotel,
            "hotel_entrance_warm_light",
            (0, -70, 13),
            (1.0, 0.65, 0.30),
            attenuation=(1.0, 0.006, 0.00015),
            intensity=1.5,
        )

    def add_tower_lighting(
        self,
        tower,
        width,
        depth,
        height,
    ):

        h = height * 0.55

        specs = [
            ("front", (0, -depth / 2 - 9, h), (0.40, 0.65, 1.0)),
            ("rear", (0, depth / 2 + 9, h), (0.30, 0.50, 0.95)),
            ("left", (-width / 2 - 9, 0, h), (0.35, 0.55, 1.0)),
            ("right", (width / 2 + 9, 0, h), (0.35, 0.55, 1.0)),
            ("roof", (0, 0, height + 8), (0.45, 0.60, 1.0)),
        ]

        for label, position, color in specs:

            self.add_point_light(
                tower,
                f"{tower.getName()}_{label}_light",
                position,
                color,
                attenuation=(1.0, 0.008, 0.00015),
                intensity=1.6,
            )

 
    def _make_unlit(self, node, boost=1.35):

        node.setLightOff(1)
        node.setShaderOff(1)
        node.setColorScale(boost, boost, boost, 1.0)

    def _set_additive(self, node):

        node.setTransparency(TransparencyAttrib.MAlpha)
        node.setDepthWrite(False)

        node.setAttrib(
            ColorBlendAttrib.make(
                ColorBlendAttrib.MAdd,
                ColorBlendAttrib.OIncomingAlpha,
                ColorBlendAttrib.OOne,
            )
        )

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
        self._make_unlit(node)

        return node

    def _build_boxes_node(self, name, boxes, alpha=1.0):
        """
        boxes = list of (center, size, color).
        All boxes are merged in ONE geom, with outward facing
        triangles and outward normals.
        """

        data = GeomVertexData(
            name,
            GeomVertexFormat.getV3n3c4(),
            Geom.UHStatic,
        )

        vertex_writer = GeomVertexWriter(data, "vertex")
        normal_writer = GeomVertexWriter(data, "normal")
        color_writer = GeomVertexWriter(data, "color")

        triangles = GeomTriangles(Geom.UHStatic)

        for center, size, color in boxes:

            cx, cy, cz = [float(v) for v in center]
            sx, sy, sz = [float(v) / 2.0 for v in size]

            vertices = [
                (cx - sx, cy - sy, cz - sz),
                (cx + sx, cy - sy, cz - sz),
                (cx + sx, cy + sy, cz - sz),
                (cx - sx, cy + sy, cz - sz),
                (cx - sx, cy - sy, cz + sz),
                (cx + sx, cy - sy, cz + sz),
                (cx + sx, cy + sy, cz + sz),
                (cx - sx, cy + sy, cz + sz),
            ]

            for face, normal in self.BOX_FACES:

                start = data.getNumRows()

                for index in face:

                    vertex_writer.addData3(*vertices[index])

                    normal_writer.addData3(
                        float(normal[0]),
                        float(normal[1]),
                        float(normal[2]),
                    )

                    color_writer.addData4(
                        float(color[0]),
                        float(color[1]),
                        float(color[2]),
                        float(alpha),
                    )

                triangles.addVertices(start, start + 1, start + 2)
                triangles.addVertices(start, start + 2, start + 3)

        geom = Geom(data)
        geom.addPrimitive(triangles)

        node = GeomNode(name)
        node.addGeom(geom)

        return node

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

        node = self._build_boxes_node(
            name,
            [((0, 0, 0), size, color)],
            alpha,
        )

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

    def make_box_batch(
        self,
        parent,
        name,
        boxes,
        emissive=False,
        alpha=1.0,
        additive=False,
    ):

        if not boxes:
            return None

        node = self._build_boxes_node(name, boxes, alpha)

        result = parent.attachNewNode(node)

        if alpha < 1.0:

            result.setTransparency(
                TransparencyAttrib.MAlpha
            )

            result.setDepthWrite(False)

        if emissive:
            self._make_unlit(result)

        if additive:
            self._set_additive(result)

        return result

    def make_glow_batch(
        self,
        parent,
        name,
        boxes,
        halo=0.45,
        halo_alpha=0.20,
    ):
        """Bright unlit core + larger additive halo = glowing strips."""

        core = self.make_box_batch(
            parent,
            name,
            boxes,
            emissive=True,
        )

        grown = [
            (
                center,
                (
                    size[0] + halo * 2,
                    size[1] + halo * 2,
                    size[2] + halo * 2,
                ),
                color,
            )
            for center, size, color in boxes
        ]

        glow = self.make_box_batch(
            parent,
            f"{name}_halo",
            grown,
            emissive=True,
            alpha=halo_alpha,
            additive=True,
        )

        return core, glow

    def make_line(
        self,
        parent,
        name,
        points,
        color,
        thickness=0.25,
    ):

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
        result.setShaderOff(1)

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
        alpha=1.0,
        glow=False,
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

        vertex_writer = GeomVertexWriter(data, "vertex")
        normal_writer = GeomVertexWriter(data, "normal")
        color_writer = GeomVertexWriter(data, "color")

        triangles = GeomTriangles(Geom.UHStatic)

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
                    float(vertex[0]),
                    float(vertex[1]),
                    float(vertex[2]),
                )

                normal_writer.addData3(
                    float(normal[0]),
                    float(normal[1]),
                    float(normal[2]),
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

        if glow:

            self._make_unlit(result, 1.0)
            self._set_additive(result)

        return result

    def add_glow_volume(
        self,
        parent,
        name,
        cx,
        cy,
        width,
        depth,
        z0,
        z1,
        led_levels,
        palette,
        offset_y=0.15,
        offset_x=0.15,
        overhang=0.0,
        roof_z=None,
        window_start=3.0,
        window_step=4.5,
    ):
        """
        LED bands, lit windows, glowing corner columns and a glowing
        roof outline on ALL FOUR sides of a rectangular volume.
        """

        leds = []
        lit = []
        dark = []

        faces = (
            (0, -1, width, offset_y),   # front  (-Y)
            (0, 1, width, offset_y),    # rear   (+Y)
            (-1, 0, depth, offset_x),   # left   (-X)
            (1, 0, depth, offset_x),    # right  (+X)
        )

        for face_index, (nx, ny, span, off) in enumerate(faces):

            half = depth / 2.0 if ny else width / 2.0

            # LED bands.

            for k, z in enumerate(led_levels):

                color = palette[(k + face_index) % len(palette)]

                if ny:
                    leds.append((
                        (cx, cy + ny * (half + off + 0.12), z),
                        (span * 0.96, 0.3, 0.45),
                        color,
                    ))
                else:
                    leds.append((
                        (cx + nx * (half + off + 0.12), cy, z),
                        (0.3, span * 0.96, 0.45),
                        color,
                    ))

            # Windows.

            count = max(3, int(span / 6.0))

            floor = 0
            z = z0 + window_start

            while z < z1 - 2.0:

                for i in range(count):

                    u = -span / 2 + (i + 0.5) * span / count

                    if ny:
                        position = (cx + u, cy + ny * (half + off), z)
                        size = (3.0, 0.16, 2.4)
                    else:
                        position = (cx + nx * (half + off), cy + u, z)
                        size = (0.16, 3.0, 2.4)

                    if (i * 3 + floor * 5 + face_index * 2) % 4 != 0:

                        if (i + floor) % 5 == 0:
                            color = (1.0, 0.80, 0.45)
                        else:
                            color = (0.75, 0.90, 1.0)

                        lit.append((position, size, color))

                    else:

                        dark.append((
                            position,
                            size,
                            (0.06, 0.16, 0.26),
                        ))

                floor += 1
                z += window_step

        # Corner columns.

        corner_height = z1 - z0

        for sx in (-1, 1):
            for sy in (-1, 1):

                leds.append((
                    (
                        cx + sx * (width / 2 + offset_x + 0.1),
                        cy + sy * (depth / 2 + offset_y + 0.1),
                        (z0 + z1) / 2,
                    ),
                    (0.6, 0.6, corner_height),
                    palette[0],
                ))

        # Roof outline.

        rz = roof_z if roof_z is not None else z1 + 0.3

        rw = width + overhang * 2
        rd = depth + overhang * 2

        for sy in (-1, 1):
            leds.append((
                (cx, cy + sy * rd / 2, rz),
                (rw, 0.5, 0.5),
                palette[1 % len(palette)],
            ))

        for sx in (-1, 1):
            leds.append((
                (cx + sx * rw / 2, cy, rz),
                (0.5, rd, 0.5),
                palette[1 % len(palette)],
            ))

        self.make_glow_batch(parent, f"{name}_leds", leds)

        self.make_box_batch(
            parent,
            f"{name}_windows_lit",
            lit,
            emissive=True,
        )

        self.make_box_batch(
            parent,
            f"{name}_windows_dark",
            dark,
            emissive=False,
        )

    def create_glow_tower(
        self,
        parent,
        name,
        x,
        y,
        width,
        depth,
        height,
        body_color,
        lights=True,
    ):
        """Building glowing on all four sides, with its own lights."""

        tower = parent.attachNewNode(name)
        tower.setPos(float(x), float(y), 0.0)

        self.make_box(
            tower,
            f"{name}_main",
            (0, 0, height / 2),
            (width, depth, height),
            body_color,
        )

        self.make_box(
            tower,
            f"{name}_roof",
            (0, 0, height + 0.8),
            (width + 3, depth + 3, 1.6),
            (0.07, 0.10, 0.15),
        )

        levels = [
            round(height * f, 1)
            for f in (0.25, 0.50, 0.75, 0.96)
        ]

        self.add_glow_volume(
            tower,
            name,
            0,
            0,
            width,
            depth,
            0,
            height,
            levels,
            self.TOWER_PALETTE,
            overhang=1.5,
            roof_z=height + 1.8,
        )

        if lights:
            self.add_tower_lighting(
                tower,
                width,
                depth,
                height,
            )

        return tower

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

        # Full circumference (all sides of the dome).
        u_min = -math.pi
        u_max = math.pi

        v_min = -0.20
        v_max = 1.05

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

            intercept = -5.0
            line_index = 0

            while intercept <= 5.0:

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
                        3.0,
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
                (0.15, 0.85, 1.0, 1.0),
                4.0,
            )

    def create_hotel_rings(
        self,
        parent,
        name,
        center_x,
        rx,
        ry,
        rz,
        center_z,
        levels,
        color=(0.2, 0.85, 1.0, 1.0),
        thickness=4.0,
    ):
        """Full glowing rings around the dome (all sides)."""

        for k, v in enumerate(levels):

            points = []

            for i in range(145):

                u = -math.pi + 2 * math.pi * i / 144

                points.append((
                    center_x + (rx + 0.3) * math.cos(v) * math.cos(u),
                    (ry + 0.3) * math.cos(v) * math.sin(u),
                    center_z + (rz + 0.3) * math.sin(v),
                ))

            self.make_line(
                parent,
                f"{name}_ring_{k}",
                points,
                color,
                thickness,
            )


    def create(self):

        self.create_pit_building()
        self.create_paddock()
        self.create_hotel()

        self.create_surrounding_buildings()

 
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
            footprint=self.footprint_polygon(
                position, heading, -80, 80, -20, 20
            ),
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
            footprint=self.footprint_polygon(
                position, heading, -90, 90, -45, 45
            ),
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


    def create_hotel(self):

        site = self.find_building_site(
            self.hotel_fraction,
            self.hotel_side,
            extents=(-150, 150, -70, 56),
        )

        if site is None:
            raise RuntimeError(
                "Could not find a safe position for the hotel."
            )

        position, heading, footprint = site

        # Facade (-Y side, entrance) looks toward the track.
        hotel = self.make_anchor(
            "w_abu_dhabi_hotel",
            position,
            heading,
            footprint=footprint,
            check_facing=True,
        )

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

            # Soft additive glow around the dome.
            self.make_ellipsoid(
                hotel,
                f"hotel_{name}_halo",
                (center_x, 0, center_z),
                (rx * 1.035, ry * 1.035, rz * 1.035),
                (0.15, 0.45, 1.0),
                segments=64,
                alpha=0.10,
                glow=True,
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

            self.create_hotel_rings(
                hotel,
                f"hotel_{name}",
                center_x,
                rx,
                ry,
                rz,
                center_z,
                levels=[-0.64, -0.50, -0.36],
            )

        # Glow on all four sides of the base.

        self.add_glow_volume(
            hotel,
            "hotel_base_glow",
            0,
            0,
            270,
            100,
            0,
            20,
            [3.0, 8.0, 13.0, 19.0],
            self.HOTEL_PALETTE,
            offset_y=1.3,
            offset_x=0.15,
            overhang=2.5,
            roof_z=22.3,
        )

        front_leds = []

        for i, color in enumerate(self.HOTEL_PALETTE):

            front_leds.append((
                (0, -52.2, 19 + i * 1.2),
                (250, 0.35, 0.45),
                color,
            ))

        self.make_glow_batch(
            hotel,
            "hotel_front_led",
            front_leds,
        )

        # Entrance.

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

        self.make_glow_batch(
            hotel,
            "hotel_entrance_light",
            [(
                (0, -65.7, 18),
                (38, 0.35, 0.5),
                (0.05, 0.85, 1.0),
            )],
        )

        self.add_hotel_lighting(hotel)

        print(
            "[Facilities] Large illuminated "
            "W Abu Dhabi hotel created."
        )


    def create_yas_plaza_complex(self):

        site = self.find_building_site(
            0.36,
            1,
            extents=(-85, 105, -60, 36),
        )

        if site is None:
            return

        position, heading, footprint = site

        # Entrance side (-Y) looks toward the track.
        complex_root = self.make_anchor(
            "yas_plaza_hotel_complex",
            position,
            heading,
            footprint=footprint,
            check_facing=True,
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

            self.create_glow_tower(
                complex_root,
                f"yas_plaza_{data['name']}",
                data["x"],
                data["y"],
                data["width"],
                data["depth"],
                data["height"],
                data["color"],
            )

        self.make_box(
            complex_root,
            "yas_plaza_entrance_platform",
            (15, -42, 0.35),
            (160, 30, 0.7),
            (0.17, 0.19, 0.22),
        )

        entrance_lights = []

        for i in range(8):

            x = -52 + i * 19

            entrance_lights.append((
                (x, -56, 1.0),
                (5, 0.4, 0.25),
                (0.0, 0.75, 1.0),
            ))

        self.make_glow_batch(
            complex_root,
            "yas_plaza_entrance_lights",
            entrance_lights,
            halo=0.5,
        )

        print(
            "[Facilities] Yas Plaza-inspired hotel "
            "complex created."
        )


    def create_modern_arena(self):

        site = self.find_building_site(
            0.53,
            -1,
            extents=(-90, 90, -72, 62),
        )

        if site is None:
            return

        position, heading, footprint = site

        arena = self.make_anchor(
            "yas_island_modern_arena",
            position,
            heading,
            footprint=footprint,
            check_facing=True,
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


    def create_shopping_complex(self):

        site = self.find_building_site(
            0.14,
            -1,
            extents=(-100, 100, -112, 62),
        )

        if site is None:
            return

        position, heading, footprint = site

        mall = self.make_anchor(
            "yas_island_shopping_complex",
            position,
            heading,
            footprint=footprint,
            check_facing=True,
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

    def create_city_skyline(self):

        site = self.find_building_site(
            0.91,
            1,
            extents=(-120, 140, -30, 30),
        )

        if site is None:
            return

        position, heading, footprint = site

        skyline = self.make_anchor(
            "yas_island_city_skyline",
            position,
            heading,
            footprint=footprint,
            check_facing=True,
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

            self.create_glow_tower(
                skyline,
                f"skyline_building_{i}",
                x,
                y,
                width,
                depth,
                height,
                colors[i],
            )

        print(
            "[Facilities] Distant Yas Island-inspired "
            "city skyline created."
        )

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


    def remove(self):

        self.root.removeNode()

        self.root = self.render.attachNewNode(
            "yas_marina_facilities"
        )

        self.occupied_sites.clear()
        self.occupied_footprints.clear()