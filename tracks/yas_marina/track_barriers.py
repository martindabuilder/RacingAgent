import numpy as np
from shapely.geometry import LineString, Polygon, MultiPolygon
from shapely.ops import triangulate
from panda3d.core import (Geom, GeomNode, GeomTriangles, GeomVertexData, GeomVertexFormat, GeomVertexWriter,)

class TrackBarriers:

    def __init__(self, render, track):

        self.render = render
        self.track = track

        self.root = render.attachNewNode(
            "yas_marina_barriers"
        )

        # Distance from asphalt edge to inner wall face.
        self.clearance = 15.0
        self.curb_width = 0.8

        self.wall_width = 0.45
        self.wall_height = 1.2

        self.fence_height = 4.0
        self.post_spacing = 6.0
        self.post_width = 0.16

        self.round_resolution = 12

        self.base_z = 0.04

        self.wall_color = (0.40, 0.42, 0.44, 1.0)
        self.post_color = (0.65, 0.67, 0.69, 1.0)
        self.rail_color = (0.48, 0.51, 0.54, 1.0)

        self.center_line = None
        self.is_counter_clockwise = None

   
    def _initialize_centerline(self):

        points = np.asarray(
            self.track.centerline,
            dtype=float
        )[:, :2]

        if len(points) < 10:
            raise ValueError(
                "Centerline contains too few points."
            )

        coordinates = np.vstack((
            points,
            points[0]
        ))

        line = LineString(coordinates)

        if not line.is_valid or not line.is_simple:
            raise ValueError(
                "Track centerline is not a valid simple closed line."
            )

        if not line.is_closed:
            raise ValueError(
                "Track centerline must be closed."
            )

        self.center_line = line

        # Signed area determines the direction of the track.
        area = 0.5 * np.sum(
            points[:, 0] * np.roll(points[:, 1], -1)
            - np.roll(points[:, 0], -1) * points[:, 1]
        )

        self.is_counter_clockwise = area > 0

        print(
            "Centerline:",
            round(line.length, 2),
            "m"
        )

        print(
            "Direction:",
            "counter-clockwise"
            if self.is_counter_clockwise
            else "clockwise"
        )

 
    def _get_side_width(self, side):

        if side == "left":

            values = np.asarray(
                self.track.width_left,
                dtype=float
            )

        else:

            values = np.asarray(
                self.track.width_right,
                dtype=float
            )

        values = values[np.isfinite(values)]

        if len(values) == 0:
            raise ValueError(
                f"No valid {side} track widths."
            )

        # Median prevents isolated width anomalies
        # from moving the barrier unexpectedly.
        return float(np.median(values))

 
    def _get_offset_ring(self, side):

        width = self._get_side_width(side)

        offset_distance = (
            width
            + self.curb_width
            + self.clearance
            + self.wall_width / 2.0
        )

        buffered = self.center_line.buffer(
            offset_distance,
            quad_segs=self.round_resolution,
            cap_style="round",
            join_style="round",
            mitre_limit=2.0
        )

        if buffered.is_empty:
            raise RuntimeError(
                f"Empty offset geometry for {side} side."
            )

        if not buffered.is_valid:
            buffered = buffered.buffer(0)

        if isinstance(buffered, MultiPolygon):

            buffered = max(
                buffered.geoms,
                key=lambda polygon: polygon.area
            )

        if not isinstance(buffered, Polygon):
            raise RuntimeError(
                f"Unexpected offset geometry: "
                f"{buffered.geom_type}"
            )

        if self.is_counter_clockwise:

            use_inner_ring = side == "left"

        else:

            use_inner_ring = side == "right"

        if use_inner_ring:

            if not buffered.interiors:

                raise RuntimeError(
                    f"The inner offset on the {side} side "
                    "has collapsed. The requested clearance "
                    "is too large for this part of the track."
                )

            ring = max(
                buffered.interiors,
                key=lambda item: Polygon(item).area
            )

        else:

            ring = buffered.exterior

        result = LineString(ring.coords)

        if not result.is_closed or not result.is_valid:
            raise RuntimeError(
                f"Invalid closed offset ring on {side} side."
            )

        print(
            side,
            "offset:",
            round(offset_distance, 2),
            "m | length:",
            round(result.length, 2),
            "m"
        )

        return result

    def _make_wall_footprint(self, centerline):

        footprint = centerline.buffer(
            self.wall_width / 2.0,
            quad_segs=self.round_resolution,
            cap_style="round",
            join_style="round",
            mitre_limit=2.0
        )

        if not footprint.is_valid:
            footprint = footprint.buffer(0)

        if footprint.is_empty:
            raise RuntimeError(
                "Could not create wall footprint."
            )

        return footprint

    def _create_mesh(
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

        mesh.setTwoSided(True)

        return mesh


    def _create_extruded_polygon(
        self,
        polygon,
        name,
        z_bottom,
        z_top,
        color
    ):

        if polygon.is_empty:
            return

        if not polygon.is_valid:
            polygon = polygon.buffer(0)

        if isinstance(polygon, MultiPolygon):

            for index, part in enumerate(polygon.geoms):

                self._create_extruded_polygon(
                    part,
                    f"{name}_{index}",
                    z_bottom,
                    z_top,
                    color
                )

            return

        if not isinstance(polygon, Polygon):
            raise RuntimeError(
                "Expected Polygon for mesh extrusion."
            )

        vertices = []
        colors = []
        triangles = []

   
        rings = [polygon.exterior]
        rings.extend(polygon.interiors)

        for ring in rings:

            coordinates = np.asarray(
                ring.coords[:-1],
                dtype=float
            )

            count = len(coordinates)

            if count < 3:
                continue

            base = len(vertices)

            for point in coordinates:

                vertices.append((
                    point[0],
                    point[1],
                    z_bottom
                ))

            for point in coordinates:

                vertices.append((
                    point[0],
                    point[1],
                    z_top
                ))

            colors.extend([color] * (count * 2))

            for i in range(count):

                j = (i + 1) % count

                triangles.extend([
                    (base + i, base + j, base + count + j),
                    (base + i, base + count + j, base + count + i)
                ])

        for triangle in triangulate(polygon):

            if not polygon.covers(triangle):
                continue

            coordinates = list(
                triangle.exterior.coords
            )[:-1]

            if len(coordinates) != 3:
                continue

            top_base = len(vertices)

            for point in coordinates:

                vertices.append((
                    point[0],
                    point[1],
                    z_top
                ))

            colors.extend([color] * 3)

            triangles.append((
                top_base,
                top_base + 1,
                top_base + 2
            ))

            bottom_base = len(vertices)

            for point in coordinates:

                vertices.append((
                    point[0],
                    point[1],
                    z_bottom
                ))

            colors.extend([color] * 3)

            triangles.append((
                bottom_base + 2,
                bottom_base + 1,
                bottom_base
            ))

        self._create_mesh(
            name,
            vertices,
            colors,
            triangles
        )

    def _create_wall(self, offset_line, name):

        footprint = self._make_wall_footprint(
            offset_line
        )

        self._create_extruded_polygon(
            footprint,
            name,
            self.base_z,
            self.base_z + self.wall_height,
            self.wall_color
        )

        return offset_line

    def _create_posts(self, line, name):

        vertices = []
        colors = []
        triangles = []

        half = self.post_width / 2.0

        distance = 0.0

        while distance < line.length:

            point = line.interpolate(distance)

            before = line.interpolate(
                max(0.0, distance - 0.25)
            )

            after = line.interpolate(
                min(line.length, distance + 0.25)
            )

            tangent = np.array([
                after.x - before.x,
                after.y - before.y
            ])

            length = np.linalg.norm(tangent)

            if length < 1e-9:
                distance += self.post_spacing
                continue

            tangent /= length

            normal = np.array([
                -tangent[1],
                tangent[0]
            ])

            center = np.array([
                point.x,
                point.y
            ])

            corners = [
                center - tangent * half - normal * half,
                center - tangent * half + normal * half,
                center + tangent * half + normal * half,
                center + tangent * half - normal * half
            ]

            base = len(vertices)

            z0 = self.base_z + self.wall_height
            z1 = z0 + self.fence_height

            for corner in corners:

                vertices.append((
                    corner[0],
                    corner[1],
                    z0
                ))

            for corner in corners:

                vertices.append((
                    corner[0],
                    corner[1],
                    z1
                ))

            colors.extend([self.post_color] * 8)

            faces = [
                (0, 1, 5), (0, 5, 4),
                (1, 2, 6), (1, 6, 5),
                (2, 3, 7), (2, 7, 6),
                (3, 0, 4), (3, 4, 7),
                (4, 5, 6), (4, 6, 7)
            ]

            for face in faces:

                triangles.append(tuple(
                    base + index for index in face
                ))

            distance += self.post_spacing

        self._create_mesh(
            name,
            vertices,
            colors,
            triangles
        )


    def _create_rails(self, line, name):

        rail_levels = (0.5, 2.0, 3.8)

        for level in rail_levels:

            rail_footprint = line.buffer(
                0.055,
                quad_segs=4,
                cap_style="round",
                join_style="round"
            )

            z0 = (
                self.base_z
                + self.wall_height
                + level
            )

            self._create_extruded_polygon(
                rail_footprint,
                f"{name}_{level}",
                z0,
                z0 + 0.10,
                self.rail_color
            )

 
    def _create_side(self, side):

        print(
            "\nCreating",
            side,
            "barriers..."
        )

        offset_line = self._get_offset_ring(
            side
        )

        fence_line = self._create_wall(
            offset_line,
            side + "_concrete_wall"
        )

        self._create_posts(
            fence_line,
            side + "_fence_posts"
        )

        # Horizontal rails.
        self._create_rails(
            fence_line,
            side + "_fence_rails"
        )

        print(
            side,
            "barriers completed."
        )


    def create(self):
        self._initialize_centerline()
        self._create_side("left")
        self._create_side("right")

        return self.root