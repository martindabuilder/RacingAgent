from panda3d.core import (Geom, GeomNode, GeomTriangles, GeomVertexData, GeomVertexFormat, GeomVertexWriter, NodePath)
import numpy as np

class TrackEnvironment:
    def __init__(self, render, track):
        self.render = render
        self.track = track

        self.ground = None
        self.left_curbs = None
        self.right_curbs = None

        self.ground_margin = 150.0
        self.ground_height = -0.15

        self.curb_width = 0.8
        self.curb_inner_height = 0.045
        self.curb_outer_height = 0.12

        self.stripe_length = 5.0

        self.ground_color = (0.08, 0.10, 0.14, 1.0)
        self.red_color = ( 0.85, 0.035, 0.025,1.0)
        self.white_color = (0.90, 0.90, 0.86, 1.0)

    def create(self):
        self.create_ground()
        self.create_curbs()
        print("Environment created successfully!")

    def _create_mesh(self, name, vertices, colors, triangles):

        vertex_data = GeomVertexData(name, GeomVertexFormat.getV3c4(), Geom.UHStatic)
        vertex_data.setNumRows(len(vertices))
        vertex_writer = GeomVertexWriter(vertex_data, "vertex")
        color_writer = GeomVertexWriter(vertex_data, "color")

        for vertex, color in zip(vertices, colors):
            vertex_writer.addData3f(float(vertex[0]), float(vertex[1]), float(vertex[2]))
            color_writer.addData4f(float(color[0]), float(color[1]), float(color[2]), float(color[3]))

        primitive = GeomTriangles(Geom.UHStatic)

        for triangle in triangles:
            primitive.addVertices(triangle[0], triangle[1], triangle[2])

        primitive.closePrimitive()

        geometry = Geom(vertex_data)
        geometry.addPrimitive(primitive)

        node = GeomNode(name)
        node.addGeom(geometry)

        mesh = self.render.attachNewNode(node)

        mesh.setTwoSided(True)
        mesh.setLightOff()

        return mesh

    def create_ground(self):
        centerline = self.track.centerline

        min_x = float(np.min(centerline[:, 0]))
        max_x = float(np.max(centerline[:, 0]))

        min_y = float(np.min(centerline[:, 1]))
        max_y = float(np.max(centerline[:, 1]))

        margin = self.ground_margin

        vertices = [
            (min_x - margin, min_y - margin, self.ground_height),
            (max_x + margin, min_y - margin, self.ground_height),
            (max_x + margin, max_y + margin, self.ground_height),
            (min_x - margin, max_y + margin, self.ground_height)
        ]

        colors = [self.ground_color for _ in range(4)]
        triangles = [(0, 1, 2), (0, 2, 3)]

        self.ground = self._create_mesh("yas_marina_ground", vertices, colors, triangles)
        print("Grey trackside ground created.")

        return self.ground

    def create_curbs(self):

        centerline = self.track.centerline

        left = self.track.left_boundary_3d
        right = self.track.right_boundary_3d

        point_count = len(centerline)
        previous = np.roll(centerline, 1, axis=0)
        following = np.roll(centerline, -1, axis=0)
        tangents = following - previous

        lengths = np.linalg.norm(tangents, axis=1)
        lengths = np.maximum(lengths, 1e-8)
        tangents = tangents / lengths[:, None]
        normals = np.column_stack((-tangents[:, 1], tangents[:, 0]))

        left_vertices = []
        left_colors = []
        left_triangles = []

        right_vertices = []
        right_colors = []
        right_triangles = []

        distance_along_track = 0.0

        for i in range(point_count):

            next_i = (i + 1) % point_count

            segment_length = float(
                np.linalg.norm(centerline[next_i] - centerline[i]))

            stripe_index = int(distance_along_track / self.stripe_length)

            if stripe_index % 2 == 0:
                color = self.red_color
            else:
                color = self.white_color

            # LEFT CURB
            left_inner_a = left[i]
            left_inner_b = left[next_i]

            left_outer_a = np.array([
                left[i, 0] + normals[i, 0] * self.curb_width,
                left[i, 1] + normals[i, 1] * self.curb_width,
                left[i, 2]
            ])

            left_outer_b = np.array([
                left[next_i, 0] + normals[next_i, 0] * self.curb_width,
                left[next_i, 1] + normals[next_i, 1] * self.curb_width,
                left[next_i, 2]
            ])

            left_vertices.extend([
                (left_inner_a[0], left_inner_a[1], left_inner_a[2] + self.curb_inner_height),
                (left_outer_a[0], left_outer_a[1], left_outer_a[2] + self.curb_outer_height),
                (left_inner_b[0], left_inner_b[1], left_inner_b[2] + self.curb_inner_height),
                (left_outer_b[0], left_outer_b[1], left_outer_b[2] + self.curb_outer_height)
            ])

            left_colors.extend([color] * 4)
            base = len(left_vertices) - 4
            left_triangles.extend([(base, base + 1, base + 2), (base + 1, base + 3, base + 2)])

            # RIGHT CURB
            right_inner_a = right[i]
            right_inner_b = right[next_i]

            right_outer_a = np.array([
                right[i, 0] - normals[i, 0] * self.curb_width,
                right[i, 1] - normals[i, 1] * self.curb_width,
                right[i, 2]
            ])

            right_outer_b = np.array([
                right[next_i, 0] - normals[next_i, 0] * self.curb_width,
                right[next_i, 1] - normals[next_i, 1] * self.curb_width,
                right[next_i, 2]
            ])

            right_vertices.extend([
                (right_inner_a[0], right_inner_a[1], right_inner_a[2] + self.curb_inner_height),
                (right_outer_a[0], right_outer_a[1], right_outer_a[2] + self.curb_outer_height),
                (right_inner_b[0], right_inner_b[1], right_inner_b[2] + self.curb_inner_height),
                (right_outer_b[0], right_outer_b[1], right_outer_b[2] + self.curb_outer_height)
            ])

            right_colors.extend([color] * 4)
            base = len(right_vertices) - 4
            right_triangles.extend([(base, base + 1, base + 2), (base + 1, base + 3, base + 2)])

            distance_along_track += segment_length

        self.left_curbs = self._create_mesh("yas_marina_left_curbs", left_vertices, left_colors, left_triangles)
        self.right_curbs = self._create_mesh("yas_marina_right_curbs", right_vertices, right_colors,right_triangles)

        print("Left curbs created:", point_count, "sections")
        print("Right curbs created:", point_count, "sections")

        return self.left_curbs, self.right_curbs