import numpy as np
from panda3d.core import (Geom, GeomNode, GeomTriangles, GeomVertexData, GeomVertexFormat, GeomVertexWriter,)

class TrackRunoff:
    def __init__(self, render, track, light_field=None):

        self.render = render
        self.track = track
        self.light_field = light_field
        self.root = None

        self.z = -0.02
        self.clearance = 15.4

        white = (0.78, 0.80, 0.80)
        teal = (0.00, 0.42, 0.50)
        asphalt = (0.17, 0.18, 0.20)

        self.bands = [
            (0.8, 1.3, white),
            (1.3, 7.5, teal),
            (7.5, 8.0, white),
            (8.0, self.clearance, asphalt),
        ]

    def _normals(self):
        centerline = np.asarray(self.track.centerline, dtype=float)[:, :2]
        tangents = np.roll(centerline, -1, axis=0) - np.roll(centerline, 1, axis=0)
        lengths = np.maximum(np.linalg.norm(tangents, axis=1), 1e-9)
        tangents /= lengths[:, None]
        return np.column_stack((-tangents[:, 1], tangents[:, 0]))

    def create(self):

        if self.root is not None:
            self.root.removeNode()

        left = np.asarray(self.track.left_boundary_3d, dtype=float)
        right = np.asarray(self.track.right_boundary_3d, dtype=float)
        normals = self._normals()

        count = len(left)

        vertices = []
        colors = []
        triangles = []

        for boundary, sign in ((left, 1.0), (right, -1.0)):
            for start, end, color in self.bands:
                base = len(vertices)

                for offset in (start, end):
                    xy = boundary[:, :2] + sign * normals * offset
                    z = boundary[:, 2] + self.z

                    for i in range(count):
                        vertices.append((xy[i, 0], xy[i, 1], z[i]))
                        colors.append(color)

                for i in range(count):
                    j = (i + 1) % count
                    a = base + i
                    b = base + j
                    c = base + count + i
                    d = base + count + j
                    triangles.append((a, b, c))
                    triangles.append((b, d, c))

        vertices = np.asarray(vertices, dtype=float)
        colors = np.asarray(colors, dtype=float)

        if self.light_field is not None:
            colors = colors * self.light_field.illumination(vertices, up=True)

        colors = np.clip(colors, 0.0, 1.0)

        data = GeomVertexData("runoff", GeomVertexFormat.getV3c4(), Geom.UHStatic)
        data.setNumRows(len(vertices))

        vertex_writer = GeomVertexWriter(data, "vertex")
        color_writer = GeomVertexWriter(data, "color")

        for p, c in zip(vertices, colors):
            vertex_writer.addData3f(float(p[0]), float(p[1]), float(p[2]))
            color_writer.addData4f(float(c[0]), float(c[1]), float(c[2]), 1.0)

        primitive = GeomTriangles(Geom.UHStatic)

        for triangle in triangles:
            primitive.addVertices(*triangle)

        primitive.closePrimitive()

        geom = Geom(data)
        geom.addPrimitive(primitive)

        node = GeomNode("yas_marina_runoff")
        node.addGeom(geom)

        self.root = self.render.attachNewNode(node)
        self.root.setTwoSided(True)
        self.root.setLightOff()

        print("Run-off zones created.")
        return self.root