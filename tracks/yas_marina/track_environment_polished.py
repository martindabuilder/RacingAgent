import numpy as np
from panda3d.core import (Geom, GeomNode, GeomTriangles, GeomVertexData, GeomVertexFormat, GeomVertexWriter,)
from .track_environment import TrackEnvironment
from .track_light_tools import bake_vertex_lighting


class PolishedEnvironment(TrackEnvironment):
    def __init__(self, render, track, light_field=None, cell=16.0):
        super().__init__(render, track)
        self.light_field = light_field
        self.cell = float(cell)
        self.ground_margin = 300.0
        self.ground_color = (0.13, 0.15, 0.19, 1.0)

    def create_ground(self):
        centerline = np.asarray(self.track.centerline, dtype=float)

        min_x = float(centerline[:, 0].min()) - self.ground_margin
        max_x = float(centerline[:, 0].max()) + self.ground_margin
        min_y = float(centerline[:, 1].min()) - self.ground_margin
        max_y = float(centerline[:, 1].max()) + self.ground_margin

        nx = max(1, int(np.ceil((max_x - min_x) / self.cell)))
        ny = max(1, int(np.ceil((max_y - min_y) / self.cell)))

        xs = np.linspace(min_x, max_x, nx + 1)
        ys = np.linspace(min_y, max_y, ny + 1)

        grid_x, grid_y = np.meshgrid(xs, ys, indexing="ij")
        points = np.column_stack((grid_x.ravel(), grid_y.ravel(), np.full(grid_x.size, self.ground_height),))

        base = np.array(self.ground_color[:3])
        colors = np.tile(base, (len(points), 1))

        if self.light_field is not None:
            colors = colors * self.light_field.illumination(points, up=True)

        colors = np.clip(colors, 0.0, 1.0)

        data = GeomVertexData("ground", GeomVertexFormat.getV3c4(), Geom.UHStatic)
        data.setNumRows(len(points))

        vertex_writer = GeomVertexWriter(data, "vertex")
        color_writer = GeomVertexWriter(data, "color")

        for p, c in zip(points, colors):
            vertex_writer.addData3f(float(p[0]), float(p[1]), float(p[2]))
            color_writer.addData4f(float(c[0]), float(c[1]), float(c[2]), 1.0)

        primitive = GeomTriangles(Geom.UHStatic)

        for i in range(nx):
            for j in range(ny):

                a = i * (ny + 1) + j
                b = (i + 1) * (ny + 1) + j
                c = (i + 1) * (ny + 1) + j + 1
                d = i * (ny + 1) + j + 1

                primitive.addVertices(a, b, c)
                primitive.addVertices(a, c, d)

        primitive.closePrimitive()

        geom = Geom(data)
        geom.addPrimitive(primitive)

        node = GeomNode("yas_marina_ground")
        node.addGeom(geom)

        self.ground = self.render.attachNewNode(node)
        self.ground.setTwoSided(True)
        self.ground.setLightOff()

        print(f"Ground grid created: {nx}x{ny} cells.")
        return self.ground

    def create(self):

        print("\nCREATING YAS MARINA ENVIRONMENT (polished)")
        print("-" * 45)

        self.create_ground()
        left, right = self.create_curbs()

        if self.light_field is not None:
            bake_vertex_lighting(left, self.light_field, up=True)
            bake_vertex_lighting(right, self.light_field, up=True)

        print("Environment created successfully!")