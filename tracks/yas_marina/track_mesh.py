import numpy as np
from panda3d.core import (Geom, GeomNode, GeomTriangles, GeomVertexData, GeomVertexFormat, GeomVertexWriter, NodePath, PNMImage, SamplerState, Texture, TextureStage,)

def make_asphalt_texture(size=256, seed=11):
    rng = np.random.default_rng(seed)

    fine = rng.normal(0.0, 1.0, (size, size))
    mid = np.kron(rng.normal(0.0, 1.0, (size // 4, size // 4)), np.ones((4, 4)))
    coarse = np.kron(rng.normal(0.0, 1.0, (size // 16, size // 16)), np.ones((16, 16)))

    noise = 0.55 * fine + 0.30 * mid + 0.15 * coarse
    noise = (noise - noise.mean()) / (noise.std() + 1e-9)

    brightness = np.clip(0.80 + 0.05 * noise, 0.0, 1.0)
    image = PNMImage(size, size, 3)

    for y in range(size):
        for x in range(size):
            v = float(brightness[y, x])
            image.setXel(x, y, v, v, v)

    texture = Texture("procedural_asphalt")
    texture.load(image)
    texture.setWrapU(SamplerState.WM_repeat)
    texture.setWrapV(SamplerState.WM_repeat)
    texture.setMinfilter(SamplerState.FT_linear_mipmap_linear)
    texture.setMagfilter(SamplerState.FT_linear)
    texture.setAnisotropicDegree(8)

    return texture


class TrackMesh:
    def __init__(self, render: NodePath, track, light_field=None):
        self.render = render
        self.track = track
        self.light_field = light_field
        self.road = None
        self.road_height = 0.04
        self.columns = 9
        self.tile_length = 20.0
        self.road_color = (0.30, 0.31, 0.33)
        self.texture_gain = 1.25

    def create(self):
        if self.road is not None:
            self.road.removeNode()
            self.road = None

        left = np.asarray(self.track.left_boundary_3d, dtype=float)
        right = np.asarray(self.track.right_boundary_3d, dtype=float)

        count = len(left)

        if count < 3 or len(right) != count:
            raise ValueError("Track boundaries must contain at least 3 matching points.")

        t = np.linspace(0.0, 1.0, self.columns)
        rows = np.arange(count + 1) % count

        positions = (left[rows][:, None, :] * (1.0 - t)[None, :, None] + right[rows][:, None, :] * t[None, :, None])
        positions[:, :, 2] += self.road_height

        centerline = np.asarray(self.track.centerline, dtype=float)[:, :2]
        segment = np.linalg.norm(np.roll(centerline, -1, axis=0) - centerline, axis=1)
        along = np.concatenate(([0.0], np.cumsum(segment)))

        widths = np.linalg.norm(left[:, :2] - right[:, :2], axis=1)
        mean_width = float(np.mean(widths))

        racing_line = np.exp(-((t - 0.5) / 0.17) ** 2)
        edge = np.exp(-((np.abs(t - 0.5) - 0.5) / 0.07) ** 2)

        lane = 1.0 - 0.16 * racing_line + 0.06 * edge

        base = np.array(self.road_color)
        colors = np.empty((count + 1, self.columns, 3))
        colors[:] = base[None, None, :] * lane[None, :, None]
        colors *= self.texture_gain

        if self.light_field is not None:
            flat = positions.reshape(-1, 3)
            factor = self.light_field.illumination(flat, up=True)
            colors *= factor.reshape(count + 1, self.columns, 3)

        colors = np.clip(colors, 0.0, 1.0)

        vertex_data = GeomVertexData("road", GeomVertexFormat.getV3c4t2(), Geom.UHStatic)
        vertex_data.setNumRows((count + 1) * self.columns)

        vertex_writer = GeomVertexWriter(vertex_data, "vertex")
        color_writer = GeomVertexWriter(vertex_data, "color")
        texcoord_writer = GeomVertexWriter(vertex_data, "texcoord")

        for i in range(count + 1):
            for j in range(self.columns):
                p = positions[i, j]
                c = colors[i, j]

                vertex_writer.addData3f(float(p[0]), float(p[1]), float(p[2]))
                color_writer.addData4f(float(c[0]), float(c[1]), float(c[2]), 1.0)
                texcoord_writer.addData2f(float(t[j] * mean_width / self.tile_length), float(along[i] / self.tile_length),)

        triangles = GeomTriangles(Geom.UHStatic)

        for i in range(count):
            for j in range(self.columns - 1):
                a = i * self.columns + j
                b = a + 1
                c = (i + 1) * self.columns + j
                d = c + 1
                triangles.addVertices(a, b, c)
                triangles.addVertices(b, d, c)

        triangles.closePrimitive()

        geometry = Geom(vertex_data)
        geometry.addPrimitive(triangles)

        node = GeomNode("yas_marina_road")
        node.addGeom(geometry)

        self.road = self.render.attachNewNode(node)
        self.road.setTwoSided(True)

        stage = TextureStage("asphalt")
        stage.setMode(TextureStage.MModulate)
        self.road.setTexture(stage, make_asphalt_texture())
        self.road.setLightOff()

        return self.road