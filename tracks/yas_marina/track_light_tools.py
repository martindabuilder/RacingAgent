import numpy as np
from panda3d.core import (CardMaker, GeomNode, GeomVertexReader, GeomVertexWriter, PerspectiveLens, PNMImage, Spotlight, Texture, TransparencyAttrib, Vec3, Vec4, SamplerState,)

class LightField:
    def __init__(self, poles, color=(1.00, 0.90, 0.75), intensity=3.5, exponent=5.0, r0=40.0, ambient=(0.16, 0.19, 0.27), wall_lambert=0.6,):

        self.positions = np.array([p for p, _ in poles], dtype=float)
        targets = np.array([t for _, t in poles], dtype=float)

        aims = targets - self.positions
        aims /= np.linalg.norm(aims, axis=1)[:, None]
        self.aims = aims

        self.color = np.array(color, dtype=float)
        self.intensity = float(intensity)
        self.exponent = float(exponent)
        self.r0 = float(r0)
        self.ambient = np.array(ambient, dtype=float)
        self.wall_lambert = float(wall_lambert)

    def illumination(self, points, up=True, chunk=4000):

        points = np.asarray(points, dtype=float)
        result = np.zeros(len(points))

        for start in range(0, len(points), chunk):

            block = points[start:start + chunk]

            delta = block[:, None, :] - self.positions[None, :, :]
            dist = np.linalg.norm(delta, axis=2)
            dist = np.maximum(dist, 1e-6)
            direction = delta / dist[:, :, None]

            cos_spot = np.einsum("nmk,mk->nm", direction, self.aims)
            spot = np.clip(cos_spot, 0.0, 1.0) ** self.exponent

            if up:
                lambert = np.clip(-direction[:, :, 2], 0.0, 1.0)
            else:
                lambert = self.wall_lambert

            attenuation = self.r0 ** 2 / (dist ** 2 + self.r0 ** 2)

            result[start:start + chunk] = np.sum(spot * lambert * attenuation, axis=1)

        return (self.ambient[None, :] + self.color[None, :] * self.intensity * result[:, None])


def _geom_nodes(node_path):

    nodes = []

    if isinstance(node_path.node(), GeomNode):
        nodes.append(node_path)

    for child in node_path.findAllMatches("**/+GeomNode"):
        nodes.append(child)

    return nodes


def bake_vertex_lighting(node_path, field, up=True, gain=1.0):
    for geom_np in _geom_nodes(node_path):
        to_world = geom_np.getMat(node_path.getTop())
        geom_node = geom_np.node()

        for index in range(geom_node.getNumGeoms()):

            geom = geom_node.modifyGeom(index)
            vdata = geom.modifyVertexData()
            count = vdata.getNumRows()

            if count == 0:
                continue

            reader = GeomVertexReader(vdata, "vertex")
            points = np.empty((count, 3))

            for i in range(count):
                v = to_world.xformPoint(reader.getData3f())
                points[i] = (v[0], v[1], v[2])

            factor = field.illumination(points, up=up) * gain

            color_reader = GeomVertexReader(vdata, "color")
            color_writer = GeomVertexWriter(vdata, "color")

            for i in range(count):
                c = color_reader.getData4f()
                color_writer.setData4f(min(1.0, c[0] * factor[i, 0]), min(1.0, c[1] * factor[i, 1]), min(1.0, c[2] * factor[i, 2]), c[3],)

    node_path.setLightOff(1)
    node_path.setShaderOff(1)

    return node_path


class LightPool:
    def __init__(self, render, poles, count=6, color=(1.0, 0.92, 0.80, 1.0), fov=100.0, exponent=6.0, light_range=300.0,):

        self.render = render
        self.positions = np.array([p for p, _ in poles], dtype=float)
        self.targets = np.array([t for _, t in poles], dtype=float)

        self.count = min(int(count), len(poles))
        self.assigned = [-1] * self.count
        self.nodes = []

        for i in range(self.count):

            spot = Spotlight(f"pooled_spotlight_{i}")

            lens = PerspectiveLens()
            lens.setFov(fov)
            lens.setNearFar(1.0, light_range)

            spot.setLens(lens)
            spot.setColor(Vec4(*color))
            spot.setExponent(exponent)
            spot.setAttenuation(Vec3(1.0, 0.0, 0.00001))

            node = render.attachNewNode(spot)
            render.setLight(node)

            self.nodes.append(node)

    def update(self, x, y):

        d2 = (
            (self.positions[:, 0] - x) ** 2
            + (self.positions[:, 1] - y) ** 2
        )

        wanted = list(np.argsort(d2)[: self.count])
        wanted_set = set(int(w) for w in wanted)

        free_slots = [
            slot for slot, pole in enumerate(self.assigned)
            if pole not in wanted_set
        ]

        missing = [
            int(w) for w in wanted if int(w) not in self.assigned
        ]

        for slot, pole in zip(free_slots, missing):

            self.assigned[slot] = pole

            position = self.positions[pole]
            target = self.targets[pole]

            node = self.nodes[slot]
            node.setPos(*[float(v) for v in position])
            node.lookAt(*[float(v) for v in target])

    def remove(self):
        for node in self.nodes:
            self.render.clearLight(node)
            node.removeNode()

        self.nodes.clear()


class CarBlobShadow:
    def __init__(self, render, length=8.0, width=3.6, z=0.09, strength=0.7, heading_offset=0.0,):

        self.render = render
        self.z = z
        self.heading_offset = heading_offset

        size = 64
        image = PNMImage(size, size, 4)

        for y in range(size):
            for x in range(size):
                u = (x + 0.5) / size * 2.0 - 1.0
                v = (y + 0.5) / size * 2.0 - 1.0
                d = (abs(u) ** 3 + abs(v) ** 3) ** (1.0 / 3.0)
                alpha = np.clip(1.0 - d, 0.0, 1.0)
                alpha = alpha * alpha * (3.0 - 2.0 * alpha)

                image.setXelA(x, y, 0.0, 0.0, 0.0, float(alpha * strength))

        texture = Texture("car_blob_shadow")
        texture.load(image)
        texture.setWrapU(SamplerState.WM_clamp)
        texture.setWrapV(SamplerState.WM_clamp)

        maker = CardMaker("car_blob_shadow")
        maker.setFrame(-width / 2.0, width / 2.0, -length / 2.0, length / 2.0)

        self.node = render.attachNewNode(maker.generate())
        self.node.setP(-90)          # картата лежи върху пътя
        self.node.setTexture(texture)
        self.node.setTransparency(TransparencyAttrib.MAlpha)
        self.node.setLightOff()
        self.node.setShaderOff()
        self.node.setDepthWrite(False)
        self.node.setDepthOffset(2)
        self.node.setBin("fixed", 5)

    def follow(self, car):
        position = car.getPos(self.render)
        heading = car.getH(self.render)

        self.node.setPos(position.x, position.y, self.z)
        self.node.setH(self.render, heading + self.heading_offset)