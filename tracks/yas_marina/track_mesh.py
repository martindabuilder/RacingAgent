from panda3d.core import (
    Geom,
    GeomNode,
    GeomTriangles,
    GeomVertexData,
    GeomVertexFormat,
    GeomVertexWriter,
    Material,
    NodePath
)


class TrackMesh:
    def __init__(self, render: NodePath, track):
        self.render = render
        self.track = track
        self.road = None

        self.road_height = 0.04

        self.road_color = (
            0.105,
            0.11,
            0.12,
            1.0
        )

    def create(self):

        if self.road is not None:
            self.road.removeNode()
            self.road = None

        left = self.track.left_boundary_3d
        right = self.track.right_boundary_3d

        point_count = len(left)

        if point_count < 3 or len(right) != point_count:
            raise ValueError(
                "Track boundaries must contain at least 3 matching points."
            )

        vertex_format = GeomVertexFormat.getV3c4t2()
        vertex_data = GeomVertexData("yas_marina_road", vertex_format, Geom.UHStatic)
        vertex_data.setNumRows(point_count * 2)
        vertex_writer = GeomVertexWriter(vertex_data, "vertex")
        color_writer = GeomVertexWriter(vertex_data, "color")
        texcoord_writer = GeomVertexWriter(vertex_data, "texcoord")

        for i in range(point_count):

            left_x, left_y, left_z = left[i]
            right_x, right_y, right_z = right[i]

            vertex_writer.addData3f(
                float(left_x),
                float(left_y),
                float(left_z + self.road_height)
            )

            color_writer.addData4f(
                *self.road_color
            )

            texcoord_writer.addData2f(
                0.0,
                i / 10.0
            )

            vertex_writer.addData3f(
                float(right_x),
                float(right_y),
                float(right_z + self.road_height)
            )

            color_writer.addData4f(
                *self.road_color
            )

            texcoord_writer.addData2f(
                1.0,
                i / 10.0
            )

        triangles = GeomTriangles(Geom.UHStatic)

        for i in range(point_count):

            next_i = (i + 1) % point_count

            left_current = i * 2
            right_current = i * 2 + 1

            left_next = next_i * 2
            right_next = next_i * 2 + 1

            triangles.addVertices(
                left_current,
                right_current,
                left_next
            )

            triangles.addVertices(
                right_current,
                right_next,
                left_next
            )

        triangles.closePrimitive()

        geometry = Geom(vertex_data)
        geometry.addPrimitive(triangles)

        node = GeomNode("yas_marina_road")
        node.addGeom(geometry)

        self.road = self.render.attachNewNode(node)
        self.road.setTwoSided(True)
        self.road.clearLight()

        material = Material()
        material.setName("yas_marina_asphalt")
        material.setDiffuse((0.16, 0.17, 0.19, 1.0))
        material.setAmbient((0.025, 0.028, 0.035, 1.0))
        material.setSpecular((0.08, 0.08, 0.09, 1.0))
        material.setShininess(24.0)

        self.road.setMaterial(material, 1)

        return self.road
