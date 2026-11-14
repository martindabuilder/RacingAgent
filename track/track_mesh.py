from panda3d.core import (
    Geom,
    GeomNode,
    GeomTriangles,
    GeomVertexData,
    GeomVertexFormat,
    GeomVertexWriter,
    NodePath
)


class TrackMesh:
    def __init__(self, render: NodePath, track):
        self.render = render
        self.track = track
        self.road = None

    def create(self):
        left = self.track.left_boundary
        right = self.track.right_boundary
        point_count = len(left)

        # Създаваме структура за върховете на 3D модела
        vertex_data = GeomVertexData(
            "yas_marina_road",
            GeomVertexFormat.getV3c4(),
            Geom.UHStatic
        )

        vertex_data.setNumRows(point_count * 2)

        vertex_writer = GeomVertexWriter(
            vertex_data,
            "vertex"
        )

        color_writer = GeomVertexWriter(
            vertex_data,
            "color"
        )

        for i in range(point_count):

            vertex_writer.addData3f(
                float(left[i][0]),
                float(left[i][1]),
                0.02
            )

            color_writer.addData4f(
                0.12, 0.13, 0.15, 1.0
            )

            vertex_writer.addData3f(
                float(right[i][0]),
                float(right[i][1]),
                0.02
            )

            color_writer.addData4f(
                0.12, 0.13, 0.15, 1.0
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
        self.road.setLightOff()

        print("3D Yas Marina road mesh created!")

        return self.road
