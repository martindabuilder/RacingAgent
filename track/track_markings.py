import numpy as np
from panda3d.core import (Geom, GeomNode, GeomTriangles, GeomVertexData, GeomVertexFormat, GeomVertexWriter, NodePath, TextNode,)


class TrackMarkings:
    def __init__(self, render: NodePath, track):

        self.render = render
        self.track = track
        self.root = None

        self.marking_height = 0.055
        self.z_offset = 0.001

        self.white = (0.62, 0.62, 0.58, 1.0)
        self.yellow = (0.62, 0.50, 0.08, 1.0)

        # DRS signs: white background, black text
        self.drs_color = (1.0, 1.0, 1.0, 1.0)
        self.drs_text_color = (0.01, 0.01, 0.01, 1.0)

    
        self.edge_line_width = 0.16

     
        self.start_progress = 0.0
        self.grid_forward_offset = 10.0
        self.finish_line_length = 0.85

  
        self.grid_slots = 20

        self.grid_longitudinal_spacing = 10.0
        self.grid_lateral_spacing = 4.0

        self.grid_box_length = 5.5
        self.grid_box_width = 2.4

        self.grid_line_width = 0.12

        self.grid_first_row_offset = 3.0
        self.grid_stagger = 1.5

   
        self.drs_zones = [
            {
                "detection": 900.0,
                "activation": 1250.0
            },
            {
                "detection": 3200.0,
                "activation": 3500.0
            }
        ]

    
        self.drs_sign_width = 3.0
        self.drs_sign_height = 3.0
        self.drs_sign_depth = 0.25

        self.drs_sign_offset = 3.5
        self.drs_sign_bottom_offset = 0.1

        self.corner_positions = []


    def create(self):

        self.clear()

        self.root = self.render.attachNewNode(
            "yas_marina_track_markings"
        )

        self._create_edge_lines()
        self._create_start_finish_line()
        self._create_starting_grid()
        self._create_drs_markings()
        self._create_corner_markings()

        return self.root

    def clear(self):

        if self.root is not None:
            self.root.removeNode()
            self.root = None

   
    def _get_track_sample(self, progress):

        progress = float(progress) % self.track.track_length

        distances = self.track.distance

        index = int(
            np.searchsorted(
                distances,
                progress,
                side="right"
            ) - 1
        )

        index %= self.track.number_of_points

        next_index = (
            index + 1
        ) % self.track.number_of_points

        segment_length = self.track.segment_lengths[index]

        if segment_length <= 1e-8:
            t = 0.0
        else:
            t = (
                progress - distances[index]
            ) / segment_length

        t = float(np.clip(t, 0.0, 1.0))

        center = (
            self.track.centerline[index] * (1.0 - t)
            + self.track.centerline[next_index] * t
        )

        direction = self.track.segment_directions[index].copy()

        direction_length = np.linalg.norm(direction)

        if direction_length < 1e-8:
            raise ValueError("Invalid track direction.")

        direction /= direction_length

        normal = np.array([
            -direction[1],
            direction[0]
        ])

        elevation = (
            self.track.centerline_3d[index, 2] * (1.0 - t)
            + self.track.centerline_3d[next_index, 2] * t
        )

        width_left = (
            self.track.width_left[index] * (1.0 - t)
            + self.track.width_left[next_index] * t
        )

        width_right = (
            self.track.width_right[index] * (1.0 - t)
            + self.track.width_right[next_index] * t
        )

        return {
            "position": center,
            "direction": direction,
            "normal": normal,
            "elevation": float(elevation),
            "width_left": float(width_left),
            "width_right": float(width_right),
            "progress": progress,
        }

    def _create_quad(self, name, corners, color):

        vertex_format = GeomVertexFormat.getV3c4()

        vertex_data = GeomVertexData(
            name,
            vertex_format,
            Geom.UHStatic
        )

        vertex_data.setNumRows(4)

        vertex_writer = GeomVertexWriter(
            vertex_data,
            "vertex"
        )

        color_writer = GeomVertexWriter(
            vertex_data,
            "color"
        )

        for point in corners:

            vertex_writer.addData3f(
                float(point[0]),
                float(point[1]),
                float(point[2])
            )

            color_writer.addData4f(*color)

        triangles = GeomTriangles(Geom.UHStatic)

        triangles.addVertices(0, 1, 2)
        triangles.addVertices(0, 2, 3)
        triangles.closePrimitive()

        geometry = Geom(vertex_data)
        geometry.addPrimitive(triangles)

        node = GeomNode(name)
        node.addGeom(geometry)

        result = self.root.attachNewNode(node)

        result.setTwoSided(True)
        result.setLightOff(1)

        return result

    def _create_rectangle(
        self,
        center,
        direction,
        normal,
        length,
        width,
        z,
        color,
        name
    ):

        half_length = length * 0.5
        half_width = width * 0.5

        corners = [
            center - direction * half_length - normal * half_width,
            center + direction * half_length - normal * half_width,
            center + direction * half_length + normal * half_width,
            center - direction * half_length + normal * half_width,
        ]

        corners_3d = [
            (point[0], point[1], z)
            for point in corners
        ]

        return self._create_quad(
            name,
            corners_3d,
            color
        )

 
    def _create_outline(
        self,
        center,
        direction,
        normal,
        length,
        width,
        line_width,
        z,
        color,
        name
    ):

        half_length = length * 0.5
        half_width = width * 0.5

        # Front crossbar
        front_center = (
            center + direction * half_length
        )

        self._create_rectangle(
            front_center,
            direction,
            normal,
            line_width,
            width,
            z,
            color,
            f"{name}_front"
        )

        # Left side
        left_center = (
            center + normal * half_width
        )

        self._create_rectangle(
            left_center,
            direction,
            normal,
            length,
            line_width,
            z,
            color,
            f"{name}_left"
        )

        # Right side
        right_center = (
            center - normal * half_width
        )

        self._create_rectangle(
            right_center,
            direction,
            normal,
            length,
            line_width,
            z,
            color,
            f"{name}_right"
        )

    def _create_edge_lines(self):

        left_boundary = self.track.left_boundary_3d
        right_boundary = self.track.right_boundary_3d

        point_count = len(left_boundary)

        for side_name, boundary in (
            ("left", left_boundary),
            ("right", right_boundary),
        ):

            for i in range(point_count):

                next_i = (i + 1) % point_count

                p1 = boundary[i]
                p2 = boundary[next_i]

                direction = p2[:2] - p1[:2]

                length = np.linalg.norm(direction)

                if length < 1e-6:
                    continue

                direction /= length

                normal = np.array([
                    -direction[1],
                    direction[0]
                ])

                center = (
                    p1[:2] + p2[:2]
                ) * 0.5

                corners = [
                    center - direction * length * 0.5
                    - normal * self.edge_line_width * 0.5,

                    center + direction * length * 0.5
                    - normal * self.edge_line_width * 0.5,

                    center + direction * length * 0.5
                    + normal * self.edge_line_width * 0.5,

                    center - direction * length * 0.5
                    + normal * self.edge_line_width * 0.5,
                ]

                z = (
                    (p1[2] + p2[2]) * 0.5
                    + self.marking_height
                    + self.z_offset
                )

                corners_3d = [
                    (point[0], point[1], z)
                    for point in corners
                ]

                self._create_quad(
                    f"edge_{side_name}_{i}",
                    corners_3d,
                    self.white
                )

  
    def _create_start_finish_line(self):

        start_progress = (
            self.start_progress
            + self.grid_forward_offset
        )

        sample = self._get_track_sample(
            start_progress
        )

        center = sample["position"]
        direction = sample["direction"]
        normal = sample["normal"]

        width = (
            sample["width_left"]
            + sample["width_right"]
        )

        z = (
            sample["elevation"]
            + self.marking_height
            + self.z_offset * 2.0
        )

        self._create_rectangle(
            center,
            direction,
            normal,
            self.finish_line_length,
            width,
            z,
            self.white,
            "start_finish_line"
        )

    def _create_starting_grid(self):

        new_start_progress = (
            self.start_progress
            + self.grid_forward_offset
        )

        for slot in range(self.grid_slots):

            row = slot // 2
            side = slot % 2

            lateral_position = (
                self.grid_lateral_spacing * 0.5
                if side == 0
                else -self.grid_lateral_spacing * 0.5
            )

            stagger = (
                self.grid_stagger
                if row % 2 == 1
                else 0.0
            )

            progress = (
                new_start_progress
                - self.grid_first_row_offset
                - row * self.grid_longitudinal_spacing
                - stagger
            )

            sample = self._get_track_sample(progress)

            center = sample["position"].copy()
            direction = sample["direction"]
            normal = sample["normal"]

            center += normal * lateral_position

            z = (
                sample["elevation"]
                + self.marking_height
                + self.z_offset * 3.0
            )

            self._create_outline(
                center=center,
                direction=direction,
                normal=normal,
                length=self.grid_box_length,
                width=self.grid_box_width,
                line_width=self.grid_line_width,
                z=z,
                color=self.white,
                name=f"grid_slot_{slot + 1}"
            )

 
    def _create_drs_markings(self):

        print("\n========== DRS SIGNS ==========")

        for index, zone in enumerate(self.drs_zones):

            detection = float(zone["detection"])
            activation = float(zone["activation"])

            print(
                f"DRS Zone {index + 1}: "
                f"Detection = {detection:.2f} m, "
                f"Activation = {activation:.2f} m"
            )

            self._create_drs_marker(
                detection,
                f"drs_detection_{index + 1}"
            )

            self._create_drs_marker(
                activation,
                f"drs_activation_{index + 1}"
            )

    def _create_drs_marker(self, progress, name):

        sample = self._get_track_sample(progress)

        center = sample["position"]
        direction = sample["direction"]
        normal = sample["normal"]

        elevation = sample["elevation"]

        width_left = sample["width_left"]
        width_right = sample["width_right"]

        left_position = (
            center
            + normal * (
                width_left + self.drs_sign_offset
            )
        )

        right_position = (
            center
            - normal * (
                width_right + self.drs_sign_offset
            )
        )

        self._create_drs_sign(
            name=f"{name}_left",
            position=left_position,
            direction=direction,
            elevation=elevation
        )

        self._create_drs_sign(
            name=f"{name}_right",
            position=right_position,
            direction=direction,
            elevation=elevation
        )

    def _create_drs_sign(
        self,
        name,
        position,
        direction,
        elevation
    ):

        width = self.drs_sign_width
        height = self.drs_sign_height
        depth = self.drs_sign_depth

        forward = direction.copy()
        forward /= np.linalg.norm(forward)

        horizontal = np.array([
            -forward[1],
            forward[0]
        ])

        bottom_z = elevation + self.drs_sign_bottom_offset
        top_z = bottom_z + height

        half_width = width * 0.5
        half_depth = depth * 0.5

        front_center = position + forward * half_depth
        back_center = position - forward * half_depth

        front_corners = [
            (
                front_center[0] - horizontal[0] * half_width,
                front_center[1] - horizontal[1] * half_width,
                bottom_z
            ),
            (
                front_center[0] + horizontal[0] * half_width,
                front_center[1] + horizontal[1] * half_width,
                bottom_z
            ),
            (
                front_center[0] + horizontal[0] * half_width,
                front_center[1] + horizontal[1] * half_width,
                top_z
            ),
            (
                front_center[0] - horizontal[0] * half_width,
                front_center[1] - horizontal[1] * half_width,
                top_z
            )
        ]

        back_corners = [
            (
                back_center[0] - horizontal[0] * half_width,
                back_center[1] - horizontal[1] * half_width,
                bottom_z
            ),
            (
                back_center[0] + horizontal[0] * half_width,
                back_center[1] + horizontal[1] * half_width,
                bottom_z
            ),
            (
                back_center[0] + horizontal[0] * half_width,
                back_center[1] + horizontal[1] * half_width,
                top_z
            ),
            (
                back_center[0] - horizontal[0] * half_width,
                back_center[1] - horizontal[1] * half_width,
                top_z
            )
        ]

        # Matte white sign material
        color = (0.72, 0.72, 0.72, 1.0)

        # Front
        front_node = self._create_quad(
            f"{name}_front",
            front_corners,
            color
        )

        # Back
        back_node = self._create_quad(
            f"{name}_back",
            back_corners,
            color
        )

        # Sides
        for i in range(4):

            next_i = (i + 1) % 4

            side_node = self._create_quad(
                f"{name}_side_{i}",
                [
                    front_corners[i],
                    front_corners[next_i],
                    back_corners[next_i],
                    back_corners[i]
                ],
                color
            )

            side_node.setLightOff(1)
            side_node.setShaderOff(1)
            side_node.setTextureOff(1)

        # Disable lighting, shaders and textures on the sign body.
        for node in (front_node, back_node):
            node.setLightOff(1)
            node.setShaderOff(1)
            node.setTextureOff(1)

   
        text_node = TextNode(f"{name}_text")

        text_node.setText("DRS")
        text_node.setTextColor(*self.drs_text_color)
        text_node.setAlign(TextNode.ACenter)
        text_node.setTextScale(1.5)

        text_path = self.root.attachNewNode(text_node)

        # Text goes on the side facing the approaching cars (back_center).
        text_position = back_center - forward * 0.04

        text_path.setPos(
            float(text_position[0]),
            float(text_position[1]),
            float(bottom_z + height * 0.32)
        )

        # Text face (-Y at H=0) must point toward -forward.
        heading = np.degrees(
            np.arctan2(-forward[0], forward[1])
        )

        text_path.setHpr(float(heading), 0.0, 0.0)

        text_path.setLightOff(1)
        text_path.setShaderOff(1)
        text_path.setTwoSided(True)

        # Positive depth offset = closer to the camera.
        text_path.setDepthOffset(2)
        text_path.setBin("fixed", 10)

   
    def _create_corner_markings(self):

        for corner_number, progress in enumerate(
            self.corner_positions,
            start=1
        ):

            sample = self._get_track_sample(progress)

            center = sample["position"]
            normal = sample["normal"]

            position = (
                center
                + normal * (
                    sample["width_left"] + 1.5
                )
            )

            z = (
                sample["elevation"]
                + self.marking_height
                + self.z_offset * 5.0
            )

            self._create_corner_number(
                corner_number,
                position,
                z
            )

    def _create_corner_number(
        self,
        number,
        position,
        z
    ):

        text_node = TextNode(
            f"corner_number_{number}"
        )

        text_node.setText(str(number))
        text_node.setTextColor(*self.white)
        text_node.setAlign(TextNode.ACenter)
        text_node.setTextScale(2.5)

        node = self.root.attachNewNode(text_node)

        node.setPos(
            float(position[0]),
            float(position[1]),
            float(z)
        )

        node.setHpr(0, -90, 0)
        node.setLightOff(1)