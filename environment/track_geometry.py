import numpy as np

class TrackGeometry:
    def __init__(self, track_data):
        self.track_data = track_data

        self.centerline_3d = np.asarray(
            self.track_data.get_centerline(),
            dtype=float
        )

        if (
            self.centerline_3d.ndim != 2
            or self.centerline_3d.shape[1] != 3
        ):
            raise ValueError(
                "Centerline must have shape (N, 3)."
            )

        self.centerline = self.centerline_3d[:, :2]
        self.number_of_points = len(self.centerline)

        if self.number_of_points < 3:
            raise ValueError(
                "Track requires at least 3 centerline points."
            )

        if not np.all(np.isfinite(self.centerline_3d)):
            raise ValueError(
                "Centerline contains invalid coordinates."
            )

        widths = np.asarray(
            self.track_data.get_track_widths(),
            dtype=float
        )

        if widths.shape != (self.number_of_points, 2):
            raise ValueError(
                "Track widths must have shape (N, 2)."
            )

        if not np.all(np.isfinite(widths)) or np.any(widths <= 0.0):
            raise ValueError(
                "Track widths must contain positive finite values."
            )

        self.width_left = widths[:, 0]
        self.width_right = widths[:, 1]
        self.width = self.width_left + self.width_right

        self.segment_starts = self.centerline

        self.segment_ends = np.roll(
            self.centerline,
            -1,
            axis=0
        )

        self.segment_vectors = (
            self.segment_ends - self.segment_starts
        )

        self.segment_lengths = np.linalg.norm(
            self.segment_vectors,
            axis=1
        )

        if np.any(self.segment_lengths < 1e-8):
            raise ValueError(
                "Track contains zero-length segments."
            )

        self.segment_directions = (
            self.segment_vectors
            / self.segment_lengths[:, None]
        )

        self.track_length = float(
            np.sum(self.segment_lengths)
        )

        self.distance = np.zeros(
            self.number_of_points,
            dtype=float
        )

        self.distance[1:] = np.cumsum(
            self.segment_lengths[:-1]
        )

        self.heading = self._calculate_heading()
        self.curvature = self._calculate_curvature()

        (
            self.left_boundary_3d,
            self.right_boundary_3d
        ) = self._calculate_boundaries()

        self.left_boundary = self.left_boundary_3d[:, :2]
        self.right_boundary = self.right_boundary_3d[:, :2]

    def _calculate_heading(self):
        headings = np.arctan2(
            self.segment_directions[:, 1],
            self.segment_directions[:, 0]
        )

        previous = np.roll(headings, 1)

        angle_difference = np.arctan2(
            np.sin(headings - previous),
            np.cos(headings - previous)
        )

        return previous + angle_difference / 2.0

    def _calculate_curvature(self):
        directions = self.segment_directions

        previous_directions = np.roll(
            directions,
            1,
            axis=0
        )

        cross = (
            previous_directions[:, 0] * directions[:, 1]
            - previous_directions[:, 1] * directions[:, 0]
        )

        dot = np.sum(
            previous_directions * directions,
            axis=1
        )

        angle_change = np.arctan2(cross, dot)

        average_length = (
            np.roll(self.segment_lengths, 1)
            + self.segment_lengths
        ) / 2.0

        return angle_change / np.maximum(
            average_length,
            1e-8
        )

    def _calculate_boundaries(self):
        previous_points = np.roll(
            self.centerline,
            1,
            axis=0
        )

        next_points = np.roll(
            self.centerline,
            -1,
            axis=0
        )

        tangents = next_points - previous_points

        tangent_lengths = np.linalg.norm(
            tangents,
            axis=1
        )

        if np.any(tangent_lengths < 1e-8):
            raise ValueError(
                "Cannot calculate track boundaries: invalid tangent."
            )

        tangents = tangents / tangent_lengths[:, None]

        normals = np.column_stack((
            -tangents[:, 1],
            tangents[:, 0]
        ))

        left_xy = (
            self.centerline
            + normals * self.width_left[:, None]
        )

        right_xy = (
            self.centerline
            - normals * self.width_right[:, None]
        )

        left_3d = np.column_stack((
            left_xy,
            self.centerline_3d[:, 2]
        ))

        right_3d = np.column_stack((
            right_xy,
            self.centerline_3d[:, 2]
        ))

        return left_3d, right_3d

    def get_track_length(self):
        return self.track_length

    def get_heading(self, index):
        index = int(index) % self.number_of_points

        return float(
            np.degrees(self.heading[index])
        )

    def get_curvature(self, index):
        index = int(index) % self.number_of_points

        return float(self.curvature[index])

    def get_width(self, index):
        index = int(index) % self.number_of_points

        return float(self.width[index])

    def get_nearest_point(self, position):
        position = np.asarray(
            position,
            dtype=float
        )

        if position.size < 2:
            raise ValueError(
                "Position must contain at least X and Y."
            )

        position = position[:2]

        if not np.isfinite(position).all():
            raise ValueError(
                "Position contains invalid coordinates."
            )

        relative = position - self.segment_starts

        projection = np.sum(
            relative * self.segment_vectors,
            axis=1
        ) / (self.segment_lengths ** 2)

        projection = np.clip(
            projection,
            0.0,
            1.0
        )

        projected_points = (
            self.segment_starts
            + projection[:, None] * self.segment_vectors
        )

        differences = position - projected_points

        distances = np.linalg.norm(
            differences,
            axis=1
        )

        segment_index = int(
            np.argmin(distances)
        )

        t = float(projection[segment_index])

        nearest_xy = projected_points[segment_index]
        direction = self.segment_directions[segment_index]
        offset_vector = position - nearest_xy

        cross_product = (
            direction[0] * offset_vector[1]
            - direction[1] * offset_vector[0]
        )

        lateral_offset = float(cross_product)

        segment_progress = (
            t * self.segment_lengths[segment_index]
        )

        progress = (
            self.distance[segment_index]
            + segment_progress
        ) % self.track_length

        next_index = (
            segment_index + 1
        ) % self.number_of_points

        elevation = (
            self.centerline_3d[segment_index, 2] * (1.0 - t)
            + self.centerline_3d[next_index, 2] * t
        )

        curvature = (
            self.curvature[segment_index] * (1.0 - t)
            + self.curvature[next_index] * t
        )

        width_left = (
            self.width_left[segment_index] * (1.0 - t)
            + self.width_left[next_index] * t
        )

        width_right = (
            self.width_right[segment_index] * (1.0 - t)
            + self.width_right[next_index] * t
        )

        heading_rad = float(
            np.arctan2(
                direction[1],
                direction[0]
            )
        )

        return {
            "index": segment_index,
            "point": np.array([
                nearest_xy[0],
                nearest_xy[1],
                elevation
            ]),
            "distance": float(distances[segment_index]),
            "lateral_offset": lateral_offset,
            "progress": float(progress),
            "heading_rad": heading_rad,
            "heading_deg": float(np.degrees(heading_rad)),
            "curvature": float(curvature),
            "width_left": float(width_left),
            "width_right": float(width_right),
            "width": float(width_left + width_right)
        }

    def is_within_track(self, position):
        nearest = self.get_nearest_point(position)

        offset = nearest["lateral_offset"]

        if offset >= 0:
            return offset <= nearest["width_left"]

        return abs(offset) <= nearest["width_right"]

    def get_distance_from_centerline(self, position):
        nearest = self.get_nearest_point(position)

        return nearest["distance"]

    def get_track_position(self, position):
        nearest = self.get_nearest_point(position)

        return {
            **nearest,
            "within_track": self.is_within_track(position)
        }
