import numpy as np
from track.track_data import TrackData

class TrackGeometry:

    def __init__(self):
        self.track_data = TrackData()
        self.centerline_3d = self.track_data.centerline
        self.left_boundary_3d = self.track_data.left_boundary
        self.right_boundary_3d = self.track_data.right_boundary
        self.centerline = self.centerline_3d[:, :2]
        self.left_boundary = self.left_boundary_3d[:, :2]
        self.right_boundary = self.right_boundary_3d[:, :2]

        self.distance = self.track_data.distance
        self.heading = self.track_data.heading
        self.curvature = self.track_data.curvature
        self.width = self.track_data.width

        self.width_left = np.linalg.norm(
            self.left_boundary - self.centerline,
            axis=1
        )

        self.width_right = np.linalg.norm(
            self.right_boundary - self.centerline,
            axis=1
        )

        self.number_of_points = len(self.centerline)

        # Create segments, including the final segment
        # connecting the last point to the first point.
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
            self.segment_lengths.sum()
        )

        print("\nTRACK GEOMETRY INITIALIZED")
        print("-" * 50)
        print("Points:", self.number_of_points)
        print("Track length:", round(self.track_length, 2), "m")

    def get_track_length(self):
        return self.track_length

    def get_heading(self, index):
        index = int(index) % self.number_of_points
        return float(np.degrees(self.heading[index]))

    def get_curvature(self, index):
        index = int(index) % self.number_of_points
        return float(self.curvature[index])

    def get_width(self, index):
        index = int(index) % self.number_of_points
        return float(self.width[index])


    def get_nearest_point(self, position):
        position = np.asarray(position, dtype=float)
        if position.size < 2:
            raise ValueError("Position must contain at least X and Y.")

        position = position[:2]

        if not np.isfinite(position).all():
            raise ValueError("Position contains invalid coordinates.")

        relative = position - self.segment_starts

        projection = np.sum(
            relative * self.segment_vectors,
            axis=1
        ) / (self.segment_lengths ** 2)

        projection = np.clip(projection, 0.0, 1.0)

        projected_points = (
            self.segment_starts
            + projection[:, None] * self.segment_vectors
        )

        differences = position - projected_points

        distances = np.linalg.norm(
            differences,
            axis=1
        )

        segment_index = int(np.argmin(distances))
        nearest_xy = projected_points[segment_index]
        direction = self.segment_directions[segment_index]
        offset_vector = position - nearest_xy

        cross_product = (
            direction[0] * offset_vector[1]
            - direction[1] * offset_vector[0]
        )

        lateral_offset = float(cross_product)

        segment_progress = (
            projection[segment_index]
            * self.segment_lengths[segment_index]
        )

        progress = (
            self.distance[segment_index]
            + segment_progress
        )

        progress %= self.track_length

        next_index = (
            segment_index + 1
        ) % self.number_of_points

        t = projection[segment_index]

        elevation = (
            self.centerline_3d[segment_index, 2]
            * (1.0 - t)
            + self.centerline_3d[next_index, 2] * t
        )

        curvature = (
            self.curvature[segment_index] * (1.0 - t)
            + self.curvature[next_index] * t
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
        }

    def is_within_track(self, position):

        nearest = self.get_nearest_point(position)

        index = nearest["index"]
        offset = nearest["lateral_offset"]

        if offset >= 0:
            return offset <= self.width_left[index]

        return abs(offset) <= self.width_right[index]

    def get_distance_from_centerline(self, position):
        nearest = self.get_nearest_point(position)
        return nearest["distance"]

    def get_track_position(self, position):
        nearest = self.get_nearest_point(position)
        index = nearest["index"]

        return {
            **nearest,
            "width": float(self.width[index]),
            "width_left": float(self.width_left[index]),
            "width_right": float(self.width_right[index]),
            "within_track": self.is_within_track(position),
        }


if __name__ == "__main__":
    track = TrackGeometry()
    print("\n" + "=" * 55)
    print("YAS MARINA TRACK GEOMETRY TEST")
    print("=" * 55)

    print("\nTRACK INFORMATION")
    print("Number of points:", track.number_of_points)

    print(
        "Track length:",
        round(track.get_track_length(), 2),
        "m"
    )

    print("\nFIRST TRACK POINT")
    print("Center:", track.centerline[0])
    print("Left:", track.left_boundary[0])
    print("Right:", track.right_boundary[0])

    test_position = track.centerline[0]

    result = track.get_track_position(test_position)

    print("\nVEHICLE POSITION TEST")
    print("Test position:", test_position)
    print("Nearest point:", np.round(result["point"], 3))
    print("Distance:", round(result["distance"], 4), "m")
    print("Lateral offset:", round(result["lateral_offset"], 4), "m")
    print("Progress:", round(result["progress"], 3), "m")
    print("Heading:", round(result["heading_deg"], 3), "degrees")
    print("Curvature:", round(result["curvature"], 6))
    print("Within track:", result["within_track"])

    test_position_outside = (
        track.centerline[0] + np.array([0.0, 30.0])
    )

    print("\nSECOND POSITION TEST")
    print("Test position:", test_position_outside)
    print(
        "Within track:",
        track.is_within_track(test_position_outside)
    )

    print("\nTRACK GEOMETRY TEST FINISHED")
