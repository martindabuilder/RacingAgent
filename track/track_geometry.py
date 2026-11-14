import numpy as np
from pathlib import Path

class TrackGeometry:
    def __init__(self, csv_path=None):
        if csv_path is None:
            project_root = Path(__file__).resolve().parents[1]
            csv_path = (
                project_root
                / "assets"
                / "tracks"
                / "yas_marina"
                / "YasMarina.csv"
            )

        self.csv_path = Path(csv_path)

        self.load_track()
        self.calculate_boundaries()

    def load_track(self):
        data = np.loadtxt(self.csv_path,delimiter=",", comments="#")

        if data.ndim != 2 or data.shape[1] != 4:
            raise ValueError("CSV file must contain exactly 4 columns.")

        if len(data) < 3:
            raise ValueError("Track needs at least 3 points.")

        if not np.isfinite(data).all():
            raise ValueError("Track contains invalid values.")

        self.centerline = data[:, :2]
        self.width_right = data[:, 2]
        self.width_left = data[:, 3]

        print(f"Track loaded: {len(self.centerline)} points")

    def calculate_boundaries(self):
        points = self.centerline
        previous_points = np.roll(points, 1, axis=0)
        next_points = np.roll(points, -1, axis=0)
        tangent = next_points - previous_points
        lengths = np.linalg.norm(tangent, axis=1)

        if np.any(lengths < 1e-8):
            raise ValueError("Invalid track direction vectors.")

        tangent = tangent / lengths[:, None]
        left_normal = np.column_stack((-tangent[:, 1],tangent[:, 0]))
        self.left_boundary = (points + left_normal * self.width_left[:, None])
        self.right_boundary = (points - left_normal * self.width_right[:, None])

        print("Track boundaries calculated!")

    def get_track_length(self):
        differences = np.diff(self.centerline, axis=0)
        segment_lengths = np.linalg.norm(differences,axis=1)
        closing_distance = np.linalg.norm(self.centerline[0] - self.centerline[-1])

        return np.sum(segment_lengths) + closing_distance


if __name__ == "__main__":

    track = TrackGeometry()
    print("\n--- YAS MARINA TRACK ---")
    print("Number of points:", len(track.centerline))

    print(
        "First centerline point:",
        track.centerline[0]
    )

    print(
        "First left boundary:",
        track.left_boundary[0]
    )

    print(
        "First right boundary:",
        track.right_boundary[0]
    )

    print(
        "Approximate track length:",
        round(track.get_track_length(), 2),
        "meters"
    )
