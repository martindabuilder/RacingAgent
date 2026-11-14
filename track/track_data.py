from pathlib import Path
import numpy as np

class TrackData:
   
    def __init__(self):
        self.project_root = Path(__file__).resolve().parent.parent
        self.csv_path = (
            self.project_root
            / "assets"
            / "tracks"
            / "yas_marina"
            / "yas_marina_2021_track.csv"
        )

        self.data = None

        self.load_data()
        self.validate_data()

    def load_data(self):

        if not self.csv_path.exists():
            raise FileNotFoundError(
                f"Track CSV not found: {self.csv_path}"
            )

        self.data = np.genfromtxt(
            self.csv_path,
            delimiter=",",
            names=True,
            dtype=float,
            encoding="utf-8"
        )

        if self.data.size == 0:
            raise ValueError("Track CSV is empty.")

        print("\nTRACK DATA LOADED")
        print("-" * 50)
        print("File:", self.csv_path.name)
        print("Number of points:", self.data.size)

 
    def validate_data(self):

        required_columns = [
            "s_m",
            "x_m",
            "y_m",
            "z_m",
            "heading_rad",
            "curvature_1pm",
            "width_m",
            "left_x_m",
            "left_y_m",
            "right_x_m",
            "right_y_m",
        ]

        available_columns = self.data.dtype.names

        missing = [
            column
            for column in required_columns
            if column not in available_columns
        ]

        if missing:
            raise ValueError(
                f"Missing CSV columns: {missing}"
            )

        for column in required_columns:

            values = self.data[column]

            if not np.isfinite(values).all():
                raise ValueError(
                    f"Invalid values found in column: {column}"
                )

        if len(self.data) < 3:
            raise ValueError(
                "Track requires at least 3 points."
            )

        if np.any(self.data["width_m"] <= 0):
            raise ValueError(
                "Track width must be positive."
            )

        if np.any(np.diff(self.data["s_m"]) <= 0):
            raise ValueError(
                "Track distance values must increase."
            )

        print("Data validation: PASSED")

    @property
    def centerline(self):

        return np.column_stack((
            self.data["x_m"],
            self.data["y_m"],
            self.data["z_m"]
        ))

    @property
    def left_boundary(self):

        return np.column_stack((
            self.data["left_x_m"],
            self.data["left_y_m"],
            self.data["z_m"]
        ))

    @property
    def right_boundary(self):

        return np.column_stack((
            self.data["right_x_m"],
            self.data["right_y_m"],
            self.data["z_m"]
        ))

    @property
    def distance(self):
        return self.data["s_m"]

    @property
    def heading(self):
        return self.data["heading_rad"]

    @property
    def curvature(self):
        return self.data["curvature_1pm"]

    @property
    def width(self):
        return self.data["width_m"]

    @property
    def elevation(self):
        return self.data["z_m"]

    @property
    def number_of_points(self):
        return len(self.data)

    @property
    def length(self):
        return float(self.distance[-1] - self.distance[0])

    @property
    def average_width(self):
        return float(np.mean(self.width))

    def print_summary(self):

        print("\nYAS MARINA TRACK SUMMARY")
        print("=" * 50)

        print("Points:", self.number_of_points)
        print("Length:", round(self.length, 2), "m")
        print("Average width:", round(self.average_width, 2), "m")

        print("\nCenterline:")
        print("  First:", np.round(self.centerline[0], 3))
        print("  Last:", np.round(self.centerline[-1], 3))

        print("\nElevation:")
        print("  Minimum:", round(float(self.elevation.min()), 3))
        print("  Maximum:", round(float(self.elevation.max()), 3))

        print("\nCurvature:")
        print("  Minimum:", round(float(self.curvature.min()), 6))
        print("  Maximum:", round(float(self.curvature.max()), 6))

        print("\nTrack data source is provisional.")
        print("GLB alignment has not yet been fully verified.")


if __name__ == "__main__":
    track = TrackData()
    track.print_summary()

    print("\nARRAY SHAPES")
    print("-" * 50)
    print("Centerline:", track.centerline.shape)
    print("Left boundary:", track.left_boundary.shape)
    print("Right boundary:", track.right_boundary.shape)
    print("Heading:", track.heading.shape)
    print("Curvature:", track.curvature.shape)
    print("Width:", track.width.shape)