from pathlib import Path
import csv
import math

class TrackData:
    def __init__(self, track_name, scale=10.0):
        project_root = Path(__file__).resolve().parent.parent

        self.track_name = track_name.lower().strip()
        self.scale = float(scale)

        self.data_dir = (
            project_root
            / "assets"
            / "tracks"
            / self.track_name
        )

        self.centerline_file = None
        self.raceline_file = None
        self.map_file = None
        self.map_image_file = None

        self.centerline = []
        self.raceline = []

        self.resolution = 0.0
        self.origin = (0.0, 0.0, 0.0)

        self.length = 0.0

        if not self.track_name:
            raise ValueError("Track name cannot be empty.")

        if self.scale <= 0.0:
            raise ValueError("Track scale must be positive.")

    def load(self):
        print(f"\nLOADING {self.track_name.upper()} TRACK DATA")

        self._find_track_files()
        self.centerline = self._read_csv(self.centerline_file,["x_m", "y_m", "w_tr_right_m", "w_tr_left_m"])

        if self.raceline_file is not None:
            self.raceline = self._read_csv(self.raceline_file, ["s_m", "x_m", "y_m", "psi_rad", "kappa_radpm", "vx_mps", "ax_mps2"])

        if self.map_file is not None:
            self._load_map_metadata()

        self._validate()
        self._calculate_length()
        self._print_report()

        return self

    def _find_track_files(self):
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Track directory not found: {self.data_dir}")

        files = list(self.data_dir.iterdir())

        centerline_files = [
            file for file in files
            if file.is_file()
            and file.suffix.lower() == ".csv"
            and "centerline" in file.stem.lower()
        ]

        raceline_files = [
            file for file in files
            if file.is_file()
            and file.suffix.lower() == ".csv"
            and "raceline" in file.stem.lower()
        ]

        map_files = [
            file for file in files
            if file.is_file()
            and file.suffix.lower() in (".yaml", ".yml")
        ]

        map_images = [
            file for file in files
            if file.is_file()
            and file.suffix.lower() in (".png", ".pgm")
        ]

        if len(centerline_files) != 1:
            raise FileNotFoundError(f"Expected exactly one centerline CSV in {self.data_dir}, found {len(centerline_files)}.")

        if len(raceline_files) > 1:
            raise ValueError(f"Multiple raceline CSV files found in {self.data_dir}.")

        self.centerline_file = centerline_files[0]

        if raceline_files:
            self.raceline_file = raceline_files[0]

        if map_files:
            self.map_file = map_files[0]

        if map_images:
            self.map_image_file = map_images[0]

    @staticmethod
    def _read_csv(file_path, expected_columns):
        rows = []
        header = None

        with open(file_path, "r", encoding="utf-8-sig", newline="") as file:

            for raw_line in file:
                line = raw_line.strip()

                if not line:
                    continue

                if line.startswith("#"):
                    line = line.lstrip("#").strip()

                delimiter = ";" if ";" in line else ","
                parts = next(csv.reader([line], delimiter=delimiter))

                parts = [
                    value.replace("\u00a0", " ").strip()
                    for value in parts
                ]

                normalized = [
                    value.lstrip("#").strip()
                    for value in parts
                ]

                if all(
                    column in normalized
                    for column in expected_columns
                ):
                    header = normalized
                    continue

                if header is None:
                    continue

                if all(
                    value and set(value) <= {"-", ":"}
                    for value in parts
                ):
                    continue

                try:
                    values = [
                        float(value)
                        for value in parts
                    ]
                except ValueError:
                    continue

                if len(values) != len(header):
                    continue

                rows.append(dict(zip(header, values)))

        if header is None:
            raise ValueError(
                f"Could not find expected columns in {file_path.name}. "
                f"Expected: {expected_columns}"
            )

        if not rows:
            raise ValueError(f"No valid data rows in {file_path.name}.")

        return rows

    def _load_map_metadata(self):
        metadata = {}

        with open(self.map_file, "r", encoding="utf-8") as file:

            for line in file:
                line = line.strip()

                if not line or line.startswith("#"):
                    continue

                if ":" not in line:
                    continue

                key, value = line.split(":", 1)
                metadata[key.strip()] = value.strip()

        if "resolution" in metadata:
            self.resolution = float(metadata["resolution"])

        if "origin" in metadata:
            self.origin = tuple(
                float(value.strip())
                for value in metadata["origin"].strip("[]").split(",")
            )

    def _validate(self):
        if len(self.centerline) < 3:
            raise ValueError(
                "Centerline needs at least 3 points."
            )

        for point in self.centerline:
            if point["w_tr_right_m"] <= 0.0:
                raise ValueError("Invalid right track width.")

            if point["w_tr_left_m"] <= 0.0:
                raise ValueError("Invalid left track width.")

            if not all(
                math.isfinite(point[column])
                for column in ("x_m", "y_m", "w_tr_right_m", "w_tr_left_m")):
                raise ValueError("Centerline contains invalid values.")

        if self.raceline:
            for point in self.raceline:
                if point["vx_mps"] < 0.0:
                    raise ValueError("Invalid raceline speed.")

    def _calculate_length(self):
        total = 0.0

        for i in range(len(self.centerline)):
            current = self.centerline[i]
            following = self.centerline[
                (i + 1) % len(self.centerline)
            ]

            dx = following["x_m"] - current["x_m"]
            dy = following["y_m"] - current["y_m"]

            total += math.hypot(dx, dy)

        self.length = total * self.scale

    def get_centerline(self):
        return [
            (point["x_m"] * self.scale, point["y_m"] * self.scale, 0.0)
            for point in self.centerline
        ]

    def get_raceline(self):
        return [
            (point["x_m"] * self.scale, point["y_m"] * self.scale, 0.0)
            for point in self.raceline
        ]

    def get_track_widths(self):
        return [
            (point["w_tr_left_m"] * self.scale, point["w_tr_right_m"] * self.scale)
            for point in self.centerline
        ]

    def _print_report(self):
        average_left = sum(
            point["w_tr_left_m"]
            for point in self.centerline
        ) / len(self.centerline)

        average_right = sum(
            point["w_tr_right_m"]
            for point in self.centerline
        ) / len(self.centerline)

        print("Track:", self.track_name)
        print("Centerline points:", len(self.centerline), "Raceline points:", len(self.raceline), "Track length:", round(self.length, 2), "m")

        print("Average left width:", round(average_left * self.scale, 2), "m")
        print("Average right width:", round(average_right * self.scale, 2), "m")
        print("Scale:", f"1:{int(self.scale)}")
        print("Centerline file:", self.centerline_file.name)

        if self.raceline_file is not None:
            print("Raceline file:", self.raceline_file.name)

        if self.map_file is not None:
            print("Map metadata:", self.map_file.name)