from pathlib import Path
import csv
import math

class TrackData:
    def __init__(self):
        project_root = Path(__file__).resolve().parent.parent.parent

        self.data_dir = (
            project_root
            / "assets"
            / "tracks"
            / "yas_marina"
        )

        self.centerline_file = self.data_dir / "YasMarina_centerline.csv"
        self.raceline_file = self.data_dir / "YasMarina_raceline.csv"
        self.map_file = self.data_dir / "YasMarina_map.yaml"
        self.map_image_file = self.data_dir / "YasMarina_map.png"

        self.centerline = []
        self.raceline = []

        self.resolution = 0.0
        self.origin = (0.0, 0.0, 0.0)

        # F1TENTH data is in 1:10 scale.
        self.scale = 10.0

        self.length = 0.0

    def load(self):
        print("\nLOADING YAS MARINA TRACK DATA")
        print("=" * 45)

        self._check_files()

        self.centerline = self._read_csv(
            self.centerline_file,
            [
                "x_m",
                "y_m",
                "w_tr_right_m",
                "w_tr_left_m"
            ]
        )

        self.raceline = self._read_csv(
            self.raceline_file,
            [
                "s_m",
                "x_m",
                "y_m",
                "psi_rad",
                "kappa_radpm",
                "vx_mps",
                "ax_mps2"
            ]
        )

        self._load_map_metadata()
        self._validate()
        self._calculate_length()
        self._print_report()

        return self

    def _check_files(self):
        required_files = [
            self.centerline_file,
            self.raceline_file,
            self.map_file,
            self.map_image_file
        ]

        for file_path in required_files:
            if not file_path.exists():
                raise FileNotFoundError(
                    f"Missing track file: {file_path}"
                )

    @staticmethod
    def _read_csv(file_path, expected_columns):
        rows = []
        header = None

        with open(
            file_path,
            "r",
            encoding="utf-8-sig",
            newline=""
        ) as file:

            for raw_line in file:

                line = raw_line.strip()

                if not line:
                    continue

                # Ignore UUID and other comment lines,
                # but keep the column header.
                if line.startswith("#"):
                    line = line[1:].strip()

                # Both files may use semicolons or commas.
                if ";" in line:
                    delimiter = ";"
                else:
                    delimiter = ","

                parts = next(
                    csv.reader(
                        [line],
                        delimiter=delimiter
                    )
                )

                parts = [
                    value.replace("\u00a0", " ").strip()
                    for value in parts
                ]

                normalized = [
                    value.lstrip("#").strip()
                    for value in parts
                ]

                # Find the header.
                if all(
                    column in normalized
                    for column in expected_columns
                ):
                    header = normalized
                    continue

                if header is None:
                    continue

                # Skip Markdown separator rows.
                if all(
                    value and set(value) <= {"-", ":"}
                    for value in parts
                ):
                    continue

                # Read numeric values.
                try:
                    values = [
                        float(value)
                        for value in parts
                    ]
                except ValueError:
                    continue

                if len(values) != len(header):
                    continue

                rows.append(
                    dict(zip(header, values))
                )

        if header is None:
            raise ValueError(
                f"Could not find expected columns in {file_path.name}. "
                f"Expected: {expected_columns}"
            )

        if not rows:
            raise ValueError(
                f"No valid data rows in {file_path.name}"
            )

        return rows

    def _load_map_metadata(self):
        metadata = {}

        with open(
            self.map_file,
            "r",
            encoding="utf-8"
        ) as file:

            for line in file:
                line = line.strip()

                if not line or line.startswith("#"):
                    continue

                if ":" not in line:
                    continue

                key, value = line.split(":", 1)
                metadata[key.strip()] = value.strip()

        if "resolution" not in metadata:
            raise ValueError("Map resolution is missing.")

        if "origin" not in metadata:
            raise ValueError("Map origin is missing.")

        self.resolution = float(metadata["resolution"])

        self.origin = tuple(
            float(value.strip())
            for value in metadata["origin"].strip("[]").split(",")
        )

    def _validate(self):
        if len(self.centerline) < 3:
            raise ValueError("Centerline needs at least 3 points.")

        if len(self.raceline) < 3:
            raise ValueError("Raceline needs at least 3 points.")

        for point in self.centerline:
            if point["w_tr_right_m"] <= 0:
                raise ValueError("Invalid right track width.")

            if point["w_tr_left_m"] <= 0:
                raise ValueError("Invalid left track width.")

        for point in self.raceline:
            if point["vx_mps"] < 0:
                raise ValueError("Invalid raceline speed.")

        if self.resolution <= 0:
            raise ValueError("Invalid map resolution.")

        if len(self.origin) != 3:
            raise ValueError("Invalid map origin.")

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
            (
                point["x_m"] * self.scale,
                point["y_m"] * self.scale,
                0.0
            )
            for point in self.centerline
        ]

    def get_raceline(self):
        return [
            (
                point["x_m"] * self.scale,
                point["y_m"] * self.scale,
                0.0
            )
            for point in self.raceline
        ]

    def get_track_widths(self):
        return [
            (
                point["w_tr_left_m"] * self.scale,
                point["w_tr_right_m"] * self.scale
            )
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

        print("\nTRACK DATA LOADED")
        print("-" * 45)

        print("Centerline points:", len(self.centerline))
        print("Raceline points:", len(self.raceline))
        print("Track length:", round(self.length, 2), "m")

        print(
            "Average left width:",
            round(average_left * self.scale, 2),
            "m"
        )

        print(
            "Average right width:",
            round(average_right * self.scale, 2),
            "m"
        )

        print("Scale:", f"1:{int(self.scale)}")
        print("Map resolution:", self.resolution, "m/pixel")
        print("Map origin:", self.origin)

        print("\nTRACK DATA READY")
        print("=" * 45)
