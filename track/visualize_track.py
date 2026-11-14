from pathlib import Path

from direct.showbase.ShowBase import ShowBase
from direct.gui.OnscreenText import OnscreenText

from panda3d.core import (
    LineSegs,
    AmbientLight,
    DirectionalLight,
    Filename,
    WindowProperties,
)

import gltf

from track.track_data import TrackData


class TrackVisualizer(ShowBase):

    def __init__(self):
        super().__init__()

        # -------------------------------------------------
        # 1. ПЪТИЩА ДО ФАЙЛОВЕТЕ
        # -------------------------------------------------

        project_root = Path(__file__).resolve().parent.parent

        model_path = (
            project_root
            / "assets"
            / "tracks"
            / "yas_marina"
            / "yas_marina_circuit_abu_dhabi_2021_layout.glb"
        )

        # -------------------------------------------------
        # 2. ЗАРЕЖДАНЕ НА CSV ГЕОМЕТРИЯТА
        # -------------------------------------------------

        self.track_data = TrackData()

        # -------------------------------------------------
        # 3. НАСТРОЙКИ НА ПРОЗОРЕЦА
        # -------------------------------------------------

        properties = WindowProperties()
        properties.setTitle(
            "Yas Marina - Track Geometry Inspector"
        )
        properties.setSize(1400, 900)

        self.win.requestProperties(properties)

        # -------------------------------------------------
        # 4. ЗАРЕЖДАНЕ НА 3D МОДЕЛА
        # -------------------------------------------------

        if not model_path.exists():
            raise FileNotFoundError(
                f"3D model not found: {model_path}"
            )

        self.track_model = self.loader.loadModel(
            Filename.fromOsSpecific(str(model_path))
        )

        if self.track_model.isEmpty():
            raise RuntimeError(
                "Could not load Yas Marina 3D model."
            )

        self.track_model.reparentTo(self.render)

        print("\n3D model loaded successfully.")

        # -------------------------------------------------
        # 5. ДИАГНОСТИКА НА КООРДИНАТИТЕ
        # -------------------------------------------------

        self.print_geometry_bounds()

        # -------------------------------------------------
        # 6. ОСВЕТЛЕНИЕ
        # -------------------------------------------------

        self.setup_lighting()

        # -------------------------------------------------
        # 7. ВИЗУАЛИЗАЦИЯ НА ЛИНИИТЕ
        # -------------------------------------------------

        self.draw_track_lines()

        # -------------------------------------------------
        # 8. КАМЕРА
        # -------------------------------------------------

        self.setup_camera()

        # -------------------------------------------------
        # 9. ИНФОРМАЦИЯ НА ЕКРАНА
        # -------------------------------------------------

        self.show_info()

        self.disableMouse()

        print("\nYas Marina visualization started.")

    # =====================================================
    # ДИАГНОСТИКА
    # =====================================================

    def print_geometry_bounds(self):
        """Извежда координатните граници на модела и CSV."""

        # Граници на 3D модела в координатите на render
        model_bounds = self.track_model.getTightBounds(
            self.render
        )

        print("\n===================================")
        print("          3D MODEL BOUNDS")
        print("===================================")

        if model_bounds:
            model_min, model_max = model_bounds

            print("Minimum:", model_min)
            print("Maximum:", model_max)
            print("Size:", model_max - model_min)

        else:
            print("Could not calculate model bounds.")

        # Граници на CSV централната линия
        centerline = self.track_data.centerline

        csv_min = centerline.min(axis=0)
        csv_max = centerline.max(axis=0)

        print("\n===================================")
        print("          CSV TRACK BOUNDS")
        print("===================================")

        print("Minimum:", csv_min)
        print("Maximum:", csv_max)
        print("Size:", csv_max - csv_min)

        print("\n===================================")
        print("          FIRST CSV POINT")
        print("===================================")

        print("Center:", self.track_data.centerline[0])
        print("Left:", self.track_data.left_boundary[0])
        print("Right:", self.track_data.right_boundary[0])

        print("\n===================================\n")

    # =====================================================
    # ОСВЕТЛЕНИЕ
    # =====================================================

    def setup_lighting(self):
        """Добавя осветление към 3D модела."""

        ambient = AmbientLight("ambient")
        ambient.setColor((0.65, 0.65, 0.65, 1))

        ambient_np = self.render.attachNewNode(ambient)
        self.render.setLight(ambient_np)

        directional = DirectionalLight("directional")
        directional.setColor((0.8, 0.8, 0.8, 1))

        directional_np = self.render.attachNewNode(
            directional
        )

        directional_np.setHpr(-45, -60, 0)

        self.render.setLight(directional_np)

    # =====================================================
    # НАЧЕРТАВАНЕ НА ЛИНИИТЕ
    # =====================================================

    def draw_line(
        self,
        name,
        points,
        color,
        thickness=2
    ):
        """Начертава линия през списък от 3D точки."""

        lines = LineSegs(name)

        lines.setColor(*color)
        lines.setThickness(thickness)

        height_offset = 0.5

        lines.moveTo(
            points[0][0],
            points[0][1],
            points[0][2] + height_offset
        )

        for point in points[1:]:

            lines.drawTo(
                point[0],
                point[1],
                point[2] + height_offset
            )

        # Затваряме обиколката
        lines.drawTo(
            points[0][0],
            points[0][1],
            points[0][2] + height_offset
        )

        node = lines.create()

        self.render.attachNewNode(node)

    def draw_track_lines(self):
        """Начертава централната линия и границите."""

        # Централна линия - циан
        self.draw_line(
            "centerline",
            self.track_data.centerline,
            (0, 1, 1, 1),
            3
        )

        # Лява граница - зелена
        self.draw_line(
            "left_boundary",
            self.track_data.left_boundary,
            (0, 1, 0, 1),
            2
        )

        # Дясна граница - червена
        self.draw_line(
            "right_boundary",
            self.track_data.right_boundary,
            (1, 0, 0, 1),
            2
        )

    # =====================================================
    # КАМЕРА
    # =====================================================

    def setup_camera(self):
        """Поставя камерата над пистата."""

        centerline = self.track_data.centerline

        min_x = centerline[:, 0].min()
        max_x = centerline[:, 0].max()

        min_y = centerline[:, 1].min()
        max_y = centerline[:, 1].max()

        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2

        self.camera.setPos(
            center_x,
            center_y,
            2200
        )

        self.camera.lookAt(
            center_x,
            center_y,
            0
        )

        self.camLens.setFov(45)

    # =====================================================
    # ИНФОРМАЦИЯ НА ЕКРАНА
    # =====================================================

    def show_info(self):
        """Показва легенда на екрана."""

        OnscreenText(
            text="YAS MARINA - TRACK GEOMETRY",
            pos=(-1.28, 0.92),
            scale=0.055,
            fg=(1, 1, 1, 1),
            align=0
        )

        OnscreenText(
            text=(
                "CYAN: Centerline   "
                "GREEN: Left   "
                "RED: Right"
            ),
            pos=(-1.28, 0.85),
            scale=0.04,
            fg=(1, 1, 1, 1),
            align=0
        )


# =========================================================
# СТАРТИРАНЕ НА ПРОГРАМАТА
# =========================================================

if __name__ == "__main__":

    app = TrackVisualizer()

    app.run()
