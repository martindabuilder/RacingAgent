from pathlib import Path
from panda3d.core import (
    Filename,
    NodePath,
    MaterialAttrib,
    TextureAttrib,
    LineSegs,
)


class TrackModel:
    def __init__(self, render: NodePath, loader):
        self.render = render
        self.loader = loader
        self.model = None
        self.centerline_node = None

        project_root = Path(__file__).resolve().parents[1]

        self.model_path = (
            project_root
            / "assets"
            / "tracks"
            / "yas_marina"
            / "yas_marina_circuit_abu_dhabi_2021_layout.glb"
        )

    def load(self):
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Track model not found: {self.model_path}"
            )

        self.model = self.loader.loadModel(
            Filename.fromOsSpecific(str(self.model_path))
        )

        if self.model is None:
            raise RuntimeError(
                "Failed to load Yas Marina 3D model."
            )

        self.model.reparentTo(self.render)

        print("\n--- 3D MODEL COMPONENTS ---")

        geometry_nodes = self.model.findAllMatches("**/+GeomNode")

        print(
            "Number of geometry nodes:",
            geometry_nodes.getNumPaths()
        )

        print("\n--- COMPONENT MATERIALS AND DIMENSIONS ---")

        for i in range(geometry_nodes.getNumPaths()):
            node_path = geometry_nodes.getPath(i)
            bounds = node_path.getTightBounds(self.render)

            if bounds:
                min_point, max_point = bounds

                size_x = max_point.x - min_point.x
                size_y = max_point.y - min_point.y
                size_z = max_point.z - min_point.z

                center_x = (min_point.x + max_point.x) / 2
                center_y = (min_point.y + max_point.y) / 2
                center_z = (min_point.z + max_point.z) / 2

                print(f"\nComponent {i}: {node_path.getName()}")

                print(
                    f"Center: X={center_x:.2f}, "
                    f"Y={center_y:.2f}, Z={center_z:.2f}"
                )

                print(
                    f"Size: X={size_x:.2f}, "
                    f"Y={size_y:.2f}, Z={size_z:.2f}"
                )

            geom_node = node_path.node()

            for geom_index in range(geom_node.getNumGeoms()):
                state = geom_node.getGeomState(geom_index)

                material_attrib = state.getAttrib(MaterialAttrib)
                material_name = "-"

                if material_attrib:
                    material = material_attrib.getMaterial()

                    if material:
                        material_name = material.getName() or "-"

                texture_attrib = state.getAttrib(TextureAttrib)
                texture_names = []

                if texture_attrib:
                    for j in range(texture_attrib.getNumOnStages()):
                        stage = texture_attrib.getOnStage(j)
                        texture = texture_attrib.getOnTexture(stage)

                        if texture:
                            texture_names.append(texture.getName())

                print(
                    f"  Geom {geom_index} | "
                    f"Material: {material_name} | "
                    f"Textures: {texture_names}"
                )

        print("\n--- MODEL POSITION ---")
        print("Position:", self.model.getPos())
        print("Rotation:", self.model.getHpr())
        print("Scale:", self.model.getScale())

        print("\nYas Marina 3D model loaded!")

        return self.model

    def show_centerline(self, centerline, height=0.8):
        if len(centerline) < 2:
            raise ValueError(
                "Centerline must contain at least 2 points."
            )

        lines = LineSegs()
        lines.setThickness(2.5)
        lines.setColor(0.0, 1.0, 1.0, 1.0)

        first_point = centerline[0]

        lines.moveTo(
            float(first_point[0]),
            float(first_point[1]),
            height
        )

        for point in centerline[1:]:
            lines.drawTo(
                float(point[0]),
                float(point[1]),
                height
            )

        lines.drawTo(
            float(first_point[0]),
            float(first_point[1]),
            height
        )

        self.centerline_node = self.render.attachNewNode(
            lines.create()
        )

        print("CSV centerline visualization created!")
