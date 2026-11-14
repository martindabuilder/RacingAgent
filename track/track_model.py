from pathlib import Path
from panda3d.core import Filename, NodePath

class TrackModel:
    def __init__(self, render: NodePath, loader):
        self.render = render
        self.loader = loader
        self.model = None

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
            raise RuntimeError("Failed to load Yas Marina 3D model.")

        self.model.reparentTo(self.render)

        print("Yas Marina 3D model loaded!")
        return self.model
