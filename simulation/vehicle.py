from panda3d.core import NodePath, Vec3
from track.track_geometry import TrackGeometry

class F1Vehicle:

    def __init__(
        self,
        render: NodePath,
        loader,
        track_geometry: TrackGeometry = None
    ):
        self.render = render
        self.loader = loader

        self.track_geometry = track_geometry

        self.position = Vec3(0, 0, 0)
        self.speed = 0.0
        self.heading = 0.0

        self.car = self.render.attachNewNode("f1_car")
        model_path = "assets/models/2020_f1_mercedes-benz_w11.glb"
        self.model = self.loader.loadModel(model_path)
        self.model.reparentTo(self.car)

        self.model.setScale(2)
        self.model.setPos(0, 0, 0)
        print("Mercedes AMG W11 model loaded!")

    def set_position(self, x, y, z=0):

        self.position = Vec3(x, y, z)
        self.car.setPos(self.position)

    def get_position(self):
        return self.position

    def set_speed(self, speed):
        self.speed = max(0.0, float(speed))

    def get_speed(self):
        return self.speed
        
    def set_rotation(self, heading):
        self.heading = float(heading)
        self.car.setH(self.heading)

    def get_rotation(self):
        return self.heading

    def get_track_state(self):

        if self.track_geometry is None:
            raise RuntimeError(
                "TrackGeometry has not been connected to F1Vehicle."
            )

        return self.track_geometry.get_track_position(
            [
                self.position.x,
                self.position.y,
                self.position.z
            ]
        )
