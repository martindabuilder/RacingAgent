from direct.showbase.ShowBase import ShowBase
from panda3d.core import WindowProperties
from simulation.vehicle import F1Vehicle
from track.track_geometry import TrackGeometry
from track.track_model import TrackModel

class F1Simulator(ShowBase):
    def __init__(self):
        super().__init__()
        self.setBackgroundColor(0.05, 0.05, 0.08, 1)
        properties = WindowProperties()
        properties.setTitle("F1 Autonomous Racing Lab")

        self.win.requestProperties(properties)
        self.track = TrackGeometry()
        self.track_model = TrackModel(self.render, self.loader)
        self.track_model.load()

        self.vehicle = F1Vehicle(self.render, self.loader)
        start_x = self.track.centerline[0][0]
        start_y = self.track.centerline[0][1]

        self.vehicle.set_position(start_x, start_y, 0)
        start_heading = self.track.get_heading(0)
        self.vehicle.set_rotation(start_heading)

        self.camera.reparentTo(self.vehicle.car)
        self.camera.setPos(0, 15, 7)
        self.camera.lookAt(0, 0, 1)

        self.disableMouse()

        print("F1 Simulator started!")

if __name__ == "__main__":
    simulator = F1Simulator()
    simulator.run()

    