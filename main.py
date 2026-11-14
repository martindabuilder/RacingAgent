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

        self.track_model = TrackModel(
            self.render,
            self.loader
        )

        self.track_model.load()
        self.track_model.show_centerline(
            self.track.centerline_3d,
            height=1.0
        )

        self.vehicle = F1Vehicle(
            self.render,
            self.loader,
            self.track
        )

        start_x = self.track.centerline_3d[0, 0]
        start_y = self.track.centerline_3d[0, 1]
        start_z = self.track.centerline_3d[0, 2]

        self.vehicle.set_position(
            start_x,
            start_y,
            start_z
        )

        start_heading = self.track.get_heading(0)
        self.vehicle.set_rotation(start_heading)

        track_state = self.vehicle.get_track_state()
        
        print("Position:", self.vehicle.get_position())
        print("Speed:", self.vehicle.get_speed())
        print("Heading:", self.vehicle.get_rotation())
        print("Distance from centerline:", track_state["distance"])
        print("Track progress:", track_state["progress"])
        print("Within track:", track_state["within_track"])

        self.camera.reparentTo(self.render)
        self.camera.setPos(0, 0, 100)
        self.camera.lookAt(0, 0, 0)

        self.disableMouse()

        print("\nF1 Simulator started!")

if __name__ == "__main__":
    simulator = F1Simulator()
    simulator.run()