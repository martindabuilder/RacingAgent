from direct.showbase.ShowBase import ShowBase
from panda3d.core import WindowProperties
from simulation.vehicle import F1Vehicle

class F1Simulator(ShowBase):
    def __init__(self):
        super().__init__()
        self.setBackgroundColor(0.05, 0.05, 0.08, 1)
        properties = WindowProperties()
        properties.setTitle("F1 Autonomous Racing Lab")
        self.win.requestProperties(properties)
        self.vehicle = F1Vehicle(self.render, self.loader)
        self.disableMouse()
        self.camera.setPos(0, 8, 3.5)
        self.camera.lookAt(0, 0, 0.5)
        print("F1 Simulator started!")

if __name__ == "__main__":
    simulator = F1Simulator()
    simulator.run()