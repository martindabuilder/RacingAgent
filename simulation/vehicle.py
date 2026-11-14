from panda3d.core import NodePath, Vec3

class F1Vehicle:
    def __init__(self, render: NodePath, loader):
        self.render = render
        self.loader = loader
        self.position = Vec3(0, 0, 0)
        self.speed = 0.0
        self.car = self.render.attachNewNode("f1_car")
        model_path = "assets/models/2020_f1_mercedes-benz_w11.glb"
        self.model = self.loader.loadModel(model_path)
        self.model.reparentTo(self.car)
        self.model.setScale(0.8)
        self.model.setPos(0, 0, 0)
        print("Mercedes AMG W11 model loaded!")

    def set_position(self, x, y, z=0):
        self.position = Vec3(x, y, z)
        self.car.setPos(self.position)

    def set_speed(self, speed):
        self.speed = speed

    def set_rotation(self, heading):
        self.car.setH(heading)
