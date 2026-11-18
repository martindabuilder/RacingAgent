import math
from panda3d.core import (NodePath, Vec3, Vec4, Material, MaterialAttrib)
from environment.track_geometry import TrackGeometry

class F1Vehicle:
    def __init__(self, render: NodePath, loader, track_geometry: TrackGeometry = None):
        self.render = render
        self.loader = loader
        self.track_geometry = track_geometry

        self.position = Vec3(0, 0, 0)
        self.speed = 0.0
        self.heading = 0.0

        self.front_left_wheel_pivot = None
        self.front_right_wheel_pivot = None

        self.front_left_wheel_initial_h = 0.0
        self.front_right_wheel_initial_h = 0.0

        self.wheel_roll_pivots = []
        self.wheel_radius = 0.39
        self.wheel_rotation = 0.0

        self.car = self.render.attachNewNode("f1_car")

        model_path = "assets/models/2020_f1_mercedes-benz_w11.glb"
        self.model = self.loader.loadModel(model_path)

        if self.model.isEmpty():
            raise RuntimeError(f"Could not load vehicle model: {model_path}")

        self.model.reparentTo(self.car)
        self.model.setScale(1.3)
        self.model.setH(-270)

        bounds = self.model.getTightBounds(self.car)

        if bounds is None:
            raise RuntimeError("Could not calculate Mercedes model bounds.")

        minimum, maximum = bounds
        center_x = (minimum.x + maximum.x) / 2.0
        center_y = (minimum.y + maximum.y) / 2.0
        self.model.setPos(-center_x, -center_y, -minimum.z)

        self._apply_visual_materials()
        self._setup_wheels()

    def _apply_visual_materials(self):
        geom_nodes = self.model.findAllMatches("**/+GeomNode")
        enhanced_count = 0

        for material_np in geom_nodes:
            geom_node = material_np.node()

            for i in range(geom_node.getNumGeoms()):
                state = geom_node.getGeomState(i)
                material_attrib = state.getAttrib(MaterialAttrib)

                if material_attrib is None:
                    continue

                original_material = material_attrib.getMaterial()

                if original_material is None:
                    continue

                material_name = original_material.getName().lower()
                material = Material(original_material)

                if "ext_44" in material_name:
                    material.setSpecular(Vec4(1, 1, 1, 1))
                    material.setShininess(180)

                elif "wheel" in material_name:
                    material.setSpecular(Vec4(0.95, 0.95, 0.95, 1))
                    material.setShininess(110)

                elif "tyre_tread" in material_name:
                    material.setSpecular(Vec4(0.025, 0.025, 0.025, 1))
                    material.setShininess(5)

                elif "misc" in material_name:
                    material.setSpecular(Vec4(0.85, 0.85, 0.85, 1))
                    material.setShininess(130)

                elif "car_chassis" in material_name:
                    material.setSpecular(Vec4(0.80, 0.80, 0.80, 1))
                    material.setShininess(140)

                elif "badges" in material_name:
                    material.setSpecular(Vec4(0.90, 0.90, 0.90, 1))
                    material.setShininess(130)

                else:
                    continue

                geom_node.setGeomState(i, state.addAttrib(MaterialAttrib.make(material)))
                enhanced_count += 1

        print(f"[Vehicle] Enhanced glossy materials: {enhanced_count}")

    def _setup_wheels(self):
        wheel_parts = {
            "front_left": [
                "LOD_A_WHEEL_mm_wheel.001_3",
                "LOD_A_TYRE_mm_tyre.001_2"
            ],
            "front_right": [
                "LOD_A_WHEEL_mm_wheel_13",
                "LOD_A_TYRE_mm_tyre_34"
            ],
            "rear_left": [
                "LOD_A_WHEEL_REAR_mm_wheel.001_1",
                "LOD_A_TYRE_REAR_mm_tyre.001_4"
            ],
            "rear_right": [
                "LOD_A_WHEEL_REAR_mm_wheel_39",
                "LOD_A_TYRE_REAR_mm_tyre_12"
            ]
        }

        for wheel, node_names in wheel_parts.items():
            parts = []

            for node_name in node_names:
                part = self.model.find(f"**/{node_name}")

                if part.isEmpty():
                    print(f"[Vehicle] Missing wheel component: {node_name}")
                    continue

                parts.append(part)

            if len(parts) != 2:
                print(f"[Vehicle] Could not initialize {wheel} wheel.")
                continue

            wheel_position = parts[0].getPos(self.car)
            pivot = self.car.attachNewNode(f"{wheel}_pivot")
            pivot.setPos(self.car, wheel_position)
            roll_pivot = pivot.attachNewNode(f"{wheel}_roll_pivot")

            for part in parts:
                if "TYRE" in part.getName():
                    bounds = part.getTightBounds(self.car)
                    if bounds is not None:
                        minimum, maximum = bounds
                        self.wheel_radius = (maximum.z - minimum.z) / 2.0
                part.wrtReparentTo(roll_pivot)

            self.wheel_roll_pivots.append(roll_pivot)

            if wheel == "front_left":
                self.front_left_wheel_pivot = pivot
                self.front_left_wheel_initial_h = pivot.getH(self.car)
            elif wheel == "front_right":
                self.front_right_wheel_pivot = pivot
                self.front_right_wheel_initial_h = pivot.getH(self.car)

            label = wheel.replace("_", " ").capitalize()
            print(f"[Vehicle] {label} wheel pivot ready.")

        print(f"[Vehicle] Wheel radius: {self.wheel_radius:.3f} m")

    def update_wheel_rotation(self, dt):
        dt = float(dt)

        if not math.isfinite(dt) or dt < 0.0:
            raise ValueError("Wheel rotation time step must be finite and non-negative.")

        if dt == 0.0 or self.wheel_radius <= 0.0:
            return

        rotation_delta = math.degrees(self.speed * dt / self.wheel_radius)
        self.wheel_rotation = (self.wheel_rotation + rotation_delta) % 360.0

        for pivot in self.wheel_roll_pivots:
            pivot.setR(self.wheel_rotation)

    def set_steering_angle(self, steering_angle):
        heading = math.degrees(float(steering_angle))

        if self.front_left_wheel_pivot is not None:
            self.front_left_wheel_pivot.setH(self.car, self.front_left_wheel_initial_h + heading)

        if self.front_right_wheel_pivot is not None:
            self.front_right_wheel_pivot.setH(self.car, self.front_right_wheel_initial_h + heading)

    def set_position(self, x, y, z=0):
        self.position = Vec3(float(x), float(y), float(z))
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
            raise RuntimeError("TrackGeometry has not been connected to F1Vehicle.")

        return self.track_geometry.get_track_position([self.position.x, self.position.y, self.position.z])
