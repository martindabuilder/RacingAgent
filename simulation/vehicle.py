from panda3d.core import (NodePath, Vec3, Vec4, Material,MaterialAttrib)
from tracks.yas_marina.track_geometry import TrackGeometry

class F1Vehicle:
    def __init__(self, render: NodePath,loader, track_geometry: TrackGeometry = None):
        self.render = render
        self.loader = loader
        self.track_geometry = track_geometry

        self.position = Vec3(0, 0, 0)
        self.speed = 0.0
        self.heading = 0.0

        self.car = self.render.attachNewNode("f1_car")
        model_path = "assets/models/2020_f1_mercedes-benz_w11.glb"
        self.model = self.loader.loadModel(model_path)

        if self.model.isEmpty():
            raise RuntimeError(f"Could not load vehicle model: {model_path}")

        self.model.reparentTo(self.car)

        self.model.setScale(2.0)
        self.model.setH(-270)

        bounds = self.model.getTightBounds(self.car)

        if bounds is None:
            raise RuntimeError(
                "Could not calculate Mercedes model bounds."
            )

        minimum, maximum = bounds
        center_x = (minimum.x + maximum.x) / 2.0
        center_y = (minimum.y + maximum.y) / 2.0
        self.model.setPos( -center_x, -center_y, -minimum.z)
        dimensions = maximum - minimum
        self._apply_visual_materials()

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
                    material.setSpecular(Vec4(0.75, 0.75, 0.75, 1.0))
                    material.setShininess(96.0)
                    category = "Glossy bodywork"

                elif "wheel" in material_name:
                    material.setSpecular(Vec4(0.85, 0.85, 0.85, 1.0))
                    material.setShininess(90.0)
                    category = "Metallic wheels"

                elif "tyre_tread" in material_name:
                    material.setSpecular(Vec4(0.045, 0.045, 0.045, 1.0))
                    material.setShininess(6.0)
                    category = "Matte tyres"

                elif "misc" in material_name:
                    material.setSpecular(Vec4(0.42, 0.42, 0.42, 1.0))
                    material.setShininess(64.0)
                    category = "Carbon and details"

                elif "car_chassis" in material_name:
                    material.setSpecular(Vec4(0.38, 0.38, 0.38, 1.0))
                    material.setShininess(52.0)
                    category = "Chassis"

                elif "badges" in material_name:
                    material.setSpecular(Vec4(0.55, 0.55, 0.55, 1.0))
                    material.setShininess(72.0)
                    category = "Badges"

                else:
                    continue

                new_state = state.addAttrib(MaterialAttrib.make(material))
                geom_node.setGeomState(i, new_state)
                enhanced_count += 1
                
    def set_position(self, x, y, z=0):

        self.position = Vec3(
            float(x),
            float(y),
            float(z)
        )

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

        return self.track_geometry.get_track_position([self.position.x, self.position.y, self.position.z])
