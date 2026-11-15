from direct.showbase.ShowBase import ShowBase
from direct.filter.CommonFilters import CommonFilters
from panda3d.core import (WindowProperties, AmbientLight, DirectionalLight, Vec4,)
from simulation.vehicle import F1Vehicle
from track.track_geometry import TrackGeometry
from track.track_mesh import TrackMesh
from track.track_environment import TrackEnvironment
from track.track_barriers import TrackBarriers
from track.track_grandstands import TrackGrandstands
from track.track_lighting import TrackLighting
from track.track_facilities import TrackFacilities
from track.track_markings import TrackMarkings


class YasMarinaSimulator(ShowBase):
    def __init__(self):
        super().__init__()

        self.render.setShaderAuto()

        properties = WindowProperties()
        properties.setTitle("Yas Marina - F1 Autonomous Racing Simulator")
        properties.setSize(1600, 900)

        self.win.requestProperties(properties)

        self.setBackgroundColor(
            0.008,
            0.015,
            0.045,
            1.0,
        )

   
        ambient_light = AmbientLight("night_ambient")
        ambient_light.setColor(
            Vec4(0.16, 0.19, 0.30, 1.0)
        )

        self.ambient_light_node = (
            self.render.attachNewNode(ambient_light)
        )

        self.render.setLight(
            self.ambient_light_node
        )

        moon_light = DirectionalLight("moon_light")
        moon_light.setColor(
            Vec4(0.30, 0.34, 0.48, 1.0)
        )

        self.moon_light_node = (
            self.render.attachNewNode(moon_light)
        )

        self.moon_light_node.setHpr(
            35,
            -55,
            0,
        )

        self.render.setLight(
            self.moon_light_node
        )

        self.track = TrackGeometry()

        self.track_mesh = TrackMesh(
            self.render,
            self.track,
        )

        self.track_mesh.create()

  
        self.environment = TrackEnvironment(
            self.render,
            self.track,
        )

        self.environment.create()


        self.barriers = TrackBarriers(
            self.render,
            self.track,
        )

        self.barriers.create()

        self.track_markings = TrackMarkings(
            self.render,
            self.track
        )   

        self.track_markings.create()


        self.grandstands = TrackGrandstands(
            self.track,
            self.render,
        )

        self.grandstands.create()

        self.track_lighting = TrackLighting(
            self.render,
            self.track,
        )

        self.track_lighting.create()

        self.vehicle = F1Vehicle(
            self.render,
            self.loader,
            self.track,
        )

        start_position = self.track.centerline[0]

        self.facilities = TrackFacilities(
            self.render,
            self.track,
        )

        self.facilities.create()

  
        self.vehicle.set_position(
            float(start_position[0]),
            float(start_position[1]),
            0.2,
        )

        start_heading = self.track.get_heading(0)

        self.vehicle.set_rotation(
            start_heading
        )

        centerline = self.track.centerline

        min_x = float(centerline[:, 0].min())
        max_x = float(centerline[:, 0].max())

        min_y = float(centerline[:, 1].min())
        max_y = float(centerline[:, 1].max())

        center_x = (min_x + max_x) / 2.0
        center_y = (min_y + max_y) / 2.0

        self.camera.setPos(
            center_x,
            center_y,
            1100,
        )

        self.camera.lookAt(
            center_x,
            center_y,
            0,
        )

        self.camLens.setFov(50)

        self.filters = CommonFilters(
            self.win,
            self.cam,
        )

        try:
            bloom_enabled = self.filters.setBloom(
                blend=(0.3, 0.3, 0.3, 0.0),
                desat=0.0,
                intensity=1.5,
                size="medium",
                mintrigger=0.6,
                maxtrigger=1.0,
            )

            if bloom_enabled:
                print(
                    "[Graphics] Bloom enabled successfully."
                )
            else:
                print(
                    "[Graphics] Bloom could not be enabled. "
                    "Graphics hardware or shaders may not support it."
                )

        except Exception as error:
            print(
                "[Graphics] Bloom initialization failed:",
                error,
            )


if __name__ == "__main__":
    simulator = YasMarinaSimulator()
    simulator.run()