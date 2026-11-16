from environment.track_data import TrackData
from environment.track_geometry import TrackGeometry

class TrackLoader:
    @staticmethod
    def load(track_name):
        track_data = TrackData(track_name).load()

        track = TrackGeometry(track_data)

        print(f"\nTRACK GEOMETRY READY: {track_name.upper()}")
        print("Track length:", round(track.track_length, 2), "m")
        print("Centerline points:", track.number_of_points)

        return track
