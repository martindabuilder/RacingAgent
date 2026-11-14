import matplotlib.pyplot as plt
from track_geometry import TrackGeometry

track = TrackGeometry()
plt.figure(figsize=(14, 10))
plt.plot(track.centerline[:, 0],track.centerline[:, 1],color="blue",label="Centerline",linewidth=1.5)
plt.plot(track.left_boundary[:, 0],track.left_boundary[:, 1],color="green",label="Left Boundary",linewidth=1)
plt.plot(track.right_boundary[:, 0],track.right_boundary[:, 1],color="red",label="Right Boundary",linewidth=1)
plt.title("Yas Marina Circuit - Track Geometry")
plt.xlabel("X (meters)")
plt.ylabel("Y (meters)")
plt.axis("equal")
plt.grid(True)
plt.legend()
plt.show()