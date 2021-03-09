import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt

H = np.array([
    [-0.974793639765174],
    [ 0.157761465309759],
    [ 0.157761465309759],
])

fig = plt.figure()
ax = fig.gca(projection="3d")

x, y, z = np.meshgrid(np.arange(-0.80, 1.0, 0.2),
                      np.arange(-0.80, 1.0, 0.2),
                      np.arange(-0.80, 1.0, 0.2))


u = H[0] * y * z
v = H[1] * x * z
w = H[2] * x * y

ax.quiver(x, y, z, u, v, w, length=0.2)
plt.show()
"""

nx = 10
ny = 10
nz = 10

origin = (-(nx - 1)*0.1/2, -(ny - 1)*0.1/2, -(nz - 1)*0.1/2)
mesh = pv.UniformGrid((nx, ny, nz), (.1, .1, .1), origin)
x = mesh.points[:, 0]
y = mesh.points[:, 1]
z = mesh.points[:, 2]
vectors = np.empty((mesh.n_points, 3))
vectors[:, 0] = H[0] * y * z
vectors[:, 1] = H[1] * x * z
vectors[:, 2] = H[2] * x * y

mesh['vectors'] = vectors

stream, src = mesh.streamlines('vectors', return_source=True,
                               terminal_speed=0.0, n_points=200,
                               source_radius=0.1)
cpos = [(1.2, 1.2, 1.2), (-0.0, -0.0, -0.0), (0.0, 0.0, 1.0)]
stream.tube(radius=0.0015).plot(cpos=cpos)
"""
