import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt

fig = plt.figure(figsize=plt.figaspect(1))
ax = fig.add_subplot(111, projection='3d')

coefs = (0.25, 1, 1)  # Coefficients in a0/c x**2 + a1/c y**2 + a2/c z**2 = 1 
# Radii corresponding to the coefficients:
rx, ry, rz = 1/np.sqrt(coefs)

# Set of all spherical angles: u = np.linspace(0, 2 * np.pi, 100)
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)

# Cartesian coordinates that correspond to the spherical angles:
# (this is the equation of an ellipsoid):
x = rx * np.outer(np.cos(u), np.sin(v))
y = ry * np.outer(np.sin(u), np.sin(v))
z = rz * np.outer(np.ones_like(u), np.cos(v))

# Plot:
ax.plot_surface(x, y, z,  rstride=4, cstride=4, color='b')

# Adjustment of the axes, so that they all have the same span:
max_radius = max(rx, ry, rz)
for axis in 'xyz':
    getattr(ax, 'set_{}lim'.format(axis))((-max_radius, max_radius))


H = np.array([
    [-0.964679416731502],
     [-0.186270801435128],
     [-0.186270801435128]
])


x, y, z = np.meshgrid(np.arange(-0.80, 1.0, 0.2),
                      np.arange(-0.80, 1.0, 0.2),
                      np.arange(-0.80, 1.0, 0.2))


u = H[0] * y * z
v = H[1] * x * z
w = H[2] * x * y

ax.quiver(x, y, z, u, v, w, length=0.2)
ax.set_title(r"$Q^{(3)}$")
fig.tight_layout(rect=[0, 0, 0.95, 1])
fig.savefig("4-1-1_Q3_flow_quiver.pdf", format="pdf")
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
