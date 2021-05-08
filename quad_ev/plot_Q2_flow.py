import numpy as np
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
    [-0.974793639765174],
    [ 0.157761465309759],
    [ 0.157761465309759],
])


x, y, z = np.meshgrid(np.arange(-2., 2.1, 1.0),
                      np.arange(-2., 2.1, 1.0),
                      np.arange(-2., 2.1, 2.0))


u = H[0] * y * z
v = H[1] * x * z
w = H[2] * x * y

ax.quiver(x, y, z, u, v, w, length=0.2, linewidth=1.0)
ax.set_title(r"$Q^{(2)}$")
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$y$")
ax.set_zlabel(r"$z$")
fig.tight_layout(rect=[0, 0, 0.95, 1])
fig.savefig("4-1-1_Q2_flow_quiver.pdf", format="pdf")
