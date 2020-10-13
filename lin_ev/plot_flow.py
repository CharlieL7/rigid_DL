import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.gca(projection="3d")

E_1 = np.array([
    [0., 1., 0.],
    [1., 0., 0.],
    [0., 0., 0.],
    ]
)

E_2 = np.array([
    [0., 1., 0.],
    [-1., 0., 0.],
    [0., 0., 0.],
    ]
)

E_tot = E_1 + E_2

x, y, z = np.meshgrid(np.arange(-0.8, 1, 0.2),
                      np.arange(-0.8, 1, 0.2),
                      np.arange(-0.8, 1, 0.8))

pos = np.array([x, y, z])

#u = E[0, 0] * x + E[0, 1] * y + E[0, 2] * z
#v = E[1, 0] * x + E[1, 1] * y + E[1, 2] * z
#w = E[2, 0] * x + E[2, 1] * y + E[2, 2] * z

# weird einsum to do same as comment above
vels = np.einsum("ij, jlmn -> ilmn", E_tot, pos)

#ax.quiver(x, y, z, u, v, w, length=0.1)
ax.quiver(x, y, z, vels[0], vels[1], vels[2], length=0.1)

plt.show()
