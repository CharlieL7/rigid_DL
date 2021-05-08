import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt


E_1 = np.array([
    [0., 1., 0.],
    [1., 0., 0.],
    [0., 0., 0.],
    ]
)

nx = 4
ny = 4
nz = 2

origin = (-(nx - 1)*0.1/2, -(ny - 1)*0.1/2, -(nz - 1)*0.1/2)
mesh = pv.UniformGrid((nx, ny, nz), (.1, .1, .1), origin)
x = mesh.points[:, 0]
y = mesh.points[:, 1]
z = mesh.points[:, 2]
vectors = np.empty((mesh.n_points, 3))
vectors[:, 0] = y
vectors[:, 1] = x
vectors[:, 2] = 0

mesh['vectors'] = vectors

stream, src = mesh.streamlines('vectors', return_source=True,
                               terminal_speed=0.0, n_points=200,
                               source_radius=0.1)
cpos = [(1.2, 1.2, 1.2), (-0.0, -0.0, -0.0), (0.0, 0.0, 1.0)]
stream.tube(radius=0.0015).plot(cpos=cpos)
