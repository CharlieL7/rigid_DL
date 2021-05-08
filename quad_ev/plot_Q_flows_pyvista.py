import argparse as argp
from configparser import ConfigParser
import numpy as np
import pyvista as pv

pv.rcParams['transparent_background'] = True

def main():

    parser = argp.ArgumentParser(description="Q flows plot in 3D with ellipsoid insert")
    args = parser.parse_args()

    config = ConfigParser()
    config.read('options.cfg')

    H = np.empty(3)
    H[0] = config.getfloat("main", "H_0")
    H[1] = config.getfloat("main", "H_1")
    H[2] = config.getfloat("main", "H_2")

    a = config.getint('main', 'a')
    b = config.getint('main', 'b')
    c = config.getint('main', 'c')
     
    nx = config.getint('main', 'nx')
    ny = config.getint('main', 'ny')
    nz = config.getint('main', 'nz')

    out_name = config.get("main", "out_name")

    spacing = [3, 3, 3]
    origin = (-(nx - 1)*spacing[0]/2, -(ny - 1)*spacing[1]/2, -(nz - 1)*spacing[2]/2)
    mesh = pv.UniformGrid((nx, ny, nz), spacing, origin)
    vectors = np.empty((mesh.n_points, 3))
    x = mesh.points[:, 0]
    y = mesh.points[:, 1]
    z = mesh.points[:, 2]
    vectors[:, 0] = H[0] * y * z
    vectors[:, 1] = H[1] * x * z
    vectors[:, 2] = H[2] * x * y
    mesh['vectors'] = vectors

    streamlines, src = mesh.streamlines(
            'vectors',
            return_source=True,
            n_points=200,
    )

    ellipsoid_mesh = pv.ParametricEllipsoid(a/2, b/2, c/2)

    p = pv.Plotter()
    #p.add_mesh(mesh)
    p.add_mesh(ellipsoid_mesh)
    p.add_mesh(streamlines.tube(radius=0.05))
    p.add_mesh(src)
    p.show_axes()
    p.show(screenshot="{}.png".format(out_name))


if __name__ == "__main__":
    main()
