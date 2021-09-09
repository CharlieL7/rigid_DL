import argparse as argp
import eigenvalues as ev
import elliptic_integrals as e_int
from scipy.integrate import quad
from scipy.linalg import null_space
import numpy as np
import pyvista as pv

pv.rcParams['transparent_background'] = True

def diag_eigvec(pm, dims):
    """
    Calculates the eigenfunctions associated with the diagonal terms
    of the linear rate of strain field.

    Parameters:
        pm: the plus or minus eigenvalue
        dims: ellipsoid dimensions
    Returns:
        E_d and E_c matricies for dotting with position
    """
    if np.allclose(dims, [1., 1., 1.,]):
        print("Sphere detected, diagonal linear ROS field set to arbitrary flow")
        if pm == "+":
            return uni_x()
        return uni_z()
    kapp = ev.kappa_pm(pm, dims)
    app_0, bpp_0, gpp_0 = ev.ellip_pp_cnst(dims)
    d = bpp_0 * gpp_0 + gpp_0 * app_0 + app_0 * bpp_0
    M = np.array(
        [
            [(kapp - 1) + (4*app_0)/(3*d), -(2*bpp_0)/(3*d), -(2*gpp_0)/(3*d)],
            [-(2*app_0)/(3*d), (kapp - 1) + (4*bpp_0)/(3*d), -(2*gpp_0)/(3*d)],
            [1., 1., 1.]
        ]
    )
    if np.linalg.matrix_rank(M) != 2:
        print(M)
        raise RuntimeError("Diagonal eigenfunction failed; full rank matrix")
    e = null_space(M).reshape((3,))
    E = e * np.identity(3)
    A, B, C = ev.ABC_const(E, dims)
    D = 4. * np.identity(3) * np.array([A, B, C])
    E_d = D - E
    tmp_max = np.amax(np.abs(E_d))
    E_d = np.divide(E_d, tmp_max) # normalize the matrix to largest value = 1
    print(E_d)
    E_c = np.array([0, 0, 0])
    return (E_d, E_c)

def uni_x():
    E_d = np.array([[2., 0., 0.], [0., -1., 0.], [0., 0., -1.]])
    E_c = np.zeros(3)
    return (E_d, E_c)

def uni_z():
    E_d = np.array([[-1., 0., 0.], [0., -1., 0.], [0., 0., 2.]])
    E_c = np.zeros(3)
    return (E_d, E_c)

def main():

    parser = argp.ArgumentParser(description="Ep or Em flow plot in 3D with ellipsoid insert")
    parser.add_argument("type", type=str, help="+ for Ep flow, - for Em flow")
    parser.add_argument("dims", nargs=3, type=float, help="ellipsoid dimensions")
    parser.add_argument(
        "-o",
        "--out_tag",
        help="tag to prepend to output plot",
        default="Epm_flow_3D_quiver"
    )
    args = parser.parse_args()

    E_d, _E_c = diag_eigvec(args.type, args.dims)

    nx = 4
    ny = 4
    nz = 4

    spacing = [3, 3, 3]
    origin = (-(nx - 1)*spacing[0]/2, -(ny - 1)*spacing[1]/2, -(nz - 1)*spacing[2]/2)
    mesh = pv.UniformGrid((nx, ny, nz), spacing, origin)
    vectors = np.empty((mesh.n_points, 3))
    vectors = E_d @ np.transpose(mesh.points)
    vectors = np.transpose(vectors)

    mesh['vectors'] = vectors

    streamlines, src = mesh.streamlines(
            'vectors',
            return_source=True,
            n_points=200,
    )
    ellipsoid_mesh = pv.ParametricEllipsoid(args.dims[0]/2, args.dims[1]/2, args.dims[2]/2)

    p = pv.Plotter()
    #p.add_mesh(mesh)
    p.add_mesh(ellipsoid_mesh)
    p.add_mesh(streamlines.tube(radius=0.05))
    p.add_mesh(src)
    p.show_axes()
    p.show(screenshot="{}.png".format(args.out_tag))


if __name__ == "__main__":
    main()
