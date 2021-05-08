import eigenvalues as ev
import elliptic_integrals as e_int
from scipy.integrate import quad
from scipy.linalg import null_space
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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

    fig = plt.figure(figsize=plt.figaspect(1))
    ax = fig.add_subplot(111, projection='3d')

    coefs = (0.25, 1, 1)  # Coefficients in a0/c x**2 + a1/c y**2 + a2/c z**2 = 1 
    # Radii corresponding to the coefficients:
    rx, ry, rz = 1/np.sqrt(coefs)

    # Set of all spherical angles:
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

    dims = (4, 1, 1)
    E_d, E_c = diag_eigvec("+", dims)
    ax = fig.gca(projection="3d")
    x, y, z = np.meshgrid(np.arange(-2, 2.1, 1.00),
                          np.arange(-2, 2.1, 1.00),
                          np.arange(-2, 2.1, 1.0))

    pos = np.array([x, y, z])
    vels = np.einsum("ij, jlmn -> ilmn", E_d, pos)
    ax.quiver(x, y, z, vels[0], vels[1], vels[2], length=0.2, linewidth=0.70)
    ax.set_title(r"$E^{(+)}$")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_zlabel(r"$z$")
    #ax.view_init(elev=0., azim=0)
    fig.tight_layout(rect=[0, 0, 0.95, 1])
    fig.savefig("Ep_flow_quiver.pdf", format="pdf")


if __name__ == "__main__":
    main()
