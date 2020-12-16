"""
Eigenfunctions for the ellipsoidal particle
"""

import numpy as np
from scipy.linalg import null_space
import rigid_DL.eigenvalues as ev
import rigid_DL.geometric as geo
import rigid_DL.elliptic_integrals as e_int

def E_12(dims):
    """
    Off diagonal (12) rate of strain field eigenfunction
    Returns velocites at each vertex in cartesional coordinates

    Parameters:
        dims: ellipsoid dimensions
    Returns
    """
    E_d = np.array([[0., 1., 0.], [1., 0., 0.], [0., 0., 0.]])
    a, b, _c = dims
    E_c = (a**2 - b**2)/(a**2 + b**2) * E_d[0, 1] * np.array([0., 0., 1.])
    return (E_d, E_c)


def E_31(dims):
    E_d = np.array([[0., 0., 1.], [0., 0., 0.], [1., 0., 0.]])
    a, _b, c = dims
    E_c = (c**2 - a**2)/(c**2 + a**2) * E_d[0, 2] * np.array([0., 1., 0.])
    return (E_d, E_c)


def E_23(dims):
    E_d = np.array([[0., 0., 0.], [0., 0., 1.], [0., 1., 0.]])
    _a, b, c = dims
    E_c = (b**2 - c**2)/(b**2 + c**2) * E_d[1, 2] * np.array([1., 0., 0.])
    return (E_d, E_c)


def uni_x():
    E_d = np.array([[2., 0., 0.], [0., -1., 0.], [0., 0., -1.]])
    E_c = np.zeros(3)
    return (E_d, E_c)


def hyper_yz():
    E_d = np.array([[0., 0., 0.], [0., 1., 0.], [0., 0., -1.]])
    E_c = np.zeros(3)
    return (E_d, E_c)


def uni_z():
    E_d = np.array([[-1., 0., 0.], [0., -1., 0.], [0., 0., 2.]])
    E_c = np.zeros(3)
    return (E_d, E_c)


def hyper_xy():
    E_d = np.array([[1., 0., 0.], [0., -1., 0.], [0., 0., 0.]])
    E_c = np.zeros(3)
    return (E_d, E_c)


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


def calc_3x3_evec(dims, kappa):
    """
    Calculates the H^(1) vector associated with the eigenvector of the quadratic flow
    for the 3x3 system.

    Parameters:
        dims: ellipsoidal dimensions (a > b > c)
        kappa: kappa value associated with the eigenvalue to get the eigenvector for
    Returns:
        the H^(1) vector which is double dotted with position to get the disturbance
        velocity
    """
    K_12, K_23, K_13 = e_int.ellip_K_ij(dims)
    K_123 = e_int.ellip_K_123(dims)
    a, b, c = dims
    
    # Matrix generated by SymPy
    B = np.array([
        [2*(K_123*a**3*b*c*(-b**2*kappa + b**2 - c**2*kappa + c**2) - 2)/(a*b*c),
            2*b**2*(-K_123*b**2*kappa + K_123*b**2 - K_123*c**2*kappa + K_123*c**2 + K_13*kappa - K_13),
            2*c**2*(K_12*kappa - K_12 - K_123*b**2*kappa + K_123*b**2 - K_123*c**2*kappa + K_123*c**2)],

        [2*a**2*(-K_123*a**2*kappa + K_123*a**2 - K_123*c**2*kappa + K_123*c**2 + K_23*kappa - K_23),
            2*(K_123*a*b**3*c*(-a**2*kappa + a**2 - c**2*kappa + c**2) - 2)/(a*b*c),
            2*c**2*(K_12*kappa - K_12 - K_123*a**2*kappa + K_123*a**2 - K_123*c**2*kappa + K_123*c**2)],

        [2*a**2*(-K_123*a**2*kappa + K_123*a**2 - K_123*b**2*kappa + K_123*b**2 + K_23*kappa - K_23),
            2*b**2*(-K_123*a**2*kappa + K_123*a**2 - K_123*b**2*kappa + K_123*b**2 + K_13*kappa - K_13),
            2*(K_123*a*b*c**3*(-a**2*kappa + a**2 - b**2*kappa + b**2) - 2)/(a*b*c)]
    ])

    C = np.array([
        [K_123*a**2 + K_23, K_123*b**2 - K_13, -K_12 + K_123*c**2],
        [K_123*a**2 - K_23, K_123*b**2 + K_13, -K_12 + K_123*c**2],
        [K_123*a**2 - K_23, K_123*b**2 - K_13, K_12 + K_123*c**2]
    ])
    C_inv = np.linalg.inv(C)

    D = B @ C_inv
    H = null_space(D)
    return H
