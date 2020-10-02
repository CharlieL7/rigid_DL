"""
Gaussian quadrature for abritrary f(xi, eta, nodes) function
"""

import numpy as np
import rigid_DL.geometric as geo

# 6-point quadrature
a = 0.816847572980459
b = 0.445948490915965
c = 0.108103018168070
d = 0.091576213509771
omega1 = 0.109951743655322
omega2 = 0.223381589678011
PARA_PTS = np.array([
    (d, d),
    (a, d),
    (d, a),
    (b, b),
    (c, b),
    (b, c)
])
W = np.array([omega1, omega1, omega1, omega2, omega2, omega2])


def int_over_tri_lin(func, nodes, hs):
    """
    Integrate a function over a flat triangle surface using Gaussian quadrature.
    Seven point quadrature.

    Parameters:
        func: function to integrate, can return any order tensor
               expecting f(xi, eta, nodes)
        nodes: 3x3 ndarray with nodes as column vectors
        hs: triangle area
    Returns:
        integrated function
    """
    f = []
    for xi, eta in PARA_PTS:
        f.append(func(xi, eta, nodes))
    f = np.transpose(np.array(f))
    ret = 0.5 * hs * np.dot(f, W)
    return ret


def int_over_tri_quad(func, nodes, hs):
    """
    Integrate a function over a curved triangle surface using Gaussian quadrature.
    Seven point quadrature.

    Parameters:
        func: function to integrate, can return any order tensor
               expecting f(eta, xi, nodes)
        nodes: 3x6 ndarray with nodes as column vectors
        hs: areas at the seven quadrature points (7,) ndarray
    Returns:
        integrated function
    """
    f = []
    for i, (xi, eta) in enumerate(PARA_PTS):
        f.append(func(xi, eta, nodes) * hs[i])
    f = np.transpose(np.array(f))
    ret = 0.5 * np.dot(f, W)
    return ret


def int_over_tri_quad_n(func, nodes, n):
    """
    Version that integrates function dotted with an array of normal vectors.
    This is used to minimize number of times normal vector calculation over
    an element is called.

    Parameters:
        func: function to integrate, must return (3,3,3) ndarray
               expecting f(eta, xi, nodes)
        nodes: 3x6 ndarray with nodes as column vectors
        n: normal vectors at the seven quadrature points (3, 6) ndarray
            (not normalized vectors, hs magnitude)
    Returns:
        integrated (function . n)
    """
    f = np.empty([3, 3, 6])
    for i, (xi, eta) in enumerate(PARA_PTS):
        f[:, :, i] = np.dot(func(xi, eta, nodes), n[:, i])
    ret = 0.5 * np.dot(f, W)
    return ret


def quad_n(nodes, xc, tri_c):
    """
    Calculate the normal vector values over a curved triangle for severn point Gaussian quadrature.
    This function is use to reduce the number of times these values are recalculated.

    Parameters:
        nodes: 3x6 ndarray with nodes as column vectors
    Returns:
        normals: (3, 6) ndarray
    """
    normals = np.empty([3, 6])
    for i, (xi, eta) in enumerate(PARA_PTS):
        e_xi = np.einsum("ij,j->i", nodes, geo.dphi_dxi_quadratic(xi, eta, nodes))
        e_eta = np.einsum("ij,j->i", nodes, geo.dphi_deta_quadratic(xi, eta, nodes))
        n = np.cross(e_xi, e_eta)
        if np.dot(n, tri_c - xc) < 0.:
            print("n reori")
            n = -n
        normals[:, i] = n
    return normals


# ///////////////
# DEBUG STUFF
# ///////////////


def int_over_tri_quad_slow(func, nodes, xc, tri_c):
    """
    Integrate a function over a curved triangle surface using Gaussian quadrature.
    Seven point quadrature.
    SLOW VERSION.

    Parameters:
        func : function to integrate, can return any order tensor
               expecting f(eta, xi, n, nodes)
        nodes : 3x6 ndarray with nodes as column vectors
        xc: mesh centroid
        tri_c: triangle center
    Returns:
        integrated function
    """
    """
    # Gaussian quadrature weights
    w = 1./60. * np.array([3., 8., 3., 8., 3., 8., 27.])

    # Gaussian quadrature points
    para_pts = [
        (0.0, 0.0),
        (0.5, 0.0),
        (1.0, 0.0),
        (0.5, 0.5),
        (0.0, 1.0),
        (0.0, 0.5),
        (1./3., 1./3.),
    ]

    """
    # 6-point quadrature
    a = 0.816847572980459
    b = 0.445948490915965
    c = 0.108103018168070
    d = 0.091576213509771
    omega1 = 0.109951743655322
    omega2 = 0.223381589678011
    para_pts = np.array([
        (d, d),
        (a, d),
        (d, a),
        (b, b),
        (c, b),
        (b, c)
    ])
    w = np.array([omega1, omega1, omega1, omega2, omega2, omega2])

    f = []
    for (xi, eta) in para_pts:
        e_xi = np.matmul(nodes, geo.dphi_dxi_quadratic(xi, eta, nodes))
        e_eta = np.matmul(nodes, geo.dphi_deta_quadratic(xi, eta, nodes))
        n = np.cross(e_xi, e_eta)
        if np.dot(n, tri_c - xc) < 0.:
            print("reoriented normal")
            n = -n
        h_s = np.linalg.norm(n)
        n_unit = n / h_s
        f.append(func(xi, eta, n_unit, nodes) * h_s)
    f = np.array(f)
    f = np.transpose(np.array(f)) # make (3,6)

    ret = 0.5 * np.dot(f, w)
    return ret
