"""
Seven point Gaussian quadrature for abritrary f(xi, eta, nodes) function
"""

import numpy as np
import geometric as geo

# Gaussian quadrature weights
W = 1./60. * np.array([3., 8., 3., 8., 3., 8., 27.])
# Gaussian quadrature points
PARA_PTS = [
    (0.0, 0.0),
    (0.5, 0.0),
    (1.0, 0.0),
    (0.5, 0.5),
    (0.0, 1.0),
    (0.0, 0.5),
    (1./3., 1./3.),
]

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


def int_over_tri_quad_n(func, nodes, hs, n):
    """
    Version that integrates function dotted with an array of normal vectors.
    This is used to minimize number of times normal vector calculation over
    an element is called.

    Parameters:
        func: function to integrate, must return (3,3,3) ndarray
               expecting f(eta, xi, nodes)
        nodes: 3x6 ndarray with nodes as column vectors
        hs: areas at the seven quadrature points (7,) ndarray
        n: normal vectors at the seven quadrature points (3, 7) ndarray
    Returns:
        integrated (function . n)
    """
    f = np.empty([3, 3, 7])
    for i, (xi, eta) in enumerate(PARA_PTS):
        f[:, :, i] = np.einsum("ijk,k->ij", func(xi, eta, nodes), n[:, i]) * hs[i]
    ret = 0.5 * np.dot(f, W)
    return ret


def quad_n(nodes):
    """
    Calculate the normal vector values over a curved triangle for severn point Gaussian quadrature.
    This function is use to reduce the number of times these values are recalculated.

    Parameters:
        nodes: 3x6 ndarray with nodes as column vectors
    Returns:
        normals: (3, 7) ndarray
    """
    normals = np.empty([3, 7])
    for i, (xi, eta) in enumerate(PARA_PTS):
        e_xi = np.matmul(nodes, geo.dphi_dxi_quadratic(xi, eta, nodes))
        e_eta = np.matmul(nodes, geo.dphi_deta_quadratic(xi, eta, nodes))
        normals[:, i] = np.cross(e_xi, e_eta)
    return normals
