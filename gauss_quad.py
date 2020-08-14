import math
import numpy as np
import geometric as geo

def int_over_tri_linear(func, nodes):
    """
    Integrate a function over a flat triangle surface using Gaussian quadrature.
    Seven point quadrature.

    Parameters:
        func : function to integrate, can return any order tensor
               expecting f(xi, eta, nodes)
        nodes : 3x3 ndarray with nodes as column vectors
    Returns:
        integrated function
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
    f = []
    for xi, eta in para_pts:
        f.append(func(xi, eta, nodes))
    f = np.array(f)

    # metric function for surface area
    n = np.cross(nodes[1] - nodes[0], nodes[2] - nodes[0])
    h_s = np.linalg.norm(n)

    ret = 0.5 * h_s * np.dot(f, w)
    return ret


def int_over_tri_quadratic(func, nodes):
    """
    Integrate a function over a curved triangle surface using Gaussian quadrature.
    Seven point quadrature.

    Parameters:
        func : function to integrate, can return any order tensor
               expecting f(eta, xi, nodes)
        nodes : 3x6 ndarray with nodes as column vectors
    Returns:
        integrated function
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
    f = []
    for xi, eta in para_pts:
        e_xi = np.matmul(nodes, geo.dphi_dxi_quadratic(xi, eta, nodes))
        e_eta = np.matmul(nodes, geo.dphi_deta_quadratic(xi, eta, nodes))
        h_s = np.linalg.norm(np.cross(e_xi, e_eta))
        f.append(func(xi, eta, nodes) * h_s)
    f = np.array(f)

    ret = 0.5 * h_s * np.dot(f, w)
    return ret
