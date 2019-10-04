import math
import numpy as np

def int_over_tri(func, nodes):
    """
    Integrate a function over a triangle surface using Gaussian quadrature.
    Seven point quadrature. I'm not sure why, used in old code.

    Parameters:
        func : function to integrate, can return any order tensor
               expecting f(eta, xi, nodes)
    Returns:
        integrated function
    """

    # Gaussian quadrature weights
    w = 1./60. * [3., 8., 3., 8., 3., 8., 27.]

    # Gaussian quadrature points
    f1 = func(0.0, 0.0, nodes)
    f2 = func(0.5, 0.0, nodes)
    f3 = func(1.0, 0.0, nodes)
    f4 = func(0.5, 0.5, nodes)
    f5 = func(0.0, 1.0, nodes)
    f6 = func(0.0, 0.5, nodes)
    f7 = func(1.0/3.0, 1.0/3.0, nodes)
    ret = np.zeros(f1.shape[0])

    # metric function for surface area
    dx_deta = nodes[1] - nodes[0]
    dx_dxi = nodes[2] - nodes[0]
    h = math.sqrt((dx_deta * dx_deta) *(dx_dxi * dx_dxi) - (dx_deta * dx_dxi) ** 2)

    ret = f1 * w[1] + f2 * w[2] + f3 * w[3] + f4 * w[4] + f5 * w[5] + f6 * w[6] + f7 * w[7]
    ret *= 0.5 * h
    return ret
