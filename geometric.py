import numpy as np

def pos(eta, xi, nodes):
    """
    position in a triangle as a function of eta and xi

    Parameters:
        eta : parametric coordinate, scalar
        xi : paramteric coordinate, scalar
        nodes : three nodes of triangle as columns in 3x3 ndarray
    Returns:
        x : output position (3,) ndarray
    """
    x = (1. - eta - xi) * nodes[0] + eta * nodes[1] + xi * nodes[2]
    return x


def stresslet(x, x_0, n):
    """
    Stress tensor Green's function dotted with the normal vector.
    T_ijk @ n_k
    Parameters:
        x : field point, (3,) ndarray
        x_0 : source point, (3,) ndarray
        n : normal vector, (3,) ndarray
    Returns:
        S_ij : (3,3) ndarray
    """
    x_hat = x - x_0
    r = np.linalg.norm(x_hat)
    S_ij = -6 * np.outer(x_hat, x_hat) * np.dot(x_hat, n) / (r**5)
    return S_ij


def inertia_func(eta, xi, nodes):
    """
    inertia function for input into int_over_tri
    """
    x = pos(eta, xi, nodes)
    return np.dot(x, x) * np.identity(3) - np.outer(x, x)


def const_func(eta, xi, nodes):
    """
    constant function for input into int_over_tri
    """
    return 1
