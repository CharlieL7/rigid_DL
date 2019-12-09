import numpy as np
import sys
import math

def pos(eta, xi, nodes):
    """
    position in a flat, linear triangle as a function of eta and xi

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


def shape_func_linear(eta, xi, num):
    """
    shape functions for linear elements
    """
    if num == 0:
        return 1 - eta - xi
    elif num == 1:
        return eta
    elif num == 2:
        return xi
    else:
        sys.exit("failure on shape_func(), unexpected num")


def cart2sph(vec):
    """
    cartesional coordinates to spherical coordinates
    """
    (x, y, z) = vec
    xy = x**2 + y**2
    r = math.sqrt(xy + z**2)
    theta = math.acos(z/r)
    phi = math.atan2(y, x)
    return np.array([r, theta, phi])


def sph2cart(vec):
    """
    spherical coordinates to cartesional coordinates
    """
    (r, theta, phi) = vec
    x = r * math.sin(theta) * math.cos(phi)
    y = r * math.sin(theta) * math.sin(phi)
    z = r * math.cos(theta)
    return np.array([x, y, z])


def v_sph2cart(x, v):
    """
    spherical velocity to cartesional velocity
    """
    (r, theta, phi) = x
    (rd, td, pd) = v
    dxdt = (
        rd * math.sin(theta) * math.cos(phi) + r * math.cos(theta) *
        td * math.cos(phi) - r * math.sin(theta) * math.sin(phi) * pd
    )
    dydt = (
        rd * math.sin(theta) * math.sin(phi) + r * math.cos(theta) *
        td * math.sin(phi) + r * math.sin(theta) * math.cos(phi) * pd
    )
    dzdt = rd * math.cos(theta) - r * math.sin(theta) * td
    return np.array([dxdt, dydt, dzdt])
