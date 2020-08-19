import sys
import math
import numpy as np

def pos_linear(xi, eta, nodes):
    """
    position in a flat, 3 node triangle as a function of eta and xi

    Parameters:
        eta : parametric coordinate, scalar
        xi : paramteric coordinate, scalar
        nodes : three nodes of triangle as columns in 3x3 ndarray
    Returns:
        x : output position (3,) ndarray
    """
    x = (1. - xi - eta) * nodes[0] + xi * nodes[1] + eta * nodes[2]
    return x


def pos_quadratic(xi, eta, nodes):
    """
    position in a curved 6 node triangle as a function of eta and xi

    Parameters:
        eta : parametric coordinate, scalar
        xi : paramteric coordinate, scalar
        nodes : six nodes of triangle as columns in 3x6 ndarray
    Returns:
        x : output position (3,) ndarray
    """
    alpha, beta, gamma = calc_abg(nodes)
    phi = np.array([
        0.,
        (1. / (1. - alpha)) * xi * (xi - alpha + ((alpha - gamma) / (1. - gamma)) * eta),
        (1. / (1. - beta)) * eta * (eta - beta + ((beta + gamma - 1.) / (gamma)) * xi),
        1. / (alpha * (1. - alpha)) * xi * (1. - xi - eta),
        1. / (gamma * (1. - gamma)) * xi * eta,
        1. / (beta * (1. - beta)) * eta * (1. - xi - eta),
        ])
    phi[0] = 1. - np.sum(phi)
    x = np.matmul(nodes, phi)
    return x


def dphi_dxi_quadratic(xi, eta, nodes):
    """
    Calculates the dphi/dxi vector for a 2nd order curved triangular element.

    Parameters:
        eta : parametric coordinate, scalar
        xi : paramteric coordinate, scalar
        nodes : six nodes of triangle as columns in 3x6 ndarray
    Returns:
        dphi_dxi : dphi/dxi vector as ndarray shape (6)
    """
    alpha, beta, gamma = calc_abg(nodes)
    dphi_dxi = np.array([
        0.,
        (1. / (1. - alpha)) * (2. * xi - alpha + (alpha - gamma) / (1. - gamma) * eta),
        (1. / (1. - beta)) * (beta + gamma - 1.) / (gamma) * eta,
        (1. / (alpha * (1. - alpha)) * (1. - 2. * xi - eta)),
        eta / (gamma * (1. - gamma)),
        -eta / (beta * (1. - beta)),
    ])
    dphi_dxi[0] = 1. - np.sum(dphi_dxi)
    return dphi_dxi


def dphi_deta_quadratic(xi, eta, nodes):
    """
    Calculates the dphi/deta vector for a 2nd order curved triangular element.

    Parameters:
        eta : parametric coordinate, scalar
        xi : paramteric coordinate, scalar
        nodes : six nodes of triangle as columns in 3x6 ndarray
    Returns:
        dphi_deta : dphi/deta vector as ndarray shape (6)
    """
    alpha, beta, gamma = calc_abg(nodes)
    dphi_deta = np.array([
        0.,
        (1. / (1. - alpha)) * (alpha - gamma) / (1. - gamma) * xi,
        (1. / (1. - beta)) * (2. * eta - beta + (beta + gamma - 1.) / (gamma) * xi),
        -xi / (alpha * (1. - alpha)),
        xi / (gamma * (1. - gamma)),
        1. / (beta * (1. - beta)) * (1. - xi - 2. * eta),
    ])
    dphi_deta[0] = 1. - np.sum(dphi_deta)
    return dphi_deta


def calc_abg(nodes):
    """
    Calculate the alpha, beta, and gamma geometrical parameters for the parameterization
    of a six-node curved triangle

    Parameters:
        nodes : six nodes of triangle as columns in 3x6 ndarray
    Returns:
        (alpha, beta, gamma) : tuple of floats
    """
    alpha = 1. / (1. + np.linalg.norm(nodes[:, 3] - nodes[:, 1]) /
                  np.linalg.norm(nodes[:, 3] - nodes[:, 0]))
    beta = 1. / (1. + np.linalg.norm(nodes[:, 5] - nodes[:, 2]) /
                 np.linalg.norm(nodes[:, 5] - nodes[:, 0]))
    gamma = 1. / (1. + np.linalg.norm(nodes[:, 4] - nodes[:, 1]) /
                  np.linalg.norm(nodes[:, 4] - nodes[:, 2]))
    return (alpha, beta, gamma)


def stresslet(x, x_0, n):
    """
    Green's function associated with stress tensor of point force or flow from a point source.
    Dotted with the normal vector.
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
    S_ij = -6. * np.outer(x_hat, x_hat) * np.dot(x_hat, n) / (r**5.)
    return S_ij


def inertia_func_linear(xi, eta, nodes):
    """
    inertia function for input into int_over_tri_linear()
    """
    x = pos_linear(xi, eta, nodes)
    return np.dot(x, x) * np.identity(3) - np.outer(x, x)


def inertia_func_quadratic(xi, eta, nodes):
    """
    inertia function for input into int_over_tri_quadratic()
    """
    x = pos_quadratic(xi, eta, nodes)
    return np.dot(x, x) * np.identity(3) - np.outer(x, x)


def const_func(xi, eta, nodes):
    """
    constant function for input into int_over_tri
    """
    return 1.


def shape_func_linear(xi, eta, num):
    """
    shape functions for linear elements
    """
    if num == 0:
        return 1 - xi - eta
    elif num == 1:
        return xi
    elif num == 2:
        return eta
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