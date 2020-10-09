"""
Elliptical integrals that are used as constants for the eigenvalue functions
"""
import math
import sys
import numpy as np
from scipy.integrate import quad

def ellip_cnst(dims):
    """
    Base elliptical integral at the surface

    Parameters:
        dims : elliptical dimensions (a, b, c)
    Returns:
        float, the constant
    """
    assert len(dims) == 3
    abc = dims[0] * dims[1] * dims[2]
    cons = np.empty(3)
    for i in range(3):
        cons[i] = abc * quad(ellip_integrand, 0., np.inf, args=(dims, i))[0]
    return cons


def ellip_integrand(t, dims, start_ind):
    """
    Integrand for the elliptical integral
    """
    delta_t = np.sqrt((dims[0]**2 + t) * (dims[1]**2 + t) * (dims[2]**2 + t))
    return ((dims[start_ind]**2 + t) * delta_t)**(-1)


def ellip_p_cnst(dims):
    """
    Primed elliptical integral at the surface

    Parameters:
        dims: ellipsoid dimenisons
    Returns:
        [alpha, beta, gamma]: the constants as ndarray
    """
    assert len(dims) == 3
    abc = dims[0] * dims[1] * dims[2]
    cons = np.empty(3)
    for i in range(3):
        cons[i] = abc * quad(ellip_p_integrand, 0, np.inf, args=(dims, i))[0]
    return cons


def ellip_p_integrand(t, dims, i):
    """
    Integrand for the primed elliptical integral

    Parameters:
        t : dependent variable
        dims : ellipsoidal dimensions
        i : starting index for dimension (0 = a, 1 = b, 2 = c)
    """
    delta_t = np.sqrt((dims[0]**2 + t) * (dims[1]**2 + t) * (dims[2]**2 + t))
    tmp = (
        (dims[(i+2) % 3]**2 + t) *
        (dims[(i+1) % 3]**2 + t) *
        delta_t
    )**(-1)
    return tmp


def ellip_pp_cnst(dims):
    """
    Constants based on the ellipsoid dimensions, alpha, beta, gamma

    Parameters:
        dims: ellipsoid dimensions
    Returns:
        [alpha, beta, gamma]: the constants as ndarray
    """
    assert len(dims) == 3
    abc = dims[0] * dims[1] * dims[2]
    cons = np.empty(3)
    for i in range(3):
        q = quad(ellip_pp_integrand, 0, np.inf, args=(dims, i))
        cons[i] = (abc * q[0])
    return cons


def ellip_pp_integrand(t, dims, i):
    """
    Integrand for the double prime elliptical integral

    Parameters:
        t : dependent variable
        dims : ellipsoidal dimensions
        i : starting index for dimension (0 = a, 1 = b, 2 = c)
    """
    delta_t = np.sqrt((dims[0]**2 + t) * (dims[1]**2 + t) * (dims[2]**2 + t))
    tmp = t / (
        (dims[(i+1) % 3]**2 + t) *
        (dims[(i+2) % 3]**2 + t) *
        delta_t
    )
    return tmp

# Reformatted stuff for quadratic eigenfunctions

def ellip_K_ij(dims):
    """
    Alternative version of primed elliptical integral at the surface
    following notation in Martin 2019

    Parameters:
        dims: ellipsoid dimenisons
    Returns:
        [K_12, K_23, K_13]: the constants as ndarray
    """
    assert len(dims) == 3
    cons = np.empty(3)
    for i in range(3):
        cons[i] = quad(K_ij_inte, 0, np.inf, args=(dims, i))[0]
    return cons


def K_ij_inte(t, dims, i):
    """
    Integrand for the K_ij elliptical integral

    Parameters:
        t : dependent variable
        dims : ellipsoidal dimensions
        i : starting index for dimension (0 = K_12, 1 = K_23, 2 = K_13)
    """
    delta_t = np.sqrt((dims[0]**2 + t) * (dims[1]**2 + t) * (dims[2]**2 + t))
    tmp = (
        (dims[(i) % 3]**2 + t) *
        (dims[(i+1) % 3]**2 + t) *
        delta_t
    )**(-1)
    return tmp


def ellip_K_123(dims):
    """
    Other version seen in Martin 2019
    """
    assert len(dims) == 3
    cons = quad(K_123_inte, 0, np.inf, args=(dims))[0]
    return cons


def K_123_inte(t, dims):
    delta_t = np.sqrt((dims[0]**2 + t) * (dims[1]**2 + t) * (dims[2]**2 + t))
    tmp = (
        (dims[0]**2 + t) *
        (dims[1]**2 + t) *
        (dims[2]**2 + t) *
        delta_t
    )**(-1)
    return tmp
