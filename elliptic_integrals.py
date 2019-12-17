"""
Elliptical integrals that are used as constants for the eigenvalue functions
"""
import math
import sys
import numpy as np
from scipy.integrate import quad

def ellip_cnst(dims, version):
    """
    Base elliptical integral at the surface

    Parameters:
        dims : elliptical dimensions (a, b, c)
        version : which constant, expecting "alpha", "beta", or "gamma"
    Returns:
        float, the constant
    """
    assert len(dims) == 3
    i = get_start_ind(version)
    abc = dims[0] * dims[1] * dims[2]
    return abc * quad(ellip_integrand, 0., np.inf, args=(dims, i))[0]


def ellip_integrand(t, dims, start_ind):
    """
    Integrand for the elliptical integral
    """
    delta_t = np.sqrt((dims[0]**2 + t) * (dims[1]**2 + t) * (dims[2]**2 + t))
    return ((dims[start_ind]**2 + t) * delta_t)**(-1)


def ellip_p_cnst(dims, version):
    """
    Primed elliptical integral at the surface
    """
    assert len(dims) == 3
    i = get_start_ind(version)
    abc = dims[0] * dims[1] * dims[2]
    return abc * quad(ellip_p_integrand, 0, np.inf, args=(dims, i))[0]


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


def ellip_pp_cnst(dims, version):
    assert len(dims) == 3
    i = get_start_ind(version)
    abc = dims[0] * dims[1] * dims[2]
    return abc * quad(ellip_pp_integrand, 0, np.inf, args=(dims, i))[0]


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


def get_start_ind(version):
    if version == "alpha":
        start_ind = 0
    elif version == "beta":
        start_ind = 1
    elif version == "gamma":
        start_ind = 2
    else:
        sys.exit("Unrecognized version input in get_start_ind()")
    return start_ind
