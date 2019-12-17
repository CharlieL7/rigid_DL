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
    cnst = dims[0] * dims[1] * dims[2]
    return cnst * quad(ellip_integrand, 0, np.inf, args=(dims, i))[0]


def ellip_integrand(t, dims, start_ind):
    """
    Integrand for the ellipstical integral
    """
    delta_t = math.sqrt((dims[0]**2 * t) * (dims[1]**2 + t) * (dims[2]**2 + t))
    return 1./((dims[start_ind]**2 + t) * delta_t)


def ellip_p_cnst(dims, version):
    """
    Primed elliptical integral at the surface
    """
    assert len(dims) == 3
    i = get_start_ind(version)
    types = ["alpha", "beta", "gamma"]
    tmp0 = ellip_cnst(dims, types[(i + 2) % 3]) - ellip_cnst(dims, types[(i + 1) % 3])
    tmp1 = dims[(i + 1) % 3]**2 - dims[(i + 2) % 3]**2
    return tmp0 / tmp1


def ellip_pp_cnst(dims, version):
    assert len(dims) == 3
    i = get_start_ind(version)
    types = ["alpha", "beta", "gamma"]
    tmp0 = (
        dims[(i+1) % 3]**2 * ellip_cnst(dims, types[(i+1) % 3]) -
        dims[(i+2) % 3]**2 * ellip_cnst(dims, types[(i+2) % 3])
    )
    tmp1 = dims[(i+1) % 3]**2 - dims[(i+2) % 3]**2
    return tmp0 / tmp1


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
