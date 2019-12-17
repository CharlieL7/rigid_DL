"""
Functions to get a eigenvalues of an ellipsoid
"""
import math
from elliptic_integrals import ellip_pp_cnst, ellip_p_cnst

def lambda_12(dims):
    """
    eigenvalue for (12) off-diagonal rate-of-strain field

    parameters:
        dims : ellipsoidal dimensions
    returns:
        float, the eigenvalue
    """
    a, b, c = dims
    return (a**2 + b**2) * ellip_p_cnst(dims, "gamma") - 1


def lambda_23(dims):
    """
    eigenvalue for (23) off-diagonal rate-of-strain field

    parameters:
        dims : ellipsoidal dimensions
    returns:
        float, the eigenvalue
    """
    a, b, c = dims
    return (b**2 + c**2) * ellip_p_cnst(dims, "alpha") - 1


def lambda_31(dims):
    """
    eigenvalue for (31) off-diagonal rate-of-strain field

    parameters:
        dims : ellipsoidal dimensions
    returns:
        float, the eigenvalue
    """
    a, b, c = dims
    return (c**2 + a**2) * ellip_p_cnst(dims, "beta") - 1


def kappa_pm(dims, p_m):
    alpha_0_pp = ellip_pp_cnst(dims, "alpha")
    beta_0_pp = ellip_pp_cnst(dims, "beta")
    gamma_0_pp = ellip_pp_cnst(dims, "gamma")
    d = beta_0_pp * gamma_0_pp + gamma_0_pp * alpha_0_pp + alpha_0_pp * beta_0_pp
    tmp0 = 1 - (2 /(3 * d)) * (alpha_0_pp + beta_0_pp + gamma_0_pp)
    tmp1 = (2 / (3*d)) * math.sqrt(alpha_0_pp**2 + beta_0_pp**2 + gamma_0_pp**2 - d)
    if p_m == "+":
        return tmp0 + tmp1
    else:
        return tmp0 - tmp1


def lambda_p(p_m, dims):
    """
    'plus or minus' eigenvalue for diagonal rate-of-strain field

    parameters:
        dims : ellipsoidal dimensions
    returns:
        float, the eigenvalue
    """
    return (1. + kappa_pm(dims, p_m)) / (1. - kappa_pm(dims, p_m))
