"""
Functions to get a eigenvalues of an ellipsoid
"""
import math
from rigid_DL.elliptic_integrals import ellip_pp_cnst, ellip_p_cnst

def lambda_12(dims):
    """
    eigenvalue for (12) off-diagonal rate-of-strain field

    parameters:
        dims : ellipsoidal dimensions
    returns:
        float, the eigenvalue
    """
    a, b, c = dims
    _, _, gamma_p = ellip_p_cnst(dims)
    return (a**2 + b**2) * gamma_p - 1


def lambda_23(dims):
    """
    eigenvalue for (23) off-diagonal rate-of-strain field

    parameters:
        dims : ellipsoidal dimensions
    returns:
        float, the eigenvalue
    """
    a, b, c = dims
    alpha_p, _, _ = ellip_p_cnst(dims)
    return (b**2 + c**2) * alpha_p - 1


def lambda_31(dims):
    """
    eigenvalue for (31) off-diagonal rate-of-strain field

    parameters:
        dims : ellipsoidal dimensions
    returns:
        float, the eigenvalue
    """
    a, b, c = dims
    _, beta_p, _ = ellip_p_cnst(dims)
    return (c**2 + a**2) * beta_p - 1


def kappa_pm(p_m, dims):
    alpha_pp_0, beta_pp_0, gamma_pp_0 = ellip_pp_cnst(dims)
    d = beta_pp_0 * gamma_pp_0 + gamma_pp_0 * alpha_pp_0 + alpha_pp_0 * beta_pp_0
    tmp0 = 1 - (2 /(3 * d)) * (alpha_pp_0 + beta_pp_0 + gamma_pp_0)
    tmp1 = (2 / (3*d)) * math.sqrt(alpha_pp_0**2 + beta_pp_0**2 + gamma_pp_0**2 - d)
    if p_m == "+":
        return tmp0 + tmp1
    else:
        return tmp0 - tmp1


def lambda_pm(p_m, dims):
    """
    'plus or minus' eigenvalue for diagonal rate-of-strain field

    parameters:
        dims : ellipsoidal dimensions
    returns:
        float, the eigenvalue
    """
    return (1. + kappa_pm(p_m, dims)) / (1. - kappa_pm(p_m, dims))


def ABC_const(E, dims):
    """
    Diagonal ROS field constants defined by Jeffery

    Parameters:
        E: the diagonal ROS field
        dims: ellipsoidal dimensions
    Returns:
        (A, B, C)
    """
    app_0, bpp_0, gpp_0 = ellip_pp_cnst(dims)
    (E11, E22, E33) = E.diagonal()
    A = (
        1/6. * (2. * app_0 * E11 - bpp_0 * E22 - gpp_0 * E33) /
        (bpp_0 * gpp_0 + gpp_0 * app_0 + app_0 * bpp_0)
    )
    B = (
        1/6. * (2. * bpp_0 * E22 - gpp_0 * E33 - app_0 * E11) /
        (bpp_0 * gpp_0 + gpp_0 * app_0 + app_0 * bpp_0)
    )
    C = (
        1/6. * (2. * gpp_0 * E33 - app_0 * E11 - bpp_0 * E22) /
        (bpp_0 * gpp_0 + gpp_0 * app_0 + app_0 * bpp_0)
    )
    return (A, B, C)
