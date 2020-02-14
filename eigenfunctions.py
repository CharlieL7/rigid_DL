"""
Eigenfunctions for the ellipsoidal particle
"""

import numpy as np
from scipy.linalg import null_space
import eigenvalues as ev

def make_linear_vels(E_d, E_c, mesh, const_or_linear):
    """
    Makes the velocitiy field from the rate of strain field
Parameters:
        E_d : the rate of strain field dotted with position
        E_c : the rate of strain field crossed with position
        mesh : simple mesh input
        const_or_linear : if constant or linear density distributions
    Returns:
        v_list : velocities at each element or node
    """
    if const_or_linear == 'c':
        num_faces = mesh.faces.shape[0]
        v_list = np.zeros((num_faces, 3))
        for m in range(num_faces):
            face = mesh.faces[m]
            center = mesh.calc_tri_center(face)
            v_list[m] = np.dot(E_d, center) - np.cross(E_c, center)
        return v_list
    else:
        num_vert = mesh.vertices.shape[0]
        v_list = np.zeros((num_vert, 3))
        for m in range(num_vert):
            vert = mesh.vertices[m]
            v_list[m] = np.dot(E_d, vert) - np.cross(E_c, vert)
        return v_list


def make_translation_vels(v_cnst, mesh, const_or_linear):
    """
    Makes the rigid body motion velocity field
Parameters:
        v_cnst : the velocity at each point
        mesh : simple mesh input
        const_or_linear : if constant or linear density distributions
    Returns:
        v_list : velocities at each element or node
    """
    if const_or_linear == 'c':
        num_faces = mesh.faces.shape[0]
        v_list = np.zeros((num_faces, 3))
        for m in range(num_faces):
            face = mesh.faces[m]
            v_list[m] = v_cnst
        return v_list
    else:
        num_vert = mesh.vertices.shape[0]
        v_list = np.zeros((num_vert, 3))
        for m in range(num_vert):
            v_list[m] = v_cnst
        return v_list


def E_12(mesh, const_or_linear):
    """
    Off diagonal (12) rate of strain field eigenfunction
    Returns velocites at each vertex in cartesional coordinates

    Parameters:
        mesh : simple mesh input
        const_or_linear : if constant or linear density distributions
    Returns;
        v_list : velocities at each element or node
    """
    E_d = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
    a, b, _c = mesh.dims
    E_c = (a**2 - b**2)/(a**2 + b**2) * E_d[0, 1] * np.array([0, 0, 1])
    return make_linear_vels(E_d, E_c, mesh, const_or_linear)


def E_13(mesh, const_or_linear):
    E_d = np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]])
    a, _b, c = mesh.dims
    E_c = (c**2 - a**2)/(c**2 + a**2) * E_d[0, 2] * np.array([0, 1, 0])
    return make_linear_vels(E_d, E_c, mesh, const_or_linear)


def E_23(mesh, const_or_linear):
    E_d = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]])
    _a, b, c = mesh.dims
    E_c = (b**2 - c**2)/(b**2 + c**2) * E_d[1, 2] * np.array([1, 0, 0])
    return make_linear_vels(E_d, E_c, mesh, const_or_linear)


def uni_x(mesh, const_or_linear):
    E_d = np.array([[2, 0, 0], [0, -1, 0], [0, 0, -1]])
    E_c = np.array([0, 0, 0])
    return make_linear_vels(E_d, E_c, mesh, const_or_linear)


def hyper_yz(mesh, const_or_linear):
    E_d = np.array([[0, 0, 0], [0, 1, 0], [0, 0, -1]])
    E_c = np.array([0, 0, 0])
    return make_linear_vels(E_d, E_c, mesh, const_or_linear)


def uni_z(mesh, const_or_linear):
    E_d = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 2]])
    E_c = np.array([0, 0, 0])
    return make_linear_vels(E_d, E_c, mesh, const_or_linear)


def hyper_xy(mesh, const_or_linear):
    E_d = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]])
    E_c = np.array([0, 0, 0])
    return make_linear_vels(E_d, E_c, mesh, const_or_linear)


def diag_eigvec(pm, mesh, const_or_linear):
    dims = mesh.dims
    kapp = ev.kappa_pm(pm, dims)
    alpha_pp_0 = ev.ellip_pp_cnst(dims, "alpha")
    beta_pp_0 = ev.ellip_pp_cnst(dims, "beta")
    gamma_pp_0 = ev.ellip_pp_cnst(dims, "gamma")
    d = beta_pp_0 * gamma_pp_0 + gamma_pp_0 * alpha_pp_0 + alpha_pp_0 * beta_pp_0
    A = np.array(
        [
            [(kapp - 1) + (4*alpha_pp_0)/(3*d), -(2*beta_pp_0)/(3*d), -(2*gamma_pp_0)/(3 * d)],
            [-(2*alpha_pp_0)/(3*d), (kapp - 1) + (4*beta_pp_0)/(3*d), -(2*gamma_pp_0)/(3*d)],
            [1., 1., 1.]
        ]
    )
    e = null_space(A)
    E_d = np.array([[e[0], 0, 0], [0, e[1], 0], [0, 0, e[2]]])
    E_c = np.array([0, 0, 0])
    return make_linear_vels(E_d, E_c, mesh, const_or_linear)


def calc_ROS_eigs(field, dims):
    """
    Calculates the 5 matricies that when dotted with position are
    the eigenvectors of a rate-of-strain flow field

    Parameters:
        field : the input ROS flow field
        dims : elliptical dimensions, (a, b, c)
    Returns:
        (Ee1, Ee2, Ee3, Ee4, Ee5) : tuple of the five matrix-eigenvalue pairs
            for the eigenvectors where Ee = {"eval":, "emat":}
    """
    E1 = np.array([
        [0, field[0, 1], 0],
        [field[1, 0], 0, 0],
        [0, 0, 0]]
    ) # NOT DONE, NEED ROTATIONAL TERM
    E2 = np.array([
        [0, 0, field[0, 2]],
        [0, 0, 0],
        [field[2, 0], 0, 0]]
    )
    E3 = np.array([
        [0, 0, 0],
        [0, 0, field[1, 2]],
        [0, field[2, 1], 0]]
    )
    #Ee1 = {"eval":lambda_12(dims), "emat":E1}
    #Ee2 = {"eval":lambda_13(dims), "emat":E2}
    #Ee3 = {"eval":lambda_23(dims), "emat":E3}
    #kapp_p = kappa_pm("+", dims)
    #kapp_n = kappa_pm("-", dims)
    #TODO
