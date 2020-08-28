"""
Eigenfunctions for the ellipsoidal particle
"""

import numpy as np
from scipy.linalg import null_space
import eigenvalues as ev

def make_cp_le_lin_vels(E_d, E_c, mesh):
    """
    Makes the velocitiy field from the rate of strain field
    constant potential, linear elements
Parameters:
        E_d : the rate of strain field dotted with position
        E_c : the rate of strain field crossed with position
        mesh : simple linear mesh input
    Returns:
        v_list : velocities at each element or node
    """
    num_faces = mesh.faces.shape[0]
    v_list = np.zeros((num_faces, 3))
    for m in range(num_faces):
        face = mesh.faces[m]
        nodes = mesh.get_nodes(face)
        center = mesh.calc_tri_center(nodes)
        v_list[m] = np.dot(E_d, center) - np.cross(E_c, center)
    return v_list


def make_lp_le_lin_vels(E_d, E_c, mesh):
    """
    Makes the velocitiy field from the rate of strain field
    linear potential, linear elements
Parameters:
        E_d : the rate of strain field dotted with position
        E_c : the rate of strain field crossed with position
        mesh : simple linear mesh input
    Returns:
        v_list : velocities at each element or node
    """
    num_vert = mesh.vertices.shape[0]
    v_list = np.zeros((num_vert, 3))
    for m in range(num_vert):
        vert = mesh.vertices[m]
        v_list[m] = np.dot(E_d, vert) - np.cross(E_c, vert)
    return v_list


def make_cp_qe_lin_vels(E_d, E_c, mesh):
    """
    Make the velocity field from the rate of strain field
    constant potential, quadratic elements
Parameters:
        E_d : the rate of strain field dotted with position
        E_c : the rate of strain field crossed with position
        mesh : simple quadratic mesh input
    Returns:
        v_list : velocities at each element or node
    """
    num_faces = mesh.faces.shape[0]
    v_list = np.zeros((num_faces, 3))
    for m in range(num_faces):
        face = mesh.faces[m]
        nodes = mesh.get_nodes(face)
        center = mesh.calc_tri_center(nodes)
        v_list[m] = np.dot(E_d, center) - np.cross(E_c, center)
    return v_list


def make_lp_qe_lin_vels(E_d, E_c, mesh):
    """
    Make the velocity field from the rate of strain field
    linear potential, quadratic elements
Parameters:
        E_d : the rate of strain field dotted with position
        E_c : the rate of strain field crossed with position
        mesh : simple quadratic mesh input
    Returns:
        v_list : velocities at each element or node
    """
    num_vert = mesh.lin_verts.shape[0]
    v_list = np.zeros((num_vert, 3))
    for m in range(num_vert):
        vert = mesh.lin_verts[m]
        v_list[m] = np.dot(E_d, vert) - np.cross(E_c, vert)
    return v_list


def E_12(mesh):
    """
    Off diagonal (12) rate of strain field eigenfunction
    Returns velocites at each vertex in cartesional coordinates

    Parameters:
        mesh : simple mesh input for dimensions
    Returns;
    """
    E_d = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
    a, b, _c = mesh.dims
    E_c = (a**2 - b**2)/(a**2 + b**2) * E_d[0, 1] * np.array([0, 0, 1])
    return (E_d, E_c)


def E_13(mesh):
    E_d = np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]])
    a, _b, c = mesh.dims
    E_c = (c**2 - a**2)/(c**2 + a**2) * E_d[0, 2] * np.array([0, 1, 0])
    return (E_d, E_c)


def E_23(mesh):
    E_d = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]])
    _a, b, c = mesh.dims
    E_c = (b**2 - c**2)/(b**2 + c**2) * E_d[1, 2] * np.array([1, 0, 0])
    return (E_d, E_c)


def uni_x():
    E_d = np.array([[2, 0, 0], [0, -1, 0], [0, 0, -1]])
    E_c = np.array([0, 0, 0])
    return (E_d, E_c)


def hyper_yz():
    E_d = np.array([[0, 0, 0], [0, 1, 0], [0, 0, -1]])
    E_c = np.array([0, 0, 0])
    return (E_d, E_c)


def uni_z():
    E_d = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 2]])
    E_c = np.array([0, 0, 0])
    return (E_d, E_c)


def hyper_xy():
    E_d = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]])
    E_c = np.array([0, 0, 0])
    return (E_d, E_c)


def diag_eigvec(pm, mesh):
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
    return (E_d, E_c)


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
