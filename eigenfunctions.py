"""
Eigenfunctions for the ellipsoidal particle
"""

import numpy as np

def make_vels(E_d, E_c, mesh, const_or_linear):
    """
    Makes the velocitiy field from the rate of strain field

    Parameters:
        E_d : the rate of strain field dotted with position
        E_c : the rate of strain field crossed with position
        mesh : simple mesh input
        const_or_linear : if constant or linear density distributions
    Returns;
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
    return make_vels(E_d, E_c, mesh, const_or_linear)


def E_13(mesh, const_or_linear):
    E_d = np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]])
    a, _b, c = mesh.dims
    E_c = (c**2 - a**2)/(c**2 + a**2) * E_d[0, 2] * np.array([0, 1, 0])
    return make_vels(E_d, E_c, mesh, const_or_linear)


def E_23(mesh, const_or_linear):
    E_d = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]])
    _a, b, c = mesh.dims
    E_c = (b**2 - c**2)/(b**2 + c**2) * E_d[1, 2] * np.array([1, 0, 0])
    return make_vels(E_d, E_c, mesh, const_or_linear)


def uni_x(mesh, const_or_linear):
    E_d = np.array([[2, 0, 0], [0, -1, 0], [0, 0, -1]])
    E_c = np.array([0, 0, 0])
    return make_vels(E_d, E_c, mesh, const_or_linear)


def hyper_yz(mesh, const_or_linear):
    E_d = np.array([[0, 0, 0], [0, 1, 0], [0, 0, -1]])
    E_c = np.array([0, 0, 0])
    return make_vels(E_d, E_c, mesh, const_or_linear)


def uni_z(mesh, const_or_linear):
    E_d = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 2]])
    E_c = np.array([0, 0, 0])
    return make_vels(E_d, E_c, mesh, const_or_linear)


def hyper_xy(mesh, const_or_linear):
    E_d = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]])
    E_c = np.array([0, 0, 0])
    return make_vels(E_d, E_c, mesh, const_or_linear)
