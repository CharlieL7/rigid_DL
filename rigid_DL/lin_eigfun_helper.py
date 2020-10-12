"""
Helper functions for creating eigenvectors for the linear
ROS eigenfunctions depending on mesh and potential parameterization.
"""
import numpy as np
import rigid_DL.geometric as geo

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


def make_lin_psi_func(E_d, E_c):
    """
    Makes the eigenvector function to integrate over the surface
    """
    def quad_func(xi, eta, nodes):
        x = geo.pos_linear(xi, eta, nodes)
        return np.linalg.norm(np.dot(E_d, x) - np.cross(E_c, x))
    return quad_func


def make_quad_psi_func(E_d, E_c):
    """
    Makes the eigenvector function to integrate over the surface
    """
    def quad_func(xi, eta, nodes):
        x = geo.pos_quadratic(xi, eta, nodes)
        return np.linalg.norm(np.dot(E_d, x) - np.cross(E_c, x))
    return quad_func



