"""
Module to generate all of the kernels
"""
import math
import numpy as np
import simple_mesh as sm
import gauss_quad as gq


def form_matrix():
    """
    Assembles the A matrix from x + Ax = b. For the double layer rigid body problem.
    Parameters:
    Returns:
    """
    #TODO


def form_dl_mat(mesh):
    """
    Forms the matrix for the regular and singular double layer potential contributions.

    NOTE: can optimize this by vectorizing the sub_mat calculations
    Parameters:
        mesh : simple_mesh object to calculate over
    Returns:
        C : (3 * num_faces, 3 * num_faces) ndarray
    """
    num_faces = mesh.faces.shape[0]
    C = np.zeros((3 * num_faces, 3 * num_faces))
    c_0 = (1. / (4. * math.pi))

    # regular points
    for m in range(num_faces): # source points
        src_center = mesh.calc_tri_center(mesh.faces[m])
        for n in range(num_faces): # over field points
            if n != m:
                field_nodes = mesh.get_nodes(mesh.faces[n])
                field_normal = mesh.calc_normal(mesh.faces[n])
                sub_mat = -c_0 * gq.int_over_tri(make_quad_func(src_center, field_normal), field_nodes)
                C[(3 * m):(3 * m + 3), (3 * n):(3 * n + 3)] = sub_mat

    # singular points as function of all regular points
    for m in range(num_faces): # source points
        src_center = mesh.calc_tri_center(mesh.faces[m])
        sub_mat = np.zeros(3, 3)
        for n in range(num_faces): # over field points
            if n != m:
                sub_mat += c_0 * C[(3 * m):(3 * m + 3), (3 * n):(3 * n + 3)]

        C[(3 * m):(3 * m + 3), (3 * m):(3 * m + 3)] = sub_mat + np.identity(3)

    return C


def form_rb_mat(mesh):
    """
    Forms the matrix for the rigid body motions.
    Added to the double layer matrix to remove (-1) eigenvalue and complete
    part of the space of all possible external flows.

    Parameters:
        mesh : simple_mesh object to calculate over
    Returns:
        (num_faces, num_faces, 3, 3) ndarray
    """
    # translation, many diagonal matrix
    num_faces = mesh.faces.shape[0]
    D = np.zeros((3 * num_faces, 3 * num_faces))
    for m in range(num_faces):
        src_center = mesh.calc_tri_center(mesh.faces[m])
        for n in range(num_faces):
            field_nodes = mesh.get_nodes(mesh.faces[n])
            field_normal = mesh.calc_normal(mesh.faces[n])
            gq.int_over_tri(const_quad_func, field_nodes)
            

    # rotational
    #TODO



def form_nm_mat(mesh):
    """
    Forms the matrix to remove the (+1) eigenvalue from the double layer formulations.

    Parameters:
        mesh : simple_mesh object to calculate over
    Returns:
        (num_faces, num_faces, 3, 3) ndarray
    """
    #TODO


def calc_external_velocity(mesh):
    """
    Forms the vector of external velocity contributions.

    Parameters:
        mesh : simple_mesh object to calculate over
    Returns:
        (num_faces, num_faces, 3, 3) ndarray
    """
    #TODO


def make_quad_func(x_0, n):
    """
    Throwaway function for interfacing
    """
    def quad_func(eta, xi, nodes):
        x = pos(eta, xi, nodes)
        return stresslet(x, x_0, n)
    return quad_func


def const_quad_func(eta, xi, nodes):
    """
    Throwaway function that just returns 1
    """
    return 1
