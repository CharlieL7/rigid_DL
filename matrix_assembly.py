"""
Module to generate all of the kernels
"""
import math
import numpy as np
import simple_mesh as sm
import gauss_quad as gq

C0 = (1. / (4. * math.pi))

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

    for m in range(num_faces): # source points
        src_center = mesh.calc_tri_center(mesh.faces[m])
        for n in range(num_faces): # field points
            field_nodes = mesh.get_nodes(mesh.faces[n])
            field_normal = mesh.calc_normal(mesh.faces[n])
            if n != m:
                sub_mat = gq.int_over_tri(make_quad_func(src_center, field_normal), field_nodes)
                C[(3 * m):(3 * m + 3), (3 * n):(3 * n + 3)] += sub_mat
                C[(3 * m):(3 * m + 3), (3 * m):(3 * m + 3)] -= sub_mat
                # for constant potential, the integral over singular elements just cancels
                # so do nothing for n == m
        C[(3 * m):(3 * m + 3), (3 * m):(3 * m + 3)] -= np.identity(3)

    return C


def form_rb_mat(mesh):
    """
    Forms the matrix for the rigid body motions.
    Added to the double layer matrix to remove (-1) eigenvalue and complete
    part of the space of all possible external flows.  Parameters:
        mesh : simple_mesh object to calculate over
    Returns:
        (num_faces, num_faces, 3, 3) ndarray
    """
    D = np.zeros((3 * num_faces, 3 * num_faces))
    # translation, many diagonal matrix
    num_faces = mesh.faces.shape[0]
    trans_vals = np.zeros((num_faces))
    for n in range(num_faces):
        field_nodes = mesh.get_nodes(mesh.faces[n])
        trans_vals = gq.int_over_tri(const_quad_func, field_nodes)/mesh.surf_area
    for m in range(num_faces):
        for n in range(num_faces):
            D[(3 * m):(3 * m + 3), (3 * m):(3 * m + 3)] += trans_vals[m] * np.identity(3)

    # rotational
    rot_m = np.zeros((num_faces, 3))
    rot_n = np.zeros((num_faces, 3))
    for k in range(3):
        for m in range(num_faces): # over sources
            src_center = mesh.calc_tri_center(mesh.faces[m])
            rot_m[m, :] = np.cross( mesh.w(:, k), sec_center - mesh.centroid)
        for n in range(num_faces): # over fields
            field_nodes = mesh.get_nodes(mesh.faces[n])
            rot_n[n, :] = gq.int_over_tri( make_rot_func(mesh.w[:, k], mesh.centroid), field_nodes )

        for m in range(num_faces):
            for n in range(num_faces):
                D[(3 * m):(3 * m + 3), (3 * m):(3 * m + 3)] += np.outer(rot_m, rot_n)

    return D
        

def form_nm_mat(mesh):
    """
    Forms the matrix to remove the (+1) eigenvalue from the double layer formulations.

    Parameters:
        mesh : simple_mesh object to calculate over
    Returns:
        (num_faces, num_faces, 3, 3) ndarray
    """
    E = np.zeros((3 * num_faces, 3 * num_faces))
    nm_vecs = np.zeros((num_faces, 3))
    num_faces = mesh.faces.shape[0]
    for n in range(num_faces): # field points
        normal = mesh.calc_normal(mesh.faces[n])
        nm_vecs[m, :] = trans_vals * normal
    for m in range(num_faces): # source points
        normal_m = mesh.calc_normal(mesh.faces[m])
        E[(3 * m):(3 * m + 3), (3 * m):(3 * m + 3)] -= np.outer(normal_m, nm_vecs)
    return E


def calc_external_velocity(mesh):
    """
    Forms the vector of external velocity contributions.
    part of RHS vector

    Parameters:
        mesh : simple_mesh object to calculate over
    Returns:
        (num_faces, num_faces, 3, 3) ndarray
    """
    #TODO


def calc_RHS(mesh):
    """
    Forms the vector of force and torque RHS contributions.
    part of RHS vector

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


def make_rot_func(w, centroid)
    def quad_func(eta, xi, nodes):
        x = pos(eta, xi, nodes)
        return np.cross(w, x - centroid)
    return quad_func


def const_quad_func(eta, xi, nodes):
    """
    Throwaway function that just returns 1
    """
    return 1
