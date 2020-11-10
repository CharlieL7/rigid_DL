"""
Helper file to calculate translational and rotational velocities
from a given solution vector and mesh
"""

import numpy as np
import rigid_DL.geometric as geo
import rigid_DL.gauss_quad as gq

def calc_le_trans_vel(geo_mesh, q):
    """
    Calculate the translational velocity of a particle.
    Parameters:
        geo_mesh: geometric mesh
        q: DL potential solution; (3 * num_nodes,) ndarray
    Returns:
        part_v: particle translational velocity; (3,) ndarray
    """
    part_v = np.zeros(3)
    num_faces = geo_mesh.faces.shape[0]
    S_D = geo_mesh.get_surface_area()
    for face_num in range(num_faces):
        face_hs = geo_mesh.get_hs(face_num)
        j = 3 * face_num
        part_v += q[j : j+3] * 0.5 * face_hs
    part_v = part_v * (-4. * np.pi) / S_D
    return part_v


def calc_le_rot_vel(lin_geo_mesh, q):
    """
    Calculate the rotational velocity of a particle.
    Parameters:
        lin_geo_mesh: linear geometric mesh
        q: DL potential solutions; (3 * num_nodes,) ndarray
    Returns:
        omega_vec: particle rotational velocity; (3,) ndarray
    """
    num_faces = lin_geo_mesh.get_faces().shape[0]
    x_c = lin_geo_mesh.get_centroid()
    w = lin_geo_mesh.calc_rotation_vectors()
    omega_vec = 0.
    A_m = calc_le_Am_vec(lin_geo_mesh)
    for face_num in range(num_faces):
        face_nodes = lin_geo_mesh.get_tri_nodes(face_num)
        face_hs = lin_geo_mesh.get_hs(face_num)
        def make_omega_quad(q_vec):
            def omega_quad(xi, eta, nodes):
                pos = geo.pos_linear(xi, eta, nodes)
                X = pos - x_c
                return np.cross(X, q_vec)
            return omega_quad
        j = 3 * face_num
        tmp_omega = gq.int_over_tri_lin(
            make_omega_quad(q[j : j+3]),
            face_nodes,
            face_hs,
        )
        tmp_arr = []
        for m in range(3):
            tmp_arr.append((1./ A_m[m]) * w[m] * np.dot(w[m], tmp_omega))
        omega_vec += -4. * np.pi * np.sum(tmp_arr, axis=0)
    return omega_vec


def calc_le_Am_vec(lin_geo_mesh):
    """
    Calculate the A_m vector given a linear geometric mesh.
    Parameters:
        lin_geo_mesh: linear geometric mesh
    Returns:
        A_m: A_m vector for rotational velocity calculation
    """
    num_faces = lin_geo_mesh.get_faces().shape[0]
    x_c = lin_geo_mesh.get_centroid()
    w = lin_geo_mesh.calc_rotation_vectors()
    A_m = np.zeros(3)
    for m in range(3):
        def make_am_quad_func(w_m):
            def am_quad(xi, eta, nodes):
                pos = geo.pos_linear(xi, eta, nodes)
                X = pos - x_c
                return np.dot(np.cross(w_m, X), np.cross(w_m, X))
            return am_quad
        for face_num in range(num_faces): # get A_m terms
            face_nodes = lin_geo_mesh.get_tri_nodes(face_num)
            face_hs = lin_geo_mesh.get_hs(face_num)
            A_m[m] += gq.int_over_tri_lin(
                make_am_quad_func(w[m]),
                face_nodes,
                face_hs,
            )
    return A_m
