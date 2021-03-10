"""
Helper file to calculate translational and rotational velocities
from a given solution vector and mesh
"""

import numpy as np
import rigid_DL.geometric as geo
import rigid_DL.gauss_quad as gq

def calc_cp_le_trans_vel(lin_geo_mesh, q):
    """
    Calculate the translational velocity of a particle.
    Constant potential, linear elements.
    Parameters:
        lin_geo_mesh: geometric mesh
        q: DL potential solution; (3 * num_nodes,) ndarray
    Returns:
        part_v: particle translational velocity; (3,) ndarray
    """
    part_v = np.zeros(3)
    num_faces = lin_geo_mesh.get_faces().shape[0]
    S_D = lin_geo_mesh.get_surface_area()
    for face_num in range(num_faces):
        face_hs = lin_geo_mesh.get_hs(face_num)
        j = 3 * face_num
        part_v += q[j : j+3] * 0.5 * face_hs
    part_v *= (-4. * np.pi) / S_D
    return part_v


def calc_cp_le_rot_vel(lin_geo_mesh, q):
    """
    Calculate the rotational velocity of a particle.
    Constant potential, linear elements.
    Parameters:
        lin_geo_mesh: linear geometric mesh
        q: DL potential solutions; (3 * num_nodes,) ndarray
    Returns:
        omega_vec: particle rotational velocity; (3,) ndarray
    """
    num_faces = lin_geo_mesh.get_faces().shape[0]
    q_res = np.reshape(q, (num_faces, 3))
    x_c = lin_geo_mesh.get_centroid()
    w = lin_geo_mesh.get_w()
    w = np.identity(3)
    A_m = lin_geo_mesh.get_A_m()
    omega_vec = 0.
    for face_num in range(num_faces):
        face_nodes = lin_geo_mesh.get_tri_nodes(face_num)
        face_hs = lin_geo_mesh.get_hs(face_num)
        def make_omega_quad(q_vec):
            def omega_quad(xi, eta, nodes):
                pos = geo.linear_interp(xi, eta, nodes)
                X = pos - x_c
                return np.cross(X, q_vec)
            return omega_quad
        tmp_omega = gq.int_over_tri_lin(
            make_omega_quad(q_res[face_num]),
            face_nodes,
            face_hs,
        )
        tmp_arr = []
        for m in range(3):
            tmp_arr.append((1./ A_m[m]) * w[m] * np.dot(w[m], tmp_omega))
        tmp_arr = np.array(tmp_arr)
        omega_vec += -4. * np.pi * np.sum(tmp_arr, axis=0)
    return omega_vec


def calc_lp_le_trans_vel(lin_pot_mesh, lin_geo_mesh, q):
    """
    Calculate the translational velocity of a particle.
    Linear potential, linear elements
    Parameters:
        lin_pot_mesh: linear potential mesh
        lin_geo_mesh: geometric mesh
        q: DL potential solution; (3 * num_nodes,) ndarray
    Returns:
        part_v: particle translational velocity; (3,) ndarray
    """
    part_v = np.zeros(3)
    pot_faces = lin_pot_mesh.get_faces()
    num_faces = pot_faces.shape[0]
    S_D = lin_geo_mesh.get_surface_area()
    for face_num in range(num_faces):
        face_nodes = lin_geo_mesh.get_tri_nodes(face_num)
        face_hs = lin_geo_mesh.get_hs(face_num)
        def make_v_sol_quad(face_num):
            def v_quad(xi, eta, _nodes):
                node_0 = pot_faces[face_num, 0]
                node_1 = pot_faces[face_num, 1]
                node_2 = pot_faces[face_num, 2]
                phi_0 = geo.shape_func_linear(xi, eta, 0)
                phi_1 = geo.shape_func_linear(xi, eta, 1)
                phi_2 = geo.shape_func_linear(xi, eta, 2)
                ret = (
                    q[3 * node_0 : 3 * node_0 + 3] * phi_0 +
                    q[3 * node_1 : 3 * node_1 + 3] * phi_1 +
                    q[3 * node_2 : 3 * node_2 + 3] * phi_2
                )
                return ret
            return v_quad
        part_v += gq.int_over_tri_lin(
            make_v_sol_quad(face_num),
            face_nodes,
            face_hs,
        )
    part_v *= (-4. * np.pi) / S_D
    return part_v


def calc_lp_le_rot_vel(lin_pot_mesh, lin_geo_mesh, q):
    """
    Calculate the rotational velocity of a particle.
    Linear potential, linear elements
    Parameters:
        lin_pot_mesh: linear potential mesh
        lin_geo_mesh: linear geometric mesh
        q: DL potential solutions; (3 * num_nodes,) ndarray
    Returns:
        omega_vec: particle rotational velocity; (3,) ndarray
    """
    num_faces = lin_pot_mesh.get_faces().shape[0]
    pot_faces = lin_pot_mesh.get_faces()
    x_c = lin_geo_mesh.get_centroid()
    w = lin_geo_mesh.get_w()
    A_m = lin_geo_mesh.get_A_m()
    omega_vec = 0.
    for face_num in range(num_faces):
        face_nodes = lin_geo_mesh.get_tri_nodes(face_num)
        face_hs = lin_geo_mesh.get_hs(face_num)
        def make_omega_quad(face_num):
            def omega_quad(xi, eta, nodes):
                pos = geo.linear_interp(xi, eta, nodes)
                X = pos - x_c
                node_0 = pot_faces[face_num, 0]
                node_1 = pot_faces[face_num, 1]
                node_2 = pot_faces[face_num, 2]
                phi_0 = geo.shape_func_linear(xi, eta, 0)
                phi_1 = geo.shape_func_linear(xi, eta, 1)
                phi_2 = geo.shape_func_linear(xi, eta, 2)
                ret = (
                    np.cross(X, phi_0 * q[3 * node_0 : 3 * node_0 + 3]) +
                    np.cross(X, phi_1 * q[3 * node_1 : 3 * node_1 + 3]) +
                    np.cross(X, phi_2 * q[3 * node_2 : 3 * node_2 + 3])
                )
                return ret
            return omega_quad
        tmp_omega = gq.int_over_tri_lin(
            make_omega_quad(face_num),
            face_nodes,
            face_hs,
        )
        tmp_arr = []
        for m in range(3):
            tmp_arr.append((1./ A_m[m]) * w[m] * np.dot(w[m], tmp_omega))
        omega_vec += -4. * np.pi * np.sum(tmp_arr, axis=0)
    return omega_vec


def calc_cp_qe_trans_vel(quad_geo_mesh, q):
    """
    Calculate the translational velocity of a particle.
    Constant potential, quadratidc elements.
    Parameters:
        quad_geo_mesh: geometric mesh
        q: DL potential solution; (3 * num_nodes,) ndarray
    Returns:
        part_v: particle translational velocity; (3,) ndarray
    """
    part_v = np.zeros(3)
    num_faces = quad_geo_mesh.get_faces().shape[0]
    S_D = quad_geo_mesh.get_surface_area()
    for face_num in range(num_faces):
        face_nodes = quad_geo_mesh.get_tri_nodes(face_num)
        face_hs = quad_geo_mesh.get_hs(face_num)
        j = 3 * face_num
        def make_v_quad(q_vec):
            def v_quad(_xi, _eta, _nodes):
                return q_vec
            return v_quad
        part_v += gq.int_over_tri_quad(make_v_quad(q[j : j+3]), face_nodes, face_hs)
    part_v *= (-4. * np.pi) / S_D
    return part_v


def calc_cp_qe_rot_vel(quad_geo_mesh, q):
    """
    Calculate the rotational velocity of a particle.
    Constant potential, quadratic elements.
    Parameters:
        quad_geo_mesh: quadratic geometric mesh
        q: DL potential solutions; (3 * num_nodes,) ndarray
    Returns:
        omega_vec: particle rotational velocity; (3,) ndarray
    """
    num_faces = quad_geo_mesh.get_faces().shape[0]
    x_c = quad_geo_mesh.get_centroid()
    w = quad_geo_mesh.get_w()
    A_m = quad_geo_mesh.get_A_m()
    omega_vec = 0.
    for face_num in range(num_faces):
        face_nodes = quad_geo_mesh.get_tri_nodes(face_num)
        face_hs = quad_geo_mesh.get_hs(face_num)
        def make_omega_quad(q_vec):
            def omega_quad(xi, eta, nodes):
                pos = geo.quadratic_interp(xi, eta, nodes)
                X = pos - x_c
                return np.cross(X, q_vec)
            return omega_quad
        j = 3 * face_num
        tmp_omega = gq.int_over_tri_quad(
            make_omega_quad(q[j : j+3]),
            face_nodes,
            face_hs,
        )
        tmp_arr = []
        for m in range(3):
            tmp_arr.append((1./ A_m[m]) * w[m] * np.dot(w[m], tmp_omega))
        omega_vec += -4. * np.pi * np.sum(tmp_arr, axis=0)
    return omega_vec


def calc_lp_qe_trans_vel(lin_pot_mesh, quad_geo_mesh, q):
    """
    Calculate the translational velocity of a particle.
    Linear potential, quadratic elements
    Parameters:
        lin_pot_mesh: linear potential mesh
        quad_geo_mesh: geometric mesh
        q: DL potential solution; (3 * num_nodes,) ndarray
    Returns:
        part_v: particle translational velocity; (3,) ndarray
    """
    part_v = np.zeros(3)
    pot_faces = lin_pot_mesh.get_faces()
    num_faces = pot_faces.shape[0]
    S_D = quad_geo_mesh.get_surface_area()
    for face_num in range(num_faces):
        face_nodes = quad_geo_mesh.get_tri_nodes(face_num)
        face_hs = quad_geo_mesh.get_hs(face_num)
        def make_v_sol_quad(face_num):
            def v_quad(xi, eta, _nodes):
                node_0 = pot_faces[face_num, 0]
                node_1 = pot_faces[face_num, 1]
                node_2 = pot_faces[face_num, 2]
                phi_0 = geo.shape_func_linear(xi, eta, 0)
                phi_1 = geo.shape_func_linear(xi, eta, 1)
                phi_2 = geo.shape_func_linear(xi, eta, 2)
                ret = (
                    q[3 * node_0 : 3 * node_0 + 3] * phi_0 +
                    q[3 * node_1 : 3 * node_1 + 3] * phi_1 +
                    q[3 * node_2 : 3 * node_2 + 3] * phi_2
                )
                return ret
            return v_quad
        part_v += gq.int_over_tri_quad(
            make_v_sol_quad(face_num),
            face_nodes,
            face_hs,
        )
    part_v *= (-4. * np.pi) / S_D
    return part_v


def calc_lp_qe_rot_vel(lin_pot_mesh, quad_geo_mesh, q):
    """
    Calculate the rotational velocity of a particle.
    Linear potential, quadratic elements
    Parameters:
        lin_pot_mesh: linear potential mesh
        quad_geo_mesh: quadratic geometric mesh
        q: DL potential solutions; (3 * num_nodes,) ndarray
    Returns:
        omega_vec: particle rotational velocity; (3,) ndarray
    """
    num_faces = lin_pot_mesh.get_faces().shape[0]
    pot_faces = lin_pot_mesh.get_faces()
    x_c = quad_geo_mesh.get_centroid()
    w = quad_geo_mesh.get_w()
    A_m = quad_geo_mesh.get_A_m()
    omega_vec = 0.
    for face_num in range(num_faces):
        face_nodes = quad_geo_mesh.get_tri_nodes(face_num)
        face_hs = quad_geo_mesh.get_hs(face_num)
        def make_omega_quad(face_num):
            def omega_quad(xi, eta, nodes):
                pos = geo.linear_interp(xi, eta, nodes)
                X = pos - x_c
                node_0 = pot_faces[face_num, 0]
                node_1 = pot_faces[face_num, 1]
                node_2 = pot_faces[face_num, 2]
                phi_0 = geo.shape_func_linear(xi, eta, 0)
                phi_1 = geo.shape_func_linear(xi, eta, 1)
                phi_2 = geo.shape_func_linear(xi, eta, 2)
                ret = (
                    np.cross(X, phi_0 * q[3 * node_0 : 3 * node_0 + 3]) +
                    np.cross(X, phi_1 * q[3 * node_1 : 3 * node_1 + 3]) +
                    np.cross(X, phi_2 * q[3 * node_2 : 3 * node_2 + 3])
                )
                return ret
            return omega_quad
        tmp_omega = gq.int_over_tri_quad(
            make_omega_quad(face_num),
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
    Deprecated, version calculated from moment of inertia tensor better.
    Calculate the A_m vector given a linear geometric mesh.
    Parameters:
        lin_geo_mesh: linear geometric mesh
    Returns:
        A_m: A_m vector for rotational velocity calculation
    """
    num_faces = lin_geo_mesh.get_faces().shape[0]
    x_c = lin_geo_mesh.get_centroid()
    w = np.identity(3)
    A_m = np.zeros(3)
    for m in range(3):
        def make_am_quad_func(w_m):
            def am_quad(xi, eta, nodes):
                pos = geo.linear_interp(xi, eta, nodes)
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
