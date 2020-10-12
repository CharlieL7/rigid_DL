"""
Helper functions for creating eigenvectors for the quadratic
ROS eigenfunctions depending on mesh and potential parameterization.
"""

import numpy as np
import rigid_DL.eigenfunctions as eigfun

def make_cp_le_quad_vels(mesh, dims, kappa_vec):
    num_faces = mesh.faces.shape[0]
    v_3x3 = np.empty((3, num_faces, 3))
    for i, kappa in enumerate(kappa_vec):
        H = eigfun.calc_3x3_evec(dims, kappa)
        for m in range(num_faces):
            face = mesh.faces[m]
            nodes = mesh.get_nodes(face)
            center = mesh.calc_tri_center(nodes)
            x_tmp = np.array([
                [center[1] * center[2]],
                [center[2] * center[0]],
                [center[0] * center[1]],
            ])
            v_3x3[i, m] = np.ravel(H * x_tmp)
    return v_3x3


def make_cp_qe_quad_vels(mesh, dims, kappa_vec):
    num_faces = mesh.faces.shape[0]
    v_3x3 = np.empty((3, num_faces, 3))
    for i, kappa in enumerate(kappa_vec):
        H = eigfun.calc_3x3_evec(dims, kappa)
        for m in range(num_faces):
            face = mesh.faces[m]
            nodes = mesh.get_nodes(face)
            center = mesh.calc_tri_center(nodes)
            x_tmp = np.array([
                [center[1] * center[2]],
                [center[2] * center[0]],
                [center[0] * center[1]],
            ])
            v_3x3[i, m] = np.ravel(H * x_tmp)
    return v_3x3


def make_lp_le_quad_vels(mesh, dims, kappa_vec):
    num_verts = mesh.vertices.shape[0]
    v_3x3 = np.empty((3, num_verts, 3))
    for i, kappa in enumerate(kappa_vec):
        H = eigfun.calc_3x3_evec(dims, kappa)
        for m in range(num_verts):
            vert = mesh.vertices[m]
            x_tmp = np.array([
                [vert[1] * vert[2]],
                [vert[2] * vert[0]],
                [vert[0] * vert[1]],
            ])
            v_3x3[i, m] = np.ravel(H * x_tmp)
    return v_3x3


def make_lp_qe_quad_vels(mesh, dims, kappa_vec):
    num_verts = mesh.lin_verts.shape[0]
    v_3x3 = np.empty((3, num_verts, 3))
    for i, kappa in enumerate(kappa_vec):
        H = eigfun.calc_3x3_evec(dims, kappa)
        for m in range(num_verts):
            vert = mesh.lin_verts[m]
            x_tmp = np.array([
                [vert[1] * vert[2]],
                [vert[2] * vert[0]],
                [vert[0] * vert[1]],
            ])
            v_3x3[i, m] = np.ravel(H * x_tmp)
    return v_3x3
