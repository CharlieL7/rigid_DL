"""
Helper functions for creating eigenvectors 
"""
import numpy as np
import rigid_DL.geometric as geo
import rigid_DL.eigenfunctions as eigfun
import rigid_DL.gauss_quad as gq
from rigid_DL.enums import Mesh_Type, Pot_Type

def make_lin_eig_vels(pot_mesh, E_d, E_c):
    """
    Make the eigenfunction velocity field from the rate of strain matrix

    Parameters:
        pot_mesh: potential mesh
        E_d: ROS field, dotted with position
        E_c: ROS field, crossed with position
    Returns:
        v_list: velocities at each node
    """
    pot_nodes = pot_mesh.get_nodes()
    num_nodes = pot_nodes.shape[0]
    v_list = np.zeros((num_nodes, 3))
    for m in range(num_nodes):
        node = pot_nodes[m]
        v_list[m] = E_d @ node - np.cross(E_c, node)
    return v_list


def make_quad_eig_vels(pot_mesh, dims, kappa_vec):
    """
    Makes the eigenfunction velocity field for the 3x3 quadratic ROS
    field

    Parameters:
        pot_mesh: potential mesh
        dims: ellipsoidal dimensions
        kappa_vec: kappa values for 3x3 system
    Returns:
        v_3x3: (3, N, 3) ndarray (N, 3) for each kappa value
    """
    pot_nodes = pot_mesh.get_nodes()
    num_nodes = pot_nodes.shape[0]
    v_3x3 = np.empty((3, num_nodes, 3))
    for i, kappa in enumerate(kappa_vec):
        H = eigfun.calc_3x3_evec(dims, kappa)
        for m in range(num_nodes):
            node = pot_nodes[m]
            x_tmp = np.array([
                [node[1] * node[2]],
                [node[2] * node[0]],
                [node[0] * node[1]],
            ])
            v_3x3[i, m] = np.ravel(H * x_tmp)
    return v_3x3


def calc_inner_product_2(types, pot_mesh, geo_mesh, psi):
    """
    Inner product of eigenfunction with itself
    Parameters:
        types: (pot_type, mesh_type)
        pot_mesh: potential mesh
        psi: eigenvector
    Returns:
        external flow magnitude: scalar value
    """
    num_nodes = pot_mesh.get_nodes().shape[0]
    tmp_psi = np.reshape(psi, (num_nodes, 3))
    psi_2 = np.einsum("ij,ij->i", tmp_psi, tmp_psi)
    func_map = {
        (Pot_Type.CONSTANT, Mesh_Type.LINEAR): inner_product_cp_le,
        (Pot_Type.CONSTANT, Mesh_Type.QUADRATIC): inner_product_cp_qe,
        (Pot_Type.LINEAR, Mesh_Type.LINEAR): inner_product_lp_le,
        (Pot_Type.LINEAR, Mesh_Type.QUADRATIC): inner_product_lp_qe,
        (Pot_Type.QUADRATIC, Mesh_Type.QUADRATIC): inner_product_qp_qe,
    }
    return func_map[types](pot_mesh, geo_mesh, psi_2)


def inner_product_cp_le(cons_pot_mesh, lin_geo_mesh, psi_2):
    num_faces = cons_pot_mesh.get_faces().shape[0]
    ret = 0.
    for face_num in range(num_faces):
        face_hs = lin_geo_mesh.get_hs(face_num)
        ret += 0.5 * face_hs * psi_2[face_num]
    return ret


def inner_product_cp_qe(lin_pot_mesh, quad_geo_mesh, psi_2):
    num_faces = lin_pot_mesh.get_faces().shape[0]
    ret = 0.
    for face_num in range(num_faces):
        face_nodes = quad_geo_mesh.get_tri_nodes(face_num)
        face_hs = quad_geo_mesh.get_hs(face_num)
        face_area = gq.int_over_tri_quad(geo.const_func, face_nodes, face_hs)
        ret += psi_2[face_num] * face_area
    return ret


def inner_product_lp_le(lin_pot_mesh, lin_geo_mesh, psi_2):
    pot_faces = lin_pot_mesh.get_faces()
    num_faces = pot_faces.shape[0]
    ret = 0.
    for face_num in range(num_faces):
        face_nodes = lin_geo_mesh.get_tri_nodes(face_num)
        face_hs = lin_geo_mesh.get_hs(face_num)
        def make_func(face_num):
            def quad_func(xi, eta, _nodes):
                node_0 = pot_faces[face_num, 0]
                node_1 = pot_faces[face_num, 1]
                node_2 = pot_faces[face_num, 2]
                phi_0 = geo.shape_func_linear(xi, eta, 0)
                phi_1 = geo.shape_func_linear(xi, eta, 1)
                phi_2 = geo.shape_func_linear(xi, eta, 2)
                ret = (
                    psi_2[node_0] * phi_0 +
                    psi_2[node_1] * phi_1 +
                    psi_2[node_2] * phi_2
                )
                return ret
            return quad_func
        ret += gq.int_over_tri_lin(
            make_func(face_num),
            face_nodes,
            face_hs,
        )
    return ret


def inner_product_lp_qe(lin_pot_mesh, quad_geo_mesh, psi_2):
    pot_faces = lin_pot_mesh.get_faces()
    num_faces = pot_faces.shape[0]
    ret = 0.
    for face_num in range(num_faces):
        face_nodes = quad_geo_mesh.get_tri_nodes(face_num)
        face_hs = quad_geo_mesh.get_hs(face_num)
        def make_func(face_num):
            def quad_func(xi, eta, _nodes):
                node_0 = pot_faces[face_num, 0]
                node_1 = pot_faces[face_num, 1]
                node_2 = pot_faces[face_num, 2]
                phi_0 = geo.shape_func_linear(xi, eta, 0)
                phi_1 = geo.shape_func_linear(xi, eta, 1)
                phi_2 = geo.shape_func_linear(xi, eta, 2)
                ret = (
                    psi_2[node_0] * phi_0 +
                    psi_2[node_1] * phi_1 +
                    psi_2[node_2] * phi_2
                )
                return ret
            return quad_func
        ret += gq.int_over_tri_quad(
            make_func(face_num),
            face_nodes,
            face_hs,
        )
    return ret


def inner_product_qp_qe(quad_pot_mesh, quad_geo_mesh, psi_2):
    pot_faces = quad_pot_mesh.get_faces()
    num_faces = pot_faces.shape[0]
    ret = 0.
    for face_num in range(num_faces):
        face_nodes = quad_geo_mesh.get_tri_nodes(face_num)
        face_hs = quad_geo_mesh.get_hs(face_num)
        def make_func(face_num):
            def quad_func(xi, eta, _nodes):
                node_0 = pot_faces[face_num, 0]
                node_1 = pot_faces[face_num, 1]
                node_2 = pot_faces[face_num, 2]
                node_3 = pot_faces[face_num, 3]
                node_4 = pot_faces[face_num, 4]
                node_5 = pot_faces[face_num, 5]
                psi_2_arr = [
                    psi_2[node_0],
                    psi_2[node_1],
                    psi_2[node_2],
                    psi_2[node_3],
                    psi_2[node_4],
                    psi_2[node_5],
                ]
                return geo.quadratic_interp(xi, eta, psi_2_arr)
            return quad_func
        ret += gq.int_over_tri_quad(
            make_func(face_num),
            face_nodes,
            face_hs,
        )
    return ret


def make_lin_psi_func(E_d, E_c):
    """
    Makes the eigenvector function to integrate over the surface
    """
    def quad_func(xi, eta, nodes):
        x = geo.linear_interp(xi, eta, nodes)
        return np.linalg.norm(np.dot(E_d, x) - np.cross(E_c, x))
    return quad_func


def make_quad_psi_func(E_d, E_c):
    """
    Makes the eigenvector function to integrate over the surface
    """
    def quad_func(xi, eta, nodes):
        x = geo.quadratic_interp(xi, eta, nodes)
        return np.linalg.norm(np.dot(E_d, x) - np.cross(E_c, x))
    return quad_func
