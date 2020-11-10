"""
Matrix assembly functions for the different parameterizations
"""
import numpy as np
import rigid_DL.gauss_quad as gq
import rigid_DL.geometric as geo

def make_mat_cp_le(cons_pot_mesh, lin_geo_mesh):
    """
    Makes the stiffness matrix using closed surface singularity subtraction.
    For constant potentials over a linear elements.
    Faces loop over source points loop.
    Parameters:
        cons_pot_mesh: constant potential mesh
        lin_geo_mesh : linear geometric mesh
    Returns:
        the stresslet matrix
    """
    pot_faces = cons_pot_mesh.get_faces()
    assert pot_faces.shape[0] == lin_geo_mesh.get_faces().shape[0]
    num_faces = pot_faces.shape[0] # should be same for either pot or geo
    c_0 = 1. / (4. * np.pi)
    C = np.zeros((3 * num_faces, 3 * num_faces))
    for face_num in range(num_faces):
        face_nodes = lin_geo_mesh.get_tri_nodes(face_num)
        face_n = lin_geo_mesh.get_normal(face_num)
        face_hs = lin_geo_mesh.get_hs(face_num)
        for src_num in range(num_faces):
            src_center = cons_pot_mesh.get_node(src_num)
            if face_num != src_num:
                sub_mat = gq.int_over_tri_lin(
                    make_cp_le_quad_func(face_n, src_center),
                    face_nodes,
                    face_hs
                )
                C[(3 * src_num):(3 * src_num + 3),
                  (3 * face_num):(3 * face_num + 3)] += sub_mat
                C[(3 * src_num):(3 * src_num + 3),
                  (3 * src_num):(3 * src_num + 3)] -= sub_mat
            # do nothing face_num == src_num, how it works out for constant elements
    for src_num in range(num_faces):
        C[(3 * src_num):(3 * src_num + 3),
          (3 * src_num):(3 * src_num + 3)] -= 4. * np.pi * np.identity(3)
    C = np.dot(c_0, C)
    return C


def make_mat_lp_le(lin_pot_mesh, lin_geo_mesh):
    """
    Makes the stiffness matrix using closed surface singularity subtraction.
    For linear potentials over a linear elements.
    Parameters:
        lin_pot_mesh: linear potential mesh
        lin_geo_mesh : linear geometric mesh
    Returns:
        the stresslet matrix
    """
    geo_faces = lin_geo_mesh.get_faces()
    pot_faces = lin_pot_mesh.get_faces()
    assert geo_faces.shape[0] == pot_faces.shape[0]
    num_faces = geo_faces.shape[0]
    pot_nodes = lin_pot_mesh.get_nodes()
    num_nodes = pot_nodes.shape[0]
    c_0 = 1. / (4. * np.pi)
    C = np.zeros((3 * num_nodes, 3 * num_nodes))

    for face_num in range(num_faces): # integrate over faces
        face_nodes = lin_geo_mesh.get_tri_nodes(face_num)
        face_n = lin_geo_mesh.get_normal(face_num)
        face_hs = lin_geo_mesh.get_hs(face_num)
        for src_num in range(num_nodes): # source points
            src_pt = pot_nodes[src_num]
            is_singular, local_singular_ind = lin_pot_mesh.check_in_face(src_num, face_num)

            if is_singular: # singular triangle
                for node_num in range(3):
                    node_global_num = pot_faces[face_num, node_num] # global index for vert
                    sub_mat = gq.int_over_tri_lin(
                        make_sing_lp_le_quad_func(
                            face_n, src_pt, node_num, local_singular_ind
                            ),
                        face_nodes,
                        face_hs
                    )
                    C[(3 * src_num):(3 * src_num + 3),
                      (3 * node_global_num):(3 * node_global_num + 3)] += sub_mat

            else: # regular triangle
                for node_num in range(3):
                    node_global_num = pot_faces[face_num, node_num] # global index for vert
                    sub_mat = gq.int_over_tri_lin(
                        make_reg_lp_le_quad_func(
                            face_n, src_pt, node_num
                        ),
                        face_nodes,
                        face_hs
                    )
                    C[(3 * src_num):(3 * src_num + 3),
                      (3 * node_global_num):(3 * node_global_num + 3)] += sub_mat
                # subtracting the q(x_0) term
                sub_mat = gq.int_over_tri_lin(
                    make_cp_le_quad_func(face_n, src_pt),
                    face_nodes,
                    face_hs
                )
                C[(3 * src_num):(3 * src_num + 3), (3 * src_num):(3 * src_num + 3)] -= sub_mat

    for src_num in range(num_nodes): # source points
        # whole surface q(x_0) term
        C[(3 * src_num):(3 * src_num + 3), (3 * src_num):(3 * src_num + 3)] -= (
            4. * np.pi * np.identity(3)
        )

    C = np.dot(c_0, C)
    return C


def make_mat_cp_qe(cons_pot_mesh, quad_geo_mesh):
    """
    Makes the stiffness matrix using closed surface singularity subtraction.
    For constant potentials over quadratic elements.
    Parameters:
        cons_pot_mesh: constant potential mesh
        quad_geo_mesh : quadratic geometric mesh
    Returns:
        the stresslet matrix
    """
    geo_faces = quad_geo_mesh.get_faces()
    pot_faces = cons_pot_mesh.get_faces()
    assert geo_faces.shape[0] == pot_faces.shape[0]
    num_faces = geo_faces.shape[0]
    c_0 = 1. / (4. * np.pi)
    C = np.zeros((3 * num_faces, 3 * num_faces))
    for face_num in range(num_faces): # field points
        face_nodes = quad_geo_mesh.get_tri_nodes(face_num)
        face_n = quad_geo_mesh.get_quad_n(face_num)
        for src_num in range(num_faces): # source points
            src_center = cons_pot_mesh.get_node(src_num)
            if face_num != src_num:
                sub_mat = gq.int_over_tri_quad_n(
                    make_cp_qe_quad_func(src_center),
                    face_nodes,
                    face_n
                )
                C[(3 * src_num):(3 * src_num + 3),
                  (3 * face_num):(3 * face_num + 3)] += sub_mat
                C[(3 * src_num):(3 * src_num + 3),
                  (3 * src_num):(3 * src_num + 3)] -= sub_mat
            # do nothing face_num == src_num, how it works out for constant elements

    for src_num in range(num_faces):
        C[(3 * src_num):(3 * src_num + 3),
          (3 * src_num):(3 * src_num + 3)] -= 4. * np.pi * np.identity(3)
    C = np.dot(c_0, C)
    return C


def make_mat_lp_qe(lin_pot_mesh, quad_geo_mesh):
    """
    Makes the stiffness matrix using closed surface singularity subtraction.
    For linear potentials over quadratic elements.
    Parameters:
        lin_pot_mesh: linear potential mesh
        quad_geo_mesh: quadratic geometric mesh
    Returns:
        the stresslet matrix
    """
    geo_faces = quad_geo_mesh.get_faces()
    pot_faces = lin_pot_mesh.get_faces()
    assert geo_faces.shape[0] == pot_faces.shape[0]
    num_faces = geo_faces.shape[0]
    pot_nodes = lin_pot_mesh.get_nodes()
    num_nodes = pot_nodes.shape[0]
    c_0 = 1. / (4. * np.pi)
    C = np.zeros((3 * num_nodes, 3 * num_nodes))

    for face_num in range(num_faces): # integrate over faces
        face_nodes = quad_geo_mesh.get_tri_nodes(face_num)
        face_n = quad_geo_mesh.quad_n[face_num]
        for src_num in range(num_nodes): # source points
            src_pt = lin_pot_mesh.get_node(src_num)
            is_singular, local_singular_ind = lin_pot_mesh.check_in_face(src_num, face_num)

            if is_singular: # singular triangle
                for node_num in range(3):
                    node_global_num = pot_faces[face_num, node_num] # global index for vert
                    sub_mat = gq.int_over_tri_quad_n(
                        make_sing_lp_qe_quad_func(
                            src_pt, node_num, local_singular_ind
                            ),
                        face_nodes,
                        face_n
                    )
                    C[(3 * src_num):(3 * src_num + 3),
                      (3 * node_global_num):(3 * node_global_num + 3)] += sub_mat

            else: # regular triangle
                for node_num in range(3):
                    node_global_num = pot_faces[face_num, node_num] # global index for vert
                    sub_mat = gq.int_over_tri_quad_n(
                        make_reg_lp_qe_quad_func(src_pt, node_num),
                        face_nodes,
                        face_n
                    )
                    C[(3 * src_num):(3 * src_num + 3),
                      (3 * node_global_num):(3 * node_global_num + 3)] += sub_mat
                # subtracting the q(x_0) term
                sub_mat = gq.int_over_tri_quad_n(
                    make_cp_qe_quad_func(src_pt),
                    face_nodes,
                    face_n
                )
                C[(3 * src_num):(3 * src_num + 3), (3 * src_num):(3 * src_num + 3)] -= sub_mat

    for src_num in range(num_nodes): # source points
        # whole surface q(x_0) term
        C[(3 * src_num):(3 * src_num + 3), (3 * src_num):(3 * src_num + 3)] -= (
            4. * np.pi * np.identity(3)
        )

    C = np.dot(c_0, C)
    return C


def make_cp_le_quad_func(n, x_0):
    """
    Makes the constant potential function that is integrated over
    linear elements for the stiffness matrix
    Parameters:
        n: unit normal vector
        x_0: the source point
    """
    def quad_func(xi, eta, nodes):
        x = geo.pos_linear(xi, eta, nodes)
        return geo.stresslet_n(x, x_0, n)
    return quad_func


def make_cp_qe_quad_func(x_0):
    """"
    Makes the constant potential function that is integrated over
    quadratic elements for the stiffness matrix
    Parameters:
        x_0: the source point
    """
    def quad_func(xi, eta, nodes):
        x = geo.pos_quadratic(xi, eta, nodes)
        return geo.stresslet(x, x_0)
    return quad_func


def make_reg_lp_le_quad_func(n, x_0, node_num):
    """
    Makes the regular (non-singular) linear potential, linear element function
    that is integrated for the stiffness matrix
    Parameters:
        n: unit normal vector
        x_0: source point
        node_num: which potential shape function [0, 1, 2]
    """
    def quad_func(xi, eta, nodes):
        x = geo.pos_linear(xi, eta, nodes)
        S = geo.stresslet_n(x, x_0, n)
        phi = geo.shape_func_linear(xi, eta, node_num)
        return phi * S
    return quad_func


def make_reg_lp_qe_quad_func(x_0, node_num):
    """
    Makes the regular (non-singular) linear potential, quadratic element function
    that is integrated for the stiffness matrix
    Parameters:
        x_0: source point
        node_num: which potential shape function [0, 1, 2]
    """
    def quad_func(xi, eta, nodes):
        x = geo.pos_quadratic(xi, eta, nodes)
        S = geo.stresslet(x, x_0)
        phi = geo.shape_func_linear(xi, eta, node_num)
        return phi * S
    return quad_func


def make_sing_lp_le_quad_func(n, x_0, node_num, singular_ind):
    """
    Makes the sinuglar linear potential, linear element function
    that is integrated for the stiffness matrix
    
    NOTE: This always evaluates to zero because the \hat{x} is perpendicular to the normal vector
    for linear elements.
    Parameters:
        n: unit normal vector
        x_0: source point
        node_num: which shape function
        singular_ind: local singular index for a face
    """
    def quad_func(xi, eta, nodes):
        x = geo.pos_linear(xi, eta, nodes)
        phi = geo.shape_func_linear(xi, eta, node_num)
        # shape function for source point is [1, 0, 0], [0, 1, 0], or [0, 0, 1]
        if node_num == singular_ind:
            if (phi - 1) == 0: # getting around division by 0
                return np.zeros([3, 3])
            else:
                return (phi - 1) * geo.stresslet_n(x, x_0, n)
        else:
            if phi == 0:
                return np.zeros([3, 3])
            else:
                return (phi) * geo.stresslet_n(x, x_0, n)
    return quad_func


def make_sing_lp_qe_quad_func(x_0, node_num, singular_ind):
    """
    Makes the sinuglar linear potential, quadratic element function
    that is dotted with normal vectors and integrated for the stiffness matrix
    Parameters:
        x_0: source point
        node_num: which potential shape function [0, 1, 2]
        singular_ind: local singular index for a face
    """
    def quad_func(xi, eta, nodes):
        x = geo.pos_quadratic(xi, eta, nodes)
        phi = geo.shape_func_linear(xi, eta, node_num)
        if node_num == singular_ind:
            if (phi - 1) == 0: # getting around division by 0
                return np.zeros([3, 3, 3])
            else:
                if np.linalg.norm(x - x_0) < 1e-6:
                    print("nearly singular lp qe x_hat error")
                return (phi - 1) * geo.stresslet(x, x_0)
        else:
            if phi == 0:
                return np.zeros([3, 3, 3])
            else:
                if np.linalg.norm(x - x_0) < 1e-6:
                    print("nearly singular lp qe x_hat error")
                return (phi) * geo.stresslet(x, x_0)
    return quad_func
