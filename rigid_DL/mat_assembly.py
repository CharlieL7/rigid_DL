"""
Matrix assembly functions for the different parameterizations
"""
import numpy as np
import rigid_DL.gauss_quad as gq
import rigid_DL.geometric as geo
import ctypes as ct


# Cpp linked assembly versions
def make_mat_cp_le_cpp(cons_pot_mesh, lin_geo_mesh):
    """
    Assembles the DL operator matrix for a constant potential,
    linear geometry discretization.
    Links to C library for speed.
    Parameters:
        cons_pot_mesh: constant potential mesh
        lin_geo_mesh: linear geometric mesh
    Returns:
        the stresslet matrix
    """
    mata_lib = ct.CDLL("/home/charlie/local_git/rigid_DL/c_src/matrix_assem.so")
    mata_lib.add_cp_le_DL_terms.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(dtype=np.int32, ndim=2, flags="C_CONTIGUOUS"),
        ct.c_int,
        ct.c_int,
        ct.c_int,
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
    ]
    mata_lib.add_cp_le_DL_terms.restype = None
    num_nodes = int(cons_pot_mesh.get_nodes().shape[0])
    num_verts = int(lin_geo_mesh.get_verts().shape[0])
    num_faces = int(lin_geo_mesh.get_faces().shape[0])
    assert num_nodes == num_faces
    K = np.zeros((3 * num_faces, 3 * num_faces)).astype(np.float64)
    nodes = cons_pot_mesh.get_nodes().astype(np.float64)
    verts = lin_geo_mesh.get_verts().astype(np.float64)
    faces = lin_geo_mesh.get_faces().astype(np.int32)
    normals = lin_geo_mesh.normals.astype(np.float64)
    hs_arr = lin_geo_mesh.hs.astype(np.float64)
    mata_lib.add_cp_le_DL_terms(
        K,
        nodes,
        verts,
        faces,
        num_nodes,
        num_verts,
        num_faces,
        normals,
        hs_arr
    )
    return K


def make_mat_lp_le_cpp(lin_pot_mesh, lin_geo_mesh):
    """
    Assembles the DL operator matrix for a linear potential,
    linear geometry discretization.
    Links to C library for speed
    Parameters:
        lin_pot_mesh: linear potential mesh
        lin_geo_mesh: linear geometric mesh
    Returns:
        the stresslet matrix
    """
    mata_lib = ct.CDLL("/home/charlie/local_git/rigid_DL/c_src/matrix_assem.so")
    mata_lib.add_lp_le_DL_terms.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(dtype=np.int32, ndim=2, flags="C_CONTIGUOUS"),
        ct.c_int,
        ct.c_int,
        ct.c_int,
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
    ]
    mata_lib.add_cp_le_DL_terms.restype = None
    num_nodes = int(lin_pot_mesh.get_nodes().shape[0])
    num_verts = int(lin_geo_mesh.get_verts().shape[0])
    num_faces = int(lin_geo_mesh.get_faces().shape[0])
    assert num_nodes == num_verts
    K = np.zeros((3 * num_nodes, 3 * num_nodes)).astype(np.float64)
    nodes = lin_pot_mesh.get_nodes().astype(np.float64)
    verts = lin_geo_mesh.get_verts().astype(np.float64)
    faces = lin_geo_mesh.get_faces().astype(np.int32)
    normals = lin_geo_mesh.normals.astype(np.float64)
    hs_arr = lin_geo_mesh.hs.astype(np.float64)
    mata_lib.add_lp_le_DL_terms(
        K,
        nodes,
        verts,
        faces,
        num_nodes,
        num_verts,
        num_faces,
        normals,
        hs_arr
    )
    return K


def make_mat_cp_qe_cpp(cons_pot_mesh, quad_geo_mesh):
    """
    Assembles the DL operator matrix for a linear potential,
    linear geometry discretization.
    Links to C library for speed
    Parameters:
        cons_pot_mesh: constant potential mesh
        quad_geo_mesh: quadratic geometric mesh
    Returns:
        the stresslet matrix
    """
    return 0


def make_mat_lp_qe_cpp(lin_pot_mesh, quad_geo_mesh):
    """
    Assembles the DL operator matrix for a linear potential,
    linear geometry discretization.
    Links to C library for speed
    Parameters:
        lin_pot_mesh: linear potential mesh
        quad_geo_mesh: quadratic geometric mesh
    Returns:
        the stresslet matrix
    """
    return 0


def make_mat_qp_qe_cpp(quad_pot_mesh, quad_geo_mesh):
    """
    This version links to C library for speed
    INCOMPLETE
    """
    return 0




# all python assembly versions
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
    K = np.zeros((3 * num_faces, 3 * num_faces))
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
                K[(3 * src_num):(3 * src_num + 3),
                  (3 * face_num):(3 * face_num + 3)] += sub_mat
                K[(3 * src_num):(3 * src_num + 3),
                  (3 * src_num):(3 * src_num + 3)] -= sub_mat
            # do nothing face_num == src_num, how it works out for constant elements
    for src_num in range(num_faces):
        K[(3 * src_num):(3 * src_num + 3),
          (3 * src_num):(3 * src_num + 3)] -= 4. * np.pi * np.identity(3)
    K = np.dot(c_0, K)
    return K


def make_mat_lp_le(lin_pot_mesh, lin_geo_mesh):
    """
    Version that uses linear interpolation for normal vector
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
    K = np.zeros((3 * num_nodes, 3 * num_nodes))

    for face_num in range(num_faces): # integrate over faces
        face_nodes = lin_geo_mesh.get_tri_nodes(face_num)
        face_n = lin_geo_mesh.get_normal(face_num)
        face_hs = lin_geo_mesh.get_hs(face_num)
        for src_num in range(num_nodes): # source points
            src_pt = pot_nodes[src_num]
            is_singular, _local_singular_ind = lin_pot_mesh.check_in_face(src_num, face_num)
            if not is_singular: # regular triangle
                for node_num in range(3):
                    node_global_num = pot_faces[face_num, node_num] # global index for vert
                    sub_mat = gq.int_over_tri_lin(
                        make_reg_lp_le_quad_func(
                            face_n, src_pt, node_num
                        ),
                        face_nodes,
                        face_hs
                    )
                    K[(3 * src_num):(3 * src_num + 3),
                      (3 * node_global_num):(3 * node_global_num + 3)] += c_0 * sub_mat
                # subtracting the q(x_0) term
                sub_mat = gq.int_over_tri_lin(
                    make_cp_le_quad_func(face_n, src_pt),
                    face_nodes,
                    face_hs
                )
                K[(3 * src_num):(3 * src_num + 3), (3 * src_num):(3 * src_num + 3)] -= c_0 * sub_mat
            # always evaluates to zero for singular flat triangles
            # from \hat{x} vector being orthogonal to normal vector

    for src_num in range(num_nodes): # source points
        # whole surface q(x_0) term
        K[(3 * src_num):(3 * src_num + 3), (3 * src_num):(3 * src_num + 3)] -= c_0 * (
            4. * np.pi * np.identity(3)
        )
    return K


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
    K = np.zeros((3 * num_faces, 3 * num_faces))
    for face_num in range(num_faces): # field points
        face_nodes = quad_geo_mesh.get_tri_nodes(face_num)
        face_n = quad_geo_mesh.get_quad_n(face_num)
        face_hs = quad_geo_mesh.get_hs(face_num)
        for src_num in range(num_faces): # source points
            src_center = cons_pot_mesh.get_node(src_num)
            if face_num != src_num:
                sub_mat = gq.int_over_tri_quad_n(
                    make_cp_qe_quad_func(src_center),
                    face_nodes,
                    face_n,
                    face_hs,
                )
                K[(3 * src_num):(3 * src_num + 3),
                  (3 * face_num):(3 * face_num + 3)] += sub_mat
                K[(3 * src_num):(3 * src_num + 3),
                  (3 * src_num):(3 * src_num + 3)] -= sub_mat
            # do nothing face_num == src_num, how it works out for constant elements

    for src_num in range(num_faces):
        K[(3 * src_num):(3 * src_num + 3),
          (3 * src_num):(3 * src_num + 3)] -= 4. * np.pi * np.identity(3)
    K = np.dot(c_0, K)
    return K


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
    K = np.zeros((3 * num_nodes, 3 * num_nodes))

    for face_num in range(num_faces): # integrate over faces
        face_nodes = quad_geo_mesh.get_tri_nodes(face_num)
        face_n = quad_geo_mesh.get_quad_n(face_num)
        face_hs = quad_geo_mesh.get_hs(face_num)
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
                        face_n,
                        face_hs,
                    )
                    K[(3 * src_num):(3 * src_num + 3),
                      (3 * node_global_num):(3 * node_global_num + 3)] += sub_mat

            else: # regular triangle
                for node_num in range(3):
                    node_global_num = pot_faces[face_num, node_num] # global index for vert
                    sub_mat = gq.int_over_tri_quad_n(
                        make_reg_lp_qe_quad_func(src_pt, node_num),
                        face_nodes,
                        face_n,
                        face_hs,
                    )
                    K[(3 * src_num):(3 * src_num + 3),
                      (3 * node_global_num):(3 * node_global_num + 3)] += sub_mat
                # subtracting the q(x_0) term
                sub_mat = gq.int_over_tri_quad_n(
                    make_cp_qe_quad_func(src_pt),
                    face_nodes,
                    face_n,
                    face_hs,
                )
                K[(3 * src_num):(3 * src_num + 3), (3 * src_num):(3 * src_num + 3)] -= sub_mat

    for src_num in range(num_nodes): # source points
        # whole surface q(x_0) term
        K[(3 * src_num):(3 * src_num + 3), (3 * src_num):(3 * src_num + 3)] -= (
            4. * np.pi * np.identity(3)
        )

    K = np.dot(c_0, K)
    return K


def make_mat_qp_qe(quad_pot_mesh, quad_geo_mesh):
    """
    Makes the stiffness matrix using closed surface singularity subtraction.
    For linear potentials over quadratic elements.
    Parameters:
        quad_pot_mesh: quadratic potential mesh
        quad_geo_mesh: quadratic geometric mesh
    Returns:
        the stresslet matrix
    """
    geo_faces = quad_geo_mesh.get_faces()
    pot_faces = quad_pot_mesh.get_faces()
    assert geo_faces.shape[0] == pot_faces.shape[0]
    num_faces = geo_faces.shape[0]
    pot_nodes = quad_pot_mesh.get_nodes()
    num_nodes = pot_nodes.shape[0]
    c_0 = 1. / (4. * np.pi)
    K = np.zeros((3 * num_nodes, 3 * num_nodes))

    for face_num in range(num_faces): # integrate over faces
        face_nodes = quad_geo_mesh.get_tri_nodes(face_num)
        face_n = quad_geo_mesh.get_quad_n(face_num)
        face_hs = quad_geo_mesh.get_hs(face_num)
        for src_num in range(num_nodes): # source points
            src_pt = quad_pot_mesh.get_node(src_num)
            is_singular, local_singular_ind = quad_pot_mesh.check_in_face(src_num, face_num)

            if is_singular: # singular triangle
                for node_num in range(6):
                    node_global_num = pot_faces[face_num, node_num] # global index for vert
                    sub_mat = gq.int_over_tri_quad_n(
                        make_sing_qp_qe_quad_func(
                            src_pt, node_num, local_singular_ind
                            ),
                        face_nodes,
                        face_n,
                        face_hs,
                    )
                    K[(3 * src_num):(3 * src_num + 3),
                      (3 * node_global_num):(3 * node_global_num + 3)] += sub_mat

            else: # regular triangle
                for node_num in range(6):
                    node_global_num = pot_faces[face_num, node_num] # global index for vert
                    sub_mat = gq.int_over_tri_quad_n(
                        make_reg_qp_qe_quad_func(src_pt, node_num),
                        face_nodes,
                        face_n,
                        face_hs,
                    )
                    K[(3 * src_num):(3 * src_num + 3),
                      (3 * node_global_num):(3 * node_global_num + 3)] += sub_mat
                # subtracting the q(x_0) term
                sub_mat = gq.int_over_tri_quad_n(
                    make_cp_qe_quad_func(src_pt),
                    face_nodes,
                    face_n,
                    face_hs,
                )
                K[(3 * src_num):(3 * src_num + 3), (3 * src_num):(3 * src_num + 3)] -= sub_mat

    for src_num in range(num_nodes): # source points
        # whole surface q(x_0) term
        K[(3 * src_num):(3 * src_num + 3), (3 * src_num):(3 * src_num + 3)] -= (
            4. * np.pi * np.identity(3)
        )

    K = np.dot(c_0, K)
    return K


def make_cp_le_quad_func(n, x_0):
    """
    Makes the constant potential function that is integrated over
    linear elements for the stiffness matrix
    Parameters:
        n: unit normal vector
        x_0: the source point
    """
    def quad_func(xi, eta, nodes):
        x = geo.linear_interp(xi, eta, nodes)
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
        x = geo.quadratic_interp(xi, eta, nodes)
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
        x = geo.linear_interp(xi, eta, nodes)
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
        x = geo.quadratic_interp(xi, eta, nodes)
        S = geo.stresslet(x, x_0)
        phi = geo.shape_func_linear(xi, eta, node_num)
        return phi * S
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
        x = geo.quadratic_interp(xi, eta, nodes)
        phi = geo.shape_func_linear(xi, eta, node_num)
        if node_num == singular_ind:
            if (phi - 1) == 0: # getting around division by 0
                return np.zeros([3, 3, 3])
            else:
                if np.linalg.norm(x - x_0) < 1e-6:
                    print("nearly singular lp_qe x_hat error")
                return (phi - 1) * geo.stresslet(x, x_0)
        else:
            if phi == 0:
                return np.zeros([3, 3, 3])
            else:
                if np.linalg.norm(x - x_0) < 1e-6:
                    print("nearly singular lp_qe x_hat error")
                return (phi) * geo.stresslet(x, x_0)
    return quad_func


def make_reg_qp_qe_quad_func(x_0, node_num):
    """
    Makes the regular (non-singular) quadratic potential, quadratic element function
    that is integrated for the stiffness matrix
    Parameters:
        x_0: source point
        node_num: which potential shape function [0-5]
    """
    def quad_func(xi, eta, nodes):
        x = geo.quadratic_interp(xi, eta, nodes)
        S = geo.stresslet(x, x_0)
        phi = geo.shape_func_quadratic(xi, eta, nodes, node_num)
        return phi * S
    return quad_func


def make_sing_qp_qe_quad_func(x_0, node_num, singular_ind):
    """
    Makes the sinuglar quadratic potential, quadratic element function
    that is dotted with normal vectors and integrated for the stiffness matrix
    Parameters:
        x_0: source point
        node_num: which potential shape function [0-5]
        singular_ind: local singular index for a face
    """
    def quad_func(xi, eta, nodes):
        x = geo.quadratic_interp(xi, eta, nodes)
        phi = geo.shape_func_quadratic(xi, eta, nodes, node_num)
        if node_num == singular_ind:
            if (phi - 1) == 0: # getting around division by 0
                return np.zeros([3, 3, 3])
            else:
                if np.linalg.norm(x - x_0) < 1e-6:
                    print("nearly singular qp_qe x_hat error")
                return (phi - 1) * geo.stresslet(x, x_0)
        else:
            if phi == 0:
                return np.zeros([3, 3, 3])
            else:
                if np.linalg.norm(x - x_0) < 1e-6:
                    print("nearly singular qp_qe x_hat error")
                return (phi) * geo.stresslet(x, x_0)
    return quad_func




# Analytical Normal vector using versions
def make_mat_cp_le_NV(cons_pot_mesh, lin_geo_mesh_NV):
    """
    This version uses linear interpolation from analytical normal vectors
    at the mesh vertices
    Parameters:
        cons_pot_mesh: constant potential mesh
        lin_geo_mesh_NV : linear geometric mesh using interpolated normal vectors
    Returns:
        the stresslet matrix
    """
    pot_faces = cons_pot_mesh.get_faces()
    assert pot_faces.shape[0] == lin_geo_mesh_NV.get_faces().shape[0]
    num_faces = pot_faces.shape[0] # should be same for either pot or geo
    c_0 = 1. / (4. * np.pi)
    K = np.zeros((3 * num_faces, 3 * num_faces))
    for face_num in range(num_faces):
        face_nodes = lin_geo_mesh_NV.get_tri_nodes(face_num)
        face_normals = lin_geo_mesh_NV.get_tri_normals(face_num)
        face_hs = lin_geo_mesh_NV.get_hs(face_num)
        for src_num in range(num_faces):
            src_center = cons_pot_mesh.get_node(src_num)
            if face_num != src_num:
                sub_mat = gq.int_over_tri_lin(
                    make_cp_le_NV_quad_func(face_normals, src_center),
                    face_nodes,
                    face_hs
                )
                K[(3 * src_num):(3 * src_num + 3),
                  (3 * face_num):(3 * face_num + 3)] += sub_mat
                K[(3 * src_num):(3 * src_num + 3),
                  (3 * src_num):(3 * src_num + 3)] -= sub_mat
            # do nothing face_num == src_num, how it works out for constant elements
    for src_num in range(num_faces):
        K[(3 * src_num):(3 * src_num + 3),
          (3 * src_num):(3 * src_num + 3)] -= 4. * np.pi * np.identity(3)
    K = np.dot(c_0, K)
    return K


def make_mat_lp_le_NV(lin_pot_mesh, lin_geo_mesh_NV):
    """
    Makes the stiffness matrix using closed surface singularity subtraction.
    For linear potentials over a linear elements.
    Parameters:
        lin_pot_mesh: linear potential mesh
        lin_geo_mesh_NV : linear geometric mesh using interpolated normal vectors
    Returns:
        the stresslet matrix
    """
    geo_faces = lin_geo_mesh_NV.get_faces()
    pot_faces = lin_pot_mesh.get_faces()
    assert geo_faces.shape[0] == pot_faces.shape[0]
    num_faces = geo_faces.shape[0]
    pot_nodes = lin_pot_mesh.get_nodes()
    num_nodes = pot_nodes.shape[0]
    c_0 = 1. / (4. * np.pi)
    K = np.zeros((3 * num_nodes, 3 * num_nodes))

    for face_num in range(num_faces): # integrate over faces
        face_nodes = lin_geo_mesh_NV.get_tri_nodes(face_num)
        face_normals = lin_geo_mesh_NV.get_tri_normals(face_num)
        face_hs = lin_geo_mesh_NV.get_hs(face_num)
        for src_num in range(num_nodes): # source points
            src_pt = pot_nodes[src_num]
            is_singular, local_singular_ind = lin_pot_mesh.check_in_face(src_num, face_num)
            if not is_singular: # regular triangle
                for node_num in range(3):
                    node_global_num = pot_faces[face_num, node_num] # global index for vert
                    sub_mat = gq.int_over_tri_lin(
                        make_reg_lp_le_NV_quad_func(
                            face_normals, src_pt, node_num
                        ),
                        face_nodes,
                        face_hs
                    )
                    K[(3 * src_num):(3 * src_num + 3),
                      (3 * node_global_num):(3 * node_global_num + 3)] += c_0 * sub_mat
                # subtracting the q(x_0) term
                sub_mat = gq.int_over_tri_lin(
                    make_cp_le_NV_quad_func(face_normals, src_pt),
                    face_nodes,
                    face_hs
                )
                K[(3 * src_num):(3 * src_num + 3), (3 * src_num):(3 * src_num + 3)] -= c_0 * sub_mat
            else: # singular triangle
                for node_num in range(3):
                    node_global_num = pot_faces[face_num, node_num] # global index for vert
                    sub_mat = gq.int_over_tri_lin(
                        make_sing_lp_le_NV_quad_func(
                            face_normals, src_pt, node_num, local_singular_ind
                        ),
                        face_nodes,
                        face_hs
                    )
                    K[(3 * src_num):(3 * src_num + 3),
                      (3 * node_global_num):(3 * node_global_num + 3)] += c_0 * sub_mat

    for src_num in range(num_nodes): # source points
        # whole surface q(x_0) term
        K[(3 * src_num):(3 * src_num + 3), (3 * src_num):(3 * src_num + 3)] -= c_0 * (
            4. * np.pi * np.identity(3)
        )
    return K


def make_cp_le_NV_quad_func(normals, x_0):
    """
    This version uses linear intepolation for the normal vector
    Parameters:
        normals: (3,3) ndarray of normal vectors at element verticies
        x_0: the source point
    """
    def quad_func(xi, eta, nodes):
        x = geo.linear_interp(xi, eta, nodes)
        n = geo.linear_interp(xi, eta, normals)
        return geo.stresslet_n(x, x_0, n)
    return quad_func


def make_reg_lp_le_NV_quad_func(normals, x_0, node_num):
    """
    This version for using linear interpolated normal vector
    Parameters:
        normals: (3,3) ndarray of normal vectors at element veritices as rows
        x_0: source point
        node_num: which potential shape function [0, 1, 2]
    """
    def quad_func(xi, eta, nodes):
        x = geo.linear_interp(xi, eta, nodes)
        n = geo.linear_interp(xi, eta, normals)
        S = geo.stresslet_n(x, x_0, n)
        phi = geo.shape_func_linear(xi, eta, node_num)
        return phi * S
    return quad_func


def make_sing_lp_le_NV_quad_func(normals, x_0, node_num, singular_ind):
    """
    Normal vector will be linearly interpolated, therefore singular
    elements can be non-zero now
    Parameters:
        normals: (3,3) ndarray of normal vectors at element vertices as rows
        x_0: source point
        node_num: which potential shape function [0, 1, 2]
        singular_ind: local singular index
    """
    def quad_func(xi, eta, nodes):
        x = geo.linear_interp(xi, eta, nodes)
        n = geo.linear_interp(xi, eta, normals)
        phi = geo.shape_func_linear(xi, eta, node_num)
        if node_num == singular_ind:
            if (phi - 1) == 0: # getting around division by 0
                return np.zeros([3, 3])
            else:
                if np.linalg.norm(x - x_0) < 1e-6:
                    print("nearly singular lp_le_NV x_hat error")
                return (phi - 1) * geo.stresslet_n(x, x_0, n)
        else:
            if phi == 0:
                return np.zeros([3, 3])
            else:
                if np.linalg.norm(x - x_0) < 1e-6:
                    print("nearly singular lp_le_NV x_hat error")
                return (phi) * geo.stresslet_n(x, x_0, n)
    return quad_func
