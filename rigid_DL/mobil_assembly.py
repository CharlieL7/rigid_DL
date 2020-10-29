"""
Matrix assembly for the mobility problem
"""
import numpy as np
import rigid_DL.gauss_quad as gq
import rigid_DL.geometric as geo

def make_mat_cp_le(cons_pot_mesh, lin_geo_mesh):
    """
    Mobility problem.
    Makes the stiffness matrix using closed surface singularity subtraction.
    For constant potentials over a linear elements.

    Parameters:
        cons_pot_mesh: constant potential mesh
        lin_geo_mesh : linear geometric mesh
    Returns:
        the stresslet matrix
    """
    pot_faces = cons_pot_mesh.get_faces()
    assert pot_faces.shape[0] == lin_geo_mesh.get_faces().shape[0]
    num_faces = pot_faces.shape[0]
    c_0 = 1. / (4. * np.pi)
    K = np.zeros((3 * num_faces, 3 * num_faces))

    # Make DL terms
    # $-\frac{1}{4\pi} \Int{q_j(\bm{x}) T_{jik}(\bm{x}, \bm{x}_0) n_k(\bm{x})}{S(\bm{x}), D, PV}$
    for face_num in range(num_faces):
        face_nodes = lin_geo_mesh.get_tri_nodes(face_num)
        face_n = lin_geo_mesh.get_normal(face_num)
        face_hs = lin_geo_mesh.get_hs(face_num)
        face_unit_n = face_n / face_hs
        for src_num in range(num_faces):
            src_center = cons_pot_mesh.get_node(src_num)
            if face_num != src_num:
                sub_mat = gq.int_over_tri_lin(
                    make_DL_cp_le_quad_func(face_unit_n, src_center),
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

    # Make potential dotted with normal vector terms
    # $\frac{1}{S_D} n_i(\bm{x}_0) \Int{q_j(\bm{x}) n_j(\bm{x})}{S(\bm{x}), D}$
    S_D = lin_geo_mesh.get_surface_area()
    c_1 = 1. / S_D
    for face_num in range(num_faces):
        face_nodes = lin_geo_mesh.get_tri_nodes(face_num)
        face_n = lin_geo_mesh.get_normal(face_num)
        face_hs = lin_geo_mesh.get_hs(face_num)
        face_unit_n = face_n / face_hs
        for src_num in range(num_faces):
            src_n = lin_geo_mesh.get_normal(src_num)
            def n_quad(_xi, _eta, _nodes):
                return face_n
            sub_mat = np.einsum("i,j->ij", src_n, gq.int_over_tri_lin(
                n_quad,
                face_nodes,
                face_hs,
                )
            )
        K[(3 * src_num):(3 * src_num+ 3),
          (3 * face_num):(3 * face_num + 3)] += c_1 * sub_mat

    # Make rigid body motion terms
    lc_3 = np.zeros((3, 3)) # 3D Levi_Civita Tensor
    lc_3[[0, 1, 2], [1, 2, 0], [2, 0, 1]] = 1.
    lc_3[[0, 2, 1], [1, 0, 2], [2, 1, 0]] = -1.

    w = lin_geo_mesh.calc_rotation_vectors()
    A_m = np.zeros(3)
    for m in range(3):
        def make_am_quad_func(w_m):
            def am_quad(xi, eta, nodes):
                pos = geo.pos_linear(xi, eta, nodes)
                X = pos - x_c
                return np.cross(w_m, X) @ np.cross(w_m, X)
            return am_quad
        for face_num in range(num_faces): # get A_m terms
            face_nodes = lin_geo_mesh.get_tri_nodes(face_num)
            face_n = lin_geo_mesh.get_normal(face_num)
            face_hs = lin_geo_mesh.get_hs(face_num)
            face_unit_n = face_n / face_hs
            A_m[m] += gq.int_over_tri_lin(
                make_am_quad_func(w[m]),
                face_nodes,
                face_hs,
            )

    x_c = lin_geo_mesh.get_centroid(face_num)
    for face_num in range(num_faces):
        face_nodes = lin_geo_mesh.get_tri_nodes(face_num)
        face_n = lin_geo_mesh.get_normal(face_num)
        face_hs = lin_geo_mesh.get_hs(face_num)
        face_unit_n = face_n / face_hs
        def v_quad(_xi, _eta, _nodes):
            return 1.
        v_sub_mat = np.identity(3) * gq.int_over_tri_lin(
            v_quad,
            face_nodes,
            face_hs,
        ) # added to the whole column
        for src_num in range(num_faces):
            K[(3 * src_num):(3 * src_num + 3),
              (3 * face_num):(3 * face_num + 3)] += (-1. / S_D) * v_sub_mat
            src_center = cons_pot_mesh.get_node(src_num)
            def omega_quad(xi, eta, nodes):
                pos = geo.pos_linear(xi, eta, nodes)
                X = pos - x_c
                return np.einsum("lrs,r->ls", lc_3, X)
            tmp_omega = gq.int_over_tri_lin(
                omega_quad,
                face_nodes,
                face_hs,
            )
            tmp_arr = []
            for m in range(3):
                tmp_arr.append((1./ A_m[m]) * np.einsum("j,l,ls->js", w[m], w[m], tmp_omega))
            omega_mat = np.sum(tmp_arr) * -4. * np.pi
            X_0 = src_center - x_c
            omega_mat = np.einsum("ijk,js,k->is", lc_3, omega_mat, X_0)
            K[(3 * src_num):(3 * src_num + 3),
              (3 * face_num):(3 * face_num + 3)] -= omega_mat
    return K


def make_init_sol_vector():
    """
    Makes the inital solution vector of {q, v, omega} for the mobility problem.
    Sets random values for usage in method of successive substitutions.

    Parameters:

    Returns:

    """
    #TODO


def make_forcing_vector():
    """
    Makes the forcing vector of {c, 0, 0} for the mobility problem

    Parameters:

    Returns:

    """
    #TODO


def make_DL_cp_le_quad_func(n, x_0):
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
