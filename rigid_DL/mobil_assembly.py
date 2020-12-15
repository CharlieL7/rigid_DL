"""
Matrix assembly for the mobility problem
"""
import numpy as np
import rigid_DL.gauss_quad as gq
import rigid_DL.geometric as geo
import rigid_DL.mobil_helper as mobil_helper

def make_mat_cp_le(cons_pot_mesh, lin_geo_mesh):
    """
    Mobility problem.
    Makes the stiffness matrix using closed surface singularity subtraction.
    For constant potentials over linear elements.
    Parameters:
        cons_pot_mesh: constant potential mesh
        lin_geo_mesh : linear geometric mesh
    Returns:
        the stresslet matrix
    """
    pot_faces = cons_pot_mesh.get_faces()
    assert pot_faces.shape[0] == lin_geo_mesh.get_faces().shape[0]
    num_faces = pot_faces.shape[0]
    K = np.zeros((3 * num_faces, 3 * num_faces))
    add_cp_le_DL_terms(K, cons_pot_mesh, lin_geo_mesh)
    add_cp_le_RBM_terms(K, cons_pot_mesh, lin_geo_mesh)
    return K


def add_cp_le_DL_terms(K, cons_pot_mesh, lin_geo_mesh):
    """
    Make DL terms for constant potential, linear elements mesh
    $-\frac{1}{4\pi} \Int{q_j(\bm{x}) T_{jik}(\bm{x}, \bm{x}_0) n_k(\bm{x})}{S(\bm{x}), D, PV}$
    Parameters:
        K: stiffness matrix to add terms to (3 * face_num, 3 * face_num) ndarray
        cons_pot_mesh: constant potential mesh
        lin_geo_mesh : linear geometric mesh
    Returns:
        None
    """
    c_0 = 1. / (4. * np.pi)
    num_faces = cons_pot_mesh.get_faces().shape[0]
    for face_num in range(num_faces):
        face_nodes = lin_geo_mesh.get_tri_nodes(face_num)
        face_n = lin_geo_mesh.get_normal(face_num)
        face_hs = lin_geo_mesh.get_hs(face_num)
        for src_num in range(num_faces):
            src_center = cons_pot_mesh.get_node(src_num)
            if face_num != src_num:
                sub_mat = gq.int_over_tri_lin(
                    make_DL_cp_le_quad_func(face_n, src_center),
                    face_nodes,
                    face_hs
                )
                K[(3 * src_num):(3 * src_num + 3),
                  (3 * face_num):(3 * face_num + 3)] += c_0 * sub_mat
                K[(3 * src_num):(3 * src_num + 3),
                  (3 * src_num):(3 * src_num + 3)] -= c_0 * sub_mat
            # do nothing face_num == src_num, how it works out for constant elements
    for src_num in range(num_faces):
        K[(3 * src_num):(3 * src_num + 3),
          (3 * src_num):(3 * src_num + 3)] += c_0 * -4. * np.pi * np.identity(3)


def add_cp_le_RBM_terms(K, cons_pot_mesh, lin_geo_mesh):
    """
    Add rigid body motion terms to the given stiffness matrix K.
    For constant potential and linear elements.
    Parameters:
        K: stiffness matrix to add terms to (3 * face_num, 3 * face_num) ndarray
        cons_pot_mesh: constant potential mesh
        lin_geo_mesh : linear geometric mesh
    Returns:
        None
    """
    num_faces = cons_pot_mesh.get_faces().shape[0]
    x_c = lin_geo_mesh.get_centroid()
    w = np.identity(3)
    A_m = mobil_helper.calc_le_Am_vec(lin_geo_mesh)
    print("A_m:")
    print(A_m)
    S_D = lin_geo_mesh.get_surface_area()

    for face_num in range(num_faces):
        face_nodes = lin_geo_mesh.get_tri_nodes(face_num)
        face_hs = lin_geo_mesh.get_hs(face_num)
        v_sub_mat = (-4. * np.pi / S_D) * (np.identity(3) * 0.5 * face_hs) # simple integral
        def omega_quad(xi, eta, nodes):
            pos = geo.pos_linear(xi, eta, nodes)
            X = pos - x_c
            return np.einsum("lrs,r", geo.LC_3, X)
        tmp_omega = gq.int_over_tri_lin(
            omega_quad,
            face_nodes,
            face_hs,
        )
        tmp_arr = []
        for m in range(3):
            tmp_arr.append((1./ A_m[m]) * np.einsum("j,l,ls", w[m], w[m], tmp_omega))
        tmp_omega_mat = -4. * np.pi * np.sum(tmp_arr, axis=0)
        for src_num in range(num_faces):
            K[(3 * src_num):(3 * src_num + 3),
              (3 * face_num):(3 * face_num + 3)] += -1. / (4. * np.pi) * v_sub_mat
            src_center = cons_pot_mesh.get_node(src_num)
            X_0 = src_center - x_c
            omega_mat = np.einsum("ijk,js,k->is", geo.LC_3, tmp_omega_mat, X_0)
            K[(3 * src_num):(3 * src_num + 3),
              (3 * face_num):(3 * face_num + 3)] += -1. / (4. * np.pi) * omega_mat # error?


def add_cp_le_n_terms(K, cons_pot_mesh, lin_geo_mesh):
    """
    Add the potential dotted with normal vector terms to remove to +1 eigenvalue
    from the DL operator.
    # $\frac{1}{S_D} n_i(\bm{x}_0) \Int{q_j(\bm{x}) n_j(\bm{x})}{S(\bm{x}), D}$
    Used for the successive substitutions method.
    Parameters:
        K: stiffness matrix to add terms to (3 * face_num, 3 * face_num) ndarray
        cons_pot_mesh: constant potential mesh
        lin_geo_mesh : linear geometric mesh
    Returns:
        None
    """
    num_faces = cons_pot_mesh.get_faces().shape[0]
    S_D = lin_geo_mesh.get_surface_area()
    for face_num in range(num_faces):
        face_n = lin_geo_mesh.get_normal(face_num)
        face_hs = lin_geo_mesh.get_hs(face_num)
        for src_num in range(num_faces):
            src_n = lin_geo_mesh.get_normal(src_num)
            sub_mat = np.einsum("i,j->ij",
                src_n,
                face_n * face_hs * 0.5
            )
            K[(3 * src_num):(3 * src_num + 3),
              (3 * face_num):(3 * face_num + 3)] += (-1. / S_D) * sub_mat


def make_cp_le_forcing_vec_SS(cons_pot_mesh, lin_geo_mesh, u_d, f, l, mu):
    """
    Makes the forcing vector ( f ) for the mobility problem given a
    constant potential mesh and linear geometric mesh.
    For the successive substitutions iterative method.
    Parameters:
        cons_pot_mesh: potential mesh
        lin_geo_mesh: geometric mesh
        u_d: disturbance velocity; (3 * N,) ndarray
        f: force on particle; (3,) ndarray
        l: torque on particle; (3,) ndarray
        mu: fluid viscosity; scalar
    Returns:
        fv: forcing vector (3 * N,) ndarray
    """
    pot_faces = cons_pot_mesh.get_faces()
    assert pot_faces.shape[0] == lin_geo_mesh.get_faces().shape[0]
    num_faces = pot_faces.shape[0]

    x_c = lin_geo_mesh.get_centroid()
    c_0 = 1. / (4. * np.pi)

    # make Power and Miranda supplementary flow vector
    f_s = f / (-8. * np.pi * mu) # the script F seen in Pozrikidis
    l_s = l / (-8. * np.pi * mu) # the script L seen in Pozrikidis
    v_s = np.empty(3 * num_faces)
    for src_num in range(num_faces):
        node = cons_pot_mesh.get_node(src_num)
        v_s[(3 * src_num) : (3 * src_num + 3)] = np.einsum(
            "il,l->i", geo.stokeslet(node, x_c), f_s
        ) + np.einsum(
            "il,l->i", geo.rotlet(node, x_c), l_s
        )
    c_s = c_0 * (u_d - v_s) # script C term from Pozrikidis
    fv = np.copy(c_s) # must copy

    # make integral of c_s dotted with normal vector term
    S_D = lin_geo_mesh.get_surface_area()
    for face_num in range(num_faces):
        face_n = lin_geo_mesh.get_normal(face_num)
        face_hs = lin_geo_mesh.get_hs(face_num)
        for src_num in range(num_faces):
            src_n = lin_geo_mesh.get_normal(src_num)
            # setting c_s as constant over element
            j = 3 * face_num
            k = 3 * src_num
            sub_vec = src_n * np.dot(c_s[j : j+3], face_n) * face_hs * 0.5
            fv[k : k+3] += (-1. / (2. * S_D)) * sub_vec
    return fv


def make_forcing_vec(pot_mesh, geo_mesh, u_d, f, l, mu):
    """
    Makes the forcing vector ( f ) for the mobility problem given a
    constant potential mesh and linear geometric mesh.
    For the direct solution method.
    Parameters:
        pot_mesh: potential mesh
        geo_mesh: geometric mesh
        u_d: disturbance velocity; (3 * N,) ndarray
        f: force on particle; (3,) ndarray
        l: torque on particle; (3,) ndarray
        mu: fluid viscosity; scalar
    Returns:
        fv: forcing vector (3 * N,) ndarray
    """
    pot_nodes = pot_mesh.get_nodes()
    num_nodes = pot_nodes.shape[0]

    x_c = geo_mesh.get_centroid()
    c_0 = 1. / (4. * np.pi)

    # make Power and Miranda supplementary flow vector
    f_s = f / (-8. * np.pi * mu) # the script F seen in Pozrikidis
    l_s = l / (-8. * np.pi * mu) # the script L seen in Pozrikidis
    v_s = np.empty(3 * num_nodes)
    for src_num in range(num_nodes):
        node = pot_nodes[src_num]
        v_s[(3 * src_num) : (3 * src_num + 3)] = np.einsum(
            "il,l->i", geo.stokeslet(node, x_c), f_s
        ) + np.einsum(
            "il,l->i", geo.rotlet(node, x_c), l_s
        )
    fv = c_0 * (u_d - v_s) # script C term from Pozrikidis
    return fv


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


def make_mat_lp_le(lin_pot_mesh, lin_geo_mesh):
    """
    Mobility problem.
    Makes the stiffness matrix using closed surface singularity subtraction.
    For constant potentials over linear elements.
    Parameters:
        lin_pot_mesh: linear potential mesh
        lin_geo_mesh: linear geometric mesh
    Returns:
        the stresslet matrix
    """
    num_nodes = lin_pot_mesh.get_nodes().shape[0]
    K = np.zeros((3 * num_nodes, 3 * num_nodes))
    add_lp_le_DL_terms(K, lin_pot_mesh, lin_geo_mesh)
    add_lp_le_RBM_terms(K, lin_pot_mesh, lin_geo_mesh)
    return K


def add_lp_le_DL_terms(K, lin_pot_mesh, lin_geo_mesh):
    """
    Make DL terms for linear potential, linear elements mesh
    $-\frac{1}{4\pi} \Int{q_j(\bm{x}) T_{jik}(\bm{x}, \bm{x}_0) n_k(\bm{x})}{S(\bm{x}), D, PV}$
    Parameters:
        K: stiffness matrix to add terms to (3 * node_num, 3 * node_num) ndarray
        lin_pot_mesh: constant potential mesh
        lin_geo_mesh : linear geometric mesh
    Returns:
        None
    """
    c_0 = 1. / (4. * np.pi)
    num_faces = lin_pot_mesh.get_faces().shape[0]
    pot_faces = lin_pot_mesh.get_faces()
    pot_nodes = lin_pot_mesh.get_nodes()
    num_nodes = pot_nodes.shape[0]
    for face_num in range(num_faces): # integrate over faces
        face_nodes = lin_geo_mesh.get_tri_nodes(face_num)
        face_n = lin_geo_mesh.get_normal(face_num)
        face_hs = lin_geo_mesh.get_hs(face_num)
        for src_num in range(num_nodes): # source points
            src_pt = pot_nodes[src_num]
            is_singular, _local_singular_ind = lin_pot_mesh.check_in_face(src_num, face_num)
            if not is_singular: # regular triangle
                for node_num in range(3): # over 3 triangle nodes
                    face_node_global_num = pot_faces[face_num, node_num] # global index for vert
                    sub_mat = gq.int_over_tri_lin(
                        make_DL_reg_lp_le_quad_func(
                            face_n, src_pt, node_num
                        ),
                        face_nodes,
                        face_hs
                    )
                    j = 3 * face_node_global_num
                    K[(3 * src_num):(3 * src_num + 3),
                      j:j + 3] += c_0 * sub_mat
                # subtracting the q(x_0) term
                sub_mat = gq.int_over_tri_lin(
                    make_DL_cp_le_quad_func(face_n, src_pt),
                    face_nodes,
                    face_hs
                )
                K[(3 * src_num):(3 * src_num + 3), (3 * src_num):(3 * src_num + 3)] -= c_0 * sub_mat
            # for singular elements, \hat{x} dotted with normal vector always zero
            # due to flat triangular elements

    for src_num in range(num_nodes): # source points
        # whole surface q(x_0) term
        j = 3 * src_num
        K[j:j+3, j:j+3] -= c_0 * (
            4. * np.pi * np.identity(3)
        )


def add_lp_le_RBM_terms(K, lin_pot_mesh, lin_geo_mesh):
    """
    Add rigid body motion terms to the given stiffness matrix K.
    For linear potential and linear elements.
    Parameters:
        K: stiffness matrix to add terms to (3 * node_num, 3 * node_num) ndarray
        lin_pot_mesh: constant potential mesh
        lin_geo_mesh : linear geometric mesh
    Returns:
        None
    """
    pot_faces = lin_pot_mesh.get_faces()
    num_faces = pot_faces.shape[0]
    pot_nodes = lin_pot_mesh.get_nodes()
    num_nodes = pot_nodes.shape[0]
    S_D = lin_geo_mesh.get_surface_area()
    x_c = lin_geo_mesh.get_centroid()
    w = np.identity(3)
    A_m = mobil_helper.calc_le_Am_vec(lin_geo_mesh)

    for face_num in range(num_faces):
        face_nodes = lin_geo_mesh.get_tri_nodes(face_num)
        face_hs = lin_geo_mesh.get_hs(face_num)
        for node_num in range(3): # face nodes
            face_node_global_num = pot_faces[face_num, node_num]
            v_sub_mat = (-4. * np.pi / S_D) * gq.int_over_tri_lin(
                make_lp_le_v_quad(node_num),
                face_nodes,
                face_hs,
            )
            j = 3 * face_node_global_num
            tmp_omega = gq.int_over_tri_lin(
                make_lp_le_omega_quad(node_num, x_c),
                face_nodes,
                face_hs,
            )
            tmp_arr = []
            for m in range(3):
                tmp_arr.append((1./ A_m[m]) * np.einsum("j,l,ls", w[m], w[m], tmp_omega))
            tmp_omega_mat = -4. * np.pi * np.sum(tmp_arr, axis=0)
            for src_num in range(num_nodes):
                K[(3 * src_num):(3 * src_num + 3),
                  j:j+3] += -1. / (4. * np.pi) * v_sub_mat
                src_pt = lin_pot_mesh.get_node(src_num)
                X_0 = src_pt - x_c
                omega_mat = np.einsum("ijk,js,k", geo.LC_3, tmp_omega_mat, X_0)
                K[(3 * src_num):(3 * src_num + 3),
                  j:j+3] += 1. / (4. * np.pi) * (omega_mat) #error?


def make_DL_reg_lp_le_quad_func(n, x_0, node_num):
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


def make_lp_le_v_quad(node_num):
    def v_quad(xi, eta, _nodes):
        phi = geo.shape_func_linear(xi, eta, node_num)
        return phi * np.identity(3)
    return v_quad


def make_lp_le_omega_quad(node_num, x_c):
    def omega_quad(xi, eta, nodes):
        pos = geo.pos_linear(xi, eta, nodes)
        X = pos - x_c
        phi = geo.shape_func_linear(xi, eta, node_num)
        return phi * np.einsum("lrs,r", geo.LC_3, X)
    return omega_quad


def make_lp_le_forcing_vec(lin_pot_mesh, lin_geo_mesh, u_d, f, l, mu):
    """
    Makes the forcing vector ( f ) for the mobility problem
    For the direct solution method.
    Parameters:
        lin_pot_mesh: potential mesh
        lin_geo_mesh: geometric mesh
        u_d: disturbance velocity; (3 * N,) ndarray
        f: force on particle; (3,) ndarray
        l: torque on particle; (3,) ndarray
        mu: fluid viscosity; scalar
    Returns:
        fv: forcing vector (3 * N,) ndarray
    """
    pot_nodes = lin_pot_mesh.get_nodes()
    num_nodes = pot_nodes.shape[0]

    x_c = lin_geo_mesh.get_centroid()
    c_0 = 1. / (4. * np.pi)

    # make Power and Miranda supplementary flow vector
    f_s = f / (-8. * np.pi * mu) # the script F seen in Pozrikidis
    l_s = l / (-8. * np.pi * mu) # the script L seen in Pozrikidis
    v_s = np.empty(3 * num_nodes)
    for src_num in range(num_nodes):
        node = lin_pot_mesh.get_node(src_num)
        v_s[(3 * src_num) : (3 * src_num + 3)] = np.einsum(
            "il,l->i", geo.stokeslet(node, x_c), f_s
        ) + np.einsum(
            "il,l->i", geo.rotlet(node, x_c), l_s
        )
    fv = c_0 * (u_d - v_s) # script C term from Pozrikidis
    return fv

