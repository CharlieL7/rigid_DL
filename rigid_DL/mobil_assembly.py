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
    w = lin_geo_mesh.get_w()
    A_m = lin_geo_mesh.get_A_m()
    S_D = lin_geo_mesh.get_surface_area()

    for face_num in range(num_faces):
        face_nodes = lin_geo_mesh.get_tri_nodes(face_num)
        face_hs = lin_geo_mesh.get_hs(face_num)
        v_sub_mat = (1. / S_D) * (np.identity(3) * 0.5 * face_hs) # simple integral
        def omega_quad(xi, eta, nodes):
            pos = geo.linear_interp(xi, eta, nodes)
            X = pos - x_c
            return np.einsum("lrs,s->lr", geo.LC_3, X)
        tmp_omega = gq.int_over_tri_lin(
            omega_quad,
            face_nodes,
            face_hs,
        )
        tmp_arr = []
        for m in range(3):
            tmp_arr.append((1./ A_m[m]) * np.outer(w[m], np.einsum("l,ls", w[m], tmp_omega)))
        tmp_arr = np.array(tmp_arr)
        tmp_omega_mat = np.sum(tmp_arr, axis=0)
        for src_num in range(num_faces):
            K[(3 * src_num):(3 * src_num + 3),
              (3 * face_num):(3 * face_num + 3)] += v_sub_mat
            #src_center = cons_pot_mesh.get_node(src_num)
            src_center = lin_geo_mesh.get_tri_center(src_num)
            X_0 = src_center - x_c
            omega_mat = np.einsum("ijk,js,k->is", geo.LC_3, tmp_omega_mat, X_0)
            K[(3 * src_num):(3 * src_num + 3),
              (3 * face_num):(3 * face_num + 3)] += omega_mat


def add_cp_le_RBM_terms_alt(K, cons_pot_mesh, lin_geo_mesh):
    """
    Alternative using np.cross()
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
    w = lin_geo_mesh.get_w()
    A_m = lin_geo_mesh.get_A_m()
    S_D = lin_geo_mesh.get_surface_area()

    for face_num in range(num_faces):
        face_nodes = lin_geo_mesh.get_tri_nodes(face_num)
        face_hs = lin_geo_mesh.get_hs(face_num)
        face_center = lin_geo_mesh.get_tri_center(face_num)
        v_sub_mat = (1. / S_D) * (np.identity(3) * 0.5 * face_hs) # simple integral
        tmp_omega = []
        for m in range(3):
            def make_omega_quad(w, A_m):
                def omega_quad(xi, eta, nodes):
                    pos = geo.linear_interp(xi, eta, nodes)
                    X = pos - x_c
                    return np.cross(w, X)
                return omega_quad
            tmp_omega.append(gq.int_over_tri_lin(
                make_omega_quad(w[m], A_m[m]),
                face_nodes,
                face_hs,
            ))
        for src_num in range(num_faces):
            K[(3 * src_num):(3 * src_num + 3),
              (3 * face_num):(3 * face_num + 3)] += v_sub_mat
            src_center = cons_pot_mesh.get_node(src_num)
            X_0 = src_center - x_c
            tmp_arr = []
            for m in range(3):
                tmp_arr.append(np.outer((1./ A_m[m]) * np.cross(w[m], X_0), tmp_omega[m]))
            tmp_arr = np.array(tmp_arr)
            omega_mat = np.sum(tmp_arr, axis=0)
            K[(3 * src_num):(3 * src_num + 3),
              (3 * face_num):(3 * face_num + 3)] += omega_mat


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
    Makes the forcing vector ( f ) for the mobility problem
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
    c_0 = -1. / (4. * np.pi)

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
        x = geo.linear_interp(xi, eta, nodes)
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
    w = lin_geo_mesh.get_w()
    A_m = lin_geo_mesh.get_A_m()

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
                tmp_arr.append((1./ A_m[m]) * np.outer(w[m], np.einsum("l, ls", w[m], tmp_omega)))
            tmp_omega_mat = -4. * np.pi * np.sum(tmp_arr, axis=0)
            for src_num in range(num_nodes):
                K[(3 * src_num):(3 * src_num + 3),
                  j:j+3] += -1. / (4. * np.pi) * v_sub_mat
                src_pt = lin_pot_mesh.get_node(src_num)
                X_0 = src_pt - x_c
                omega_mat = np.einsum("ijk,js,k", geo.LC_3, tmp_omega_mat, X_0)
                K[(3 * src_num):(3 * src_num + 3),
                  j:j+3] += 1. / (4. * np.pi) * (omega_mat)


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
        x = geo.linear_interp(xi, eta, nodes)
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
        pos = geo.linear_interp(xi, eta, nodes)
        X = pos - x_c
        phi = geo.shape_func_linear(xi, eta, node_num)
        return phi * np.einsum("lrs,r", geo.LC_3, X)
    return omega_quad


def make_mat_cp_qe(cons_pot_mesh, quad_geo_mesh):
    """
    Mobility problem.
    Makes the stiffness matrix using closed surface singularity subtraction.
    For constant potentials over quadratic elements.
    Parameters:
        cons_pot_mesh: constant potential mesh
        quad_geo_mesh: quadratic geometric mesh
    Returns:
        the stresslet matrix
    """
    pot_faces = cons_pot_mesh.get_faces()
    assert pot_faces.shape[0] == quad_geo_mesh.get_faces().shape[0]
    num_faces = pot_faces.shape[0]
    K = np.zeros((3 * num_faces, 3 * num_faces))
    add_cp_qe_DL_terms(K, cons_pot_mesh, quad_geo_mesh)
    add_cp_qe_RBM_terms(K, cons_pot_mesh, quad_geo_mesh)
    return K


def add_cp_qe_DL_terms(K, cons_pot_mesh, quad_geo_mesh):
    """
    Make DL terms for constant potential, quadratic elements mesh
    $-\frac{1}{4\pi} \Int{q_j(\bm{x}) T_{jik}(\bm{x}, \bm{x}_0) n_k(\bm{x})}{S(\bm{x}), D, PV}$
    Parameters:
        K: stiffness matrix to add terms to (3 * face_num, 3 * face_num) ndarray
        cons_pot_mesh: constant potential mesh
        quad_geo_mesh : quadratic geometric mesh
    Returns:
        None
    """
    geo_faces = quad_geo_mesh.get_faces()
    pot_faces = cons_pot_mesh.get_faces()
    assert geo_faces.shape[0] == pot_faces.shape[0]
    num_faces = geo_faces.shape[0]
    c_0 = 1. / (4. * np.pi)
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
    K *= c_0


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


def add_cp_qe_RBM_terms(K, cons_pot_mesh, quad_geo_mesh):
    """
    Add rigid body motion terms to the given stiffness matrix K.
    For constant potential and quadratic elements.
    Parameters:
        K: stiffness matrix to add terms to (3 * face_num, 3 * face_num) ndarray
        cons_pot_mesh: constant potential mesh
        quad_geo_mesh: quadratic geometric mesh
    Returns:
        None
    """
    num_faces = cons_pot_mesh.get_faces().shape[0]
    x_c = quad_geo_mesh.get_centroid()
    w = quad_geo_mesh.get_w()
    A_m = quad_geo_mesh.get_A_m()
    S_D = quad_geo_mesh.get_surface_area()

    for face_num in range(num_faces):
        face_nodes = quad_geo_mesh.get_tri_nodes(face_num)
        face_hs = quad_geo_mesh.get_hs(face_num)
        def v_quad(_xi, _eta, _nodes):
            return np.identity(3)
        v_sub_mat = (1. / S_D) * gq.int_over_tri_quad(v_quad, face_nodes, face_hs)
        def omega_quad(xi, eta, nodes):
            pos = geo.quadratic_interp(xi, eta, nodes)
            X = pos - x_c
            return np.einsum("lrs,s->lr", geo.LC_3, X)
        tmp_omega = gq.int_over_tri_quad(
            omega_quad,
            face_nodes,
            face_hs,
        )
        tmp_arr = []
        for m in range(3):
            tmp_arr.append((1./ A_m[m]) * np.outer(w[m], np.einsum("l,ls", w[m], tmp_omega)))
        tmp_arr = np.array(tmp_arr)
        tmp_omega_mat = np.sum(tmp_arr, axis=0)
        for src_num in range(num_faces):
            K[(3 * src_num):(3 * src_num + 3),
              (3 * face_num):(3 * face_num + 3)] += v_sub_mat
            src_center = cons_pot_mesh.get_node(src_num)
            X_0 = src_center - x_c
            omega_mat = np.einsum("ijk,js,k->is", geo.LC_3, tmp_omega_mat, X_0)
            K[(3 * src_num):(3 * src_num + 3),
              (3 * face_num):(3 * face_num + 3)] += omega_mat


def make_mat_lp_qe(lin_pot_mesh, quad_geo_mesh):
    """
    Mobility problem.
    Makes the stiffness matrix using closed surface singularity subtraction.
    For linear potentials over quadratic elements.
    Parameters:
        lin_pot_mesh: linear potential mesh
        quad_geo_mesh: quadratic geometric mesh
    Returns:
        the stresslet matrix
    """
    num_nodes = lin_pot_mesh.get_nodes().shape[0]
    K = np.zeros((3 * num_nodes, 3 * num_nodes))
    add_lp_qe_DL_terms(K, lin_pot_mesh, quad_geo_mesh)
    add_lp_qe_RBM_terms(K, lin_pot_mesh, quad_geo_mesh)
    return K


def add_lp_qe_DL_terms(K, lin_pot_mesh, quad_geo_mesh):
    """
    Make DL terms for linear potential, quadratic elements mesh
    $-\frac{1}{4\pi} \Int{q_j(\bm{x}) T_{jik}(\bm{x}, \bm{x}_0) n_k(\bm{x})}{S(\bm{x}), D, PV}$
    Parameters:
        K: stiffness matrix to add terms to (3 * face_num, 3 * face_num) ndarray
        lin_pot_mesh: linear potential mesh
        quad_geo_mesh : quadratic geometric mesh
    Returns:
        None
    """
    geo_faces = quad_geo_mesh.get_faces()
    pot_faces = lin_pot_mesh.get_faces()
    assert geo_faces.shape[0] == pot_faces.shape[0]
    num_faces = geo_faces.shape[0]
    pot_nodes = lin_pot_mesh.get_nodes()
    num_nodes = pot_nodes.shape[0]
    c_0 = 1. / (4. * np.pi)

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

    K *= c_0


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


def add_lp_qe_RBM_terms(K, lin_pot_mesh, quad_geo_mesh):
    """
    Add rigid body motion terms to the given stiffness matrix K.
    For linear potential and quadratic elements.
    Parameters:
        K: stiffness matrix to add terms to (3 * node_num, 3 * node_num) ndarray
        lin_pot_mesh: constant potential mesh
        quad_geo_mesh: quadratic geometric mesh
    Returns:
        None
    """
    pot_faces = lin_pot_mesh.get_faces()
    num_faces = pot_faces.shape[0]
    pot_nodes = lin_pot_mesh.get_nodes()
    num_nodes = pot_nodes.shape[0]
    S_D = quad_geo_mesh.get_surface_area()
    x_c = quad_geo_mesh.get_centroid()
    w = quad_geo_mesh.get_w()
    A_m = quad_geo_mesh.get_A_m()

    for face_num in range(num_faces):
        face_nodes = quad_geo_mesh.get_tri_nodes(face_num)
        face_hs = quad_geo_mesh.get_hs(face_num)
        for node_num in range(3): # face nodes
            face_node_global_num = pot_faces[face_num, node_num]
            v_sub_mat = (-4. * np.pi / S_D) * gq.int_over_tri_quad(
                make_lp_le_v_quad(node_num),
                face_nodes,
                face_hs,
            )
            j = 3 * face_node_global_num
            tmp_omega = gq.int_over_tri_quad(
                make_lp_le_omega_quad(node_num, x_c),
                face_nodes,
                face_hs,
            )
            tmp_arr = []
            for m in range(3):
                tmp_arr.append((1./ A_m[m]) * np.outer(w[m], np.einsum("l, ls", w[m], tmp_omega)))
            tmp_omega_mat = -4. * np.pi * np.sum(tmp_arr, axis=0)
            for src_num in range(num_nodes):
                K[(3 * src_num):(3 * src_num + 3),
                  j:j+3] += -1. / (4. * np.pi) * v_sub_mat
                src_pt = lin_pot_mesh.get_node(src_num)
                X_0 = src_pt - x_c
                omega_mat = np.einsum("ijk,js,k", geo.LC_3, tmp_omega_mat, X_0)
                K[(3 * src_num):(3 * src_num + 3),
                  j:j+3] += 1. / (4. * np.pi) * (omega_mat)
