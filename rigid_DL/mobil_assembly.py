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
        for src_num in range(num_faces):
            src_center = cons_pot_mesh.get_node(src_num)
            if face_num != src_num:
                sub_mat = gq.int_over_tri_lin(
                    make_DL_cp_le_quad_func(face_n, src_center),
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

    x_c = lin_geo_mesh.get_centroid()
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
            A_m[m] += gq.int_over_tri_lin(
                make_am_quad_func(w[m]),
                face_nodes,
                face_hs,
            )

    for face_num in range(num_faces):
        face_nodes = lin_geo_mesh.get_tri_nodes(face_num)
        face_n = lin_geo_mesh.get_normal(face_num)
        face_hs = lin_geo_mesh.get_hs(face_num)
        def v_quad(_xi, _eta, _nodes):
            return 1.
        v_sub_mat = np.identity(3) * gq.int_over_tri_lin(
            v_quad,
            face_nodes,
            face_hs,
        ) # added to the whole column
        for src_num in range(num_faces):
            K[(3 * src_num):(3 * src_num + 3),
              (3 * face_num):(3 * face_num + 3)] += (1. / S_D) * v_sub_mat
            src_center = cons_pot_mesh.get_node(src_num)
            def omega_quad(xi, eta, nodes):
                pos = geo.pos_linear(xi, eta, nodes)
                X = pos - x_c
                return np.einsum("lrs,r->ls", geo.LC_3, X)
            tmp_omega = gq.int_over_tri_lin(
                omega_quad,
                face_nodes,
                face_hs,
            )
            tmp_arr = []
            for m in range(3):
                tmp_arr.append((1./ A_m[m]) * np.einsum("j,l,ls->js", w[m], w[m], tmp_omega))
            omega_mat = np.sum(tmp_arr, axis=0) * -4. * np.pi
            X_0 = src_center - x_c
            omega_mat = np.einsum("ijk,js,k->is", geo.LC_3, omega_mat, X_0)
            K[(3 * src_num):(3 * src_num + 3),
              (3 * face_num):(3 * face_num + 3)] += omega_mat
    return K


def make_cp_le_forcing_vec(cons_pot_mesh, lin_geo_mesh, u_d, f, l, mu):
    """
    Makes the forcing vector ( f ) for the mobility problem given a
    constant potential mesh and linear geometric mesh.

    Parameters:
        cons_pot_mesh: potential mesh
        lin_geo_mesh: geometric mesh
        u_d: disturbance velocity; (3 * N,) ndarray
        f: force on particle; (3,) ndarray
        l: torque on particle; (3,) ndarray
        mu: fluid viscosity; scalar
    Returns:
        f: forcing vector (3 * N,) ndarray
    """
    pot_faces = cons_pot_mesh.get_faces()
    assert pot_faces.shape[0] == lin_geo_mesh.get_faces().shape[0]
    num_faces = pot_faces.shape[0]
    x_c = cons_pot_mesh.get_centroid()
    c_0 = 1. / (4. * np.pi)

    # make Power and Miranda supplementary flow vector
    f_s = f / (-8. * np.pi * mu) # the script F seen in Pozrikidis
    l_s = l / (-8. * np.pi * mu) # the script L seen in Pozrikidis
    v_s = np.empty(3 * num_faces)
    for src_num in range(num_faces):
        node = cons_pot_mesh.get_node(src_num)
        v_s[(3 * src_num) : (3 * src_num + 3)] = f_s * geo.stokeslet(node, x_c) + l_s * geo.rotlet(node, x_c)
    c_s = c_0 * (u_d - v_s) # script C term from Pozrikidis
    
    # make integral of c_s dotted with normal vector term


def make_init_sol_vector(pot_mesh):
    """
    Makes the inital solution vector ( q ) for the mobility problem.
    Sets random values for usage in method of successive substitutions.

    Parameters:
        pot_mesh: potential mesh
    Returns:
        q : random initialized solution vector; (N, 3) ndarray
    """
    num_nodes = pot_mesh.get_nodes().shape[0]
    return np.random.rand(num_nodes, 3)


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
