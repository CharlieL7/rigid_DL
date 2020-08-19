"""
Matrix assembly functions for the different parameterizations
"""
import numpy as np
import gauss_quad as gq
import geometric as geo


def make_mat_cp_le(lin_mesh):
    """
    Makes the stiffness matrix using closed surface singularity subtraction.
    For constant potentials over a linear elements.

    Parameters:
        lin_mesh : simple linear mesh input
    Returns:
        the stresslet matrix
    """
    num_faces = lin_mesh.faces.shape[0]
    c_0 = 1. / (4. * np.pi)
    C = np.zeros((3 * num_faces, 3 * num_faces))
    for src_num in range(num_faces): # source points
        src_center = lin_mesh.calc_tri_center(lin_mesh.get_nodes(lin_mesh.faces[src_num]))
        for face_num in range(num_faces): # field points
            face_nodes = lin_mesh.get_nodes(lin_mesh.faces[face_num])
            if face_num != src_num:
                sub_mat = gq.int_over_tri_linear(
                    make_cp_le_quad_func(lin_mesh, src_center),
                    face_nodes
                )
                C[(3 * src_num):(3 * src_num + 3),
                  (3 * face_num):(3 * face_num + 3)] += sub_mat
                C[(3 * src_num):(3 * src_num + 3),
                  (3 * src_num):(3 * src_num + 3)] -= sub_mat
            # do nothing face_num == src_num, how it works out for constant elements
        C[(3 * src_num):(3 * src_num + 3),
          (3 * src_num):(3 * src_num + 3)] -= 4. * np.pi * np.identity(3)
    C = np.dot(c_0, C)
    return C


def make_mat_cp_le_f2s(lin_mesh):
    """
    Makes the stiffness matrix using closed surface singularity subtraction.
    For constant potentials over a linear elements.
    Faces loop over source points loop.

    Parameters:
        lin_mesh : simple linear mesh input
    Returns:
        the stresslet matrix
    """
    num_faces = lin_mesh.faces.shape[0]
    c_0 = 1. / (4. * np.pi)
    C = np.zeros((3 * num_faces, 3 * num_faces))
    for face_num in range(num_faces):
        face_nodes = lin_mesh.get_nodes(lin_mesh.faces[face_num])
        face_n = np.cross(face_nodes[1] - face_nodes[0], face_nodes[2] - face_nodes[0])
        face_h_s = np.linalg.norm(face_n)
        face_unit_n = face_n / face_h_s
        for src_num in range(num_faces):
            src_center = lin_mesh.calc_tri_center(lin_mesh.get_nodes(lin_mesh.faces[src_num]))
            if face_num != src_num:
                sub_mat = gq.int_over_tri_linear_f2s(
                    make_cp_le_quad_func_f2s(face_unit_n, src_center),
                    face_nodes, face_h_s
                )
                C[(3 * src_num):(3 * src_num + 3),
                  (3 * face_num):(3 * face_num + 3)] += sub_mat
                C[(3 * src_num):(3 * src_num + 3),
                  (3 * src_num):(3 * src_num + 3)] -= sub_mat
            # do nothing face_num == src_num, how it works out for constant elements
        C[(3 * src_num):(3 * src_num + 3),
          (3 * src_num):(3 * src_num + 3)] -= 4. * np.pi * np.identity(3)
    C = np.dot(c_0, C)
    return C


def make_mat_lp_le(lin_mesh):
    """
    Makes the stiffness matrix using closed surface singularity subtraction.
    For linear potentials over a linear elements.

    Parameters:
        lin_mesh : simple linear mesh input
    Returns:
        the stresslet matrix
    """
    num_faces = lin_mesh.faces.shape[0]
    num_verts = lin_mesh.vertices.shape[0]
    c_0 = 1. / (4. * np.pi)
    C = np.zeros((3 * num_verts, 3 * num_verts))

    for src_num in range(num_verts): # source points
        src_pt = lin_mesh.vertices[src_num]
        for face_num in range(num_faces): # integrate over faces
            face_nodes = lin_mesh.get_nodes(lin_mesh.faces[face_num])
            is_singular, local_singular_ind = lin_mesh.check_in_face(src_num, face_num)

            if is_singular: # singular triangle
                for node_num in range(3):
                    sub_mat = gq.int_over_tri_linear(
                        make_sing_lp_le_quad_func(
                            lin_mesh, src_pt, node_num, local_singular_ind
                            ),
                        face_nodes
                    )
                    C[(3 * src_num):(3 * src_num + 3),
                      (3 * node_global_num):(3 * node_global_num + 3)] += sub_mat

            else: # regular triangle
                for node_num in range(3):
                    node_global_num = lin_mesh.faces[face_num, node_num] # global index for vert
                    sub_mat = gq.int_over_tri_linear(
                        make_reg_lp_le_quad_func(lin_mesh, src_pt, node_num),
                        face_nodes
                    )
                    C[(3 * src_num):(3 * src_num + 3),
                      (3 * node_global_num):(3 * node_global_num + 3)] += sub_mat
                # subtracting the q(x_0) term
                sub_mat = gq.int_over_tri_linear(make_cp_le_quad_func(lin_mesh, src_pt), face_nodes)
                C[(3 * src_num):(3 * src_num + 3), (3 * src_num):(3 * src_num + 3)] -= sub_mat

        # whole surface q(x_0) term
        C[(3 * src_num):(3 * src_num + 3), (3 * src_num):(3 * src_num + 3)] -= (
            4. * np.pi * np.identity(3)
        )

    C = np.dot(c_0, C)
    return C


def make_mat_cp_qe(quad_mesh):
    """
    Makes the stiffness matrix using closed surface singularity subtraction.
    For constant potentials over quadratic elements.

    Parameters:
        quad_mesh : simple quadratic mesh input
    Returns:
        the stresslet matrix
    """
    num_faces = quad_mesh.faces.shape[0]
    c_0 = 1. / (4. * np.pi)
    C = np.zeros((3 * num_faces, 3 * num_faces))
    for src_num in range(num_faces): # source points
        src_center = quad_mesh.calc_tri_center(quad_mesh.get_nodes(quad_mesh.faces[src_num]))
        for face_num in range(num_faces): # field points
            face_nodes = quad_mesh.get_nodes(quad_mesh.faces[face_num])
            if face_num != src_num:
                sub_mat = gq.int_over_tri_quadratic(
                    make_cp_qe_quad_func(quad_mesh, src_center),
                    face_nodes
                )
                C[(3 * src_num):(3 * src_num + 3),
                  (3 * face_num):(3 * face_num + 3)] += sub_mat
                C[(3 * src_num):(3 * src_num + 3),
                  (3 * src_num):(3 * src_num + 3)] -= sub_mat
            # do nothing face_num == src_num, how it works out for constant elements
        C[(3 * src_num):(3 * src_num + 3),
          (3 * src_num):(3 * src_num + 3)] -= 4. * np.pi * np.identity(3)
    C = np.dot(c_0, C)
    return C


def make_mat_lp_qe(quad_mesh):
    """
    Makes the stiffness matrix using closed surface singularity subtraction.
    For linear potentials over quadratic elements.

    Parameters:
        quad_mesh : simple quadratic mesh input
    Returns:
        the stresslet matrix
    """
    num_faces = quad_mesh.faces.shape[0]
    num_verts = quad_mesh.vertices.shape[0]
    c_0 = 1. / (4. * np.pi)
    C = np.zeros((3 * num_verts, 3 * num_verts))

    for src_num in range(num_verts): # source points
        src_pt = quad_mesh.vertices[src_num]
        for face_num in range(num_faces): # integrate over faces
            face_nodes = quad_mesh.get_nodes(quad_mesh.faces[face_num])
            is_singular, local_singular_ind = quad_mesh.check_in_face(src_num, face_num)

            if is_singular: # singular triangle
                for node_num in range(3):
                    sub_mat = gq.int_over_tri_linear(
                        make_sing_lp_le_quad_func(
                            quad_mesh, src_pt, node_num, local_singular_ind
                            ),
                        face_nodes
                    )
                    C[(3 * src_num):(3 * src_num + 3),
                      (3 * node_global_num):(3 * node_global_num + 3)] += sub_mat

            else: # regular triangle
                for node_num in range(3):
                    node_global_num = quad_mesh.faces[face_num, node_num] # global index for vert
                    sub_mat = gq.int_over_tri_linear(
                        make_reg_lp_le_quad_func(quad_mesh, src_pt, node_num),
                        face_nodes
                    )
                    C[(3 * src_num):(3 * src_num + 3),
                      (3 * node_global_num):(3 * node_global_num + 3)] += sub_mat
                # subtracting the q(x_0) term
                sub_mat = gq.int_over_tri_linear(make_cp_le_quad_func(quad_mesh, src_pt), face_nodes)
                C[(3 * src_num):(3 * src_num + 3), (3 * src_num):(3 * src_num + 3)] -= sub_mat

        # whole surface q(x_0) term
        C[(3 * src_num):(3 * src_num + 3), (3 * src_num):(3 * src_num + 3)] -= (
            4. * np.pi * np.identity(3)
        )

    C = np.dot(c_0, C)
    return C


def off_diag_ros_field(mesh, const_or_linear):
    """
    Off diagonal rate of strain field eigenfunction
    Returns velocites at each vertex in cartesional coordinates
    Note that this is already in cartesional coordinates

    Parameters:
        mesh : simple mesh input
        const_or_linear : if constant or linear density distributions
    """
    E = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
    if const_or_linear == 'c':
        num_faces = mesh.faces.shape[0]
        v_list = np.zeros((num_faces, 3))
        for m in range(num_faces):
            face = mesh.faces[m]
            nodes = mesh.get_nodes(face)
            center = mesh.calc_tri_center(nodes)
            v_list[m] = np.dot(E, center)
        return v_list
    elif const_or_linear == 'l':
        num_vert = mesh.vertices.shape[0]
        v_list = np.zeros((num_vert, 3))
        for m in range(num_vert):
            vert = mesh.vertices[m]
            v_list[m] = np.dot(E, vert)
        return v_list
    else:
        sys.exit("unknown const_or_linear value")


def make_cp_le_quad_func(lin_mesh, x_0):
    """
    Makes the constant potential function that is integrated over 
    linear elements for the stiffness matrix

    Parameters:
        lin_mesh: the linear mesh object
        x_0: the source point
    """
    def quad_func(xi, eta, nodes):
        x = geo.pos_linear(xi, eta, nodes)
        n = lin_mesh.calc_normal(nodes)
        return geo.stresslet(x, x_0, n)
    return quad_func


def make_cp_le_quad_func_f2s(n, x_0):
    """
    Makes the constant potential function that is integrated over 
    linear elements for the stiffness matrix
    F2S version.

    Parameters:
        lin_mesh: the linear mesh object
        x_0: the source point
    """
    def quad_func(xi, eta, nodes):
        x = geo.pos_linear(xi, eta, nodes)
        return geo.stresslet(x, x_0, n)
    return quad_func


def make_cp_qe_quad_func(quad_mesh, x_0):
    """"
    Makes the constant potential function that is integrated over 
    quadratic elements for the stiffness matrix

    Parameters:
        quad_mesh: the quadratic mesh object
        x_0: the source point
    """
    def quad_func(xi, eta, nodes):
        x = geo.pos_quadratic(xi, eta, nodes)
        n = quad_mesh.normal_func(xi, eta, nodes)
        return geo.stresslet(x, x_0, n)
    return quad_func


def make_reg_lp_le_quad_func(lin_mesh, x_0, node_num):
    """
    Makes the regular (non-singular) linear potential, linear element function
    that is integrated for the stiffness matrix

    Parameters:
        lin_mesh: the linear mesh object
        x_0: source point
        node_num: which potential shape function [0, 1, 2]
    """
    def quad_func(xi, eta, nodes):
        x = geo.pos_linear(xi, eta, nodes)
        n = lin_mesh.calc_normal(nodes)
        S = geo.stresslet(x, x_0, n)
        phi = geo.shape_func_linear(xi, eta, node_num)
        return phi * S
    return quad_func


def make_reg_lp_qe_quad_func(quad_mesh, x_0, node_num):
    """
    Makes the regular (non-singular) linear potential, quadratic element function
    that is integrated for the stiffness matrix

    Parameters:
        quad_mesh: the quadratic mesh object
        x_0: source point
        node_num: which potential shape function [0, 1, 2]
    """
    def quad_func(xi, eta, nodes):
        x = geo.pos_quadratic(xi, eta, nodes)
        n = quad_mesh.normal_func(xi, eta, nodes)
        S = geo.stresslet(x, x_0, n)
        phi = geo.shape_func_linear(xi, eta, node_num)
        return phi * S
    return quad_func


def make_sing_lp_le_quad_func(lin_mesh, x_0, node_num, singular_ind):
    """
    Makes the sinuglar linear potential, linear element function
    that is integrated for the stiffness matrix

    Parameters:
        lin_mesh: linear mesh object
        x_0: source point
        node_num: which shape function
        singular_ind: local singular index for a face
    """
    def quad_func(xi, eta, nodes):
        x = geo.pos_linear(xi, eta, nodes)
        x_hat = x - x_0
        n = lin_mesh.calc_normal(nodes)
        r = np.linalg.norm(x_hat)
        phi = geo.shape_func_linear(xi, eta, node_num)
        # shape function for source point is [1, 0, 0], [0, 1, 0], or [0, 0, 1]
        if node_num == singular_ind:
            if (phi - 1) == 0: # getting around division by 0
                return 0
            else:
                return (phi - 1) * (-6 * np.outer(x_hat, x_hat) * np.dot(x_hat, n) / (r**5))
        else:
            if phi == 0:
                return 0
            else:
                return (phi) * (-6 * np.outer(x_hat, x_hat) * np.dot(x_hat, n) / (r**5))
    return quad_func


def make_sing_lp_qe_quad_func(quad_mesh, x_0, node_num, singular_ind):
    """
    Makes the sinuglar linear potential, quadratic element function
    that is integrated for the stiffness matrix

    Parameters:
        quad_mesh: the curved triangular mesh
        x_0: source point
        node_num: which potential shape function [0, 1, 2]
        singular_ind: local singular index for a face
    """
    def quad_func(xi, eta, nodes):
        x = geo.pos_quadratic(xi, eta, nodes)
        n = quad_mesh.normal_func(xi, eta, nodes)
        x_hat = x - x_0
        r = np.linalg.norm(x_hat)
        phi = geo.shape_func_linear(xi, eta, node_num)
        # shape function for source point is [1, 0, 0], [0, 1, 0], or [0, 0, 1]
        if node_num == singular_ind:
            if (phi - 1) == 0: # getting around division by 0
                return 0
            else:
                return (phi - 1) * (-6 * np.outer(x_hat, x_hat) * np.dot(x_hat, n) / (r**5))
        else:
            if phi == 0:
                return 0
            else:
                return (phi) * (-6 * np.outer(x_hat, x_hat) * np.dot(x_hat, n) / (r**5))
    return quad_func


if __name__ == "__main__":
    main()
