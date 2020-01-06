"""
Direct calulation of the velocity solution from an eigenfunction density input
To check if the method we are employing makes sense.
"""
import sys
import math
import time
import ast
import numpy as np
import simple_linear_mesh as slm
import input_output as io
import gauss_quad as gq
import geometric as geo
from geometric import shape_func_linear
import eigenfunctions as efun

def main():
    if len(sys.argv) != 4:
        print("Usage: mesh name, constant or linear (c, l), output name")
        sys.exit()
    mesh_name = sys.argv[1]
    const_or_linear = sys.argv[2]
    assert const_or_linear in ('c', 'l')
    out_name = sys.argv[3]

    t0 = time.time()

    (x_data, f2v, _params) = io.read_short_dat(mesh_name)
    f2v = f2v - 1 # indexing change
    mesh = slm.simple_linear_mesh(x_data, f2v)
    mesh.dims = ast.literal_eval(input("ellipsoid dimensions (a, b, c): "))
    v_in = efun.diag_eigvec("-", mesh, const_or_linear)

    t1 = time.time()
    print("{}, before construct stiffness matrix".format(t1 - t0))

    if const_or_linear == 'c':
        C = make_mat_const(mesh) # stiffness matrix
    elif const_or_linear == 'l':
        C = make_mat_linear(mesh) # stiffness matrix
    v_out = np.dot(C, v_in.flatten('C'))
    v_out = v_out.reshape(v_in.shape, order='C')

    t2 = time.time()
    print("{}, matrix forming and dotting walltime".format(t2 - t1))

    w, _v = np.linalg.eig(C)

    io.write_vel(v_in, v_out, "{}_vel.csv".format(out_name))
    io.write_eig(w, "{}_eig.csv".format(out_name))

    t3 = time.time()
    print("{}, write out walltime".format(t3 - t2))


def make_mat_const(mesh):
    """
    Makes the stiffness matrix using point singularity subtraction.
    For constant elements.

    Parameters:
        mesh : simple mesh input
    Returns:
        the stresslet matrix
    """
    num_faces = mesh.faces.shape[0]
    c_0 = 1. / (4. * math.pi)
    C = np.zeros((3 * num_faces, 3 * num_faces))
    for src_num in range(num_faces): # source points
        src_center = mesh.calc_tri_center(mesh.faces[src_num])
        for face_num in range(num_faces): # field points
            field_nodes = mesh.get_nodes(mesh.faces[face_num])
            field_normal = mesh.calc_normal(mesh.faces[face_num])
            if face_num != src_num:
                sub_mat = gq.int_over_tri(
                    make_const_quad_func(src_center, field_normal),
                    field_nodes
                )
                C[(3 * src_num):(3 * src_num + 3),
                  (3 * face_num):(3 * face_num + 3)] += sub_mat
                C[(3 * src_num):(3 * src_num + 3),
                  (3 * src_num):(3 * src_num + 3)] -= sub_mat
            # do nothing face_num == src_num, how it works out for constant elements
        C[(3 * src_num):(3 * src_num + 3),
          (3 * src_num):(3 * src_num + 3)] -= 4. * math.pi * np.identity(3)
    C = np.dot(c_0, C)
    return C


def make_mat_linear(mesh):
    """
    Makes the stiffness matrix using point singularity subtraction.
    For constant elements.

    Parameters:
        mesh : simple mesh input
    Returns:
        the stresslet matrix
    """
    num_faces = mesh.faces.shape[0]
    num_verts = mesh.vertices.shape[0]
    c_0 = 1. / (4. * math.pi)
    C = np.zeros((3 * num_verts, 3 * num_verts))

    for src_num in range(num_verts): # source points
        src_pt = mesh.vertices[src_num]
        for face_num in range(num_faces): # integrate over faces
            field_nodes = mesh.get_nodes(mesh.faces[face_num])
            field_normal = mesh.calc_normal(mesh.faces[face_num])
            is_singular, local_singular_ind = mesh.check_in_face(src_num, face_num)

            if is_singular: # singular triangle
                for node_num in range(3):
                    sub_mat = gq.int_over_tri(
                        make_sing_linear_quad_func(
                            src_pt, field_normal, node_num, local_singular_ind
                            ),
                        field_nodes
                    )
                    C[(3 * src_num):(3 * src_num + 3),
                      (3 * node_global_num):(3 * node_global_num + 3)] += sub_mat

            else: # regular triangle
                for node_num in range(3):
                    node_global_num = mesh.faces[face_num, node_num] # global index for vert
                    sub_mat = gq.int_over_tri(
                        make_reg_linear_quad_func(src_pt, field_normal, node_num),
                        field_nodes
                    )
                    C[(3 * src_num):(3 * src_num + 3),
                      (3 * node_global_num):(3 * node_global_num + 3)] += sub_mat
                # subtracting the q(x_0) term
                sub_mat = gq.int_over_tri(make_const_quad_func(src_pt, field_normal), field_nodes)
                C[(3 * src_num):(3 * src_num + 3), (3 * src_num):(3 * src_num + 3)] -= sub_mat

        # whole surface q(x_0) term
        C[(3 * src_num):(3 * src_num + 3), (3 * src_num):(3 * src_num + 3)] -= (
            4. * math.pi * np.identity(3)
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
            center = mesh.calc_tri_center(face)
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


def make_const_quad_func(x_0, n):
    """
    Makes the constant potential function that is integrated for the stiffness matrix
    """
    def quad_func(eta, xi, nodes):
        x = geo.pos(eta, xi, nodes)
        return geo.stresslet(x, x_0, n)
    return quad_func


def make_reg_linear_quad_func(x_0, n, node_num):
    """
    Makes the regular (non-singular) linear potential function that is
    integrated for the stiffness matrix

    Parameters:
        x_0: source point
        n: normal vector
        num: which shape function
    """
    def quad_func(eta, xi, nodes):
        x = geo.pos(eta, xi, nodes)
        S = geo.stresslet(x, x_0, n)
        phi = shape_func_linear(eta, xi, node_num)
        return np.dot(phi, S)
    return quad_func


def make_sing_linear_quad_func(x_0, n, node_num, singular_ind):
    """
    Makes the sinuglar linear potential function that is integrated for the stiffness matrix

    Parameters:
        x_0: source point
        n: normal vector
        node_num: which shape function
        singular_ind: local singular index for a face
    """
    def quad_func(eta, xi, nodes):
        x = geo.pos(eta, xi, nodes)
        x_hat = x - x_0
        r = np.linalg.norm(x_hat)
        phi = shape_func_linear(eta, xi, node_num)
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
