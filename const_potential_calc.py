"""
Direct calulation of the velocity solution from an eigenfunction density input
To check if the method we are employing makes sense.
"""
import sys
import csv
import math
import time
import numpy as np
import simple_mesh as sm
import input_output as io
import gauss_quad as gq

def main1():
    if len(sys.argv) != 2:
        print("Usage: mesh_name")
        sys.exit()
    mesh_name = sys.argv[1]
    (x_data, f2v, _params) = io.read_short_dat(mesh_name)
    f2v = f2v - 1 # indexing change
    mesh = sm.simple_mesh(x_data, f2v)
    num_faces = mesh.faces.shape[0]
    v_in = sphere_eigen_func(mesh, 1.) # eigenfunction
    v_out = np.zeros((num_faces, 3))
    C = np.zeros((3 * num_faces, 3 * num_faces))
    c_0 = (1. / (4. * math.pi)) # not sure why this is positive
    for m in range(num_faces): # source points
        src_center = mesh.calc_tri_center(mesh.faces[m])
        for n in range(num_faces): # field points
            if n != m:
                field_nodes = mesh.get_nodes(mesh.faces[n])
                field_normal = mesh.calc_normal(mesh.faces[n])
                sub_mat = gq.int_over_tri(make_quad_func(src_center, field_normal), field_nodes)
                C[(3 * m):(3 * m + 3), (3 * n):(3 * n + 3)] += sub_mat

    # singularity subtraction
    # total - regular points
    for m in range(num_faces): # source points
        src_center = mesh.calc_tri_center(mesh.faces[m])
        sub_mat = np.zeros((3, 3))
        for n in range(num_faces): # over field points
            if n != m:
                sub_mat += C[(3 * m):(3 * m + 3), (3 * n):(3 * n + 3)]
        C[(3 * m):(3 * m + 3), (3 * m):(3 * m + 3)] = -4. * math.pi * np.identity(3) - sub_mat

    C = np.dot(c_0, C)
    v_out = np.dot(C, v_in.flatten('C'))
    v_out = v_out.reshape(v_in.shape, order='C')

    w, _v = np.linalg.eig(C)
    io.write_vel(v_in, v_out, "test_vels.csv")
    io.write_eig(w, "test_eigs.csv")


def main2():
    """
    other singularity subtraction
    """
    if len(sys.argv) != 2:
        print("Usage: mesh_name")
        sys.exit()
    mesh_name = sys.argv[1]

    t0 = time.time()

    (x_data, f2v, _params) = io.read_short_dat(mesh_name)
    f2v = f2v - 1 # indexing change
    mesh = sm.simple_mesh(x_data, f2v)
    num_faces = mesh.faces.shape[0]

    t1 = time.time()
    print("{}, mesh input walltime".format(t1 - t0))
    v_in = off_diag_ros_field(mesh)
    v_out = np.zeros((num_faces, 3))
    C = np.zeros((3 * num_faces, 3 * num_faces))
    c_0 = (1. / (4. * math.pi)) # note to future self: think of formulation without eigenvalue, always this constant (inside or outside particle), then find eigenvalues, setting direction of forcing function

    t0 = time.time()
    print("{}, eigenfunction calc walltime".format(t0 - t1))

    for m in range(num_faces): # source points
        src_center = mesh.calc_tri_center(mesh.faces[m])
        for n in range(num_faces): # field points
            if n != m:
                field_nodes = mesh.get_nodes(mesh.faces[n])
                field_normal = mesh.calc_normal(mesh.faces[n])
                sub_mat = gq.int_over_tri(make_quad_func(src_center, field_normal), field_nodes)
                C[(3 * m):(3 * m + 3), (3 * n):(3 * n + 3)] += sub_mat
                C[(3 * m):(3 * m + 3), (3 * m):(3 * m + 3)] -= sub_mat
            # do nothing n == m for constant elements
        C[(3 * m):(3 * m + 3), (3 * m):(3 * m + 3)] -= 4. * math.pi * np.identity(3)
    C = np.dot(c_0, C)
    v_out = np.dot(C, v_in.flatten('C'))
    v_out = v_out.reshape(v_in.shape, order='C')

    t1 = time.time()
    print("{}, matrix forming and dotting walltime".format(t1 - t0))

    w, _v = np.linalg.eig(C)

    t0 = time.time()
    print("{}, solving eigenfunction walltime".format(t0 - t1))

    io.write_vel(v_in, v_out, "vel_ros.csv")
    io.write_eig(w, "eig_ros.csv")

    t1 = time.time()
    print("{}, write out walltime".format(t1 - t0))


def sphere_eigen_func(mesh, w):
    """
    Eigenfunction for the double layer potential to check the solutions from the code integrals
    Rotating rigid sphere
    Returns velocities at each vertex in cartesional coordinates
    for constant potentials

    Parameters:
        mesh : simple mesh input
        w : rotational velocity magnitude
    """
    num_faces = mesh.faces.shape[0]
    v_list = np.zeros((num_faces, 3))
    for m in range(num_faces):
        face = mesh.faces[m]
        center = mesh.calc_tri_center(face)
        c_sph = cart2sph(center) # all the radii should be essentially the same
        v_sph = np.array([0., 0., w * c_sph[0] * math.sin(c_sph[1])])
        v_list[m] = v_sph2cart(c_sph, v_sph)
    return v_list


def off_diag_ros_field(mesh):
    """
    Off diagonal rate of strain field eigenfunction
    Returns velocites at each vertex in cartesional coordinates
    Note that this is already in cartesional coordinates
    for constant potentials

    Parameters:
        mesh : simple mesh input
    """
    E = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
    num_faces = mesh.faces.shape[0]
    v_list = np.zeros((num_faces, 3))
    for m in range(num_faces):
        face = mesh.faces[m]
        center = mesh.calc_tri_center(face)
        v_list[m] = np.dot(E, center)
    return v_list


def cart2sph(vec):
    (x, y, z) = vec
    xy = x**2 + y**2
    r = math.sqrt(xy + z**2)
    theta = math.acos(z/r)
    phi = math.atan2(y, x)
    return np.array([r, theta, phi])


def sph2cart(vec):
    (r, theta, phi) = vec
    x = r * math.sin(theta) * math.cos(phi)
    y = r * math.sin(theta) * math.sin(phi)
    z = r * math.cos(theta)
    return np.array([x, y, z])


def v_sph2cart(x, v):
    (r, theta, phi) = x
    (rd, td, pd) = v
    dxdt = rd * math.sin(theta) * math.cos(phi) + r * math.cos(theta) * td * math.cos(phi) - r * math.sin(theta) * math.sin(phi) * pd
    dydt = rd * math.sin(theta) * math.sin(phi) + r * math.cos(theta) * td * math.sin(phi) + r * math.sin(theta) * math.cos(phi) * pd
    dzdt = rd * math.cos(theta) - r * math.sin(theta) * td
    return np.array([dxdt, dydt, dzdt])


def make_quad_func(x_0, n):
    """
    Throwaway function for interfacing
    """
    def quad_func(eta, xi, nodes):
        x = pos(eta, xi, nodes)
        return stresslet(x, x_0, n)
    return quad_func


def pos(eta, xi, nodes):
    """
    position in a triangle as a function of eta and xi

    Parameters:
        eta : parametric coordinate, scalar
        xi : paramteric coordinate, scalar
        nodes : three nodes of triangle as columns in 3x3 ndarray
    Returns:
        x : output position (3,) ndarray
    """
    x = (1. - eta - xi) * nodes[0] + eta * nodes[1] + xi * nodes[2]
    return x


def stresslet(x, x_0, n):
    """
    Stress tensor Green's function dotted with the normal vector.
    T_ijk @ n_k
    Parameters:
        x : field point, (3,) ndarray
        x_0 : source point, (3,) ndarray
        n : normal vector, (3,) ndarray
    Returns:
        S_ij : (3,3) ndarray
    """
    x_hat = x - x_0
    r = np.linalg.norm(x_hat)
    S_ij = -6 * np.outer(x_hat, x_hat) * np.dot(x_hat, n) / (r**5)
    return S_ij


if __name__ == "__main__":
    main2()
