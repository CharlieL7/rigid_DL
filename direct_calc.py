"""
Direct calulation of the velocity solution from an eigenfunction density input
To check if the method we are employing makes sense.
"""
import sys
import math
import numpy as np
import simple_mesh as sm
import input_output as io
import gauss_quad as gq

def main():
    if len(sys.argv) != 3:
        print("Usage: mesh_name, rotational velocity")
    mesh_name = sys.argv[1]
    w = float(sys.argv[2])
    (x_data, f2v, _params) = io.read_short_dat(mesh_name)
    f2v = f2v - 1 # indexing change
    mesh = sm.simple_mesh(x_data, f2v)
    v_in = sphere_eigen_func(mesh, w) # eigenfunction
    num_faces = mesh.faces.shape[0]
    v_out = np.zeros((num_faces, 3))
    C = np.zeros((3 * num_faces, 3 * num_faces))
    c_0 = (1. / (4. * math.pi))
    for m in range(num_faces): # source points
        src_center = mesh.calc_tri_center(mesh.faces[m])
        for n in range(num_faces): # field points
            if n != m:
                field_nodes = mesh.get_nodes(mesh.faces[n])
                field_normal = mesh.calc_normal(mesh.faces[n])
                sub_mat = -c_0 * gq.int_over_tri(make_quad_func(src_center, field_normal), field_nodes)
                C[(3 * m):(3 * m + 3), (3 * n):(3 * n + 3)] = sub_mat
                v_out[m] += np.dot(v_in[n], sub_mat)

    # singular points as function of all regular points
    for m in range(num_faces): # source points
        src_center = mesh.calc_tri_center(mesh.faces[m])
        sub_mat = np.zeros((3, 3))
        for n in range(num_faces): # over field points
            if n != m:
                sub_mat += c_0 * C[(3 * m):(3 * m + 3), (3 * n):(3 * n + 3)]

        C[(3 * m):(3 * m + 3), (3 * m):(3 * m + 3)] = sub_mat + np.identity(3)
        v_out[m] += np.dot(v_in[m], C[(3 * m):(3 * m + 3), (3 * m):(3 * m + 3)])

    print("v_out")
    print("-------------------")
    print(v_out)


def sphere_eigen_func(mesh, w):
    """
    Eigenfunction for the double layer potential to check the solutions from the code integrals
    Rotating rigid sphere
    Returns velocities at each vertex in cartesional coordinates

    Parameters:
        mesh : simple mesh input
        w : rotational velocity magnitude
    """
    num_faces = mesh.faces.shape[0]
    v_list = np.zeros((num_faces, 3))
    for m in range(num_faces):
        face = mesh.faces[m]
        center = mesh.calc_tri_center(face)
        (r, theta, phi) = cart2sph(center) # all the radii should be essentially the same
        v_sph = np.array([0., 0., w * r * math.sin(theta)])
        v_list[m] = sph2cart(v_sph)
    return v_list


def cart2sph(vec):
    (x, y, z) = vec 
    xy = x**2 + y**2
    r = math.sqrt(xy + z**2)
    theta = math.atan2(z, math.sqrt(xy))
    phi = math.atan2(y, x)
    return np.array([r, theta, phi])


def sph2cart(vec):
    (r, theta, phi) = vec
    x = r * math.sin(theta) * math.cos(phi)
    y = r * math.sin(theta) * math.sin(phi)
    z = r * math.cos(theta)
    return np.array([x, y, z])


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
    main()
