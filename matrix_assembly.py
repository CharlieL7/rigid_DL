"""
Module to generate all of the kernels
"""
import math
import simple_mesh
import gauss_quad

def form_matrix():
    """
    Assembles the A matrix from x + Ax = b. For the double layer rigid body problem.
    Parameters:
    Returns:
    """
    #TODO


def calc_double_layer(mesh):
    """
    Forms the matrix for the regular and singular double layer potential contributions.
    
    NOTE: can optimize this by vectorizing the sub_mat calculations
    Parameters:
        mesh : simple_mesh object to calculate over
    Returns:
        C : (3 * num_faces, 3 * num_faces) ndarray
    """
    num_faces = mesh.faces.shape[0]
    C = np.zeros((3 * num_faces, 3 * num_faces))
    c_0 = (1. / (4. * math.pi))
    
    # regular points
    for m in range(num_faces): # source points
        src_center = mesh.calc_tri_center(mesh.faces[m])
        for n in [x in range(num_faces) if x != m]: # field points
            field_nodes = mesh.get_nodes(mesh.faces[n])
            field_normal = mesh.calc_normal(mesh.faces[n])
            sub_mat = -c_0 * gauss_quad(make_quad_func(src_center, field_normal), field_nodes)
            C[(3 * m):(3 * m + 3), (3 * n):(3 * n + 3)] = sub_mat

    # singular points as function of all regular points
    for m in range(num_faces): # source points
        src_center = mesh.calc_tri_center(mesh.faces[m])
        sub_mat = np.zeros(3, 3)
        for n in [x in range(num_faces) if x != m]: # field points
            sub_mat += c_0 * C[(3 * m ):(3 * m + 3), (3 * n):(3 * n + 3)]
        C[(3 * m):(3 * n + 3), (3 * m):(3 * m + 3)] = sub_mat + np.identity(3)

    return C


def calc_rigid_body(mesh):
    """
    Forms the matrix for the rigid body motions.
    Added to the double layer matrix to remove (-1) eigenvalue and complete
    part of the space of all possible external flows.

    Parameters:
        mesh : simple_mesh object to calculate over
    Returns:
        (num_faces, num_faces, 3, 3) ndarray
    """
    # translation, many diagonal matrix
    num_faces = mesh.faces.shape[0]
    D = np.zeros((3 * num_faces, 3 * num_faces))


    
    


def calc_normal_kernel(mesh):
    """
    Forms the matrix to remove the (+1) eigenvalue from the double layer formulations.

    Parameters:
        mesh : simple_mesh object to calculate over
    Returns:
        (num_faces, num_faces, 3, 3) ndarray
    """
    #TODO


def calc_external_velocity(mesh):
    """
    Forms the vector of external velocity contributions.

    Parameters:
        mesh : simple_mesh object to calculate over
    Returns:
        (num_faces, num_faces, 3, 3) ndarray
    """
    #TODO


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
    S_ij = -6 np.outer(x_hat, x_hat) * np.dot(x_hat, n) / (r**5)
    return S_ij


def make_quad_func(x_0, n):
    def quad_func(eta, xi, nodes):
        x = pos(eta, xi, nodes)
        return stresslet(x, x_0, n)
    return quad_func
