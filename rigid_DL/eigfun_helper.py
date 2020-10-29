"""
Helper functions for creating eigenvectors 
"""
import numpy as np
import rigid_DL.geometric as geo
import rigid_DL.eigenfunctions as eigfun

def make_lin_eig_vels(pot_mesh, E_d, E_c):
    """
    Make the eigenfunction velocity field from the rate of strain matrix

    Parameters:
        pot_mesh: potential mesh
        E_d: ROS field, dotted with position
        E_c: ROS field, crossed with position
    Returns:
        v_list: velocities at each node
    """
    pot_nodes = pot_mesh.get_nodes()
    num_nodes = pot_nodes.shape[0]
    v_list = np.zeros((num_nodes, 3))
    for m in range(num_nodes):
        node = pot_nodes[m]
        v_list[m] = E_d @ node - np.cross(E_c, node)
    return v_list


def make_quad_eig_vels(pot_mesh, dims, kappa_vec):
    """
    Makes the eigenfunction velocity field for the 3x3 quadratic ROS
    field

    Parameters:
        pot_mesh: potential mesh
        dims: ellipsoidal dimensions
        kappa_vec: kappa values for 3x3 system
    Returns:
        v_3x3: (3, N, 3) ndarray (N, 3) for each kappa value
    """
    pot_nodes = pot_mesh.get_nodes()
    num_nodes = pot_nodes.shape[0]
    v_3x3 = np.empty((3, num_nodes, 3))
    for i, kappa in enumerate(kappa_vec):
        H = eigfun.calc_3x3_evec(dims, kappa)
        for m in range(num_nodes):
            node = pot_nodes[m]
            x_tmp = np.array([
                [node[1] * node[2]],
                [node[2] * node[0]],
                [node[0] * node[1]],
            ])
            v_3x3[i, m] = np.ravel(H * x_tmp)
    return v_3x3


def make_lin_psi_func(E_d, E_c):
    """
    Makes the eigenvector function to integrate over the surface
    """
    def quad_func(xi, eta, nodes):
        x = geo.pos_linear(xi, eta, nodes)
        return np.linalg.norm(np.dot(E_d, x) - np.cross(E_c, x))
    return quad_func


def make_quad_psi_func(E_d, E_c):
    """
    Makes the eigenvector function to integrate over the surface
    """
    def quad_func(xi, eta, nodes):
        x = geo.pos_quadratic(xi, eta, nodes)
        return np.linalg.norm(np.dot(E_d, x) - np.cross(E_c, x))
    return quad_func
