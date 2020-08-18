"""
Creates the stiffness matrix and then finds the eigenvalue of the system
corresponding to an eigenfunction
Used to get a measure of the global error in a discretization
"""
import sys
import numpy as np
import input_output as io
import simple_linear_mesh as slm
import potential_calc_v2 as pot_calc
import eigenfunctions as efun

def main():
    if len(sys.argv) != 3:
        print("Usage: mesh, piece-wise constant or linear (c, l)")
        sys.exit()
    mesh_name = sys.argv[1]
    const_or_linear = sys.argv[2]
    assert const_or_linear in ('c', 'l')

    (x_data, f2v, _params) = io.read_short_dat(mesh_name)
    f2v = f2v - 1 # indexing change
    mesh = slm.simple_linear_mesh(x_data, f2v)
    v_cnst = np.array([1, 0, 0])
    eig_vec_in = efun.make_translation_vels(v_cnst, mesh, const_or_linear)
    # normalize into unit vector
    eig_vec_in /= np.linalg.norm(eig_vec_in)
    eig_vec_in = eig_vec_in.flatten('C')

    if const_or_linear == 'c':
        C = pot_calc.make_mat_const(mesh) # stiffness matrix
    elif const_or_linear == 'l':
        C = pot_calc.make_mat_linear(mesh) # stiffness matrix

    eig_vals, eig_vecs = np.linalg.eig(C)

    # find the discrete eigenvector that best approximates the actual eigenvector
    min_L2 = float("inf")
    min_ind = 0
    for i, vec in enumerate(eig_vecs):
        tmp = np.linalg.norm(eig_vec_in - np.real(vec))
        if tmp < min_L2:
            min_L2 = tmp
            min_ind = i

    print("Discrete eigenvalue for E12: {}".format(eig_vals.real[min_ind]))
    print("Error in L2 norm of eigenvector: {}".format(min_L2))


if __name__ == "__main__":
    main()
