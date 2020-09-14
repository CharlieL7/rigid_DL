"""
Creates the stiffness matrix and then finds the eigenvalue of the system
corresponding to an eigenfunction
Used to get a measure of the global error in a discretization
"""
import sys
import numpy as np
import rigid_DL.input_output as io
import rigid_DL.simple_linear_mesh as slm
import rigid_DL.mat_assembly as mata
import rigid_DL.eigenfunctions as efun

def main():
    if len(sys.argv) != 3:
        print("Usage: mesh, piece-wise constant or linear (c, l)")
        sys.exit()
    mesh_name = sys.argv[1]

    (x_data, f2v, _params) = io.read_short_dat(mesh_name)
    f2v = f2v - 1 # indexing change
    mesh = slm.simple_linear_mesh(x_data, f2v)
    v_cnst = np.array([1, 0, 0])
    v_trans_in = np.zeros((mesh.faces.shape[0], 1), dtype=v_cnst.dtype) + v_cnst
    v_trans_in = v_trans_in.flatten("C")

    C = mata.make_mat_cp_le(mesh) # stiffness matrix

    eig_vals, eig_vecs = np.linalg.eig(C)

    # find the discrete eigenvector that best approximates the actual eigenvector
    min_L2 = float("inf")
    min_ind = 0
    for i, vec in enumerate(eig_vecs):
        tmp = np.linalg.norm(v_trans_in - np.real(vec))
        if tmp < min_L2:
            min_L2 = tmp
            min_ind = i

    print("Discrete eigenvalue for E12: {}".format(eig_vals.real[min_ind]))
    print("Error in L2 norm of eigenvector: {}".format(min_L2))


if __name__ == "__main__":
    main()
