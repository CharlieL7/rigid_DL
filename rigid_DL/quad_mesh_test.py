"""
Quadratic mesh test
"""

import time
import argparse as argp
import numpy as np
import meshio
import rigid_DL.simple_quad_mesh as sqm
import rigid_DL.eigenfunctions as efun
import rigid_DL.input_output as io
import rigid_DL.mat_assembly as mata

def main():
    parser = argp.ArgumentParser(description="Quadratic mesh test")
    parser.add_argument("mesh", help="input quadratic triangular mesh")
    args = parser.parse_args()

    t0 = time.time()
    mesh = meshio.read(args.mesh)
    verts = mesh.points
    for cell_block in mesh.cells:
        if cell_block.type == "triangle6":
            faces = cell_block.data
    t1 = time.time()
    print("{}, meshio read time".format(t1 - t0))
    quad_mesh = sqm.simple_quad_mesh(verts, faces)
    print("Calculated mesh dims: {}".format(quad_mesh.dims))
    C = mata.make_mat_cp_qe(quad_mesh)

    v_cnst = np.array([1, 0, 0])
    #v_trans_in = np.zeros((quad_mesh.lin_verts.shape[0], 1), dtype=v_cnst.dtype) + v_cnst
    v_trans_in = np.zeros((quad_mesh.lin_faces.shape[0], 1), dtype=v_cnst.dtype) + v_cnst
    v_trans_out = np.dot(C, v_trans_in.flatten('C'))
    v_trans_out = v_trans_out.reshape(v_trans_in.shape, order='C')

    E_d, E_c = efun.E_12(quad_mesh)
    v_12_in = efun.make_cp_qe_lin_vels(E_d, E_c, quad_mesh)
    v_12_out = np.dot(C, v_12_in.flatten('C'))
    v_12_out = v_12_out.reshape(v_12_in.shape, order='C')

    t0 = time.time()
    print("{}, matrix forming and dotting walltime".format(t0 - t1))

    w, v = np.linalg.eig(C)
    io.write_eigval(w, "quad_test_eigval.csv")
    #io.write_eigvec(v, "quad_test_eigvec.csv")
    #io.write_vel(v_trans_in, v_trans_out, "quad_test_trans_vel.csv")
    #io.write_vel(v_12_in, v_12_out, "quad_test_E12_vel.csv")
    t1 = time.time()
    print("{}, write out walltime".format(t1 - t0))

if __name__ == "__main__":
    main()
