"""
Linear mesh eigenvalue tests
"""
import time
import argparse as argp
import numpy as np
import input_output as io
import simple_linear_mesh as slm
import mat_assembly as mata
import eigenfunctions as efun

def main():
    parser = argp.ArgumentParser(description="Linear mesh eigenvalue tests")
    parser.add_argument("mesh", help="linear mesh file input")
    parser.add_argument("-o", "--out_tag", help="tag to prepend to output files")
    args = parser.parse_args()

    mesh_name = args.mesh
    out_name = "test"
    if args.out_tag:
        out_name = args.out_tag

    t0 = time.time()

    (x_data, f2v, _params) = io.read_short_dat(mesh_name)
    f2v = f2v - 1 # indexing change
    mesh = slm.simple_linear_mesh(x_data, f2v)


    t1 = time.time()
    print("{}, before construct stiffness matrix".format(t1 - t0))

    C = mata.make_mat_cp_le(mesh) # stiffness matrix

    v_cnst = np.array([1, 0, 0])
    v_trans_in = np.zeros((mesh.faces.shape[0], 1), dtype=v_cnst.dtype) + v_cnst
    v_trans_out = np.dot(C, v_trans_in.flatten('C'))
    v_trans_out = v_trans_out.reshape(v_trans_in.shape, order='C')

    E_d, E_c = efun.E_12(mesh)
    v_12_in = efun.make_cp_le_lin_vels(E_d, E_c, mesh)
    v_12_out = np.dot(C, v_12_in.flatten('C'))
    v_12_out = v_12_out.reshape(v_12_in.shape, order='C')

    t2 = time.time()
    print("{}, matrix forming and dotting walltime".format(t2 - t1))

    #w, v = np.linalg.eig(C)
    #io.write_eigval(w, "{}_eigval.csv".format(out_name))
    #io.write_eigvec(v, "{}_eigvec.csv".format(out_name))
    io.write_vel(v_trans_in, v_trans_out, "{}_trans_vel.csv".format(out_name))
    io.write_vel(v_12_in, v_12_out, "{}_12_vel.csv".format(out_name))

    t3 = time.time()
    print("{}, write out walltime".format(t3 - t2))


if __name__ == "__main__":
    main()
