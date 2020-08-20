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
    parser.add_argument("cl", help="constant or linear potentials")
    parser.add_argument("-o", "--out_tag", help="tag to prepend to output files")
    args = parser.parse_args()

    mesh_name = args.mesh
    const_or_linear = args.cl
    assert const_or_linear in ('c', 'l')
    out_name = "test"
    if args.out_tag:
        out_name = args.out_tag

    t0 = time.time()

    (x_data, f2v, _params) = io.read_short_dat(mesh_name)
    f2v = f2v - 1 # indexing change
    mesh = slm.simple_linear_mesh(x_data, f2v)
    #v_in = efun.diag_eigvec("-", mesh, const_or_linear)
    v_cnst = np.array([1, 0, 0])
    v_trans_in = efun.make_translation_vels(v_cnst, mesh, const_or_linear)
    v_12_in = efun.E_12(mesh, const_or_linear)

    t1 = time.time()
    print("{}, before construct stiffness matrix".format(t1 - t0))

    if const_or_linear == 'c':
        C = mata.make_mat_cp_le(mesh) # stiffness matrix
    elif const_or_linear == 'l':
        C = mata.make_mat_lp_le(mesh) # stiffness matrix

    v_trans_out = np.dot(C, v_trans_in.flatten('C'))
    v_trans_out = v_trans_out.reshape(v_trans_in.shape, order='C')
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
