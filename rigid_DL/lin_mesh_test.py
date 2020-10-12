"""
Linear mesh eigenvalue tests
"""
import argparse as argp
import numpy as np
import meshio
import rigid_DL.input_output as io
import rigid_DL.simple_linear_mesh as SLM
import rigid_DL.mat_assembly as mata
import rigid_DL.eigenfunctions as e_fun
import rigid_DL.eigenvalues as e_val

def main():
    parser = argp.ArgumentParser(description="Linear mesh eigenvalue tests")
    parser.add_argument("mesh", help="linear mesh file input")
    parser.add_argument("dims", nargs=3, type=float, help="expected mesh dimensions")
    parser.add_argument(
        "-o",
        "--out_tag",
        help="tag to prepend to output files",
        default="lin_mesh_test"
    )
    args = parser.parse_args()

    expected_dims = args.dims
    out_name = args.out_tag
    if args.out_tag:
        out_name = args.out_tag

    out_name = args.out_tag

    io_mesh = meshio.read(args.mesh)
    verts = io_mesh.points
    for cell_block in io_mesh.cells:
        if cell_block.type == "triangle":
            faces = cell_block.data
            break

    mesh = SLM.simple_linear_mesh(verts, faces)


    C = mata.make_mat_lp_le(mesh) # stiffness matrix

    """
    v_cnst = np.array([1, 0, 0])
    v_trans_in = np.zeros((mesh.vertices.shape[0], 1), dtype=v_cnst.dtype) + v_cnst
    v_trans_out = np.dot(C, v_trans_in.flatten('C'))
    v_trans_out = v_trans_out.reshape(v_trans_in.shape, order='C')

    E_d, E_c = e_fun.E_12(mesh)
    v_12_in = e_fun.make_lp_le_lin_vels(E_d, E_c, mesh)
    v_12_out = np.dot(C, v_12_in.flatten('C'))
    v_12_out = v_12_out.reshape(v_12_in.shape, order='C')

    E_d, E_c = e_fun.diag_eigvec("+", mesh)
    v_p_in = e_fun.make_cp_le_lin_vels(E_d, E_c, mesh)
    v_p_out = np.dot(C, v_p_in.flatten("C"))
    v_p_out = v_p_out.reshape(v_p_in.shape, order="C")
    """
    kappa_vec = e_val.calc_3x3_eval(expected_dims)
    e_val_3x3 = -(1 + kappa_vec) / (kappa_vec -1)
    print("3x3 eigenvalues: {}".format(e_val_3x3))

    num_vert = mesh.vertices.shape[0]
    v_3x3_in = np.empty((3, num_vert, 3))
    for i, kappa in enumerate(kappa_vec):
        H = e_fun.calc_3x3_evec(expected_dims, kappa)
        for m in range(num_vert):
            vert = mesh.vertices[m]
            x_tmp = np.array([
                [vert[1] * vert[2]],
                [vert[2] * vert[0]],
                [vert[0] * vert[1]],
            ])
            v_3x3_in[i, m] = np.ravel(H * x_tmp)

    v_3x3_out = np.empty((3, num_vert, 3))
    for i, v_in in enumerate(v_3x3_in):
        v_3x3_out[i] = (np.dot(C, v_in.flatten("C"))).reshape(v_in.shape, order="C")

    #w, v = np.linalg.eig(C)
    #io.write_eigval(w, "{}_eigval.csv".format(out_name))
    #io.write_eigvec(v, "{}_eigvec.csv".format(out_name))
    #io.write_vel(v_trans_in, v_trans_out, "{}_trans_vel.csv".format(out_name))
    #io.write_vel(v_12_in, v_12_out, "{}_12_vel.csv".format(out_name))
    #io.write_vel(v_p_in, v_p_out, "{}_p_vel.csv".format(out_name))
    for i, v_in in enumerate(v_3x3_in):
        io.write_vel(v_in, v_3x3_out[i], "{}_33{}_vel.csv".format(out_name, i))


if __name__ == "__main__":
    main()
