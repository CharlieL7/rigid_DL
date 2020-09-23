"""
Linear mesh eigenvalue tests
"""
import argparse as argp
import numpy as np
import meshio
import rigid_DL.input_output as io
import rigid_DL.simple_linear_mesh as SLM
import rigid_DL.mat_assembly as mata
import rigid_DL.eigenfunctions as efun
import rigid_DL.eigenvalues as eigvals

def main():
    parser = argp.ArgumentParser(description="Linear potential, linear element eigenvalue tests")
    parser.add_argument("mesh", help="linear mesh file input")
    parser.add_argument("dims", nargs=3, type=float, help="expected mesh dimensions")
    parser.add_argument("-o", "--out_tag", help="tag to prepend to output files", default="lp_le_test")
    args = parser.parse_args()

    out_name = args.out_tag
    expected_dims = args.dims

    io_mesh = meshio.read(args.mesh)
    verts = io_mesh.points
    for cell_block in io_mesh.cells:
        if cell_block.type == "triangle":
            faces = cell_block.data

    slm = SLM.simple_linear_mesh(verts, faces)

    C = mata.make_mat_lp_le(slm) # stiffness matrix

    cells = [("triangle", slm.faces)]

    E_d, E_c = efun.E_12(slm)
    eigval_12 = eigvals.lambda_12(expected_dims)
    mse_err_12, v_in_norm_12, err_12 = eigval_err(slm, C, eigval_12, E_d, E_c)

    E_d, E_c = efun.E_23(slm)
    eigval_23 = eigvals.lambda_23(expected_dims)
    mse_err_23, v_in_norm_23, err_23 = eigval_err(slm, C, eigval_23, E_d, E_c)

    E_d, E_c = efun.E_31(slm)
    eigval_31 = eigvals.lambda_31(expected_dims)
    mse_err_31, v_in_norm_31, err_31 = eigval_err(slm, C, eigval_31, E_d, E_c)

    E_d, E_c = efun.diag_eigvec("+", slm)
    eigval_p = eigvals.lambda_pm("+", expected_dims)
    mse_err_p, v_in_norm_p, err_p = eigval_err(slm, C, eigval_p, E_d, E_c)

    E_d, E_c = efun.diag_eigvec("-", slm)
    eigval_m = eigvals.lambda_pm("-", expected_dims)
    mse_err_m, v_in_norm_m, err_m = eigval_err(slm, C, eigval_m, E_d, E_c)

    mesh_io = meshio.Mesh(
        slm.vertices,
        cells,
        point_data={
            "mse_err_12": mse_err_12,
            "v_in_norm_12": v_in_norm_12,
            "percent_err_12": err_12,

            "mse_err_23": mse_err_23,
            "v_in_norm_23": v_in_norm_23,
            "percent_err_23": err_23,

            "mse_err_31": mse_err_31,
            "v_in_norm_31": v_in_norm_31,
            "percent_err_31": err_31,

            "mse_err_p": mse_err_p,
            "v_in_norm_p": v_in_norm_p,
            "percent_err_p": err_p,

            "mse_err_m": mse_err_m,
            "v_in_norm_m": v_in_norm_m,
            "percent_err_m": err_m,
        }
    )
    meshio.write("{}_out.vtk".format(out_name), mesh_io, file_format="vtk")

    #w, v = np.linalg.eig(C)
    #io.write_eigval(w, "{}_eigval.csv".format(out_name))
    #io.write_eigvec(v, "{}_eigvec.csv".format(out_name))
    #io.write_vel(v_trans_in, v_trans_out, "{}_trans_vel.csv".format(out_name))
    #io.write_vel(v_12_in, v_12_out, "{}_12_vel.csv".format(out_name))
    #io.write_vel(v_p_in, v_p_out, "{}_p_vel.csv".format(out_name))


def eigval_err(mesh, C, eigval, E_d, E_c):
    v_in = efun.make_lp_le_lin_vels(E_d, E_c, mesh)
    lambda_mat = eigval * np.identity(C.shape[0])
    g = np.dot((lambda_mat - C), v_in.flatten("C"))
    g = g.reshape(v_in.shape, order="C")
    MSE = 1/3. * np.einsum("ij,ij->i", g, g)
    v_in_norms = np.einsum("ij,ij->i", v_in, v_in)
    return (MSE, v_in_norms, MSE / v_in_norms)


def trans_eigval_err(mesh, C):
    # error in mean squared value sense at each vertex: (v_analytical - v_discrete)
    v_cnst = np.array([1, 0, 0])
    v_in = np.zeros((mesh.vertices.shape[0], 1), dtype=v_cnst.dtype) + v_cnst
    eigval = -1.
    lambda_mat = eigval * np.identity(C.shape[0])
    g = np.dot((lambda_mat - C), v_in.flatten("C"))
    g = g.reshape(v_in.shape, order="C")
    MSE = 1/3. * np.einsum("ij,ij->i", g, g)
    v_in_norms = np.einsum("ij,ij->i", v_in, v_in)
    return (MSE, v_in_norms, MSE / v_in_norms)


if __name__ == "__main__":
    main()
