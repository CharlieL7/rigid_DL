"""
Mesh eigenvector tests
"""
import argparse as argp
from enum import Enum
import numpy as np
import meshio
import rigid_DL.input_output as io
import rigid_DL.simple_linear_mesh as SLM
import rigid_DL.simple_quad_mesh as SQM
import rigid_DL.mat_assembly as mata
import rigid_DL.eigenfunctions as efun
import rigid_DL.eigenvalues as eigvals
import rigid_DL.gauss_quad as gq

def main():
    parser = argp.ArgumentParser(description="Double layer eigenvector tests that handles all parameterizations.")
    parser.add_argument("mesh", help="mesh file input, parameterization determined by file")
    parser.add_argument("dims", nargs=3, type=float, help="expected mesh dimensions")
    parser.add_argument(
        "-o",
        "--out_tag",
        help="tag to prepend to output files",
        default="combined_test"
    )
    parser.add_argument(
        "-p", "--potential",
        type=int,
        help="parameterization for potential (0 = constant, 1 = linear)",
        default=0
    )
    args = parser.parse_args()

    out_name = args.out_tag
    expected_dims = args.dims
    pot_type = Pot_Type(args.potential)

    io_mesh = meshio.read(args.mesh)
    verts = io_mesh.points
    for cell_block in io_mesh.cells:
        if cell_block.type == "triangle":
            faces = cell_block.data
            mesh_type = Mesh_Type.LINEAR
            break
        if cell_block.type == "triangle6":
            faces = cell_block.data
            mesh_type = Mesh_Type.QUADRATIC
            break

    if mesh_type == Mesh_Type.LINEAR:
        mesh = SLM.simple_linear_mesh(verts, faces)
    elif mesh_type == Mesh_Type.QUADRATIC:
        mesh = SQM.simple_quad_mesh(verts, faces)

    stiff_map = {
        (Pot_Type.CONSTANT, Mesh_Type.LINEAR): mata.make_mat_cp_le,
        (Pot_Type.LINEAR, Mesh_Type.LINEAR): mata.make_mat_lp_le,
        (Pot_Type.CONSTANT, Mesh_Type.QUADRATIC): mata.make_mat_cp_qe,
        (Pot_Type.LINEAR, Mesh_Type.QUADRATIC): mata.make_mat_lp_qe,
    }
    C =stiff_map[(pot_type, mesh_type)](mesh) # stiffness matrix

    E_d, E_c = efun.E_12(mesh)
    eigval_12 = eigvals.lambda_12(expected_dims)
    abs_err_12, v_in_12, percent_err_12 = eigval_err(mesh, pot_type, mesh_type, C, eigval_12, E_d, E_c)

    E_d, E_c = efun.E_23(mesh)
    eigval_23 = eigvals.lambda_23(expected_dims)
    abs_err_23, v_in_23, percent_err_23 = eigval_err(mesh, pot_type, mesh_type, C, eigval_23, E_d, E_c)

    E_d, E_c = efun.E_31(mesh)
    eigval_31 = eigvals.lambda_31(expected_dims)
    abs_err_31, v_in_31, percent_err_31 = eigval_err(mesh, pot_type, mesh_type, C, eigval_31, E_d, E_c)

    E_d, E_c = efun.diag_eigvec("+", mesh)
    eigval_p = eigvals.lambda_pm("+", expected_dims)
    abs_err_p, v_in_p, percent_err_p = eigval_err(mesh, pot_type, mesh_type, C, eigval_p, E_d, E_c)

    E_d, E_c = efun.diag_eigvec("-", mesh)
    eigval_m = eigvals.lambda_pm("-", expected_dims)
    abs_err_m, v_in_m, percent_err_m = eigval_err(mesh, pot_type, mesh_type, C, eigval_m, E_d, E_c)

    if mesh_type == Mesh_Type.LINEAR:
        cells = [("triangle", mesh.faces)]
    elif mesh_type == Mesh_Type.QUADRATIC:
        cells = [("triangle6", mesh.faces)]

    if pot_type == Pot_Type.CONSTANT:
        mesh_io = meshio.Mesh(
            mesh.vertices,
            cells,
            cell_data={
                "abs_err_12": [abs_err_12],
                "v_in_12": [v_in_12],
                "normalized_err_12": [percent_err_12],

                "abs_err_23": [abs_err_23],
                "v_in_23": [v_in_23],
                "normalized_err_23": [percent_err_23],

                "abs_err_31": [abs_err_31],
                "v_in_31": [v_in_31],
                "normalized_err_31": [percent_err_31],

                "abs_err_p": [abs_err_p],
                "v_in_p": [v_in_p],
                "normalized_err_p": [percent_err_p],

                "abs_err_m": [abs_err_m],
                "v_in_m": [v_in_m],
                "normalized_err_m": [percent_err_m],
            }
        )
    elif pot_type == Pot_Type.LINEAR:
        mesh_io = meshio.Mesh(
            mesh.vertices,
            cells,
            point_data={
                "abs_err_12": abs_err_12,
                "v_in_12": v_in_12,
                "normalized_err_12": percent_err_12,

                "abs_err_23": abs_err_23,
                "v_in_23":v_in_23,
                "normalized_err_23": percent_err_23,

                "abs_err_31": abs_err_31,
                "v_in_31": v_in_31,
                "normalized_err_31": percent_err_31,

                "abs_err_p": abs_err_p,
                "v_in_p": v_in_p,
                "normalized_err_p": percent_err_p,

                "abs_err_m": abs_err_m,
                "v_in_m": v_in_m,
                "normalized_err_m": percent_err_m,
            }
        )
    meshio.write("{}_out.vtk".format(out_name), mesh_io, file_format="vtk")


def eigval_err(mesh, pot_type, mesh_type, C, eigval, E_d, E_c):
    """
    Linear eigenvalue/eigenvector error function
    """
    tol = 1e-6 #lower bound for norm about equal to zero
    v_fun_map = {
        (Pot_Type.CONSTANT, Mesh_Type.LINEAR): efun.make_cp_le_lin_vels,
        (Pot_Type.LINEAR, Mesh_Type.LINEAR): efun.make_lp_le_lin_vels,
        (Pot_Type.CONSTANT, Mesh_Type.QUADRATIC): efun.make_cp_qe_lin_vels,
        (Pot_Type.LINEAR, Mesh_Type.QUADRATIC): efun.make_lp_qe_lin_vels,
    }
    v_in = v_fun_map[(pot_type, mesh_type)](E_d, E_c, mesh)
    lambda_mat = eigval * np.identity(C.shape[0])
    g = np.dot((lambda_mat - C), v_in.flatten("C"))
    g = g.reshape(v_in.shape, order="C")
    err_arr = np.linalg.norm(g, axis=1)
    v_in_norms = np.linalg.norm(v_in, axis=1)
    # only divides when v_in_norm is > than tol, otherwise sets to zero
    per_err_arr = np.divide(err_arr, v_in_norms, out=np.zeros_like(err_arr), where=v_in_norms>tol)
    if (pot_type == Pot_Type.LINEAR and mesh_type == Mesh_Type.QUADRATIC):
        # fill unused verticies with lp_qe parameterization with zeros
        num_quad_verts = mesh.vertices.shape[0]
        quad_err_arr = np.zeros(num_quad_verts)
        quad_v_in_arr = np.zeros((num_quad_verts, 3))
        quad_per_err_arr = np.zeros(num_quad_verts)
        for i, err in enumerate(err_arr):
            quad_ind = mesh.lin_to_quad_map[i]
            quad_err_arr[quad_ind] = err
            quad_v_in_arr[quad_ind] = v_in[i]
            quad_per_err_arr[quad_ind] = per_err_arr[i]
        return(quad_err_arr, quad_v_in_arr, quad_per_err_arr)
    return (err_arr, v_in, per_err_arr)


def calc_avg_v_in_norm(mesh, mesh_type, E_d, E_c):
    """
    Surface average of eigenfunction
    """
    avg_v_in_norm = 0.
    if mesh_type == Mesh_Type.LINEAR:
        for i, face in enumerate(mesh.faces):
            nodes = mesh.get_nodes(face)
            avg_v_in_norm += gq.int_over_tri_lin(
                efun.make_lin_psi_func(E_d, E_c),
                nodes,
                mesh.hs[i],
            )
    elif mesh_type == Mesh_Type.QUADRATIC:
        for i, face in enumerate(mesh.faces):
            nodes = mesh.get_nodes(face)
            avg_v_in_norm += gq.int_over_tri_quad(
                efun.make_lin_psi_func(E_d, E_c),
                nodes,
                mesh.hs[i],
            )
    avg_v_in_norm /= mesh.surf_area
    return avg_v_in_norm


class Mesh_Type(Enum):
    LINEAR = 1
    QUADRATIC = 2


class Pot_Type(Enum):
    CONSTANT = 0
    LINEAR = 1

if __name__ == "__main__":
    main()
