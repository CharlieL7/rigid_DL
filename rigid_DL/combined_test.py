"""
Mesh eigenvector tests
"""
import argparse as argp
from enum import Enum
import numpy as np
import meshio
from rigid_DL import lin_geo_mesh, quad_geo_mesh, cons_pot_mesh, lin_pot_mesh
import rigid_DL.mat_assembly as mata
import rigid_DL.eigenfunctions as eigfuns
import rigid_DL.eigenvalues as eigvals
import rigid_DL.gauss_quad as gq
import rigid_DL.eigfun_helper as eig_helper


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
        geo_mesh = lin_geo_mesh.Lin_Geo_Mesh(verts, faces)
    elif mesh_type == Mesh_Type.QUADRATIC:
        geo_mesh = quad_geo_mesh.Quad_Geo_Mesh(verts, faces)

    pot_mesh_map = {
        (Pot_Type.CONSTANT, Mesh_Type.LINEAR): cons_pot_mesh.Cons_Pot_Mesh.make_from_geo_mesh(geo_mesh),
        (Pot_Type.CONSTANT, Mesh_Type.QUADRATIC): cons_pot_mesh.Cons_Pot_Mesh.make_from_geo_mesh(geo_mesh),
        (Pot_Type.LINEAR, Mesh_Type.LINEAR): lin_pot_mesh.Lin_Pot_Mesh.make_from_lin_geo_mesh(geo_mesh),
        (Pot_Type.LINEAR, Mesh_Type.QUADRATIC): lin_pot_mesh.Lin_Pot_Mesh.make_from_quad_geo_mesh(geo_mesh),
    }
    pot_mesh = pot_mesh_map[(pot_type, mesh_type)]

    stiff_map = {
        (Pot_Type.CONSTANT, Mesh_Type.LINEAR): mata.make_mat_cp_le,
        (Pot_Type.LINEAR, Mesh_Type.LINEAR): mata.make_mat_lp_le,
        (Pot_Type.CONSTANT, Mesh_Type.QUADRATIC): mata.make_mat_cp_qe,
        (Pot_Type.LINEAR, Mesh_Type.QUADRATIC): mata.make_mat_lp_qe,
    }
    C = stiff_map[(pot_type, mesh_type)](pot_mesh, geo_mesh) # stiffness matrix

    # Linear eigenfunctions
    E_d, E_c = eigfuns.E_12(geo_mesh, expected_dims)
    eigval_12 = eigvals.lambda_12(expected_dims)
    abs_err_12, v_in_12, percent_err_12 = lin_eigval_err(
        pot_mesh,
        {"C": C, "eigval": eigval_12, "E_d": E_d, "E_c": E_c},
    )

    E_d, E_c = eigfuns.E_23(geo_mesh, expected_dims)
    eigval_23 = eigvals.lambda_23(expected_dims)
    abs_err_23, v_in_23, percent_err_23 = lin_eigval_err(
        pot_mesh,
        {"C": C, "eigval": eigval_23, "E_d": E_d, "E_c": E_c},
    )

    E_d, E_c = eigfuns.E_31(geo_mesh, expected_dims)
    eigval_31 = eigvals.lambda_31(expected_dims)
    abs_err_31, v_in_31, percent_err_31 = lin_eigval_err(
        pot_mesh,
        {"C": C, "eigval": eigval_31, "E_d": E_d, "E_c": E_c},
    )

    E_d, E_c = eigfuns.diag_eigvec("+", geo_mesh, expected_dims)
    eigval_p = eigvals.lambda_pm("+", expected_dims)
    abs_err_p, v_in_p, percent_err_p = lin_eigval_err(
        pot_mesh,
        {"C": C, "eigval": eigval_p, "E_d": E_d, "E_c": E_c},
    )

    E_d, E_c = eigfuns.diag_eigvec("-", geo_mesh, expected_dims)
    eigval_m = eigvals.lambda_pm("-", expected_dims)
    abs_err_m, v_in_m, percent_err_m = lin_eigval_err(
        pot_mesh,
        {"C": C, "eigval": eigval_m, "E_d": E_d, "E_c": E_c},
    )

    if np.allclose(expected_dims, [1., 1., 1.,]):
        print("Sphere, setting quadratic ROS eigenfunctions to 0")
        abs_err_3x3 = np.zeros((3, pot_mesh.get_nodes().shape[0], 3))
        v_in_3x3 = abs_err_3x3
        percent_err_3x3 = abs_err_3x3
    else:
        # 3x3 system quadratic eigenfunctions
        kappa_vec = eigvals.calc_3x3_eval(expected_dims)
        abs_err_3x3, v_in_3x3, percent_err_3x3 = quad_eigval_err(pot_mesh, C, expected_dims, kappa_vec)

    if pot_type == Pot_Type.CONSTANT:
        if mesh_type == Mesh_Type.LINEAR:
            cells = [("triangle", geo_mesh.faces)]
        elif mesh_type == Mesh_Type.QUADRATIC:
            cells = [("triangle6", geo_mesh.faces)]
        mesh_io = meshio.Mesh(
            geo_mesh.get_verts(),
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

                "abs_err_330": [abs_err_3x3[0]],
                "v_in_330": [v_in_3x3[0]],
                "normalized_err_330": [percent_err_3x3[0]],

                "abs_err_331": [abs_err_3x3[1]],
                "v_in_331": [v_in_3x3[1]],
                "normalized_err_331": [percent_err_3x3[1]],

                "abs_err_332": [abs_err_3x3[2]],
                "v_in_332": [v_in_3x3[2]],
                "normalized_err_332": [percent_err_3x3[2]],
            }
        )
    elif pot_type == Pot_Type.LINEAR:
        cells = [("triangle", pot_mesh.faces)] # to plot point data correctly must be this, might
        # also be able to map point_data to specific points, not sure how to do this with meshio
        mesh_io = meshio.Mesh(
            pot_mesh.get_nodes(),
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

                "abs_err_330": abs_err_3x3[0],
                "v_in_330": v_in_3x3[0],
                "normalized_err_330": percent_err_3x3[0],

                "abs_err_331": abs_err_3x3[1],
                "v_in_331": v_in_3x3[1],
                "normalized_err_331": percent_err_3x3[1],

                "abs_err_332": abs_err_3x3[2],
                "v_in_332":v_in_3x3[2],
                "normalized_err_332": percent_err_3x3[2],
            }
        )
    meshio.write("{}_out.vtk".format(out_name), mesh_io, file_format="vtk")


def lin_eigval_err(pot_mesh, C_ev):
    """
    Linear eigenvalue/eigenvector error function

    Parameters:
        pot_mesh: potential mesh
        C_ev: dict of {K, eigval, E_d, E_v}
    Returns:
        err_arr: linear error at each node
        v_in: eigenfunction at each node
        per_err_arr: linear error normalized by L2 norm at node
    """
    C = C_ev["C"]
    eigval = C_ev["eigval"]
    E_d = C_ev["E_d"]
    E_c = C_ev["E_c"]
    tol = 1e-6 #lower bound for norm about equal to zero
    v_in = eig_helper.make_lin_eig_vels(pot_mesh, E_d, E_c)
    lambda_mat = eigval * np.identity(C.shape[0])
    g = np.dot((lambda_mat - C), v_in.flatten("C"))
    g = g.reshape(v_in.shape, order="C")
    err_arr = np.linalg.norm(g, axis=1)
    base = eigval * v_in
    base = np.linalg.norm(base, axis=1)
    # only divides when base is > than tol, otherwise sets to zero
    per_err_arr = np.divide(err_arr, base, out=np.zeros_like(err_arr), where=base>tol)
    return (err_arr, v_in, per_err_arr)


def quad_eigval_err(pot_mesh, C, dims, kappa_vec):
    """
    Quadratic 3x3 system eigenfunction error function

    Parameters:
        pot_mesh: potential mesh
        C: discrete DL operator for mesh
        dims: ellipsoidal dimensions
        kappa_vec: the three kappa values for the 3x3 system associated with
        the eigenfunctions
    """
    tol = 1e-6 #lower bound for norm about equal to zero
    eigval_3x3 = -(1 + kappa_vec) / (kappa_vec -1)
    v_3x3_in = eig_helper.make_quad_eig_vels(pot_mesh, dims, kappa_vec)
    err_3x3 = []
    v_in_norm_3x3 = []
    per_err_3x3 = []
    for i, v_in in enumerate(v_3x3_in):
        lambda_mat = eigval_3x3[i] * np.identity(C.shape[0])
        tmp_v_in_norm = np.linalg.norm(v_in, axis=1)
        tmp_err = np.dot((lambda_mat - C), v_in.flatten("C"))
        tmp_err = tmp_err.reshape(v_in.shape, order="C")
        tmp_err = np.linalg.norm(tmp_err, axis=1)
        base = eigval_3x3[i] * v_in
        base = np.linalg.norm(base, axis=1)
        tmp_per_err = np.divide(tmp_err, base, out=np.zeros_like(tmp_err), where=base>tol)
        v_in_norm_3x3.append(tmp_v_in_norm)
        err_3x3.append(tmp_err)
        per_err_3x3.append(tmp_per_err)
    return (err_3x3, v_in_norm_3x3, per_err_3x3)


def calc_avg_v_in_norm(mesh, mesh_type, E_d, E_c):
    """
    Surface average of eigenfunction
    """
    avg_v_in_norm = 0.
    if mesh_type == Mesh_Type.LINEAR:
        for i, face in enumerate(mesh.faces):
            nodes = mesh.get_nodes(face)
            avg_v_in_norm += gq.int_over_tri_lin(
                eigfuns.make_lin_psi_func(E_d, E_c),
                nodes,
                mesh.hs[i],
            )
    elif mesh_type == Mesh_Type.QUADRATIC:
        for i, face in enumerate(mesh.faces):
            nodes = mesh.get_nodes(face)
            avg_v_in_norm += gq.int_over_tri_quad(
                eigfuns.make_lin_psi_func(E_d, E_c),
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
