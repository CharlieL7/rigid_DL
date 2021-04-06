"""
Mesh eigenvector tests
"""
import argparse as argp
import numpy as np
import meshio
import csv
from rigid_DL import lin_geo_mesh
from rigid_DL import quad_geo_mesh
from rigid_DL import cons_pot_mesh
from rigid_DL import lin_pot_mesh
from rigid_DL import quad_pot_mesh
import rigid_DL.mat_assembly as mata
import rigid_DL.eigfun_helper as RDL_eig_helper
import rigid_DL.mobil_helper as RDL_mobil_helper
import rigid_DL.eigenfunctions as RDL_eig_funs
import rigid_DL.eigenvalues as RDL_eig_vals
from rigid_DL.enums import Mesh_Type, Pot_Type


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

    dims = args.dims
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
        #(Pot_Type.QUADRATIC, Mesh_Type.LINEAR): quad_pot_mesh.Quad_Pot_Mesh.make_from_lin_geo_mesh(geo_mesh),
        (Pot_Type.QUADRATIC, Mesh_Type.QUADRATIC): quad_pot_mesh.Quad_Pot_Mesh.make_from_quad_geo_mesh(geo_mesh),
    }
    pot_mesh = pot_mesh_map[(pot_type, mesh_type)]

    stiff_map = {
        (Pot_Type.CONSTANT, Mesh_Type.LINEAR): mata.make_mat_cp_le,
        (Pot_Type.LINEAR, Mesh_Type.LINEAR): mata.make_mat_lp_le,
        (Pot_Type.CONSTANT, Mesh_Type.QUADRATIC): mata.make_mat_cp_qe,
        (Pot_Type.LINEAR, Mesh_Type.QUADRATIC): mata.make_mat_lp_qe,
        (Pot_Type.QUADRATIC, Mesh_Type.QUADRATIC): mata.make_mat_qp_qe,
    }
    K = stiff_map[(pot_type, mesh_type)](pot_mesh, geo_mesh) # stiffness matrix
    
    # RBM eigenfunctions
    num_nodes = pot_mesh.get_nodes().shape[0]

    u_d = np.zeros((num_nodes, 3))
    u_d[:, 0] = 1.
    ret_Tx = cons_eigval_err(
        pot_mesh,
        geo_mesh,
        {
            "K": K,
            "u_d": u_d,
        }
    )

    u_d = np.zeros((num_nodes, 3))
    u_d[:, 1] = 1.
    ret_Ty = cons_eigval_err(
        pot_mesh,
        geo_mesh,
        {
            "K": K,
            "u_d": u_d,
        }
    )

    u_d = np.zeros((num_nodes, 3))
    u_d[:, 2] = 1.
    ret_Tz = cons_eigval_err(
        pot_mesh,
        geo_mesh,
        {
            "K": K,
            "u_d": u_d,
        }
    )

    E_d = np.array([
        [0., 1., 0.,],
        [-1., 0., 0.,],
        [0., 0., 0.,],
    ])
    ret_R12 = lin_eigval_err(
        pot_mesh,
        geo_mesh,
        {
            "K": K,
            "eigval": -1.,
            "E_d": E_d,
            "E_c": np.zeros(3),
            "pot_type": pot_type,
            "mesh_type": mesh_type,
        }
    )

    E_d = np.array([
        [0., 0., 0.,],
        [0., 0., 1.,],
        [0., -1., 0.,],
    ])
    ret_R23 = lin_eigval_err(
        pot_mesh,
        geo_mesh,
        {
            "K": K,
            "eigval": -1.,
            "E_d": E_d,
            "E_c": np.zeros(3),
            "pot_type": pot_type,
            "mesh_type": mesh_type,
        }
    )

    E_d = np.array([
        [0., 0., 1.,],
        [0., 0., 0.,],
        [-1., 0., 0.,],
    ])
    ret_R31 = lin_eigval_err(
        pot_mesh,
        geo_mesh,
        {
            "K": K,
            "eigval": -1.,
            "E_d": E_d,
            "E_c": np.zeros(3),
            "pot_type": pot_type,
            "mesh_type": mesh_type,
        }
    )

    # Linear ROS eigenfunctions
    E_d, E_c = RDL_eig_funs.E_12(dims)
    eigval_12 = RDL_eig_vals.lambda_12(dims)
    ret_E12 = lin_eigval_err(
        pot_mesh,
        geo_mesh,
        {
            "K": K,
            "eigval": eigval_12,
            "E_d": E_d,
            "E_c": E_c,
            "pot_type": pot_type,
            "mesh_type": mesh_type,
        },
    )
    E_d, E_c = RDL_eig_funs.E_23(dims)
    eigval_23 = RDL_eig_vals.lambda_23(dims)
    ret_E23 = lin_eigval_err(
        pot_mesh,
        geo_mesh,
        {
            "K": K,
            "eigval": eigval_23,
            "E_d": E_d,
            "E_c": E_c,
            "pot_type": pot_type,
            "mesh_type": mesh_type,
        },
    )
    E_d, E_c = RDL_eig_funs.E_31(dims)
    eigval_31 = RDL_eig_vals.lambda_31(dims)
    ret_E31 = lin_eigval_err(
        pot_mesh,
        geo_mesh,
        {
            "K": K,
            "eigval": eigval_31,
            "E_d": E_d,
            "E_c": E_c,
            "pot_type": pot_type,
            "mesh_type": mesh_type,
        },
    )
    E_d, E_c = RDL_eig_funs.diag_eigvec("+", dims)
    eigval_p = RDL_eig_vals.lambda_pm("+", dims)
    ret_Ep = lin_eigval_err(
        pot_mesh,
        geo_mesh,
        {
            "K": K,
            "eigval": eigval_p,
            "E_d": E_d,
            "E_c": E_c,
            "pot_type": pot_type,
            "mesh_type": mesh_type,
        },
    )
    E_d, E_c = RDL_eig_funs.diag_eigvec("-", dims)
    eigval_m = RDL_eig_vals.lambda_pm("-", dims)
    ret_Em = lin_eigval_err(
        pot_mesh,
        geo_mesh,
        {
            "K": K,
            "eigval": eigval_m,
            "E_d": E_d,
            "E_c": E_c,
            "pot_type": pot_type,
            "mesh_type": mesh_type,
        },
    )

    # Quadratic flows
    kappa_vec = RDL_eig_vals.calc_3x3_eval(dims)
    eigval_3x3 = -(1 + kappa_vec) / (kappa_vec -1)
    ret_3x3 = quad_eigval_err(
        pot_mesh,
        geo_mesh,
        {
            "K": K,
            "eigval_3x3": eigval_3x3,
            "kappa_vec": kappa_vec,
            "dims": dims,
            "pot_type": pot_type,
            "mesh_type": mesh_type,
        },
    )

    # Write out to vtk file
    cells = [("triangle", pot_mesh.faces)] # to plot point data correctly must be this, might
    # also be able to map point_data to specific points, not sure how to do this with meshio
    if pot_type == Pot_Type.CONSTANT:
        if mesh_type == Mesh_Type.LINEAR:
            cells = [("triangle", geo_mesh.faces)]
        elif mesh_type == Mesh_Type.QUADRATIC:
            cells = [("triangle6", geo_mesh.faces)]
        mesh_io = meshio.Mesh(
            geo_mesh.get_verts(),
            cells,
            cell_data={
                "rel_err_Tx": [ret_Tx["local_relative_L2_error"]],
                "rel_err_Ty": [ret_Ty["local_relative_L2_error"]],
                "rel_err_Tz": [ret_Tz["local_relative_L2_error"]],
                "rel_err_R12": [ret_R12["local_relative_L2_error"]],
                "rel_err_R23": [ret_R23["local_relative_L2_error"]],
                "rel_err_R31": [ret_R31["local_relative_L2_error"]],
                "rel_err_E12": [ret_E12["local_relative_L2_error"]],
                "rel_err_E23": [ret_E23["local_relative_L2_error"]],
                "rel_err_E31": [ret_E31["local_relative_L2_error"]],
                "rel_err_Ep": [ret_Ep["local_relative_L2_error"]],
                "rel_err_Em": [ret_Em["local_relative_L2_error"]],
                "abs_err_E12": [ret_E12["local_absolute_L2_error"]],
                "abs_err_E23": [ret_E23["local_absolute_L2_error"]],
                "abs_err_E31": [ret_E31["local_absolute_L2_error"]],
                "abs_err_Ep": [ret_Ep["local_absolute_L2_error"]],
                "abs_err_Em": [ret_Em["local_absolute_L2_error"]],
                "collin_E12": [ret_E12["local_collinearity"]],
                "collin_E23": [ret_E23["local_collinearity"]],
                "collin_E31": [ret_E31["local_collinearity"]],
                "collin_Ep": [ret_Ep["local_collinearity"]],
                "collin_Em": [ret_Em["local_collinearity"]],
                "abs_err_3x3_0": [ret_3x3["local_absolute_L2_error"][0]],
                "abs_err_3x3_1": [ret_3x3["local_absolute_L2_error"][1]],
                "abs_err_3x3_2": [ret_3x3["local_absolute_L2_error"][2]],
                "rel_err_3x3_0": [ret_3x3["local_relative_L2_error"][0]],
                "rel_err_3x3_1": [ret_3x3["local_relative_L2_error"][1]],
                "rel_err_3x3_2": [ret_3x3["local_relative_L2_error"][2]],
                "collin_3x3_0": [ret_3x3["local_collinearity"][0]],
                "collin_3x3_1": [ret_3x3["local_collinearity"][1]],
                "collin_3x3_2": [ret_3x3["local_collinearity"][2]],
            },
        )
    elif pot_type == Pot_Type.LINEAR:
        cells = [("triangle", pot_mesh.faces)] # to plot point data correctly must be this, might
        # also be able to map point_data to specific points, not sure how to do this with meshio
        mesh_io = meshio.Mesh(
            pot_mesh.get_nodes(),
            cells,
            point_data={
                "rel_err_Tx": ret_Tx["local_relative_L2_error"],
                "rel_err_Ty": ret_Ty["local_relative_L2_error"],
                "rel_err_Tz": ret_Tz["local_relative_L2_error"],
                "rel_err_R12": ret_R12["local_relative_L2_error"],
                "rel_err_R23": ret_R23["local_relative_L2_error"],
                "rel_err_R31": ret_R31["local_relative_L2_error"],
                "rel_err_E12": ret_E12["local_relative_L2_error"],
                "rel_err_E23": ret_E23["local_relative_L2_error"],
                "rel_err_E31": ret_E31["local_relative_L2_error"],
                "rel_err_Ep": ret_Ep["local_relative_L2_error"],
                "rel_err_Em": ret_Em["local_relative_L2_error"],
                "abs_err_E12": ret_E12["local_absolute_L2_error"],
                "abs_err_E23": ret_E23["local_absolute_L2_error"],
                "abs_err_E31": ret_E31["local_absolute_L2_error"],
                "abs_err_Ep": ret_Ep["local_absolute_L2_error"],
                "abs_err_Em": ret_Em["local_absolute_L2_error"],
                "collin_E12": ret_E12["local_collinearity"],
                "collin_E23": ret_E23["local_collinearity"],
                "collin_E31": ret_E31["local_collinearity"],
                "collin_Ep": ret_Ep["local_collinearity"],
                "collin_Em": ret_Em["local_collinearity"],
                "abs_err_3x3_0": ret_3x3["local_absolute_L2_error"][0],
                "abs_err_3x3_1": ret_3x3["local_absolute_L2_error"][1],
                "abs_err_3x3_2": ret_3x3["local_absolute_L2_error"][2],
                "rel_err_3x3_0": ret_3x3["local_relative_L2_error"][0],
                "rel_err_3x3_1": ret_3x3["local_relative_L2_error"][1],
                "rel_err_3x3_2": ret_3x3["local_relative_L2_error"][2],
                "collin_3x3_0": ret_3x3["local_collinearity"][0],
                "collin_3x3_1": ret_3x3["local_collinearity"][1],
                "collin_3x3_2": ret_3x3["local_collinearity"][2],
            },
        )
    elif pot_type == Pot_Type.QUADRATIC:
        if mesh_type == Mesh_Type.LINEAR:
            cells = [("triangle", geo_mesh.faces)]
        elif mesh_type == Mesh_Type.QUADRATIC:
            cells = [("triangle6", geo_mesh.faces)]
        mesh_io = meshio.Mesh(
            pot_mesh.get_nodes(),
            cells,
            point_data={
                "rel_err_Tx": ret_Tx["local_relative_L2_error"],
                "rel_err_Ty": ret_Ty["local_relative_L2_error"],
                "rel_err_Tz": ret_Tz["local_relative_L2_error"],
                "rel_err_R12": ret_R12["local_relative_L2_error"],
                "rel_err_R23": ret_R23["local_relative_L2_error"],
                "rel_err_R31": ret_R31["local_relative_L2_error"],
                "rel_err_E12": ret_E12["local_relative_L2_error"],
                "rel_err_E23": ret_E23["local_relative_L2_error"],
                "rel_err_E31": ret_E31["local_relative_L2_error"],
                "rel_err_Ep": ret_Ep["local_relative_L2_error"],
                "rel_err_Em": ret_Em["local_relative_L2_error"],
                "abs_err_E12": ret_E12["local_absolute_L2_error"],
                "abs_err_E23": ret_E23["local_absolute_L2_error"],
                "abs_err_E31": ret_E31["local_absolute_L2_error"],
                "abs_err_Ep": ret_Ep["local_absolute_L2_error"],
                "abs_err_Em": ret_Em["local_absolute_L2_error"],
                "collin_E12": ret_E12["local_collinearity"],
                "collin_E23": ret_E23["local_collinearity"],
                "collin_E31": ret_E31["local_collinearity"],
                "collin_Ep": ret_Ep["local_collinearity"],
                "collin_Em": ret_Em["local_collinearity"],
                "abs_err_3x3_0": ret_3x3["local_absolute_L2_error"][0],
                "abs_err_3x3_1": ret_3x3["local_absolute_L2_error"][1],
                "abs_err_3x3_2": ret_3x3["local_absolute_L2_error"][2],
                "rel_err_3x3_0": ret_3x3["local_relative_L2_error"][0],
                "rel_err_3x3_1": ret_3x3["local_relative_L2_error"][1],
                "rel_err_3x3_2": ret_3x3["local_relative_L2_error"][2],
                "collin_3x3_0": ret_3x3["local_collinearity"][0],
                "collin_3x3_1": ret_3x3["local_collinearity"][1],
                "collin_3x3_2": ret_3x3["local_collinearity"][2],
            },
        )
    meshio.write("{}.vtk".format(args.out_tag), mesh_io, file_format="vtk")

    ov_data={
        "ov_rel_err_Tx": ret_Tx["overall_relative_L2_error"],
        "ov_rel_err_Ty": ret_Ty["overall_relative_L2_error"],
        "ov_rel_err_Tz": ret_Tz["overall_relative_L2_error"],
        "ov_rel_err_R12": ret_R12["overall_relative_L2_error"],
        "ov_rel_err_R23": ret_R23["overall_relative_L2_error"],
        "ov_rel_err_R31": ret_R31["overall_relative_L2_error"],
        "ov_rel_err_E12": ret_E12["overall_relative_L2_error"],
        "ov_rel_err_E23": ret_E23["overall_relative_L2_error"],
        "ov_rel_err_E31": ret_E31["overall_relative_L2_error"],
        "ov_rel_err_Ep": ret_Ep["overall_relative_L2_error"],
        "ov_rel_err_Em": ret_Em["overall_relative_L2_error"],
        "ov_rel_err_3x3_0": ret_3x3["overall_relative_L2_error"][0],
        "ov_rel_err_3x3_1": ret_3x3["overall_relative_L2_error"][1],
        "ov_rel_err_3x3_2": ret_3x3["overall_relative_L2_error"][2],
    }
    with open("{}_ov_rel.csv".format(args.out_tag), "w", newline='') as csvfile:
        writer = csv.DictWriter(csvfile, ov_data.keys())
        writer.writeheader()
        writer.writerow(ov_data)

    max_data={
        "ov_abs_err_Tx": np.max(ret_Tx["overall_absolute_L2_error"]),
        "ov_abs_err_Ty": np.max(ret_Ty["overall_absolute_L2_error"]),
        "ov_abs_err_Tz": np.max(ret_Tz["overall_absolute_L2_error"]),
        "ov_abs_err_R12": np.max(ret_R12["overall_absolute_L2_error"]),
        "ov_abs_err_R23": np.max(ret_R23["overall_absolute_L2_error"]),
        "ov_abs_err_R31": np.max(ret_R31["overall_absolute_L2_error"]),
        "ov_abs_err_E12": np.max(ret_E12["overall_absolute_L2_error"]),
        "ov_abs_err_E23": np.max(ret_E23["overall_absolute_L2_error"]),
        "ov_abs_err_E31": np.max(ret_E31["overall_absolute_L2_error"]),
        "ov_abs_err_Ep": np.max(ret_Ep["overall_absolute_L2_error"]),
        "ov_abs_err_Em": np.max(ret_Em["overall_absolute_L2_error"]),
        "ov_abs_err_3x3_0": np.max(ret_3x3["overall_absolute_L2_error"][0]),
        "ov_abs_err_3x3_1": np.max(ret_3x3["overall_absolute_L2_error"][1]),
        "ov_abs_err_3x3_2": np.max(ret_3x3["overall_absolute_L2_error"][2]),
    }
    with open("{}_ov_abs.csv".format(args.out_tag), "w", newline='') as csvfile:
        writer = csv.DictWriter(csvfile, max_data.keys())
        writer.writeheader()
        writer.writerow(max_data)


def cons_eigval_err(pot_mesh, geo_mesh, K_ev):
    """
    Constant eigenvalue/eigenvector error function
    The translation eigenvectors, always -1 eigenvalue
    Parameters:
        pot_mesh: potential mesh
        geo_mesh: geometric mesh
        K_ev: dict of {K, u_d}
    """
    K = K_ev["K"]
    u_d = K_ev["u_d"]
    eigval = -1.
    lambda_mat = eigval * np.identity(K.shape[0])
    g = np.dot((lambda_mat - K), np.ravel(u_d))
    ov_abs_err = np.linalg.norm(g)
    ov_rel_err = ov_abs_err / np.linalg.norm(eigval * u_d)
    g = g.reshape(u_d.shape, order="C")
    rel_err_arr = np.linalg.norm(g, axis=1) / np.linalg.norm(u_d[0]) # flow type constant over position
    ret = {
        "overall_absolute_L2_error": ov_abs_err,
        "overall_relative_L2_error": ov_rel_err,
        "local_relative_L2_error": rel_err_arr,
    }
    return ret


def lin_eigval_err(pot_mesh, geo_mesh, K_ev):
    """
    Linear eigenvalue/eigenvector error function
    Parameters:
        pot_mesh: potential mesh
        geo_mesh: geometric mesh
        K_ev: dict of {K, eigval, E_d, E_c, pot_type, mesh_type}
    """
    num_nodes = pot_mesh.get_nodes().shape[0]
    K = K_ev["K"]
    eigval = K_ev["eigval"]
    E_d = K_ev["E_d"]
    E_c = K_ev["E_c"]
    pot_type = K_ev["pot_type"]
    mesh_type = K_ev["mesh_type"]
    psi = RDL_eig_helper.make_lin_eig_vels(pot_mesh, E_d, E_c)
    psi = 
    lambda_mat = eigval * np.identity(K.shape[0])
    out_vec = np.reshape(K @ np.ravel(psi), (num_nodes, 3))
    g = np.dot((lambda_mat - K), np.ravel(psi))
    ov_abs_err = np.linalg.norm(g)
    ov_rel_err = ov_abs_err / np.linalg.norm(eigval * np.ravel(psi))
    g = g.reshape(psi.shape, order="C")
    abs_err_arr = np.linalg.norm(g, axis=1)
    loc_collin = np.einsum(
        "ij,ij->i",
        out_vec / np.linalg.norm(out_vec, axis=1)[:,None],
        psi / np.linalg.norm(psi, axis=1)[:,None]
    )
    rel_err_arr = abs_err_arr / (RDL_eig_helper.calc_ext_flow_magnitude((pot_type, mesh_type), pot_mesh, geo_mesh, eigval * psi))
    ret = {
        "overall_absolute_L2_error": ov_abs_err,
        "overall_relative_L2_error": ov_rel_err,
        "local_absolute_L2_error": abs_err_arr,
        "local_relative_L2_error":rel_err_arr,
        "local_collinearity": loc_collin,
        "eigenvalue": eigval,
    }
    return ret


def quad_eigval_err(pot_mesh, geo_mesh, K_ev):
    """
    Quadratic 3x3 system eigenfunction error function

    Parameters:
        pot_mesh: potential mesh
        geo_mesh: geometric mesh
        dims: ellipsoidal dimensions
        K_ev: dict of {K, eigval_3x3, dims, kappa_vec, pot_type, mesh_type}
    """
    num_nodes = pot_mesh.get_nodes().shape[0]
    K = K_ev["K"]
    eigval_3x3 = K_ev["eigval_3x3"]
    dims = K_ev["dims"]
    kappa_vec = K_ev["kappa_vec"]
    pot_type = K_ev["pot_type"]
    mesh_type = K_ev["mesh_type"]
    eigval_3x3 = -(1 + kappa_vec) / (kappa_vec -1)
    v_3x3_in = RDL_eig_helper.make_quad_eig_vels(pot_mesh, dims, kappa_vec)
    abs_err_3x3 = []
    rel_err_3x3 = []
    loc_collin_3x3 = []
    ov_abs_err_3x3 = []
    ov_rel_err_3x3 = []
    for i, v_in in enumerate(v_3x3_in):
        out_vec = np.reshape(K @ np.ravel(v_in), (num_nodes, 3))
        loc_collin_3x3.append(
            np.einsum(
                "ij,ij->i",
                out_vec / np.linalg.norm(out_vec, axis=1)[:,None],
                v_in / np.linalg.norm(v_in, axis=1)[:,None]
            )
        )
        lambda_mat = eigval_3x3[i] * np.identity(K.shape[0])
        tmp_err = np.dot((lambda_mat - K), v_in.flatten("C"))
        ov_abs_err_3x3.append(np.linalg.norm(tmp_err))
        ov_rel_err_3x3.append(np.linalg.norm(tmp_err) / np.linalg.norm(eigval_3x3[i] * np.ravel(v_in)))
        tmp_err = tmp_err.reshape(v_in.shape, order="C")
        tmp_err = np.linalg.norm(tmp_err, axis=1)
        abs_err_3x3.append(tmp_err)
        rel_err_3x3.append(tmp_err / (RDL_eig_helper.calc_ext_flow_magnitude((pot_type, mesh_type), pot_mesh, geo_mesh, eigval_3x3[i] * v_in)))

    ret = {
        "overall_absolute_L2_error": ov_abs_err_3x3,
        "overall_relative_L2_error": ov_rel_err_3x3,
        "local_absolute_L2_error": abs_err_3x3,
        "local_relative_L2_error":rel_err_3x3,
        "local_collinearity": loc_collin_3x3,
        "eigenvalues": eigval_3x3,
    }
    return ret


if __name__ == "__main__":
    main()
