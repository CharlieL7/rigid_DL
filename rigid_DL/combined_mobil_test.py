"""
Testing the stiffness matrix assembly for the mobility problem
"""
import sys
import csv
from enum import Enum
import argparse as argp
import numpy as np
import meshio
import rigid_DL.mobil_assembly as m_a
from rigid_DL import lin_geo_mesh, quad_geo_mesh, cons_pot_mesh, lin_pot_mesh
import rigid_DL.eigfun_helper as RDL_eig_helper
import rigid_DL.mobil_helper as RDL_mobil_helper
import rigid_DL.eigenfunctions as RDL_eig_funs
import rigid_DL.eigenvalues as RDL_eig_vals

def main():
    parser = argp.ArgumentParser(description="Testing the mobility problem for set of linear and quadratic flows")
    parser.add_argument("mesh", help="mesh file input, parameterization determined by file")
    parser.add_argument("dims", nargs=3, type=float, help="expected mesh dimensions")
    parser.add_argument(
        "-o",
        "--out_tag",
        help="tag to prepend to output files",
        default="combined_mobil_test"
    )
    parser.add_argument(
        "-m",
        "--mu",
        help="viscosity",
        default=1.
    )
    parser.add_argument(
        "-p", "--potential",
        type=int,
        help="parameterization for potential (0 = constant, 1 = linear)",
        default=0
    )
    args = parser.parse_args()
    io_mesh = meshio.read(args.mesh)
    pot_type = Pot_Type(args.potential)
    mu = args.mu
    f = np.array([0., 0., 0.,]) # force
    l = np.array([0., 0., 0,]) # torque

    verts = io_mesh.points
    for cell_block in io_mesh.cells:
        if cell_block.type == "triangle":
            faces = cell_block.data
            mesh_type = Mesh_Type.LINEAR
            geo_mesh = lin_geo_mesh.Lin_Geo_Mesh(verts, faces)
            break
        if cell_block.type == "triangle6":
            faces = cell_block.data
            mesh_type = Mesh_Type.QUADRATIC
            geo_mesh = quad_geo_mesh.Quad_Geo_Mesh(verts, faces)
            break

    dims = args.dims
    #dims = geo_mesh.dims
    print("dims:")
    print(dims)

    pot_mesh_map = {
        (Pot_Type.CONSTANT, Mesh_Type.LINEAR): cons_pot_mesh.Cons_Pot_Mesh.make_from_geo_mesh(geo_mesh),
        (Pot_Type.LINEAR, Mesh_Type.LINEAR): lin_pot_mesh.Lin_Pot_Mesh.make_from_lin_geo_mesh(geo_mesh),
        (Pot_Type.CONSTANT, Mesh_Type.QUADRATIC): cons_pot_mesh.Cons_Pot_Mesh.make_from_geo_mesh(geo_mesh),
        (Pot_Type.LINEAR, Mesh_Type.QUADRATIC): lin_pot_mesh.Lin_Pot_Mesh.make_from_quad_geo_mesh(geo_mesh),
    }
    pot_mesh = pot_mesh_map[(pot_type, mesh_type)]
    num_nodes = pot_mesh.get_nodes().shape[0]

    stiff_map = {
        (Pot_Type.CONSTANT, Mesh_Type.LINEAR): m_a.make_mat_cp_le,
        (Pot_Type.LINEAR, Mesh_Type.LINEAR): m_a.make_mat_lp_le,
        (Pot_Type.CONSTANT, Mesh_Type.QUADRATIC): m_a.make_mat_cp_qe,
        (Pot_Type.LINEAR, Mesh_Type.QUADRATIC): m_a.make_mat_lp_qe,
    }

    K = stiff_map[(pot_type, mesh_type)](pot_mesh, geo_mesh) # stiffness matrix
    eig_vals, _eig_vecs = np.linalg.eig(K)
    np.savetxt("{}_eig.txt".format(args.out_tag), np.sort(np.real(eig_vals)))


    # Linear eigenfunctions
    E_d, E_c = RDL_eig_funs.E_12(dims)
    eigval_12 = RDL_eig_vals.lambda_12(dims)
    ret_E12 = lin_flow_solves(
        pot_mesh,
        geo_mesh,
        {
            "K": K,
            "eigval": eigval_12,
            "E_d": E_d,
            "E_c": E_c,
            "f": f,
            "l": l,
            "mu": mu,
        },
    )
    E_d, E_c = RDL_eig_funs.E_23(dims)
    eigval_23 = RDL_eig_vals.lambda_23(dims)
    ret_E23 = lin_flow_solves(
        pot_mesh,
        geo_mesh,
        {
            "K": K,
            "eigval": eigval_23,
            "E_d": E_d,
            "E_c": E_c,
            "f": f,
            "l": l,
            "mu": mu,
        },
    )
    E_d, E_c = RDL_eig_funs.E_31(dims)
    eigval_31 = RDL_eig_vals.lambda_31(dims)
    ret_E31 = lin_flow_solves(
        pot_mesh,
        geo_mesh,
        {
            "K": K,
            "eigval": eigval_31,
            "E_d": E_d,
            "E_c": E_c,
            "f": f,
            "l": l,
            "mu": mu,
        },
    )
    E_d, E_c = RDL_eig_funs.diag_eigvec("+", dims)
    eigval_p = RDL_eig_vals.lambda_pm("+", dims)
    ret_Ep = lin_flow_solves(
        pot_mesh,
        geo_mesh,
        {
            "K": K,
            "eigval": eigval_p,
            "E_d": E_d,
            "E_c": E_c,
            "f": f,
            "l": l,
            "mu": mu,
        },
    )
    E_d, E_c = RDL_eig_funs.diag_eigvec("-", dims)
    eigval_m = RDL_eig_vals.lambda_pm("-", dims)
    ret_Em = lin_flow_solves(
        pot_mesh,
        geo_mesh,
        {
            "K": K,
            "eigval": eigval_m,
            "E_d": E_d,
            "E_c": E_c,
            "f": f,
            "l": l,
            "mu": mu,
        },
    )

    # Quadratic flows
    kappa_vec = RDL_eig_vals.calc_3x3_eval(dims)
    eigval_3x3 = -(1 + kappa_vec) / (kappa_vec -1)
    ret_3x3 = quad_flow_solves(
        pot_mesh,
        geo_mesh,
        {
            "K": K,
            "eigval_3x3": eigval_3x3,
            "dims": dims,
            "kappa_vec": kappa_vec,
            "f": f,
            "l": l,
            "mu": mu,
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
            },
        )
    elif pot_type == Pot_Type.LINEAR:
        cells = [("triangle", pot_mesh.faces)] # to plot point data correctly must be this, might
        # also be able to map point_data to specific points, not sure how to do this with meshio
        mesh_io = meshio.Mesh(
            pot_mesh.get_nodes(),
            cells,
            point_data={
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
            },
        )
    meshio.write("{}_out.vtk".format(args.out_tag), mesh_io, file_format="vtk")


    out_file = "{}_data.txt".format(args.out_tag)
    with open(out_file, 'w') as out:
        csv_writer = csv.writer(out, delimiter=',')
        out.write("trans_vel, rot_vel, collinearity, tot_abs_err, tot_rel_err, eigenvalue\n")
        csv_writer.writerow([ret_E12["trans_velocity"], ret_E12["rot_velocity"],
            ret_E12["collinearity"], ret_E12["total_absolute_L2_error"],
            ret_E12["total_relative_L2_error"], eigval_12])
        csv_writer.writerow([ret_E23["trans_velocity"], ret_E23["rot_velocity"],
            ret_E23["collinearity"], ret_E23["total_absolute_L2_error"],
            ret_E23["total_relative_L2_error"], eigval_23])
        csv_writer.writerow([ret_E31["trans_velocity"], ret_E31["rot_velocity"],
            ret_E31["collinearity"], ret_E31["total_absolute_L2_error"],
            ret_E31["total_relative_L2_error"], eigval_31])
        csv_writer.writerow([ret_Ep["trans_velocity"], ret_Ep["rot_velocity"],
            ret_Ep["collinearity"], ret_Ep["total_absolute_L2_error"],
            ret_Ep["total_relative_L2_error"], eigval_p])
        csv_writer.writerow([ret_Em["trans_velocity"], ret_Em["rot_velocity"],
            ret_Em["collinearity"], ret_Em["total_absolute_L2_error"],
            ret_Em["total_relative_L2_error"], eigval_m])

        csv_writer.writerow([ret_3x3["trans_velocity"][0], ret_3x3["rot_velocity"][0],
            ret_3x3["collinearity"][0], ret_3x3["total_absolute_L2_error"][0],
            ret_3x3["total_relative_L2_error"][0], eigval_3x3[0]])
        csv_writer.writerow([ret_3x3["trans_velocity"][1], ret_3x3["rot_velocity"][1],
            ret_3x3["collinearity"][1], ret_3x3["total_absolute_L2_error"][1],
            ret_3x3["total_relative_L2_error"][1], eigval_3x3[1]])
        csv_writer.writerow([ret_3x3["trans_velocity"][2], ret_3x3["rot_velocity"][2],
            ret_3x3["collinearity"][2], ret_3x3["total_absolute_L2_error"][2],
            ret_3x3["total_relative_L2_error"][2], eigval_3x3[2]])



def lin_flow_solves(pot_mesh, geo_mesh, K_ev):
    """
    Linear eigenvalue/eigenvector error function
    Parameters:
        pot_mesh: potential mesh
        geo_mesh: geometric mesh
        K_ev: dict of {K, eigval, E_d, E_c, f, l, mu}
    Returns:
        dict of keys:
            local_relative_L2_error: local L2 norm error (N,) ndarray
            total_relative_L2_error: L2 norm of error between q and psi
            collinearity: collinearity of q and psi (1 = exactly collinear)
            trans_velocity: translational velocity (3,) ndarray
            rot_velocity: rotational velocity (3,) ndarray
    """
    tol = 1e-6 #lower bound for norm about equal to zero
    num_nodes = pot_mesh.get_nodes().shape[0]
    K = K_ev["K"]
    eigval = K_ev["eigval"]
    E_d = K_ev["E_d"]
    E_c = K_ev["E_c"]
    f = K_ev["f"]
    l = K_ev["l"]
    mu = K_ev["mu"]
    u_d = RDL_eig_helper.make_lin_eig_vels(pot_mesh, E_d, E_c) # (N, 3)
    psi = m_a.make_forcing_vec(pot_mesh, geo_mesh, np.ravel(u_d), f, l, mu) # eigenvector (3N,)
    q = np.linalg.solve(K + np.identity(3*num_nodes), (eigval + 1) * psi) # (3N,)
    diff = q - psi
    base = np.linalg.norm(np.reshape(psi, (num_nodes, 3)), axis=1)
    loc_abs_err = np.linalg.norm(np.reshape(diff, (num_nodes, 3)), axis=1)
    #loc_rel_err = np.divide(loc_abs_err, base, out=np.zeros_like(loc_abs_err), where=(base > tol))
    loc_rel_err = loc_abs_err / np.mean(base)
    tot_abs_err = np.linalg.norm(diff)
    tot_rel_err = np.linalg.norm(diff) / np.linalg.norm(psi)
    collin = np.dot(q / np.linalg.norm(q), psi / np.linalg.norm(psi))
    if isinstance(pot_mesh, cons_pot_mesh.Cons_Pot_Mesh):
        if isinstance(geo_mesh, lin_geo_mesh.Lin_Geo_Mesh):
            trans_v = (RDL_mobil_helper.calc_cp_le_trans_vel(geo_mesh, q))
            rot_v = (RDL_mobil_helper.calc_cp_le_rot_vel(geo_mesh, q))
        else:
            trans_v = (RDL_mobil_helper.calc_cp_qe_trans_vel(geo_mesh, q))
            rot_v = (RDL_mobil_helper.calc_cp_qe_rot_vel(geo_mesh, q))
    else:
        if isinstance(geo_mesh, lin_geo_mesh.Lin_Geo_Mesh):
            trans_v = (RDL_mobil_helper.calc_lp_le_trans_vel(pot_mesh, geo_mesh, q))
            rot_v = (RDL_mobil_helper.calc_lp_le_rot_vel(pot_mesh, geo_mesh, q))
        else:
            trans_v = (RDL_mobil_helper.calc_lp_qe_trans_vel(pot_mesh, geo_mesh, q))
            rot_v = (RDL_mobil_helper.calc_lp_qe_rot_vel(pot_mesh, geo_mesh, q))

    ret = {
        "local_relative_L2_error": loc_rel_err,
        "local_absolute_L2_error": loc_abs_err,
        "total_relative_L2_error": tot_rel_err,
        "total_absolute_L2_error": tot_abs_err,
        "collinearity": collin,
        "trans_velocity": trans_v,
        "rot_velocity": rot_v,
    }
    return ret


def quad_flow_solves(pot_mesh, geo_mesh, K_ev):
    """
    Quadratic eigenvalue/eigenvector error function
    Parameters:
        pot_mesh: potential mesh
        geo_mesh: geometric mesh
        K_ev: dict of {K, eigval_3x3, dims, kappa_vec, f, l, mu}
    Returns:
        dict of keys:
            local_relative_L2_error: local L2 norm error (N,) ndarray
            total_relative_L2_error: L2 norm of error between q and psi
            collinearity: collinearity of q and psi (1 = exactly collinear)
            trans_velocity: translational velocity (3,) ndarray
            rot_velocity: rotational velocity (3,) ndarray
    """
    tol = 1e-6 #lower bound for norm about equal to zero
    K = K_ev["K"]
    eigval_3x3 = K_ev["eigval_3x3"]
    dims = K_ev["dims"]
    kappa_vec = K_ev["kappa_vec"]
    f = K_ev["f"]
    l = K_ev["l"]
    mu = K_ev["mu"]
    num_nodes = pot_mesh.get_nodes().shape[0]
    u_d_3x3 = RDL_eig_helper.make_quad_eig_vels(pot_mesh, dims, kappa_vec) # (N, 3)
    loc_rel_err = []
    tot_rel_err = []
    tot_abs_err = []
    collin = []
    trans_v = []
    rot_v = []
    for i, u_d in enumerate(u_d_3x3):
        psi = m_a.make_forcing_vec(pot_mesh, geo_mesh, np.ravel(u_d), f, l, mu) # eigenvector (3N,)
        q = np.linalg.solve(K + np.identity(3*num_nodes), (eigval_3x3[i] + 1) * psi) # (3N,)
        diff = q - psi
        base = np.linalg.norm(np.reshape(psi, (num_nodes, 3)), axis=1)
        loc_abs_err = np.linalg.norm(np.reshape(diff, (num_nodes, 3)), axis=1)
        #loc_rel_err.append(np.divide(tmp_0, base, out=np.zeros_like(tmp_0), where=(base > tol)))
        loc_rel_err = loc_abs_err / np.mean(base)
        tot_abs_err.append(np.linalg.norm(diff))
        tot_rel_err.append(np.linalg.norm(diff) / np.linalg.norm(psi))
        collin.append(np.dot(q / np.linalg.norm(q), psi / np.linalg.norm(psi)))
        if isinstance(pot_mesh, cons_pot_mesh.Cons_Pot_Mesh):
            if isinstance(geo_mesh, lin_geo_mesh.Lin_Geo_Mesh):
                trans_v.append(RDL_mobil_helper.calc_cp_le_trans_vel(geo_mesh, q))
                rot_v.append(RDL_mobil_helper.calc_cp_le_rot_vel(geo_mesh, q))
            else:
                trans_v.append(RDL_mobil_helper.calc_cp_qe_trans_vel(geo_mesh, q))
                rot_v.append(RDL_mobil_helper.calc_cp_qe_rot_vel(geo_mesh, q))
        else:
            if isinstance(geo_mesh, lin_geo_mesh.Lin_Geo_Mesh):
                trans_v.append(RDL_mobil_helper.calc_lp_le_trans_vel(pot_mesh, geo_mesh, q))
                rot_v.append(RDL_mobil_helper.calc_lp_le_rot_vel(pot_mesh, geo_mesh, q))
            else:
                trans_v.append(RDL_mobil_helper.calc_lp_qe_trans_vel(pot_mesh, geo_mesh, q))
                rot_v.append(RDL_mobil_helper.calc_lp_qe_rot_vel(pot_mesh, geo_mesh, q))

    ret = {
        "local_relative_L2_error": loc_rel_err,
        "local_absolute_L2_error": loc_abs_err,
        "total_relative_L2_error": tot_rel_err,
        "total_absolute_L2_error": tot_abs_err,
        "collinearity": collin,
        "trans_velocity": trans_v,
        "rot_velocity": rot_v,
    }
    return ret


class Mesh_Type(Enum):
    LINEAR = 1
    QUADRATIC = 2


class Pot_Type(Enum):
    CONSTANT = 0
    LINEAR = 1


if __name__ == "__main__":
    main()
