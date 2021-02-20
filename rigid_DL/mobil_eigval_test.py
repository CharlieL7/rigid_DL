"""
Testing the stiffness matrix assembly for the mobility problem
"""
import sys
import csv
import argparse as argp
import numpy as np
import meshio
import rigid_DL.mobil_assembly as m_a
from rigid_DL import lin_geo_mesh, quad_geo_mesh, cons_pot_mesh, lin_pot_mesh
import rigid_DL.eigfun_helper as RDL_eig_helper
import rigid_DL.mobil_helper as RDL_mobil_helper
import rigid_DL.eigenfunctions as RDL_eig_funs
import rigid_DL.eigenvalues as RDL_eig_vals
from rigid_DL.enums import Mesh_Type, Pot_Type

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
    #eig_vals, _eig_vecs = np.linalg.eig(K)
    #np.savetxt("{}_eig.txt".format(args.out_tag), np.sort(np.real(eig_vals)))
    
    #print("Rotational flow")
    #E_d = np.array(
    #    [
    #        [0., 1., 0.,],
    #        [-1., 0., 0.,],
    #        [0., 0., 0.,]
    #    ]
    #)
    #E_c = np.zeros(3)

    print("E12 flow")
    E_d, E_c = RDL_eig_funs.E_12(dims)
    eigval = RDL_eig_vals.lambda_12(dims)

    pot_nodes = pot_mesh.get_nodes()
    num_nodes = pot_nodes.shape[0]
    psi = np.zeros((num_nodes, 3))
    for m in range(num_nodes):
        node = pot_nodes[m]
        xx = node - geo_mesh.get_centroid()
        psi[m] = E_d @ xx - np.cross(E_c, xx)

    q = K @ np.ravel(psi)

    out_vec = np.reshape(q, (num_nodes, 3))
    norm_err = np.linalg.norm(out_vec - (eigval) * psi, axis=1)
    mean_err = np.mean(norm_err)
    print("Avg. absolute L2 error:")
    print(mean_err)
    print("Avg. relative L2 error:")
    print(mean_err / RDL_eig_helper.calc_ext_flow_magnitude((pot_type, mesh_type), pot_mesh, geo_mesh, psi))


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
    loc_err = np.linalg.norm(np.reshape(diff, (num_nodes, 3)), axis=1)
    #loc_err = np.divide(loc_err, base, out=np.zeros_like(loc_err), where=(base > tol))
    tot_err = np.linalg.norm(diff) / np.linalg.norm(psi)
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
        "local_relative_L2_error": loc_err,
        "total_relative_L2_error": tot_err,
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
    loc_err = []
    tot_err = []
    collin = []
    trans_v = []
    rot_v = []
    for i, u_d in enumerate(u_d_3x3):
        psi = m_a.make_forcing_vec(pot_mesh, geo_mesh, np.ravel(u_d), f, l, mu) # eigenvector (3N,)
        q = np.linalg.solve(K + np.identity(3*num_nodes), (eigval_3x3[i] + 1) * psi) # (3N,)
        diff = q - psi
        base = np.linalg.norm(np.reshape(psi, (num_nodes, 3)), axis=1)
        tmp_0 = np.linalg.norm(np.reshape(diff, (num_nodes, 3)), axis=1)
        #loc_err.append(np.divide(tmp_0, base, out=np.zeros_like(tmp_0), where=(base > tol)))
        loc_err.append(tmp_0)
        tot_err.append(np.linalg.norm(diff) / np.linalg.norm(psi))
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
        "local_relative_L2_error": loc_err,
        "total_relative_L2_error": tot_err,
        "collinearity": collin,
        "trans_velocity": trans_v,
        "rot_velocity": rot_v,
    }
    return ret

if __name__ == "__main__":
    main()
