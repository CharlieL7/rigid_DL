"""
Mesh eigenvector tests
"""
import argparse as argp
from enum import Enum
import numpy as np
import meshio
import csv
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
    K = stiff_map[(pot_type, mesh_type)](pot_mesh, geo_mesh) # stiffness matrix

    eig_vals, _eig_vecs = np.linalg.eig(K)
    np.savetxt("{}_eig.txt".format(args.out_tag), np.sort(np.real(eig_vals)))

    
    """
    # Linear eigenfunctions
    E_d, E_c = eigfuns.E_12(geo_mesh, expected_dims)
    eigval_12 = eigvals.lambda_12(expected_dims)
    eig_vec = eig_helper.make_lin_eig_vels(pot_mesh, E_d, E_c)
    out_vec = np.reshape(K @ eig_vec.flatten("C"), eig_vec.shape, order="C")
    tmp_0 = np.ravel(eigval_12 * eig_vec)
    tmp_1 = np.ravel(out_vec)
    print("Colinearlity:")
    print(np.dot(tmp_0 / np.linalg.norm(tmp_0), tmp_1 / np.linalg.norm(tmp_1)))
    print("Whole Relative L2 Error:")
    print(np.linalg.norm(tmp_1 - tmp_0) / np.linalg.norm(tmp_0))
    np.savetxt("base_vec_12.csv", eig_vec * eigval_12, delimiter=",")
    np.savetxt("out_vec_12.csv", out_vec, delimiter=",")
    """


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


class Mesh_Type(Enum):
    LINEAR = 1
    QUADRATIC = 2


class Pot_Type(Enum):
    CONSTANT = 0
    LINEAR = 1

if __name__ == "__main__":
    main()
