"""
Mesh eigenvector tests
"""
import argparse as argp
import csv
import numpy as np
import meshio
from rigid_DL import cons_pot_mesh
from rigid_DL import lin_pot_mesh
from rigid_DL import lin_geo_mesh_NV
from rigid_DL import quad_geo_mesh_NV
import rigid_DL.mat_assembly as mata
import rigid_DL.eigenfunctions as eigfuns
import rigid_DL.eigenvalues as eigvals
import rigid_DL.gauss_quad as gq
import rigid_DL.eigfun_helper as eig_helper
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

    out_name = args.out_tag
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
        geo_mesh = lin_geo_mesh_NV.Lin_Geo_Mesh_NV(args.dims, verts, faces)
    elif mesh_type == Mesh_Type.QUADRATIC:
        geo_mesh = quad_geo_mesh_NV.Quad_Geo_Mesh_NV(args.dims, verts, faces)

    pot_mesh_map = {
        (Pot_Type.CONSTANT, Mesh_Type.LINEAR): cons_pot_mesh.Cons_Pot_Mesh.make_from_geo_mesh(geo_mesh),
        (Pot_Type.CONSTANT, Mesh_Type.QUADRATIC): cons_pot_mesh.Cons_Pot_Mesh.make_from_geo_mesh(geo_mesh),
        (Pot_Type.LINEAR, Mesh_Type.LINEAR): lin_pot_mesh.Lin_Pot_Mesh.make_from_lin_geo_mesh(geo_mesh),
        (Pot_Type.LINEAR, Mesh_Type.QUADRATIC): lin_pot_mesh.Lin_Pot_Mesh.make_from_quad_geo_mesh(geo_mesh),
    }
    pot_mesh = pot_mesh_map[(pot_type, mesh_type)]

    stiff_map = {
        (Pot_Type.CONSTANT, Mesh_Type.LINEAR): mata.make_mat_cp_le_NV,
        (Pot_Type.LINEAR, Mesh_Type.LINEAR): mata.make_mat_lp_le_NV,
        (Pot_Type.CONSTANT, Mesh_Type.QUADRATIC): mata.make_mat_cp_qe,
        (Pot_Type.LINEAR, Mesh_Type.QUADRATIC): mata.make_mat_lp_qe,
    }
    K = stiff_map[(pot_type, mesh_type)](pot_mesh, geo_mesh) # stiffness matrix

    eig_vals, _eig_vecs = np.linalg.eig(K)

    # Linear eigenfunctions
    num_nodes = pot_mesh.get_nodes().shape[0]
    pot_nodes = pot_mesh.get_nodes()

    print("E12 flow")
    E_d, E_c = eigfuns.E_12(args.dims)
    eigval = eigvals.lambda_12(args.dims)
    psi = np.zeros((num_nodes, 3))
    for m in range(num_nodes):
        node = pot_nodes[m]
        xx = node - geo_mesh.get_centroid()
        psi[m] = E_d @ xx - np.cross(E_c, xx)

    #print("Rotational flow")
    #E = np.array([
    #    [0., 1., 0.,],
    #    [-1., 0., 0.,],
    #    [0., 0., 0.,],
    #])
    #eigval = -1.
    #psi = eig_helper.make_lin_eig_vels(pot_mesh, E, np.zeros(3))
    
    #print("Translational flow")
    #psi = np.zeros((num_nodes, 3))
    #eigval = -1.
    #psi[:,0] = 1.

    out_vec = np.reshape(K @ np.ravel(psi), psi.shape, order="C")
    norm_err = np.linalg.norm(out_vec - (eigval * psi), axis=1)
    mean_err = np.mean(norm_err)
    print("Avg. absolute L2 error:")
    print(mean_err)
    print("Avg. relative L2 error:")
    print(mean_err / eig_helper.calc_ext_flow_magnitude((pot_type, mesh_type), pot_mesh, geo_mesh, psi))

    #print("Colinearlity:")
    #print(np.dot(tmp_0 / np.linalg.norm(tmp_0), tmp_1 / np.linalg.norm(tmp_1)))
    #print("Whole Relative L2 Error:")
    #print(np.linalg.norm(tmp_1 - tmp_0) / np.linalg.norm(tmp_0))
    #np.savetxt("base_vec_12.csv", eig_vec * eigval_12, delimiter=",")
    #np.savetxt("out_vec_12.csv", out_vec, delimiter=",")

if __name__ == "__main__":
    main()
