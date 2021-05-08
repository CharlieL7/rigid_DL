"""
Mesh eigenvector tests
"""
import sys
import time
import argparse as argp
import numpy as np
import meshio
from rigid_DL import lin_geo_mesh, quad_geo_mesh, cons_pot_mesh, lin_pot_mesh
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
            print("Unsupported mesh type")
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
    t1 = time.process_time()
    K_py = stiff_map[(pot_type, mesh_type)](pot_mesh, geo_mesh) # stiffness matrix
    t2 = time.process_time()
    print("normals py")
    print("K_py time: {}".format(t2 - t1))
    K_c = mata.make_mat_lp_le_cpp(pot_mesh, geo_mesh)
    t1 = time.process_time()
    print("K_c time: {}".format(t1 - t2))
    
    np.savetxt("K_py.csv", K_py, delimiter=",")
    np.savetxt("K_c.csv", K_c, delimiter=",")
    np.savetxt("K_py_eig.csv", sorted(np.linalg.eig(K_py)[0]), delimiter=",")
    np.savetxt("K_c_eig.csv", sorted(np.linalg.eig(K_c)[0]), delimiter=",")
    print("Are matrices allclose?: {}".format(np.allclose(K_py, K_c)))


if __name__ == "__main__":
    main()
