"""
Testing the DL matrix assembly
"""
import argparse as argp
import numpy as np
import meshio
import rigid_DL.mat_assembly as mata
from rigid_DL import lin_geo_mesh, quad_geo_mesh, cons_pot_mesh, lin_pot_mesh
import rigid_DL.eigfun_helper as eig_helper

parser = argp.ArgumentParser(description="Testing the DL matrix assembly")
parser.add_argument("mesh", help="mesh file input, parameterization determined by file")
parser.add_argument(
    "-o",
    "--out_tag",
    help="tag to prepend to output files",
    default="DL_test"
)
args = parser.parse_args()
io_mesh = meshio.read(args.mesh)
verts = io_mesh.points
for cell_block in io_mesh.cells:
    if cell_block.type == "triangle":
        faces = cell_block.data
        break

geo_mesh = lin_geo_mesh.Lin_Geo_Mesh(verts, faces)
pot_mesh = lin_pot_mesh.Lin_Pot_Mesh.make_from_lin_geo_mesh(geo_mesh)
K = mata.make_mat_lp_le(pot_mesh, geo_mesh)
eig_vals, eig_vecs = np.linalg.eig(K)

out_file = "{}_eig.txt".format(args.out_tag)
with open(out_file, 'w') as out:
    out.write("eigenvalues\n")
    for a in eig_vals.real:
        out.write("{}\n".format(a))


E_d = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 0]])
E_c = np.zeros(3)
u_d = eig_helper.make_lin_eig_vels(pot_mesh, E_d, E_c)
u_d = np.ravel(u_d)

out_vec = K @ u_d

sol_file= "{}_sol.txt".format(args.out_tag)
with open(sol_file, 'w') as out:
    out.write("sol, u_d\n")
    for i, a in enumerate(out_vec):
        out.write("{}, {}\n".format(a, u_d[i]))
