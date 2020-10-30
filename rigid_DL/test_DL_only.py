"""
Testing the DL matrix assembly
"""
import argparse as argp
import numpy as np
import meshio
import rigid_DL.mat_assembly as mata
from rigid_DL import lin_geo_mesh, quad_geo_mesh, cons_pot_mesh, lin_pot_mesh

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
pot_mesh = cons_pot_mesh.Cons_Pot_Mesh.make_from_geo_mesh(geo_mesh)
K = mata.make_mat_cp_le(pot_mesh, geo_mesh)
eig_vals, eig_vecs = np.linalg.eig(K)

out_file = "{}_eig.txt".format(args.out_tag)
with open(out_file, 'w') as out:
    out.write("eigenvalues\n")
    for a in eig_vals.real:
        out.write("{}\n".format(a))
