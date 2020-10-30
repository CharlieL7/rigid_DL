"""
Testing the stiffness matrix assembly for the mobility problem
"""
import argparse as argp
import numpy as np
import meshio
import rigid_DL.mobil_assembly as m_a
from rigid_DL import lin_geo_mesh, quad_geo_mesh, cons_pot_mesh, lin_pot_mesh

parser = argp.ArgumentParser(description="Testing the mobility problem assembly")
parser.add_argument("mesh", help="mesh file input, parameterization determined by file")
parser.add_argument(
    "-o",
    "--out_tag",
    help="tag to prepend to output files",
    default="mobil_test"
)
parser.add_argument(
    "-m",
    "--mu",
    help="viscosity",
    default=1.
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
K = m_a.make_mat_cp_le(pot_mesh, geo_mesh)
eig_vals, eig_vecs = np.linalg.eig(K)

out_file = "{}_eig.txt".format(args.out_tag)
with open(out_file, 'w') as out:
    out.write("eigenvalues\n")
    for a in eig_vals.real:
        out.write("{}\n".format(a))

mu = args.mu
num_nodes = pot_mesh.get_nodes().shape[0]
u_d = np.zeros(3 * num_nodes)
u_d[0:None:3] = 1.
f = np.zeros(3)
l = np.zeros(3)
q = m_a.make_init_sol_vector(pot_mesh)
fv = m_a.make_cp_le_forcing_vec(pot_mesh, geo_mesh, u_d, f, l, mu)
cont = True
iter_num = 0
while cont:
    q_old = q
    q = K @ q + fv
    iter_num += 1
    if np.linalg.norm(q - q_old) < 1e-6:
        cont = False
    elif iter_num > 100:
        cont = False
        print("Iterations > 100, failed")

print("Iterations: {}".format(iter_num))
q_file= "{}_sol.txt".format(args.out_tag)
with open(q_file, 'w') as out:
    out.write("solution vector\n")
    for a in q:
        out.write("{}\n".format(a))
