"""
Testing the stiffness matrix assembly for the mobility problem
"""
import argparse as argp
import numpy as np
import meshio
import rigid_DL.mobil_assembly as m_a
from rigid_DL import lin_geo_mesh, quad_geo_mesh, cons_pot_mesh, lin_pot_mesh
import rigid_DL.eigfun_helper as eig_helper
import rigid_DL.mobil_helper as mobil_helper
import rigid_DL.eigenfunctions as eig_funs

parser = argp.ArgumentParser(description="Testing the mobility problem assembly")
parser.add_argument("mesh", help="mesh file input, parameterization determined by file")
parser.add_argument("dims", nargs=3, type=float, help="expected mesh dimensions")
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
parser.add_argument(
    "-s",
    "--is_sphere",
    help="mesh is a sphere",
    default=False,
    action="store_true",
)
args = parser.parse_args()
io_mesh = meshio.read(args.mesh)
verts = io_mesh.points
for cell_block in io_mesh.cells:
    if cell_block.type == "triangle":
        faces = cell_block.data
        break

geo_mesh = lin_geo_mesh.Lin_Geo_Mesh(verts, faces)
if args.is_sphere:
    geo_mesh.is_sphere = True
mu = args.mu
E_d = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
E_c = np.zeros(3)
f = np.array([0., 0., 0.,]) # force
l = np.array([1., 0., 0,]) # torque

"""
pot_mesh = cons_pot_mesh.Cons_Pot_Mesh.make_from_geo_mesh(geo_mesh)
K = m_a.make_mat_cp_le(pot_mesh, geo_mesh)
eig_vals, eig_vecs = np.linalg.eig(K)

out_file = "{}_cp_le_eig.txt".format(args.out_tag)
with open(out_file, 'w') as out:
    out.write("eigenvalues\n")
    for a in sorted(eig_vals.real):
        out.write("{}\n".format(a))
num_nodes = pot_mesh.get_nodes().shape[0]

u_d = eig_helper.make_lin_eig_vels(pot_mesh, E_d, E_c)
u_d = np.ravel(u_d)

fv = m_a.make_forcing_vec(pot_mesh, geo_mesh, u_d, f, l, mu)
q = np.linalg.solve(K + np.identity(3*num_nodes), fv)

q_file= "{}_cp_le_sol.txt".format(args.out_tag)
with open(q_file, 'w') as out:
    out.write("q, u_d\n")
    for i, a in enumerate(q):
        out.write("{}, {}\n".format(a, u_d[i]))

trans_v = mobil_helper.calc_cp_le_trans_vel(geo_mesh, q)
print("cp_le Particle translational velocity: {}".format(trans_v))

rot_v = mobil_helper.calc_cp_le_rot_vel(geo_mesh, q)
print("cp_le Particle rotational velocity: {}".format(rot_v))

"""

pot_mesh = lin_pot_mesh.Lin_Pot_Mesh.make_from_lin_geo_mesh(geo_mesh)
num_nodes = pot_mesh.get_nodes().shape[0]
K = m_a.make_mat_lp_le(pot_mesh, geo_mesh)
eig_vals, eig_vecs = np.linalg.eig(K)

out_file = "{}_lp_le_eig.txt".format(args.out_tag)
with open(out_file, 'w') as out:
    out.write("eigenvalues\n")
    for a in sorted(eig_vals.real):
        out.write("{}\n".format(a))

u_d = eig_helper.make_lin_eig_vels(pot_mesh, E_d, E_c)
u_d = np.ravel(u_d)
fv = m_a.make_forcing_vec(pot_mesh, geo_mesh, u_d, f, l, mu)
q = np.linalg.solve(K + np.identity(3 * num_nodes), fv)

q_file = "{}_lp_le_sol.txt".format(args.out_tag)
with open(q_file, 'w') as out:
    out.write("q, u_d\n")
    for i, a in enumerate(q):
        out.write("{}, {}\n".format(a, u_d[i]))

trans_v = mobil_helper.calc_lp_le_trans_vel(pot_mesh, geo_mesh, q)
print("lp_le Particle translational velocity: {}".format(trans_v))

rot_v = mobil_helper.calc_lp_le_rot_vel(pot_mesh, geo_mesh, q)
print("lp_le Particle rotational velocity: {}".format(rot_v))
