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
import rigid_DL.eigenvalues as eig_vals

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
args = parser.parse_args()
io_mesh = meshio.read(args.mesh)
verts = io_mesh.points
for cell_block in io_mesh.cells:
    if cell_block.type == "triangle":
        faces = cell_block.data
        break

dims = args.dims
geo_mesh = lin_geo_mesh.Lin_Geo_Mesh(verts, faces)
mu = args.mu
print("Dims:")
print(dims)
"""
E_d = np.array([
    [0., 1., 0.,],
    [-1., 0., 0.,],
    [0., 0., 0.,]
])
E_c = np.zeros(3)
"""
E_d, E_c = eig_funs.E_12(dims)
#E_d, E_c = eig_funs.diag_eigvec("+", dims)
eigval = eig_vals.lambda_12(dims)
#eigval = eig_vals.lambda_pm("+", dims)
f = np.array([0., 0., 0.,]) # force
l = np.array([0., 0., 0,]) # torque
print("eigenvalue: {}".format(eigval))

# cp_le version
pot_mesh = cons_pot_mesh.Cons_Pot_Mesh.make_from_geo_mesh(geo_mesh)
num_nodes = pot_mesh.get_nodes().shape[0]
K = m_a.make_mat_cp_le(pot_mesh, geo_mesh)
eig_vals, eig_vecs = np.linalg.eig(K)

out_file = "{}_cp_le_eig.txt".format(args.out_tag)
with open(out_file, 'w') as out:
    out.write("eigenvalues\n")
    for a in sorted(eig_vals.real):
        out.write("{}\n".format(a))

u_d = eig_helper.make_lin_eig_vels(pot_mesh, E_d, E_c)

psi = m_a.make_forcing_vec(pot_mesh, geo_mesh, np.ravel(u_d), f, l, mu) #eigenvector
q = np.linalg.solve(K + np.identity(3*num_nodes), (eigval + 1) * psi)
np.savetxt("q_vec_cp_le_R12.csv", np.reshape(q, (num_nodes, 3)), delimiter=",")
np.savetxt("eig_vec_cp_le_E12.csv", np.reshape(psi, (num_nodes, 3)), delimiter=",")

trans_v = mobil_helper.calc_cp_le_trans_vel(geo_mesh, q)
print("cp_le Particle translational velocity: {}".format(repr(trans_v)))

rot_v = mobil_helper.calc_cp_le_rot_vel(geo_mesh, q)
print("cp_le Particle rotational velocity: {}".format(repr(rot_v)))

tmp_0 = psi
tmp_1 = q
print("cp_le Colinearity:")
print(np.dot(tmp_0 / np.linalg.norm(tmp_0), tmp_1 / np.linalg.norm(tmp_1)))
print("cp_le Total Relative L2 Error:")
print(np.linalg.norm(tmp_1 - tmp_0) / np.linalg.norm(tmp_0))
print("Max Abs Error:")
print(np.max(np.linalg.norm(np.reshape(tmp_1 - tmp_0, (num_nodes, 3)), axis=1)))
#print("Abs Errors:")
#print(np.linalg.norm(np.reshape(tmp_1 - tmp_0, (num_nodes, 3)), axis=1))
#print("cp_le L2 Norm psi")
#print(np.linalg.norm(tmp_0))

"""
# lp_le version
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

psi = m_a.make_forcing_vec(pot_mesh, geo_mesh, np.ravel(u_d), f, l, mu)
q = np.linalg.solve(K + np.identity(3 * num_nodes), (eigval + 1) * psi)
q_reshape = np.reshape(q, u_d.shape, order="C")
np.savetxt("q_vec_lp_le_E12.csv", q_reshape, delimiter=",")
np.savetxt("eig_vec_lp_le_E12.csv", np.reshape(psi, (num_nodes, 3)), delimiter=",")

trans_v = mobil_helper.calc_lp_le_trans_vel(pot_mesh, geo_mesh, q)
print("lp_le Particle translational velocity: {}".format(repr(trans_v)))

rot_v = mobil_helper.calc_lp_le_rot_vel(pot_mesh, geo_mesh, q)
print("lp_le Particle rotational velocity: {}".format(repr(rot_v)))

tmp_0 = psi
tmp_1 = q
print("lp_le Colinearity:")
print(np.dot(tmp_0 / np.linalg.norm(tmp_0), tmp_1 / np.linalg.norm(tmp_1)))
print("lp_le Total Relative L2 Error:")
print(np.linalg.norm(tmp_1 - tmp_0) / np.linalg.norm(tmp_0))
"""
