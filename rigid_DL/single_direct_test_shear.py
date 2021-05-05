"""
Mesh eigenvector tests
"""
import argparse as argp
import numpy as np
import meshio
import csv
from rigid_DL import lin_geo_mesh, quad_geo_mesh, cons_pot_mesh, lin_pot_mesh
import rigid_DL.mat_assembly as mata
import rigid_DL.eigfun_helper as RDL_eig_helper
import rigid_DL.mobil_helper as RDL_mobil_helper
import rigid_DL.eigenfunctions as RDL_eig_funs
import rigid_DL.eigenvalues as RDL_eig_vals
from rigid_DL.enums import Mesh_Type, Pot_Type


def main():
    parser = argp.ArgumentParser(description="Double layer eigenvector tests that handles all parameterizations.")
    parser.add_argument("mesh", help="mesh file input, parameterization determined by file")
    parser.add_argument("dims", nargs=3, type=float, help="expected mesh dimensions")
    parser.add_argument(
        "-o",
        "--out_tag",
        help="tag to prepend to output files",
        default="direct_test_shear"
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

    # Linear eigenfunctions
    num_nodes = pot_mesh.get_nodes().shape[0]
    pot_nodes = pot_mesh.get_nodes()

    eigval_12 = RDL_eig_vals.lambda_12(args.dims)

    print("Shear flow")
    E_d = np.array([
        [0., 1., 0.,],
        [0., 0., 0.,],
        [0., 0., 0.,],
    ])
    u_d = np.zeros((num_nodes, 3))
    for m in range(num_nodes):
        node = pot_nodes[m]
        xx = node - geo_mesh.get_centroid()
        u_d[m] = E_d @ xx
    u_d /= (RDL_eig_helper.calc_inner_product_2((pot_type, mesh_type), pot_mesh, geo_mesh, u_d))**(1./2.)
    out_vec = np.reshape(K @ np.ravel(u_d), u_d.shape, order="C")
    
    a, b, _c = args.dims
    E_ext_12 = np.array([
        [0., 1., 0.,],
        [1., 0., 0.,],
        [0., 0., 0.,],
    ])
    E_rot_12 = np.array([
        [0., 1., 0.,],
        [-1., 0., 0.,],
        [0., 0., 0.,],
    ])

    tmp = (a**2 - b**2) / (a**2 + b**2)
    E_expected = eigval_12 * (E_ext_12 + tmp * E_rot_12) + (tmp - 1.) * E_rot_12
    E_expected *= 0.5
    expected = np.zeros((num_nodes, 3))
    for m in range(num_nodes):
        node = pot_nodes[m]
        xx = node - geo_mesh.get_centroid()
        expected[m] = E_expected @ xx
    expected /= (RDL_eig_helper.calc_inner_product_2((pot_type, mesh_type), pot_mesh, geo_mesh, expected))**(1./2.)

    print("out_vec")
    print(out_vec)
    print("expected")
    print(expected)
    local_err = np.linalg.norm(out_vec - expected, axis=1) * np.sqrt(geo_mesh.get_surface_area())
    #print(RDL_eig_helper.calc_ext_flow_magnitude((pot_type, mesh_type), pot_mesh, geo_mesh, expected))
    #rel_err = abs_err / RDL_eig_helper.calc_ext_flow_magnitude((pot_type, mesh_type), pot_mesh, geo_mesh, expected)
    #print("rel_err")
    #print(rel_err)

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
                "local_err_shear": [local_err],
            },
        )
    elif pot_type == Pot_Type.LINEAR:
        cells = [("triangle", pot_mesh.faces)] # to plot point data correctly must be this, might
        # also be able to map point_data to specific points, not sure how to do this with meshio
        mesh_io = meshio.Mesh(
            pot_mesh.get_nodes(),
            cells,
            point_data={
                "local_err_shear": local_err,
            },
        )
    meshio.write("{}_out.vtk".format(args.out_tag), mesh_io, file_format="vtk")


if __name__ == "__main__":
    main()
