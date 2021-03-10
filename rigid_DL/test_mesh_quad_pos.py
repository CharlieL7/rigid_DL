"""
Testing mesh quadrature positions against ellipsoid surface error for linear and quadratic meshes 
"""
import argparse as argp
import meshio
from rigid_DL.enums import Mesh_Type
from rigid_DL import lin_geo_mesh, quad_geo_mesh


def main():
    parser = argp.ArgumentParser(description="Testing mesh quadrature positions to see error from ellipsoidal surface")
    parser.add_argument("mesh", help="mesh file input, parameterization determined by file")
    parser.add_argument("dims", nargs=3, type=float, help="expected mesh dimensions")
    parser.add_argument(
        "-o",
        "--out_tag",
        help="tag to prepend to output files",
        default="test_mesh_quad_pos"
    )
    
    args = parser.parse_args()
    io_mesh = meshio.read(args.mesh)
    dims = args.dims

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

    for face in geo_mesh.faces:


if __name__ == "__main__":
    main()
