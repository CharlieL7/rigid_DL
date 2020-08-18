"""
Quadratic mesh test
"""

import math
import time
import argparse as argp
import numpy as np
import meshio
import simple_quad_mesh as sqm
import gauss_quad as gq
import geometric as geo
import eigenfunctions as efun
import potential_calc as poc

def main():
    parser = argp.ArgumentParser(description="Quadratic mesh test")
    parser.add_argument("mesh", help="input quadratic triangular mesh")
    args = parser.parse_args()
    mesh = meshio.read(args.mesh)
    verts = mesh.points
    for cell_block in mesh.cells:
        if cell_block.type == "triangle6":
            faces = cell_block.data

    quad_mesh = sqm.simple_quad_mesh(verts, faces)
    C = poc.make_mat_cp_qe(quad_mesh)
    print(C)

if __name__ == "__main__":
    main()
