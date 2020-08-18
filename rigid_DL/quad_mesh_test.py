"""
Quadratic mesh test
"""

import argparse as argp
import numpy as np
import meshio
import simple_quad_mesh as sqm
import eigenfunctions as efun
import input_output as io
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
    v_cnst = np.array([1, 0, 0])
    v_trans_in = efun.make_translation_vels(v_cnst, quad_mesh, 'c')
    v_trans_out = np.dot(C, v_trans_in.flatten('C'))
    v_trans_out = v_trans_out.reshape(v_trans_in.shape, order='C')
    io.write_vel(v_trans_in, v_trans_out, "quad_test_trans_vel.csv")

if __name__ == "__main__":
    main()
