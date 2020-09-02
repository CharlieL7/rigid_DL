"""
Creates an ellipsoidal mesh from a spherical mesh input
"""
import sys
import ast
import rigid_DL.simple_linear_mesh as slm

def main():
    if len(sys.argv) != 2:
        print("Usage: make_ellipsoid mesh_name")
        sys.exit()
    mesh_name = sys.argv[1]
    dims = ast.literal_eval(input("multiply dimensions as (a, b, c): "))
    if len(dims) != 3:
        sys.exit("invalid ellipsoid dimensions")
    mesh = slm.simple_linear_mesh.read_dat(mesh_name)
    for i in range(3):
        mesh.vertices[:, i] *= dims[i]
    out_name = input("output name: ")
    mesh.write_to_dat(out_name)

if __name__ == "__main__":
        main()
