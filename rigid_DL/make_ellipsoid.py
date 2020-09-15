"""
Creates an ellipsoidal mesh from a spherical mesh input
"""
import argparse as argp
import numpy as np
import meshio

def main():
    parser = argp.ArgumentParser(
        description="Makes a subdivided icosahedron and does scaling transforms to an ellipsoid"
    )
    parser.add_argument("-d", "--dims", nargs=3, type=float, default=[1., 1., 1.], help="ellipsoid dimensions")
    parser.add_argument("-o", "--out", help="output mesh name", default="ellip_mesh")
    parser.add_argument("-s", "--subdiv", type=int, help="number of subdivisions", default=0)
    args = parser.parse_args()
    dims = args.dims
    out_name = args.out
    num_subdiv = args.subdiv
    verts, faces = make_icosahedron()
    i = 0
    while i < num_subdiv:
        verts, faces = subdiv_mesh(verts, faces)
        i += 1
    scale_mesh(dims, verts)
    cells = [("triangle", faces)]
    mesh = meshio.Mesh(np.array(verts), cells)
    meshio.write("{}.vtk".format(out_name), mesh, file_format="vtk",)


def scale_mesh(dims, verts):
    """
    Scales the icosasphere mesh to ellipsoid.
    Works in place.

    Parameters:
        dims: [a, b, c] ellipsoid dimensions
        verts: vertices as numpy array
    Returns:
        None
    """
    verts[:, 0] *= dims[0]
    verts[:, 1] *= dims[1]
    verts[:, 2] *= dims[2]


def subdiv_mesh(verts, faces):
    """
    Subdivide the icosahedron mesh by midpointx projected on the unit circle.
    Appends to the original mesh. Check if copy or reference?

    Parameters:
        verts: vertices of the original mesh as (N, 3) ndarray
        faces: faces of the original mesh (indicies)
    Returns:
        (verts, new_faces)
    """
    edge_cache = dict() # track edges divided: "edge as tuple of indicies": index of new mid
    new_verts = list(verts)
    new_faces = []
    for tri in faces:
        edge_set = (
            frozenset((tri[0], tri[1])),
            frozenset((tri[2], tri[1])),
            frozenset((tri[0], tri[2]))
        )
        verts_tup = (
            (verts[tri[0]], verts[tri[1]]),
            (verts[tri[2]], verts[tri[1]]),
            (verts[tri[0]], verts[tri[2]]),
        )
        x_i = [tri[0], tri[1], tri[2], 0., 0., 0.,]
        for j, ind in enumerate(edge_set):
            if ind not in edge_cache:
                new_ind = len(new_verts)
                edge_cache[ind] = new_ind
                x_i[j+3] = new_ind
                new_verts.append(mid_pt(verts_tup[j]))
            else:
                x_i[j+3] = edge_cache[ind]
        new_faces.append([x_i[0], x_i[3], x_i[5]])
        new_faces.append([x_i[3], x_i[1], x_i[4]])
        new_faces.append([x_i[5], x_i[4], x_i[2]])
        new_faces.append([x_i[3], x_i[4], x_i[5]])
    return (np.array(new_verts), np.array(new_faces))


def make_icosahedron():
    """
    Makes an icosahedron mesh scaled to the unit sphere.

    Returns:
    (verts, faces): 12 vertices and 20 faces
    """
    phi = (1. + np.sqrt(5)) / (2.)
    verts = np.array([
        [-1., phi, 0],
        [1., phi, 0],
        [-1., -phi, 0],
        [1., -phi, 0],

        [0., -1., phi],
        [0., 1., phi],
        [0., -1., -phi],
        [0., 1., -phi],

        [phi, 0., -1.],
        [phi, 0., 1.],
        [-phi, 0., -1.],
        [-phi, 0., 1.],
    ])
    faces = np.array([
        # around point 0
        [0, 11, 5],
        [0, 5, 1],
        [0, 1, 7],
        [0, 7, 10],
        [0, 10, 11],

        [1, 5, 9],
        [5, 11, 4],
        [11, 10, 2],
        [10, 7, 6],
        [7, 1, 8],

        [3, 9, 4],
        [3, 4, 2],
        [3, 2, 6],
        [3, 6, 8],
        [3, 8, 9],

        [4, 9, 5],
        [2, 4, 11],
        [6, 2, 10],
        [8, 6, 7],
        [9, 8, 1],
    ])
    length = np.linalg.norm(verts[0])
    return (verts / length, faces)


def mid_pt(pts):
    """
    Finds the middle point of the two vertices projected on the unit sphere

    Parameters:
        pts: two vertices as tuple of (3,) ndarray
    Returns:
        mid_pt: middle point projected on unit sphere (length to 1.)
    """
    mid_v = (pts[0] + pts[1]) / 2.
    length = np.linalg.norm(mid_v)
    return mid_v / length


if __name__ == "__main__":
    main()
