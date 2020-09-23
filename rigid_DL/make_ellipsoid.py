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
    parser.add_argument("-s", "--subdiv", type=int, help="number of subdivisions (used for both midpoint and geodesic)", default=0)
    parser.add_argument("-q", "--is_quad", help="convert to quadratic mesh", default=False, action="store_true")
    parser.add_argument("-g", "--geodesic", help="use the geodesic subdivision", default=False, action="store_true")
    args = parser.parse_args()
    dims = args.dims
    out_name = args.out
    num_subdiv = args.subdiv
    verts, faces = make_icosahedron()
    if args.geodesic:
        verts, faces = geodesic_div(verts, faces, num_subdiv)
    else:
        i = 0
        while i < num_subdiv:
            verts, faces = subdiv_mesh(verts, faces)
            i += 1
    verts = scale_mesh(dims, verts)
    if args.is_quad:
        verts, faces = conv_to_quad(verts, faces)
        cells = [("triangle6", faces)]
        mesh = meshio.Mesh(np.array(verts), cells)
    else:
        cells = [("triangle", faces)]
        mesh = meshio.Mesh(np.array(verts), cells)
    meshio.write("{}.vtk".format(out_name), mesh, file_format="vtk",)


def scale_mesh(dims, verts):
    """
    Scales the icosasphere mesh to ellipsoid.

    Parameters:
        dims: [a, b, c] ellipsoid dimensions
        verts: vertices as numpy array
    Returns:
        scaled vertices
    """
    # starting with a unit sphere, so transformation divided by 2
    abc = np.array(dims) / 2.
    return np.einsum("ij,j->ij", verts, abc)


def subdiv_mesh(verts, faces):
    """
    Subdivide the icosahedron mesh by midpointx projected on the unit circle.
    Appends to the original mesh.

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
        # vertices in original tri
        x_i = [tri[0], tri[1], tri[2], 0., 0., 0.,]
        for j, ind in enumerate(edge_set):
            if ind not in edge_cache:
                new_ind = len(new_verts)
                edge_cache[ind] = new_ind
                x_i[j+3] = new_ind
                new_verts.append(calc_mid_pt(verts_tup[j]))
            else:
                x_i[j+3] = edge_cache[ind]
        new_faces.append([x_i[0], x_i[3], x_i[5]])
        new_faces.append([x_i[3], x_i[1], x_i[4]])
        new_faces.append([x_i[5], x_i[4], x_i[2]])
        new_faces.append([x_i[3], x_i[4], x_i[5]])
    return (np.array(new_verts), np.array(new_faces))


def geodesic_div(verts, faces, n):
    """
    Splits each edge of the triangular mesh into n segments and adds in
    the nessesary edges. Equivalent to (n, 0) frequency geodesic sphere.
    Using an algorithm that builds the new faces from the base to top as
    would be done by hand.
    Everything is counterclockwise.

    Parameters:
        verts: verticies
        faces: triangular faces
        n: number segments to split each edge into
    """
    assert n >= 1
    # track edges divided: "edge as tuple of indicies": indicies of edge nodes incl. original
    # not the same as in subdiv_mesh
    edge_cache = dict()
    new_verts = list(verts)
    new_faces = []
    for tri in faces:
        edge_set = (# frozenset for hashing, order vertx do not matter
            frozenset((tri[0], tri[1])),
            frozenset((tri[2], tri[1])),
            frozenset((tri[0], tri[2])),
        )
        edge_ori = (
            (tri[0], tri[1]),
            (tri[1], tri[2]),
            (tri[0], tri[2]),
        )
        verts_tup = (
            (verts[tri[0]], verts[tri[1]]),
            (verts[tri[2]], verts[tri[1]]),
            (verts[tri[0]], verts[tri[2]]),
        )
        # make or get divided edge nodes
        e = [] # divided edge vertex indicies
        for j, ed_set in enumerate(edge_set):
            if ed_set not in edge_cache:
                new_inds = add_n_pts(new_verts, verts_tup[j], n)
                div_edge = np.concatenate(edge_ori[j][0], new_inds, edge_ori[j][1])
                edge_cache[ed_set] = div_edge
            else:
                div_edge = edge_cache[ed_set]
                if div_edge[0] != edge_ori[j][0]:
                    div_edge = div_edge[::-1] # reverse order
            e.append(div_edge)

        # get triangle rows and make interior points
        rows = [] # jagged arrays building the triangle up (vert indicies)
        rows.append(e[0]) # bottom row
        m = n - 1 # row divisions counter
        for r in range(1, n+1):
            ind_0 = e[2][-r]
            ind_1 = e[1][r]
            x_0 = new_verts[ind_0]
            x_1 = new_verts[ind_1]
            new_inds = add_n_pts(new_verts, [x_0, x_1], m)
            div_row = np.concatenate(ind_0, new_inds, ind_1)
            rows.append(div_row)
            m -= 1

        # iterate over rows to make all faces
        g = n # tri to make counter
        for i, row in enumerate(rows[:-1]): # till second to last row
            a = row
            b = rows[i+1]
            for j in range(g): # lowers
                new_faces.append(np.array([a[j], a[j+1], b[j]]))
            for k in range(g - 1): # uppers
                new_faces.append(np.array([b[k], a[k+1], b[k+1]]))
            g -= 1
    return (np.array(new_verts), np.array(new_faces))


def proj_usphere(verts):
    """
    Project verticies onto the unit sphere
    """
    norms = np.linalg.norm(verts, axis=1)
    verts /= norms


def add_n_pts(all_verts, verts, n):
    """
    Splits line between two vertices into n segments.
    Works in place on the vertex list.
    Will return nothing for n = 1

    Parameters:
        all_verts: whole vertex list
        verts: two vertices to split as two (3,) ndarray
        n: number of segments to split into
    Returns:
        new_inds: new indicies for added points
    """
    assert n >= 1
    l_verts = len(all_verts)
    new_inds = np.arange(l_verts, l_verts + n - 1)
    for v in calc_n_split(n, verts):
        all_verts.append(v)
    return new_inds



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


def calc_mid_pt(pts):
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


def calc_n_split(n, pts):
    """
    Finds verticies to split two points into n sections

    Parameters:
        n: number of new segments
        pts: two vertices as tuple of (3,) ndarray
    Returns:
        new_pts: new points in a list of  (3,) ndarray
    """
    assert(n >= 1)
    new_verts = []
    for i in range(1, n):
        new_verts.append((pts[0] + pts[1]) * (i/(n)))
    return new_verts


def conv_to_quad(verts, faces):
    """
    Converts a linear mesh into a curved quadratic mesh by adding veritices
    at the midpoints of each triangle. Then updating the face list (keep
    number of faces the same).
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
        # indices of vertices in new triangle
        x_i = [tri[0], tri[1], tri[2], 0., 0., 0.,]
        for j, ind in enumerate(edge_set):
            if ind not in edge_cache:
                new_ind = len(new_verts)
                edge_cache[ind] = new_ind
                x_i[j+3] = new_ind
                v_0, v_1 = verts_tup[j]
                mid_pt = (v_0 + v_1) / 2 # exactly halfway
                new_verts.append(mid_pt)
            else:
                x_i[j+3] = edge_cache[ind]
        new_faces.append(x_i)
    return (np.array(new_verts), np.array(new_faces))


if __name__ == "__main__":
    main()
