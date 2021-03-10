"""
Derivative version of Lin_Geo_Mesh
This version uses the analytical normal vectors on the ellipsoidal surface.
Requires mesh verticies to be on the ellipsoidal surface.
"""
import numpy as np
from rigid_DL.lin_geo_mesh import Lin_Geo_Mesh

class Lin_Geo_Mesh_NV(Lin_Geo_Mesh):

    def __init__(self, dims, x, f):
        """
        Constructor for the simple_mesh object.
        Parameters:
            dims: ellipsoidal dimensions (3,)
            x : verts in list-like object (Nv, 3)
            f : list-like with indices to verts of a triangle
                expecting 3 node triangles (Nf, 3)
        """
        # (Nv, 3) ndarray
        self.verts = np.array(x)
        # (Nf, 3) ndarray
        self.faces = np.array(f)
        self.dims = dims
        self.normals = self.calc_all_n() # normals at veritices
        self.hs = self.calc_all_hs()
        self.surf_area = self.calc_surf_area()
        self.centroid = self.calc_mesh_centroid()
        self.mom_inertia = self.calc_moment_inertia_tensor()
        (self.w, self.A_m) = self.calc_rotation_eig()
        print("Surface area:")
        print(self.surf_area)
        print("Moment of inertia:")
        print(self.mom_inertia)


    def get_normal(self, vert_num):
        """
        Return normal vector for vertex.
        Parameters:
            vert_num: vertex number
        Returns:
            normal vector: (3,) ndarray
        """
        return self.normals[vert_num]


    def get_tri_normals(self, face_num):
        """
        Return the normal vectors at each vertex of a face
        """
        face = self.faces[face_num]
        n_0 = self.normals[face[0]]
        n_1 = self.normals[face[1]]
        n_2 = self.normals[face[2]]
        tri_normals = np.stack((n_0, n_1, n_2), axis=0)
        return tri_normals


    def calc_all_n(self):
        """
        Calculates all of the vertex normals
        """
        Nv = self.verts.shape[0]
        normals = np.empty([Nv, 3])
        for i, vert in enumerate(self.verts):
            n = 2. * np.divide(vert, np.power(self.dims, 2.))
            n /= np.linalg.norm(n) # normalize the vector
            normals[i] = n
        return normals


    def calc_all_hs(self):
        """
        Calculates all of the hs values on the mesh
        where hs is the triangle area * 2
        """
        Nf = self.faces.shape[0]
        normals = np.empty([Nf, 3])
        for i, face in enumerate(self.faces):
            nodes = self.get_tri_nodes(i)
            normals[i] = np.cross(nodes[1] - nodes[0], nodes[2] - nodes[0])
        return np.linalg.norm(normals, axis=1)
