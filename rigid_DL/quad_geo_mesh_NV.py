"""
Derivative version of Quad_Geo_Mesh
This version uses the analytical normal vectors on the ellipsoidal surface.
Requires mesh verticies to be on the ellipsoidal surface.
"""
import numpy as np
from rigid_DL.quad_geo_mesh import Quad_Geo_Mesh
import rigid_DL.gauss_quad as gq

class Quad_Geo_Mesh_NV(Quad_Geo_Mesh):

    def __init__(self, dims, x, f):
        """
        Constructor for the simple_mesh object.
        Face numbers must be equivalent to potential mesh's.

        Parameters:
            dims: ellipsoidal dimensions (3,)
            x : verts in list-like object (Nv, 3)
            f : list-like with indices to verts of a triangle
                expecting 6 node curved triangles (Nf, 6)
                Indicies 0, 1, 2 should be the triangle verts
        """
        self.verts = np.array(x) # (Nv, 3) ndarray
        self.faces = np.array(f) # (Nf, 6) ndarray
        self.dims = dims
        self.vert_normals = self.calc_all_n()
        self.quad_n = self.calc_all_quad_n()
        self.quad_hs = self.calc_all_quad_hs()
        self.surf_area = self.calc_surf_area()
        self.centroid = self.calc_mesh_centroid()
        self.center_mesh()
        self.mom_inertia = self.calc_moment_inertia_tensor()
        (self.w, self.A_m) = self.calc_rotation_eig() #w is 3 ROWS of eigenvectors
        print("Surface area:")
        print(self.surf_area)
        print("Moment of inertia:")
        print(self.mom_inertia)
    

    def get_tri_normals(self, face_num):
        """
        Gets the normal vectors of a face and puts nodes into a
        (6, 3) matrix

        Paramters:
            face_num : face number
        Returns:
            nodes : (6, 3) ndarray of nodes as rows
        """
        face = self.faces[face_num]
        n_0 = self.vert_normals[face[0]]
        n_1 = self.vert_normals[face[1]]
        n_2 = self.vert_normals[face[2]]
        n_3 = self.vert_normals[face[3]]
        n_4 = self.vert_normals[face[4]]
        n_5 = self.vert_normals[face[5]]
        nodes = np.stack((n_0, n_1, n_2, n_3, n_4, n_5), axis=0)
        return nodes


    def calc_all_n(self):
        """
        Calculate all the vertex normals based on ellipsoid equation
        """
        Nv = self.verts.shape[0]
        normals = np.empty([Nv, 3])
        for i, vert in enumerate(self.verts):
            n = 2. * np.divide(vert, np.power(self.dims, 2.))
            n /= np.linalg.norm(n) # normalize the vector
            normals[i] = n
        return normals


    def calc_all_quad_n(self):
        """
        Calculates all of the normal vector values that will be used for six point
        Gaussian quadrature using quadratic interpolation of analytical normal vectors
        at the mesh verticies
        """
        Nf = self.faces.shape[0]
        quad_n = np.empty([Nf, 6, 3])
        for i, face in enumerate(self.faces):
            node_normals = self.get_tri_normals(i)
            quad_n[i] = gq.quad_n_NV(node_normals)
        return quad_n


    def calc_all_quad_hs(self):
        """
        Calculates all of the hs values that will be used for six point
        Gaussian quadrature. This version still requires over element normal
        vector calculation to get hs values.
        Returns:
            (Nf, 6, 3) ndarray
        """
        Nf = self.faces.shape[0]
        normals = np.empty([Nf, 6, 3])
        for i, face in enumerate(self.faces):
            nodes = self.get_tri_nodes(i)
            normals[i] = gq.quad_n(nodes)
        return np.linalg.norm(normals, axis=2)
