"""
Simple class to hold the mesh positions and faces.
Three node flat triangles
Has functions to calculate properties on the mesh.
"""
import numpy as np
from rigid_DL.geo_mesh import Geo_Mesh
import rigid_DL.geometric as geo
import rigid_DL.gauss_quad as gq

class Lin_Geo_Mesh(Geo_Mesh):

    def __init__(self, x, f):
        """
        Constructor for the simple_mesh object.

        Parameters:
            x : verts in list-like object (Nv, 3)
            f : list-like with indices to verts of a triangle
                expecting 3 node triangles (Nf, 3)
        """
        # (Nv, 3) ndarray
        self.verts = np.array(x)
        # (Nf, 3) ndarray
        self.faces = np.array(f)
        self.normals = self.calc_all_n()
        self.hs = self.calc_all_hs()
        self.surf_area = self.calc_surf_area()
        self.centroid = self.calc_mesh_centroid()
        self.normalize_n() # normal vectors normalized
        self.mom_inertia = self.calc_moment_inertia_tensor()
        self.dims = self.calc_ellip_dims()


    def get_verts(self):
        return self.verts


    def get_faces(self):
        return self.faces


    def get_centroid(self):
        return self.centroid


    def get_surface_area(self):
        return self.surf_area


    def get_tri_nodes(self, face_num):
        """
        Gets the nodes of a face and puts nodes into a (3, 3) matrix

        Paramters:
            face_num : face number
        Returns:
            nodes : (3, 3) ndarray of nodes as columns
        """
        face = self.faces[face_num]
        x_0 = self.verts[face[0]]
        x_1 = self.verts[face[1]]
        x_2 = self.verts[face[2]]
        nodes = np.stack((x_0, x_1, x_2), axis=1)
        return nodes


    def get_tri_center(self, face_num):
        """
        Calculates the centroid point on a face

        Parameters:
            face_num : face number
        Returns:
            tri_c : (3, ) ndarray for triangle center
        """
        nodes = self.get_tri_nodes(face_num)
        pt = geo.pos_linear(1/3., 1/3., nodes)
        return pt


    def get_normal(self, face_num):
        """
        Return normal vector for face.

        Parameters:
            face_num: face numbers
        Returns:
            normal vector: (3,) ndarray
        """
        return self.normals[face_num]


    def get_hs(self, face_num):
        """
        Return hs value for face.

        Parameters:
            face_num: face numbers
        Returns:
            hs: scalar
        """
        return self.hs[face_num]


    def calc_all_n(self):
        """
        Calculates all of the mesh normals, (Nf, 3) ndarray.
        Orientation still needs to be checked.
        """
        Nf = self.faces.shape[0]
        normals = np.empty([Nf, 3])
        for i, face in enumerate(self.faces):
            nodes = self.get_tri_nodes(i)
            normals[i] = np.cross(nodes[:, 1] - nodes[:, 0], nodes[:, 2] - nodes[:, 0])
        return normals


    def calc_all_hs(self):
        """
        Calculates all of the hs values on the mesh
        """
        return np.linalg.norm(self.normals, axis=1)


    def normalize_n(self):
        """
        Checks the orientation of the normals and reorients them to point
        outwards from the mesh if required. Then normalized the normal vectors
        """
        for i, face in enumerate(self.faces):
            x_c2tri = self.get_tri_center(i) - self.centroid
            if np.dot(self.normals[i], x_c2tri) < 0.:
                self.normals[i] = -self.normals[i]
            self.normals[i] = self.normals[i] / self.hs[i]


    def calc_surf_area(self):
        """
        Calculates the surface area of the mesh
        """
        return 0.5 * np.sum(self.hs)


    def calc_mesh_centroid(self):
        """
        Calculates the centroid of the mesh weighted by the element area
        """
        x_c = np.zeros(3)
        for i, face in enumerate(self.faces):
            nodes = self.get_tri_nodes(i)
            x_c += gq.int_over_tri_lin(geo.pos_linear, nodes, self.hs[i])
        x_c /= self.surf_area
        return x_c


    def calc_moment_inertia_tensor(self):
        """
        Calculates the moment of inertia tensor
        Uses element area weighting
        """
        inertia_tensor = np.zeros((3, 3))
        for i, face in enumerate(self.faces):
            nodes = self.get_tri_nodes(i)
            inertia_tensor += gq.int_over_tri_lin(geo.inertia_func_linear, nodes, self.hs[i])
        inertia_tensor /= self.surf_area
        return inertia_tensor


    def calc_rotation_vectors(self):
        """
        Calculates the rotation vectors (eigenvectors of moment of inertia)
        Be careful of sphere case when this basis is no longer orthogonal.
        Pretty sure can just use cartesional unit vectors for every case.
        """
        if self.is_sphere:
            return np.identity(3)
        eig_vals, eig_vecs = np.linalg.eig(self.mom_inertia)
        w = eig_vecs.T
        return w


    def calc_ellip_dims(self):
        """
        Rotate the mesh by eigenvectors of the moment of inertia tensor and then
        get the lengths along each axis
        """

        [eigvals, eigvecs] = np.linalg.eig(self.mom_inertia)
        idx = eigvals.argsort()
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        # first vector positive x
        if eigvecs[0, 0] < 0:
            eigvecs[:, 0] = -eigvecs[:, 0]

        # making sure right-handed
        temp = np.cross(eigvecs[:, 0], eigvecs[:, 1])
        if np.dot(temp.T, eigvecs[:, 2]) < 0:
            eigvecs[:, 2] = -eigvecs[:, 2]

        x_rot = np.matmul(self.verts, eigvecs)
        a = (np.amax(x_rot[:, 0]) - np.amin(x_rot[:, 0])) / 2.
        b = (np.amax(x_rot[:, 1]) - np.amin(x_rot[:, 1])) / 2.
        c = (np.amax(x_rot[:, 2]) - np.amin(x_rot[:, 2])) / 2.
        return (a, b, c)
