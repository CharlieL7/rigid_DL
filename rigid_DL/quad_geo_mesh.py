"""
Class to hold the mesh positions and faces.
Six node curved triangles
Has functions to calculate properties on the mesh.
"""

import numpy as np
from rigid_DL.geo_mesh import Geo_Mesh
import rigid_DL.geometric as geo
import rigid_DL.gauss_quad as gq

class Quad_Geo_Mesh(Geo_Mesh):

    def __init__(self, x, f):
        """
        Constructor for the simple_mesh object.
        Face numbers must be equivalent to potential mesh's.

        Parameters:
            x : verts in list-like object (Nv, 3)
            f : list-like with indices to verts of a triangle
                expecting 6 node curved triangles (Nf, 6)
                Indicies 0, 1, 2 should be the triangle verts
        """
        self.verts = np.array(x) # (Nv, 3) ndarray
        self.faces = np.array(f) # (Nf, 6) ndarray
        self.quad_n = self.calc_all_quad_n()
        self.quad_hs = self.calc_all_quad_hs()
        self.surf_area = self.calc_surf_area()
        self.centroid = self.calc_mesh_centroid()
        self.center_mesh();
        print("Center of mass:")
        print(self.centroid)
        self.flip_n()
        self.mom_inertia = self.calc_moment_inertia_tensor()
        (self.w, self.A_m) = self.calc_rotation_eig() #w is 3 ROWS of eigenvectors
        print("Surface area:")
        print(self.surf_area)
        print("Moment of inertia tensor:")
        print(self.mom_inertia)


    def get_verts(self):
        return self.verts


    def get_faces(self):
        return self.faces


    def get_centroid(self):
        return self.centroid


    def get_surface_area(self):
        return self.surf_area


    def get_w(self):
        """
        Return rotation vectors for rotation eigenfunction
        """
        return self.w


    def get_A_m(self):
        """
        Return eigenvalues of moment of inertia tensor
        """
        return self.A_m


    def get_tri_nodes(self, face_num):
        """
        Gets the nodes of a face and puts nodes into a (6, 3) matrix

        Paramters:
            face_num : face number
        Returns:
            nodes : (6, 3) ndarray of nodes as columns
        """
        face = self.faces[face_num]
        x_0 = self.verts[face[0]]
        x_1 = self.verts[face[1]]
        x_2 = self.verts[face[2]]
        x_3 = self.verts[face[3]]
        x_4 = self.verts[face[4]]
        x_5 = self.verts[face[5]]
        nodes = np.stack((x_0, x_1, x_2, x_3, x_4, x_5), axis=1)
        return nodes


    def get_tri_center(self, face_num):
        """
        Gets the center point on the face surface.

        Parameters:
            face_num: face number
        Returns:
            tri_c : (3, ) ndarray for triangle center
        """
        nodes = self.get_tri_nodes(face_num)
        pt = geo.pos_quadratic(1./3., 1./3., nodes)
        return pt


    def get_quad_n(self, face_num):
        """
        Returns normal vectors for Guassian quadrature.

        Parameters:
            face_num: face number
        Returns:
            quad_n: (3, 6) ndarray of normal vectors as columns
        """
        return self.quad_n[face_num]


    def get_hs(self, face_num):
        """
        Returns (6,) ndarray of hs value at quadrature points
        """
        return self.quad_hs[face_num]


    def calc_all_quad_n(self):
        """
        Calculates all of the normal vector values that will be used for six point
        Gaussian quadrature
        """
        Nf = self.faces.shape[0]
        normals = np.empty([Nf, 3, 6])
        for i, face in enumerate(self.faces):
            nodes = self.get_tri_nodes(i)
            normals[i] = gq.quad_n(nodes)
        return normals


    def calc_all_quad_hs(self):
        """
        Calculates all of the hs values that will be used for six point
        Gaussian quadrature
        """
        assert self.quad_n.any()
        return np.linalg.norm(self.quad_n, axis=1)


    def flip_n(self):
        """
        Checks the orientation of the normals and reorients them to point
        outwards from the mesh if required.
        """
        count = 0
        for i, face in enumerate(self.faces):
            x_c2tri = self.get_tri_center(i) - self.centroid
            normals = self.quad_n[i]
            for j in range(normals.shape[1]): #iterate over 6 normal vectors
                n = normals[:, j]
                if np.dot(n, x_c2tri) < 0.:
                    count += 1
                    self.quad_n[i][:,j] *= -1.
        if count > 0:
            print("flipped {} normal vectors".format(count))


    def calc_surf_area(self):
        """
        Calculates the surface area of the mesh

        Parameters:
            requires verts, faces to be set
        Returns:
            total surface area of mesh
        """
        s_a = 0.0
        for i, face in enumerate(self.faces):
            nodes = self.get_tri_nodes(i)
            s_a += gq.int_over_tri_quad(geo.const_func, nodes, self.quad_hs[i])
        return s_a


    def calc_mesh_centroid(self):
        """
        Calculates the centroid of the mesh weighted by the element area

        Parameters:
            requires verts, faces, surf_area to be set
        Returns:
            centroid as ndarray of shape (3, )
        """
        x_c = np.zeros(3)
        for i, face in enumerate(self.faces):
            nodes = self.get_tri_nodes(i)
            x_c += gq.int_over_tri_quad(geo.pos_quadratic, nodes, self.quad_hs[i])
        x_c /= self.surf_area
        return x_c
    

    def center_mesh(self):
        """
        Center the mesh so the centroid is at the origin
        """
        self.verts -= self.centroid
        self.centroid = self.calc_mesh_centroid()


    def calc_mesh_centroid_pc(self):
        """
        Centroid of mesh assuming point cloud
        """
        x_c = np.mean(self.verts, axis=0)
        return x_c


    def calc_moment_inertia_tensor(self):
        """
        Calculates the moment of inertia tensor
        Uses element area weighting
        """
        inertia_tensor = np.zeros((3, 3))
        for i, face in enumerate(self.faces):
            nodes = self.get_tri_nodes(i)
            inertia_tensor += gq.int_over_tri_quad(geo.inertia_func_quadratic, nodes, self.get_hs(i))
        return inertia_tensor


    def calc_rotation_eig(self):
        """
        Calculates the rotation vectors (eigenvectors of moment of inertia)
        Be careful of sphere case when this basis is no longer orthogonal.
        """
        eig_vals, eig_vecs = np.linalg.eig(self.mom_inertia)
        idx = eig_vals.argsort()
        eig_vals = eig_vals[idx]
        eig_vecs = eig_vecs[:, idx]
        w = eig_vecs.T
        return (w, eig_vals)


    def normal_func(self, xi, eta, nodes):
        """
        Normal function of the face.
        Values are cached for better runtime.

        Paramters:
            xi : first parametric variable
            eta : second parameteric variable
            nodes : six nodes of triangle as columns in 3x6 ndarray
        Returns:
            n : normal vector (3,) ndarray
        """
        e_xi = np.matmul(nodes, geo.dphi_dxi_quadratic(xi, eta, nodes))
        e_eta = np.matmul(nodes, geo.dphi_deta_quadratic(xi, eta, nodes))
        n = np.cross(e_xi, e_eta)

        pt2tri = self.get_tri_center(nodes) - self.centroid
        if np.dot(n, pt2tri) < 0.:
            n = -n

        n /= np.linalg.norm(n)
        return n


    def calc_ellip_dims(self):
        """
        Rotate the mesh by eigenvectors of the moment of inertia tensor and then get the lengths
        along each axis
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
        a = np.amax(x_rot[:, 0]) - np.amin(x_rot[:, 0])
        b = np.amax(x_rot[:, 1]) - np.amin(x_rot[:, 1])
        c = np.amax(x_rot[:, 2]) - np.amin(x_rot[:, 2])
        return (a, b, c)
