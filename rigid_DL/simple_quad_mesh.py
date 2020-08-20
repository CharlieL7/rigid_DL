"""
Simple class to hold the mesh positions and faces.
Six node curved triangles
Has functions to calculate properties on the mesh.
"""

import numpy as np
import geometric as geo
import gauss_quad as gq

class simple_quad_mesh:

    def __init__(self, x, f):
        """
        Constructor for the simple_mesh object.

        Parameters:
            x : vertices in list-like object (Nv, 3)
            f : list-like with indices to vertices of a triangle
                expecting 6 node curved triangles (Nf, 3)
                Indicies 0, 1, 2 should be the triangle vertices
        """
        self.vertices = np.array(x) # (N, 3) ndarray
        self.faces = np.array(f) # (N, 3) ndarray
        self.quad_n = self.calc_all_quad_n()
        self.quad_hs = self.calc_all_quad_hs()
        self.surf_area = self.calc_surf_area()
        self.center_mesh()
        self.centroid = self.calc_mesh_centroid()
        self.mom_inertia = self.calc_moment_inertia_tensor()


    def calc_all_quad_n(self):
        """
        Calculates all of the normal vector values that will be used for seven point
        Gaussian quadrature
        """
        Nf = self.faces.shape[0]
        normals = np.empty([Nf, 3, 7])
        for i, face in enumerate(self.faces):
            nodes = self.get_nodes(face)
            normals[i] = gq.quad_n(nodes)
        return normals


    def calc_all_quad_hs(self):
        """
        Calculates all of the hs values that will be used for seven point
        Gaussian quadrature
        """
        assert self.quad_n.any()
        return np.linalg.norm(self.quad_n, axis=1)


    def calc_surf_area(self):
        """
        Calculates the surface area of the mesh

        Parameters:
            requires vertices, faces to be set
        Returns:
            total surface area of mesh
        """
        s_a = 0.0
        for i, face in enumerate(self.faces):
            nodes = self.get_nodes(face)
            s_a += gq.int_over_tri_quad(geo.const_func, nodes, self.quad_hs[i])
        return s_a


    def calc_mesh_centroid(self):
        """
        Calculates the centroid of the mesh weighted by the element area

        Parameters:
            requires verticies, faces, surf_area to be set
        Returns:
            centroid as ndarray of shape (3, )
        """
        x_c = np.zeros(3)
        for i, face in enumerate(self.faces):
            nodes = self.get_nodes(face)
            x_c += gq.int_over_tri_quad(geo.pos_quadratic, nodes, self.quad_hs[i])
        x_c /= self.surf_area
        return x_c


    def calc_moment_inertia_tensor(self):
        """
        Calculates the moment of inertia tensor
        Uses element area weighting

        Parameters:
            requires verticies, faces, surf_area, centroid
        Returns:
            moment of inertia as ndarray of shape (3, 3)
        """

        inertia_tensor = np.zeros((3, 3))
        for i, face in enumerate(self.faces):
            nodes = self.get_nodes(face)
            inertia_tensor += gq.int_over_tri_quad(
                geo.inertia_func_quadratic,
                nodes,
                self.quad_hs[i]
            )
        inertia_tensor /= self.surf_area
        return inertia_tensor


    def calc_rotation_vectors(self):
        """
        Calculates the rotation vectors (eigensolutions)

        Parameters:
            requires verticies, faces, surf_area, centroid, mom_inertia
        Returns:
            w : eigensolutions for rotations, ndarray (3, 3), rows are the vectors
        """
        eig_vals, eig_vecs = np.linalg.eig(self.mom_inertia)
        w = np.zeros((3, 3))
        for i in range(3):
            w[i] = eig_vecs[:, i] / (np.sqrt(eig_vals[i] * self.surf_area))
        return w


    def get_nodes(self, face):
        """
        Gets the nodes of a face and puts nodes into a (6, 3) matrix

        Paramters:
            face : the face to get nodes for, (3,) list-like integers
        Returns:
            nodes : (6, 3) ndarray of nodes as columns
        """
        x_0 = self.vertices[face[0]]
        x_1 = self.vertices[face[1]]
        x_2 = self.vertices[face[2]]
        x_3 = self.vertices[face[3]]
        x_4 = self.vertices[face[4]]
        x_5 = self.vertices[face[5]]
        nodes = np.stack((x_0, x_1, x_2, x_3, x_4, x_5), axis=1)
        return nodes


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

        pt2tri = self.calc_tri_center(nodes) - self.centroid
        if np.dot(n, pt2tri) < 0.:
            n = -n

        n /= np.linalg.norm(n)
        return n


    def calc_tri_center(self, nodes):
        """
        Gets the center point on the face surface.

        Parameters:
            nodes : six nodes of triangle as columns in 3x6 ndarray
        Returns:
            tri_c : (3, ) ndarray for triangle center
        """
        pt = geo.pos_quadratic(1./3., 1./3., nodes)
        return pt


    def center_mesh(self):
        """
        Move the mesh such that the centroid is at [0, 0, 0]
        """
        old_centroid = self.calc_mesh_centroid()
        self.vertices -= old_centroid


    def check_in_face(self, vert_num, face_num):
        """
        Checks if a vertex is contained in a face
        Return the local node index if found in the face
        Gives the first index if multiple (there should not be multiple for a valid mesh)

        Parameters:
            vert_num : global index for vertex
            face_num : index for face
        Returns:
            (is_singular, local_singular_index)
            is_sinuglar : if integral is singular
            local_singular_index : local index [0:N) of singular node
        """
        for i, node_global_ind in enumerate(self.faces[face_num]):
            if node_global_ind == vert_num:
                return (True, i)
        return (False, None)
