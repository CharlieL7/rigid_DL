"""
Simple class to hold the mesh positions and faces.
Three node flat triangles
Has functions to calculate properties on the mesh.
"""
import csv
import sys
import numpy as np
import rigid_DL.geometric as geo
import rigid_DL.gauss_quad as gq

class simple_linear_mesh:

    def __init__(self, x, f):
        """
        Constructor for the simple_mesh object.

        Parameters:
            x : vertices in list-like object (Nv, 3)
            f : list-like with indices to vertices of a triangle
                expecting 3 node triangles (Nf, 3)
        """
        # (N, 3) ndarray
        self.vertices = np.array(x)

        # (N, 3) ndarray
        self.faces = np.array(f)
        self.normals = self.calc_all_n()
        self.hs = self.calc_all_hs()
        self.surf_area = self.calc_surf_area()
        self.centroid = self.calc_mesh_centroid()
        self.reori_n()
        self.center_mesh()
        self.mom_inertia = self.calc_moment_inertia_tensor()
        self.dims = self.calc_ellip_dims()

        # eigensolutions to translation and rotation
        self.v = np.identity(3) / np.sqrt(self.surf_area)
        self.w = self.calc_rotation_vectors() # rows are the vectors


    @classmethod
    def read_dat(cls, in_name):
        """
        Reads a Tecplot human readable dat file to a simple linear mesh object
        Only reads the positions & connectivity

        Parameters:
            in_name : input file name
        Returns:
            simple_linear_mesh class with the read in data
        """
        with open(in_name, 'r') as dat_file:
        # ignore all headers
            is_header = True
            while is_header:
                tmp_line = dat_file.readline()
                if tmp_line.find('#') != 0:
                    is_header = False
                    # get the next two Tecplot lines and then go back one line

            try:
                reader = csv.reader(dat_file, delimiter=' ')
                type_line = next(reader)
                nvert = int(type_line[1][2:])
                nface = int(type_line[2][2:])
                x_data = [] # position
                f2v = [] # connectivity

                count = 0
                while count < nvert:
                    lst = next(reader)[0:3]
                    x_data.append(lst)
                    count += 1
                x_data = np.array(x_data, dtype=float)

                count = 0
                while count < nface:
                    lst = next(reader)[0:3] # should just be 3 values
                    f2v.append([int(i) for i in lst])
                    count += 1
                f2v = np.array(f2v, dtype=int)
                f2v -= 1 # indexing change

            except csv.Error as e:
                sys.exit('file %s, line %d: %s' % (in_name, reader.line_num, e))

        return cls(x_data, f2v)


    def calc_all_n(self):
        """
        Calculates all of the mesh normals, (Nf, 3) ndarray.
        Orientation still needs to be checked.
        """
        Nf = self.faces.shape[0]
        normals = np.empty([Nf, 3])
        for i, face in enumerate(self.faces):
            nodes = self.get_nodes(face)
            normals[i] = np.cross(nodes[:, 1] - nodes[:, 0], nodes[:, 2] - nodes[:, 0])
        return normals


    def calc_all_hs(self):
        """
        Calculates all of the hs values on the mesh
        """
        return np.linalg.norm(self.normals, axis=1)


    def reori_n(self):
        """
        Checks the orientation of the normals and reorients them to point
        outwards from the mesh if required
        """
        for i, face in enumerate(self.faces):
            nodes = self.get_nodes(face)
            x_c2tri = self.calc_tri_center(nodes) - self.centroid
            if np.dot(self.normals[i], x_c2tri) < 0.:
                self.normals[i] = -self.normals[i]


    def calc_surf_area(self):
        """
        Calculates the surface area of the mesh
        """
        s_a = 0.0
        for i, face in enumerate(self.faces): # get rows
            nodes = self.get_nodes(face)
            s_a += gq.int_over_tri_lin(geo.const_func, nodes, self.hs[i])
        return s_a


    def calc_mesh_centroid(self):
        """
        Calculates the centroid of the mesh weighted by the element area
        """
        x_c = np.zeros(3)
        for i, face in enumerate(self.faces):
            nodes = self.get_nodes(face)
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
            nodes = self.get_nodes(face)
            inertia_tensor += gq.int_over_tri_lin(geo.inertia_func_linear, nodes, self.hs[i])
        inertia_tensor /= self.surf_area
        return inertia_tensor


    def calc_rotation_vectors(self):
        """
        Calculates the rotation vectors (eigensolutions)
        """
        eig_vals, eig_vecs = np.linalg.eig(self.mom_inertia)
        w = np.zeros((3, 3))
        for i in range(3):
            w[i] = eig_vecs[:, i] / (np.sqrt(eig_vals[i] * self.surf_area))
        return w


    def get_nodes(self, face):
        """
        Gets the nodes of a face and puts nodes into a (3, 3) matrix

        Paramters:
            face : the face to get nodes for, (3,) list-like integers
        Returns:
            nodes : (3, 3) ndarray of nodes as columns
        """
        x_0 = self.vertices[face[0]]
        x_1 = self.vertices[face[1]]
        x_2 = self.vertices[face[2]]
        nodes = np.stack((x_0, x_1, x_2), axis=1)
        return nodes


    def calc_tri_center(self, nodes):
        """
        Calculates the centroid point on a face

        Parameters:
            nodes : three nodes of triangle as columns in 3x3 ndarray
        Returns:
            tri_c : (3, ) ndarray for triangle center
        """
        tri_c = (1./3.) * np.sum(nodes, axis=1)
        return tri_c


    def center_mesh(self):
        """
        Move the mesh such that the centroid is at [0, 0, 0]
        """
        old_centroid = self.calc_mesh_centroid()
        self.vertices -= old_centroid


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

        x_rot = np.matmul(self.vertices, eigvecs)
        a = np.amax(x_rot[:, 0]) - np.amin(x_rot[:, 0])
        b = np.amax(x_rot[:, 1]) - np.amin(x_rot[:, 1])
        c = np.amax(x_rot[:, 2]) - np.amin(x_rot[:, 2])
        return (a, b, c)



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


    def write_to_dat(self, out_name):
        """
        Writes the mesh to a Tecplot human readable dat file

        Parameters:
            out_name : string to write to
        """
        str_header = [
            "VARIABLES = X, Y, Z\n",
            "ZONE N={0} E={1} F=FEPOINT ET=TRIANGLE\n".format(self.vertices.shape[0], self.faces.shape[0])
            ]
        with open(out_name, 'w') as out:
            out.writelines(str_header)
            writer = csv.writer(out, delimiter=' ', lineterminator="\n")
            writer.writerows(self.vertices)
            writer.writerows(self.faces + 1)
