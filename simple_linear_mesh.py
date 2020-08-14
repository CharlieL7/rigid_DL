"""
Simple class to hold the mesh positions and faces.
Three node flat triangles
Has functions to calculate properties on the mesh.
"""
import csv
import sys
import numpy as np
import geometric as geo
import gauss_quad as gq

class simple_linear_mesh:

    def __init__(self, x, f):
        """
        Constructor for the simple_mesh object.

        Parameters:
            x : vertices in list-like object (N, 3)
            f : list-like with indices to vertices of a triangle
                expecting 3 node triangles (N, 3)
        """
        # (N, 3) ndarray
        self.vertices = np.array(x)

        # (N, 3) ndarray
        self.faces = np.array(f)
        self.surf_area = self.calc_surf_area()
        self.center_mesh()
        self.centroid = self.calc_mesh_centroid()
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


    def calc_surf_area(self):
        """
        Calculates the surface area of the mesh

        Parameters:
            requires vertices, faces to be set
        Returns:
            total surface area of mesh
        """
        s_a = 0.0
        for f in self.faces: # get rows
            nodes = self.get_nodes(f)
            s_a += gq.int_over_tri_linear(geo.const_func, nodes)
        return s_a


    def calc_mesh_centroid(self):
        """
        Calculates the centroid of the mesh weighted by the element area

        Parameters:
            requires vertices, faces, surf_area to be set
        Returns:
            centroid as ndarray of shape (3, )
        """
        x_c = np.zeros(3)
        for f in self.faces:
            nodes = self.get_nodes(f)
            x_c += gq.int_over_tri_linear(geo.pos_linear, nodes)
        x_c /= self.surf_area
        return x_c


    def calc_moment_inertia_tensor(self):
        """
        Calculates the moment of inertia tensor
        Uses element area weighting

        Parameters:
            requires vertices, faces, surf_area, centroid
        Returns:
            moment of inertia as ndarray of shape (3, 3)
        """

        inertia_tensor = np.zeros((3, 3))
        for f in self.faces:
            nodes = self.get_nodes(f)
            inertia_tensor += gq.int_over_tri_linear(geo.inertia_func_linear, nodes)

        inertia_tensor /= self.surf_area
        return inertia_tensor


    def calc_rotation_vectors(self):
        """
        Calculates the rotation vectors (eigensolutions)

        Parameters:
            requires vertices, faces, surf_area, centroid, mom_inertia
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


    def calc_normal(self, face):
        """
        Calculates the normal vector for a face

        Paramters:
            face : the face to get normal vector for, (3, ) list-like
        Returns:
            n : normalized normal vector, (3, ) ndarray
        """
        nodes = self.get_nodes(face)
        n = np.cross(nodes[:, 1] - nodes[:, 0], nodes[:, 2] - nodes[:, 0])
        # make outwards pointing
        x_c2tri = self.calc_tri_center(face) - self.centroid
        if np.dot(n, x_c2tri) < 0.:
            n = -n
        return n / np.linalg.norm(n)


    def calc_tri_center(self, face):
        """
        Calculates the centroid point on a face

        Parameters:
            face : the face to get triangle center for, (3, ) list-like
        Returns:
            tri_c : (3, ) ndarray for triangle center
        """
        nodes = self.get_nodes(face)
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
