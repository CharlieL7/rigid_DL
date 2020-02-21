"""
Simple class to hold the mesh positions, faces, velocties, and tractions
Three node flat triangles
Has functions to calculate properties on the mesh.
"""
import csv
import sys
import re
import numpy as np
import geometric as gm
import gauss_quad as gq

class slm_UF:

    def __init__(self, par_dict):
        """
        Constructor for the simple_mesh object.

        Parameters:
            par_dict : dictionary of parameters
                x : vertices in list-like object (N, 3)
                faces : list-like with indices to vertices of a triangle
                v : velocities at each vertex
                f : tractions at each vertex
                visc_rat : viscosity ratio
                Ca : capillary number
                De : Deborah number
        """
        # (N, 3) ndarray
        self.vertices = par_dict["x"] # positions

        # (N, 3) ndarray
        self.faces = par_dict["faces"] # connectivity

        # (N, 3) ndarray
        self.v = par_dict["v"] # velocities

        # (N, 3) ndarray
        self.f = par_dict["f"] # tractions

        self.visc_rat = par_dict["visc_rat"]
        self.Ca = par_dict["Ca"]
        self.De = par_dict["De"]
        self.time = par_dict["time"]

        self.surf_area = self.calc_surf_area()
        self.center_mesh()
        self.centroid = self.calc_mesh_centroid()
        self.volume = self.calc_volume()
        self.mom_inertia = self.calc_moment_inertia_tensor()
        self.dims = (0., 0., 0.)


    @classmethod
    def read_dat(cls, in_name):
        """
        Reads a Tecplot human readable dat file to a simple linear mesh object
        Reads positions, connectivity, and tractions

        Parameters:
            in_name : input file name
        Returns:
            simple_linear_mesh class with the read in data
        """
        visc_rat = -1
        with open(in_name, 'r') as dat_file:
            is_header = True
            while is_header:
                tmp_line = dat_file.readline()
                if tmp_line.find('#') == 0: # if first character is #
                    eq_pos = tmp_line.find('=')
                    if eq_pos != -1:
                        if tmp_line.find("viscRat") != -1:
                            visc_rat = float(tmp_line[eq_pos+1:])
                        elif tmp_line.find("Ca") != -1:
                            Ca = float(tmp_line[eq_pos+1:])
                        elif tmp_line.find("shRate") != -1:
                            Ca = float(tmp_line[eq_pos+1:])
                        elif tmp_line.find("deformRate") != -1:
                            Ca = float(tmp_line[eq_pos+1:])
                        elif tmp_line.find("De") != -1:
                            De = float(tmp_line[eq_pos+1:])
                        elif tmp_line.find("time") != -1:
                            time = float(tmp_line[eq_pos+1:])
                else:
                    is_header = False
                    # should be at VARIABLES line
                    if tmp_line.find("VARIABLES") != -1:
                        field_names = re.sub("[^\w]", " ", tmp_line).split()[1:]
                        type_line = dat_file.readline().split()
                        nvert = int(type_line[1][2:])
                        nface = int(type_line[2][2:])
                    else:
                        sys.exit("Unrecognized data file, missing VARIABLES line")

            assert visc_rat != -1
            x_data = [] # position
            v_data = [] # velocities
            f_data = [] # tractions
            f2v = [] # connectivity
            try:
                dict_reader = csv.DictReader(dat_file, fieldnames=field_names, delimiter=' ')
                count = 0
                while count < nvert:
                    row = next(dict_reader)
                    x_data.append([row["X"], row["Y"], row["Z"]])
                    v_data.append([row["U"], row["V"], row["W"]])
                    f_data.append([row["F_0"], row["F_1"], row["F_2"]])
                    count += 1
                x_data = np.array(x_data, dtype=float)
                v_data = np.array(v_data, dtype=float)
                f_data = np.array(f_data, dtype=float)
            except csv.Error as e:
                sys.exit("dict_reader, file %s, line %d: %s" % (in_name, dict_reader.line_num, e))

            try:
                # switching readers as data changes to connectivity
                list_reader = csv.reader(dat_file, delimiter=' ')
                count = 0
                while count < nface:
                    lst = next(list_reader)[0:3] # should just be 3 values
                    f2v.append([int(i) for i in lst])
                    count += 1
                f2v = np.array(f2v, dtype=int)
                f2v -= 1 # indexing change
            except csv.Error as e:
                sys.exit("list_reader, file %s, line %d: %s" % (in_name, list_reader.line_num, e))
            ret = {"x":x_data, "faces":f2v, "v":v_data, "f":f_data, "visc_rat":visc_rat,
                    "Ca":Ca, "De":De, "time":time}

        return cls(ret)


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
            s_a += gq.int_over_tri(gm.const_func, nodes)
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
            x_c += gq.int_over_tri(gm.make_pos(nodes), nodes)
        x_c /= self.surf_area
        return x_c


    def calc_volume(self):
        """
        Calculates the volume of the linear mesh
        Adds up tetrahedron volumes from centroid to triange veritices

        Parameters:
            requires veritices, faces, and centroid to be set
        Returns:
            total volume of mesh
        """
        volume = 0.
        for f in self.faces:
            nodes = self.get_nodes(f)
            face_area = gq.int_over_tri(gm.const_func, nodes)
            xx = self.calc_tri_center(f) - self.centroid
            n = self.calc_normal(f)
            volume += 1./3. * face_area * np.dot(xx, n)
        return volume


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
            inertia_tensor += gq.int_over_tri(gm.make_inertia_func(nodes), nodes)

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
            nodes : (3, 3) ndarray of nodes as rows
                    note that numpy doesn't distinguish between rows and columns for "1D" arrays
        """
        x_0 = self.vertices[face[0]]
        x_1 = self.vertices[face[1]]
        x_2 = self.vertices[face[2]]
        nodes = np.stack((x_0, x_1, x_2), axis=0)
        return nodes


    def get_vels(self, face):
        """
        Gets the velocities of a face and puts into a (3, 3) matrix

        Paramters:
            face : the face to get nodes for, (3,) list-like integers
        Returns:
            vels : (3, 3) ndarray of vels as rows
                    note that numpy doesn't distinguish between rows and columns for "1D" arrays
        """
        v_0 = self.v[face[0]]
        v_1 = self.v[face[1]]
        v_2 = self.v[face[2]]
        vels = np.stack((v_0, v_1, v_2), axis=0)
        return vels


    def get_tractions(self, face):
        """
        Gets tractions of a face and puts into a (3, 3) matrix

        Paramters:
            face : the face to get nodes for, (3,) list-like integers
        Returns:
            tractions: (3, 3) ndarray of tractions as rows
                    note that numpy doesn't distinguish between rows and columns for "1D" arrays
        """
        f_0 = self.f[face[0]]
        f_1 = self.f[face[1]]
        f_2 = self.f[face[2]]
        tractions = np.stack((f_0, f_1, f_2), axis=0)
        return tractions



    def calc_normal(self, face):
        """
        Calculates the normal vector for a face

        Paramters:
            face : the face to get normal vector for, (3, ) list-like
        Returns:
            n : normalized normal vector, (3, ) ndarray
        """
        nodes = self.get_nodes(face)
        n = np.cross(nodes[1] - nodes[0], nodes[2] - nodes[0])
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
        tri_c = (1./3.) * (nodes[0] + nodes[1] + nodes[2])
        return tri_c


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
