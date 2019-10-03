import numpy as np
import geometric
import gauss_quad

class simple_mesh:
    """
    Simple class to hold the mesh positions and faces.
    three node flat triangles
    Has functions to calculate properties on the mesh.
    """

    def __init__(self, x, f):
        """
        Constructor for the simple_mesh object.

        Parameters:
            x : vertices in list-like object (N, 3)
            f : list-like with indices to vertices of a triangle
                expecting 3 node triangles (N, 3)
        """
        self.vertices = np.array(x)
        self.faces = np.array(f)
        self.surf_area = self.calc_surf_area()
        self.centroid = self.calc_centroid()

        # eigensolutions to translation and rotation
        self.v, self.w = np.array([])


    def calc_surf_area(self):
        """
        Calculates the surface area of the mesh

        Parameters:
            requires verticies, faces to be set
        Returns:
            total surface area of mesh
        """
        # tmp function that just returns 1.
        def tmp(_eta, _xi, _nodes): return 1.
        s_a = 0.0
        for f in self.faces: # get rows
            nodes = self.get_nodes(f)
            s_a += gauss_quad.int_over_tri(tmp, nodes)
        return s_a


    def calc_centroid(self):
        """
        Calculates the centroid of the mesh weighted by the element area

        Parameters:
            requires verticies, faces, surf_area to be set
        Returns:
            centroid as ndarray of shape (3, )
        """
        x_c = np.zeros([3, 1])
        for f in self.faces:
            nodes = self.get_nodes(f)
            x_c += gauss_quad.int_over_tri(geometric.pos, nodes)
        x_c /= self.surf_area
        return x_c


    def calc_eigensols(self):
        """
        Calculates the eigensolutions for translation and rotation

        Parameters:
            requires verticies, faces, surf_area, centroid
        Returns:
            (v, w)
            v : eigensolutions for translations
            w : eigensolutions for rotations
        """

    def get_nodes(self, face):
        """
        Gets the nodes of a face and puts nodes into a (3, 3) matrix of column vectors

        Paramters:
            face : the face to get nodes for
        Returns:
            nodes : (3, 3) ndarray of nodes as columns
        """
        x_0 = self.vertices[face[0]]
        x_1 = self.vertices[face[1]]
        x_2 = self.vertices[face[2]]
        nodes = np.stack((x_0, x_1, x_2), axis=1) # making into column vectors
        return nodes
