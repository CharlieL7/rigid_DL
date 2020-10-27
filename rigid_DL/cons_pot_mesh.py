"""
Constant potential mesh
"""

import numpy as np
from rigid_DL.pot_mesh import Pot_Mesh

class Cons_Pot_Mesh(Pot_Mesh):

    def get_node(self, face_num):
        return self.nodes[face_num]


    def get_nodes(self):
        return self.nodes


    def get_faces(self):
        return self.faces


    @classmethod
    def make_from_geo_mesh(cls, geo_mesh):
        """
        Makes the constant potential mesh from the center points
        on the triangles of the given geometric mesh.

        Parameters:
            geo_mesh: geometric mesh
        Returns:
            Cons_Pot_Mesh object
        """
        nodes = []
        faces = []
        for face_num, geo_face in enumerate(geo_mesh.faces):
            nodes.append(geo_mesh.get_tri_center(face_num))
            faces.append(geo_face)
        nodes = np.array(nodes)
        faces = np.array(faces)
        return cls(nodes, faces)
