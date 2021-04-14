"""
Quadratic potential mesh
"""

import numpy as np
from rigid_DL.pot_mesh import Pot_Mesh

class Quad_Pot_Mesh(Pot_Mesh):

    def __init__(self, x, f, l2q=None, q2l=None):
        self.nodes = x
        self.faces = f
        self.lin_to_quad_map = l2q
        self.quad_to_lin_map = q2l


    def get_node(self, node_num):
        return self.nodes[node_num]


    def get_nodes(self):
        return self.nodes


    def get_faces(self):
        return self.faces


    @classmethod
    def make_from_lin_geo_mesh(cls, lin_geo_mesh):
        """
        Makes the linear poential mesh from the given linear geometric mesh.

        Parameters:
            lin_geo_mesh: quadratic geometric mesh
        Returns:
            Quad_Pot_Mesh object
        """
        # TODO


    @classmethod
    def make_from_quad_geo_mesh(cls, quad_geo_mesh):
        """
        Makes the linear poential mesh from the given quadratic geometric mesh
        Just uses the geometric nodes directly.

        Parameters:
            quad_geo_mesh: quadratic geometric mesh
        Returns:
            Quad_Pot_Mesh object
        """
        return cls(quad_geo_mesh.get_verts(), quad_geo_mesh.get_faces())

