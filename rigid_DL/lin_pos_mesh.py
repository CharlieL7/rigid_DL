"""
Linear potential mesh
"""

import numpy as np
from rigid_DL.pot_mesh import Pot_Mesh

class Lin_Pot_Mesh(Pot_Mesh)

    def __init__(self, x, f, l2q, q2l):
        self.nodes = x
        self.faces = f
        self.lin_to_quad_map = l2q
        self.quad_to_lin_map = q2l


    def get_node(self, node_num):
        return self.node[node_num]
    

    def get_nodes(self):
        return self.nodes


    def get_faces(self):
        return self.faces

    
    @classmethod
    def make_from_quad_geo_mesh(cls, quad_geo_mesh):
        """
        Makes the linear poential mesh from the given quadratic geometric mesh

        Parameters:
            quad_geo_mesh: quadratic geometric mesh
        Returns:
            Lin_Pot_Mesh object
        """
        tmp_faces = []
        lin_vert_nums = []
        l2q = {}

        for quad_face in quad_geo_mesh.faces:
            node_nums = quad_face[0:3]
            tmp_faces.append(node_nums)
            for v_num in node_nums:
                if v_num not in lin_vert_nums:
                    lin_vert_nums.append(v_num)
        lin_vert_nums.sort()

        tmp_verts = []
        for vert_num in lin_vert_nums:
            tmp_verts.append(self.vertices[vert_num])
        lin_verts = np.array(tmp_verts)

        for i, v_num in enumerate(lin_vert_nums):
            l2q[i] = v_num
        q2l = dict((v, k) for k, v in l2q.items()) # values must be unique
        lin_faces = []
        for lin_face in tmp_faces:
            arr = np.empty(3)
            for i, quad_node_num in enumerate(lin_face):
                arr[i] = self.quad_to_lin_map[quad_node_num]
            lin_faces.append(arr)
        lin_faces = np.array(lin_faces, dtype=int)
        return cls(lin_verts, lin_faces, l2q, q2l)

