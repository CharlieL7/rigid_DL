"""
Abstract class for potential meshes
"""

from abc import ABC, abstractmethod

class Pot_Mesh(ABC):

    def __init__(self, x, f):
        self.nodes = x
        self.faces = f


    @abstractmethod
    def get_node(self, node_num):
        pass


    @abstractmethod
    def get_nodes(self):
        pass


    @abstractmethod
    def get_faces(self):
        pass


    def check_in_face(self, node_num, face_num):
        """
        Checks if a node is contained in a face
        Return the local node index if found in the face
        Gives the first index if multiple (there should not be multiple for a valid mesh)

        Parameters:
            node_num : global index for vertex
            face_num : index for face
        Returns:
            (is_singular, local_singular_index)
            is_sinuglar : if integral is singular
            local_singular_index : local index [0:N) of singular node
        """
        for i, node_global_ind in enumerate(self.faces[face_num]):
            if node_global_ind == node_num:
                return (True, i)
        return (False, None)
