"""
Abstract class for geometric meshes
"""

from abc import ABC, abstractmethod

class Geo_Mesh(ABC):

    def __init__(self, x, f):
        self.verts = x
        self.faces = f


    @abstractmethod
    def get_verts(self):
        pass


    @abstractmethod
    def get_faces(self):
        pass


    @abstractmethod
    def get_centroid(self):
        pass


    @abstractmethod
    def get_tri_nodes(self, face_num):
        pass


    @abstractmethod
    def get_tri_center(self, face_num):
        pass


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
