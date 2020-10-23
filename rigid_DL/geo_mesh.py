"""
Abstract class for geometric meshes
"""

from abc import ABC, abstractmethod

class Geo_Mesh(ABC):

    def __init__(self, x, f):
        self.verticies = x
        self.faces = f


    @abstractmethod
    def get_verticies(self):
        pass


    @abstractmethod
    def get_faces(self):
        pass


    @abstractmethod
    def get_centroid(self):
        pass


    @abstractmethod
    def get_nodes(self, face_num):
        pass


    @abstractmethod
    def get_tri_center(self, face):
        pass
