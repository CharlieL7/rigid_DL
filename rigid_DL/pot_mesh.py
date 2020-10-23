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
