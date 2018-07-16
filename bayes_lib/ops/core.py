import abc
from ..core import Node
from ..model import *

class Operation(Node):

    is_differentiable = False
    
    def __init__(self, name, input_nodes):
        super().__init__(name)
        self.input_nodes = []
        self.set_dependencies(input_nodes)
        Model.get_context().add_operation(self)

    def set_dependencies(self,input_nodes):
        for node in input_nodes:
            node.consumers.append(self)
            self.input_nodes.append(node)
        
    @abc.abstractmethod
    def compute(self, *args):
        return


