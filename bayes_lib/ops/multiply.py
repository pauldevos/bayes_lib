from .core import *
import autograd.numpy as agnp

class multiply(Operation):

    is_differentiable = True

    def __init__(self, name, nodes):
        super().__init__(name, nodes)

    def compute(self, *nodes):
        return agnp.prod(agnp.array(nodes))



