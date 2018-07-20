from .core import *
import autograd.numpy as agnp

class make_covariance(Operation):

    is_differentiable = True

    def __init__(self, name, nodes, dim):
        self.dim = dim
        super().__init__(name, nodes)

    def compute(self, *nodes):
        return agnp.hstack(nodes).reshape(self.dim, self.dim)



