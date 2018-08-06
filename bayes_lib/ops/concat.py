from .core import *
import autograd.numpy as agnp

class concat(Operation):

    is_differentiable = True

    def __init__(self, name, nodes):
        super().__init__(name, nodes)

    def compute(self, *nodes):
        return agnp.hstack(nodes)


