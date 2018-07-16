from .core import *
import autograd.numpy as agnp

class Matmul(Operation):

    is_differentiable = True

    def __init__(self, name, a, b):
        super().__init__(name, [a,b])

    def compute(self, a, b):
        return agnp.dot(a, b)



