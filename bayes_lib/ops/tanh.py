from .core import *
import autograd.numpy as agnp

class tanh(Operation):

    is_differentiable = True

    def __init__(self, name, a):
        super().__init__(name, [a])

    def compute(self, a):
        return agnp.tanh(a)



