from .core import *

import autograd.scipy as agsp
import autograd.numpy as agnp

class Variable(RandomVariable):

    is_differentiable = True
    
    def __init__(self, name, dimensions = 1, transform = None, observed = None):
        super().__init__(name, dimensions = dimensions, transform = transform, observed = observed)
        self.set_dependencies([])

    def log_density(self, value):
        return 0
