from .core import *

import autograd.scipy as agsp
import autograd.numpy as agnp

class Uniform(BoundedRandomVariable):

    is_differentiable = True

    def __init__(self, name, lb, ub, dimensions = 1, transform = None, observed = None):
        super().__init__(name, lb, ub, dimensions = dimensions, transform = transform, observed = observed)
        self.lb = lb
        self.ub = ub
        self.set_dependencies([lb, ub])

    def log_density(self, value, lb, ub):
        if agnp.all(self.dimensions == agnp.array([1])):
            return -agnp.log(ub - lb)
        else:
            return -agnp.log(ub - lb) * value.shape[0]

