from .core import *

import autograd.scipy as agsp
import autograd.numpy as agnp

class Bernoulli(RandomVariable):

    is_differentiable = True
    
    def __init__(self, name, theta, dimensions = 1, transform = None, observed = None):
        super().__init__(name, dimensions = dimensions, transform = transform, observed = observed)
        self.theta = theta
        self.set_dependencies([theta])

    def log_density(self, value, theta):
        if agnp.all(self.dimensions == agnp.array([1])):
            return (value) * theta + (1 - value) * (1 - theta)
        else:
            return agnp.sum(value * theta + (1 - value) * (1 - theta))
