from .core import *

import autograd.numpy as agnp
from scipy.stats import invgamma

class InverseGamma(PositiveRandomVariable):

    is_differentiable = False
    
    def __init__(self, name, shape, scale, dimensions = 1, transform = None, observed = None):
        super().__init__(name, dimensions = dimensions, transform = transform, observed = observed)
        self.shape = shape
        self.scale = scale
        self.set_dependencies([shape, scale])

    def log_density(self, value, shape, scale):
        if agnp.all(self.dimensions == agnp.array([1])):
            return invgamma.logpdf(value, shape, scale = scale)
        else:
            return agnp.sum(invgamma.logpdf(value, shape, scale = scale))
