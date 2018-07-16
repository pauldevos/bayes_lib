from .core import *

import autograd.numpy as agnp
import autograd.scipy as agsp

class Gamma(PositiveRandomVariable):

    is_differentiable = True 
    
    def __init__(self, name, shape, scale, dimensions = 1, transform = None, observed = None):
        super().__init__(name, transform = transform, observed = observed)
        self.shape = shape
        self.scale = scale
        self.set_dependencies([shape, scale])
    
    def log_density(self, value, shape, scale):
        if agnp.all(self.dimensions == agnp.array([1])):
            return agsp.stats.gamma.logpdf(value, shape, scale = scale)
        else:
            return agnp.sum(agsp.stats.gamma.logpdf(value,shape,scale = scale))
