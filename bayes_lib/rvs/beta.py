from .core import *

import autograd.scipy as agsp
from autograd.scipy.stats import beta

class Beta(PositiveRandomVariable):

    is_differentiable = True 
    
    def __init__(self, name, alpha, beta, dimensions = 1, transform = None, observed = None):
        super().__init__(name, dimensions = dimensions, transform = transform, observed = observed)
        self.alpha = alpha
        self.beta = beta
        self.set_dependencies([alpha, beta])

    def log_density(self, value, alpha, beta):
        if agnp.all(self.dimensions == agnp.array([1])):
            return agsp.stats.beta.logpdf(value, alpha, scale = beta)
        else:
            return agnp.sum(agsp.stats.beta.logpdf(value, alpha, scale = beta))
