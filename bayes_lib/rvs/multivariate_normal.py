from .core import *

import autograd.scipy as agsp
import autograd.numpy as agnp

class Multivariate_Normal(RandomVariable):

    is_differentiable = True
    
    def __init__(self, name, mu, cov, dimensions = 1, transform = None, observed = None):
        super().__init__(name, dimensions = dimensions, transform = transform, observed = observed)
        self.mu = mu
        self.cov = cov
        self.set_dependencies([mu, cov])

    def log_density(self, value, mu, cov):
        cov = cov + 1e-6 * agnp.eye(cov.shape[0])
        if agnp.all(self.dimensions == agnp.array([1])):
            return agsp.stats.multivariate_normal.logpdf(value, mu, cov)
        else:
            return agnp.sum(agsp.stats.multivariate_normal.logpdf(value, mu, cov))

