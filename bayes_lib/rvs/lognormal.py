from .core import *

import autograd.scipy as agsp
import autograd.numpy as agnp

from scipy.stats import norm

class LogNormal(PositiveRandomVariable):

    is_differentiable = True

    def __init__(self, name, mu, sigma, dimensions = 1, transform = None, observed = None):
        super().__init__(name, dimensions = dimensions, transform = transform, observed = observed)
        self.mu = mu
        self.sigma = sigma
        self.set_dependencies([mu, sigma])

    def log_density(self, value, mu, sigma):
        if agnp.all(self.dimensions == agnp.array([1])):
            return agsp.stats.norm.logpdf(value, mu, sigma)
        else:
            return agnp.sum(agsp.stats.norm.logpdf(value, mu, sigma))
