from .core import *

import autograd.scipy as agsp
import autograd.numpy as agnp
import numpy as np

from scipy.stats import norm

class LogNormal(PositiveRandomVariable):

    def __init__(self, name, mu, sigma, transform = None, observed = None):
        super().__init__(name, transform = transform, observed = observed)
        self.mu = mu
        self.sigma = sigma

    def check_value(self, v):
        return True

    def log_density(self):
        v = self.cvalue
        jd = self.jdet
        lpdf = agnp.sum(agsp.stats.norm.logpdf(v, get_rv_value(self.mu), get_rv_value(self.sigma)) + agnp.log(jd))
        return lpdf

    def sample(self):
        return self.apply_transform(np.random.normal(get_rv_value(self.mu, s = True), get_rv_value(self.sigma, s = True)))

