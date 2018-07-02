from .core import *

import autograd.scipy as agsp
import autograd.numpy as agnp
import numpy as np

from scipy.stats import norm

class Normal(RandomVariable):

    is_differentiable = True

    def __init__(self, name, mean, sd, transform = None, observed = None):
        super().__init__(name, transform = transform, observed = observed)
        self.mean = mean
        self.sd = sd
        if observed is None:
            self.value = self.sample(apply_transform = False)

    def check_value(self, v):
        return True

    def log_density(self):
        v = self.cvalue
        jd = self.jdet
        lpdf = agnp.sum(agsp.stats.norm.logpdf(v, get_rv_value(self.mean), get_rv_value(self.sd)) + agnp.log(jd))
        return lpdf

    def sample(self, apply_transform = True):
        z = np.random.normal(get_rv_value(self.mean, s = True), get_rv_value(self.sd, s = True))
        if apply_transform:
            return self.apply_transform(z)
        else:
            return z


