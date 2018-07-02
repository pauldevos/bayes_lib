from .core import *

import autograd.numpy as agnp
from autograd.scipy.stats import beta

class Beta(PositiveRandomVariable):

    is_differentiable = True 
    
    def __init__(self, name, alpha, beta, transform = None, observed = None):
        super().__init__(name, transform = transform, observed = observed)
        self.alpha = alpha
        self.beta = beta
        if observed is None:
            self.value = self.sample(apply_transform = False)

    def check_value(self, v):
        return True

    def log_density(self):
        v = self.cvalue
        jd = self.jdet
        lpdf = agnp.sum(beta.logpdf(v, get_rv_value(self.alpha), scale = get_rv_value(self.beta))) + agnp.log(jd)
        return lpdf

    def sample(self, apply_transform = True):
        z = beta.rvs(get_rv_value(self.shape, s = True), scale = get_rv_value(self.scale, s = True))
        if apply_transform:
            return self.apply_transform(z)
        else:
            return z


