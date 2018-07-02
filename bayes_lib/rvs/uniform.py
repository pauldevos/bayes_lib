from .core import *

import autograd.scipy as agsp
import autograd.numpy as agnp
import numpy as np

class Uniform(BoundedRandomVariable):

    is_differentiable = True

    def __init__(self, name, lb, ub, transform = None, observed = None):
        super().__init__(name, lb, ub, transform = transform, observed = observed)
        self.lb = lb
        self.ub = ub
        if observed is None:
            self.cvalue = self.sample(apply_transform = False)

    def check_value(self, v):
        return True

    def log_density(self, v = None):
        if v is None:
            v = self.cvalue
            jd = self.jdet
        sz = 1
        if not agnp.isscalar(v):
            sz = v.shape[0]
        lpdf = (-agnp.log(get_rv_value(self.ub) - get_rv_value(self.lb)) * sz) + agnp.log(jd)
        return lpdf

    def sample(self, apply_transform = True):
        z = np.random.uniform(get_rv_value(self.lb, s = True), get_rv_value(self.ub, s = True))
        if apply_transform:
            return self.apply_transform(z)
        else:
            return z

