from .core import *

import autograd.scipy as agsp
import autograd.numpy as agnp
import numpy as np

from scipy.stats import norm

class SurrogateLikelihood(RandomVariable):
 
    def __init__(self, name, log_density, dependencies):
        super().__init__(name, transform = None, observed = log_density)
        self.dependencies = dependencies
        self.is_observed = True
        self.log_density = lambda: log_density(get_rv_value(self.dependencies))

    def check_value(self, v):
        return True

    def log_density(self):
        return None

    def sample(self, apply_transform = True):
        return None


