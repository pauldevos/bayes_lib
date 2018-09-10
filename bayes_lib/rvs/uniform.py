from .core import *

import tensorflow as tf

class Uniform(BoundedRandomVariable):

    is_differentiable = True
    is_reparameterizable = True

    def __init__(self, lb, ub, dims = 1, observed = None, transform = None, default_value = 0., *args, **kwargs):
        self.lb = float(lb)
        self.ub = float(ub)
        self.N = dims
        super().__init__(observed, default_value, self.lb, self.ub, transform = transform, *args, **kwargs)
    
    def log_density(self):
        return tf.log(self.ub - self.lb) * self.N

    def sample(self, shape):
        return tf.random_uniform(shape,0,1) * (self.ub - self.lb) + self.lb

