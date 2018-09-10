from .core import *

import autograd.numpy as agnp
import autograd.scipy as agsp

class Gamma(PositiveRandomVariable):

    is_differentiable = True 
    
    def __init__(self, alpha, beta, observed = None, transform = None, default_value = 1.0, *args, **kwargs):
        super().__init__(observed, default_value, transform = transform, *args, **kwargs)
        self.alpha = alpha
        self.beta = beta
    
    def log_density(self):
        return tf.reduce_sum(tf.distributions.Gamm(self.alpha, self.beta).log_prob(self.value()))

    def sample(self, shape):
        return tf.distributions.Gamma(self.alpha, self.beta).sample(shape)
