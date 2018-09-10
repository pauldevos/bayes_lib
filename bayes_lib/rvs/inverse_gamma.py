from .core import *

import autograd.numpy as agnp
from scipy.stats import invgamma

class InverseGamma(PositiveRandomVariable):

    is_differentiable = True

    def __init__(self, name, alpha, beta, observed = None, transform = None, default_value = 1., *args, **kwargs):
        super().__init__(observed, default_value, transform = transform, *args, **kwargs)
        self.alpha = alpha
        self.beta = beta

    def log_density(self):
        return tf.reduce_sum(tf.distributions.InverseGamma(self.alpha, self.beta).log_prob(self.value()))

    def sample(self, shape):
        return tf.distributions.InverseGamma(self.alpha, self.beta).sample(shape)
