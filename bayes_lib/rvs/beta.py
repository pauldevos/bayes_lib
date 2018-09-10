from .core import *

import autograd.scipy as agsp
from autograd.scipy.stats import beta

class Beta(PositiveRandomVariable):

    is_differentiable = True 

    def __init__(self, alpha, beta, observed = None, transform = None, default_value = 1., *args, **kwargs):
        super().__init__(observed, default_value, transform = transform, *args, **kwargs)
        self.alpha = alpha
        self.beta = beta

    def log_density(self):
        return tf.reduce_sum(tf.distributions.Beta(self.alpha, self.beta).log_prob(self.value()))

    def sample(self, shape):
        return tf.distributions.Beta(self.alpha, self.beta).sample(shape)
