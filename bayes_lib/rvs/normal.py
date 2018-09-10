from .core import *

import tensorflow as tf

class Normal(RandomVariable):
    
    is_differentiable = True
    is_reparameterizable = True

    def __init__(self, mu, sigma, observed = None, transform = None, default_value = 0., *args, **kwargs):
        super().__init__(observed, default_value, transform = transform, *args, **kwargs)
        self.mu = mu
        self.sigma = sigma
    
    def log_density(self):
        return tf.reduce_sum(tf.distributions.Normal(self.mu, self.sigma).log_prob(self.value()))

    def sample(self, shape):
        return tf.random_normal(shape) * self.sigma + self.mu
