from .core import *

import tensorflow as tf

class Multivariate_Normal(RandomVariable):

    is_differentiable = True
    is_reparameterizable = True
    
    def __init__(self, mu, cov, observed = None, transform = None, default_value = 0., *args, **kwargs):
        super().__init__(observed, default_value, transform = transform, *args, **kwargs)
        self.mu = mu
        self.cov = cov

    def log_density(self):
        return tf.reduce_sum(tf.contrib.distributions.MultivariateNormalFullCovariance(self.mu, self.cov).log_prob(self.value()))

    def sample(self, shape):
        return tf.random_normal(shape) * self.cov + self.mu

