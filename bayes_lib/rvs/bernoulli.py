from .core import *

import tensorflow as tf

class Bernoulli(RandomVariable):

    is_differentiable = True
    
    def __init__(self, theta, observed = None, transform = None, default_value = 0.5, *args, **kwargs):
        super().__init__(observed, default_value, transform = transform, *args, **kwargs)
        self.theta = theta

    def log_density(self):
        return tf.reduce_sum((self.value()) * tf.log(self.theta) + (1 - self.value()) * tf.log(1 - self.theta))
