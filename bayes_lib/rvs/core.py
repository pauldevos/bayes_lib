import tensorflow as tf

from .transform import *
from ..model import *

class RandomVariable(tf.Variable):
    
    # Can the log density be differentiated w.r.t. the parameters?
    is_differentiable = False

    # Can samples be reparameterized as a function of simple noise?
    is_reparameterizable = False

    def __init__(self, observed, default_value, transform = None, *args, **kwargs):
        self.transform = transform
        if observed is None:
            self.observed = default_value
            self.is_observed = False
            trainable = True
        else:
            self.observed = observed
            self.is_observed = True
            self.jdet = 1.
            trainable = False
        super().__init__(default_value, trainable = trainable, *args, **kwargs)
        Model.get_context().add_random_variable(self)

    def transform_det(self, value):
        if self.transform is not None:
            uv = self.transform.inverse_transform(value)
            det = self.transform.transform_jacobian_det(value)
            return uv, det
        else:
            return value, 1.

    def transform_assign(self, value):
        uv, det = self.transform_det(value)
        tv = self.assign(uv)
        self.jdet = det
        return tv

    def transform_value(self, value):
        uv, _ = self.transform_det(value)
        return uv

    def value(self):
        if self.is_observed:
            return self.observed
        else:
            return super().value()

    @abc.abstractmethod
    def log_density(self):
        return
    
    @abc.abstractmethod
    def sample(self, shape):
        return

    def log_density_and_jacobian(self):
        return self.log_density() + tf.log(self.jdet)

class DefaultConstrainedRandomVariable(RandomVariable):

    def transform_det(self, value):
        x = self.default_transform.inverse_transform(value)
        jdet = self.default_transform.transform_jacobian_det(value)
        x2, jdet2 = super().transform_det(x)
        return x2, jdet * jdet2

class PositiveRandomVariable(DefaultConstrainedRandomVariable):
    
    def __init__(self, observed, default_value, trasnform = None, *args, **kwargs):
        self.default_transform = LowerBoundRVTransform(0)
        if isinstance(transform, LowerBoundRVTransform):
            transform = None
        super().__init__(observed, default_value, transform = transform, *args, **kwargs)

class BoundedRandomVariable(DefaultConstrainedRandomVariable):

    def __init__(self, observed, default_value, a, b, transform = None, *args, **kwargs):

        self.default_transform = LowerUpperBoundRVTransform(a, b)
        if isinstance(transform, LowerUpperBoundRVTransform):
            transform = None
        super().__init__(observed, default_value, transform = transform, *args, **kwargs)
