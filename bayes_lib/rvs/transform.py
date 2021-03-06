import tensorflow as tf
import abc

from ..math.utils import logit, inv_logit

"""
Abstract Class defining a Random Variable transformation.
Defines how to transform from constrained to unconstrained,
how to transfrom from unconstrained to constrained, and how
to compute the determinant of the jacobian of the transform.
"""
class RVTransform(abc.ABC):
    
    # Defines transform from constrained to unconstrained
    @abc.abstractmethod
    def transform(self, x):
        return
    
    # Defines inverse transform from unconstrained to constrained
    @abc.abstractmethod
    def inverse_transform(self, y):
        return
    
    # Defines the determinant of jacobian of transform
    @abc.abstractmethod
    def transform_jacobian_det(self, y):
        return

"""
Collection of pre-defined transforms
"""
class LowerBoundRVTransform(RVTransform):

    def __init__(self, lb):
        self.__lb = lb

    def transform(self, x):
        return tf.log(x - self.__lb)

    def inverse_transform(self, y):
        return tf.exp(y) + self.__lb

    def transform_jacobian_det(self, y):
        return tf.exp(y)

class UpperBoundRVTransform(RVTransform):

    def __init__(self, ub):
        self.__ub = ub

    def transform(self, x):
        return tf.log(self.__ub - x)
    
    def inverse_transform(self, y):
        return self.__ub - tf.exp(y)

    def transform_jacobian_det(self, y):
        return tf.exp(y)

class LowerUpperBoundRVTransform(RVTransform):

    def __init__(self, lb, ub):
        self.__lb = lb
        self.__ub = ub

    def transform(self, x):
        u = (x - self.__lb)/(self.__ub - self.__lb)
        return logit(u)
    
    def inverse_transform(self, y):
        return self.__lb + (self.__ub - self.__lb) * inv_logit(y)

    def transform_jacobian_det(self, y):
        u = inv_logit(y)
        return (self.__ub - self.__lb) * u * (1 - u)

