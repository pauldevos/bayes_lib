import numpy as np
import abc

from .math.utils import logit, inv_logit

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
    def transform_jacobian_det(self, x):
        return

class LowerBoundRVTransform(RVTransform):
    
    def __init__(self, lb):
        self.__lb = lb

    def transform(self, x):
        return np.log(x - self.__lb)

    def inverse_transform(self, y):
        return np.exp(y) + self.__lb

    def transform_jacobian_det(self, y):
        return np.exp(y)

class UpperBoundRVTransform(RVTransform):

    def __init__(self, ub):
        self.__ub = ub

    def transform(self, x):
        return np.log(self.__ub - x)
    
    def inverse_transform(self, y):
        return self.__ub - np.exp(y)

    def transform_jacobian_det(self, y):
        return np.exp(y)

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

