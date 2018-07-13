import autograd.numpy as agnp
import autograd.scipy as agsp
import numpy as np
import abc
import collections

from ..model import Model
from .transform import LowerBoundRVTransform, LowerUpperBoundRVTransform

"""
Option like method which returns a sampled value from a random variable
or the passed in value if it is not a random variable
"""
def get_rv_value(rv, s = False):
    if isinstance(rv, RandomVariable):
        if s:
            return rv.sample(apply_transform = True)
        else:
            return rv.cvalue
    elif isinstance(rv, collections.Iterable):
        return agnp.array([get_rv_value(v, s = s) for v in rv])
    elif isinstance(rv, RandomVariableOperation):
        return rv.compute()
    else:
        return rv 

"""
Random Variable base class.  Defines a log density.
"""
# Abstract Base Class to represent a parameter that needs to be estimated
class RandomVariable(abc.ABC):

    # All values are R and unconstrained
    __value = None
    __cvalue = None
    __jdet = 1.
    __is_observed = False
    is_differentiable = False
    
    def __init__(self, name, transform = None, observed = None):
        self.name = name
        self.transform = transform
        if observed is not None:
            self.value = observed
            self.is_observed = True
        Model.get_context().append_param(self)
    
    # Transform stored value from unconstrained to constrained
    # and returns the correction term
    def apply_transform(self, v, det = False):
        if self.transform is not None:
            x = self.transform.inverse_transform(v)
            if det:
                jdet = self.transform.transform_jacobian_det(v)
                return x, jdet
            else:
                return x
        else:
            if det:
                return v, 1
            else:
                return v

    @property
    def name(self):
        return self.__name

    @name.setter
    def name(self, n):
        self.__name = n

    @property
    def cvalue(self):
        return self.__cvalue

    @cvalue.setter
    def cvalue(self, cv):
        self.__cvalue = cv

    @property
    def jdet(self):
        return self.__jdet

    @jdet.setter
    def jdet(self, jd):
        self.__jdet = jd
    
    @property
    def value(self):
        return self.__value

    @value.setter
    def value(self, v):
        if self.check_value(v):
            self.__value = v
            cval, jdet = self.apply_transform(self.__value, det = True)
            self.cvalue = cval
            self.jdet = jdet
        else:
            raise TypeError("Invalid value for random variable %s" % self.__name)

    @property
    def is_observed(self):
        return self.__is_observed

    @is_observed.setter
    def is_observed(self, o):
        self.__is_observed = o

    @property
    def transform(self):
        return self.__transform

    @transform.setter
    def transform(self, t):
        self.__transform = t

    @abc.abstractmethod
    def check_value(self, v):
        return

    @abc.abstractmethod
    def log_density(self):
        return

    def log_density_p(self, p):
        return
    
    @abc.abstractmethod
    def sample(self, apply_transform = True):
        return

class DefaultConstrainedRandomVariable(RandomVariable):

    # Transform stored value from unconstrained to constrained
    # and returns the correction term
    def apply_transform(self, v, det = False):
        x = self.default_transform.inverse_transform(v)
        if det:
            jdet = self.default_transform.transform_jacobian_det(v)
            x2, jdet_ret = super().apply_transform(x, det = det)
            return x2, jdet * jdet_ret
        else:
            x2 = super().apply_transform(x, det = det)
            return x2

"""
Defines an extension of RandomVariable that by 
default defines a Lower Bound Transform.
"""
class PositiveRandomVariable(DefaultConstrainedRandomVariable):
    
    def __init__(self, name, transform = None, observed = None):
        self.default_transform = LowerBoundRVTransform(0)
        if isinstance(transform, LowerBoundRVTransform):
            transform = None
        super().__init__(name, transform = transform, observed = observed)

"""
Defines an extenstion of a RandomVariable that by
default defines a set of bounds on a Random Variable
"""
class BoundedRandomVariable(DefaultConstrainedRandomVariable):

    def __init__(self, name, a, b, transform = None, observed = None):
        self.default_transform = LowerUpperBoundRVTransform(a, b)
        if isinstance(transform, LowerUpperBoundRVTransform):
            transform = None
        super().__init__(name, transform = transform, observed = observed)


class RandomVariableOperation(object):

    def __init__(self, name, inputs):
        self.name = name
        self.inputs = inputs

    @abc.abstractmethod
    def compute(self):
        return
