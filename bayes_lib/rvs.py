import numpy as np
from scipy.stats import norm, gamma, lognorm
import abc
import collections

from .model import Model
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
        return np.array([get_rv_value(v, s = s) for v in rv])
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
    
    def __init__(self, name, transform = None, observed = None):
        self.name = name
        self.transform = transform
        Model.get_context().append_param(self)
        if observed is not None:
            self.value = observed
            self.is_observed = True
    
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
    def log_density(self, v):
        return

    @abc.abstractmethod
    def sample(self, apply_transform = True):
        return

"""
Defines an extension of RandomVariable that by 
default defines a Lower Bound Transform.
"""
class PositiveRandomVariable(RandomVariable):
    
    def __init__(self, name, transform = None, observed = None):
        self.default_transform = LowerBoundRVTransform(0)
        if isinstance(transform, LowerBoundRVTransform):
            transform = None
        super().__init__(name, transform = transform, observed = observed)

    @property
    def value(self):
        return self.__value

    @value.setter
    def value(self, v):
        if self.check_value(v):
            self.__value = v
            pos = self.default_transform.inverse_transform(v)
            pos_jdet = self.default_transform.transform_jacobian_det(v)
            cval, jdet = self.apply_transform(pos, det = True)
            self.cvalue = cval
            self.jdet = pos_jdet * jdet

"""
Defines an extenstion of a RandomVariable that by
default defines a set of bounds on a Random Variable
"""
class BoundedRandomVariable(RandomVariable):

    def __init__(self, name, a, b, transform = None, observed = None):
        self.default_transform = LowerUpperBoundRVTransform(a, b)
        if isinstance(transform, LowerUpperBoundRVTransform):
            transform = None
        super().__init__(name, transform = transform, observed = observed)

    @property
    def value(self):
        return self.__value

    @value.setter
    def value(self, v):
        if self.check_value(v):
            self.__value = v
            pos = self.default_transform.inverse_transform(v)
            pos_jdet = self.default_transform.transform_jacobian_det(v)
            cval, jdet = self.apply_transform(pos, det = True)
            self.cvalue = cval
            self.jdet = pos_jdet * jdet


"""
Collection of Pre-defined standard random variables.
"""
class Normal(RandomVariable):

    def __init__(self, name, mean, sd, transform = None, observed = None):
        super().__init__(name, transform = transform, observed = observed)
        self.mean = mean
        self.sd = sd
        if observed is None:
            self.value = self.sample(apply_transform = False)

    def check_value(self, v):
        return True

    def log_density(self, v = None):
        if v is None:
            v = self.cvalue
            jd = self.jdet
        lpdf = np.sum(norm.logpdf(v, get_rv_value(self.mean), get_rv_value(self.sd)) + np.log(jd))
        return lpdf

    def sample(self, apply_transform = True):
        z = np.random.normal(get_rv_value(self.mean, s = True), get_rv_value(self.sd, s = True))
        if apply_transform:
            return self.apply_transform(z)
        else:
            return z

"""
NOT WORKING YET
"""
class Uniform(BoundedRandomVariable):

    def __init__(self, name, lb, ub, transform = None, observed = None):
        super().__init__(name, lb, ub, transform = transform, observed = observed)
        self.lb = lb
        self.ub = ub
        if observed is None:
            self.cvalue = self.sample(apply_transform = False)

    def check_value(self, v):
        return True

    def log_density(self, v = None):
        if v is None:
            v = self.cvalue
            jd = self.jdet
        sz = 1
        if not np.isscalar(v):
            sz = v.shape[0]
        lpdf = (-np.log(get_rv_value(self.ub) - get_rv_value(self.lb)) * sz) + np.log(jd)
        return lpdf

    def sample(self, apply_transform = True):
        z = np.random.uniform(get_rv_value(self.lb, s = True), get_rv_value(self.ub, s = True))
        if apply_transform:
            return self.apply_transform(z)
        else:
            return z

class Gamma(PositiveRandomVariable):
    
    def __init__(self, name, shape, scale, transform = None, observed = None):
        super().__init__(name, transform = transform, observed = observed)
        self.shape = shape
        self.scale = scale
        if observed is None:
            self.cvalue = self.sample(apply_transform = False)

    def check_value(self, v):
        return True

    def log_density(self, v = None):
        if v is None:
            v = self.cvalue
            jd = self.jdet
        lpdf = np.sum(gamma.logpdf(v, get_rv_value(self.shape), get_rv_value(self.scale))) + np.log(jd)
        return lpdf

    def sample(self, apply_transform = True):
        z = gamma.rvs(get_rv_value(self.shape, s = True), get_rv_value(self.scale, s = True))
        if apply_transform:
            return self.apply_transform(z)
        else:
            return z

class Beta(PositiveRandomVariable):

    def __init__(self, name, alpha, beta, transform = None, observed = None):
        super().__init__(name, transform = transform, observed = observed)
        self.alpha = alpha
        self.beta = beta
        if observed is None:
            self.cvalue = self.sample(apply_transform = False)

    def check_value(self, v):
        return True

    def log_density(self, v = None):
        if v is None:
            v = self.cvalue
            jd = self.jdet
        lpdf = np.sum(beta.logpdf(v, get_rv_value(self.alpha), get_rv_value(self.beta))) + np.log(jd)
        return lpdf
    
    def sample(self):
        z = beta.rvs(get_rv_value(self.alpha, s = True), get_rv_value(self.scale, s = True))
        if apply_transform:
            return self.apply_transform(z)
        else:
            return z

class LogNormal(PositiveRandomVariable):

    def __init__(self, name, mu, sigma, transform = None, observed = None):
        super().__init__(name, transform = transform, observed = observed)
        self.mu = mu
        self.sigma = sigma

    def check_value(self, v):
        return True

    def log_density(self, v = None):
        if v is None:
            v = self.cvalue
            jd = self.jdet
        lpdf = np.sum(norm.logpdf(v, get_rv_value(self.mu), get_rv_value(self.sigma)) + np.log(jd))
        return lpdf

    def sample(self):
        return self.apply_transform(np.random.normal(get_rv_value(self.mu, s = True), get_rv_value(self.sigma, s = True)))
