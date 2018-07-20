import abc
from ..core import Node
from .transform import *
from ..model import *

class RandomVariable(Node):

    is_differentiable = False
    transform = None
    #input_nodes = []

    def __init__(self, name, observed, dimensions = 1, transform = None):
        super().__init__(name)
        if dimensions == 1:
            self.dimensions = agnp.array([1])
        else:
            self.dimensions = dimensions
        if transform is not None:
            if not transform.is_differentiable:
                self.is_differentiable = False
            self.transform = transform
        if observed is not None:
            self.is_observed = True
            self.value = observed
            self.constrained_value = self.value
            self.jdet = 1
            self.dimensions = agnp.array(observed.shape)
        else:
            self.is_observed = False
            self.value = None
            self.constrained_value = None
        Model.get_context().add_random_variable(self)

    def set_dependencies(self, inodes):
        self.input_nodes = []
        for node in inodes:
            if isinstance(node, Node):
                node.consumers.append(self)
                self.input_nodes.append(node)
            else:
                n = Constant("C",node)
                n.consumers.append(self)
                self.input_nodes.append(n)

    # Transform stored value from unconstrained to constrained
    # and returns the correction term
    def apply_transform(self, v, det = False):
        if self.transform is not None:
            # Constrained
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

    def set_value(self, value):
        if not self.is_observed:
            self.value = value
            self.constrained_value, self.jdet = self.apply_transform(value, det = True) 

    @abc.abstractmethod
    def log_density(self, constrained_value, *args):
        return
    
    def log_density_and_jacobian(self, *args):
        return self.log_density(self.constrained_value, *args) + agnp.log(self.jdet)

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

class PositiveRandomVariable(DefaultConstrainedRandomVariable):
    
    def __init__(self, name, dimensions = 1, transform = None, observed = None):
        self.default_transform = LowerBoundRVTransform(0)
        if isinstance(transform, LowerBoundRVTransform):
            transform = None
        super().__init__(name, dimensions = dimensions, transform = transform, observed = observed)

class BoundedRandomVariable(DefaultConstrainedRandomVariable):

    def __init__(self, name, a, b, dimensions = 1, transform = None, observed = None):
        self.default_transform = LowerUpperBoundRVTransform(a, b)
        if isinstance(transform, LowerUpperBoundRVTransform):
            transform = None
        super().__init__(name, dimensions = dimensions, transform = transform, observed = observed)
