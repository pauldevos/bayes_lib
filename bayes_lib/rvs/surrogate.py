from .core import *

import autograd.scipy as agsp
import autograd.numpy as agnp

class SurrogateLikelihood(RandomVariable):

    def __init__(self, name, dependencies, observed, dimensions = 1):
        super().__init__(name, dimensions = 1, transform = transform, observed = observed)
        self.dependencies = dependencies
        self.set_dependencies(dependencies)
    
    @abc.abstractmethod
    def log_density(self, value, dependencies):
        return

class AutoregressiveSurrogateLikelihood(RandomVariable):

    is_differentiable = True

    def log_density(self, value, dependencies):
        return agnp.prod(

