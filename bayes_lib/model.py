import autograd.numpy as agnp
import autograd
import threading
import collections

class ModelNotDifferentiableException(Exception):
   pass 

class DuplicateParameterNameException(Exception):
    pass
    
"""
Defines a Context object for a Model
"""

class Context(object):

    contexts = threading.local()
    
    # Attach self to context stack
    def __enter__(self):
        type(self).get_contexts().append(self)
        return self
    
    # Remove self from context stack
    def __exit__(self, typ, value, traceback):
        type(self).get_contexts().pop()
    
    @classmethod
    def get_context(cls):
        try:
            return cls.get_contexts()[-1]
        except:
            raise TypeError("No context!")

    @classmethod
    def get_contexts(cls):
        if not hasattr(cls.contexts, 'stack'):
            cls.contexts.stack = []
        return cls.contexts.stack

"""
Model wrapper encompasses a probability model that contains
parameters and specifies the computation of a log_density through
the parameters
"""
class Model(Context):

    _gld = None
    is_differentiable = True 

    def __init__(self, name = None):
        self.name = name
        self.param_lkup = {}
        self.params = []
        self.observed_params = []
        self.unobserved_params = []
        # Number of unobserved params
        self.n_params = 0

    def append_param(self, param):
        if param.name in self.param_lkup:
            raise DuplicateParameterNameException
        else:
            self.param_lkup[param.name] = param

        self.params.append(param)
        if not param.is_observed:
            self.unobserved_params.append(param)
            self.n_params += 1
        else:
            self.observed_params.append(param)
        
        if not param.is_differentiable:
            self.is_differentiable = False
        
    def get_param_vector(self):
        param_vec = []
        for i in range(len(self.unobserved_params)):
            param_vec.append(self.unobserved_params[i].value)
        return agnp.array(param_vec)

    def set_param_vector(self, p_vals):
        for i in range(len(self.unobserved_params)):
            self.unobserved_params[i].value = p_vals[i]

    def transform_param_vector(self, p_vals):
        param_vec = []
        for i in range(len(self.unobserved_params)):
            param_vec.append(self.unobserved_params[i].apply_transform(p_vals[i]))
        return agnp.array(param_vec)
    
    def get_constrained_params(self):
        param_vec = agnp.array([p.cvalue for p in self.unobserved_params])
        return param_vec

    def log_density(self):
        logp = 0
        for i in range(len(self.params)):
            logp += self.params[i].log_density()
        return logp

    def log_likelihood(self):
        logp = 0
        for i in range(len(self.observed_params)):
            logp += self.observed_params[i].log_density()
        return logp

    def log_density_p(self, param_vec):
        self.set_param_vector(param_vec)
        return self.log_density()

    def compile_gradient_(self):
        if not self.is_differentiable:
            raise ModelNotDifferentiableException
        self._gld = autograd.grad(self.log_density_p)

    def grad_log_density(self):
        if self._gld is None:
            self.compile_gradient_()
        return self._gld(self.get_param_vector())

    def grad_log_density_p(self,param_vec):
        if self._gld is None:
            self.compile_gradient_()
        return self._gld(param_vec)

        
