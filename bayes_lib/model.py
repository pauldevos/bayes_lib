import numpy as np
import threading
import collections

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

    def __init__(self, name = None):
        self.name = name
        self.params = []

    def append_param(self, param):
        self.params.append(param)

    def get_param_vector(self):
        param_vec = []
        for i in range(len(self.params)):
            if not self.params[i].is_observed:
                param_vec.append(self.params[i].value)
        return np.array(param_vec)

    def set_param_vector(self, p_vals):
        for i in range(len(self.params)):
            if not self.params[i].is_observed:
                self.params[i].value = p_vals[i]
    
    def get_constrained_params(self):
        param_vec = [p.cvalue for p in self.params if not p.is_observed]
        return param_vec

    def log_density(self):
        logp = 0
        for i in range(len(self.params)):
            logp += self.params[i].log_density()
        return logp
