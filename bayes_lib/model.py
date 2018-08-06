import bayes_lib as bl

import autograd.numpy as agnp
import autograd.scipy as agsp
import autograd
import threading
import collections
import abc
from .core import *

class DuplicateNodeException(Exception):
    pass

class ModelNotDifferentiableException(Exception):
    pass

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


class Model(Context):

    is_differentiable = True
    _grad_log_density = None
    nodes_lp = None
    _n_params = 0

    def __init__(self):
        self.node_lookup = {}
        self.computation_graph = Graph()
        self.nodes_postorder = {}

    def add_placeholder(self, node):
        if node.name not in self.node_lookup:
            self.computation_graph.placeholders.append(node)
            self.node_lookup[node.name] = node
        else:
            raise DuplicateNodeException

    def add_constant(self, node):
        self.computation_graph.constants.append(node)

    def add_random_variable(self, node):
        if node.name not in self.node_lookup:
            self.computation_graph.random_variables.append(node)
            self.node_lookup[node.name] = node
            if not node.is_differentiable:
                self.is_differentiable = False
        else:
            raise DuplicateNodeException

    def add_operation(self, node):
        if node.name not in self.node_lookup:
            self.computation_graph.operations.append(node)
            self.node_lookup[node.name] = node
            if not node.is_differentiable:
                self.is_differentiable = False
        else:
            raise DuplicateNodeException
    
    @property
    def n_params(self):
        if self.nodes_lp is None:
            self._compile()
        return self._n_params

    @n_params.setter
    def n_params(self, n_params):
        self._n_params = n_params
    
    def _compile(self):

        # Get the computational nodes to iterate over in 
        # order of necessary computation
        self.nodes_lp = []
        def recurse(node):
            if not node.visited:
                if isinstance(node, bl.ops.Operation) or isinstance(node, bl.rvs.RandomVariable):
                    for input_node in node.input_nodes:
                        recurse(input_node)
                self.nodes_lp.append(node)
                node.visited = True
        for node in self.computation_graph.random_variables:
            recurse(node)
        
        # Get the indicies of unobserved r.v.s
        self.unobserved_indices = []
        start_idx = 0
        cur_parameter_pos = 0
        n_total_params = 0
        for i in range(len(self.computation_graph.random_variables)):
            if not self.computation_graph.random_variables[i].is_observed:
                start_idx = cur_parameter_pos
                n_params = agnp.prod(self.computation_graph.random_variables[i].dimensions)
                end_idx = start_idx + agnp.prod(self.computation_graph.random_variables[i].dimensions)
                cur_parameter_pos = end_idx
                self.unobserved_indices.append((i, slice(start_idx,end_idx)))
                n_total_params += n_params
        self.n_params = n_total_params

        # Create autograd derivative function
        if self.is_differentiable:
            self._grad_log_density = autograd.grad(self._log_density)

        # Reset node for later compilation
        self.computation_graph.reset()

    def grad_log_density(self, parameters, feed_dict = {}):
        if not self.is_differentiable:
            raise ModelNotDifferentiableException
        else:
            if self._grad_log_density is None:
                self._compile()

        if len(parameters.shape) >= 2:
            grad_lps = agnp.array([self._grad_log_density(parameters[i,:], feed_dict = feed_dict) for i in range(parameters.shape[0])])
            return grad_lps
        else:
            return self._grad_log_density(parameters, feed_dict = feed_dict)

    def log_density(self, parameters, feed_dict = {}):
        if self.nodes_lp is None:
            self._compile()
        
        if len(parameters.shape) >= 2:
            lps = agnp.array([self._log_density(parameters[i,:], feed_dict) for i in range(parameters.shape[0])])
            return lps
        else:
            return self._log_density(parameters, feed_dict = feed_dict)

    def _log_density(self, parameter, feed_dict = {}):
        lp =  0
        for i in range(len(self.unobserved_indices)):
            rv_idx, parameter_range = self.unobserved_indices[i]
            self.computation_graph.random_variables[rv_idx].set_value(parameter[parameter_range])
        
        for node in self.nodes_lp:
            if isinstance(node,Placeholder):
                node.output = feed_dict[node]
            elif isinstance(node,Constant):
                node.output = node.value
            elif isinstance(node,bl.rvs.RandomVariable):
                node.inputs = [input_node.output for input_node in node.input_nodes]
                node.output = node.constrained_value
                t = node.log_density_and_jacobian(*node.inputs)
                lp += t
            else:
                node.inputs = [input_node.output for input_node in node.input_nodes]
                node.output = node.compute(*node.inputs)
        return lp

    def set_param(self, parameter):
        if self.nodes_lp is None:
            self._compile()

        for i in range(len(self.unobserved_indices)):
            rv_idx, parameter_range = self.unobserved_indices[i]
            self.computation_graph.random_variables[rv_idx].set_value(parameter[parameter_range])

    def evaluate(self, node, feed_dict = {}):
        if node.name in self.nodes_postorder:
            nodes_postorder = self.nodes_postorder[node.name]
        else:
            nodes_postorder = []
            def recurse(node):
                if not node.visited:
                    if isinstance(node, bl.ops.Operation) or isinstance(node, bl.rvs.RandomVariable):
                        for input_node in node.input_nodes:
                            recurse(input_node)
                    nodes_postorder.append(node)
                    node.visited = True
            recurse(node)
            self.nodes_postorder[node.name] = nodes_postorder
        
        for node in nodes_postorder:
            if isinstance(node,Placeholder):
                node.output = feed_dict[node]
            elif isinstance(node,Constant):
                node.output = node.value
            elif isinstance(node,bl.rvs.RandomVariable):
                node.inputs = [input_node.output for input_node in node.input_nodes]
                node.output = node.constrained_value
            else:
                node.inputs = [input_node.output for input_node in node.input_nodes]
                node.output = node.compute(*node.inputs)
        return node.output

    def constrain_parameters(self, unconstrained_parameters):
        if self.nodes_lp is None:
            self._compile()
        
        if len(unconstrained_parameters.shape) == 1:
            unconstrained_parameters = unconstrained_parameters.reshape(1,-1)
        constrained_parameters = agnp.zeros(unconstrained_parameters.shape)
        for k in range(constrained_parameters.shape[0]):
            for i in range(len(self.unobserved_indices)):
                rv_idx, parameter_range = self.unobserved_indices[i]
                constrained_parameters[k,parameter_range] = self.computation_graph.random_variables[rv_idx].apply_transform(unconstrained_parameters[k,parameter_range])
        return constrained_parameters

class Placeholder(Node):

    def __init__(self, name, dimensions):
        super().__init__(name)
        self.value = None
        self.dimensions = dimensions
        Model.get_context().add_placeholder(self)

    def set_value(self, value):
        self.value = value

class Constant(Node):
    
    def __init__(self, name, value):
        super().__init__(name)
        if not isinstance(value, agnp.ndarray):
            self.dimensions = agnp.array([1])
        else:
            self.dimensions = agnp.array(value.shape)
        self.value = value
        Model.get_context().add_constant(self)
