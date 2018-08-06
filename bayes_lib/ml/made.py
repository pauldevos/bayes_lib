import abc
import bayes_lib as bl
from bayes_lib.math.utils import *
from ..core import *
from ..ml.neural_network import *

import autograd.numpy as agnp
import autograd.scipy as agsp
import autograd

class MADE(bl.rvs.RandomVariable):

    is_differentiable = True

    def assign_degrees(self, input_dims, hidden_nodes):
        degrees = []
        degrees.append(np.arange(1, input_dims + 1))
        for N in hidden_nodes:
            degrees_l = agnp.arange(N) % max(1, input_dims - 1) + min(1, input_dims - 1)
            degrees.append(degrees_l)
        return degrees

    def create_masks(self, degrees, mode = 'sequential'):

        masks = []
        for i, (d0, d1) in enumerate(zip(degrees[:-1], degrees[1:])):
            M = d0[:, np.newaxis] <= d1
            masks.append(M)

        Mmp = degrees[-1][:, np.newaxis] < degrees[0]
        return masks, Mmp
    
    @abc.abstractmethod
    def log_density(self, value, *args):
        return

class GaussianMADE(MADE):

    def __init__(self, name, input_dimensions, hidden_nodes, observed, nonlinearity = sigmoid, last_layer_nonlinearity = linear):

        super().__init__(name, dimensions = input_dimensions, observed = observed)
        degrees = self.assign_degrees(input_dimensions, hidden_nodes)
        masks, llm = self.create_masks(degrees)
    
        self.made_nn_pre = MaskedNeuralNetwork('nn_base_%s' % (name), observed, [input_dimensions] + hidden_nodes, masks, nonlinearity = nonlinearity, last_layer_nonlinearity = nonlinearity)
        self.made_nn_mean = MaskedNeuralNetwork('nn_mean_%s' % (name), self.made_nn_pre, [hidden_nodes[-1], input_dimensions], [llm], nonlinearity = nonlinearity, last_layer_nonlinearity = last_layer_nonlinearity)
        self.made_nn_std = MaskedNeuralNetwork('nn_std_%s' % (name), self.made_nn_pre, [hidden_nodes[-1], input_dimensions], [llm], nonlinearity = nonlinearity, last_layer_nonlinearity = last_layer_nonlinearity)
        self.set_dependencies([self.made_nn_mean, self.made_nn_std])
    
    def log_density(self, value, made_nn_mean, made_nn_std):
        z = agnp.sum(agsp.stats.norm.logpdf(value, made_nn_mean, agnp.exp(made_nn_std)))
        return z

class BernoulliMADE(MADE):

    def __init__(self, name, input_dimensions, hidden_nodes, observed, nonlinearity = sigmoid, last_layer_nonlinearity = sigmoid):

        super().__init__(name, dimensions = input_dimensions, observed = observed)
        degrees = self.assign_degrees(input_dimensions, hidden_nodes)
        masks, llm = self.create_masks(degrees)
    
        self.made_nn_pre = MaskedNeuralNetwork('nn_base_%s' % (name), observed, [input_dimensions] + hidden_nodes, masks, nonlinearity = nonlinearity, last_layer_nonlinearity = nonlinearity)
        self.made_nn_prob = MaskedNeuralNetwork('nn_mean_%s' % (name), self.made_nn_pre, [hidden_nodes[-1], input_dimensions], [llm], nonlinearity = nonlinearity, last_layer_nonlinearity = last_layer_nonlinearity)
        self.set_dependencies([self.made_nn_prob])

    def log_density(self, value, made_nn_prob):
        z = agnp.sum((value) * agnp.log(made_nn_prob) + (1 - value) * agnp.log(1 - made_nn_prob))
        return z

class ConditionalBernoulliMADE(BernoulliMADE):
    
    def __init__(self, name, input_dimensions, conditional_dims, conditional_input, hidden_nodes, observed, nonlinearity = sigmoid, last_layer_nonlinearity = sigmoid):

        super(BernoulliMADE, self).__init__(name, dimensions = input_dimensions, observed = observed)
        degrees = self.assign_degrees(input_dimensions, hidden_nodes)
        masks, llm = self.create_masks(degrees)
        masks[0] = agnp.vstack((masks[0],agnp.ones((conditional_dims, hidden_nodes[0]))))
        
        self.made_nn_pre = MaskedNeuralNetwork('nn_base_%s' % (name), bl.ops.concat('concat', [observed, conditional_input]), [input_dimensions + conditional_dims] + hidden_nodes, masks, nonlinearity = nonlinearity, last_layer_nonlinearity = nonlinearity)
        self.made_nn_prob = MaskedNeuralNetwork('nn_mean_%s' % (name), self.made_nn_pre, [hidden_nodes[-1], input_dimensions], [llm], nonlinearity = nonlinearity, last_layer_nonlinearity = last_layer_nonlinearity)
        self.set_dependencies([self.made_nn_prob])


class ConditionalGaussianMADE(GaussianMADE):

    is_differentiable = True

    def __init__(self, name, input_dimensions, conditional_dims, conditional_input, hidden_nodes, observed, nonlinearity = sigmoid, last_layer_nonlinearity = linear):

        super(GaussianMADE, self).__init__(name, dimensions = input_dimensions, observed = observed)
        degrees = self.assign_degrees(input_dimensions, hidden_nodes)
        masks, llm = self.create_masks(degrees)
        masks[0] = agnp.vstack((masks[0],agnp.ones((conditional_dims, hidden_nodes[0]))))
        
        self.made_nn_pre = MaskedNeuralNetwork('nn_base_%s' % (name), bl.ops.concat('concat', [observed, conditional_input]), [input_dimensions + conditional_dims] + hidden_nodes, masks, nonlinearity = nonlinearity, last_layer_nonlinearity = nonlinearity)
        self.made_nn_mean = MaskedNeuralNetwork('nn_mean_%s' % (name), self.made_nn_pre, [hidden_nodes[-1], input_dimensions], [llm], nonlinearity = nonlinearity, last_layer_nonlinearity = last_layer_nonlinearity)
        self.made_nn_std = MaskedNeuralNetwork('nn_std_%s' % (name), self.made_nn_pre, [hidden_nodes[-1], input_dimensions], [llm], nonlinearity = nonlinearity, last_layer_nonlinearity = last_layer_nonlinearity)
        self.set_dependencies([self.made_nn_mean, self.made_nn_std])
