import abc
import bayes_lib as bl
import numpy as np

from ..ml.neural_network import *

class MADE(bl.rvs.RandomVariable):

    is_differentiable = True

    def __init__(self, observed, *args, **kwargs):
        super().__init__(observed, default_value = 0., transform = None, *args, **kwargs)

    def assign_degrees(self, input_dims, hidden_nodes):
        degrees = []
        degrees.append(np.arange(1, input_dims + 1))
        for N in hidden_nodes:
            degrees_l = np.arange(N) % max(1, input_dims - 1) + min(1, input_dims - 1)
            degrees.append(degrees_l)
        return degrees

    def create_masks(self, degrees, mode = 'sequential'):

        masks = []
        for i, (d0, d1) in enumerate(zip(degrees[:-1], degrees[1:])):
            M = d0[:, np.newaxis] <= d1
            masks.append(M)

        Mmp = degrees[-1][:, np.newaxis] < degrees[0]
        return masks, Mmp
    
class GaussianMADE(MADE):

    def __init__(self, observed, input_dims, hidden_nodes, nonlinearity = tf.nn.sigmoid, last_layer_nonlinearity = bl.math.utils.linear, *args, **kwargs):

        super().__init__(observed, *args, **kwargs)
        degrees = self.assign_degrees(input_dims, hidden_nodes)
        masks, llm = self.create_masks(degrees)
    
        self.made_nn_pre = MaskedNeuralNetwork([input_dims] + hidden_nodes, masks, nonlinearity = nonlinearity, last_layer_nonlinearity = nonlinearity)
        self.made_nn_mean = MaskedNeuralNetwork([hidden_nodes[-1], input_dims], [llm], nonlinearity = nonlinearity, last_layer_nonlinearity = last_layer_nonlinearity)
        self.made_nn_std = MaskedNeuralNetwork([hidden_nodes[-1], input_dims], [llm], nonlinearity = nonlinearity, last_layer_nonlinearity = last_layer_nonlinearity)
    
    def log_density(self):
        pre = self.made_nn_pre.compute(self.value())
        mean = self.made_nn_mean.compute(pre)
        std = tf.exp(self.made_nn_std.compute(pre))
        return tf.reduce_sum(tf.distributions.Normal(mean, std).log_prob(self.value()))

class BernoulliMADE(MADE):

    def __init__(self, observed, input_dims, hidden_nodes, nonlinearity = tf.nn.sigmoid, last_layer_nonlinearity = tf.nn.sigmoid, *args, **kwargs):

        super().__init__(observed, *args, **kwargs)
        degrees = self.assign_degrees(input_dims, hidden_nodes)
        masks, llm = self.create_masks(degrees)
    
        self.made_nn_pre = MaskedNeuralNetwork([input_dims] + hidden_nodes, masks, nonlinearity = nonlinearity, last_layer_nonlinearity = nonlinearity)
        self.made_nn_prob = MaskedNeuralNetwork([hidden_nodes[-1], input_dims], [llm], nonlinearity = nonlinearity, last_layer_nonlinearity = last_layer_nonlinearity)
    
    def log_density(self):
        pre = self.made_nn_pre.compute(self.value())
        prob = self.made_nn_prob.compute(pre)
        return tf.reduce_sum(self.value() * tf.log(prob) + (1 - self.value()) * tf.log(1 - prob))

class ConditionalBernoulliMADE(MADE):
    
    def __init__(self, observed, conditional_input, input_dims, conditional_dims, hidden_nodes, nonlinearity = tf.nn.relu, last_layer_nonlinearity = tf.nn.sigmoid, *args, **kwargs):

        super().__init__(observed, *args, **kwargs)
        degrees = self.assign_degrees(input_dims, hidden_nodes)
        masks, llm = self.create_masks(degrees)
        masks[0] = np.vstack((masks[0],np.ones((conditional_dims, hidden_nodes[0]))))
        self.conditional_input = conditional_input
    
        self.made_nn_pre = MaskedNeuralNetwork([input_dims + conditional_dims] + hidden_nodes, masks, nonlinearity = nonlinearity, last_layer_nonlinearity = nonlinearity)
        self.made_nn_prob = MaskedNeuralNetwork([hidden_nodes[-1], input_dims], [llm], nonlinearity = nonlinearity, last_layer_nonlinearity = last_layer_nonlinearity)

    def log_density(self):
        pre = self.made_nn_pre.compute(tf.concat([self.value(), self.conditional_input],1))
        prob = self.made_nn_prob.compute(pre)
        return tf.reduce_sum(self.value() * tf.log(prob) + (1 - self.value()) * tf.log(1 - prob))

"""
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
"""
