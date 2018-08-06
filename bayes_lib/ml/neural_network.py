import bayes_lib as bl
from bayes_lib.math.utils import *
from ..core import *

import numpy as np
import autograd.numpy as agnp
import autograd.scipy as agsp
import autograd
import abc
from autograd.scipy.misc import logsumexp
from autograd.scipy.signal import convolve

class Layer(object):
    
    def __init__(self, input_dim, output_dim):
        self.num_weights = 0
        self.m = input_dim
        self.n = output_dim

    @abc.abstractmethod
    def forward(self, weights, inputs):
        return
    
class FCLayer(Layer):
    
    def __init__(self, input_dim, output_dim, nonlinearity = linear):
        super().__init__(input_dim, output_dim)
        self.nonlinearity = nonlinearity
        self.num_weights = (self.m+1)*self.n
        
        # Xavier Initialization
        # self.params = np.random.normal(0, np.sqrt(2/(self.m + self.n)), size = (1,self.num_weights))
        
    def unpack_params(self,weights):
        num_weight_sets = len(weights)
        return weights[:, :self.m*self.n].reshape((num_weight_sets, self.m, self.n)),\
               weights[:, self.m*self.n:].reshape((num_weight_sets, 1, self.n))

    def forward(self, weights, inputs):
        if len(inputs.shape) > 3:
            inputs = inputs.reshape((inputs.shape[0],inputs.shape[1],-1))
        W, b = self.unpack_params(weights)
        return self.nonlinearity(agnp.einsum('mnd,mdo->mno', inputs, W) + b)

class MaskedFCLayer(FCLayer):
    
    def __init__(self, input_dim, output_dim, mask, nonlinearity = linear):
        super().__init__(input_dim, output_dim, nonlinearity = nonlinearity)
        self.mask = mask[agnp.newaxis,:,:]

    def forward(self, weights, inputs):
        if len(inputs.shape) > 3:
            inputs = inputs.reshape((inputs.shape[0],inputs.shape[1],-1))
        if len(inputs.shape) < 3:
            inputs = inputs[agnp.newaxis,:,:]
        W, b = self.unpack_params(weights)
        return self.nonlinearity(agnp.einsum('mnd,mdo->mno', inputs, (W * self.mask)) + b)


class ConvLayer(Layer):

    def __init__(self, input_dims, kernel_shape, num_filters, nonlinearity = linear):
        depth = input_dims[0]
        y = input_dims[1]
        x = input_dims[2]
        
        self.kernel_shape = kernel_shape
        self.num_filters = num_filters
        self.num_filter_weights = depth * num_filters * kernel_shape[0] * kernel_shape[1]
        self.filter_weights_shape = (depth, self.num_filters, kernel_shape[0], kernel_shape[1])
        self.bias_shape = (1, num_filters, 1, 1)
        self.nonlinearity = nonlinearity
            
        self.output_dim = (self.num_filters,) + self.conv_output_shape(input_dims[1:], self.kernel_shape)
         
        super().__init__(np.prod(input_dims), np.prod(self.output_dim))
        self.num_weights = self.num_filter_weights + num_filters 
        # Xavier Initialization
        self.params = np.random.normal(0, np.sqrt(2/(self.m + self.n)), size = (1,self.num_weights))

    def conv_output_shape(self, A, B):
        return (A[0] - B[0] + 1, A[1] - B[1] + 1)

    def get_output_shape(self):
        return self.output_dim

    def unpack_params(self, weights):
        num_weight_sets = len(weights)
        return weights[:, :self.num_filter_weights].reshape((num_weight_sets,) + self.filter_weights_shape),\
               weights[:,self.num_filter_weights:].reshape((num_weight_sets,) + self.bias_shape)

    def forward(self, weights, inputs):
        # Input dims are [num_weight_sets, 
        w,b = self.unpack_params(weights)
        convs = []
        for i in range(len(w)):
            conv = convolve(inputs[i,:], w[i,:], axes=([2,3],[2,3]), dot_axes = ([1], [0]), mode = 'valid')
            conv = conv + b[i,:]
            convs.append(self.nonlinearity(conv))
        z = agnp.array(convs)
        return z

class RNNLayer(Layer):

    def __init__(self, input_dim, hidden_dim, output_dim, hidden_nonlinearity = linear, output_nonlinearity = linear):
        super().__init__(input_dim, output_dim)
        self.hidden_dim = hidden_dim
        self.num_weights = (self.m + self.hidden_dim + 1) * self.hidden_dim + (self.hidden_dim + 1) * self.n
        self.hidden_nonlinearity = hidden_nonlinearity
        self.output_nonlinearity = output_nonlinearity

        self.hidden_init = agnp.random.randn(self.hidden_dim)
        self.params = np.random.normal(0, np.sqrt(2/(self.m + self.n)), size = (1,self.num_weights))

    def unpack_hidden_params(self, weights):
        W_hidden = weights[:,:(self.m + self.hidden_dim) * self.hidden_dim]
        b_hidden = weights[:,(self.m + self.hidden_dim) * self.hidden_dim: (self.m + self.hidden_dim + 1) * self.hidden_dim]
        return W_hidden.reshape((-1,self.m + self.hidden_dim, self.hidden_dim)), b_hidden.reshape((-1, 1, self.hidden_dim))

    def unpack_output_params(self, weights):
        W_output = weights[:,(self.m + self.hidden_dim + 1) * self.hidden_dim: (self.m + self.hidden_dim + 1) * self.hidden_dim + self.hidden_dim * self.n]
        b_output = weights[:,(self.m + self.hidden_dim + 1) * self.hidden_dim + self.hidden_dim * self.n:]
        return W_output.reshape((-1,self.hidden_dim, self.n)), b_output.reshape((-1, 1, self.n))
    
    def update_hidden(self, weights, input, hidden):
        concated_input = agnp.concatenate((input, hidden),axis = 2)
        W_hidden, b_hidden = self.unpack_hidden_params(weights)
        return self.hidden_nonlinearity(agnp.einsum('pdh,pnd->pnh', W_hidden, concated_input) + b_hidden)

    def get_output(self, weights, hidden):
        W_output, b_output = self.unpack_output_params(weights)
        return self.output_nonlinearity(agnp.einsum('pdh,pnd->pnh', W_output, hidden) + b_output)

    def forward(self, weights, inputs):
        n_param_sets = inputs.shape[0]
        sequence_length = inputs.shape[1]
        n_sequences = inputs.shape[2]
        
        hiddens = agnp.expand_dims(agnp.expand_dims(self.hidden_init, 0).repeat(n_sequences, 0),0).repeat(n_param_sets, 0)
        outputs = [self.get_output(weights, hiddens)]
        for idx in range(sequence_length):
            input = inputs[:,idx,:,:]
            hiddens = self.update_hidden(weights, input, hiddens)
            outputs.append(self.get_output(weights, hiddens))
        
        out = agnp.array(outputs).reshape((inputs.shape[0],inputs.shape[1] + 1, inputs.shape[2], inputs.shape[3]))
        return out

class LSTMLayer(Layer):

    def __init__(self, input_dim, hidden_dim, output_dim, hidden_nonlinearity = linear, output_nonlinearity = linear):
        super().__init__(input_dim, output_dim)
        self.hidden_dim = hidden_dim
        self.num_h_weights = (self.m + self.hidden_dim) * self.hidden_dim
        self.num_b_weights = (self.m + self.hidden_dim + 1) * self.hidden_dim
        self.num_weights = self.num_b_weights * 4 + (self.hidden_dim + 1) * self.n
        self.hidden_nonlinearity = hidden_nonlinearity
        self.output_nonlinearity = output_nonlinearity

        self.hidden_init = agnp.random.randn(self.hidden_dim) * 0.01
        self.cell_init = agnp.random.randn(self.hidden_dim) * 0.01
        self.params = agnp.random.randn(1, self.num_weights)

    def unpack_change_params(self, weights):
        W_change = weights[:,:self.num_h_weights]
        b_change = weights[:,self.num_h_weights:self.num_b_weights]
        return W_change.reshape((-1,self.m + self.hidden_dim, self.hidden_dim)), b_change.reshape((-1, 1, self.hidden_dim))

    def unpack_forget_params(self, weights):
        W_forget = weights[:,self.num_b_weights:self.num_b_weights + self.num_h_weights]
        b_forget = weights[:,self.num_b_weights + self.num_h_weights:self.num_b_weights*2]
        return W_forget.reshape((-1,self.m + self.hidden_dim, self.hidden_dim)), b_forget.reshape((-1, 1, self.hidden_dim))

    def unpack_ingate_params(self, weights):
        W_ingate = weights[:,self.num_b_weights*2:self.num_b_weights*2 + self.num_h_weights]
        b_ingate = weights[:,self.num_b_weights*2 + self.num_h_weights: self.num_b_weights*3]
        return W_ingate.reshape((-1,self.m + self.hidden_dim, self.hidden_dim)), b_ingate.reshape((-1, 1, self.hidden_dim))
    
    def unpack_outgate_params(self, weights):
        W_outgate = weights[:,self.num_b_weights*3:self.num_b_weights*3 + self.num_h_weights]
        b_outgate = weights[:,self.num_b_weights*3 + self.num_h_weights: self.num_b_weights*4]
        return W_outgate.reshape((-1,self.m + self.hidden_dim, self.hidden_dim)), b_outgate.reshape((-1, 1, self.hidden_dim))

    def unpack_output_params(self, weights):
        W_output = weights[:,self.num_b_weights*4:self.num_b_weights*4 + self.hidden_dim * self.n]
        b_output = weights[:,self.num_b_weights*4 + self.hidden_dim * self.n:]
        return W_output.reshape((-1,self.hidden_dim, self.n)), b_output.reshape((-1, 1, self.n))

    def update_hidden(self, weights, input, hidden, cells):
        concated_input = agnp.concatenate((input, hidden),axis = 2)
        W_change, b_change = self.unpack_change_params(weights)
        change = agnp.tanh(agnp.einsum('pdh,pnd->pnh', W_change, concated_input) + b_change)
        W_forget, b_forget = self.unpack_forget_params(weights)
        forget = self.hidden_nonlinearity(agnp.einsum('pdh,pnd->pnh', W_forget, concated_input) + b_forget)
        W_ingate, b_ingate = self.unpack_ingate_params(weights)
        ingate = self.hidden_nonlinearity(agnp.einsum('pdh,pnd->pnh', W_ingate, concated_input) + b_ingate)
        W_outgate, b_outgate = self.unpack_outgate_params(weights)
        outgate = self.hidden_nonlinearity(agnp.einsum('pdh,pnd->pnh', W_outgate, concated_input) + b_outgate)
        cells = cells * forget + ingate * change
        hidden = outgate * agnp.tanh(cells)
        return hidden, cells

    def get_output(self, weights, hidden):
        W_output, b_output = self.unpack_output_params(weights)
        return self.output_nonlinearity(agnp.einsum('pdh,pnd->pnh', W_output, hidden) + b_output)

    def forward(self, weights, inputs):
        n_param_sets = inputs.shape[0]
        sequence_length = inputs.shape[1]
        n_sequences = inputs.shape[2]
        
        hiddens = agnp.expand_dims(agnp.expand_dims(self.hidden_init, 0).repeat(n_sequences, 0),0).repeat(n_param_sets, 0)
        cells = agnp.expand_dims(agnp.expand_dims(self.cell_init, 0).repeat(n_sequences, 0),0).repeat(n_param_sets, 0)
        
        outputs = [self.get_output(weights, hiddens)]
        for idx in range(sequence_length):
            input = inputs[:,idx,:,:]
            hiddens, cells = self.update_hidden(weights, input, hiddens, cells)
            outputs.append(self.get_output(weights, hiddens))
        
        out = agnp.array(outputs).reshape((inputs.shape[0],inputs.shape[1] + 1, inputs.shape[2], inputs.shape[3]))
        return out

class BaseNeuralNetwork(bl.ops.Operation):

    layers = None
    num_weights = None

    def __init__(self, name, weights, inputs):
        self.weights = weights
        self.inputs = inputs
        super().__init__(name, [weights,inputs])
    
    def unpack_layers(self, weights):
        num_weight_sets = len(weights)
        for layer in self.layers:
            yield weights[:, :layer.num_weights]
            weights = weights[:, layer.num_weights:]

    def compute(self, weights, inputs):
        weights = weights.reshape(1,-1)
        #t = len(weights)
        if len(inputs.shape) < 3:
            inputs = agnp.expand_dims(inputs, 0).repeat(1,0)
        for i, w in enumerate(self.unpack_layers(weights)):
            inputs = self.layers[i].forward(w, inputs)
        return inputs

    def add_layer(self, layer):
        if self.layers is None:
            self.layers = []
            self.num_weights = 0
        self.layers.append(layer)
        self.num_weights += layer.num_weights

class DenseNeuralNetwork(BaseNeuralNetwork):

    is_differentiable = True

    def  __init__(self, name, inputs, layer_dims, nonlinearity = sigmoid, last_layer_nonlinearity = linear):
        shapes = list(zip(layer_dims[:-1], layer_dims[1:]))
        for m, n in shapes[:len(shapes)-1]:
            self.add_layer(FCLayer(m, n, nonlinearity))
        self.add_layer(FCLayer(shapes[-1][0], shapes[-1][1], nonlinearity = last_layer_nonlinearity))
        weights = bl.rvs.Normal('rv_%s' % (name), 0, 1, dimensions = agnp.array(self.num_weights))
        #weights = bl.rvs.Variable('rv_%s' % (name), dimensions = agnp.array(self.num_weights))
        super().__init__(name, weights, inputs)


class MaskedNeuralNetwork(BaseNeuralNetwork):
    
    is_differentiable = True

    def  __init__(self, name, inputs, layer_dims, masks, nonlinearity = sigmoid, last_layer_nonlinearity = linear):
        shapes = list(zip(layer_dims[:-1], layer_dims[1:]))
        for i, dims in enumerate(shapes[:len(shapes)-1]):
            m,n = dims
            self.add_layer(MaskedFCLayer(m, n, masks[i], nonlinearity))
        self.add_layer(MaskedFCLayer(shapes[-1][0], shapes[-1][1], masks[-1], nonlinearity = last_layer_nonlinearity))
        #weights = bl.rvs.Normal('rv_%s' % (name), 0, 1, dimensions = agnp.array(self.num_weights))
        weights = bl.rvs.Variable('rv_%s' % (name), dimensions = agnp.array(self.num_weights))
        super().__init__(name, weights, inputs)


