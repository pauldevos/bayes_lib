import bayes_lib as bl
from bayes_lib.math.utils import *
from ..core import *

import autograd.numpy as agnp
import autograd.scipy as agsp
import autograd

class MADE(bl.rvs.RandomVariable):

    is_differentiable = True

    def __init__(self, name, layer_dims, dimensions = 1, transform = None, observed = None, nonlinearity = sigmoid):
        super().__init__(name, dimensions = dimensions, transform = transform, observed = observed)
        self.layer_dims = layer_dims
        self.degrees = self.assign_degrees()
        self.masks = self.create_masks(self.degrees)
        self.weights, self.biases = self.create_weights()
        self.set_dependencies([self.weights, self.biases])

    def reconstruct_weights_and_parameters(self, weights, parameters):
        Ws = []
        bs = []
        widx = 0
        bidx = 0
        for l, (N0, N1) in enumerate(zip(self.layer_dims[:-1], self.layer_dims[1:])):
            Ws.append(weights[widx:widx + N0 * N1].reshape(N0, N1))
            bs.append(biases[bidx:bidx + N1])
        return Ws, bs

    def create_weights(self):
        n_W_weights = 0
        n_b_weights = 0
        for l, (N0, N1) in enumerate(zip(self.layer_dims[:-1], self.layer_dims[1:])):
            n_W_weights += N0 * N1
            n_b_weights += N1
        weights = bl.rvs.Variable('weights', dimensions = agnp.array([n_W_weights]))
        biases = bl.rvs.Variable('biases', dimensions = agnp.array([n_b_weights]))
        return weights, biases

    def assign_degrees(self):
        degrees = []
        for l in range(1,len(self.layer_dims) - 1):
            degrees.append(agnp.arange(self.layer_dims[l]) % max(1, self.layer_dims[0] - 1) + min(1, self.layer_dims[0] - 1))
        return degrees
    
    def create_masks(self, degrees):
        Ms = []

        for l, (d0, d1) in enumerate(zip(degrees[:-1], degrees[1:])):
            M = d0[:, agnp.newaxis] <= d1
            Ms.append(M)

        Mmp = degrees[-1][:, agnp.newaxis] < degrees[0]
        return Ms, Mmp

    def log_density(self, value, *args):
        return 0
