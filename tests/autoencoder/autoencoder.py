import bayes_lib as bl
import autograd
import autograd.numpy as agnp
import autograd.scipy as agsp
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

#x_true = agnp.random.normal(3, 1, size = (100, 10))
mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)
train_images = mnist.train.next_batch(100)[0]

with bl.Model() as m:
    X = bl.Placeholder('X', dimensions = agnp.array([100,784]))
    encoder = bl.ml.neural_network.DenseNeuralNetwork('Encoder', X, layer_dims = [784,50,20], nonlinearity = bl.math.utils.sigmoid)
    decoder = bl.ml.neural_network.DenseNeuralNetwork('Decoder', encoder, layer_dims = [20, 50, 784], nonlinearity = bl.math.utils.sigmoid, last_layer_nonlinearity = bl.math.utils.sigmoid)
    y = bl.rvs.Bernoulli('obs', decoder, observed = X)
    
    init = agnp.random.normal(0,3,size = m.n_params)
    optimizer = bl.math.optimizers.ADAM(learning_rate = 5e-1)
    
    def iter_func(t, p, o):
        print("Objective value: %f" % o)

    res = optimizer.run(lambda x: -m.log_density(x, feed_dict = {X: train_images}), lambda x: -m.grad_log_density(x, feed_dict = {X: train_images}), init, iter_func = iter_func, iter_interval = 5)
    agnp.savetxt("pos.txt", res.position, delimiter = ',')
