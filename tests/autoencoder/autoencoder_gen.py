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

    fit_params = agnp.loadtxt("pos.txt", delimiter = ',')
    m.set_param(fit_params)
    out = m.evaluate(decoder, feed_dict = {X: train_images})

    for i in range(out.shape[1]):
        out[0,i,:] = out[0,i,:] > 0.5

    fig, ax = plt.subplots(10,10)

    for i in range(10):
        for j in range(10):
            ax[i,j].imshow(out[0,i * 10 + j,:].reshape(28,28))
            ax[i,j].axis('off')

    plt.show()

