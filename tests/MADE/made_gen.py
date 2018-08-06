import bayes_lib as bl
import autograd
import autograd.numpy as agnp
import autograd.scipy as agsp
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

#mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)

with bl.Model() as m:
    X = bl.Placeholder('X', dimensions = agnp.array([100,784]))
    Y = bl.Placeholder('Y', dimensions = agnp.array([100,10]))
    made = bl.ml.made.ConditionalBernoulliMADE('made', 784, 10, Y, [20,10], X, nonlinearity = bl.math.utils.tanh)
    
    params = agnp.loadtxt("map_est_pos.txt", delimiter = ',')
    m.set_param(params)
    
    gen = agnp.zeros((100,784))
    label = agnp.tile(agnp.array([0,0,0,0,0,0,0,0,0,1]), (100,1))
    print(label.shape)
    res = m.evaluate(made.made_nn_prob, feed_dict = {X: gen, Y: label})
    ims = agnp.random.rand(100,784) < res[0,:,:]
    fig, ax = plt.subplots(10,10)
    for i in range(10):
        for j in range(10):
            ax[i,j].imshow(ims[10 * i + j,:].reshape(28,28))
            ax[i,j].axis('off')
    plt.show()

