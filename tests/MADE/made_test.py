import bayes_lib as bl
import autograd
import autograd.numpy as agnp
import autograd.scipy as agsp
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("../data/MNIST_data/", one_hot = True)

with bl.Model() as m:
    X = bl.Placeholder('X', dimensions = agnp.array([1000,784]))
    Y = bl.Placeholder('Y', dimensions = agnp.array([1000,10]))
    made = bl.ml.made.ConditionalBernoulliMADE('made', 784, 10, Y, [20,10], X, nonlinearity = bl.math.utils.tanh)

    init = agnp.random.normal(0,1,size = m.n_params)
    #init = agnp.zeros(m.n_params)

    def iter_func(t, p, o):
        print("Objective value: %f" % o)
    
    for i in range(110):
        train_images, train_labels = mnist.train.next_batch(500)
        optimizer = bl.math.optimizers.ADAM(learning_rate = 3)
        res = optimizer.run(lambda x: -m.log_density(x, feed_dict = {X: train_images, Y: train_labels}), lambda x: -m.grad_log_density(x, feed_dict = {X: train_images, Y: train_labels}), init, iter_func = iter_func, iter_interval = 5, max_iters = 1000, convergence = 1e-3)
        init = res.position
    agnp.savetxt("map_est_pos.txt", res.position, delimiter = ',')
    #print(made.log_density(agnp.array([[1,2]])))

