import bayes_lib as bl
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

#x_true = agnp.random.normal(3, 1, size = (100, 10))
mnist = input_data.read_data_sets("../data/MNIST_data/", one_hot = True)
batch_size = 256 

train_images = mnist.train.next_batch(batch_size)[0]
test_images = mnist.test.next_batch(batch_size)[0]

with bl.Model() as model:
    X = tf.placeholder(tf.float32, shape = [None,784])
    encoder = bl.ml.neural_network.DenseNeuralNetwork(layer_dims = [784, 256, 128], last_layer_nonlinearity = tf.nn.sigmoid)
    decoder = bl.ml.neural_network.DenseNeuralNetwork(layer_dims = [128, 256, 784], last_layer_nonlinearity = tf.nn.sigmoid)
    Y = bl.rvs.Normal(decoder.compute(encoder.compute(X)), 1., observed = X)
    
    ld_op = -model.get_log_density_op()
    train_op = tf.train.RMSPropOptimizer(0.01).minimize(ld_op)

    model.sess.run(tf.global_variables_initializer())

    for i in range(30000):
        train_images = mnist.train.next_batch(batch_size)[0]
        model.sess.run(train_op, feed_dict = {X: train_images})

        if i % 1000 == 0:
            print(model.sess.run(ld_op, feed_dict = {X: train_images}))

    test_images = mnist.test.next_batch(batch_size)[0]
    out = model.sess.run(decoder.compute(encoder.compute(X)), feed_dict = {X:test_images})

    fig, ax = plt.subplots(10,10)
    for i in range(10):
        for j in range(10):
            ax[i,j].imshow(out[i * 10 + j,:].reshape(28,28))
            ax[i,j].axis('off')

    plt.show()

    fig, ax = plt.subplots(10,10)
    for i in range(10):
        for j in range(10):
            ax[i,j].imshow(test_images[i * 10 + j,:].reshape(28,28))
            ax[i,j].axis('off')

    plt.show()

    #y = bl.rvs.Bernoulli('obs', decoder, observed = X)
    
    #init = agnp.random.normal(0,3,size = m.n_params)
    #optimizer = bl.math.optimizers.ADAM(learning_rate = 5e-1)
    
    #def iter_func(t, p, o):
    #    print("Objective value: %f" % o)

    #res = optimizer.run(lambda x: -m.log_density(x, feed_dict = {X: train_images}), lambda x: -m.grad_log_density(x, feed_dict = {X: train_images}), init, iter_func = iter_func, iter_interval = 5)
    #agnp.savetxt("pos.txt", res.position, delimiter = ',')
