import bayes_lib as bl
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

mnist = input_data.read_data_sets("../data/MNIST_data/", one_hot = True)
def binarize(images):
    return (images > 0.5).astype(np.float32)

with bl.Model() as model:
    X = tf.placeholder(tf.float32, shape = [None, 784])
    y = tf.placeholder(tf.float32, shape = [None, 10])
    #made = bl.ml.made.ConditionalBernoulliMADE(X, y, 784, 10, [20, 10])
    made = bl.ml.made.BernoulliMADE(X, 784, [50,100,50])
    
    ld_op = -model.get_log_density_op()
    train_op = tf.train.AdamOptimizer(0.01).minimize(ld_op)

    model.sess.run(tf.global_variables_initializer())

    for i in range(20000):
        train_images, labels = mnist.train.next_batch(128)
        model.sess.run(train_op, feed_dict = {X: binarize(train_images), y: labels})

        if i % 1000 == 0:
            print(model.sess.run(ld_op, feed_dict = {X: binarize(train_images), y: labels}))
    
    #pre = made.made_nn_pre.compute(tf.concat([X,y],1))
    pre = made.made_nn_pre.compute(X)
    prob_op = made.made_nn_prob.compute(pre)
    y_test = np.tile(np.array([0,0,0,0,0,0,0,0,0,1]), (10,1))
    probs = model.sess.run(prob_op, feed_dict = {X: np.zeros((10,784)), y: y_test})
    
    sample = np.random.rand(10,784) > probs

    fig, ax = plt.subplots(2,5)
    for i in range(2):
        for j in range(5):
            ax[i,j].imshow(sample[i * 2 + j,:].reshape((28,28)))

    plt.show()

    
