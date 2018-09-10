import bayes_lib as bl
import tensorflow as tf

import pandas as pd

import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import seaborn as sns

def sample(theta1, theta2, theta3, theta4, theta5, N):

    s1 = theta3**2
    s2 = theta4**2

    rho = np.tanh(theta5)

    return np.random.multivariate_normal(np.array([theta1,theta2]), np.array([[s1, rho * s1 * s2],[rho * s1 * s2, s2]]), size = N)

def true_lpdf(theta1, theta2, theta3, theta4, theta5, X):
    
    s1 = theta3**2
    s2 = theta4**2

    rho = np.tanh(theta5)

    return np.sum(multivariate_normal.logpdf(X, np.array([theta1,theta2]), np.array([[s1, rho * s1 * s2],[rho * s1 * s2, s2]])))

tp = np.array([0.7, -2.9, -1, -0.9, 0.6, 4])
x = sample(0.7, -2.9, -1, -0.9, 0.6, 4)

with bl.Model() as model_post:
    X = tf.placeholder(tf.float32, shape = [None, 2])

    theta1 = bl.rvs.Uniform(-3,3)
    theta2 = bl.rvs.Uniform(-3,3)
    theta3 = bl.rvs.Uniform(-3,3)
    theta4 = bl.rvs.Uniform(-3,3)
    theta5 = bl.rvs.Uniform(-3,3)

    rho = tf.tanh(theta5)
    corr = rho * (theta3**2) * (theta4**2)

    mu = tf.stack([theta1, theta2])
    cov = [[theta3**4, corr],[corr, theta4**4]]

    y = bl.rvs.Multivariate_Normal(mu, cov, observed = X)

    #inf_alg = bl.inference.samplers.A_MVNMetropolisHastings(scale = 0.1)
    inf_alg = bl.inference.samplers.A_MVNMetropolisHastings_Adapt()
    samples = []
    for i in range(4):
        chain,tchain = inf_alg.run(model_post, feed_dict = {X: x}, n_iters = 3000)
        samples.append(tchain[2000:,:])
    
    sns.pairplot(pd.DataFrame(np.vstack(samples)))
    plt.show()
 
    """    
    def iterfunc(t, x, obj):
        print("%d : %f" % (t, obj))

    inf_alg = bl.inference.variational.ReparameterizedVariationalInference(model_post, init = np.random.normal(0,1,size = 10))
    
    res, v_dist = inf_alg.run(feed_dict = {X: x}, iter_func = iterfunc)
    samples = v_dist.sample(res.position, 1000)
    
    sns.pairplot(pd.DataFrame(samples))
    plt.show()
    """

#with bl.Model() as model:
#    X = tf.placeholder(tf.float32, shape = [None, 2])
"""
made = bl.ml.made.GaussianMADE(X, 2, [50, 100, 50])

ld_op = -model.get_log_density_op()
train_op = tf.train.AdamOptimizer(0.01).minimize(ld_op)

model.sess.run(tf.global_variables_initializer())

running_train_data = []
for i in range(200):
    train_data = sample(0.7, -2.9, -1, -0.9, 0.6, 10)
    running_train_data.append(train_data)
    #model.sess.run(train_op, feed_dict = {X: np.vstack(running_train_data)})
    model.sess.run(train_op, feed_dict = {X: train_data})
    if i % 100 == 0:
        print(i)
        print(model.sess.run(-ld_op, feed_dict = {X: x}))

print("Final %f:" % model.sess.run(-ld_op, feed_dict = {X: x}))
print("True LPDF %f:" % true_lpdf(0.7, -2.9, -1, -0.9, 0.6, x))

y = np.zeros([1, 2])
pre = made.made_nn_pre.compute(X)
mean_op = made.made_nn_mean.compute(pre)
std_op = tf.exp(made.made_nn_std.compute(pre))
means, stds = model.sess.run([mean_op, std_op], feed_dict = {X: np.zeros((1, 2))})
samples = np.random.normal(means, stds)
print(samples.shape)
plt.plot(samples[:,0], samples[:,1])
plt.show()
"""
