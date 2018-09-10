import bayes_lib as bl
import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

lv_process = bl.apps.markov_jump_process.LotkaVolterra2([50,100], 0.01, 0.5, 1, 0.01, max_n_steps = 1000)

observation_times = np.arange(0, 15, 0.2)
def sample(p1, p2, p3, p4, termination_time = 30, observation_times = observation_times):
    lv_process.set_params([p1,p2,p3,p4])
    return lv_process.sample(observation_times = observation_times, termination_time = termination_time)[1:,:]

N_obs = observation_times.shape[0]
with bl.Model() as model:
    X = tf.placeholder(tf.float32, shape = [None, N_obs * 2])

    made = bl.ml.made.GaussianMADE(X, N_obs * 2, [50, 100, 50])

    ld_op = -model.get_log_density_op()
    train_op = tf.train.AdamOptimizer(0.1).minimize(ld_op)

    model.sess.run(tf.global_variables_initializer())
    
    running_train_data = []
    for i in range(2000):
        train_data = np.zeros([10, N_obs * 2])
        for j in range(10):
            y_samp = sample(0.01, 0.5, 1, 0.01)
            train_data[j,:] = np.hstack([y_samp[:,1], y_samp[:,2]])
        running_train_data.append(train_data)
        model.sess.run(train_op, feed_dict = {X: np.vstack(running_train_data)})
        
        if i % 100 == 0:
            print(i)
            print(model.sess.run(ld_op, feed_dict = {X: train_data}))

y = np.zeros([1, N_obs * 2])
pre = made.made_nn_pre.compute(X)
mean_op = made.made_nn_mean.compute(pre)
std_op = tf.exp(made.made_nn_std.compute(pre))
means, stds = model.sess.run([mean_op, std_op], feed_dict = {X: np.zeros((1, N_obs * 2))})
samples = np.random.normal(means, stds)

plt.plot(observation_times, samples[0,:N_obs], label = 'predator')
plt.plot(observation_times, samples[0,N_obs:], label = 'prey')
plt.legend()
plt.show()


