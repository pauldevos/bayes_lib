import bayes_lib as bl
import tensorflow as tf
import seaborn as sns
import pandas as pd
import autograd.numpy as np
import autograd.scipy as agsp
import matplotlib.pyplot as plt

X_data = np.linspace(-4, 4, 100)
y_data = 3 * X_data + 2 + np.random.normal(0,1,size = 100)

with bl.Model() as model:
    X = tf.placeholder(tf.float32, shape = 100)
    y = tf.placeholder(tf.float32, shape = 100)
    
    m = bl.rvs.Normal(0., 1., name = 'm_prior')
    b = bl.rvs.Normal(0., 1., name = 'b_prior')
    obs = bl.rvs.Normal(m * X + b, 1., observed = y, name = 'obs')
    
    def iterfunc(t, x, obj):
        print("%d : %f" % (t, obj))
    """
    inf_alg = bl.inference.samplers.MultivariateUniSliceSampler()
    chain, tchain = inf_alg.run(model, np.random.normal(0, 1, size = 2), feed_dict = {X: X_data, y: y_data}, n_iters = 3000)

    sns.pairplot(pd.DataFrame(tchain[2000:,:]))
    plt.show()
    
    """
    #inf_alg = bl.inference.variational.ReparameterizedVariationalInference(model, init = np.array([0,0,1,0,0,1]))
    inf_alg = bl.inference.variational.ReparameterizedVariationalInference(model, init = np.array([0,0,0,0]))

    res, v_dist = inf_alg.run(feed_dict = {X: X_data, y: y_data}, iter_func = iterfunc)
    samples = v_dist.sample(res.position, 1000)
    
    sns.pairplot(pd.DataFrame(samples))
    plt.show()
    
    """
    train_op = tf.train.AdamOptimizer(1).minimize(-model.get_log_density_op())
    model.sess.run(tf.global_variables_initializer(), feed_dict = {X: X_data, y: y_data})
    
    for i in range(100):
        model.sess.run(train_op, feed_dict = {X: X_data, y: y_data})
    m_res,b_res = model.sess.run([m,b])
    
    plt.plot(X_data, y_data)
    plt.plot(X_data, m_res * X_data + b_res)
    plt.show()

    #print(model.log_density(np.array([2.,2.], dtype = np.float32), feed_dict = {X: X_data, y: y_data}))
    """
