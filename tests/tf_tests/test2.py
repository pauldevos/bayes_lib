import bayes_lib as bl
import tensorflow as tf
import seaborn as sns
import pandas as pd
import autograd.numpy as np
import autograd.scipy as agsp
import matplotlib.pyplot as plt

with bl.Model() as model:
    x = bl.rvs.Normal(0., 2., name = 'b')
    
    inf_alg = bl.inference.samplers.MultivariateUniSliceSampler()
    chain, tchain = inf_alg.run(model, np.array(np.random.normal(0, 1)).reshape(1,))
    plt.hist(tchain)
    plt.show()
    
    """
    inf_alg = bl.inference.variational.ReparameterizedVariationalInference(model, init = np.array([0,0,0,0]))
    res, v_dist = inf_alg.run(feed_dict = {X: X_data, y: y_data}, iter_func = iterfunc)
    samples = v_dist.sample(res.position, 1000)
    
    sns.pairplot(pd.DataFrame(samples))
    plt.show()
    
    inf_alg = bl.inference.samplers.A_MVNMetropolisHastings_Adapt()
    chain = inf_alg.run(model, feed_dict = {X: X_data, y: y_data}, n_iters = 10000)
    
    sns.pairplot(pd.DataFrame(chain[5000:,:]))
    plt.show()

    plt.plot(chain[5000:,0])
    plt.plot(chain[5000:,1])
    plt.show()
    """ 
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
