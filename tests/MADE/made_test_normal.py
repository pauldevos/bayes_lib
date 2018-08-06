import bayes_lib as bl
import autograd
import autograd.numpy as agnp
import autograd.scipy as agsp
import seaborn as sns
import matplotlib.pyplot as plt

def gen_data(n):
    logpdfs = []
    #theta = agnp.random.multivariate_normal(agnp.array([3,3]), agnp.eye(2), size = n)
    theta = agnp.random.uniform(0, 5, size=(n,2))
    data = []
    for i in range(n):
        y = agnp.random.multivariate_normal(theta[i,:], agnp.eye(2))
        logpdfs.append(agsp.stats.multivariate_normal.logpdf(y, theta[i,:], agnp.eye(2)))
        data.append(y)
    return logpdfs, theta, agnp.array(data)

#data = agnp.hstack([agnp.random.normal(3, 2, size = (1000,1)),agnp.random.normal(10, .1, size = (1000,1)), agnp.random.normal(-5, 3, size = (1000,1))])
with bl.Model() as m_surrogate_density:
    #X = bl.Placeholder('X', dimensions = agnp.array([1000,3]))
    X = bl.Placeholder('X', dimensions = agnp.array([100, 2]))
    theta = bl.Placeholder('theta', dimensions = agnp.array([100,2]))
    made = bl.ml.made.ConditionalGaussianMADE('made', 2, 2, theta, [20,21], X, nonlinearity = bl.math.utils.relu)

    init = agnp.random.normal(-3,3,size = m_surrogate_density.n_params)

    def iter_func(t, p, o):
        print("Objective value: %f" % o)

    for i in range(10):
        _, pseudo_theta, pseudo_data = gen_data(100)
        optimizer = bl.math.optimizers.ADAM(learning_rate = 1)
        res = optimizer.run(lambda x: -m_surrogate_density.log_density(x, feed_dict = {X: pseudo_data, theta: pseudo_theta}), lambda x: -m_surrogate_density.grad_log_density(x, feed_dict = {X: pseudo_data, theta: pseudo_theta}), init, iter_func = iter_func, iter_interval = 5, max_iters = 1000)
        init = res.position

    Z = []
    lpdf = []
    true_lpdf = []
    xv, yv = agnp.meshgrid(agnp.linspace(0,6,100),agnp.linspace(0,6,100))
    for i in range(100):
        for j in range(100):
            Z.append([xv[i,j], yv[i,j]])
            x_d = agnp.array([xv[i,j], yv[i,j]])
            t_m = agnp.array([2,2])
            lpdf.append(m_surrogate_density.log_density(init, feed_dict = {X: x_d.reshape(1,-1), theta: t_m.reshape(1,-1)}))
            true_lpdf.append(agsp.stats.multivariate_normal.logpdf(x_d, t_m, agnp.eye(2)))
    
    fig, ax = plt.subplots(1,2)
    ax[0].contour(agnp.array(lpdf).reshape(100,100))
    ax[1].contour(agnp.array(true_lpdf).reshape(100,100))
    plt.show()

    """
    fig, ax = plt.subplots(1,2)
    sns.kdeplot(obs[:,0], ax = ax[0], label = 'est')
    sns.kdeplot(agnp.random.normal(3,1, size = 1000), ax = ax[0], label = 'true')
    ax[0].legend()
    #ax[0].hist(obs[:,0])
    #ax[0].hist(agnp.random.normal(3,1, size = 1000))
    sns.kdeplot(obs[:,1], ax = ax[1], label = 'est')
    sns.kdeplot(agnp.random.normal(3,1, size = 1000), ax = ax[1], label = 'true')
    ax[1].legend()
    #ax[1].hist(obs[:,1])
    #ax[1].hist(agnp.random.normal(3,1, size = 1000))
    plt.show()
    """        

"""
with bl.Model() as m_fit:

    X = bl.Placeholder('X', dimension = agnp.array([100,2]))
    mu_prior = bl.rvs.Multivariate_Normal('mu_prior', agnp.array([0,0]), agnp.eye(2))
    y = bl.ml.made.SingleLayerMADE('made', 2, 5, X, nonlinearity = bl.math.utils.relu)
""" 
