import bayes_lib as bl
import numpy as np
from scipy.stats import norm
from scipy.integrate import odeint

import time

import matplotlib.pyplot as plt
import seaborn as sns
import autograd
import autograd.numpy as agnp
import autograd.scipy as agsp

data = np.random.normal(3, 1, size = 100)

def special_log_density(x):
    mu = x[:,0]
    lps = []
    for i in range(mu.shape[0]):
        lps.append(agnp.sum(agsp.stats.norm.logpdf(data, mu[i], 1), axis = 0))
    return agnp.array(lps)

grad_special_log_density = autograd.elementwise_grad(special_log_density)

params = np.random.normal(3, 1, size = 100000).reshape(-1,1)

with bl.Model() as m:
    mu = bl.rvs.Normal('mu_prior', 0, 1)
    #mu = bl.rvs.Uniform('mu_prior', 1, 4)
    #std = bl.rvs.Normal('std_prior', 0, 3, transform = bl.transform.LowerBoundRVTransform(0))
    y = bl.rvs.Normal('obs_model', mu, 1, observed = data)
    st = time.time()
    g = grad_special_log_density(params)
    print("All at once %d" % (time.time() - st))
    m.grad_log_density_p(params[:4,:])
    st = time.time()
    m.grad_log_density_p(params)
    print("One at a time %d" % (time.time() - st))
    #sampler = bl.inference.samplers.M_MVNMetropolisHastings(m)
    
    """
    sampler = bl.sampling.M_MVNMetropolisHastings(m)
    sampler2 = bl.sampling.A_MVNMetropolisHastings(m)
    chain, tchain = sampler.run(n_iters = 2000)
    chain2, tchain2 = sampler2.run(n_iters = 2000)
    #plt.hist(chain[1000:,0], bins = 'auto')
    #plt.hist(tchain[50000:,1])
    #plt.scatter(tchain[50000:,0], tchain[50000:,1])
    plt.plot(list(range(1000)), tchain[1000:,0], color = 'r', label = 'multiplicative')
    plt.plot(list(range(1000)), tchain2[1000:,0], color = 'b', label = 'additive')
    plt.legend()
    plt.show()
    
    #i_dist = lambda : [np.random.normal(2, 5)]
    #i_dist_density = lambda x : norm.pdf(x, 2, 5)
    #samples = bl.sampling.importance_sampling(m, i_dist, i_dist_density, n_i_samples = 10000)
    #plt.hist(samples, bins = 'auto')
    #plt.show()
    """
