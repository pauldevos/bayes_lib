import bayes_lib as bl
import numpy as np
from scipy.stats import norm
from scipy.integrate import odeint

import matplotlib.pyplot as plt
import seaborn as sns

data = np.random.normal(3, 1, size = 100)

with bl.Model() as m:
    mu = bl.rvs.Normal('mu_prior', 0, 1)
    #mu = bl.rvs.Uniform('mu_prior', 1, 4)
    #std = bl.rvs.Normal('std_prior', 0, 3, transform = bl.transform.LowerBoundRVTransform(0))
    y = bl.rvs.Normal('obs_model', mu, 1, observed = data)

    chain, tchain = bl.sampling.metropolis_hastings(m, n_iter = 10000)
    plt.hist(tchain[5000:,0])
    #plt.hist(tchain[50000:,1])
    #plt.scatter(tchain[50000:,0], tchain[50000:,1])
    plt.show()
    
    #i_dist = lambda : [np.random.normal(2, 5)]
    #i_dist_density = lambda x : norm.pdf(x, 2, 5)
    #samples = bl.sampling.importance_sampling(m, i_dist, i_dist_density, n_i_samples = 10000)
    #plt.hist(samples, bins = 'auto')
    #plt.show()
