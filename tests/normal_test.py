import bayes_lib as bl
import autograd
import autograd.numpy as agnp
import autograd.scipy as agsp

import matplotlib.pyplot as plt

data = agnp.random.normal(3,1,size = 100)
with bl.Model() as m:
    mu = bl.rvs.Normal('mu_prior', 0, 1)
    std = bl.rvs.Normal('std_prior', 3, 1, transform = bl.rvs.transform.LowerBoundRVTransform(0.))
    y = bl.rvs.Normal('obs', mu, std, observed = data)
    
    """
    inf_alg = bl.inference.samplers.A_MVNMetropolisHastings(m)
    chain, tchain = inf_alg.run(n_iters = 2000)
    
    plt.hist(chain[1000:,1], label = 'Unconstrained')
    plt.hist(tchain[1000:,1], label = 'Constrained')
    plt.legend()
    plt.show()
    """
    inf_alg = bl.inference.variational.BlackBoxVariationalInference(m)

    def iter_func(t, cur_pos, cur_obj):
        print("Current Objective: %f)" % cur_obj)
    res, v_dist = inf_alg.run(iter_func = iter_func)
    print(res)



    
