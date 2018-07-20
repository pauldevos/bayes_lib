import bayes_lib as bl
import autograd
import autograd.numpy as agnp
import autograd.scipy as agsp

import matplotlib.pyplot as plt

data = agnp.random.normal(10,1,size = 100)
with bl.Model() as m:
    mu = bl.rvs.Normal('mu_prior', 0, 1)
    #std = bl.rvs.Normal('std_prior', 3, 1, transform = bl.rvs.transform.LowerBoundRVTransform(0.))
    y = bl.rvs.Normal('obs', mu, 1., observed = data)
    
    """
    inf_alg = bl.inference.samplers.A_MVNMetropolisHastings(m)
    chain, tchain = inf_alg.run(n_iters = 2000)
    
    plt.hist(chain[1000:,1], label = 'Unconstrained')
    plt.hist(tchain[1000:,1], label = 'Constrained')
    plt.legend()
    plt.show()
    """
    inf_alg = bl.inference.variational.ReparameterizedVariationalInference(m, variational_dist = bl.inference.variational.distributions.PlanarNormalizingFlow(5))

    def iter_func(t, cur_pos, cur_obj):
        print("Current Objective: %f)" % cur_obj)
    
    res, v_dist = inf_alg.run(iter_func = iter_func, max_opt_iters = 3000)
    samples = v_dist.sample(3000)
    agnp.savetxt("post.csv", samples, delimiter = ',')
    print(res)

    fig, axes = plt.subplots(1, 6)
    for i in range(6):
        axes[i].hist(samples[:,i])
    plt.show()



    
