import bayes_lib as bl
import numpy as np
import autograd
import autograd.numpy as agnp
import autograd.scipy as agsp
import matplotlib.pyplot as plt
from scipy.stats import norm
import seaborn as sns

data = np.random.normal(3, 4, size = 100)

with bl.Model() as m:
    mu = bl.rvs.Normal('mu_prior', 0, 1)
    std = bl.rvs.InverseGamma('std_prior', 1, 1)
    y = bl.rvs.Normal('obs_model', mu, std, observed = data)
    # VI Test
    inf_alg = bl.inference.variational.BlackBoxVariationalInference(m, init = np.array([3, np.log(4), 0, 0]))
    res, v_post = inf_alg.run()
    
    inf_alg2 = bl.inference.samplers.A_MVNMetropolisHastings(m, scale = 0.1)
    samples, tsamples = inf_alg2.run(n_iters = 5000)
    print("Variational:", np.array([v_post.v_means[0], np.exp(v_post.v_means[1])]))
    print("MCMC:", np.mean(tsamples[2500:,:], axis = 0))
    #sns.kdeplot(tsamples[2500:,0], label = 'mu')
    #sns.kdeplot(tsamples[2500:,1], label = 'std')
    #plt.show()

    #plt.plot(list(range(2500)), samples[2500:,0], label = 'mu')
    #plt.plot(list(range(2500)), samples[2500:,1], label = 'std')
    #plt.legend()
    #plt.show()

    #s = v_post.sample(1000)
    #sns.kdeplot(s[1000:,0])
    #sns.kdeplot(samples[1000:])
    #plt.plot(np.linspace(2,3, 150), analytical_pdfs)
    #z = np.random.normal((1/(1/100 + 1)) * np.mean(data) + (1/(1/100 + 1)) * 0, np.sqrt(1/(1/1 + 100/1)), size = 1000)
    #sns.kdeplot(s[:,0], label = 'VI')
    #sns.kdeplot(z, label = 'Analytical')
    #sns.kdeplot(tsamples[1000:,0], label = 'MCMC')
    #plt.show()
