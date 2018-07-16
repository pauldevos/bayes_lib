import bayes_lib as bl
from bayes_lib.apps.markov_jump_process import *
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import time 

np.random.seed(1)
obs = None

with bl.model.Model() as m:
    ir1 = 0.01#bl.rvs.Normal('ir1',np.log(0.01), np.log(1.1), transform = bl.transform.LowerBoundRVTransform(0))
    br = 0.5#bl.rvs.Normal('br', np.log(0.5), np.log(1.1), transform = bl.transform.LowerBoundRVTransform(0))
    dr = 1.0#bl.rvs.Normal('dr', np.log(1.0), np.log(1.1), transform = bl.transform.LowerBoundRVTransform(0))
    ir2 = 0.01#bl.rvs.Normal('ir2', np.log(0.01), np.log(1.1), transform = bl.transform.LowerBoundRVTransform(0))
    y0 = [50, 100]
    obs_process = LotkaVolterra2("lv", y0, ir1, br, dr, ir2, 30)
    obs = obs_process.sim(observation_times = np.linspace(0.2, 30, 15), max_n_steps = 10000, s = True)
    obs[:,1:] = obs[:,1:] + np.random.lognormal(0,np.log(5),size = obs[:,1:].shape)

with bl.model.Model() as m:
    ir1 = bl.rvs.Normal('ir1',np.log(0.01), np.log(1.1), transform = bl.rvs.transform.LowerBoundRVTransform(0))
    br = bl.rvs.Normal('br', np.log(0.5), np.log(1.1), transform = bl.rvs.transform.LowerBoundRVTransform(0))
    dr = bl.rvs.Normal('dr', np.log(1.0), np.log(1.1), transform = bl.rvs.transform.LowerBoundRVTransform(0))
    ir2 = bl.rvs.Normal('ir2', np.log(0.01), np.log(1.1), transform = bl.rvs.transform.LowerBoundRVTransform(0))
    y0 = [50, 100]
    obs_process = LotkaVolterra2("lv", y0, ir1, br, dr, ir2, 30, observed = obs)
    
    n_valid_samples = 0
    total_samples = 1000
    sampled_params = []
    llhoods = np.zeros((total_samples,1))
    while n_valid_samples < total_samples:
        s = m.sample_param_vector(1)[0,:]
        m.set_param_vector(s)
        res = obs_process.log_density()
        if res != -np.inf:
            sampled_params.append(s)
            llhoods[n_valid_samples] = res
            n_valid_samples += 1
    z = np.hstack([sampled_params, llhoods])
    bl.utils.save_chain(z, "results/synth_likelihood.csv")

    #inf_alg = bl.inference.variational.BlackBoxVariationalInference(m)
    #inf_alg.run(n_mc_samples = 2, n_grad_samples = 2)
    #sampler = bl.inference.samplers.M_MVNMetropolisHastingsSpecial(m, scale = 1e-5)
    #chain, tchain = sampler.run(n_iters = 2000, init_params = np.log([0.02,0.3,0.85,0.01]))
    #bl.utils.save_chain(tchain, "results/tchain.csv")
    #bl.utils.save_chain(chain, "results/chain.csv")
