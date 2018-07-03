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
    obs = obs_process.sim(observation_times = np.linspace(0, 30, 15), max_n_steps = 10000, s = True)
    obs[:,1:] = obs[:,1:] + np.random.normal(0,10,size = obs[:,1:].shape)

with bl.model.Model() as m:
    ir1 = bl.rvs.Normal('ir1',np.log(0.01), np.log(1.1), transform = bl.rvs.transform.LowerBoundRVTransform(0))
    br = bl.rvs.Normal('br', np.log(0.5), np.log(1.1), transform = bl.rvs.transform.LowerBoundRVTransform(0))
    dr = bl.rvs.Normal('dr', np.log(1.0), np.log(1.1), transform = bl.rvs.transform.LowerBoundRVTransform(0))
    ir2 = bl.rvs.Normal('ir2', np.log(0.01), np.log(1.1), transform = bl.rvs.transform.LowerBoundRVTransform(0))
    y0 = [50, 100]
    obs_process = LotkaVolterra2("lv", y0, ir1, br, dr, ir2, 30, observed = obs)
    
    print("Started sampling!")
    sampler = bl.inference.samplers.M_MVNMetropolisHastingsSpecial(m, scale = 1e-5)
    chain, tchain = sampler.run(n_iters = 2000, init_params = np.log([0.02,0.3,0.85,0.01]))
    bl.utils.save_chain(tchain, "results/tchain.csv")
    bl.utils.save_chain(chain, "results/chain.csv")
