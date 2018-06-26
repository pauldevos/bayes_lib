import bayes_lib as bl
from bayes_lib.apps.markov_jump_process import *
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(1)
obs = None
with bl.model.Model() as m:
    br = bl.rvs.Normal('br', 0, np.log(1.1), transform = bl.transform.LowerBoundRVTransform(0))
    ir = bl.rvs.Normal('ir', np.log(0.005), np.log(1.001), transform = bl.transform.LowerBoundRVTransform(0))
    dr = 0.6
    y0 = [50, 100]
    obs_process = LotkaVolterra("lv", y0, br, ir, dr, 30)
    obs = obs_process.sim(observation_times = np.linspace(0, 30, 10), s = True)

obs[:,1:] = obs[:,1:] + np.random.normal(0, 10, size = obs[:,1:].shape)
with bl.model.Model() as m:
    br = bl.rvs.Normal('br', 0, np.log(1.1), transform = bl.transform.LowerBoundRVTransform(0))
    ir = bl.rvs.Normal('ir', np.log(0.005), np.log(1.001), transform = bl.transform.LowerBoundRVTransform(0))
    dr = 0.6
    y0 = [50, 100]
    obs_process = LotkaVolterra("lv", y0, br, ir, dr, 30, observed = obs)
    chain, tchain = bl.sampling.metropolis_hastings(m, n_iter = 100, init_params = np.log(np.array([0.9509, 0.00499])))
    bl.utils.save_chain(tchain, "tchain.csv")
    bl.utils.save_chain(chain, "chain.csv")

"""
with bl.model.Model() as m:
    br = bl.rvs.Normal('br', 0, np.log(1.02), transform = bl.transform.LowerBoundRVTransform(0))
    dr = 0.81
    y0 = [bl.rvs.Normal('y0', np.log(100), np.log(1.2), transform = bl.transform.LowerBoundRVTransform(0))]
    obs_process = BirthDeathProcess("birth_death", y0, br, dr, 3)
    for i in range(100):
        times, path = obs_process.sim(observation_times = [0.2,0.4,0.6,0.8,1,1.2,1.4,1.6], s = True)
        plt.plot(times, path[:,0], '.-', alpha = 0.3)
    plt.show()

data = None
with bl.model.SimulationModel() as m:
    br = 1.38175
    dr = 0.81
    y0 = np.array([100])
    obs_process = BirthDeathProcess("birth_death", y0, br, dr, 3) 
    path = obs_process.sample()
    data = path[::100,:] 
    np.savetxt("data.csv", data, delimiter = ',')
    plt.plot(data[:,0], data[:,1])
    plt.show()
data = np.loadtxt("data.csv", delimiter = ',')
with bl.model.Model() as m2:
    br = bl.rvs.Normal('br', 0, np.log(2), transform = bl.transform.LowerBoundRVTransform(0))
    dr = bl.rvs.Normal('dr', 0, 1, transform = bl.transform.LowerBoundRVTransform(0))
    y0 = np.array([100])
    obs_process = BirthDeathProcess("birth_death", y0, br, dr, 3, times = data[:,0], observed = data[:,1])

    chain, tchain = bl.sampling.metropolis_hastings(m2, n_iter = 1000, init_params = np.array([1.38175, 0.81]))
    bl.utils.save_chain(tchain, "chain.csv")
    plt.hist(tchain[500:,0])
    plt.show()
"""
