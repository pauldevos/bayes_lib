import bayes_lib as bl
from bayes_lib.apps.skm import *
import numpy as np
from scipy.stats import norm
from scipy.integrate import odeint

import matplotlib.pyplot as plt
import seaborn as sns

class BirthDeathProcess(StochasticKineticRandomVariable):

    def __init__(self, name, y0, birth_rate, death_rate, end_T, times = None, observed = None):
        super().__init__(name, y0, end_T, times = times, observed = observed)
        self.birth_rate = birth_rate
        self.death_rate = death_rate
        self.reactions = np.array([[1],[-1]])

    def gen_rate_fx(self, s = False):
        br = bl.get_rv_value(self.birth_rate, s = s)
        dr = bl.get_rv_value(self.death_rate, s = s)
        def rate_fx(state):
            return np.array([br * state[1], dr * state[1]])
        return rate_fx

    def sample(self):
        y0 = bl.get_rv_value(self.y0)
        return self.gillespie_simulate(y0, self.reactions, self.gen_rate_fx(s = True))

"""
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
"""
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
