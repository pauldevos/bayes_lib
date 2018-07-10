import numpy as np
import abc
from ..model import Model
from ..rvs import RandomVariable, get_rv_value
from scipy.stats import norm, lognorm
import pathos.pools as pp
import random

class MarkovJumpProcess(RandomVariable):

    def __init__(self, name, initial_state, params, termination_time, max_n_steps = np.inf, observed = None):
        super().__init__(name, observed = observed)
        self.initial_state = initial_state
        self.params = params
        self.termination_time = termination_time

    def single_step(self, state, params, s = False):
        rates = params * self.compute_propensities(state)
        total_rate = np.sum(rates)
        if total_rate == 0:
            tau = np.inf
            return tau, state
        else:
            tau = np.random.exponential(1/total_rate)
    
        reaction_idx = random.choices(range(rates.shape[0]), weights = rates)[0]
        new_state = self.do_reaction(reaction_idx, state)
        return tau, new_state

    def sim_delta(self, initial_state, params, dt, max_n_steps = np.inf, ret_steps = False, s = False):
        current_time = 0
        current_state = initial_state
        n_steps = 0
        while current_time < dt:
            tau, new_state = self.single_step(current_state, params, s = s)
            current_time += tau
            current_state = new_state
            n_steps += 1
            if n_steps > max_n_steps:
                if ret_steps:
                    return n_steps, None
                else:
                    return None
        if ret_steps:
            return n_steps, current_state
        else:
            return current_state

    def sim(self, observation_times = None, termination_time = None, max_n_steps = np.inf, s = False):

        if observation_times is None and termination_time is None:
            raise TypeError("Must specify either a set of observation times or a termination time")
        
        initial_state = get_rv_value(self.initial_state, s = s)
        params = get_rv_value(self.params, s = s)
        states = []
        
        # If simulating at specific times, step using deltas
        n_steps = 0
        if observation_times is not None:
            times = np.hstack((np.array([0]),observation_times))
            states.append(initial_state)
            current_state = initial_state
            diffs = np.diff(times)
            for i in range(len(diffs)):
                ns, new_state = self.sim_delta(current_state, params, diffs[i], max_n_steps = max_n_steps, ret_steps = True, s = s)
                n_steps += ns
                if n_steps > max_n_steps:
                    return None
                if new_state is None:
                    return None
                states.append(new_state)
                current_state = new_state
        else:
            times = []
            current_state = initial_state
            current_time = 0
            while current_time < termination_time:
                times.append(current_time)
                states.append(current_state)
                tau, new_state = self.single_step(current_state, params, s = s)
                current_time += tau
                current_state = new_state
                n_steps += 1
                if n_steps > max_n_steps:
                    return None
            times = np.array(times)
        return np.c_[np.array(times), np.array(states)]

    @abc.abstractmethod
    def compute_propensities(self, state):
        return

    @abc.abstractmethod
    def do_reaction(self, reaction_idx, state):
        return
    
    def check_value(self, v):
        return True

    def sample(self, max_n_steps = np.inf):
        return self.sim(termination_time = self.termination_time, max_n_steps = max_n_steps, s = True)

    def obs_log_likelihood(self, v, r):
        return np.sum(norm.logpdf(v, r, 10))
    
    # Computes an approximate log density using PMCMC
    def log_density(self, n_smc = 100, v = None):
        p = pp.ProcessPool()
        if v is None:
            v = self.cvalue
            jd = self.jdet
            
        # Observations is a matrix where column 0 is observation times
        T = v[:,0]

        # Number of species
        N = v.shape[1] - 1

        # Generate particles with shape n_smc, N_dims, N_time_points
        particles = np.zeros((n_smc, T.shape[0], N))
        params = get_rv_value(self.params)

        # Sample initial positions for the particles
        for i in range(n_smc):
            particles[i,0,:] = get_rv_value(self.initial_state)

        llhood_est = 0
        
        # SMC for each time interval
        weights = 1/n_smc * np.ones(n_smc)
        deltas = np.diff(T)
        for i in range(1,T.shape[0]):
            
            # Resampling step
            resampled_indices = random.choices(range(n_smc), weights = weights, k = n_smc)
            resampled_particles = particles[resampled_indices, i-1, :]
            
            # Simulate forward in time
            res = list(p.map(lambda x: self.sim_delta(x, params, deltas[i-1]), resampled_particles))

            # Compute new particles and new weights
            new_weights = [-np.inf if r is None else self.obs_log_likelihood(v[i,1:], r) for r in res]
            particles[:,i,:] = np.array([np.zeros(N) if r is None else r for r in res])
            n = np.exp(new_weights)
            d = sum(n)
            if d == 0:
                print("Inf")
                return -np.inf
            else:
                weights = n/d

            # Update likelihood estimate
            llhood_est += np.log(np.mean(n))
        return llhood_est

class BirthDeathProcess(MarkovJumpProcess):

    def __init__(self, name, initial_state, birth_rate, death_rate, termination_time, max_n_steps = np.inf, observed = None):
        self.birth_rate = birth_rate
        self.death_rate = death_rate
        params = [birth_rate, death_rate]
        super().__init__(name, initial_state, params, termination_time, max_n_steps = max_n_steps, observed = observed)
        self.reactions = np.array([[1],[-1]])

    def compute_propensities(self, state):
        return np.array([state[0], state[0]])

    def do_reaction(self, reaction_idx, state):
        return state + self.reactions[reaction_idx,:]


class LotkaVolterra(MarkovJumpProcess):

    def __init__(self, name, initial_state, birth_rate, interaction_rate, death_rate, termination_time, max_n_steps = np.inf, observed = None):
        self.birth_rate = birth_rate
        self.interaction_rate = interaction_rate
        self.death_rate = death_rate
        params = [birth_rate, interaction_rate, death_rate]
        super().__init__(name, initial_state, params, termination_time, max_n_steps = max_n_steps, observed = observed)
        self.reactions = np.array([[1,0],[-1,1],[0,-1]])
        
    def compute_propensities(self, state):
        return np.array([state[0], state[0] * state[1], state[1]])

    def do_reaction(self, reaction_idx, state):
        return state + self.reactions[reaction_idx, :]

class LotkaVolterra2(MarkovJumpProcess):

    def __init__(self, name, initial_state, ir1, br, dr, ir2, termination_time, max_n_steps = np.inf, observed = None):
        self.ir1 = ir1
        self.br = br
        self.dr = dr
        self.ir2 = ir2
        params = [self.ir1, self.br, self.dr, self.ir2]
        super().__init__(name, initial_state, params, termination_time, max_n_steps = max_n_steps, observed = observed)
        self.reactions = np.array([[1,0],[-1,0],[0,1],[0,-1]])

    def compute_propensities(self, state):
        return np.array([state[0]*state[1], state[0], state[1], state[0]*state[1]])

    def do_reaction(self, reaction_idx, state):
        return state + self.reactions[reaction_idx, :]
