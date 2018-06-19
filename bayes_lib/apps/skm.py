import numpy as np
import abc
from scipy.stats import norm

from ..rvs import RandomVariable, get_rv_value
import time

class StochasticKineticRandomVariable(RandomVariable):

    def __init__(self, name, y0, end_T, times = None, observed = None):
        super().__init__(name, observed = observed)
        self.y0 = y0
        self.end_T = end_T
        self.times = times

    def check_value(self, v):
        return True

    @abc.abstractmethod
    def gen_rate_fx(self, state):
        return None

    def obs_log_likelihood(self, obs, particle):
        return norm.logpdf(obs, particle, 5)

    def log_density(self, n_smc = 50, v = None):
        if v is None:
            v = self.cvalue
            jd = self.jdet
        
        rate_fx = self.gen_rate_fx()
        
        T = self.times.shape[0]
        if len(v.shape) == 1:
            N = 1
        else:
            N = v.shape[1]
        particles = np.zeros((n_smc, N, T))

        # Sample initial latent paths
        for i in range(n_smc):
            particles[i,:,0] = get_rv_value(self.y0, s = True)

        llhood_est = 0
        
        # SMC for each time interval
        weights = 1/n_smc * np.ones(n_smc)
        deltas = np.diff(self.times)
        for i in range(1,T):
            resampled_indices = np.random.choice(list(range(n_smc)), n_smc, p = weights, replace = True)
            resampled_particles = particles[resampled_indices, :, i-1]
            res = list(map(lambda x: self.gillespie_simulate(x, self.reactions, rate_fx, np.array([0,deltas[i-1]])), resampled_particles))
            new_weights = [-np.inf if r is None else self.obs_log_likelihood(v[i], r[1,1:]) for r in res]
            particles[:,:,i] = np.array([0 if r is None else r[1,1:] for r in res])
            n = np.exp(new_weights)
            d = sum(n)
            if d == 0:
                return -np.inf
            else:
                weights = n/d
            weights = weights[:,0]
            llhood_est += np.mean(new_weights)
        return llhood_est
    
    def gillespie_simulate(self, initial_state, reactions, rate_fx, times = None, tau_fn = None):
        if times is None:
            times = self.times
        step_counter = 0
        if tau_fn:
            states = [np.hstack((np.array([0]),initial_state))]
            if times is not None:
                times = times[1:]
            cur_state = states[0]
            terminate = False
            while not terminate and cur_state[0] < self.end_T:
                updated_state = np.zeros(initial_state.shape[0] + 1)
                rates = rate_fx(cur_state)
                tau = tau_fn(cur_state)
                # Update time
                updated_state[0] = cur_state[0] + tau
                
                # If all of the rates are 0, then the system is effectively dead
                if not (np.array(rates) > 0).all():
                    return None

                if not (np.array(rates) < 10000).all():
                    return None
                
                # Randomly simulate the number of reactions i that occur in that interval
                num_transitions = [np.random.poisson(lam = r * tau) for r in rates]

                # Update the state based on the number of reactions of each type
                updated_state[1:] = cur_state[1:]
                for i, nt in enumerate(num_transitions):
                    updated_state[1:] += nt * reactions[i,:]
                states.append(updated_state)
                cur_state = updated_state
            return np.vstack(states)
        else: 
            states = [np.hstack((np.array([0]),initial_state))]
            if times is not None:
                times = times[1:]
                self.end_T = max(times)
            cur_state = states[0]
            terminate = False
            while not terminate and cur_state[0] < self.end_T:
                if step_counter > 200000:
                    print("Took too long!")
                    return None
                updated_state = np.zeros(initial_state.shape[0] + 1)
                rates = rate_fx(cur_state)
                total_rate = sum(rates)
                if not np.isfinite(total_rate):
                    print(cur_state)
                    print("Total rate is infinite!")
                    return None
                if total_rate == 0:
                    tau = np.inf
                else:
                    tau = np.random.exponential(1/total_rate)
                    event = np.random.choice(rates.shape[0], p = rates/total_rate)
                    step_counter += 1

                    t = reactions[event,:]
                    updated_state[1:] = cur_state[1:] + t
                updated_state[0] = cur_state[0] + tau
                if times is not None:
                    while not terminate and updated_state[0] > times[0]:
                        states.append(np.append(self.times[0], cur_state[1:]))
                        times = times[1:]
                        if len(times) == 0:
                            terminate = True
                else:
                    states.append(updated_state)
                cur_state = updated_state
            #print("Steps Taken: ", step_counter)
            return np.vstack(states)
