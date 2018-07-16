import abc
import numpy as np
from scipy.stats import norm, multivariate_normal
import copy

from bayes_lib import Model

class MCMCSampler(object):

    def __init__(self, model):
        self.model = model
    
    @abc.abstractmethod
    def run(self, init_params = None, n_iters = 1000):
        return
        
class MetropolisHastings(MCMCSampler):

    def run(self, init_params = None, n_iters = 1000):
        chain = []
        tchain = []
        if init_params is not None:
            pos = init_params
        else:
            pos = np.random.normal(-1,1,size = self.model.n_params)

        lpdf = self.model.log_density(pos)
        print("Initial LPDF %f" % (lpdf))

        for i in range(n_iters):
            print("Iteration %d" % (i))
            new_pos, forward_log_density, reverse_log_density = self.proposal_and_log_density(pos)
            
            new_lpdf = self.model.log_density(new_pos)
            print("New LPDF %f" % (new_lpdf))
            print("Accept Probability %f" % (np.exp(new_lpdf + forward_log_density - lpdf - reverse_log_density)))
            if new_lpdf + forward_log_density > -np.inf and np.log(np.random.rand()) < np.minimum(0, (new_lpdf + forward_log_density) - (lpdf + reverse_log_density)):
                print("Accept")
                chain.append(new_pos)
                pos = new_pos
                lpdf = new_lpdf
            else:
                chain.append(pos)
        return np.array(chain), self.model.constrain_parameters(np.array(chain))
    
class M_MVNMetropolisHastings(MetropolisHastings):

    def __init__(self, model, scale = 0.01):
        super().__init__(model)
        self.scale = scale

    def proposal_and_log_density(self, current_pos):
        new_pos = current_pos * np.exp(np.random.multivariate_normal(np.zeros(self.model.n_params), self.scale * np.eye(self.model.n_params)))
        return new_pos, 0, 0

class A_MVNMetropolisHastings(MetropolisHastings):

    def __init__(self, model, scale = 0.2):
        super().__init__(model)
        self.scale = scale

    def proposal_and_log_density(self, current_pos):
        new_pos = current_pos + np.random.multivariate_normal(np.zeros(self.model.n_params), self.scale * np.eye(self.model.n_params))
        return new_pos, 0, 0

def importance_sampling(m, i_dist, i_dist_density, n_i_samples = 1000, n_samples = 1000):
    if not isinstance(m, Model):
        raise TypeError("Wrong model type!")
    
    samples = []
    tsamples = []
    weights = []
    for i in range(n_i_samples):
        s = i_dist()
        m.set_param_vector(s)
        w = np.exp(m.log_density())/i_dist_density(s)
        
        samples.append(s)
        tsamples.append(m.get_constrained_params())
        weights.append(w)

    # Resampling step
    s_a = np.array(samples)
    ts_a = np.array(tsamples)
    w_a = np.array(weights)[:,0]
    idxs = np.random.choice(n_i_samples, n_samples, p = w_a/sum(w_a))
    return s_a[idx,:], ts_a[idx,:]
