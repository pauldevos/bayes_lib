import abc
import tensorflow as tf
import numpy as np
from scipy.stats import norm, multivariate_normal
import copy

from bayes_lib import Model

class MCMCSampler(object):

    @abc.abstractmethod
    def run(self, model, init = None, n_iters = 1000):
        return

class MultivariateUniSliceSampler(MCMCSampler):
    
    def run(self, model, init, width_scales = 1, feed_dict = None, n_iters = 1000):
        
        w = width_scales * np.ones(init.shape[0])
        pos = init
        tpos = model.transform_param(pos)
        chain = [pos]
        tchain = [tpos]
        lpdf = model.log_density(pos, feed_dict = feed_dict)
        for i in range(n_iters):
            print("Iteration %d" % (i+1))
            # Sample auxiliary variable to define the horizontal slice
            
            new_pos = pos.copy()
            for j in range(init.shape[0]):
                y = lpdf + np.log(1.0 - np.random.uniform())
                
                # For each dimension, perform univariate slice sampling
                u = np.random.uniform(0, w[j])
                lb = new_pos[j] - u
                rb = new_pos[j] + (w[j] - u)

                # Automatically shring intervals
                new_pos[j] = lb
                while y <= model.log_density(new_pos, feed_dict = feed_dict):
                    lb -= w[j]
                    new_pos[j] = lb

                new_pos[j] = rb
                while y <= model.log_density(new_pos, feed_dict = feed_dict):
                    rb += w[j]
                    new_pos[j] = rb

                new_pos[j] = np.random.uniform(lb, rb)
                new_lpdf = model.log_density(new_pos, feed_dict = feed_dict)
                while y > new_lpdf:
                    if new_pos[j] < pos[j]:
                        lb = new_pos[j]
                    else:
                        rb = new_pos[j]
                    new_pos[j] = np.random.uniform(lb, rb)
                    new_lpdf = model.log_density(new_pos, feed_dict = feed_dict)
            
                lpdf = new_lpdf
            pos = new_pos
            chain.append(pos)
            tchain.append(model.transform_param(pos))
        return np.array(chain), np.array(tchain)

class MultivariateHyperRectSliceSampler(MCMCSampler):

    def run(self, model, init, width_scales = 0.1, feed_dict = None, n_iters = 1000):
        
        w = width_scales * np.ones(init.shape[0])
        pos = init
        tpos = model.transform_param(pos)
        chain = [pos]
        tchain = [tpos]
        lpdf = model.log_density(pos, feed_dict = feed_dict)
        for i in range(n_iters):
            print("Iteration %d" % (i+1))
            # Sample auxiliary variable to define the horizontal slice
            y = lpdf + np.log(1.0 - np.random.uniform())
            new_pos = pos.copy()
            lbs = []
            rbs = []
            for j in range(init.shape[0]):

                # For each dimension, perform univariate slice sampling
                u = np.random.uniform(0, w[j])
                lb = new_pos[j] - u
                rb = new_pos[j] + (w[j] - u)
                lbs.append(lb)
                rbs.append(rb)
                new_pos[j] = np.random.uniform(lb, rb)
            
            new_lpdf = model.log_density(new_pos, feed_dict = feed_dict)
            while y > new_lpdf:
                for j in range(init.shape[0]):
                    if new_pos[j] < pos[j]:
                        lbs[j] = new_pos[j]
                    else:
                        rbs[j] = new_pos[j]
                    new_pos[j] = np.random.uniform(lbs[j], rbs[j])
                new_lpdf = model.log_density(new_pos, feed_dict = feed_dict) 
            
            pos = new_pos
            lpdf = new_lpdf
            chain.append(pos)
            tchain.append(model.transform_param(pos))
        return np.array(chain), np.array(tchain)

class MetropolisHastings(MCMCSampler):

    @abc.abstractmethod
    def proposal_and_log_density(self, current_pos, chain):
        return

    def run(self, model, feed_dict = None, init = None, n_iters = 1000):
        if init is not None:
            pos = init
        else:
            pos = np.random.normal(-1,1,size = model.n_params)
        
        tpos = model.transform_param(pos)
        chain = [pos]
        tchain = [tpos]
        lpdf = model.log_density(pos, feed_dict = feed_dict)
        for i in range(n_iters):
            new_pos, forward_log_density, reverse_log_density = self.proposal_and_log_density(pos, chain)
            
            new_lpdf = model.log_density(new_pos, feed_dict = feed_dict)
            if new_lpdf + forward_log_density > -np.inf and np.log(np.random.rand()) < np.minimum(0, (new_lpdf + forward_log_density) - (lpdf + reverse_log_density)):
                tpos = model.transform_param(new_pos)
                chain.append(new_pos)
                tchain.append(tpos)
                pos = new_pos
                lpdf = new_lpdf
            else:
                chain.append(pos)
                tchain.append(tpos)
        return np.array(chain), np.array(tchain)

class M_MVNMetropolisHastings(MetropolisHastings):

    def __init__(self, scale = 0.01):
        super().__init__()
        self.scale = scale

    def proposal_and_log_density(self, current_pos, chain):
        new_pos = current_pos * np.exp(np.random.multivariate_normal(np.zeros(current_pos.shape[0]), self.scale * np.eye(current_pos.shape[0])))
        return new_pos, 0, 0

class A_MVNMetropolisHastings(MetropolisHastings):

    def __init__(self, scale = 0.2):
        super().__init__()
        self.scale = scale

    def proposal_and_log_density(self, current_pos, chain):
        new_pos = current_pos + np.random.multivariate_normal(np.zeros(current_pos.shape[0]), self.scale * np.eye(current_pos.shape[0]))
        return new_pos, 0, 0

class A_MVNMetropolisHastings_Adapt(MetropolisHastings):

    def __init__(self, scale = 1, window = 100):
        super().__init__()
        self.window = window
        self.scale = scale

    def proposal_and_log_density(self, current_pos, chain):
        if len(chain) < self.window:
            cov_est = self.scale * np.eye(current_pos.shape[0])
        else:
            cov_est = np.cov(np.vstack(chain[-self.window:]).T)
        new_pos = current_pos + np.random.multivariate_normal(np.zeros(current_pos.shape[0]), (2.38**2) * cov_est/current_pos.shape[0])
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
