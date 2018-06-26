import numpy as np
from scipy.stats import norm, multivariate_normal

from .model import Model

def metropolis_hastings(m, init_params = None, n_iter = 1000):
    #if not isinstance(m, Model):
    #    raise TypeError("Wrong model type!")

    chain = []
    tchain = []
    
    if init_params is None:
        pos = m.get_param_vector()
        cpos = m.get_constrained_params()
    else:
        m.set_param_vector(init_params)
        pos = m.get_param_vector()
        cpos = m.get_constrained_params()

    N_params = pos.shape[0]
    lpdf = m.log_density()

    for i in range(n_iter):
        print(i)
        # Sample a proposal
        pert = np.random.multivariate_normal(np.zeros(pos.shape), 0.01 * np.eye(N_params))
        new_pos = pos * np.exp(pert)
        m.set_param_vector(new_pos)
        new_lpdf = m.log_density()
        forward_kernel = multivariate_normal.logpdf(pos, new_pos, np.eye(N_params))
        reverse_kernel = multivariate_normal.logpdf(new_pos, pos, np.eye(N_params))

        if new_lpdf + forward_kernel > -np.inf and np.log(np.random.rand()) < np.minimum(0.,(new_lpdf + forward_kernel) - (lpdf + reverse_kernel)):
            chain.append(new_pos)
            cpos = m.get_constrained_params()
            pos = new_pos
            lpdf = new_lpdf
        else:
            chain.append(pos)
        tchain.append(cpos)
    return np.array(chain), np.array(tchain)

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
