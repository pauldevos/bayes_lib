import numpy as np
from bayes_lib.math.optimizers import GradientDescent
from .distributions import *
import abc
import functools
import time

class VariationalInference(object):

    _v_dist = None

    @property
    def v_dist(self):
        return self._v_dist

    @v_dist.setter
    def v_dist(self, v_dist):
        self._v_dist = v_dist

    @abc.abstractmethod
    def run(self):
        return

"""
class BlackBoxVariationalInference(VariationalInference):

    def __init__(self, model, variational_dist = MeanField(), init = None):

        assert isinstance(variational_dist, DifferentiableVariationalDistribution)
        self.model = model
        self.v_dist = variational_dist
        self.v_dist.initialize(self.model.n_params, init = init)

    def run(self, n_mc_samples = 1000, n_grad_samples = 1000, optimizer = GradientDescent(learning_rate = 1e-5), iter_func = None, iter_interval = 1, max_opt_iters = 1000, convergence = 1e-4):

        def mc_elbo(v_pars, n_samples = n_mc_samples):
            self.v_dist.variational_params = v_pars
            v_samples = self.v_dist.sample(n_samples)
            logp_model = agnp.mean(self.model.log_density(v_samples))
            return logp_model + self.v_dist.entropy(v_pars, v_samples)

        def mc_grad_elbo(v_pars, n_samples = n_grad_samples):
            self.v_dist.variational_params = v_pars
            v_samples = self.v_dist.sample(n_samples)
            logp_model = self.model.log_density(v_samples)[:,0]
            logq_model = self.v_dist.log_density(v_samples)
            grad_values = self.v_dist.grad_log_density(v_samples)
            return np.mean(grad_values * (logp_model - logq_model)[:,np.newaxis], axis = 0) + self.v_dist.grad_entropy(v_pars, v_samples)

        res = optimizer.run(lambda x: -mc_elbo(x), lambda x: -mc_grad_elbo(x), self.v_dist.variational_params, iter_func = iter_func, iter_interval = iter_interval, max_iters = max_opt_iters, convergence = convergence)
        self.v_dist.variational_params = res.position
        return res, self.v_dist
"""

class ReparameterizedVariationalInference(VariationalInference):

    def __init__(self, model, variational_dist = MeanField(), init = None):

        self.model = model
        self.v_dist = variational_dist
        self.init = init

    def run(self, feed_dict = None, n_mc_samples = 10, n_grad_samples = 10, optimizer = GradientDescent(learning_rate = 0.001), iter_func = None, iter_interval = 1, max_opt_iters = 1000, convergence = 1e-4):
        
        self.v_dist.initialize(self.model, n_mc_samples, n_grad_samples)
        rng = tf.random_normal([self.model.n_params])
        z = self.v_dist.v_means + rng * tf.exp(self.v_dist.v_stds)
        test = self.model.log_density_p(z)
        grad = tf.gradients(test, [self.v_dist.v_means, self.v_dist.v_stds])
        print(grad)
        1/0
        #mc_elbo_op = tf.reduce_mean([self.model.log_density_p(z[i,:]) for i in range(n_mc_samples)]) #+ self.v_dist.entropy_op()
        #grad_mc_elbo_op = tf.gradients(mc_elbo_op, [self.v_dist.v_means, self.v_dist.v_stds])
        print(grad_mc_elbo_op)
        1/0
    
        def mc_elbo(variational_params):
            v_dict = self.v_dist.feed_dict_update(feed_dict, variational_params)
            return self.model.sess.run(mc_elbo_op, feed_dict = v_dict)

        def grad_mc_elbo(variational_params):
            v_dict = self.v_dist.feed_dict_update(feed_dict, variational_params)
            return self.model.sess.run(grad_mc_elbo_op, feed_dict = v_dict)

        print(grad_mc_elbo(np.array([0,0,1,1])))
        1/0
        """ 
        def mc_elbo(variational_params):
            _, param_samples = self.v_dist.sample_p(variational_params)
            lds = [self.model.log_density_p(param_samples[i,:], feed_dict = feed_dict) for i in range(param_samples.shape[0])]
            return np.mean(lds) + self.v_dist.entropy(variational_params, param_samples)
        """ 
        """
        def grad_mc_elbo(variational_params):
            param_samples, grad_samps = self.v_dist.grad_sample_p(variational_params)
            grad_lds = [self.model.grad_log_density(param_samples[i,:], feed_dict = feed_dict) for i in range(param_samples.shape[0])]
            grad_entropy = self.v_dist.grad_entropy(variational_params, param_samples)
            return (sum([grad_samps[i] * grad_lds[i] for i in range(n_grad_samples)])/n_grad_samples).flatten() + grad_entropy
        """ 
        res = optimizer.run(lambda x: -mc_elbo(x), lambda x: -grad_mc_elbo(x), self.init, iter_func = iter_func, iter_interval = iter_interval, max_iters = max_opt_iters, convergence = convergence)
        return res, self.v_dist






        
