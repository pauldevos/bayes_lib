import numpy as np
from bayes_lib.math.optimizers import GradientDescent
from .distributions import *
import abc

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

class BlackBoxVariationalInference(VariationalInference):

    def __init__(self, model, variational_dist = MeanField(), init = None):

        assert isinstance(variational_dist, DifferentiableVariationalDistribution)
        self.model = model
        self.v_dist = variational_dist
        self.v_dist.initialize(self.model.n_params, init = init)

    def run(self, n_mc_samples = 200, n_grad_samples = 200, optimizer = GradientDescent(learning_rate = 1e-5), max_opt_iters = 1000, convergence = 1e-4):

        def mc_elbo(v_pars, n_samples = n_mc_samples):
            self.v_dist.variational_params = v_pars
            v_samples = self.v_dist.sample(n_samples)
            logp_model = agnp.mean(self.model.log_density_p(v_samples))
            return logp_model + self.v_dist.entropy(v_pars, v_samples)

        def mc_grad_elbo(v_pars, n_samples = n_grad_samples):
            self.v_dist.variational_params = v_pars
            v_samples = self.v_dist.sample(n_samples)
            logp_model = self.model.log_density_p(v_samples)
            logq_model = self.v_dist.log_density(v_samples)
            grad_values = self.v_dist.grad_log_density(v_samples)
            return np.mean(grad_values * (logp_model - logq_model)[:,np.newaxis], axis = 0)

        res = optimizer.run(lambda x: -mc_elbo(x), lambda x: -mc_grad_elbo(x), self.v_dist.variational_params, max_iters = max_opt_iters, convergence = convergence)
        self.v_dist.variational_params = res.position
        return res, self.v_dist

class ReparameterizedVariationalInference(VariationalInference):

    def __init__(self, model, variational_dist = MeanField(), init = None):

        assert isinstance(variational_dist, ReparameterizableVariationalDistribution)
        assert model.is_differentiable
        self.model = model
        self.v_dist = variational_dist
        self.v_dist.initialize(self.model.n_params, init = init)

    def run(self, n_mc_samples = 200, n_grad_samples = 200, optimizer = GradientDescent(learning_rate = 1e-4), max_opt_iters = 1000, convergence = 1e-3):

        def mc_elbo(variational_params, n_samples = n_mc_samples):
            v_samples = self.v_dist.sample_p(variational_params, n_samples)
            logp_model = agnp.mean(self.model.log_density_p(v_samples))
            return logp_model + self.v_dist.entropy(variational_params, v_samples)

        grad_mc_elbo = autograd.grad(mc_elbo)

        res = optimizer.run(lambda x: -mc_elbo(x), lambda x: -grad_mc_elbo(x), self.v_dist.variational_params, max_iters = max_opt_iters, convergence = convergence)
        self.v_dist.variational_params = res.position
        return res, self.v_dist






        
