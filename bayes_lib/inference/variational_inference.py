import numpy as np
import autograd.scipy as agsp
from ..math.optimization import GradientDescent
import abc

class VariationalDistributionNotDifferentiableException(Exception):
    pass

class VariationalDistribution(object):

    is_differentiable = False
    _variational_params = None

    def __init__(self, n_model_params):
        self.n_model_params = n_model_params

    @property
    def variational_params(self):
        return self._variational_params

    @variational_params.setter
    def variational_params(self, variational_params):
        self._variational_params = variational_params

    @abc.abstractmethod
    def sample(self, n_samples):
        return
    
    @abc.abstractmethod
    def log_density(self, variational_samples):
        return

    def grad_log_density(self, variational_samples):
        if not is_differentiable:
            raise VariationalDistributionNotDifferentiableException
        return

class MeanField(VariationalDistribution):

    is_differentiable = True

    def __init__(self, n_model_params, init = None):
        super().__init__(n_model_params)
        if init is None:
            vm = np.zeros(self.n_model_params)
            vs = np.zeros(self.n_model_params)
            self.variational_params = np.hstack([vm, vs])
        else:
            self.variational_params = init

    @property
    def v_means(self):
        return self.__v_means

    @v_means.setter
    def v_means(self, v_means):
        self.__v_means = v_means

    @property
    def v_stds(self):
        return self.__v_stds

    @v_stds.setter
    def v_stds(self, v_stds):
        self.__v_stds = np.exp(v_stds)

    @VariationalDistribution.variational_params.setter
    def variational_params(self, variational_params):
        self.v_means = variational_params[:self.n_model_params]
        self.v_stds = variational_params[self.n_model_params:]
        self._variational_params = variational_params

    def sample(self, n_samples):
        v_samples = np.random.normal(0, 1, size = (n_samples, self.n_model_params))
        v_samples = self.v_means + v_samples * self.v_stds
        return v_samples

    def log_density(self, variational_samples):
        n_samples = variational_samples.shape[0]
        log_densities = np.zeros(n_samples)
        for i in range(n_samples):
            log_densities[i] = np.sum([agsp.stats.norm.logpdf(variational_samples[i,j], self.v_means[j], self.v_stds[j]) for j in range(self.n_model_params)])
        return log_densities

    def grad_log_density(self, variational_samples):
        n_samples = variational_samples.shape[0]
        grad_values = np.zeros((n_samples, self.n_model_params * 2))
        v_vars = self.v_stds**2
        for i in range(n_samples):
            grad_values[i,:self.n_model_params] = ((variational_samples[i] - self.v_means)/(v_vars))
            grad_values[i,self.n_model_params:] =  (-1 + ((variational_samples[i] - self.v_means)**2)/(v_vars))
        return grad_values

class BlackBoxVariationalInference(object):

    __v_dist = None

    def __init__(self, model, variational_dist = MeanField, init = None):

        assert variational_dist.is_differentiable
        self.model = model
        self.v_dist = variational_dist(self.model.n_params, init = init)

    @property
    def v_dist(self):
        return self.__v_dist

    @v_dist.setter
    def v_dist(self, v_dist):
        self.__v_dist = v_dist

    def run(self, optimizer = GradientDescent(learning_rate = 1e-4)):

        def mc_elbo(v_pars, n_samples = 10):
            self.v_dist.variational_params = v_pars
            v_samples = self.v_dist.sample(n_samples)
            logq_model = self.v_dist.log_density(v_samples)

            elbo_est = 0
            for i in range(n_samples):
                logp_model = self.model.log_density_p(v_samples[i,:])
                elbo_est += logp_model - logq_model[i]
            return elbo_est/n_samples

        def mc_grad_elbo(v_pars, n_samples = 10):
            self.v_dist.variational_params = v_pars
            v_samples = self.v_dist.sample(n_samples)
            logq_model = self.v_dist.log_density(v_samples)
            
            grad_values = self.v_dist.grad_log_density(v_samples)
            for i in range(n_samples):
                logp_model = self.model.log_density_p(v_samples[i,:])
                grad_values[i,:] = grad_values[i,:] * (logp_model - logq_model[i])
            return np.mean(grad_values, axis = 0)

        #optimizer = GradientDescent(learning_rate = 1e-3)
        res = optimizer.run(lambda x: -mc_elbo(x), lambda x: -mc_grad_elbo(x), self.v_dist.variational_params, max_iters = 1000, convergence = 1e-3)
        self.v_dist.variational_params = res.position
        return res, self.v_dist
