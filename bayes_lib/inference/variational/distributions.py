import numpy as np
import autograd.scipy as agsp
import autograd.numpy as agnp
import autograd
import abc

class VariationalDistributionNotDifferentiableException(Exception):
    pass

class VariationalDistribution(object):

    is_differentiable = False
    _variational_params = None

    @abc.abstractmethod
    def initialize(self, n_model_params, init = None):
        return

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

    def initialize(self, n_model_params, init = None):
        self.n_model_params = n_model_params
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

    def entropy(self, variational_params):
        v_stds = variational_params[self.n_model_params:]
        return 0.5 * self.n_model_params * (1.0 + agnp.log(2 * agnp.pi)) + agnp.sum(v_stds)

    def log_density(self, variational_samples):
        return agsp.stats.multivariate_normal.logpdf(variational_samples, self.v_means, np.diag(self.v_stds))

    def grad_log_density(self, variational_samples):
        grad_values = np.zeros((variational_samples.shape[0], self.n_model_params * 2))
        v_vars = self.v_stds**2
        diffs = variational_samples - self.v_means
        grad_values[:,:self.n_model_params] = diffs/v_vars
        grad_values[:,self.n_model_params:] = -1 + (diffs**2)/v_vars
        return grad_values

class FullRank(VariationalDistribution):

    is_differentiable = True

    def initialize(self, n_model_params, init = None):
        self.n_model_params = n_model_params
        if init is None:
            vm = np.zeros(self.n_model_params)
            # Parameterizer covariance but sqrt of covariance
            vs = (np.zeros((self.n_model_params, self.n_model_params)) + np.eye(self.n_model_params)).flatten()
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
    def v_cov(self):
        return self.__v_cov

    @v_cov.setter
    def v_cov(self, v_cov):
        # Transform sqrt of covariance to actual covariance
        self.__v_cov = v_cov.reshape((self.n_model_params, self.n_model_params))
        self.__v_cov = agnp.dot(self.v_cov.T, self.v_cov)

    @VariationalDistribution.variational_params.setter
    def variational_params(self, variational_params):
        self.v_means = variational_params[:self.n_model_params]
        self.v_cov = variational_params[self.n_model_params:]
        self._variational_params = variational_params

    def sample(self, n_samples):
        L = agnp.linalg.cholesky(self.v_cov)
        v_samples = self.v_means + agnp.dot(agnp.random.randn(n_samples, self.n_model_params), L)
        return v_samples

    def entropy(self, variational_params):
        v_cov = variational_params[self.n_model_params:].reshape((self.n_model_params, self.n_model_params))
        cov = agnp.dot(v_cov.T, v_cov)
        return 0.5 * self.n_params * (1.0 + agnp.log(2 * agnp.pi)) + 0.5 * agnp.log(agnp.linalg.det(cov))

    def log_density(self, variational_samples):
        return agsp.stats.multivariate_normal.logpdf(variational_samples, self.v_means, self.v_cov)

    def grad_log_density(self, variational_samples):
        # Gradient w.r.t mu and cov
        grad_values = np.zeros((variational_samples.shape[0], self.n_model_params * 3))
        precision = np.linalg.inv(self.v_cov)
        diff = (variational_samples - self.v_means)
        grad_values[:,:self.n_model_params] = diff @ precision
        def grad(x):
            x = x.reshape(-1,1)
            return (self.v_cov @ (precision - precision @ x @ x.T @ precision)).flatten()
        grad_values[:,self.n_model_params:] = np.apply_along_axis(grad, 1, diff)
        return grad_values
