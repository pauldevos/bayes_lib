import numpy as np
import autograd.scipy as agsp
import autograd.numpy as agnp
import autograd
import abc
from bayes_lib.math.utils import sigmoid, grad_sigmoid

class VariationalDistributionNotDifferentiableException(Exception):
    pass

class VariatioanlDistributionNotReparameterizableException(Exception):
    pass

class VariationalDistribution(object):
    
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

class DifferentiableVariationalDistribution(VariationalDistribution):

    is_differentiable = True 
    
    @abc.abstractmethod
    def log_density(self, variational_samples):
        return

    def grad_log_density(self, variational_samples):
        if not is_differentiable:
            raise VariationalDistributionNotDifferentiableException
        return

class ReparameterizableVariationalDistribution(VariationalDistribution):

    is_reparameterizable = True

    @abc.abstractmethod
    def sample_p(self, variataional_params, n_samples):
        return

    @abc.abstractmethod
    def grad_variational_params(self, variational_params, variational_samples):
        return

    @abc.abstractmethod
    def entropy(self, variational_params):
        return

    @abc.abstractmethod
    def grad_entropy(self, variational_params, variational_samples):
        return

class PlanarNormalizingFlow(VariationalDistribution):

    is_differentiable = True
    
    def __init__(self, n_layers, shared_params = False):
        self.n_layers = n_layers
        shared_params = shared_params

    def initialize(self, n_model_params, init = None):
        self.n_model_params = n_model_params
        if init is None:
            us = np.random.uniform(-10,10, size = (self.n_model_params, self.n_layers)).flatten()
            ws = np.random.uniform(-10,10, size = (self.n_model_params, self.n_layers)).flatten()
            bs = np.random.normal(0, 10, size = (self.n_layers))
            self.variational_params = np.hstack([us, ws, bs])
        else:
            self.variational_params = init

    @property
    def us(self):
        return self.__us

    @property
    def ws(self):
        return self.__ws

    @property
    def bs(self):
        return self.__bs

    @us.setter
    def us(self, us):
        self.__us = us

    @ws.setter
    def ws(self, ws):
        self.__ws = ws

    @bs.setter
    def bs(self, bs):
        self.__bs = bs

    def log_det_jacobian(self, w, b, u, z):
        return agnp.abs((grad_sigmoid(z @ w + b).reshape(-1,1) * w) @ u)

    def transform(self, w, b, u, z):
        return z + u * sigmoid(z @ w + b).reshape(-1,1)
    
    @VariationalDistribution.variational_params.setter
    def variational_params(self, variational_params):
        nd = self.n_model_params * self.n_layers
        self.us = variational_params[:nd]
        self.ws = variational_params[nd:nd*2]
        self.bs = variational_params[nd*2:]
        self._variational_params = variational_params

    def sample(self, n_samples):
        z0 = np.random.normal(0, 1, size = (n_samples, self.n_model_params))
        zs = [z0]
        it = 0
        for i in range(self.n_layers):
            zs.append(self.transform(self.ws[i*(self.n_model_params):(i+1) * self.n_model_params],
                                     self.bs[i:i + 1],
                                     self.us[i*(self.n_model_params):(i+1) * self.n_model_params],
                                     zs[i-1]))

        return np.hstack(zs)

    def log_density(self, variational_samples):
        z0 = variational_samples[:, :self.n_model_params]
        lpdfs = agnp.sum(agsp.stats.norm.logpdf(z0), axis = 1) 
        for k in range(self.n_layers):
            zk = variational_samples[:, k*(self.n_model_params):(k+1)*self.n_model_params]
            lpdfs += self.log_det_jacobian(self.ws[k*(self.n_model_params):(k+1) * self.n_model_params],
                                           self.bs[k:k + 1],
                                           self.us[k*(self.n_model_params):(k+1) * self.n_model_params],
                                           zk)
        return lpdfs

class MeanField(DifferentiableVariationalDistribution, ReparameterizableVariationalDistribution):

    is_differentiable = True
    is_reparameterizable = True

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
        self.__v_stds = agnp.exp(v_stds)

    @VariationalDistribution.variational_params.setter
    def variational_params(self, variational_params):
        self.v_means = variational_params[:self.n_model_params]
        self.v_stds = variational_params[self.n_model_params:]
        self._variational_params = variational_params

    def sample(self, n_samples):
        v_samples = np.random.normal(0, 1, size = (n_samples, self.n_model_params))
        v_samples = self.v_means + v_samples * self.v_stds
        return v_samples
    """
    Methods to specify that log_density of variational distribution is directly differentiable
    """
    def log_density(self, variational_samples):
        return agsp.stats.multivariate_normal.logpdf(variational_samples, self.v_means, np.diag(self.v_stds))

    def grad_log_density(self, variational_samples):
        grad_values = np.zeros((variational_samples.shape[0], self.n_model_params * 2))
        v_vars = self.v_stds**2
        diffs = variational_samples - self.v_means
        grad_values[:,:self.n_model_params] = diffs/v_vars
        grad_values[:,self.n_model_params:] = -1 + (diffs**2)/v_vars
        return grad_values

    """
    Methods to specify that log_density is reparameterizable
    """
    def sample_p(self, variational_params, n_samples):
        v_means, v_stds = variational_params[:self.n_model_params], agnp.exp(variational_params[self.n_model_params:])
        return v_means + v_stds * agnp.random.normal(0, 1, size = (n_samples, self.n_model_params))

    def entropy(self, variational_params, variational_samples):
        v_stds = variational_params[self.n_model_params:]
        return 0.5 * self.n_model_params * (1.0 + agnp.log(2 * agnp.pi)) + agnp.sum(v_stds)

class FullRank(DifferentiableVariationalDistribution, ReparameterizableVariationalDistribution):

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
        grad_values[:,self.n_model_params:] = -np.apply_along_axis(grad, 1, diff)
        return grad_values

    """
    Methods to specify reparameterizable
    """
    def sample_p(self, variational_params, n_samples):
        v_means = variational_params[:self.n_model_params]
        v_cov_sqrt = variational_params[self.n_model_params:].reshape((self.n_model_params, self.n_model_params))
        v_cov = agnp.dot(v_cov_sqrt.T,v_cov_sqrt)
        L = agnp.linalg.cholesky(v_cov)
        v_samples = v_means + agnp.dot(agnp.random.randn(n_samples, self.n_model_params), L)
        return v_samples

    def entropy(self, variational_params, variational_samples):
        v_cov = variational_params[self.n_model_params:].reshape((self.n_model_params, self.n_model_params))
        cov = agnp.dot(v_cov.T, v_cov)
        return 0.5 * self.n_model_params * (1.0 + agnp.log(2 * agnp.pi)) + 0.5 * agnp.log(agnp.linalg.det(cov))




