import numpy as np
import tensorflow as tf
import abc
import autograd.numpy as agnp
import autograd
from bayes_lib.math.utils import sigmoid, grad_sigmoid, tanh, grad_tanh

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

    @abc.abstractmethod
    def entropy(self, variational_params, variational_samples):
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

    @abc.abstractmethod
    def grad_entropy(self, variational_params, variational_samples):
        return

class ReparameterizableVariationalDistribution(VariationalDistribution):

    is_reparameterizable = True

    @abc.abstractmethod
    def sample_p(self, variational_params, n_samples):
        return

"""
class PlanarNormalizingFlow(ReparameterizableVariationalDistribution, DifferentiableVariationalDistribution):

    def __init__(self, n_layers, shared_params = False):
        self.n_layers = n_layers
        shared_params = shared_params
        self.grad_log_density_p = autograd.grad(self.log_density_p)

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
        return agnp.abs((grad_tanh(z @ w + b).reshape(-1,1) * w) @ u)

    def transform(self, w, b, u, z):
        return z + u * tanh(z @ w + b).reshape(-1,1)
    
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
    
    def sample_p(self, variational_params, n_samples):
        nd = self.n_model_params * self.n_layers
        us = variational_params[:nd]
        ws = variational_params[nd:nd*2]
        bs = variational_params[nd*2:]

        z0 = np.random.normal(0, 1, size = (n_samples, self.n_model_params))
        zs = [z0]
        it = 0
        for i in range(self.n_layers):
            zs.append(self.transform(ws[i*(self.n_model_params):(i+1) * self.n_model_params],
                                     bs[i:i + 1],
                                     us[i*(self.n_model_params):(i+1) * self.n_model_params],
                                     zs[i-1]))
        return zs[-1]

    def entropy(self, variational_params, variational_samples):
        return 0.5 * self.n_model_params * (1.0 + agnp.log(2 * agnp.pi)) + agnp.sum(self.n_model_params)

    def grad_entropy(self, variational_params, variational_samples):
        return 0

    def log_density(self, variational_samples):
        return agnp.array([self.log_density_p(self.variational_params, variational_samples[i,:]) for i in range(variational_samples.shape[0])])[:,0]

    def log_density_p(self, variational_params, zs):
        nd = self.n_model_params * self.n_layers
        us = variational_params[:nd]
        ws = variational_params[nd:nd*2]
        bs = variational_params[nd*2:]
        z0 = zs[:self.n_model_params]
        lpdfs = agnp.sum(agsp.stats.norm.logpdf(z0, 0, 1)) 
        for k in range(self.n_layers):
            zk = zs[k*(self.n_model_params):(k+1)*self.n_model_params]
            lpdfs += self.log_det_jacobian(ws[k*(self.n_model_params):(k+1) * self.n_model_params],
                                           bs[k:k + 1],
                                           us[k*(self.n_model_params):(k+1) * self.n_model_params],
                                           zk)
        return lpdfs
    
    def grad_log_density(self, variational_samples):
        return agnp.array([self.grad_log_density_p(self.variational_params, variational_samples[i,:]) for i in range(variational_samples.shape[0])])

"""
class MeanField(object):

    is_differentiable = True
    is_reparameterizable = True

    def initialize(self, model, n_mc_samples = 1, n_grad_samples = 1):
        self.model = model
        self.n_model_params = self.model.n_params
        self.v_means = tf.placeholder(tf.float32, [self.model.n_params])
        self.v_stds = tf.placeholder(tf.float32, [self.model.n_params])
        self.compile_(n_mc_samples, n_grad_samples)

    def get_variational_params(self):
        return [self.v_means, self.v_stds]

    def compile_(self, n_mc_samples, n_grad_samples):
        self.mc_rng = tf.random_normal([n_mc_samples, self.model.n_params])
        with tf.control_dependencies([self.mc_rng]):
            self.s_op = self.sample_(self.mc_rng)
        self.g_rng = tf.random_normal([n_grad_samples, self.model.n_params])
        with tf.control_dependencies([self.g_rng]):
            self.s_op_g = self.sample_(self.g_rng)
            self.g_s_op = [self.grad_sample_(self.g_rng[i,:]) for i in range(n_grad_samples)]
    
    def sample_(self, rng):
        return self.v_means + rng * tf.exp(self.v_stds)

    def grad_sample_(self, rng):
        return tf.stack(tf.gradients(self.sample_(rng), [self.v_means, self.v_stds]))

    def sample(self, variational_params, n_samples):
        rng = tf.random_normal([n_samples, self.model.n_params])
        return self.model.sess.run(self.sample_(rng), feed_dict = {self.v_means : variational_params[:self.model.n_params], self.v_stds: variational_params[self.model.n_params:]})
 
    def sample_p(self, variational_params):
        return self.model.sess.run([self.mc_rng, self.s_op], feed_dict = {self.v_means : variational_params[:self.model.n_params], self.v_stds : variational_params[self.model.n_params:]})

    def grad_sample_p(self, variational_params):
        return self.model.sess.run([self.s_op_g, self.g_s_op], feed_dict = {self.v_means : variational_params[:self.model.n_params], self.v_stds : variational_params[self.model.n_params:]})   
    def entropy(self, variational_params, variational_samples):
        v_stds = variational_params[self.n_model_params:]
        return 0.5 * self.n_model_params * (1.0 + np.log(2 * np.pi)) + np.sum(v_stds)

    def grad_entropy(self, variatianal_params, variational_samples):
        return np.hstack([np.zeros(self.n_model_params), np.ones(self.n_model_params)])

    def entropy_op(self):
        return 0.5 * self.n_model_params * (1.0 + tf.log(2 * np.pi)) + tf.reduce_sum(self.v_stds)

    def feed_dict_update(self, feed_dict, variational_params):
        if feed_dict is None:
            feed_dict = {}

        v_dict = {self.v_means: variational_params[self.n_model_params:],
                  self.v_stds : variational_params[:self.n_model_params]}
        v_dict.update(feed_dict)
        return v_dict

class FullRank(object):

    is_differentiable = True
    is_reparameterizable = True

    def initialize(self, model, n_mc_samples = 1, n_grad_samples = 1):
        self.model = model
        self.n_model_params = self.model.n_params
        self.v_means = tf.placeholder(tf.float32, [self.model.n_params])
        self.v_cov_sqrt = tf.placeholder(tf.float32, [self.model.n_params*self.model.n_params])
        self.compile_(n_mc_samples, n_grad_samples)

        self.grad_entropy_ = autograd.grad(self.entropy)

    def compile_(self, n_mc_samples, n_grad_samples):
        self.mc_rng = tf.random_normal([n_mc_samples, self.model.n_params])
        with tf.control_dependencies([self.mc_rng]):
            self.s_op = self.sample_(self.mc_rng)
        self.g_rng = tf.random_normal([n_grad_samples, self.model.n_params])
        with tf.control_dependencies([self.g_rng]):
            self.s_op_g = self.sample_(self.g_rng)
            self.g_s_op = [self.grad_sample_(tf.expand_dims(self.g_rng[i,:],0)) for i in range(n_grad_samples)]
    
    def sample_(self, rng):
        v_cov_sqrt_ = tf.reshape(self.v_cov_sqrt, [self.n_model_params, self.n_model_params])
        L = tf.cholesky(tf.matmul(tf.transpose(v_cov_sqrt_), v_cov_sqrt_))
        return self.v_means + tf.matmul(rng,L)

    def grad_sample_(self, rng):
        return tf.concat(tf.gradients(self.sample_(rng), [self.v_means, self.v_cov_sqrt]), 0)

    def sample(self, variational_params, n_samples):
        rng = tf.random_normal([n_samples, self.model.n_params])
        return self.model.sess.run(self.sample_(rng), feed_dict = {self.v_means : variational_params[:self.model.n_params], self.v_cov_sqrt: variational_params[self.model.n_params:]})
 
    def sample_p(self, variational_params):
        return self.model.sess.run([self.mc_rng, self.s_op], feed_dict = {self.v_means : variational_params[:self.model.n_params], self.v_cov_sqrt : variational_params[self.model.n_params:]})

    def grad_sample_p(self, variational_params):
        return self.model.sess.run([self.s_op_g, self.g_s_op], feed_dict = {self.v_means : variational_params[:self.model.n_params], self.v_cov_sqrt : variational_params[self.model.n_params:]})   
    
    def entropy(self, variational_params, variational_samples):
        v_cov_sqrt = variational_params[self.n_model_params:].reshape((self.n_model_params, self.n_model_params))
        cov = agnp.dot(v_cov_sqrt.T, v_cov_sqrt)
        return 0.5 * self.n_model_params * (1.0 + agnp.log(2 * agnp.pi)) + 0.5 * agnp.log(agnp.linalg.det(cov))

    def grad_entropy(self, variational_params, variational_samples):
        return self.grad_entropy_(variational_params, variational_samples)

"""
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

    #Methods to specify reparameterizable
    def sample_p(self, variational_params, n_samples):
        v_means = variational_params[:self.n_model_params]
        v_cov_sqrt = variational_params[self.n_model_params:].reshape((self.n_model_params, self.n_model_params))
        v_cov = agnp.dot(v_cov_sqrt.T,v_cov_sqrt) 
        print(v_cov)
        L = agnp.linalg.cholesky(v_cov)
        v_samples = v_means + agnp.dot(agnp.random.randn(n_samples, self.n_model_params), L)
        return v_samples

    def entropy(self, variational_params, variational_samples):
        v_cov = variational_params[self.n_model_params:].reshape((self.n_model_params, self.n_model_params))
        cov = agnp.dot(v_cov.T, v_cov)
        return 0.5 * self.n_model_params * (1.0 + agnp.log(2 * agnp.pi)) + 0.5 * agnp.log(agnp.linalg.det(cov))
    """

