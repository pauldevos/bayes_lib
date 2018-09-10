import tensorflow as tf
import bayes_lib as bl

import numpy as np
import threading
import abc

class DuplicateNodeException(Exception):
    pass

class ModelNotDifferentiableException(Exception):
    pass

class Context(object):

    contexts = threading.local()
    
    # Attach self to context stack
    def __enter__(self):
        type(self).get_contexts().append(self)
        return self
    
    # Remove self from context stack
    def __exit__(self, typ, value, traceback):
        type(self).get_contexts().pop()
    
    @classmethod
    def get_context(cls):
        try:
            return cls.get_contexts()[-1]
        except:
            raise TypeError("No context!")

    @classmethod
    def get_contexts(cls):
        if not hasattr(cls.contexts, 'stack'):
            cls.contexts.stack = []
        return cls.contexts.stack


class Model(Context):

    def __init__(self):
        self.random_variables = []
        self.unobserved_random_variables = []
        self.n_params_ = 0
        self.ld_ = None
        self.grad_ld_ = None
        self.sess = tf.Session()

    def reset(self):
        self.sess.close()
        tf.reset_default_graph()
        self.sess = tf.Session()
        self.ld_ = None
        self.grad_ld_ = None

    def add_random_variable(self, rv):
        self.random_variables.append(rv)
        if not rv.is_observed:
            self.unobserved_random_variables.append(rv)
            self.n_params_ += 1

    @property
    def n_params(self):
        return self.n_params_

    def _compile(self):
        self.log_density_fns = [rv.log_density() for rv in self.random_variables]
        self.param_placeholder_ = [tf.placeholder(tf.float32) for _ in self.unobserved_random_variables]
        self.assign_fns_ = [rv.transform_assign(self.param_placeholder_[i]) for i, rv in enumerate(self.unobserved_random_variables)]
        self.transform_fns_ = [rv.transform_value(self.param_placeholder_[i]) for i, rv in enumerate(self.unobserved_random_variables)]
        with tf.control_dependencies(self.assign_fns_):
            self.ld_ = tf.reduce_sum(self.log_density_fns)

    def _compile_grad(self):
        if self.ld_ is None:
            self._compile()
        with tf.control_dependencies(self.assign_fns_):
            self.grad_ld_ = tf.gradients(self.ld_, self.unobserved_random_variables)

    def log_density(self, parameters, feed_dict = {}):
        if self.ld_ is None:
            self._compile()
            self.sess.run(tf.global_variables_initializer())

        if feed_dict is None:
            feed_dict = {}

        #self.sess.run(self.assign_fns_, feed_dict = dict(zip(self.param_placeholder_, parameters)))
        feed_dict.update(dict(zip(self.param_placeholder_, parameters)))
        self.sess.run(self.assign_fns_, feed_dict = feed_dict)
        res = self.sess.run(self.assign_fns_ + [self.ld_], feed_dict = feed_dict)
        return res[-1]

    def log_density_p(self, parameters):
        if self.ld_ is None:
            self._compile()
            self.sess.run(tf.global_variables_initializer())
        
        assign_fns = [rv.transform_assign(parameters[i]) for i, rv in enumerate(self.unobserved_random_variables)]
        with tf.control_dependencies(assign_fns):
            ld_ = tf.reduce_sum(self.log_density_fns)
        return ld_
    
    def grad_log_density(self, parameters, feed_dict = {}):
        if self.grad_ld_ is None:
            self._compile_grad()
            self.sess.run(tf.global_variables_initializer())

        if feed_dict is None:
            feed_dict = {}

        #self.sess.run(self.assign_fns_, feed_dict = dict(zip(self.param_placeholder_, parameters)))
        feed_dict.update(dict(zip(self.param_placeholder_, parameters)))
        res = self.sess.run(self.assign_fns_ + [self.grad_ld_], feed_dict = feed_dict)
        return res[-1]

    def get_log_density_op(self):
        if self.ld_ is None:
            self._compile()
            self.sess.run(tf.global_variables_initializer())
        return self.ld_

    def transform_param(self, parameter):
        if self.ld_ is None:
            self._compile()
            self.sess.run(tf.global_variables_initializer())
        return self.sess.run(self.transform_fns_, feed_dict = dict(zip(self.param_placeholder_, parameter)))
