import numpy as np
import abc

class OptimizationResult(object):

    def __init__(self, position, objective_value, success, tolerance, iteration, optimizer):
        self.position = position
        self.objective_value = objective_value
        self.success = success
        self.iteration = iteration
        self.optimizer = optimizer

    def summary(self):
        return {'position': self.position, 'objective_value': self.objective_value, 'success' : self.success, 'iterations': self.iteration}

    def __str__(self):
        return self.summary().__str__()

class Optimizer(object):

    @abc.abstractmethod
    def run(self, objective_function, iter_func = None, iter_interval = 1):
        return

class GradientDescent(Optimizer):

    def __init__(self, learning_rate = 0.01, clip = None):
        self.learning_rate = learning_rate
        self.clip = clip

    def run(self, objective_fx, grad_fx, init, iter_func = None, iter_interval = 1, convergence = 1e-6, max_iters = 1e4):
        """Solves a minimization problem using the specified objective function, gradient function, and initialization.

        Args:
            objective_fx (f: Any -> float): A function taking an input and outputting a single real objective_value.
            grad_fx (f: Any -> Vector[float]): A function taking an input and outputting the gradient at that input.
            init (Any): An initial input position.
            convergence (float): A real number indicating the convergence tolerance.
            max_iters (int): The maximum number of iterations
            clip tuple(Maybe[None,float], Maybe[None,float]): The min and max to clip gradients, if None, will not clip

        Returns:
            An OptimizationResult object.
        """

        if self.clip is None:
            clip_fn = lambda x: x
        else:
            clip_fn = lambda x: np.clip(x, self.clip[0], self.clip[1])
        
        current_pos = init
        iteration = 0
        current_obj_value = objective_fx(init)
        converged = False

        while iteration < max_iters and not converged:
            if iter_func is not None and iteration % iter_interval == 0:
                iter_func(iteration, current_pos, current_obj_value)
            g = grad_fx(current_pos)
            new_pos = current_pos - self.learning_rate * clip_fn(g)
            new_obj_value = objective_fx(new_pos)
            iteration += 1
            #print("Grad:",g)
            if np.abs(new_obj_value - current_obj_value) < convergence:
                print("Converged by objective value")
                converged = True
            if np.sum(np.abs(new_pos - current_pos)) < convergence:
                print("Converged by absolute position change")
                converged = True
            current_pos = new_pos
            current_obj_value = new_obj_value
        
        print("Final Objective value: %f" % current_obj_value)
        opt_result = OptimizationResult(current_pos, current_obj_value, converged, convergence, iteration, self)
        return opt_result

class ADAM(Optimizer):

    def __init__(self, learning_rate = 0.01, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-8, clip = None):
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.clip = clip

    def run(self, objective_fx, grad_fx, init, iter_func = None, iter_interval = 1, convergence = 1e-6, max_iters = 1e4):
        """Solves a minimization problem using the specified objective function, gradient function, and initialization.

        Args:
            objective_fx (f: Any -> float): A function taking an input and outputting a single real objective_value.
            grad_fx (f: Any -> Vector[float]): A function taking an input and outputting the gradient at that input.
            init (Any): An initial input position.
            convergence (float): A real number indicating the convergence tolerance.
            max_iters (int): The maximum number of iterations
            clip tuple(Maybe[None,float], Maybe[None,float]): The min and max to clip gradients, if None, will not clip

        Returns:
            An OptimizationResult object.
        """

        if self.clip is None:
            clip_fn = lambda x: x
        else:
            clip_fn = lambda x: np.clip(x, self.clip[0], self.clip[1])
        
        current_pos = init
        iteration = 0
        current_obj_value = objective_fx(init)
        converged = False
        m = np.zeros(current_pos.shape)
        v = np.zeros(current_pos.shape)

        while iteration < max_iters and not converged:
            if iter_func is not None and iteration % iter_interval == 0:
                iter_func(iteration, current_pos, current_obj_value)
            g = clip_fn(grad_fx(current_pos))
            # compute moment estimates
            m = self.beta_1 * m + (1 - self.beta_1) * g
            v = self.beta_2 * v + (1 - self.beta_2) * g**2
            m_corr = m/(1 - self.beta_1)
            v_corr = v/(1 - self.beta_2)
            new_pos = current_pos - ((self.learning_rate)/(np.sqrt(v_corr) + self.epsilon))*m
            new_obj_value = objective_fx(new_pos)
            iteration += 1
            #print("Grad:",g)
            if np.abs(new_obj_value - current_obj_value) < convergence:
                print("Converged by objective value")
                converged = True
            if np.sum(np.abs(new_pos - current_pos)) < convergence:
                print("Converged by absolute position change")
                converged = True
            current_pos = new_pos
            current_obj_value = new_obj_value
        
        print("Final Objective value: %f" % current_obj_value)
        opt_result = OptimizationResult(current_pos, current_obj_value, converged, convergence, iteration, self)
        return opt_result



