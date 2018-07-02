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
    def run(self, objective_function):
        return

class GradientDescent(Optimizer):

    def __init__(self, learning_rate = 0.1):
        self.learning_rate = learning_rate

    def run(self, objective_fx, grad_fx, init, convergence = 1e-6, max_iters = 1e4):
        """Solves a minimization problem using the specified objective function, gradient function, and initialization.

        Args:
            objective_fx (f: Any -> float): A function taking an input and outputting a single real objective_value.
            grad_fx (f: Any -> Vector[float]): A function taking an input and outputting the gradient at that input.
            init (Any): An initial input position.
            convergence (float): A real number indicating the convergence tolerance.
            max_iters (int): The maximum number of iterations

        Returns:
            An OptimizationResult object.
        """
        
        current_pos = init
        iteration = 0
        current_obj_value = objective_fx(init)
        converged = False

        while iteration < max_iters and not converged:
            g = grad_fx(current_pos)
            new_pos = current_pos - self.learning_rate * g
            new_obj_value = objective_fx(new_pos)
            iteration += 1
            #print("Grad:",g)
            print("New Pos:",new_pos)
            print("New Obj:",new_obj_value)
            if np.abs(new_obj_value - current_obj_value) < convergence:
                converged = True
            if np.sum(np.abs(new_pos - current_pos)) < convergence:
                converged = True
            current_pos = new_pos
            current_obj_value = new_obj_value
        
        opt_result = OptimizationResult(current_pos, current_obj_value, converged, convergence, iteration, self)
        return opt_result


