import bayes_lib as bl
import numpy as np

data = np.random.normal(3, 1, size = 30)

def objective_function(mu):
    return np.sum(((data - mu)**2)/2) + np.sum((mu/2)**2)

def grad_objective_function(mu):
    return -np.sum((data - mu)) + np.sum(mu)

with bl.Model() as m:
    mu = bl.rvs.Normal('mu_prior', 0, 1)
    y = bl.rvs.Normal('obs_model', mu, 1, observed = data)
    z = m.get_param_vector()
    print(z)
    optimizer = bl.math.optimizers.GradientDescent(learning_rate = 0.05)
    print(m.grad_log_density_p(z))
    print(grad_objective_function(z))
    print(optimizer.run(lambda x: - m.log_density_p(x), lambda x: -m.grad_log_density_p(x), init = z))
    print(z)
    print(optimizer.run(objective_function, grad_objective_function, init = z))


