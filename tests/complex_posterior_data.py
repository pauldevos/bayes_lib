import numpy as np
import pystan
import matplotlib.pyplot as plt
import pickle

def gen_data(theta1, theta2, theta3, theta4, theta5):

    s1 = theta3**2
    s2 = theta4**2

    rho = np.tanh(theta5)

    return np.random.multivariate_normal(np.array([theta1,theta2]), np.array([[s1, rho*s1*s2],[rho * s1 * s2, s2]]), size = 4)

x = gen_data(0.7, -2.9, -1, -0.9, 0.6)

stan_model_code = """

data {
    int N;
    vector[2] y[N];
}

parameters {
    real theta1;
    real theta2;
    real theta3;
    real theta4;
    real theta5;
}

transformed parameters {

    real s1;
    real s2;
    real rho;
    vector[2] mu;
    cov_matrix[2] S;

    s1 = pow(theta3, 2);
    s2 = pow(theta4, 2);
    rho = tanh(theta5);
    mu[1] = theta1;
    mu[2] = theta2;
    S[1,1] = s1;
    S[1,2] = rho * s1 * s2;
    S[2,1] = rho * s1 * s2;
    S[2,2] = s2;
}

model {
    
    theta1 ~ uniform(-3, 3);
    theta2 ~ uniform(-3, 3);
    theta3 ~ uniform(-3, 3);
    theta4 ~ uniform(-3, 3);
    theta5 ~ uniform(-3, 3);

    y ~ multi_normal(mu, S);
}
"""

stan_model = pystan.StanModel(model_code = stan_model_code)
results = stan_model.vb(data = {'N': 4, 'y' : x})
z = np.loadtxt(results['args']['sample_file'], delimiter = ',')
plt.scatter(z[:,2], z[:,3])
plt.show()
