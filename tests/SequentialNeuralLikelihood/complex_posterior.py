import bayes_lib as bl
import numpy as np
import matplotlib.pyplot as plt

def gen_data(theta1, theta2, theta3, theta4, theta5):

    s1 = theta3**2
    s2 = theta4**2

    rho = np.tanh(theta5)

    return np.random.multivariate_normal(np.array([theta1,theta2]), np.array([[s1, rho * s1 * s2],[rho * s1 * s2, s2]]), size = 4)

x = gen_data(0.7, -2.9, -1, -0.9, 0.6)

with bl.Model() as m:
    theta1 = bl.rvs.Uniform('theta1',-3,3)
    theta2 = bl.rvs.Uniform('theta2',-3,3)
    theta3 = bl.rvs.Uniform('theta3',-3,3)
    theta4 = bl.rvs.Uniform('theta4',-3,3)
    theta5 = bl.rvs.Uniform('theta5',-3,3)

    s1 = bl.ops.square('s1',theta3)
    s2 = bl.ops.square('s2',theta4)
    rho = bl.ops.tanh('rho',theta5)

    s12 = bl.ops.square('s12',s1)
    s22 = bl.ops.square('s22',s2)
    cor = bl.ops.multiply('p12',[rho,s1,s2])
    
    mu = bl.ops.make_vector('mu',[theta1,theta2],2)
    cov = bl.ops.make_covariance('cov',[s12,cor,cor,s22],2)
    
    y = bl.rvs.Multivariate_Normal('obs', mu, cov, observed = x)

    def iter_func(t, p, o):
        print("Objective Value: %f" % (o))
    #inf_alg = bl.inference.variational.ReparameterizedVariationalInference(m)#, variational_dist = bl.inference.variational.distributions.PlanarNormalizingFlow(6))
    #res, v_dist = inf_alg.run(iter_func = iter_func, iter_interval = 1)
    #samples = v_dist.sample(1000)
    #print(samples.shape)
    #plt.scatter(samples[:,32], samples[:,33])
    #plt.scatter(samples[:,2], samples[:,3])
    chains = []
    tchains = []
    for i in range(6):
        inf_alg = bl.inference.samplers.A_MVNMetropolisHastings(m)
        chain, tchain = inf_alg.run(n_iters = 5000)
        chains.append(chain[3000:])
        tchains.append(tchain[3000:])
    
    chain = np.vstack(chains)
    tchain = np.vstack(tchains)
    plt.scatter(tchain[:,2], tchain[:,3])
    plt.show()
