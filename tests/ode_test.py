import bayes_lib as bl
import numpy as np
from scipy.stats import norm
from scipy.integrate import odeint

import matplotlib.pyplot as plt
import seaborn as sns

# ODE Testing
class LKRV(bl.rvs.RandomVariable):

    def __init__(self, name, a, b, c, d, ts, y0, observed = None):
        super().__init__(name, observed = observed)
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.ts = ts
        self.y0 = y0
            
    def check_value(self, v):
        return True

    def log_density(self, v = None):
        a = bl.get_rv_value(self.a)
        b = bl.get_rv_value(self.b)
        c = bl.get_rv_value(self.c)
        d = bl.get_rv_value(self.d)
        def dX_dt(X, t):
            return np.array([a*X[0] - b*X[0]*X[1], -c*X[1] + d*b*X[0]*X[1]])
        
        lpdf = 0
        if v is None:
            X = odeint(dX_dt, self.y0, self.ts)
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    lpdf += norm.logpdf(self.value[i,j], X[i,j], 1)
            return lpdf

    def sample(self):
        a = bl.get_rv_value(self.a, s = True)
        b = bl.get_rv_value(self.b, s = True)
        c = bl.get_rv_value(self.c, s = True)
        d = bl.get_rv_value(self.d, s = True)
        def dX_dt(X, t):
            return np.array([a*X[0] - b*X[0]*X[1], -c*X[1] + d*b*X[0]*X[1]])
        X = odeint(dX_dt, self.y0, self.ts)
        return X

ts = np.linspace(0, 15, 15)
y0 = np.array([10,15])
a = 1
b = 0.1
c = 1.5
d = 0.75
def dX_dt(X, t):
    return np.array([a*X[0] - b*X[0]*X[1], -c*X[1] + d*b*X[0]*X[1]])

data = odeint(dX_dt, y0, ts) + np.random.normal(0, 1, size = (15,2))

with bl.model.Model() as m:
    a = bl.rvs.Normal('a', 0, 2, transform = bl.rvs.transform.LowerBoundRVTransform(0))
    b = bl.rvs.Normal('b', 0, 2, transform = bl.rvs.transform.LowerBoundRVTransform(0))
    c = bl.rvs.Normal('c', 1, 2, transform = bl.rvs.transform.LowerBoundRVTransform(0))
    d = bl.rvs.Normal('d', 0, 2, transform = bl.rvs.transform.LowerBoundRVTransform(0))
    y = LKRV('LK', a, b, c, d, ts, y0, observed = data)
    
    sampler = bl.inference.samplers.A_MVNMetropolisHastings(m)
    chain, tchain = sampler.run(n_iters = 5000)
    plt.hist(tchain[1000:,1])
    plt.show()
