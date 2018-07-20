import bayes_lib as bl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import autograd

plnf_dist = bl.inference.variational.distributions.PlanarNormalizingFlow(2)
plnf_dist.initialize(2)
samples = plnf_dist.sample(1000)
#X,Y = np.meshgrid(np.linspace(-3,3,50), np.linspace(-3,3,50))
#positions = np.vstack([X.ravel(), Y.ravel(), X.ravel(), Y.ravel()]).T
log_densities = plnf_dist.log_density(samples)
grad_log_densities = plnf_dist.grad_log_density(samples)
print(grad_log_densities.shape)
#log_densities = np.array([plnf_dist.log_density(vparams, samples[i,:]) for i in range(1000)])[:,0]
#g = autograd.grad(plnf_dist.log_density)

#plt.scatter(samples[:,2], samples[:,3], c = log_densities)
#plt.contour(X, Y, log_density.reshape((50,50)))
#plt.show()
