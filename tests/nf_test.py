import bayes_lib as bl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

plnf_dist = bl.inference.variational.distributions.PlanarNormalizingFlow(2)
plnf_dist.initialize(2)
samples = plnf_dist.sample(1000)
print(samples.shape)
#X,Y = np.meshgrid(np.linspace(-3,3,50), np.linspace(-3,3,50))
#positions = np.vstack([X.ravel(), Y.ravel(), X.ravel(), Y.ravel()]).T
log_density = plnf_dist.log_density(samples)
plt.scatter(samples[:,4], samples[:,4], c = log_density)
#plt.contour(X, Y, log_density.reshape((50,50)))
plt.show()
