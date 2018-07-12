import pickle
import bayes_lib as bl
import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("results/synth_likelihood.csv", delimiter = ',')
synth_lhood = pickle.load(open("results/trained_lhood.pkl","rb"))
def ld(x):
    z = synth_lhood.predict(x.reshape(1,-1))
    print(z)
    return z
#log_density = lambda x: synth_lhood.predict(x.reshape(1,-1))
log_density = ld

with bl.model.Model() as m:
    ir1 = bl.rvs.Normal('ir1',np.log(0.01), np.log(1.1), transform = bl.rvs.transform.LowerBoundRVTransform(0))
    br = bl.rvs.Normal('br', np.log(0.5), np.log(1.1), transform = bl.rvs.transform.LowerBoundRVTransform(0))
    dr = bl.rvs.Normal('dr', np.log(1.0), np.log(1.1), transform = bl.rvs.transform.LowerBoundRVTransform(0))
    ir2 = bl.rvs.Normal('ir2', np.log(0.01), np.log(1.1), transform = bl.rvs.transform.LowerBoundRVTransform(0))
    y0 = [50, 100]
    obs_process = bl.rvs.SurrogateLikelihood("surrogate_lhood", log_density,[ir1,br,dr,ir2])
    #obs_process = LotkaVolterra2("lv", y0, ir1, br, dr, ir2, 30, observed = obs)
    print(m.log_density())
    
    print("Started sampling!")
    sampler = bl.inference.samplers.M_MVNMetropolisHastingsSpecial(m, scale = 1)
    chain, tchain = sampler.run(n_iters = 2000, init_params = np.log([0.02,0.3,0.85,0.01]))
    plt.hist(tchain[1000:,0])
    plt.show()
    #bl.utils.save_chain(tchain, "results/tchain.csv")
    #bl.utils.save_chain(chain, "results/chain.csv")
