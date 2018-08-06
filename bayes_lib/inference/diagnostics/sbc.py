import bayes_lib as bl
import numpy as np
import matplotlib.pyplot as plt

def sbc(model, inf_alg, n_runs, n_samples):
    
    res = []
    for i in range(n_runs):
        parameters, data = model.simulate(n_samples)
        samples = inf_alg.run(model, data)
        res.append(np.sum(samples < parameters))

    return res
