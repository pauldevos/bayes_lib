import numpy as np

def logit(x):
    return np.log(x/(1 - x))

def inv_logit(y):
    return 1/(1 + np.exp(-y))
