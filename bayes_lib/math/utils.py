import autograd.numpy as agnp

def logit(x):
    return agnp.log(x/(1 - x))

def inv_logit(y):
    return 1/(1 + agnp.exp(-y))
