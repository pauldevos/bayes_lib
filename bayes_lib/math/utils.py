import autograd.numpy as agnp

def logit(x):
    return agnp.log(x/(1 - x))

def inv_logit(y):
    return 1/(1 + agnp.exp(-y))

def grad_inv_logit(y):
    t = inv_logit(y)
    return t * (1 - t)

# Aliases
sigmoid = inv_logit
grad_sigmoid = grad_inv_logit


