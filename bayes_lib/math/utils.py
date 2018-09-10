import tensorflow as tf

def logit(x):
    return tf.log(x/(1 - x))

def inv_logit(y):
    return 1/(1 + tf.exp(-y))

def grad_inv_logit(y):
    t = inv_logit(y)
    return t * (1 - t)

def relu(y):
    return y * (y > 0)

def linear(y):
    return y

def exp(y):
    return tf.exp(y)

def tanh(y):
    return tf.tanh(y)

def grad_tanh(y):
    return 1 - (tf.tanh(y)**2)

# Aliases
sigmoid = inv_logit
grad_sigmoid = grad_inv_logit


