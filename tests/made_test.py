import bayes_lib as bl
import autograd
import autograd.numpy as agnp
import autograd.scipy as agsp
import matplotlib.pyplot as plt

x_true = agnp.linspace(-6, 6, 1000).reshape(1000,1)
y_true = agnp.sin(x_true)

with bl.Model() as m:
    gaussian_made = bl.ml.made.MADE('made', [3,5,5])
    print(gaussian_made.masks)
