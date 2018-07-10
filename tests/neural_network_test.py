import bayes_lib as bl
import numpy as np
import autograd.numpy as agnp
import autograd
import matplotlib.pyplot as plt

x = np.linspace(-10, 10, 100).reshape((100,1))
y = np.cos(x)

nn = bl.ml.neural_network.DenseNeuralNetwork([1,10,1])
optimizer = bl.math.optimizers.GradientDescent(learning_rate = 0.1)

def loss(params):
    return agnp.mean(agnp.square(y - nn.predict_p(params, x)))

grad_loss = autograd.elementwise_grad(loss)

plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111)
line1, = ax.plot(x, y, 'b-')
line2, = ax.plot(x, nn.predict_p(nn.get_params(), x)[0,:,:], 'r-')
def update_plot(t, current_pos, current_obj_value):
    line2.set_ydata(nn.predict_p(current_pos, x)[0,:,:])
    fig.canvas.draw()
    plt.pause(0.05)
    print("Current Objective Value: %f" % current_obj_value) 

optimizer.run(loss, grad_loss, init = nn.get_params(), iter_func = update_plot, iter_interval = 100)


