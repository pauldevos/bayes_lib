import bayes_lib as bl
import numpy as np
import autograd.numpy as agnp
import autograd
import matplotlib.pyplot as plt
import matplotlib.animation
import pickle

z = np.loadtxt("results/synth_likelihood.csv", delimiter = ',')
x = z[:,:4]
y = z[:,4]
y = bl.math.utils.exp((y - np.mean(y))/np.std(y))

nn = bl.ml.neural_network.DenseNeuralNetwork([4,3,1], last_layer_nonlinearity = bl.math.utils.exp)
optimizer = bl.math.optimizers.GradientDescent(learning_rate = 0.01)

def loss(params):
    return agnp.mean(agnp.square(y - nn.predict_p(params, x)))

grad_loss = autograd.elementwise_grad(loss)
plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlim(0,300)
ax.set_ylim(0,500000)
"""
line1, = ax.plot(x, y, 'b-')
line2, = ax.plot(x, nn.predict_p(nn.get_params(), x)[0,:,:], 'r-')
def update_plot(t, current_pos, current_obj_value):
    line2.set_ydata(nn.predict_p(current_pos, x)[0,:,:])
    fig.canvas.draw()
    plt.pause(0.05)
    print("Current Objective Value: %f" % current_obj_value) 
"""
ts = []
obj_vals = []
ax.plot(ts, obj_vals)
#sc = ax.scatter(ts, obj_vals)
def update_plot(t, current_pos, current_obj_value):
    #ax.clear()
    #ts.append(t)
    #obj_vals.append(current_obj_value)
    #sc.set_offsets(np.c_[ts,obj_vals])
    #ax.plot(ts, obj_vals)
    #fig.canvas.draw()
    #plt.pause(0.05)
    print("Current Objective Value: %f" % current_obj_value)

res = optimizer.run(loss, grad_loss, init = nn.get_params(), iter_func = update_plot, iter_interval = 100, convergence = 1e-10)
print(nn.predict(x))
nn.set_params(res.position)
print(nn.predict(x))
pickle.dump(nn, open('results/trained_lhood.pkl', 'wb'))


