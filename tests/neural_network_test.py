import bayes_lib as bl
import autograd
import autograd.numpy as agnp
import autograd.scipy as agsp
import matplotlib.pyplot as plt

x_true = agnp.linspace(-6, 6, 1000).reshape(1000,1)
y_true = agnp.sin(x_true)

with bl.Model() as m:
    X = bl.Placeholder('X', dimensions = agnp.array([1000,1]))
    network = bl.ml.neural_network.DenseNeuralNetwork('NN', X, layer_dims = [1,5,1], nonlinearity = bl.math.utils.sigmoid)
    y = bl.rvs.Normal('obs', network, 1, observed = y_true)

    optimizer = bl.math.optimizers.GradientDescent(learning_rate = 1e-6)
   
    init = agnp.random.normal(0, 3, size = m.n_params)
    plt.ion()
    ax = plt.gca()
    ax.set_autoscale_on(True)
    line_true, = ax.plot(x_true, y_true)
    line, = ax.plot(x_true,network.compute(init, x_true)[0,:,:])
    
    def iter_func(t, p, o):
        print(t)
        print("Objective Value: %f" % o)
        line.set_ydata(network.compute(p, x_true)[0,:,:])
        ax.relim()
        ax.autoscale_view(True, True, True)
        plt.draw()
        plt.pause(0.1)

    res = optimizer.run(lambda x: -m.log_density(x, feed_dict = {X: x_true}), lambda x: -m.grad_log_density(x, feed_dict = {X: x_true}), init, iter_func = iter_func, iter_interval = 5)



    
