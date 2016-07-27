import theano
import theano.tensor as T
import numpy as np
import matplotlib.pyplot as plt # this package is used to generate plot of data

# generate toy data
num_points = 1000
data_set = []

for _ in xrange(num_points):
    x = np.random.uniform(low=0.0, high=6.28)
    y = 3 * x + 2
    data_set.append([x, y])

x_set = [p[0] for p in data_set]
y_set = [p[1] for p in data_set]

# define the symbolic variables we will use
x = T.fscalar('x')
y = T.fscalar('y')
W = theano.shared(value=0., name='W')
b = theano.shared(value=0., name='b')

# define the calculate procedure of y
y_d = W * x + b

# define loss
loss = (y_d - y) ** 2

# define the learning rate of sgd
lr = 0.001

# define the optimizer to reduce the loss
g_W = T.grad(loss, W)
g_b = T.grad(loss, b)
updates = [(W, W-lr*g_W), (b, b-lr*g_b)]

# define the function to optimize the model parameters
optimizer = theano.function(inputs=[x, y], outputs=[y_d, loss], updates=updates)

# run the optimizer to optimize the model parameters
max_epoches = 10
for i in xrange(max_epoches):
    y_pred = []
    total_loss = 0
    for idx in xrange(num_points):
        this_y, this_loss = optimizer(x_set[idx], y_set[idx])
        y_pred.append(this_y)
        total_loss += this_loss
    print "Loss of epoch "+str(i)+":", total_loss
    # plot the original data into figure
    data_fig = plt.figure('data_figure')
    plt.plot(x_set, y_set, 'ro', label='original_data')
    plt.plot(x_set, y_pred, 'go', label='pred_data')
    data_fig.show()
    raw_input()
    data_fig.clear()

print "W value:", W.get_value(), 'b value:', b.get_value()