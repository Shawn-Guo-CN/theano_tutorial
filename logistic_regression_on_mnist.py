import theano
import theano.tensor as T
import numpy as np

import cPickle
import gzip
import time

def shared_dataset(data_xy):
    """
    Change the input data into theano shared variable
    :param data_xy: tuple of input data
    :return: tuple of shared data
    """
    data_x, data_y = data_xy
    shared_x = theano.shared(np.asarray(data_x, dtype=theano.config.floatX))
    shared_y = theano.shared(np.asarray(data_y, dtype='int32'))
    return shared_x, shared_y

def load_data(file_name):
    """
    Load data from mnist data set and turn it into theano shared variable.
    :param file_name: the path and name of original data
    :return: list of three tuples, [(train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)]
    """
    print "...loading data"

    with gzip.open(file_name, 'rb') as f:
        train_set, test_set = cPickle.load(f)

    # test_set_x, test_set_y = shared_dataset(test_set)
    # train_set_x, train_set_y = shared_dataset(train_set)
    # rval = [(train_set_x, train_set_y), (test_set_x, test_set_y)]
    rval = [train_set, test_set]
    return rval

def run(data_file):
    # load data
    datasets = load_data(data_file)
    train_set_x, train_set_y = datasets[0]
    test_set_x, test_set_y = datasets[1]

    # define the symbolic variables we will use
    x = T.vector('x')
    y = T.iscalar('y')
    W = theano.shared(value=np.random.uniform(-0.01, 0.01, size=(784, 10)), name='W')
    b = theano.shared(value=np.random.uniform(-0.01, 0.01, size=(10, )), name='b')

    # define the calculate procedure of y
    y_given_x = T.nnet.softmax(T.dot(x, W) + b)
    y_d = T.argmax(y_given_x)

    # define loss
    loss = -T.log(y_given_x[0])[y]

    # get_y_given_x = theano.function(inputs=[x], outputs=y_given_x)
    # get_y_d = theano.function(inputs=[x], outputs=y_d)
    # get_loss = theano.function(inputs=[x, y], outputs=loss)
    # print get_y_given_x(train_set_x[0])
    # print get_y_d(train_set_x[0])
    # print get_loss(train_set_x[0], train_set_y[0])
    # exit(0)
#
    # define the learning rate of sgd
    lr = 0.0001

    # define the optimizer to reduce the loss
    g_W = T.grad(loss, W)
    g_b = T.grad(loss, b)
    updates = [(W, W - lr * g_W), (b, b - lr * g_b)]

    # define the function to optimize the model parameters
    optimizer = theano.function(inputs=[x, y], outputs=[y_d, loss], updates=updates)

    # run the optimizer to optimize the model parameters
    max_epoches = 10
    for i in xrange(max_epoches):
        y_pred = []
        total_loss = 0
        for idx in xrange(train_set_x.shape[0]):
            this_y, this_loss = optimizer(train_set_x[idx], train_set_y[idx])
            y_pred.append(this_y)
            total_loss += this_loss
            print '\r', 'epoch:', i, 'idx:', idx, 'loss:', this_loss,
        print "Loss of epoch " + str(i) + ":", total_loss

if __name__ == '__main__':
    run('data/mnist.pkl.gz')

