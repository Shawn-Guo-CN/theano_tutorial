import theano
import theano.tensor as T
import numpy as np

import cPickle
import gzip

def load_data(file_name):
    """
    Load data from mnist data set.
    :param file_name: the path and name of original data
    :return: list of two tuples, [(train_set_x, train_set_y), (test_set_x, test_set_y)]
    """
    print "...loading data"

    with gzip.open(file_name, 'rb') as f:
        train_set, test_set = cPickle.load(f)


    rval = [train_set, test_set]
    return rval

def run(data_file):
    # load data
    datasets = load_data(data_file)
    train_set_x, train_set_y = datasets[0]
    test_set_x, test_set_y = datasets[1]

    # define the symbolic variables we will use
    x = T.matrix('x')
    y = T.ivector('y')
    W = theano.shared(value=np.random.uniform(-0.01, 0.01, size=(784, 10)), name='W')
    b = theano.shared(value=np.random.uniform(-0.01, 0.01, size=(10, )), name='b')

    # define the calculate procedure of y
    y_given_x = T.nnet.softmax(T.dot(x, W) + b)
    y_d = T.argmax(y_given_x, axis=1)

    # define loss
    loss = -T.mean(T.log(y_given_x)[T.arange(y.shape[0]), y])

    # define error
    error = T.mean(T.neq(y_d, y))
#
    # define the learning rate of sgd
    lr = 0.001

    # define the optimizer to reduce the loss
    g_W = T.grad(loss, W)
    g_b = T.grad(loss, b)
    updates = [(W, W - lr * g_W), (b, b - lr * g_b)]

    # define the function to optimize the model parameters
    optimizer = theano.function(inputs=[x, y], outputs=[y_d, loss], updates=updates)

    # define the function to test the accuracy of model's perdication
    detector = theano.function(inputs=[x, y], outputs=error)

    # run the optimizer to optimize the model parameters
    max_epoches = 20
    batch_size = 100
    batch_num = train_set_x.shape[0] / batch_size
    test_batch_num = test_set_x.shape[0] / batch_size

    error_rates = []
    for idx in xrange(test_batch_num):
        error_rates.append(detector(test_set_x[idx * batch_size: (idx + 1) * batch_size],
                                    test_set_y[idx * batch_size: (idx + 1) * batch_size]))
    print "Error Rate of Random Initialized Parameters:", np.mean(error_rates)

    for i in xrange(max_epoches):
        y_pred = []
        total_loss = 0
        for idx in xrange(batch_num):
            this_y, this_loss = optimizer(train_set_x[idx * batch_size: (idx+1) * batch_size],
                                          train_set_y[idx * batch_size: (idx+1) * batch_size])
            y_pred.append(this_y)
            total_loss += this_loss
            print '\r', 'epoch:', i, 'batch-idx:', idx, 'loss:', this_loss,
        print "Loss of epoch " + str(i) + ":", total_loss,

        error_rates = []
        for idx in xrange(test_batch_num):
            error_rates.append(detector(test_set_x[idx * batch_size: (idx+1) * batch_size],
                                        test_set_y[idx * batch_size: (idx+1) * batch_size]))
        print "Error Rate:", np.mean(error_rates)

if __name__ == '__main__':
    run('data/mnist.pkl.gz')

