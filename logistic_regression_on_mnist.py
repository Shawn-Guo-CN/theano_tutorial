import theano
import theano.tensor as T
import numpy as np

import cPickle
import gzip

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

    test_set_x, test_set_y = shared_dataset(test_set)
    train_set_x, train_set_y = shared_dataset(train_set)
    rval = [(train_set_x, train_set_y), (test_set_x, test_set_y)]
    return rval

if __name__ == '__main__':
    dataset = load_data('data/mnist.pkl.gz')

