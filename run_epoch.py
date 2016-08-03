import theano
import theano.tensor as T
import numpy as np

import cPickle
import gzip

import models
from options import options
from data_reader import data_iterator

def run_epoch(data_file):
    # define the symbolic variables we will use
    x = T.matrix('x')
    y = T.ivector('y')

    print '...building model'
    # declare the model to be used
    model = models.LogisticRegression(x, y, options['in_size'], options['out_size'])

    # declare the parameters need to be trained
    tparams = {}
    for k,v in model.params.items():
        tparams[k] = v

    # define the optimizer
    lr = T.scalar(name='lr')
    cost = model.loss
    grads = T.grad(cost, wrt=list(tparams.values()))
    optimizer = options['optimizer']
    f_grad_shared, f_update = optimizer(lr, tparams, grads, x, y, cost)

    print '...training'
    # begin optimization with data iterator
    for i in xrange(options['max_epochs']):
        total_loss = 0
        for x, y in data_iterator(data_file=data_file, batch_size=options['batch_size'], is_train=True):
            this_cost = f_grad_shared(x, y)
            f_update(options['lrate'])
            total_loss += this_cost
            print '\r', 'epoch '+str(i)+':\t', 'current loss:', this_cost,
        print 'total loss:', total_loss

if __name__ == '__main__':
    run_epoch('data/mnist.pkl.gz')