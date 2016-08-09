import theano
import theano.tensor as T
import numpy as np

from theano.tensor.nnet import conv
from theano.tensor.signal import downsample

class LogisticRegression(object):
    def __init__(self, x, y, in_size, out_size, prefix='lr_'):
        self.x = x
        self.y = y

        self.W = theano.shared(
            value=np.random.uniform(
                low=-np.sqrt(6. / (in_size + out_size)),
                high=np.sqrt(6. / (in_size + out_size)),
                size=(in_size, out_size)
            ).astype(theano.config.floatX),
            name='W',
            borrow=True
        )

        self.b = theano.shared(
            value=np.random.uniform(
                low=-np.sqrt(6. / (in_size + out_size)),
                high=np.sqrt(6. / (in_size + out_size)),
                size=(out_size,)
            ).astype(theano.config.floatX),
            name='b',
            borrow=True
        )

        self.y_given_x = T.nnet.softmax(T.dot(x, self.W) + self.b)

        self.y_d = T.argmax(self.y_given_x, axis=1)

        self.loss = -T.mean(T.log(self.y_given_x)[T.arange(y.shape[0]), y])

        self.error = T.mean(T.neq(self.y_d, y))

        self.params = {prefix+'W':self.W, prefix+'b':self.b}

class HiddenLayer(object):
    def __init__(self, x, in_size, out_size, activation=T.nnet.relu, prefix='h_'):
        self.x = x

        self.W = theano.shared(
            value=np.random.uniform(
                low=-np.sqrt(6. / (in_size + out_size)),
                high=np.sqrt(6. / (in_size + out_size)),
                size=(in_size, out_size)
            ).astype(theano.config.floatX),
            name='W',
            borrow=True
        )

        self.b = theano.shared(
            value=np.random.uniform(
                low=-np.sqrt(6. / (in_size + out_size)),
                high=np.sqrt(6. / (in_size + out_size)),
                size=(out_size,)
            ).astype(theano.config.floatX),
            name='b',
            borrow=True
        )

        self.output = activation(T.dot(x, self.W) + self.b)

        self.params = {prefix+'W': self.W, prefix+'b': self.b}

class MLP(object):
    def __init__(self, x, y, in_size, out_size, hidden_size):
        self.hidden_layer = HiddenLayer(
            x=x,
            in_size=in_size,
            out_size=hidden_size,
            prefix='hidden_'
        )

        self.logistic_layer = LogisticRegression(
            x=self.hidden_layer.output,
            y=y,
            in_size=hidden_size,
            out_size=out_size,
            prefix='lr_'
        )

        self.loss = self.logistic_layer.loss
        self.error = self.logistic_layer.error

        self.params = dict(self.hidden_layer.params.items() + self.logistic_layer.params.items())

class LeNetConvPoolLayer(object):
    def __init__(self, input, image_shape, filter_shape, poolsize=(2, 2), prefix='cpl_'):
        assert image_shape[1] == filter_shape[1]
        self.input = input

        fan_in = np.prod(filter_shape[1:])
        fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) / np.prod(poolsize))

        self.W = theano.shared(
            value=np.random.uniform(
                low=-np.sqrt(6. / (fan_in + fan_out)),
                high=np.sqrt(6. / (fan_in + fan_out)),
                size=filter_shape).astype(theano.config.floatX),
            name=prefix+'W',
            borrow=True
        )

        self.b = theano.shared(
            value=np.zeros((filter_shape[0],), dtype=theano.config.floatX),
            name=prefix+'b',
            borrow = True
        )

        conv_out = conv.conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            image_shape=image_shape
        )

        pooled_out = downsample.max_pool_2d(
            input=conv_out,
            ds=poolsize,
            ignore_border=True
        )

        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        self.params = {prefix+'W': self.W, prefix+'b': self.b}

class LeNet(object):
    def __init__(self, x, y, nkernels=[20, 50], batch_size=100):
        self.input = x.reshape((batch_size, 1, 28, 28))

        # Construct the first convolutional pooling layer:
        # filtering reduces the image size to (28-5+1, 28-5+1)=(24, 24)
        # maxpooling reduces this further to (24/2, 24/2)=(12, 12)
        # 4D output tensor is thus of shape (batch_size, nkerns[0], 12, 12)
        self.layer0 = LeNetConvPoolLayer(
            input=self.input,
            image_shape=(batch_size, 1, 28, 28),
            filter_shape=(nkernels[0], 1, 5, 5),
            poolsize=(2, 2),
            prefix='cpl0_'
        )

        # Construct the second convolutional pooling layer:
        # filtering reduces the image size to (12-5+1, 12-5+1)=(8, 8)
        # maxpooling reduces this further to (8/2, 8/2)=(4, 4)
        # 4D output tensor is thus of shape (nkerns[0], nkerns[1], 4, 4)
        self.layer1 = LeNetConvPoolLayer(
            input=self.layer0.output,
            image_shape=(batch_size, nkernels[0], 12, 12),
            filter_shape=(nkernels[1], nkernels[0], 5, 5),
            poolsize=(2, 2),
            prefix='cpl1_'
        )

        # The HiddenLayer being fully-connected, it operates on 2D matrices of shape(batch_size, num_pixels)
        # This will generate a matrix of shape (20, 32 * 4 *4) = (20, 512)
        self.layer2_input = self.layer1.output.flatten(2)

        # construct a fully-connected sigmoid layer
        self.layer2 = HiddenLayer(
            x=self.layer2_input,
            in_size=nkernels[1] * 4 * 4,
            out_size=500,
            prefix='hl_'
        )

        # classify the values of the fully-connected sigmoid layer
        self.layer3 = LogisticRegression(x=self.layer2.output, y=y, in_size=500, out_size=10, prefix='lr_')

        self.loss = self.layer3.loss
        self.error = self.layer3.error

        self.params = dict(self.layer0.params.items() +
                           self.layer1.params.items() +
                           self.layer2.params.items() +
                           self.layer3.params.items())
