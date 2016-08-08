import theano
import theano.tensor as T
import numpy as np

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