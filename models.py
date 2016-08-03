import theano
import theano.tensor as T
import numpy as np

class LogisticRegression(object):
    def __init__(self, x, y, in_size, out_size):
        self.W = theano.shared(
            value=np.random.uniform(
                low=-0.01,
                high=0.01,
                size=(in_size, out_size)
            ).astype(theano.config.floatX),
            name='W',
            borrow=True
        )

        self.b = theano.shared(
            value=np.random.uniform(
                low=-0.01,
                high=0.01,
                size=(out_size,)
            ).astype(theano.config.floatX),
            name='b',
            borrow=True
        )

        self.y_given_x = T.nnet.softmax(T.dot(x, self.W) + self.b)

        self.y_d = T.argmax(self.y_given_x, axis=1)

        self.loss = -T.mean(T.log(self.y_given_x)[T.arange(y.shape[0]), y])

        self.error = T.mean(T.neq(self.y_d, y))

        self.params = {'W':self.W, 'b':self.b}
