import theano
import theano.tensor as T
import numpy as np

x = T.scalar('x')
y = T.scalar('y')

z = x + y
f = theano.function(inputs=[x, y], outputs=z)

print f(1.0, 2.0)