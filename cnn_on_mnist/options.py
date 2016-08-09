from collections import OrderedDict
import optimizers
import models

options = OrderedDict(
    {
        'model':models.LeNet, # define the model
        'in_size':784, # input dimension
        'hidden_size':1000,  # number of hidden units in single layer
        'out_size':10, # number of units in output layer
        'patience':10,  # Number of epoch to wait before early stop if no progress
        'max_epochs':20,  # The maximum number of epoch to run
        'dispFreq':10,  # Display to stdout the training progress every N updates
        'decay_c':0.,  # Weight decay for the classifier applied to the U weights.
        'lrate':0.001,  # Learning rate for sgd (not used for adadelta and rmsprop)
        'optimizer':optimizers.adadelta,  # sgd, adadelta and rmsprop available, sgd very hard to use, not recommanded (probably need momentum and decaying learning rate).
        'valid_freq':5,  # Compute the validation error after this number of update.
        'maxlen':100,  # Sequence longer then this get ignored
        'batch_size':100,  # The batch size during training.
        'valid_batch_size':100,  # The batch size used for validation/test set.
        'dataset':'imdb',
        'nkernals':[20, 50],
    }
)