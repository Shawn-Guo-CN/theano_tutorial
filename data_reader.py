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

initialize = True
datasets = []
train_set_x = []
train_set_y = []
test_set_x = []
test_set_y = []

def data_iterator(data_file, batch_size, is_train=True):
    # load data
    global initialize
    global datasets
    global train_set_x
    global train_set_y
    global test_set_x
    global test_set_y
    if initialize:
        datasets = load_data(data_file)
        train_set_x, train_set_y = datasets[0]
        test_set_x, test_set_y = datasets[1]
        initialize = False

    if is_train:
        batch_num = train_set_x.shape[0] / batch_size
        for idx in xrange(batch_num):
            yield train_set_x[idx * batch_size: (idx + 1) * batch_size], train_set_y[idx * batch_size: (idx + 1) * batch_size]
    else:
        batch_num = test_set_x.shape[0] / batch_size
        for idx in xrange(batch_num):
            yield test_set_x[idx * batch_size: (idx + 1) * batch_size], test_set_y[idx * batch_size: (idx + 1) * batch_size]

