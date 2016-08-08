# theano_tutorial
This is the source code of my Theano tutorial. You can browse it at http://www.shawnguo.cn/2016/07/26/theano-tutorial/
## Here are the source files and their functions:
1. **a+b.py**: create a theano function that can calculate $$z=x+y$$.
 
2. **linear_regression_on_toy_data.py**: generate toy data based on equation $y=3x+2$, then train a linear regression model to fit them.

3. **logistic_regression_on_mnist.py**: create a logistic regression model to handle the recognizing of hand written numbers, i.e. the [mnist](http://yann.lecun.com/exdb/mnist/) dataset.

4. **logistic_regression_on_mnist_batch.py**: change the above model into a batched version.

5. **models_on_mnist**(python package): contains several different models aiming to address the mnist problem. The functions of every source file are listed as follow:
    * \_\_init\_\_.py: eh, in fact, this is a automatically generated file, useless in this project.
    * data_reader.py: implements a iterator that can provide batches of data $(x_{i:i+n}, y_{i:i+n})$ in mnist.
    * models.py: contains several different models aiming at perfectly recognize the numbers in mnist.
    * optimizers.py: consists of three optimizing algorithms, i.e. *sgd*, *adadelta* and *rmsprop*. 
    * options.py: a ordered dict enclosing all the parameters in training model.
    * run_epoch.py: implements the training epoch.