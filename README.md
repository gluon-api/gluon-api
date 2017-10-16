<p align="center"><img width="50%" src="_static/gluon_logo_horizontal_small.png" /></p>

# The Gluon API Specification
The Gluon API specification is an effort to improve speed, flexibility, and accessibility of deep learning technology for all developers, regardless of their deep learning framework of choice. The Gluon API offers a flexible interface that simplifies the process of prototyping, building, and training deep learning models without sacrificing training speed. It offers four distinct advantages:
*	**Simple, Easy-to-Understand Code**: Gluon offers a full set of plug-and-play neural network building blocks, including predefined layers, optimizers, and initializers.
*	**Flexible, Imperative Structure**: Gluon does not require the neural network model to be rigidly defined, but rather brings the training algorithm and model closer together to provide flexibility in the development process.
*	**Dynamic Graphs**: Gluon enables developers to define neural network models that are dynamic, meaning they can be built on the fly, with any structure, and using any of Python’s native control flow.
*	**High Performance**: Gluon provides all of the above benefits without impacting the training speed that the underlying engine provides.

## Gluon API Reference
* [Gluon API](https://github.com/gluon-api/gluon-api/blob/master/docs/gluon.rst)
* [Autograd API](https://github.com/gluon-api/gluon-api/blob/master/docs/autograd.rst)
* [Gluon Neural Network Layers API](https://github.com/gluon-api/gluon-api/blob/master/docs/nn.rst)
* [Gluon Recurrent Neural Network API](https://github.com/gluon-api/gluon-api/blob/master/docs/rnn.rst)
* [Gluon Loss API](https://github.com/gluon-api/gluon-api/blob/master/docs/loss.rst)
* [Gluon Data API](https://github.com/gluon-api/gluon-api/blob/master/docs/data.rst)
* [NDArray API](https://github.com/gluon-api/gluon-api/blob/master/docs/ndarray.rst)
* [Sparse NDArray API](https://github.com/gluon-api/gluon-api/blob/master/docs/sparse.rst)
* [Model Zoo](https://github.com/gluon-api/gluon-api/blob/master/docs/model_zoo.rst)
* [Backend C API](https://github.com/gluon-api/gluon-api/blob/master/docs/gluon_api.h)

## Getting Started with the Gluon Interface
The Gluon specification has already been implemented in Apache MXNet, so you can start using the Gluon interface by following these easy steps for [installing the latest master version of MXNet](https://mxnet.incubator.apache.org/versions/master/install/index.html). We recommend using Python version 3.3 or greater and implementing this example using a [Jupyter notebook](http://jupyter.readthedocs.io/en/latest/install.html). Setup of Jupyter is included in the MXNet installation instructions. For our example we’ll walk through how to build and train a simple two-layer neural network, called a [multilayer perceptron](http://thestraightdope.mxnet.io/chapter03_deep-neural-networks/mlp-gluon.html).

First, import `mxnet` and MXNet's implementation of the `gluon` specification. We will also need `autograd`, `ndarray`, and `numpy`.

```python
import mxnet as mx
from mxnet import gluon, autograd, ndarray
import numpy as np
```

Next, we use `gluon.data.DataLoader`, Gluon's data iterator, to hold the training and test data. Iterators are a useful object class for traversing through large datasets. We pass Gluon's DataLoader a helper, `gluon.data.vision.MNIST`, that will pre-process the MNIST handwriting dataset, getting into the right size and format, using parameters to tell it which is test set and which is the training set.

```python
train_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.MNIST(train=True, transform=lambda data, label: (data.astype(np.float32)/255, label)),
                                      batch_size=32, shuffle=True)
test_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.MNIST(train=False, transform=lambda data, label: (data.astype(np.float32)/255, label)),
                                     batch_size=32, shuffle=False)                     
```

Now, we are ready to define the actual neural network, and we can do so in five simple lines of code. First, we initialize the network with `net = gluon.nn.Sequential()`. Then, with that net, we create three layers using `gluon.nn.Dense`: the first will have 128 nodes, and the second will have 64 nodes. They both incorporate the `relu` by passing that into the `activation` function parameter. The final layer for our model, `gluon.nn.Dense(10)`, is used to set up the output layer with the number of nodes corresponding to the total number of possible outputs. In our case with MNIST, there are only 10 possible outputs because the pictures represent numerical digits of which there are only 10 (i.e., 0 to 9).

```python
# First step is to initialize your model
net = gluon.nn.Sequential()
# Then, define your model architecture
with net.name_scope():
    net.add(gluon.nn.Dense(128, activation="relu")) # 1st layer - 128 nodes
    net.add(gluon.nn.Dense(64, activation="relu")) # 2nd layer – 64 nodes
    net.add(gluon.nn.Dense(10)) # Output layer
```

Prior to kicking off the model training process, we need to initialize the model’s parameters and set up the loss with `gluon.loss.SoftmaxCrossEntropyLoss()` and model optimizer functions with `gluon.Trainer`. As with creating the model, these normally complicated functions are distilled to one line of code each.

```python
# We start with random values for all of the model’s parameters from a
# normal distribution with a standard deviation of 0.05
net.collect_params().initialize(mx.init.Normal(sigma=0.05))

# We opt to use softmax cross entropy loss function to measure how well the # model is able to predict the correct answer
softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

# We opt to use the stochastic gradient descent (sgd) training algorithm
# and set the learning rate hyperparameter to .1
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': .1})
```

Running the training is fairly typical and all the while using Gluon's functionality to make the process simple and seamless. There are four steps: (1) pass in a batch of data; (2) calculate the difference between the output generated by the neural network model and the actual truth (i.e., the loss); (3) use Gluon's `autograd` to calculate the derivatives of the model’s parameters with respect to their impact on the loss; and (4) use the Gluon's `trainer` method to optimize the parameters in a way that will decrease the loss. We set the number of epochs at 10, meaning that we will cycle through the entire training dataset 10 times.

```python
epochs = 10
for e in range(epochs):
    for i, (data, label) in enumerate(train_data):
        data = data.as_in_context(mx.cpu()).reshape((-1, 784))
        label = label.as_in_context(mx.cpu())
        with autograd.record(): # Start recording the derivatives
            output = net(data) # the forward iteration
            loss = softmax_cross_entropy(output, label)
            loss.backward()
        trainer.step(data.shape[0])
        # Provide stats on the improvement of the model over each epoch
        curr_loss = ndarray.mean(loss).asscalar()
    print("Epoch {}. Current Loss: {}.".format(e, curr_loss))
```

We now have a trained neural network model, and can see how the accuracy improves over each epoch.

A Jupyter notebook of this code has been [provided for your convenience](tutorials/mnist-gluon-example.ipynb).

To learn more about the Gluon interface and deep learning, you can reference this [comprehensive set of tutorials](http://gluon.mxnet.io/), which covers everything from an introduction to deep learning to how to implement cutting-edge neural network models.


## License
[Apache 2.0](https://github.com/gluon-api/gluon-api/blob/master/LICENSE)
