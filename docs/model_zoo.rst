
Gluon Model Zoo API
*******************


Overview
========

This document lists the model APIs in Gluon:

+---------------------------+--------------------------------------------------------------------------------------------+
| gluon.model_zoo           | Predefined and pretrained models.                                                          |
+---------------------------+--------------------------------------------------------------------------------------------+

The ``Gluon Model Zoo`` API, defined in the ``gluon.model_zoo``
package, provides pre-defined and pre-trained models to help bootstrap
machine learning applications.

In the rest of this document, we list routines provided by the
``gluon.model_zoo`` package.


Vision
------

Module for pre-defined neural network models.

This module contains definitions for the following model
architectures: 
  -  AlexNet 
  -  DenseNet 
  -  Inception V3 
  -  ResNet V1 
  -  ResNet V2 
  -  SqueezeNet 
  -  VGG 
  -  MobileNet

You can construct a model with random weights by calling its
constructor: .. code:

.. code-block:: python
   
   from gluon.model_zoo import vision
   resnet18 = vision.resnet18_v1()
   alexnet = vision.alexnet()
   squeezenet = vision.squeezenet1_0()
   densenet = vision.densenet_161()

We provide pre-trained models for all the models except ResNet V2.
These can constructed by passing ``pretrained=True``: .. code:

.. code-block:: python

   from gluon.model_zoo import vision
   resnet18 = vision.resnet18_v1(pretrained=True)
   alexnet = vision.alexnet(pretrained=True)

Pretrained models are converted from torchvision. All pre-trained
models expect input images normalized in the same way, i.e.
mini-batches of 3-channel RGB images of shape (N x 3 x H x W), where N
is the batch size, and H and W are expected to be at least 224. The
images have to be loaded in to a range of [0, 1] and then normalized
using ``mean = [0.485, 0.456, 0.406]`` and ``std = [0.229, 0.224,
0.225]``. The transformation should preferrably happen at
preprocessing. You can use ``gluon.image.color_normalize`` for such
transformation:

.. code-block:: python

   image = image/255
   normalized = gluon.image.color_normalize(image,
                                         mean=gluon.nd.array([0.485, 0.456, 0.406]),
                                         std=gluon.nd.array([0.229, 0.224, 0.225]))

+---------------+--------------------------------------------------------------------------------------------+
| ``get_model`` | Returns a pre-defined model by name                                                        |
+---------------+--------------------------------------------------------------------------------------------+


ResNet
~~~~~~

+------------------+--------------------------------------------------------------------------------------------+
| ``resnet18_v1``  | ResNet-18 V1 model from "Deep Residual Learning for Image Recognition" paper.              |
+------------------+--------------------------------------------------------------------------------------------+
| ``resnet34_v1``  | ResNet-34 V1 model from "Deep Residual Learning for Image Recognition" paper.              |
+------------------+--------------------------------------------------------------------------------------------+
| ``resnet50_v1``  | ResNet-50 V1 model from "Deep Residual Learning for Image Recognition" paper.              |
+------------------+--------------------------------------------------------------------------------------------+
| ``resnet101_v1`` | ResNet-101 V1 model from "Deep Residual Learning for Image Recognition" paper.             |
+------------------+--------------------------------------------------------------------------------------------+
| ``resnet152_v1`` | ResNet-152 V1 model from "Deep Residual Learning for Image Recognition" paper.             |
+------------------+--------------------------------------------------------------------------------------------+
| ``resnet18_v2``  | ResNet-18 V2 model from "Identity Mappings in Deep Residual Networks" paper.               |
+------------------+--------------------------------------------------------------------------------------------+
| ``resnet34_v2``  | ResNet-34 V2 model from "Identity Mappings in Deep Residual Networks" paper.               |
+------------------+--------------------------------------------------------------------------------------------+
| ``resnet50_v2``  | ResNet-50 V2 model from "Identity Mappings in Deep Residual Networks" paper.               |
+------------------+--------------------------------------------------------------------------------------------+
| ``resnet101_v2`` | ResNet-101 V2 model from "Identity Mappings in Deep Residual Networks" paper.              |
+------------------+--------------------------------------------------------------------------------------------+
| ``resnet152_v2`` | ResNet-152 V2 model from "Identity Mappings in Deep Residual Networks" paper.              |
+------------------+--------------------------------------------------------------------------------------------+

+------------------+--------------------------------------------------------------------------------------------+
| ``ResNetV1``     | ResNet V1 model from "Deep Residual Learning for Image Recognition" paper.                 |
+------------------+--------------------------------------------------------------------------------------------+
| ``ResNetV2``     | ResNet V2 model from "Identity Mappings in Deep Residual Networks" paper.                  |
+------------------+--------------------------------------------------------------------------------------------+
| ``BasicBlockV1`` | BasicBlock V1 from "Deep Residual Learning for Image Recognition" paper.                   |
+------------------+--------------------------------------------------------------------------------------------+
| ``BasicBlockV2`` | BasicBlock V2 from "Identity Mappings in Deep Residual Networks" paper.                    |
+------------------+--------------------------------------------------------------------------------------------+
| ``BottleneckV1`` | Bottleneck V1 from "Deep Residual Learning for Image Recognition" paper.                   |
+------------------+--------------------------------------------------------------------------------------------+
| ``BottleneckV2`` | Bottleneck V2 from "Identity Mappings in Deep Residual Networks" paper.                    |
+------------------+--------------------------------------------------------------------------------------------+
| ``get_resnet``   | ResNet V1 model from "Deep Residual Learning for Image Recognition" paper.                 |
+------------------+--------------------------------------------------------------------------------------------+


VGG
~~~

+--------------+--------------------------------------------------------------------------------------------+
| ``vgg11``    | VGG-11 model from the "Very Deep Convolutional Networks for Large-Scale Image Recognition" |
+--------------+--------------------------------------------------------------------------------------------+
| ``vgg13``    | VGG-13 model from the "Very Deep Convolutional Networks for Large-Scale Image Recognition" |
+--------------+--------------------------------------------------------------------------------------------+
| ``vgg16``    | VGG-16 model from the "Very Deep Convolutional Networks for Large-Scale Image Recognition" |
+--------------+--------------------------------------------------------------------------------------------+
| ``vgg19``    | VGG-19 model from the "Very Deep Convolutional Networks for Large-Scale Image Recognition" |
+--------------+--------------------------------------------------------------------------------------------+
| ``vgg11_bn`` | VGG-11 model with batch normalization from the "Very Deep Convolutional Networks for       |
+--------------+--------------------------------------------------------------------------------------------+
| ``vgg13_bn`` | VGG-13 model with batch normalization from the "Very Deep Convolutional Networks for       |
+--------------+--------------------------------------------------------------------------------------------+
| ``vgg16_bn`` | VGG-16 model with batch normalization from the "Very Deep Convolutional Networks for       |
+--------------+--------------------------------------------------------------------------------------------+
| ``vgg19_bn`` | VGG-19 model with batch normalization from the "Very Deep Convolutional Networks for       |
+--------------+--------------------------------------------------------------------------------------------+

+-------------+--------------------------------------------------------------------------------------------+
| ``VGG``     | VGG model from the "Very Deep Convolutional Networks for Large-Scale Image Recognition"    |
+-------------+--------------------------------------------------------------------------------------------+
| ``get_vgg`` | VGG model from the "Very Deep Convolutional Networks for Large-Scale Image Recognition"    |
+-------------+--------------------------------------------------------------------------------------------+


Alexnet
~~~~~~~

+-------------+--------------------------------------------------------------------------------------------+
| ``alexnet`` | AlexNet model from the "One weird trick..." paper.                                         |
+-------------+--------------------------------------------------------------------------------------------+

+-------------+--------------------------------------------------------------------------------------------+
| ``AlexNet`` | AlexNet model from the "One weird trick..." paper.                                         |
+-------------+--------------------------------------------------------------------------------------------+


DenseNet
~~~~~~~~

+-----------------+--------------------------------------------------------------------------------------------+
| ``densenet121`` | Densenet-BC 121-layer model from the "Densely Connected Convolutional Networks" paper.     |
+-----------------+--------------------------------------------------------------------------------------------+
| ``densenet161`` | Densenet-BC 161-layer model from the "Densely Connected Convolutional Networks" paper.     |
+-----------------+--------------------------------------------------------------------------------------------+
| ``densenet169`` | Densenet-BC 169-layer model from the "Densely Connected Convolutional Networks" paper.     |
+-----------------+--------------------------------------------------------------------------------------------+
| ``densenet201`` | Densenet-BC 201-layer model from the "Densely Connected Convolutional Networks" paper.     |
+-----------------+--------------------------------------------------------------------------------------------+

+--------------+--------------------------------------------------------------------------------------------+
| ``DenseNet`` | Densenet-BC model from the "Densely Connected Convolutional Networks" paper.               |
+--------------+--------------------------------------------------------------------------------------------+


SqueezeNet
~~~~~~~~~~

+-------------------+--------------------------------------------------------------------------------------------+
| ``squeezenet1_0`` | SqueezeNet 1.0 model from the "SqueezeNet: AlexNet-level accuracy with 50x fewer           |
+-------------------+--------------------------------------------------------------------------------------------+
| ``squeezenet1_1`` | SqueezeNet 1.1 model from the official SqueezeNet repo.                                    |
+-------------------+--------------------------------------------------------------------------------------------+

+----------------+--------------------------------------------------------------------------------------------+
| ``SqueezeNet`` | SqueezeNet model from the "SqueezeNet: AlexNet-level accuracy with 50x fewer parameters    |
+----------------+--------------------------------------------------------------------------------------------+


Inception
~~~~~~~~~

+------------------+--------------------------------------------------------------------------------------------+
| ``inception_v3`` | Inception v3 model from "Rethinking the Inception Architecture for Computer Vision" paper. |
+------------------+--------------------------------------------------------------------------------------------+

+----------------+--------------------------------------------------------------------------------------------+
| ``Inception3`` | Inception v3 model from "Rethinking the Inception Architecture for Computer Vision" paper. |
+----------------+--------------------------------------------------------------------------------------------+


MobileNet
~~~~~~~~~

+-------------------+--------------------------------------------------------------------------------------------+
| ``mobilenet1_0``  | MobileNet model from the "MobileNets: Efficient Convolutional Neural Networks for Mobile   |
+-------------------+--------------------------------------------------------------------------------------------+
| ``mobilenet0_75`` | MobileNet model from the "MobileNets: Efficient Convolutional Neural Networks for Mobile   |
+-------------------+--------------------------------------------------------------------------------------------+
| ``mobilenet0_5``  | MobileNet model from the "MobileNets: Efficient Convolutional Neural Networks for Mobile   |
+-------------------+--------------------------------------------------------------------------------------------+
| ``mobilenet0_25`` | MobileNet model from the "MobileNets: Efficient Convolutional Neural Networks for Mobile   |
+-------------------+--------------------------------------------------------------------------------------------+

+---------------+--------------------------------------------------------------------------------------------+
| ``MobileNet`` | MobileNet model from the "MobileNets: Efficient Convolutional Neural Networks for Mobile   |
+---------------+--------------------------------------------------------------------------------------------+


API Reference
=============

Module for pre-defined neural network models.

This module contains definitions for the following model
architectures: -  AlexNet -  DenseNet -  Inception V3 -  ResNet V1 -
ResNet V2 -  SqueezeNet -  VGG -  MobileNet

You can construct a model with random weights by calling its
constructor: .. code:

.. code-block:: python

   from gluon.model_zoo import vision
   resnet18 = vision.resnet18_v1()
   alexnet = vision.alexnet()
   squeezenet = vision.squeezenet1_0()
   densenet = vision.densenet_161()

We provide pre-trained models for all the models except ResNet V2.
These can constructed by passing ``pretrained=True``: .. code:

.. code-block:: python

   from gluon.model_zoo import vision
   resnet18 = vision.resnet18_v1(pretrained=True)
   alexnet = vision.alexnet(pretrained=True)

Pretrained models are converted from torchvision. All pre-trained
models expect input images normalized in the same way, i.e.
mini-batches of 3-channel RGB images of shape (N x 3 x H x W), where N
is the batch size, and H and W are expected to be at least 224. The
images have to be loaded in to a range of [0, 1] and then normalized
using ``mean = [0.485, 0.456, 0.406]`` and ``std = [0.229, 0.224,
0.225]``. The transformation should preferrably happen at
preprocessing. You can use ``gluon.image.color_normalize`` for such
transformation:

.. code-block:: python

   image = image/255
   normalized = gluon.image.color_normalize(image,
                                         mean=gluon.nd.array([0.485, 0.456, 0.406]),
                                         std=gluon.nd.array([0.229, 0.224, 0.225]))

**gluon.model_zoo.vision.get_model(name, **kwargs)**

   Returns a pre-defined model by name

   :Parameters:
      * **name** (*str*) -- Name of the model.

      * **pretrained** (*bool*) -- Whether to load the pretrained
        weights for model.

      * **classes** (*int*) -- Number of classes for the output layer.

      * **ctx** (*Context*, *defaultCPU*) -- The context in which
        to load the pretrained weights.

      * **root** (*str*, *default'~/.gluon/models'*) -- Location
        for keeping the model parameters.

   :Returns:
      The model.

   :Return type:
      `HybridBlock <gluon.rst#gluon.HybridBlock>`_

**class gluon.model_zoo.vision.AlexNet(classes=1000, **kwargs)**

   AlexNet model from the "One weird trick..." paper.

   :Parameters:
      **classes** (*int*, *default1000*) -- Number of classes for
      the output layer.

**class gluon.model_zoo.vision.BasicBlockV1(channels, stride,
downsample=False, in_channels=0, **kwargs)**

   BasicBlock V1 from "Deep Residual Learning for Image Recognition"
   paper. This is used for ResNet V1 for 18, 34 layers.

   :Parameters:
      * **channels** (*int*) -- Number of output channels.

      * **stride** (*int*) -- Stride size.

      * **downsample** (*bool*, *default False*) -- Whether to
        downsample the input.

      * **in_channels** (*int*, *default0*) -- Number of input
        channels. Default is 0, to infer from the graph.

**class gluon.model_zoo.vision.BasicBlockV2(channels, stride,
downsample=False, in_channels=0, **kwargs)**

   BasicBlock V2 from "Identity Mappings in Deep Residual Networks"
   paper. This is used for ResNet V2 for 18, 34 layers.

   :Parameters:
      * **channels** (*int*) -- Number of output channels.

      * **stride** (*int*) -- Stride size.

      * **downsample** (*bool*, *default False*) -- Whether to
        downsample the input.

      * **in_channels** (*int*, *default0*) -- Number of input
        channels. Default is 0, to infer from the graph.

**class gluon.model_zoo.vision.BottleneckV1(channels, stride,
downsample=False, in_channels=0, **kwargs)**

   Bottleneck V1 from "Deep Residual Learning for Image Recognition"
   paper. This is used for ResNet V1 for 50, 101, 152 layers.

   :Parameters:
      * **channels** (*int*) -- Number of output channels.

      * **stride** (*int*) -- Stride size.

      * **downsample** (*bool*, *default False*) -- Whether to
        downsample the input.

      * **in_channels** (*int*, *default0*) -- Number of input
        channels. Default is 0, to infer from the graph.

**class gluon.model_zoo.vision.BottleneckV2(channels, stride,
downsample=False, in_channels=0, **kwargs)**

   Bottleneck V2 from "Identity Mappings in Deep Residual Networks"
   paper. This is used for ResNet V2 for 50, 101, 152 layers.

   :Parameters:
      * **channels** (*int*) -- Number of output channels.

      * **stride** (*int*) -- Stride size.

      * **downsample** (*bool*, *default False*) -- Whether to
        downsample the input.

      * **in_channels** (*int*, *default0*) -- Number of input
        channels. Default is 0, to infer from the graph.

**class gluon.model_zoo.vision.DenseNet(num_init_features,
growth_rate, block_config, bn_size=4, dropout=0, classes=1000,
**kwargs)**

   Densenet-BC model from the "Densely Connected Convolutional
   Networks" paper.

   :Parameters:
      * **num_init_features** (*int*) -- Number of filters to learn in
        the first convolution layer.

      * **growth_rate** (*int*) -- Number of filters to add each layer
        (*k* in the paper).

      * **block_config** (*list of int*) -- List of integers for
        numbers of layers in each pooling block.

      * **bn_size** (*int*, *default4*) -- Multiplicative factor
        for number of bottle neck layers. (i.e. bn_size * k features
        in the bottleneck layer)

      * **dropout** (*float*, *default0*) -- Rate of dropout after
        each dense layer.

      * **classes** (*int*, *default1000*) -- Number of
        classification classes.

**class gluon.model_zoo.vision.Inception3(classes=1000,
**kwargs)**

   Inception v3 model from "Rethinking the Inception Architecture for
   Computer Vision" paper.

   :Parameters:
      **classes** (*int*, *default1000*) -- Number of
      classification classes.

**class gluon.model_zoo.vision.MobileNet(multiplier=1.0,
classes=1000, **kwargs)**

   MobileNet model from the "MobileNets: Efficient Convolutional
   Neural Networks for Mobile Vision Applications" paper.

   :Parameters:
      * **multiplier** (*float*, *default1.0*) -- The width
        multiplier for controling the model size. Only multipliers
        that are no less than 0.25 are supported. The actual number of
        channels is equal to the original channel size multiplied by
        this multiplier.

      * **classes** (*int*, *default1000*) -- Number of classes for
        the output layer.

**class gluon.model_zoo.vision.ResNetV1(block, layers, channels,
classes=1000, thumbnail=False, **kwargs)**

   ResNet V1 model from "Deep Residual Learning for Image Recognition"
   paper.

   :Parameters:
      * **block** (`HybridBlock <gluon.rst#gluon.HybridBlock>`_)
        -- Class for the residual block. Options are BasicBlockV1,
        BottleneckV1.

      * **layers** (*list of int*) -- Numbers of layers in each block

      * **channels** (*list of int*) -- Numbers of channels in each
        block. Length should be one larger than layers list.

      * **classes** (*int*, *default1000*) -- Number of
        classification classes.

      * **thumbnail** (*bool*, *default False*) -- Enable thumbnail.

**class gluon.model_zoo.vision.ResNetV2(block, layers, channels,
classes=1000, thumbnail=False, **kwargs)**

   ResNet V2 model from "Identity Mappings in Deep Residual Networks"
   paper.

   :Parameters:
      * **block** (`HybridBlock <gluon.rst#gluon.HybridBlock>`_)
        -- Class for the residual block. Options are BasicBlockV1,
        BottleneckV1.

      * **layers** (*list of int*) -- Numbers of layers in each block

      * **channels** (*list of int*) -- Numbers of channels in each
        block. Length should be one larger than layers list.

      * **classes** (*int*, *default1000*) -- Number of
        classification classes.

      * **thumbnail** (*bool*, *default False*) -- Enable thumbnail.

**class gluon.model_zoo.vision.SqueezeNet(version, classes=1000,
**kwargs)**

   SqueezeNet model from the "SqueezeNet: AlexNet-level accuracy with
   50x fewer parameters and <0.5MB model size" paper. SqueezeNet 1.1
   model from the official SqueezeNet repo. SqueezeNet 1.1 has 2.4x
   less computation and slightly fewer parameters than SqueezeNet 1.0,
   without sacrificing accuracy.

   :Parameters:
      * **version** (*str*) -- Version of squeezenet. Options are
        '1.0', '1.1'.

      * **classes** (*int*, *default1000*) -- Number of
        classification classes.

**class gluon.model_zoo.vision.VGG(layers, filters,
classes=1000, batch_norm=False, **kwargs)**

   VGG model from the "Very Deep Convolutional Networks for
   Large-Scale Image Recognition" paper.

   :Parameters:
      * **layers** (*list of int*) -- Numbers of layers in each
        feature block.

      * **filters** (*list of int*) -- Numbers of filters in each
        feature block. List length should match the layers.

      * **classes** (*int*, *default1000*) -- Number of
        classification classes.

      * **batch_norm** (*bool*, *default False*) -- Use batch
        normalization.

**gluon.model_zoo.vision.alexnet(pretrained=False, ctx=cpu(0),
root='~/.gluon/models', **kwargs)**

   AlexNet model from the "One weird trick..." paper.

   :Parameters:
      * **pretrained** (*bool*, *default False*) -- Whether to load
        the pretrained weights for model.

      * **ctx** (*Context*, *defaultCPU*) -- The context in which
        to load the pretrained weights.

      * **root** (*str*, *default'~/.gluon/models'*) -- Location
        for keeping the model parameters.

**gluon.model_zoo.vision.densenet121(**kwargs)**

   Densenet-BC 121-layer model from the "Densely Connected
   Convolutional Networks" paper.

   :Parameters:
      * **pretrained** (*bool*, *default False*) -- Whether to load
        the pretrained weights for model.

      * **ctx** (*Context*, *defaultCPU*) -- The context in which
        to load the pretrained weights.

      * **root** (*str*, *default'~/.gluon/models'*) -- Location
        for keeping the model parameters.

**gluon.model_zoo.vision.densenet161(**kwargs)**

   Densenet-BC 161-layer model from the "Densely Connected
   Convolutional Networks" paper.

   :Parameters:
      * **pretrained** (*bool*, *default False*) -- Whether to load
        the pretrained weights for model.

      * **ctx** (*Context*, *defaultCPU*) -- The context in which
        to load the pretrained weights.

      * **root** (*str*, *default'~/.gluon/models'*) -- Location
        for keeping the model parameters.

**gluon.model_zoo.vision.densenet169(**kwargs)**

   Densenet-BC 169-layer model from the "Densely Connected
   Convolutional Networks" paper.

   :Parameters:
      * **pretrained** (*bool*, *default False*) -- Whether to load
        the pretrained weights for model.

      * **ctx** (*Context*, *defaultCPU*) -- The context in which
        to load the pretrained weights.

      * **root** (*str*, *default'~/.gluon/models'*) -- Location
        for keeping the model parameters.

**gluon.model_zoo.vision.densenet201(**kwargs)**

   Densenet-BC 201-layer model from the "Densely Connected
   Convolutional Networks" paper.

   :Parameters:
      * **pretrained** (*bool*, *default False*) -- Whether to load
        the pretrained weights for model.

      * **ctx** (*Context*, *defaultCPU*) -- The context in which
        to load the pretrained weights.

      * **root** (*str*, *default'~/.gluon/models'*) -- Location
        for keeping the model parameters.

**gluon.model_zoo.vision.get_mobilenet(multiplier,
pretrained=False, ctx=cpu(0), root='~/.gluon/models', **kwargs)**

   MobileNet model from the "MobileNets: Efficient Convolutional
   Neural Networks for Mobile Vision Applications" paper.

   :Parameters:
      * **multiplier** (*float*) -- The width multiplier for
        controling the model size. Only multipliers that are no less
        than 0.25 are supported. The actual number of channels is
        equal to the original channel size multiplied by this
        multiplier.

      * **pretrained** (*bool*, *default False*) -- Whether to load
        the pretrained weights for model.

      * **ctx** (*Context*, *defaultCPU*) -- The context in which
        to load the pretrained weights.

      * **root** (*str*, *default'~/.gluon/models'*) -- Location
        for keeping the model parameters.

**gluon.model_zoo.vision.get_resnet(version, num_layers,
pretrained=False, ctx=cpu(0), root='~/.gluon/models', **kwargs)**

   ResNet V1 model from "Deep Residual Learning for Image Recognition"
   paper. ResNet V2 model from "Identity Mappings in Deep Residual
   Networks" paper.

   :Parameters:
      * **version** (*int*) -- Version of ResNet. Options are 1, 2.

      * **num_layers** (*int*) -- Numbers of layers. Options are 18,
        34, 50, 101, 152.

      * **pretrained** (*bool*, *default False*) -- Whether to load
        the pretrained weights for model.

      * **ctx** (*Context*, *defaultCPU*) -- The context in which
        to load the pretrained weights.

      * **root** (*str*, *default'~/.gluon/models'*) -- Location
        for keeping the model parameters.

**gluon.model_zoo.vision.get_vgg(num_layers, pretrained=False,
ctx=cpu(0), root='~/.gluon/models', **kwargs)**

   VGG model from the "Very Deep Convolutional Networks for
   Large-Scale Image Recognition" paper.

   :Parameters:
      * **num_layers** (*int*) -- Number of layers for the variant of
        densenet. Options are 11, 13, 16, 19.

      * **pretrained** (*bool*, *default False*) -- Whether to load
        the pretrained weights for model.

      * **ctx** (*Context*, *defaultCPU*) -- The context in which
        to load the pretrained weights.

      * **root** (*str*, *default'~/.gluon/models'*) -- Location
        for keeping the model parameters.

**gluon.model_zoo.vision.inception_v3(pretrained=False,
ctx=cpu(0), root='~/.gluon/models', **kwargs)**

   Inception v3 model from "Rethinking the Inception Architecture for
   Computer Vision" paper.

   :Parameters:
      * **pretrained** (*bool*, *default False*) -- Whether to load
        the pretrained weights for model.

      * **ctx** (*Context*, *defaultCPU*) -- The context in which
        to load the pretrained weights.

      * **root** (*str*, *default'~/.gluon/models'*) -- Location
        for keeping the model parameters.

**gluon.model_zoo.vision.mobilenet0_25(**kwargs)**

   MobileNet model from the "MobileNets: Efficient Convolutional
   Neural Networks for Mobile Vision Applications" paper, with width
   multiplier 0.25.

   :Parameters:
      * **pretrained** (*bool*, *default False*) -- Whether to load
        the pretrained weights for model.

      * **ctx** (*Context*, *defaultCPU*) -- The context in which
        to load the pretrained weights.

**gluon.model_zoo.vision.mobilenet0_5(**kwargs)**

   MobileNet model from the "MobileNets: Efficient Convolutional
   Neural Networks for Mobile Vision Applications" paper, with width
   multiplier 0.5.

   :Parameters:
      * **pretrained** (*bool*, *default False*) -- Whether to load
        the pretrained weights for model.

      * **ctx** (*Context*, *defaultCPU*) -- The context in which
        to load the pretrained weights.

**gluon.model_zoo.vision.mobilenet0_75(**kwargs)**

   MobileNet model from the "MobileNets: Efficient Convolutional
   Neural Networks for Mobile Vision Applications" paper, with width
   multiplier 0.75.

   :Parameters:
      * **pretrained** (*bool*, *default False*) -- Whether to load
        the pretrained weights for model.

      * **ctx** (*Context*, *defaultCPU*) -- The context in which
        to load the pretrained weights.

**gluon.model_zoo.vision.mobilenet1_0(**kwargs)**

   MobileNet model from the "MobileNets: Efficient Convolutional
   Neural Networks for Mobile Vision Applications" paper, with width
   multiplier 1.0.

   :Parameters:
      * **pretrained** (*bool*, *default False*) -- Whether to load
        the pretrained weights for model.

      * **ctx** (*Context*, *defaultCPU*) -- The context in which
        to load the pretrained weights.

**gluon.model_zoo.vision.resnet101_v1(**kwargs)**

   ResNet-101 V1 model from "Deep Residual Learning for Image
   Recognition" paper.

   :Parameters:
      * **pretrained** (*bool*, *default False*) -- Whether to load
        the pretrained weights for model.

      * **ctx** (*Context*, *defaultCPU*) -- The context in which
        to load the pretrained weights.

      * **root** (*str*, *default'~/.gluon/models'*) -- Location
        for keeping the model parameters.

**gluon.model_zoo.vision.resnet101_v2(**kwargs)**

   ResNet-101 V2 model from "Identity Mappings in Deep Residual
   Networks" paper.

   :Parameters:
      * **pretrained** (*bool*, *default False*) -- Whether to load
        the pretrained weights for model.

      * **ctx** (*Context*, *defaultCPU*) -- The context in which
        to load the pretrained weights.

      * **root** (*str*, *default'~/.gluon/models'*) -- Location
        for keeping the model parameters.

**gluon.model_zoo.vision.resnet152_v1(**kwargs)**

   ResNet-152 V1 model from "Deep Residual Learning for Image
   Recognition" paper.

   :Parameters:
      * **pretrained** (*bool*, *default False*) -- Whether to load
        the pretrained weights for model.

      * **ctx** (*Context*, *defaultCPU*) -- The context in which
        to load the pretrained weights.

      * **root** (*str*, *default'~/.gluon/models'*) -- Location
        for keeping the model parameters.

**gluon.model_zoo.vision.resnet152_v2(**kwargs)**

   ResNet-152 V2 model from "Identity Mappings in Deep Residual
   Networks" paper.

   :Parameters:
      * **pretrained** (*bool*, *default False*) -- Whether to load
        the pretrained weights for model.

      * **ctx** (*Context*, *defaultCPU*) -- The context in which
        to load the pretrained weights.

      * **root** (*str*, *default'~/.gluon/models'*) -- Location
        for keeping the model parameters.

**gluon.model_zoo.vision.resnet18_v1(**kwargs)**

   ResNet-18 V1 model from "Deep Residual Learning for Image
   Recognition" paper.

   :Parameters:
      * **pretrained** (*bool*, *default False*) -- Whether to load
        the pretrained weights for model.

      * **ctx** (*Context*, *defaultCPU*) -- The context in which
        to load the pretrained weights.

      * **root** (*str*, *default'~/.gluon/models'*) -- Location
        for keeping the model parameters.

**gluon.model_zoo.vision.resnet18_v2(**kwargs)**

   ResNet-18 V2 model from "Identity Mappings in Deep Residual
   Networks" paper.

   :Parameters:
      * **pretrained** (*bool*, *default False*) -- Whether to load
        the pretrained weights for model.

      * **ctx** (*Context*, *defaultCPU*) -- The context in which
        to load the pretrained weights.

      * **root** (*str*, *default'~/.gluon/models'*) -- Location
        for keeping the model parameters.

**gluon.model_zoo.vision.resnet34_v1(**kwargs)**

   ResNet-34 V1 model from "Deep Residual Learning for Image
   Recognition" paper.

   :Parameters:
      * **pretrained** (*bool*, *default False*) -- Whether to load
        the pretrained weights for model.

      * **ctx** (*Context*, *defaultCPU*) -- The context in which
        to load the pretrained weights.

      * **root** (*str*, *default'~/.gluon/models'*) -- Location
        for keeping the model parameters.

**gluon.model_zoo.vision.resnet34_v2(**kwargs)**

   ResNet-34 V2 model from "Identity Mappings in Deep Residual
   Networks" paper.

   :Parameters:
      * **pretrained** (*bool*, *default False*) -- Whether to load
        the pretrained weights for model.

      * **ctx** (*Context*, *defaultCPU*) -- The context in which
        to load the pretrained weights.

      * **root** (*str*, *default'~/.gluon/models'*) -- Location
        for keeping the model parameters.

**gluon.model_zoo.vision.resnet50_v1(**kwargs)**

   ResNet-50 V1 model from "Deep Residual Learning for Image
   Recognition" paper.

   :Parameters:
      * **pretrained** (*bool*, *default False*) -- Whether to load
        the pretrained weights for model.

      * **ctx** (*Context*, *defaultCPU*) -- The context in which
        to load the pretrained weights.

      * **root** (*str*, *default'~/.gluon/models'*) -- Location
        for keeping the model parameters.

**gluon.model_zoo.vision.resnet50_v2(**kwargs)**

   ResNet-50 V2 model from "Identity Mappings in Deep Residual
   Networks" paper.

   :Parameters:
      * **pretrained** (*bool*, *default False*) -- Whether to load
        the pretrained weights for model.

      * **ctx** (*Context*, *defaultCPU*) -- The context in which
        to load the pretrained weights.

      * **root** (*str*, *default'~/.gluon/models'*) -- Location
        for keeping the model parameters.

**gluon.model_zoo.vision.squeezenet1_0(**kwargs)**

   SqueezeNet 1.0 model from the "SqueezeNet: AlexNet-level accuracy
   with 50x fewer parameters and <0.5MB model size" paper.

   :Parameters:
      * **pretrained** (*bool*, *default False*) -- Whether to load
        the pretrained weights for model.

      * **ctx** (*Context*, *defaultCPU*) -- The context in which
        to load the pretrained weights.

      * **root** (*str*, *default'~/.gluon/models'*) -- Location
        for keeping the model parameters.

**gluon.model_zoo.vision.squeezenet1_1(**kwargs)**

   SqueezeNet 1.1 model from the official SqueezeNet repo. SqueezeNet
   1.1 has 2.4x less computation and slightly fewer parameters than
   SqueezeNet 1.0, without sacrificing accuracy.

   :Parameters:
      * **pretrained** (*bool*, *default False*) -- Whether to load
        the pretrained weights for model.

      * **ctx** (*Context*, *defaultCPU*) -- The context in which
        to load the pretrained weights.

      * **root** (*str*, *default'~/.gluon/models'*) -- Location
        for keeping the model parameters.

**gluon.model_zoo.vision.vgg11(**kwargs)**

   VGG-11 model from the "Very Deep Convolutional Networks for
   Large-Scale Image Recognition" paper.

   :Parameters:
      * **pretrained** (*bool*, *default False*) -- Whether to load
        the pretrained weights for model.

      * **ctx** (*Context*, *defaultCPU*) -- The context in which
        to load the pretrained weights.

      * **root** (*str*, *default'~/.gluon/models'*) -- Location
        for keeping the model parameters.

**gluon.model_zoo.vision.vgg11_bn(**kwargs)**

   VGG-11 model with batch normalization from the "Very Deep
   Convolutional Networks for Large-Scale Image Recognition" paper.

   :Parameters:
      * **pretrained** (*bool*, *default False*) -- Whether to load
        the pretrained weights for model.

      * **ctx** (*Context*, *defaultCPU*) -- The context in which
        to load the pretrained weights.

      * **root** (*str*, *default'~/.gluon/models'*) -- Location
        for keeping the model parameters.

**gluon.model_zoo.vision.vgg13(**kwargs)**

   VGG-13 model from the "Very Deep Convolutional Networks for
   Large-Scale Image Recognition" paper.

   :Parameters:
      * **pretrained** (*bool*, *default False*) -- Whether to load
        the pretrained weights for model.

      * **ctx** (*Context*, *defaultCPU*) -- The context in which
        to load the pretrained weights.

      * **root** (*str*, *default'~/.gluon/models'*) -- Location
        for keeping the model parameters.

**gluon.model_zoo.vision.vgg13_bn(**kwargs)**

   VGG-13 model with batch normalization from the "Very Deep
   Convolutional Networks for Large-Scale Image Recognition" paper.

   :Parameters:
      * **pretrained** (*bool*, *default False*) -- Whether to load
        the pretrained weights for model.

      * **ctx** (*Context*, *defaultCPU*) -- The context in which
        to load the pretrained weights.

      * **root** (*str*, *default'~/.gluon/models'*) -- Location
        for keeping the model parameters.

**gluon.model_zoo.vision.vgg16(**kwargs)**

   VGG-16 model from the "Very Deep Convolutional Networks for
   Large-Scale Image Recognition" paper.

   :Parameters:
      * **pretrained** (*bool*, *default False*) -- Whether to load
        the pretrained weights for model.

      * **ctx** (*Context*, *defaultCPU*) -- The context in which
        to load the pretrained weights.

      * **root** (*str*, *default'~/.gluon/models'*) -- Location
        for keeping the model parameters.

**gluon.model_zoo.vision.vgg16_bn(**kwargs)**

   VGG-16 model with batch normalization from the "Very Deep
   Convolutional Networks for Large-Scale Image Recognition" paper.

   :Parameters:
      * **pretrained** (*bool*, *default False*) -- Whether to load
        the pretrained weights for model.

      * **ctx** (*Context*, *defaultCPU*) -- The context in which
        to load the pretrained weights.

      * **root** (*str*, *default'~/.gluon/models'*) -- Location
        for keeping the model parameters.

**gluon.model_zoo.vision.vgg19(**kwargs)**

   VGG-19 model from the "Very Deep Convolutional Networks for
   Large-Scale Image Recognition" paper.

   :Parameters:
      * **pretrained** (*bool*, *default False*) -- Whether to load
        the pretrained weights for model.

      * **ctx** (*Context*, *defaultCPU*) -- The context in which
        to load the pretrained weights.

      * **root** (*str*, *default'~/.gluon/models'*) -- Location
        for keeping the model parameters.

**gluon.model_zoo.vision.vgg19_bn(**kwargs)**

   VGG-19 model with batch normalization from the "Very Deep
   Convolutional Networks for Large-Scale Image Recognition" paper.

   :Parameters:
      * **pretrained** (*bool*, *default False*) -- Whether to load
        the pretrained weights for model.

      * **ctx** (*Context*, *defaultCPU*) -- The context in which
        to load the pretrained weights.

      * **root** (*str*, *default'~/.gluon/models'*) -- Location
        for keeping the model parameters.
