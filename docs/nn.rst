
Gluon Neural Network Layers API
*******************************


Overview
========

This document lists the neural network blocks in Gluon:


Basic Layers
============

+----------------+--------------------------------------------------------------------------------------------+
| ``Dense``      | Just your regular densely-connected NN layer.                                              |
+----------------+--------------------------------------------------------------------------------------------+
| ``Activation`` | Applies an activation function to input.                                                   |
+----------------+--------------------------------------------------------------------------------------------+
| ``Dropout``    | Applies Dropout to the input.                                                              |
+----------------+--------------------------------------------------------------------------------------------+
| ``BatchNorm``  | Batch normalization layer (Ioffe and Szegedy, 2014).                                       |
+----------------+--------------------------------------------------------------------------------------------+
| ``LeakyReLU``  | Leaky version of a Rectified Linear Unit.                                                  |
+----------------+--------------------------------------------------------------------------------------------+
| ``Embedding``  | Turns non-negative integers (indexes/tokens) into dense vectors of fixed size.             |
+----------------+--------------------------------------------------------------------------------------------+


Convolutional Layers
====================

+---------------------+--------------------------------------------------------------------------------------------+
| ``Conv1D``          | 1D convolution layer (e.g. temporal convolution).                                          |
+---------------------+--------------------------------------------------------------------------------------------+
| ``Conv2D``          | 2D convolution layer (e.g. spatial convolution over images).                               |
+---------------------+--------------------------------------------------------------------------------------------+
| ``Conv3D``          | 3D convolution layer (e.g. spatial convolution over volumes).                              |
+---------------------+--------------------------------------------------------------------------------------------+
| ``Conv1DTranspose`` | Transposed 1D convolution layer (sometimes called Deconvolution).                          |
+---------------------+--------------------------------------------------------------------------------------------+
| ``Conv2DTranspose`` | Transposed 2D convolution layer (sometimes called Deconvolution).                          |
+---------------------+--------------------------------------------------------------------------------------------+
| ``Conv3DTranspose`` | Transposed 3D convolution layer (sometimes called Deconvolution).                          |
+---------------------+--------------------------------------------------------------------------------------------+


Pooling Layers
==============

+---------------------+--------------------------------------------------------------------------------------------+
| ``MaxPool1D``       | Max pooling operation for one dimensional data.                                            |
+---------------------+--------------------------------------------------------------------------------------------+
| ``MaxPool2D``       | Max pooling operation for two dimensional (spatial) data.                                  |
+---------------------+--------------------------------------------------------------------------------------------+
| ``MaxPool3D``       | Max pooling operation for 3D data (spatial or spatio-temporal).                            |
+---------------------+--------------------------------------------------------------------------------------------+
| ``AvgPool1D``       | Average pooling operation for temporal data.                                               |
+---------------------+--------------------------------------------------------------------------------------------+
| ``AvgPool2D``       | Average pooling operation for spatial data.                                                |
+---------------------+--------------------------------------------------------------------------------------------+
| ``AvgPool3D``       | Average pooling operation for 3D data (spatial or spatio-temporal).                        |
+---------------------+--------------------------------------------------------------------------------------------+
| ``GlobalMaxPool1D`` | Global max pooling operation for temporal data.                                            |
+---------------------+--------------------------------------------------------------------------------------------+
| ``GlobalMaxPool2D`` | Global max pooling operation for spatial data.                                             |
+---------------------+--------------------------------------------------------------------------------------------+
| ``GlobalMaxPool3D`` | Global max pooling operation for 3D data.                                                  |
+---------------------+--------------------------------------------------------------------------------------------+
| ``GlobalAvgPool1D`` | Global average pooling operation for temporal data.                                        |
+---------------------+--------------------------------------------------------------------------------------------+
| ``GlobalAvgPool2D`` | Global average pooling operation for spatial data.                                         |
+---------------------+--------------------------------------------------------------------------------------------+
| ``GlobalAvgPool3D`` | Global max pooling operation for 3D data.                                                  |
+---------------------+--------------------------------------------------------------------------------------------+


API Reference
=============

Neural network layers.

**class gluon.nn.Activation(activation, **kwargs)**

   Applies an activation function to input.

   :Parameters:
      **activation** (*str*) -- Name of activation function to use.
      See `Activation()
      <https://mxnet.incubator.apache.org/versions/master/api/python/ndarray/ndarray.html#mxnet.ndarray.Activation>`_ for
      available choices.

   Input shape:
      Arbitrary.

   Output shape:
      Same shape as input.

**class gluon.nn.AvgPool1D(pool_size=2, strides=None, padding=0,
layout='NCW', ceil_mode=False, **kwargs)**

   Average pooling operation for temporal data.

   :Parameters:
      * **pool_size** (*int*) -- Size of the max pooling windows.

      * **strides** (*int*, or *None*) -- Factor by which to
        downscale. E.g. 2 will halve the input size. If *None*, it
        will default to *pool_size*.

      * **padding** (*int*) -- If padding is non-zero, then the input
        is implicitly zero-padded on both sides for padding number of
        points.

      * **layout** (*str*, *default 'NCW'*) -- Dimension ordering of
        data and weight. Can be 'NCW', 'NWC', etc. 'N', 'C', 'W'
        stands for batch, channel, and width (time) dimensions
        respectively. padding is applied on 'W' dimension.

      * **ceil_mode** (*bool*, *default False*) -- When *True*, will
        use ceil instead of floor to compute the output shape.

   Input shape:
      This depends on the *layout* parameter. Input is 3D array of
      shape (batch_size, channels, width) if *layout* is *NCW*.

   Output shape:
      This depends on the *layout* parameter. Output is 3D array of
      shape (batch_size, channels, out_width) if *layout* is *NCW*.

      out_width is calculated as:

      ::

         out_width = floor((width+2*padding-pool_size)/strides)+1

      When *ceil_mode* is *True*, ceil will be used instead of floor
      in this equation.

**class gluon.nn.AvgPool2D(pool_size=(2, 2), strides=None,
padding=0, ceil_mode=False, layout='NCHW', **kwargs)**

   Average pooling operation for spatial data.

   :Parameters:
      * **pool_size** (*int or list/tuple of 2 ints*,) -- Size
        of the max pooling windows.

      * **strides** (*int*, *list/tuple of 2 ints*, or *None.*) --
        Factor by which to downscale. E.g. 2 will halve the input
        size. If *None*, it will default to *pool_size*.

      * **padding** (*int* or *list/tuple of 2 ints*,) -- If
        padding is non-zero, then the input is implicitly zero-padded
        on both sides for padding number of points.

      * **layout** (*str*, *default 'NCHW'*) -- Dimension ordering
        of data and weight. Can be 'NCHW', 'NHWC', etc. 'N', 'C', 'H',
        'W' stands for batch, channel, height, and width dimensions
        respectively. padding is applied on 'H' and 'W' dimension.

      * **ceil_mode** (*bool*, *default False*) -- When True, will
        use ceil instead of floor to compute the output shape.

   Input shape:
      This depends on the *layout* parameter. Input is 4D array of
      shape (batch_size, channels, height, width) if *layout* is
      *NCHW*.

   Output shape:
      This depends on the *layout* parameter. Output is 4D array of
      shape (batch_size, channels, out_height, out_width)  if *layout*
      is *NCHW*.

      out_height and out_width are calculated as:

      ::

         out_height = floor((height+2*padding[0]-pool_size[0])/strides[0])+1
         out_width = floor((width+2*padding[1]-pool_size[1])/strides[1])+1

      When *ceil_mode* is *True*, ceil will be used instead of floor
      in this equation.

**class gluon.nn.AvgPool3D(pool_size=(2, 2, 2), strides=None,
padding=0, ceil_mode=False, layout='NCDHW', **kwargs)**

   Average pooling operation for 3D data (spatial or spatio-temporal).

   :Parameters:
      * **pool_size** (*int* or *list/tuple of 3 ints*,) -- Size
        of the max pooling windows.

      * **strides** (*int*, *list/tuple of 3 ints*, or *None.*) --
        Factor by which to downscale. E.g. 2 will halve the input
        size. If *None*, it will default to *pool_size*.

      * **padding** (*int* or *list/tuple of 3 ints*,) -- If
        padding is non-zero, then the input is implicitly zero-padded
        on both sides for padding number of points.

      * **layout** (*str*, *default 'NCDHW'*) -- Dimension ordering
        of data and weight. Can be 'NCDHW', 'NDHWC', etc. 'N', 'C',
        'H', 'W', 'D' stands for batch, channel, height, width and
        depth dimensions respectively. padding is applied on 'D', 'H'
        and 'W' dimension.

      * **ceil_mode** (*bool*, *default False*) -- When True, will
        use ceil instead of floor to compute the output shape.

   Input shape:
      This depends on the *layout* parameter. Input is 5D array of
      shape (batch_size, channels, depth, height, width) if *layout*
      is *NCDHW*.

   Output shape:
      This depends on the *layout* parameter. Output is 5D array of
      shape (batch_size, channels, out_depth, out_height, out_width)
      if *layout* is *NCDHW*.

      out_depth, out_height and out_width are calculated as

      ::

         out_depth = floor((depth+2*padding[0]-pool_size[0])/strides[0])+1
         out_height = floor((height+2*padding[1]-pool_size[1])/strides[1])+1
         out_width = floor((width+2*padding[2]-pool_size[2])/strides[2])+1

      When *ceil_mode* is *True,* ceil will be used instead of floor
      in this equation.

**class gluon.nn.BatchNorm(axis=1, momentum=0.9, epsilon=1e-05,
center=True, scale=True, beta_initializer='zeros',
gamma_initializer='ones', running_mean_initializer='zeros',
running_variance_initializer='ones', in_channels=0, **kwargs)**

   Batch normalization layer (Ioffe and Szegedy, 2014). Normalizes the
   input at each batch, i.e. applies a transformation that maintains
   the mean activation close to 0 and the activation standard
   deviation close to 1.

   :Parameters:
      * **axis** (*int*, *default 1*) -- The axis that should be
        normalized. This is typically the channels (C) axis. For
        instance, after a *Conv2D* layer with *layout='NCHW'*, set
        *axis=1* in *BatchNorm*. If *layout='NHWC'*, then set
        *axis=3*.

      * **momentum** (*float*, *default 0.9*) -- Momentum for the
        moving average.

      * **epsilon** (*float*, *default 1e-5*) -- Small float added
        to variance to avoid dividing by zero.

      * **center** (*bool*, *default True*) -- If True, add offset
        of *beta* to normalized tensor. If False, *beta* is ignored.

      * **scale** (*bool*, *default True*) -- If True, multiply by
        *gamma*. If False, *gamma* is not used. When the next layer is
        linear (also e.g. *nn.relu*), this can be disabled since the
        scaling will be done by the next layer.

      * **beta_initializer** (str or *Initializer*, default 'zeros')
        -- Initializer for the beta weight.

      * **gamma_initializer** (str or *Initializer*, default 'ones')
        -- Initializer for the gamma weight.

      * **moving_mean_initializer** (str or *Initializer*, default
        'zeros') -- Initializer for the moving mean.

      * **moving_variance_initializer** (str or *Initializer*, default
        'ones') -- Initializer for the moving variance.

      * **in_channels** (*int*, *default 0*) -- Number of channels
        (feature maps) in input data. If not specified, initialization
        will be deferred to the first time *forward* is called and
        *in_channels* will be inferred from the shape of input data.

   Input shape:
      Arbitrary.

   Output shape:
      Same shape as input.

**class gluon.nn.Conv1D(channels, kernel_size, strides=1,
padding=0, dilation=1, groups=1, layout='NCW', activation=None,
use_bias=True, weight_initializer=None, bias_initializer='zeros',
in_channels=0, **kwargs)**

   1D convolution layer (e.g. temporal convolution).

   This layer creates a convolution kernel that is convolved with the
   layer input over a single spatial (or temporal) dimension to
   produce a tensor of outputs. If *use_bias* is True, a bias vector
   is created and added to the outputs. Finally, if *activation* is
   not *None*, it is applied to the outputs as well.

   If *in_channels* is not specified, *Parameter* initialization will
   be deferred to the first time *forward* is called and *in_channels*
   will be inferred from the shape of input data.

   :Parameters:
      * **channels** (*int*) -- The dimensionality of the output
        space, i.e. the number of output channels (filters) in the
        convolution.

      * **kernel_size** (*int* or *tuple/list of 1 int*) --
        Specifies the dimensions of the convolution window.

      * **strides** (*int* or *tuple/list of 1 int*) -- Specify
        the strides of the convolution.

      * **padding** (*int* or *a tuple/list of 1 int*,) -- If
        padding is non-zero, then the input is implicitly zero-padded
        on both sides for padding number of points

      * **dilation** (*int* or *tuple/list of 1 int*) -- Specifies
        the dilation rate to use for dilated convolution.

      * **groups** (*int*) -- Controls the connections between inputs
        and outputs. At groups=1, all inputs are convolved to all
        outputs. At groups=2, the operation becomes equivalent to
        having two conv layers side by side, each seeing half the
        input channels, and producing half the output channels, and
        both subsequently concatenated.

      * **layout** (*str*, *default 'NCW'*) -- Dimension ordering of
        data and weight. Can be 'NCW', 'NWC', etc. 'N', 'C', 'W'
        stands for batch, channel, and width (time) dimensions
        respectively. Convolution is applied on the 'W' dimension.

      * **in_channels** (*int*, *default 0*) -- The number of input
        channels to this layer. If not specified, initialization will
        be deferred to the first time *forward* is called and
        *in_channels* will be inferred from the shape of input data.

      * **activation** (*str*) -- Activation function to use. See
        `Activation()
        <https://mxnet.incubator.apache.org/versions/master/api/python/ndarray/ndarray.html#mxnet.ndarray.Activation>`_. If you
        don't specify anything, no activation is applied (ie. "linear"
        activation: *a(x) = x*).

      * **use_bias** (*bool*) -- Whether the layer uses a bias vector.

      * **weight_initializer** (str or *Initializer*) -- Initializer
        for the *weight* weights matrix.

      * **bias_initializer** (str or *Initializer*) -- Initializer for
        the bias vector.

   Input shape:
      This depends on the *layout* parameter. Input is 3D array of
      shape (batch_size, in_channels, width) if *layout* is *NCW*.

   Output shape:
      This depends on the *layout* parameter. Output is 3D array of
      shape (batch_size, channels, out_width) if *layout* is *NCW*.
      out_width is calculated as:

      ::

         out_width = floor((width+2*padding-dilation*(kernel_size-1)-1)/stride)+1

**class gluon.nn.Conv1DTranspose(channels, kernel_size,
strides=1, padding=0, output_padding=0, dilation=1, groups=1,
layout='NCW', activation=None, use_bias=True, weight_initializer=None,
bias_initializer='zeros', in_channels=0, **kwargs)**

   Transposed 1D convolution layer (sometimes called Deconvolution).

   The need for transposed convolutions generally arises from the
   desire to use a transformation going in the opposite direction of a
   normal convolution, i.e., from something that has the shape of the
   output of some convolution to something that has the shape of its
   input while maintaining a connectivity pattern that is compatible
   with said convolution.

   If *in_channels* is not specified, *Parameter* initialization will
   be deferred to the first time *forward* is called and *in_channels*
   will be inferred from the shape of input data.

   :Parameters:
      * **channels** (*int*) -- The dimensionality of the output
        space, i.e. the number of output channels (filters) in the
        convolution.

      * **kernel_size** (*int* or *tuple/list of 3 int*) --
        Specifies the dimensions of the convolution window.

      * **strides** (*int* or *tuple/list of 3 int*,) -- Specify
        the strides of the convolution.

      * **padding** (*int* or *a tuple/list of 3 int*,) -- If
        padding is non-zero, then the input is implicitly zero-padded
        on both sides for padding number of points

      * **dilation** (*int* or *tuple/list of 3 int*) -- Specifies
        the dilation rate to use for dilated convolution.

      * **groups** (*int*) -- Controls the connections between inputs
        and outputs. At groups=1, all inputs are convolved to all
        outputs. At groups=2, the operation becomes equivalent to
        having two conv layers side by side, each seeing half the
        input channels, and producing half the output channels, and
        both subsequently concatenated.

      * **layout** (*str*, *default 'NCW'*) -- Dimension ordering of
        data and weight. Can be 'NCW', 'NWC', etc. 'N', 'C', 'W'
        stands for batch, channel, and width (time) dimensions
        respectively. Convolution is applied on the 'W' dimension.

      * **in_channels** (*int*, *default 0*) -- The number of input
        channels to this layer. If not specified, initialization will
        be deferred to the first time *forward* is called and
        *in_channels* will be inferred from the shape of input data.

      * **activation** (*str*) -- Activation function to use. See
        `Activation()
        <https://mxnet.incubator.apache.org/versions/master/api/python/ndarray/ndarray.html#mxnet.ndarray.Activation>`_. If you
        don't specify anything, no activation is applied (ie. "linear"
        activation: *a(x) = x*).

      * **use_bias** (*bool*) -- Whether the layer uses a bias vector.

      * **weight_initializer** (str or *Initializer*) -- Initializer
        for the *weight* weights matrix.

      * **bias_initializer** (str or *Initializer*) -- Initializer for
        the bias vector.

   Input shape:
      This depends on the *layout* parameter. Input is 3D array of
      shape (batch_size, in_channels, width) if *layout* is *NCW*.

   Output shape:
      This depends on the *layout* parameter. Output is 3D array of
      shape (batch_size, channels, out_width) if *layout* is *NCW*.

      out_width is calculated as:

      ::

         out_width = (width-1)*strides-2*padding+kernel_size+output_padding

**class gluon.nn.Conv2D(channels, kernel_size, strides=(1, 1),
padding=(0, 0), dilation=(1, 1), groups=1, layout='NCHW',
activation=None, use_bias=True, weight_initializer=None,
bias_initializer='zeros', in_channels=0, **kwargs)**

   2D convolution layer (e.g. spatial convolution over images).

   This layer creates a convolution kernel that is convolved with the
   layer input to produce a tensor of outputs. If *use_bias* is True,
   a bias vector is created and added to the outputs. Finally, if
   *activation* is not *None*, it is applied to the outputs as well.

   If *in_channels* is not specified, *Parameter* initialization will
   be deferred to the first time *forward* is called and *in_channels*
   will be inferred from the shape of input data.

   :Parameters:
      * **channels** (*int*) -- The dimensionality of the output
        space, i.e. the number of output channels (filters) in the
        convolution.

      * **kernel_size** (*int* or *tuple/list of 2 int*) --
        Specifies the dimensions of the convolution window.

      * **strides** (*int* or *tuple/list of 2 int*,) -- Specify
        the strides of the convolution.

      * **padding** (*int* or *a tuple/list of 2 int*,) -- If
        padding is non-zero, then the input is implicitly zero-padded
        on both sides for padding number of points

      * **dilation** (*int* or *tuple/list of 2 int*) -- Specifies
        the dilation rate to use for dilated convolution.

      * **groups** (*int*) -- Controls the connections between inputs
        and outputs. At groups=1, all inputs are convolved to all
        outputs. At groups=2, the operation becomes equivalent to
        having two conv layers side by side, each seeing half the
        input channels, and producing half the output channels, and
        both subsequently concatenated.

      * **layout** (*str*, *default 'NCHW'*) -- Dimension ordering
        of data and weight. Can be 'NCHW', 'NHWC', etc. 'N', 'C', 'H',
        'W' stands for batch, channel, height, and width dimensions
        respectively. Convolution is applied on the 'H' and 'W'
        dimensions.

      * **in_channels** (*int*, *default 0*) -- The number of input
        channels to this layer. If not specified, initialization will
        be deferred to the first time *forward* is called and
        *in_channels* will be inferred from the shape of input data.

      * **activation** (*str*) -- Activation function to use. See
        `Activation()
        <https://mxnet.incubator.apache.org/versions/master/api/python/ndarray/ndarray.html#mxnet.ndarray.Activation>`_. If you
        don't specify anything, no activation is applied (ie. "linear"
        activation: *a(x) = x*).

      * **use_bias** (*bool*) -- Whether the layer uses a bias vector.

      * **weight_initializer** (str or *Initializer*) -- Initializer
        for the *weight* weights matrix.

      * **bias_initializer** (str or *Initializer*) -- Initializer for
        the bias vector.

   Input shape:
      This depends on the *layout* parameter. Input is 4D array of
      shape (batch_size, in_channels, height, width) if *layout* is
      *NCHW*.

   Output shape:
      This depends on the *layout* parameter. Output is 4D array of
      shape (batch_size, channels, out_height, out_width) if *layout*
      is *NCHW*.

      out_height and out_width are calculated as:

      ::

         out_height = floor((height+2*padding[0]-dilation[0]*(kernel_size[0]-1)-1)/stride[0])+1
         out_width = floor((width+2*padding[1]-dilation[1]*(kernel_size[1]-1)-1)/stride[1])+1

**class gluon.nn.Conv2DTranspose(channels, kernel_size,
strides=(1, 1), padding=(0, 0), output_padding=(0, 0), dilation=(1,
1), groups=1, layout='NCHW', activation=None, use_bias=True,
weight_initializer=None, bias_initializer='zeros', in_channels=0,
**kwargs)**

   Transposed 2D convolution layer (sometimes called Deconvolution).

   The need for transposed convolutions generally arises from the
   desire to use a transformation going in the opposite direction of a
   normal convolution, i.e., from something that has the shape of the
   output of some convolution to something that has the shape of its
   input while maintaining a connectivity pattern that is compatible
   with said convolution.

   If *in_channels* is not specified, *Parameter* initialization will
   be deferred to the first time *forward* is called and *in_channels*
   will be inferred from the shape of input data.

   :Parameters:
      * **channels** (*int*) -- The dimensionality of the output
        space, i.e. the number of output channels (filters) in the
        convolution.

      * **kernel_size** (*int* or *tuple/list of 3 int*) --
        Specifies the dimensions of the convolution window.

      * **strides** (*int* or *tuple/list of 3 int*,) -- Specify
        the strides of the convolution.

      * **padding** (*int* or *a tuple/list of 3 int*,) -- If
        padding is non-zero, then the input is implicitly zero-padded
        on both sides for padding number of points

      * **dilation** (*int* or *tuple/list of 3 int*) -- Specifies
        the dilation rate to use for dilated convolution.

      * **groups** (*int*) -- Controls the connections between inputs
        and outputs. At groups=1, all inputs are convolved to all
        outputs. At groups=2, the operation becomes equivalent to
        having two conv layers side by side, each seeing half the
        input channels, and producing half the output channels, and
        both subsequently concatenated.

      * **layout** (*str*, *default 'NCHW'*) -- Dimension ordering
        of data and weight. Can be 'NCHW', 'NHWC', etc. 'N', 'C', 'H',
        'W' stands for batch, channel, height, and width dimensions
        respectively. Convolution is applied on the 'H' and 'W'
        dimensions.

      * **in_channels** (*int*, *default 0*) -- The number of input
        channels to this layer. If not specified, initialization will
        be deferred to the first time *forward* is called and
        *in_channels* will be inferred from the shape of input data.

      * **activation** (*str*) -- Activation function to use. See
        `Activation()
        <https://mxnet.incubator.apache.org/versions/master/api/python/ndarray/ndarray.html#mxnet.ndarray.Activation>`_. If you
        don't specify anything, no activation is applied (ie. "linear"
        activation: *a(x) = x*).

      * **use_bias** (*bool*) -- Whether the layer uses a bias vector.

      * **weight_initializer** (str or *Initializer*) -- Initializer
        for the *weight* weights matrix.

      * **bias_initializer** (str or *Initializer*) -- Initializer for
        the bias vector.

   Input shape:
      This depends on the *layout* parameter. Input is 4D array of
      shape (batch_size, in_channels, height, width) if *layout* is
      *NCHW*.

   Output shape:
      This depends on the *layout* parameter. Output is 4D array of
      shape (batch_size, channels, out_height, out_width) if *layout*
      is *NCHW*.

      out_height and out_width are calculated as:

      ::

         out_height = (height-1)*strides[0]-2*padding[0]+kernel_size[0]+output_padding[0]
         out_width = (width-1)*strides[1]-2*padding[1]+kernel_size[1]+output_padding[1]

**class gluon.nn.Conv3D(channels, kernel_size, strides=(1, 1,
1), padding=(0, 0, 0), dilation=(1, 1, 1), groups=1, layout='NCDHW',
activation=None, use_bias=True, weight_initializer=None,
bias_initializer='zeros', in_channels=0, **kwargs)**

   3D convolution layer (e.g. spatial convolution over volumes).

   This layer creates a convolution kernel that is convolved with the
   layer input to produce a tensor of outputs. If *use_bias* is
   *True*, a bias vector is created and added to the outputs. Finally,
   if *activation* is not *None*, it is applied to the outputs as
   well.

   If *in_channels* is not specified, *Parameter* initialization will
   be deferred to the first time *forward* is called and *in_channels*
   will be inferred from the shape of input data.

   :Parameters:
      * **channels** (*int*) -- The dimensionality of the output
        space, i.e. the number of output channels (filters) in the
        convolution.

      * **kernel_size** (*int* or *tuple/list of 3 int*) --
        Specifies the dimensions of the convolution window.

      * **strides** (*int* or *tuple/list of 3 int*,) -- Specify
        the strides of the convolution.

      * **padding** (*int* or *a tuple/list of 3 int*,) -- If
        padding is non-zero, then the input is implicitly zero-padded
        on both sides for padding number of points

      * **dilation** (*int* or *tuple/list of 3 int*) -- Specifies
        the dilation rate to use for dilated convolution.

      * **groups** (*int*) -- Controls the connections between inputs
        and outputs. At groups=1, all inputs are convolved to all
        outputs. At groups=2, the operation becomes equivalent to
        having two conv layers side by side, each seeing half the
        input channels, and producing half the output channels, and
        both subsequently concatenated.

      * **layout** (*str*, *default 'NCDHW'*) -- Dimension ordering
        of data and weight. Can be 'NCDHW', 'NDHWC', etc. 'N', 'C',
        'H', 'W', 'D' stands for batch, channel, height, width and
        depth dimensions respectively. Convolution is applied on the
        'D', 'H' and 'W' dimensions.

      * **in_channels** (*int*, *default 0*) -- The number of input
        channels to this layer. If not specified, initialization will
        be deferred to the first time *forward* is called and
        *in_channels* will be inferred from the shape of input data.

      * **activation** (*str*) -- Activation function to use. See
        `Activation()
        <https://mxnet.incubator.apache.org/versions/master/api/python/ndarray/ndarray.html#mxnet.ndarray.Activation>`_. If you
        don't specify anything, no activation is applied (ie. "linear"
        activation: *a(x) = x*).

      * **use_bias** (*bool*) -- Whether the layer uses a bias vector.

      * **weight_initializer** (str or *Initializer*) -- Initializer
        for the *weight* weights matrix.

      * **bias_initializer** (str or *Initializer*) -- Initializer for
        the bias vector.

   Input shape:
      This depends on the *layout* parameter. Input is 5D array of
      shape (batch_size, in_channels, depth, height, width) if
      *layout* is *NCDHW*.

   Output shape:
      This depends on the *layout* parameter. Output is 5D array of
      shape (batch_size, channels, out_depth, out_height, out_width)
      if *layout* is *NCDHW*.

      out_depth, out_height and out_width are calculated as:

      ::

         out_depth = floor((depth+2*padding[0]-dilation[0]*(kernel_size[0]-1)-1)/stride[0])+1
         out_height = floor((height+2*padding[1]-dilation[1]*(kernel_size[1]-1)-1)/stride[1])+1
         out_width = floor((width+2*padding[2]-dilation[2]*(kernel_size[2]-1)-1)/stride[2])+1

**class gluon.nn.Conv3DTranspose(channels, kernel_size,
strides=(1, 1, 1), padding=(0, 0, 0), output_padding=(0, 0, 0),
dilation=(1, 1, 1), groups=1, layout='NCDHW', activation=None,
use_bias=True, weight_initializer=None, bias_initializer='zeros',
in_channels=0, **kwargs)**

   Transposed 3D convolution layer (sometimes called Deconvolution).

   The need for transposed convolutions generally arises from the
   desire to use a transformation going in the opposite direction of a
   normal convolution, i.e., from something that has the shape of the
   output of some convolution to something that has the shape of its
   input while maintaining a connectivity pattern that is compatible
   with said convolution.

   If *in_channels* is not specified, *Parameter* initialization will
   be deferred to the first time *forward* is called and *in_channels*
   will be inferred from the shape of input data.

   :Parameters:
      * **channels** (*int*) -- The dimensionality of the output
        space, i.e. the number of output channels (filters) in the
        convolution.

      * **kernel_size** (*int* or *tuple/list of 3 int*) --
        Specifies the dimensions of the convolution window.

      * **strides** (*int* or *tuple/list of 3 int*,) -- Specify
        the strides of the convolution.

      * **padding** (*int* or *a tuple/list of 3 int*,) -- If
        padding is non-zero, then the input is implicitly zero-padded
        on both sides for padding number of points

      * **dilation** (*int* or *tuple/list of 3 int*) -- Specifies
        the dilation rate to use for dilated convolution.

      * **groups** (*int*) -- Controls the connections between inputs
        and outputs. At groups=1, all inputs are convolved to all
        outputs. At groups=2, the operation becomes equivalent to
        having two conv layers side by side, each seeing half the
        input channels, and producing half the output channels, and
        both subsequently concatenated.

      * **layout** (*str*, *default 'NCDHW'*) -- Dimension ordering
        of data and weight. Can be 'NCDHW', 'NDHWC', etc. 'N', 'C',
        'H', 'W', 'D' stands for batch, channel, height, width and
        depth dimensions respectively. Convolution is applied on the
        'D', 'H', and 'W' dimensions.

      * **in_channels** (*int*, *default 0*) -- The number of input
        channels to this layer. If not specified, initialization will
        be deferred to the first time *forward* is called and
        *in_channels* will be inferred from the shape of input data.

      * **activation** (*str*) -- Activation function to use. See
        `Activation()
        <https://mxnet.incubator.apache.org/versions/master/api/python/ndarray/ndarray.html#mxnet.ndarray.Activation>`_. If you
        don't specify anything, no activation is applied (ie. "linear"
        activation: *a(x) = x*).

      * **use_bias** (*bool*) -- Whether the layer uses a bias vector.

      * **weight_initializer** (str or *Initializer*) -- Initializer
        for the *weight* weights matrix.

      * **bias_initializer** (str or *Initializer*) -- Initializer for
        the bias vector.

   Input shape:
      This depends on the *layout* parameter. Input is 5D array of
      shape (batch_size, in_channels, depth, height, width) if
      *layout* is *NCDHW*.

   Output shape:
      This depends on the *layout* parameter. Output is 5D array of
      shape (batch_size, channels, out_depth, out_height, out_width)
      if *layout* is *NCDHW*. out_depth, out_height and out_width are
      calculated as:

      ::

         out_depth = (depth-1)*strides[0]-2*padding[0]+kernel_size[0]+output_padding[0]
         out_height = (height-1)*strides[1]-2*padding[1]+kernel_size[1]+output_padding[1]
         out_width = (width-1)*strides[2]-2*padding[2]+kernel_size[2]+output_padding[2]

**class gluon.nn.Dense(units, activation=None, use_bias=True,
flatten=True, weight_initializer=None, bias_initializer='zeros',
in_units=0, **kwargs)**

   Just your regular densely-connected NN layer.

   *Dense* implements the operation: *output = activation(dot(input,
   weight) + bias)* where *activation* is the element-wise activation
   function passed as the *activation* argument, *weight* is a weights
   matrix created by the layer, and *bias* is a bias vector created by
   the layer (only applicable if *use_bias* is *True*).

   Note: the input must be a tensor with rank 2. Use *flatten* to
   convert it to rank 2 manually if necessary.

   :Parameters:
      * **units** (*int*) -- Dimensionality of the output space.

      * **activation** (*str*) -- Activation function to use. See help
        on *Activation* layer. If you don't specify anything, no
        activation is applied (ie. "linear" activation: *a(x) = x*).

      * **use_bias** (*bool*) -- Whether the layer uses a bias vector.

      * **flatten** (*bool*) -- Whether the input tensor should be
        flattened. If true, all but the first axis of input data are
        collapsed together. If false, all but the last axis of input
        data are kept the same, and the transformation applies on the
        last axis.

      * **weight_initializer** (str or *Initializer*) -- Initializer
        for the *kernel* weights matrix.

      * **bias_initializer** (str or *Initializer*) -- Initializer for
        the bias vector.

      * **in_units** (*int*, *optional*) -- Size of the input data.
        If not specified, initialization will be deferred to the first
        time *forward* is called and *in_units* will be inferred from
        the shape of input data.

      * **prefix** (*str* or *None*) -- See document of *Block*.

      * **params** (*ParameterDict* or *None*) -- See document of
        *Block*.

   If ``flatten`` is set to be True, then the shapes are:

   Input shape:
      An N-D input with shape *(batch_size, x1, x2, ..., xn) with x1 *
      x2 * ... * xn equal to in_units*.

   Output shape:
      The output would have shape *(batch_size, units)*.

   If ``flatten`` is set to be false, then the shapes are:

   Input shape:
      An N-D input with shape *(x1, x2, ..., xn, in_units)*.

   Output shape:
      The output would have shape *(x1, x2, ..., xn, units)*.

**class gluon.nn.Dropout(rate, **kwargs)**

   Applies Dropout to the input.

   Dropout consists in randomly setting a fraction *rate* of input
   units to 0 at each update during training time, which helps prevent
   overfitting.

   :Parameters:
      **rate** (*float*) -- Fraction of the input units to drop. Must
      be a number between 0 and 1.

   Input shape:
      Arbitrary.

   Output shape:
      Same shape as input.

   -[ References ]-

   Dropout: A Simple Way to Prevent Neural Networks from Overfitting

**class gluon.nn.Embedding(input_dim, output_dim,
dtype='float32', weight_initializer=None, **kwargs)**

   Turns non-negative integers (indexes/tokens) into dense vectors of
   fixed size. eg. [[4], [20]] -> [[0.25, 0.1], [0.6, -0.2]]

   :Parameters:
      * **input_dim** (*int*) -- Size of the vocabulary, i.e. maximum
        integer index + 1.

      * **output_dim** (*int*) -- Dimension of the dense embedding.

      * **dtype** (*str* or *np.dtype*, *default 'float32'*) --
        Data type of output embeddings.

      * **weight_initializer** (`Initializer
        <https://mxnet.incubator.apache.org/versions/master/api/python/optimization/optimization.html#mxnet.initializer.Initializer>`_)
        -- Initializer for the *embeddings* matrix.

   Input shape:
      2D tensor with shape: *(N, M)*.

   Output shape:
      3D tensor with shape: *(N, M, output_dim)*.

**class gluon.nn.Flatten(**kwargs)**

   Flattens the input to two dimensional.

   Input shape:
      Arbitrary shape *(N, a, b, c, ...)*

   Output shape:
      2D tensor with shape: *(N, a*b*c...)*

**class gluon.nn.GlobalAvgPool1D(layout='NCW', **kwargs)**

   Global average pooling operation for temporal data.

**class gluon.nn.GlobalAvgPool2D(layout='NCHW', **kwargs)**

   Global average pooling operation for spatial data.

**class gluon.nn.GlobalAvgPool3D(layout='NCDHW', **kwargs)**

   Global max pooling operation for 3D data.

**class gluon.nn.GlobalMaxPool1D(layout='NCW', **kwargs)**

   Global max pooling operation for temporal data.

**class gluon.nn.GlobalMaxPool2D(layout='NCHW', **kwargs)**

   Global max pooling operation for spatial data.

**class gluon.nn.GlobalMaxPool3D(layout='NCDHW', **kwargs)**

   Global max pooling operation for 3D data.

**class gluon.nn.LeakyReLU(alpha, **kwargs)**

   Leaky version of a Rectified Linear Unit.

   It allows a small gradient when the unit is not active

      f\left(x\right) = \left\{     \begin{array}{lr}        \alpha x
      & : x \lt 0 \\               x & : x \geq 0 \\     \end{array}
      \right.\\

   :Parameters:
      **alpha** (*float*) -- slope coefficient for the negative half
      axis. Must be >= 0.

   Input shape:
      Arbitrary.

   Output shape:
      Same shape as input.

**class gluon.nn.MaxPool1D(pool_size=2, strides=None, padding=0,
layout='NCW', ceil_mode=False, **kwargs)**

   Max pooling operation for one dimensional data.

   :Parameters:
      * **pool_size** (*int*) -- Size of the max pooling windows.

      * **strides** (*int*, or *None*) -- Factor by which to
        downscale. E.g. 2 will halve the input size. If *None*, it
        will default to *pool_size*.

      * **padding** (*int*) -- If padding is non-zero, then the input
        is implicitly zero-padded on both sides for padding number of
        points.

      * **layout** (*str*, *default 'NCW'*) -- Dimension ordering of
        data and weight. Can be 'NCW', 'NWC', etc. 'N', 'C', 'W'
        stands for batch, channel, and width (time) dimensions
        respectively. Pooling is applied on the W dimension.

      * **ceil_mode** (*bool**, **default False*) -- When *True*, will
        use ceil instead of floor to compute the output shape.

   Input shape:
      This depends on the *layout* parameter. Input is 3D array of
      shape (batch_size, channels, width) if *layout* is *NCW*.

   Output shape:
      This depends on the *layout* parameter. Output is 3D array of
      shape (batch_size, channels, out_width) if *layout* is *NCW*.

      out_width is calculated as:

      ::

         out_width = floor((width+2*padding-pool_size)/strides)+1

      When *ceil_mode* is *True*, ceil will be used instead of floor
      in this equation.

**class gluon.nn.MaxPool2D(pool_size=(2, 2), strides=None,
padding=0, layout='NCHW', ceil_mode=False, **kwargs)**

   Max pooling operation for two dimensional (spatial) data.

   :Parameters:
      * **pool_size** (*int* or *list/tuple of 2 ints*,) -- Size
        of the max pooling windows.

      * **strides** (*int*, *list/tuple of 2 ints*, or *None.*) --
        Factor by which to downscale. E.g. 2 will halve the input
        size. If *None*, it will default to *pool_size*.

      * **padding** (*int* or *list/tuple of 2 ints*,) -- If
        padding is non-zero, then the input is implicitly zero-padded
        on both sides for padding number of points.

      * **layout** (*str*, *default 'NCHW'*) -- Dimension ordering
        of data and weight. Can be 'NCHW', 'NHWC', etc. 'N', 'C', 'H',
        'W' stands for batch, channel, height, and width dimensions
        respectively. padding is applied on 'H' and 'W' dimension.

      * **ceil_mode** (*bool*, *default False*) -- When *True*, will
        use ceil instead of floor to compute the output shape.

   Input shape:
      This depends on the *layout* parameter. Input is 4D array of
      shape (batch_size, channels, height, width) if *layout* is
      *NCHW*.

   Output shape:
      This depends on the *layout* parameter. Output is 4D array of
      shape (batch_size, channels, out_height, out_width)  if *layout*
      is *NCHW*.

      out_height and out_width are calculated as:

      ::

         out_height = floor((height+2*padding[0]-pool_size[0])/strides[0])+1
         out_width = floor((width+2*padding[1]-pool_size[1])/strides[1])+1

      When *ceil_mode* is *True*, ceil will be used instead of floor
      in this equation.

**class gluon.nn.MaxPool3D(pool_size=(2, 2, 2), strides=None,
padding=0, ceil_mode=False, layout='NCDHW', **kwargs)**

   Max pooling operation for 3D data (spatial or spatio-temporal).

   :Parameters:
      * **pool_size** (*int* or *list/tuple of 3 ints*,) -- Size
        of the max pooling windows.

      * **strides** (*int*, *list/tuple of 3 ints*, or *None.*) --
        Factor by which to downscale. E.g. 2 will halve the input
        size. If *None*, it will default to *pool_size*.

      * **padding** (*int* or *list/tuple of 3 ints*,) -- If
        padding is non-zero, then the input is implicitly zero-padded
        on both sides for padding number of points.

      * **layout** (*str*, *default 'NCDHW'*) -- Dimension ordering
        of data and weight. Can be 'NCDHW', 'NDHWC', etc. 'N', 'C',
        'H', 'W', 'D' stands for batch, channel, height, width and
        depth dimensions respectively. padding is applied on 'D', 'H'
        and 'W' dimension.

      * **ceil_mode** (*bool*, *default False*) -- When *True*, will
        use ceil instead of floor to compute the output shape.

   Input shape:
      This depends on the *layout* parameter. Input is 5D array of
      shape (batch_size, channels, depth, height, width) if *layout*
      is *NCDHW*.

   Output shape:
      This depends on the *layout* parameter. Output is 5D array of
      shape (batch_size, channels, out_depth, out_height, out_width)
      if *layout* is *NCDHW*.

      out_depth, out_height and out_width are calculated as

      ::

         out_depth = floor((depth+2*padding[0]-pool_size[0])/strides[0])+1
         out_height = floor((height+2*padding[1]-pool_size[1])/strides[1])+1
         out_width = floor((width+2*padding[2]-pool_size[2])/strides[2])+1

      When *ceil_mode* is *True*, ceil will be used instead of floor
      in this equation.
