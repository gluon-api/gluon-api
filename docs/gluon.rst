
Gluon API
*************

Overview
========

The Gluon API offers a flexible interface that simplifies the process of prototyping, building, and training deep learning models without sacrificing training speed.

Parameter
=========

+-------------------+--------------------------------------------------------------------------------------------+
| ``Parameter``     | A Container holding parameters (weights) of Blocks.                                        |
+-------------------+--------------------------------------------------------------------------------------------+
| ``ParameterDict`` | A dictionary managing a set of parameters.                                                 |
+-------------------+--------------------------------------------------------------------------------------------+


Containers
==========

+-----------------+--------------------------------------------------------------------------------------------+
| ``Block``       | Base class for all neural network layers and models.                                       |
+-----------------+--------------------------------------------------------------------------------------------+
| ``HybridBlock`` | *HybridBlock* supports forwarding with both Symbol and NDArray.                            |
+-----------------+--------------------------------------------------------------------------------------------+
| ``SymbolBlock`` | Construct block from symbol.                                                               |
+-----------------+--------------------------------------------------------------------------------------------+


Trainer
=======

+-------------+--------------------------------------------------------------------------------------------+
| ``Trainer`` | Applies an *Optimizer* on a set of Parameters.                                             |
+-------------+--------------------------------------------------------------------------------------------+


Utilities
=========

+----------------------+--------------------------------------------------------------------------------------------+
| ``split_data``       | Splits an NDArray into *num_slice* slices along *batch_axis*.                              |
+----------------------+--------------------------------------------------------------------------------------------+
| ``split_and_load``   | Splits an NDArray into *len(ctx_list)* slices along *batch_axis* and loads each slice to   |
+----------------------+--------------------------------------------------------------------------------------------+
| ``clip_global_norm`` | Rescales NDArrays so that the sum of their 2-norm is smaller than *max_norm*.              |
+----------------------+--------------------------------------------------------------------------------------------+


API Reference
=============

Neural network module.

**class gluon.Block(prefix=None, params=None)**

   Base class for all neural network layers and models. Your models
   should subclass this class.

   ``Block`` can be nested recursively in a tree structure. You can
   create and assign child ``Block`` as regular attributes:

   .. code-block:: python

      from gluon import Block, nn
      from mxnet import ndarray as F

      class Model(Block):
          def __init__(self, **kwargs):
              super(Model, self).__init__(**kwargs)
              # use name_scope to give child Blocks appropriate names.
              # It also allows sharing Parameters between Blocks recursively.
              with self.name_scope():
                  self.dense0 = nn.Dense(20)
                  self.dense1 = nn.Dense(20)

          def forward(self, x):
              x = F.relu(self.dense0(x))
              return F.relu(self.dense1(x))

      model = Model()
      model.initialize(ctx=gluon.cpu(0))
      model(F.zeros((10, 10), ctx=gluon.cpu(0)))
  ..

   Child ``Block`` assigned this way will be registered and
   ``collect_params()`` will collect their Parameters recursively.

   :Parameters:
      * **prefix** (*str*) -- Prefix acts like a name space. It will
        be prepended to the names of all Parameters and child
        ``Block`` s in this ``Block`` 's ``name_scope()`` . Prefix
        should be unique within one model to prevent name collisions.

      * **params** (*ParameterDict* or *None*) --

        ``ParameterDict`` for sharing weights with the new ``Block``.
        For example, if you want ``dense1`` to share ``dense0``'s
        weights, you can do:

        ::

           dense0 = nn.Dense(20)
           dense1 = nn.Dense(20, params=dense0.collect_params())

   **__call__(*args)**

      Calls forward. Only accepts positional arguments.

   **__setattr__(name, value)**

      Registers parameters.

   ``__weakref__``

      list of weak references to the object (if defined)

   **collect_params()**

      Returns a ``ParameterDict`` containing this ``Block`` and all of
      its children's Parameters.

   **forward(*args)**

      Overrides to implement forward computation using ``NDArray``.
      Only accepts positional arguments.

      :Parameters:
         ***args** (*list of NDArray*) -- Input tensors.

   **hybridize(active=True)**

      Activates or deactivates ``HybridBlock`` s recursively. Has no
      effect on non-hybrid children.

      :Parameters:
         **active** (*bool*, *default True*) -- Whether to turn
         hybrid on or off.

   **initialize(init=<gluon.initializer.Uniform object>, ctx=None,
   verbose=False)**

      Initializes ``Parameter`` s of this ``Block`` and its children.

      Equivalent to ``block.collect_params().initialize(...)``

   **load_params(filename, ctx, allow_missing=False,
   ignore_extra=False)**

      Load parameters from file.

      filename : str
         Path to parameter file.

      ctx : Context or list of Context
         Context(s) initialize loaded parameters on.

      allow_missing : bool, default False
         Whether to silently skip loading parameters not represents in
         the file.

      ignore_extra : bool, default False
         Whether to silently ignore parameters from the file that are
         not present in this Block.

   ``name``

      Name of this ``Block``, without '_' in the end.

   **name_scope()**

      Returns a name space object managing a child ``Block`` and
      parameter names. Should be used within a ``with`` statement:

      .. code-block:: python

         with self.name_scope():
             self.dense = nn.Dense(20)
      ..

   ``params``

      Returns this ``Block``'s parameter dictionary (does not include
      its children's parameters).

   ``prefix``

      Prefix of this ``Block``.

   **register_child(block)**

      Registers block as a child of self. ``Block`` s assigned to self
      as attributes will be registered automatically.

   **save_params(filename)**

      Save parameters to file.

      filename : str
         Path to file.

**exception gluon.DeferredInitializationError**

   Error for unfinished deferred initialization.

**class gluon.HybridBlock(prefix=None, params=None)**

   *HybridBlock* supports forwarding with both Symbol and NDArray.

   Forward computation in ``HybridBlock`` must be static to work with
   ``Symbol`` s, i.e. you cannot call ``NDArray.asnumpy()``,
   ``NDArray.shape``, ``NDArray.dtype``, etc on tensors. Also, you
   cannot use branching or loop logic that bases on non-constant
   expressions like random numbers or intermediate results, since they
   change the graph structure for each iteration.

   Before activating with ``hybridize()``, ``HybridBlock`` works just
   like normal ``Block``. After activation, ``HybridBlock`` will
   create a symbolic graph representing the forward computation and
   cache it. On subsequent forwards, the cached graph will be used
   instead of ``hybrid_forward()``.

   Refer to the `Hybridize tutorial <https://github.com/zackchase/mxnet-the-straight-dope/blob/master/chapter07_distributed-learning/hybridize.ipynb>`_ to see the end-to-end usage.

   **__setattr__(name, value)**

      Registers parameters.

   **forward(x, *args)**

      Defines the forward computation. Arguments can be either
      ``NDArray`` or ``Symbol``.

   **hybrid_forward(F, x, *args, **kwargs)**

      Overrides to construct symbolic graph for this *Block*.

      :Parameters:
         * **x** ( `Symbol
           <https://mxnet.incubator.apache.org/versions/master/api/python/symbol/symbol.html#mxnet.symbol.Symbol>`_  or  `NDArray
           <https://mxnet.incubator.apache.org/versions/master/api/python/ndarray/ndarray.html#mxnet.ndarray.NDArray>`_ )-- The
           first input tensor.

         * ***args** (*list of Symbol* or *list of NDArray*) --
           Additional input tensors.

   **infer_shape(*args)**

      Infers shape of Parameters from inputs.

**class gluon.Parameter(name, grad_req='write', shape=None,
dtype=<type 'numpy.float32'>, lr_mult=1.0, wd_mult=1.0, init=None,
allow_deferred_init=False, differentiable=True)**

   A Container holding parameters (weights) of Blocks.

   ``Parameter`` holds a copy of the parameter on each ``Context``
   after it is initialized with ``Parameter.initialize(...)``. If
   ``grad_req`` is not ``'null'``, it will also hold a gradient array
   on each ``Context``:

   .. code-block:: python

      ctx = gluon.gpu(0)
      x = gluon.nd.zeros((16, 100), ctx=ctx)
      w = gluon.Parameter('fc_weight', shape=(64, 100), init=gluon.init.Xavier())
      b = gluon.Parameter('fc_bias', shape=(64,), init=gluon.init.Zero())
      w.initialize(ctx=ctx)
      b.initialize(ctx=ctx)
      out = gluon.nd.FullyConnected(x, w.data(ctx), b.data(ctx), num_hidden=64)
   ..

   :Parameters:
      * **name** (*str*) -- Name of this parameter.

      * **grad_req** (*{'write'*, *'add'*, *'null'}*, *default
        'write'*) --

        Specifies how to update gradient to grad arrays.

        * ``'write'`` means everytime gradient is written to grad
          ``NDArray``.

        * ``'add'`` means everytime gradient is added to the grad
          ``NDArray``. You need to manually call ``zero_grad()`` to
          clear the gradient buffer before each iteration when using
          this option.

        * 'null' means gradient is not requested for this parameter.
          gradient arrays will not be allocated.

      * **shape** (*tuple of int*, *default None*) -- Shape of this
        parameter. By default shape is not specified. Parameter with
        unknown shape can be used for ``Symbol`` API, but ``init``
        will throw an error when using ``NDArray`` API.

      * **dtype** (*numpy.dtype* or *str*, *default 'float32'*) --
        Data type of this parameter. For example, ``numpy.float32`` or
        ``'float32'``.

      * **lr_mult** (*float*, *default 1.0*) -- Learning rate
        multiplier. Learning rate will be multiplied by lr_mult when
        updating this parameter with optimizer.

      * **wd_mult** (*float*, *default 1.0*) -- Weight decay
        multiplier (L2 regularizer coefficient). Works similar to
        lr_mult.

      * **init** (`Initializer
        <https://mxnet.incubator.apache.org/versions/master/api/python/optimization/optimization.html#mxnet.initializer.Initializer>`_,
        *default None*) -- Initializer of this parameter. Will use
        the global initializer by default.

   ``grad_req``

      *{'write', 'add', 'null'}* -- This can be set before or after
      initialization. Setting ``grad_req`` to ``'null'`` with
      ``x.grad_req = 'null'`` saves memory and computation when you
      don't need gradient w.r.t x.

   ``lr_mult``

      *float* -- Local learning rate multiplier for this Parameter.
      The actual learning rate is calculated with ``learning_rate *
      lr_mult``. You can set it with ``param.lr_mult = 2.0``

   ``wd_mult``

      *float* -- Local weight decay multiplier for this Parameter.

   ``__weakref__``

      list of weak references to the object (if defined)

   **data(ctx=None)**

      Returns a copy of this parameter on one context. Must have been
      initialized on this context before.

      :Parameters:
         **ctx** (*Context*) -- Desired context.

      :Returns:
      :Return type:
         NDArray on ctx

   **grad(ctx=None)**

      Returns a gradient buffer for this parameter on one context.

      :Parameters:
         **ctx** (*Context*) -- Desired context.

   **initialize(init=None, ctx=None,
   default_init=<gluon.initializer.Uniform object>,
   force_reinit=False)**

      Initializes parameter and gradient arrays. Only used for
      ``NDArray`` API.

      :Parameters:
         * **init** (`Initializer
           <https://mxnet.incubator.apache.org/versions/master/api/python/optimization/optimization.html#mxnet.initializer.Initializer>`_)
           -- The initializer to use. Overrides ``Parameter.init()``
           and default_init.

         * **ctx** (Context or list of Context, defaults to
           ``context.current_context()``.) --

           Initialize Parameter on given context. If ctx is a list of
           Context, a copy will be made for each context.

           Note: Copies are independent arrays. User is responsible for keeping their values consistent when updating. 
           Normally ``gluon.Trainer`` does this for you.

         * **default_init** (`Initializer
           <https://mxnet.incubator.apache.org/versions/master/api/python/optimization/optimization.html#mxnet.initializer.Initializer>`_)
           -- Default initializer is used when both ``init()`` and
           ``Parameter.init()`` are ``None``.

         * **force_reinit** (*bool*, *default False*) -- Whether to
           force re-initialization if parameter is already
           initialized.

      Example:

      .. code-block:: python

      >>> weight = gluon.Parameter('weight', shape=(2, 2))
      >>> weight.initialize(ctx=gluon.cpu(0))
      >>> weight.data()
      [[-0.01068833  0.01729892]
       [ 0.02042518 -0.01618656]]
      <NDArray 2x2 @cpu(0)>
      >>> weight.grad()
      [[ 0.  0.]
       [ 0.  0.]]
      <NDArray 2x2 @cpu(0)>
      >>> weight.initialize(ctx=[gluon.gpu(0), gluon.gpu(1)])
      >>> weight.data(gluon.gpu(0))
      [[-0.00873779 -0.02834515]
       [ 0.05484822 -0.06206018]]
      <NDArray 2x2 @gpu(0)>
      >>> weight.data(gluon.gpu(1))
      [[-0.00873779 -0.02834515]
       [ 0.05484822 -0.06206018]]
      <NDArray 2x2 @gpu(1)>


   **list_ctx()**

      Returns a list of contexts this parameter is initialized on.

   **list_data()**

      Returns copies of this parameter on all contexts, in the same
      order as creation.

   **list_grad()**

      Returns gradient buffers on all contexts, in the same order as
      ``values()``.

   **reset_ctx(ctx)**

      Re-assign Parameter to other contexts.

      ctx : Context or list of Context, default
      ``context.current_context()``.
         Assign Parameter to given context. If ctx is a list of
         Context, a copy will be made for each context.

   **set_data(data)**

      Sets this parameter's value on all contexts to data.

   **var()**

      Returns a symbol representing this parameter.

   **zero_grad()**

      Sets gradient buffer on all contexts to 0. No action is taken if
      parameter is uninitialized or doesn't require gradient.

**class gluon.ParameterDict(prefix='', shared=None)**

   A dictionary managing a set of parameters.

   :Parameters:
      * **prefix** (str, default ``''``) -- The prefix to be prepended
        to all Parameters' names created by this dict.

      * **shared** (*ParameterDict* or *None*) -- If not ``None``,
        when this dict's ``get()`` method creates a new parameter,
        will first try to retrieve it from "shared" dict. Usually used
        for sharing parameters with another Block.

   ``__weakref__``

      list of weak references to the object (if defined)

   **get(name, **kwargs)**

      Retrieves a ``Parameter`` with name ``self.prefix+name``. If not
      found, ``get()`` will first try to retrieve it from "shared"
      dict. If still not found, ``get()`` will create a new
      ``Parameter`` with key-word arguments and insert it to self.

      :Parameters:
         * **name** (*str*) -- Name of the desired Parameter. It will
           be prepended with this dictionary's prefix.

         * ****kwargs** (*dict*) -- The rest of key-word arguments for
           the created ``Parameter``.

      :Returns:
         The created or retrieved ``Parameter``.

      :Return type:
         Parameter

   **initialize(init=<gluon.initializer.Uniform object>, ctx=None,
   verbose=False, force_reinit=False)**

      Initializes all Parameters managed by this dictionary to be used
      for ``NDArray`` API. It has no effect when using ``Symbol`` API.

      :Parameters:
         * **init** (`Initializer
           <https://mxnet.incubator.apache.org/versions/master/api/python/optimization/optimization.html#mxnet.initializer.Initializer>`_)
           -- Global default Initializer to be used when
           ``Parameter.init()`` is ``None``. Otherwise,
           ``Parameter.init()`` takes precedence.

         * **ctx** (*Context* or *list of Context*) -- Keeps a copy
           of Parameters on one or many context(s).

         * **force_reinit** (*bool*, *default False*) -- Whether to
           force re-initialization if parameter is already
           initialized.

   **load(filename, ctx, allow_missing=False, ignore_extra=False,
   restore_prefix='')**

      Load parameters from file.

      filename : str
         Path to parameter file.

      ctx : Context or list of Context
         Context(s) initialize loaded parameters on.

      allow_missing : bool, default False
         Whether to silently skip loading parameters not represents in
         the file.

      ignore_extra : bool, default False
         Whether to silently ignore parameters from the file that are
         not present in this ParameterDict.

      restore_prefix : str, default ''
         prepend prefix to names of stored parameters before loading.

   ``prefix``

      Prefix of this dict. It will be prepended to ``Parameter`s' name
      created with :py:func:`get``.

   **reset_ctx(ctx)**

      Re-assign all Parameters to other contexts.

      ctx : Context or list of Context, default
      ``context.current_context()``.
         Assign Parameter to given context. If ctx is a list of
         Context, a copy will be made for each context.

   **save(filename, strip_prefix='')**

      Save parameters to file.

      filename : str
         Path to parameter file.

      strip_prefix : str, default ''
         Strip prefix from parameter names before saving.

   **setattr(name, value)**

      Set an attribute to a new value for all Parameters.

      For example, set grad_req to null if you don't need gradient
      w.r.t a model's Parameters:

      .. code-block:: python

         model.collect_params().setattr('grad_req', 'null')

      or change the learning rate multiplier:

      .. code-block:: python

         model.collect_params().setattr('lr_mult', 0.5)

      :Parameters:
         * **name** (*str*) -- Name of the attribute.

         * **value** (*valid type for attribute name*) -- The new
           value for the attribute.

   **update(other)**

      Copies all Parameters in ``other`` to self.

   **zero_grad()**

      Sets all Parameters' gradient buffer to 0.

**class gluon.SymbolBlock(outputs, inputs, params=None)**

   Construct block from symbol. This is useful for using pre-trained
   models as feature extractors. For example, you may want to extract
   the output from fc2 layer in AlexNet.

   :Parameters:
      * **outputs** (`Symbol
        <https://mxnet.incubator.apache.org/versions/master/api/python/symbol/symbol.html#mxnet.symbol.Symbol>`_ or *list of
        Symbol* )-- The desired output for SymbolBlock.

      * **inputs** (`Symbol
        <https://mxnet.incubator.apache.org/versions/master/api/python/symbol/symbol.html#mxnet.symbol.Symbol>`_ or *list of
        Symbol*) -- The Variables in output's argument that should be
        used as inputs.

      * **params** (*ParameterDict*) -- Parameter dictionary for
        arguments and auxililary states of outputs that are not
        inputs.

   Example:

   .. code-block:: python

   >>> # To extract the feature from fc1 and fc2 layers of AlexNet:
   >>> alexnet = gluon.model_zoo.vision.alexnet(pretrained=True, ctx=gluon.cpu(),
                                                prefix='model_')
   >>> inputs = gluon.sym.var('data')
   >>> out = alexnet(inputs)
   >>> internals = out.get_internals()
   >>> print(internals.list_outputs())
   ['data', ..., 'model_dense0_relu_fwd_output', ..., 'model_dense1_relu_fwd_output', ...]
   >>> outputs = [internals['model_dense0_relu_fwd_output'],
                  internals['model_dense1_relu_fwd_output']]
   >>> # Create SymbolBlock that shares parameters with alexnet
   >>> feat_model = gluon.SymbolBlock(outputs, inputs, params=alexnet.collect_params())
   >>> x = gluon.nd.random.normal(shape=(16, 3, 224, 224))
   >>> print(feat_model(x))
   ..

**class gluon.Trainer(params, optimizer, optimizer_params=None,
kvstore='device')**

   Applies an *Optimizer* on a set of Parameters. Trainer should be
   used together with *autograd*.

   :Parameters:
      * **params** (*ParameterDict*) -- The set of parameters to
        optimize.

      * **optimizer** (*str* or `Optimizer
        <https://mxnet.incubator.apache.org/versions/master/api/python/optimization/optimization.html#mxnet.optimizer.Optimizer>`_)
        -- The optimizer to use. See help on Optimizer for a list of
        available optimizers.

      * **optimizer_params** (*dict*) -- Key-word arguments to be
        passed to optimizer constructor. For example,
        *{'learning_rate': 0.1}*. All optimizers accept learning_rate,
        wd (weight decay), clip_gradient, and lr_scheduler. See each
        optimizer's constructor for a list of additional supported
        arguments.

      * **kvstore** (*str* or `KVStore
        <https://mxnet.incubator.apache.org/versions/master/api/python/kvstore/kvstore.html#mxnet.kvstore.KVStore>`_) -- kvstore
        type for multi-gpu and distributed training. See help on
        `gluon.kvstore.create
        <https://mxnet.incubator.apache.org/versions/master/api/python/kvstore/kvstore.html#mxnet.kvstore.create>`_ for more
        information.

      * **Properties** --

      * **----------** --

      * **learning_rate** (*float*) -- The current learning rate of
        the optimizer. Given an Optimizer object optimizer, its
        learning rate can be accessed as optimizer.learning_rate.

   ``__weakref__``

      list of weak references to the object (if defined)

   **load_states(fname)**

      Loads Trainer states (e.g. optimizer, momentum) from a file.

      :Parameters:
         **fname** (*str*) -- Path to input states file.

   **save_states(fname)**

      Saves Trainer states (e.g. optimizer, momentum) to a file.

      :Parameters:
         **fname** (*str*) -- Path to output states file.

   **set_learning_rate(lr)**

      Sets a new learning rate of the optimizer.

      :Parameters:
         **lr** (*float*) -- The new learning rate of the optimizer.

   **step(batch_size, ignore_stale_grad=False)**

      Makes one step of parameter update. Should be called after
      *autograd.compute_gradient* and outside of *record()* scope.

      :Parameters:
         * **batch_size** (*int*) -- Batch size of data processed.
           Gradient will be normalized by *1/batch_size*. Set this to
           1 if you normalized loss manually with *loss = mean(loss)*.

         * **ignore_stale_grad** (*bool*, *optional*,
           *default=False*) -- If true, ignores Parameters with stale
           gradient (gradient that has not been updated by *backward*
           after last step) and skip update.

**class gluon.nn.Sequential(prefix=None, params=None)**

   Stacks Blocks sequentially.

   Example:

   .. code-block:: python

      net = nn.Sequential()
      # use net's name_scope to give child Blocks appropriate names.
      with net.name_scope():
          net.add(nn.Dense(10, activation='relu'))
          net.add(nn.Dense(20))
   ..

   **add(*blocks)**

      Adds block on top of the stack.

   **hybridize(active=True)**

      Activates or deactivates >>`<<HybridBlock`s recursively. Has no
      effect on non-hybrid children.

      :Parameters:
         **active** (*bool*, *default True*) -- Whether to turn
         hybrid on or off.

**class gluon.nn.HybridSequential(prefix=None, params=None)**

   Stacks HybridBlocks sequentially.

   Example:

   .. code-block:: python

      net = nn.Sequential()
      # use net's name_scope to give child Blocks appropriate names.
      with net.name_scope():
          net.add(nn.Dense(10, activation='relu'))
          net.add(nn.Dense(20))
   ..

   **add(*blocks)**

      Adds block on top of the stack.

Parallelization utility optimizer.

**gluon.utils.split_data(data, num_slice, batch_axis=0,
even_split=True)**

   Splits an NDArray into *num_slice* slices along *batch_axis*.
   Usually used for data parallelism where each slices is sent to one
   device (i.e. GPU).

   :Parameters:
      * **data** (`NDArray
        <https://mxnet.incubator.apache.org/versions/master/api/python/ndarray/ndarray.html#mxnet.ndarray.NDArray>`_) -- A batch
        of data.

      * **num_slice** (*int*) -- Number of desired slices.

      * **batch_axis** (*int*, *default 0*) -- The axis along which
        to slice.

      * **even_split** (*bool*, *default True*) -- Whether to force
        all slices to have the same number of elements. If *True*, an
        error will be raised when *num_slice* does not evenly divide
        *data.shape[batch_axis]*.

   :Returns:
      Return value is a list even if *num_slice* is 1.

   :Return type:
      list of NDArray

**gluon.utils.split_and_load(data, ctx_list, batch_axis=0,
even_split=True)**

   Splits an NDArray into *len(ctx_list)* slices along *batch_axis*
   and loads each slice to one context in *ctx_list*.

   :Parameters:
      * **data** (`NDArray
        <https://mxnet.incubator.apache.org/versions/master/api/python/ndarray/ndarray.html#mxnet.ndarray.NDArray>`_) -- A batch
        of data.

      * **ctx_list** (*list of Context*) -- A list of Contexts.

      * **batch_axis** (*int*, *default 0*) -- The axis along which
        to slice.

      * **even_split** (*bool*, *default True*) -- Whether to force
        all slices to have the same number of elements.

   :Returns:
      Each corresponds to a context in *ctx_list*.

   :Return type:
      list of NDArray

**gluon.utils.clip_global_norm(arrays, max_norm)**

   Rescales NDArrays so that the sum of their 2-norm is smaller than
   *max_norm*.

**gluon.utils.check_sha1(filename, sha1_hash)**

   Check whether the sha1 hash of the file content matches the
   expected hash.

   :Parameters:
      * **filename** (*str*) -- Path to the file.

      * **sha1_hash** (*str*) -- Expected sha1 hash in hexadecimal
        digits.

   :Returns:
      Whether the file content matches the expected hash.

   :Return type:
      bool

**gluon.utils.download(url, path=None, overwrite=False,
sha1_hash=None)**

   Download an given URL

   :Parameters:
      * **url** (*str*) -- URL to download

      * **path** (*str*, *optional*) -- Destination path to store
        downloaded file. By default stores to the current directory
        with same name as in url.

      * **overwrite** (*bool*, *optional*) -- Whether to overwrite
        destination file if already exists.

      * **sha1_hash** (*str*, *optional*) -- Expected sha1 hash in
        hexadecimal digits. Will ignore existing file when hash is
        specified but doesn't match.

   :Returns:
      The file path of the downloaded file.

   :Return type:
      str
