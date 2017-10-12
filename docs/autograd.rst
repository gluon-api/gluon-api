
Autograd API
****************

Overview
========

The ``autograd`` package enables automatic differentiation of NDArray
operations. In machine learning applications, ``autograd`` is often
used to calculate the gradients of loss functions with respect to
parameters.


Record vs Pause
---------------

``autograd`` records computation history on the fly to calculate
gradients later. This is only enabled inside a ``with
autograd.record():`` block. A ``with auto_grad.pause()`` block can be
used inside a ``record()`` block to temporarily disable recording.

To compute gradient with respect to an ``NDArray`` ``x``, first call
``x.attach_grad()`` to allocate space for the gradient. Then, start a
``with autograd.record()`` block, and do some computation. Finally,
call ``backward()`` on the result:

.. code-block:: python

   >>> x = gluon.nd.array([1,2,3,4])
   >>> x.attach_grad()
   >>> with gluon.autograd.record():
   ...     y = x * x + 1
   >>> y.backward()
   >>> print(x.grad)
   [ 2.  4.  6.  8.]
   <NDArray 4 @cpu(0)>


Train mode and Predict Mode
===========================

Some operators (Dropout, BatchNorm, etc) behave differently in when
training and when making predictions. This can be controlled with
``train_mode`` and ``predict_mode`` scope.

By default, gluon is in ``predict_mode``. A ``with autograd.record()``
block by default turns on ``train_mode`` (equivalent to ``with
autograd.record(train_mode=True)``). To compute a gradient in
prediction mode (as when generating adversarial examples), call record
with ``train_mode=False`` and then call ``backward(train_mode=False)``

Although training usually coincides with recording, this isn't always
the case. To control *training* vs *predict_mode* without changing
*recording* vs *not recording*, Use a ``with autograd.train_mode():``
or ``with autograd.predict_mode():`` block.

Detailed tutorials are available in Part 1 of the gluon book.


Autograd
========

+--------------------+--------------------------------------------------------------------------------------------+
| ``record``         | Returns an autograd recording scope context to be used in 'with' statement and captures    |
+--------------------+--------------------------------------------------------------------------------------------+
| ``pause``          | Returns a scope context to be used in 'with' statement for codes that do not need          |
+--------------------+--------------------------------------------------------------------------------------------+
| ``train_mode``     | Returns a scope context to be used in 'with' statement in which forward pass behavior is   |
+--------------------+--------------------------------------------------------------------------------------------+
| ``predict_mode``   | Returns a scope context to be used in 'with' statement in which forward pass behavior is   |
+--------------------+--------------------------------------------------------------------------------------------+
| ``backward``       | Compute the gradients of heads w.r.t previously marked variables.                          |
+--------------------+--------------------------------------------------------------------------------------------+
| ``set_training``   | Set status to training/predicting.                                                         |
+--------------------+--------------------------------------------------------------------------------------------+
| ``is_training``    | Get status on training/predicting.                                                         |
+--------------------+--------------------------------------------------------------------------------------------+
| ``set_recording``  | Set status to recording/not recording.                                                     |
+--------------------+--------------------------------------------------------------------------------------------+
| ``is_recording``   | Get status on recording/not recording.                                                     |
+--------------------+--------------------------------------------------------------------------------------------+
| ``mark_variables`` | Mark NDArrays as variables to compute gradient for autograd.                               |
+--------------------+--------------------------------------------------------------------------------------------+
| ``Function``       | User-defined differentiable function.                                                      |
+--------------------+--------------------------------------------------------------------------------------------+


API Reference
=============

Autograd for NDArray.

**gluon.autograd.set_recording(is_recording)**

   Set status to recording/not recording. When recording, graph will
   be constructed for gradient computation.

   :Parameters:
      **is_recording** (*bool*)

   :Returns:
   :Return type:
      previous state before this set.

**gluon.autograd.set_training(train_mode)**

   Set status to training/predicting. This affects ctx.is_train in
   operator running context. For example, Dropout will drop inputs
   randomly when train_mode=True while simply passing through if
   train_mode=False.

   :Parameters:
      **train_mode** (*bool*)

   :Returns:
   :Return type:
      previous state before this set.

**gluon.autograd.is_recording()**

   Get status on recording/not recording.

   :Returns:
   :Return type:
      Current state of recording.

**gluon.autograd.is_training()**

   Get status on training/predicting.

   :Returns:
   :Return type:
      Current state of training/predicting.

**gluon.autograd.record(train_mode=True)**

   Returns an autograd recording scope context to be used in 'with'
   statement and captures code that needs gradients to be calculated.

   Note: When forwarding with train_mode=False, the corresponding backward
     should also use train_mode=False, otherwise gradient is
     undefined.

   Example:

.. code-block:: python

      with autograd.record():
          output = net(data)
          L = loss(output, label)
          grads = L.backward(L.parameters)
          updater.step(grads)

..

  :Parameters:
      **train_mode** (*bool*, *default True*) -- Whether the forward
      pass is in training or predicting mode. This controls the
      behavior of some layers such as Dropout, BatchNorm.

**gluon.autograd.pause(train_mode=False)**

   Returns a scope context to be used in 'with' statement for codes
   that do not need gradients to be calculated.

   Example:

.. code-block:: python

      with autograd.record():
          y = model(x)
          backward([y])
          with autograd.pause():
              # testing, IO, gradient updates...
..

   :Parameters:
      **train_mode** (*bool*, *default False*) -- Whether to do
      forward for training or predicting.

**gluon.autograd.train_mode()**

   Returns a scope context to be used in 'with' statement in which
   forward pass behavior is set to training mode, without changing the
   recording states.

   Example:

.. code-block:: python

      y = model(x)
      with autograd.train_mode():
          y = dropout(y)

**gluon.autograd.predict_mode()**

   Returns a scope context to be used in 'with' statement in which
   forward pass behavior is set to inference mode, without changing
   the recording states.

   Example:

.. code-block:: python

      with autograd.record():
          y = model(x)
          with autograd.predict_mode():
              y = sampling(y)
          backward([y])

**gluon.autograd.mark_variables(variables, gradients,
grad_reqs='write')**

   Mark NDArrays as variables to compute gradient for autograd.

   :Parameters:
      * **variables** (*NDArray* or *list of
        NDArray*)

      * **gradients** (*NDArray* or *list of
        NDArray*)

      * **grad_reqs** (*str* or *list of str*)

**gluon.autograd.backward(heads, head_grads=None, retain_graph=False,
train_mode=True)**

   Compute the gradients of heads w.r.t previously marked variables.

   :Parameters:
      * **heads** (*NDArray* or *list of
        NDArray*) -- Output NDArray(s)

      * **head_grads** (*NDArray* or *list of
        NDArray* or *None*) -- Gradients with respect to heads.

      * **train_mode** (*bool*, *optional*) -- Whether to do
        backward for training or predicting.

**gluon.autograd.grad(heads, variables, head_grads=None,
retain_graph=None, create_graph=False, train_mode=True)**

   Compute the gradients of heads w.r.t variables. Gradients will be
   returned as new NDArrays instead of stored into *variable.grad*.
   Supports recording gradient graph for computing higher order
   gradients.

   gradients.

   :Parameters:
      * **heads** *NDArray* or *list of
        NDArray* -- Output NDArray(s)

      * **variables** `NDArray
        <https://mxnet.incubator.apache.org/versions/master/api/python/ndarray/ndarray.html#mxnet.ndarray.NDArray>`_ or *list of
        NDArray* -- Input variables to compute gradients for.

      * **head_grads** *NDArray* or *list of
        NDArray* or *None* -- Gradients with respect to heads.

      * **retain_graph** (*bool*) -- Whether to keep computation graph
        to differentiate again, instead of clearing history and
        release memory. Defaults to the same value as create_graph.

      * **create_graph** (*bool*) -- Whether to record gradient graph
        for computing higher order

      * **train_mode** (*bool*, *optional*) -- Whether to do
        backward for training or prediction.

   :Returns:
      Gradients with respect to variables.

   :Return type:
      NDArray or
      list of NDArray

   Example:

   >>> x = gluon.nd.ones((1,))
   >>> x.attach_grad()
   >>> with gluon.autograd.record():
   ...     z = gluon.nd.elemwise_add(gluon.nd.exp(x), x)
   >>> dx = gluon.autograd.grad(z, [x], create_graph=True)
   >>> dx.backward()
   >>> print(dx.grad)
   [ 3.71828175]
   <NDArray 1 @cpu(0)>]

**gluon.autograd.get_symbol(x)**

   Retrieve recorded computation history as *Symbol*.

   :Parameters:
      **x** NDArray -- Array
      representing the head of computation graph.

   :Returns:
      The retrieved Symbol.

   :Return type:
      `Symbol <https://mxnet.incubator.apache.org/versions/master/api/python/symbol/symbol.html#mxnet.symbol.Symbol>`_

**class gluon.autograd.Function**

   User-defined differentiable function.

   Function allows defining both forward and backward computation for
   custom operators. During gradient computation, the used-defined
   backward function will be used instead of the default chain-rule.
   You can also cast to numpy array and back for some operations in
   forward and backward.

   For example, a stable sigmoid function can be defined as:

.. code-block:: python

      class sigmoid(Function):
          def forward(self, x):
              y = 1 / (1 + gluon.nd.exp(-x))
              self.save_for_backward(y)
              return y

          def backward(self, dy):
              # backward takes as many inputs as forward's return value,
              # and returns as many NDArrays as forward's arguments.
              y, = self.saved_tensors
              return y * (1-y)
..

   **forward(*inputs)**

      Forward computation.

   **backward(*output_grads)**

      Backward computation.

      Takes as many inputs as forward's outputs, and returns as many
      NDArrays as forward's inputs.
