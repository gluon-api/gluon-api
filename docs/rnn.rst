
Gluon Recurrent Neural Network API
**********************************


Overview
========

This document lists the recurrent neural network API in Gluon:


Recurrent Layers
----------------

Recurrent layers can be used in ``Sequential`` with other regular
neural network layers. For example, to construct a sequence labeling
model where a prediction is made for each time-step:

.. code-block:: python

   model =  gluon.nn.Sequential()
   model.add(gluon.nn.Embedding(30, 10))
   model.add(gluon.rnn.LSTM(20))
   model.add(gluon.nn.Dense(5, flatten=False))
   model.initialize()
   model(nd.ones((2,3,5)))

+------------+--------------------------------------------------------------------------------------------+
| ``RNN``    | Applies a multi-layer Elman RNN with *tanh* or *ReLU* non-linearity to an input sequence.  |
+------------+--------------------------------------------------------------------------------------------+
| ``LSTM``   | Applies a multi-layer long short-term memory (LSTM) RNN to an input sequence.              |
+------------+--------------------------------------------------------------------------------------------+
| ``GRU``    | Applies a multi-layer gated recurrent unit (GRU) RNN to an input sequence.                 |
+------------+--------------------------------------------------------------------------------------------+


Recurrent Cells
---------------

Recurrent cells exposes the intermediate recurrent states and allows
for explicit stepping and unrolling, and thus provides more
flexibility.

+-----------------------+--------------------------------------------------------------------------------------------+
| ``RNNCell``           | Elman RNN recurrent neural network cell.                                                   |
+-----------------------+--------------------------------------------------------------------------------------------+
| ``LSTMCell``          | Long-Short Term Memory (LSTM) network cell.                                                |
+-----------------------+--------------------------------------------------------------------------------------------+
| ``GRUCell``           | Gated Rectified Unit (GRU) network cell.                                                   |
+-----------------------+--------------------------------------------------------------------------------------------+
| ``RecurrentCell``     | Abstract base class for RNN cells                                                          |
+-----------------------+--------------------------------------------------------------------------------------------+
| ``SequentialRNNCell`` | Sequentially stacking multiple RNN cells.                                                  |
+-----------------------+--------------------------------------------------------------------------------------------+
| ``BidirectionalCell`` | Bidirectional RNN cell.                                                                    |
+-----------------------+--------------------------------------------------------------------------------------------+
| ``DropoutCell``       | Applies dropout on input.                                                                  |
+-----------------------+--------------------------------------------------------------------------------------------+
| ``ZoneoutCell``       | Applies Zoneout on base cell.                                                              |
+-----------------------+--------------------------------------------------------------------------------------------+
| ``ResidualCell``      | Adds residual connection as described in Wu et al, 2016                                    |
+-----------------------+--------------------------------------------------------------------------------------------+


API Reference
=============

Recurrent neural network module.

**class gluon.rnn.BidirectionalCell(l_cell, r_cell,
output_prefix='bi_')**

   Bidirectional RNN cell.

   :Parameters:
      * **l_cell** (*RecurrentCell*) -- Cell for forward unrolling

      * **r_cell** (*RecurrentCell*) -- Cell for backward unrolling

**class gluon.rnn.DropoutCell(rate, prefix=None, params=None)**

   Applies dropout on input.

   :Parameters:
      **rate** (*float*) -- Percentage of elements to drop out, which
      is 1 - percentage to retain.

**class gluon.rnn.GRU(hidden_size, num_layers=1, layout='TNC',
dropout=0, bidirectional=False, input_size=0,
i2h_weight_initializer=None, h2h_weight_initializer=None,
i2h_bias_initializer='zeros', h2h_bias_initializer='zeros',
**kwargs)**

   Applies a multi-layer gated recurrent unit (GRU) RNN to an input
   sequence.

   For each element in the input sequence, each layer computes the
   following function:

      \begin{array}{ll} r_t = sigmoid(W_{ir} x_t + b_{ir} + W_{hr}
      h_{(t-1)} + b_{hr}) \\ i_t = sigmoid(W_{ii} x_t + b_{ii} + W_hi
      h_{(t-1)} + b_{hi}) \\ n_t = \tanh(W_{in} x_t + b_{in} + r_t *
      (W_{hn} h_{(t-1)}+ b_{hn})) \\ h_t = (1 - i_t) * n_t + i_t *
      h_{(t-1)} \\ \end{array}

   where h_t is the hidden state at time *t*, x_t is the hidden state
   of the previous layer at time *t* or input_t for the first layer,
   and r_t, i_t, n_t are the reset, input, and new gates,
   respectively.

   :Parameters:
      * **hidden_size** (*int*) -- The number of features in the
        hidden state h

      * **num_layers** (*int*, *default 1*) -- Number of recurrent
        layers.

      * **layout** (*str*, *default 'TNC'*) -- The format of input
        and output tensors. T, N and C stand for sequence length,
        batch size, and feature dimensions respectively.

      * **dropout** (*float*, *default 0*) -- If non-zero,
        introduces a dropout layer on the outputs of each RNN layer
        except the last layer

      * **bidirectional** (*bool*, *default False*) -- If True,
        becomes a bidirectional RNN.

      * **i2h_weight_initializer** (*str* or `Initializer
        <https://mxnet.incubator.apache.org/versions/master/api/python/optimization/optimization.html#mxnet.initializer.Initializer>`_)
        -- Initializer for the input weights matrix, used for the
        linear transformation of the inputs.

      * **h2h_weight_initializer** (*str* or `Initializer
        <https://mxnet.incubator.apache.org/versions/master/api/python/optimization/optimization.html#mxnet.initializer.Initializer>`_)
        -- Initializer for the recurrent weights matrix, used for the
        linear transformation of the recurrent state.

      * **i2h_bias_initializer** (*str* or `Initializer
        <https://mxnet.incubator.apache.org/versions/master/api/python/optimization/optimization.html#mxnet.initializer.Initializer>`_)
        -- Initializer for the bias vector.

      * **h2h_bias_initializer** (*str* or `Initializer
        <https://mxnet.incubator.apache.org/versions/master/api/python/optimization/optimization.html#mxnet.initializer.Initializer>`_)
        -- Initializer for the bias vector.

      * **input_size** (*int*, *default 0*) -- The number of
        expected features in the input x. If not specified, it will be
        inferred from input.

      * **prefix** (*str* or *None*) -- Prefix of this *Block*.

      * **params** (*ParameterDict* or *None*) -- Shared Parameters
        for this *Block*.

   Input shapes:
      The input shape depends on *layout*. For *layout='TNC'*, the
      input has shape *(sequence_length, batch_size, input_size)*

   Output shape:
      The output shape depends on *layout*. For *layout='TNC'*, the
      output has shape *(sequence_length, batch_size, num_hidden)*. If
      *bidirectional* is True, output shape will instead be
      *(sequence_length, batch_size, 2*num_hidden)*

   Recurrent state:
      The recurrent state is an NDArray with shape *(num_layers,
      batch_size, num_hidden)*. If *bidirectional* is True, the
      recurrent state shape will instead be *(2*num_layers,
      batch_size, num_hidden)* If input recurrent state is None, zeros
      are used as default begin states, and the output recurrent state
      is omitted.

   Example:

.. code-block:: python

   layer =  gluon.rnn.GRU(100, 3)
   layer.initialize()
   input = nd.random.uniform(shape=(5, 3, 10))
   # by default zeros are used as begin state
   output = layer(input)
   # manually specify begin state.
   h0 = nd.random.uniform(shape=(3, 3, 100))
   output, hn = layer(input, h0)

**class gluon.rnn.GRUCell(hidden_size,
i2h_weight_initializer=None, h2h_weight_initializer=None,
i2h_bias_initializer='zeros', h2h_bias_initializer='zeros',
input_size=0, prefix=None, params=None)**

   Gated Rectified Unit (GRU) network cell. Note: this is an
   implementation of the cuDNN version of GRUs (slight modification
   compared to Cho et al. 2014).

   Each call computes the following function:

      \begin{array}{ll} r_t = sigmoid(W_{ir} x_t + b_{ir} + W_{hr}
      h_{(t-1)} + b_{hr}) \\ i_t = sigmoid(W_{ii} x_t + b_{ii} + W_hi
      h_{(t-1)} + b_{hi}) \\ n_t = \tanh(W_{in} x_t + b_{in} + r_t *
      (W_{hn} h_{(t-1)}+ b_{hn})) \\ h_t = (1 - i_t) * n_t + i_t *
      h_{(t-1)} \\ \end{array}

   where h_t is the hidden state at time *t*, x_t is the hidden state
   of the previous layer at time *t* or input_t for the first layer,
   and r_t, i_t, n_t are the reset, input, and new gates,
   respectively.

   :Parameters:
      * **hidden_size** (*int*) -- Number of units in output symbol.

      * **i2h_weight_initializer** (*str* or `Initializer
        <https://mxnet.incubator.apache.org/versions/master/api/python/optimization/optimization.html#mxnet.initializer.Initializer>`_)
        -- Initializer for the input weights matrix, used for the
        linear transformation of the inputs.

      * **h2h_weight_initializer** (*str* or `Initializer
        <https://mxnet.incubator.apache.org/versions/master/api/python/optimization/optimization.html#mxnet.initializer.Initializer>`_)
        -- Initializer for the recurrent weights matrix, used for the
        linear transformation of the recurrent state.

      * **i2h_bias_initializer** (*str* or `Initializer
        <https://mxnet.incubator.apache.org/versions/master/api/python/optimization/optimization.html#mxnet.initializer.Initializer>`_)
        -- Initializer for the bias vector.

      * **h2h_bias_initializer** (*str* or `Initializer
        <https://mxnet.incubator.apache.org/versions/master/api/python/optimization/optimization.html#mxnet.initializer.Initializer>`_)
        -- Initializer for the bias vector.

      * **prefix** (str, default '>>gru_<<') -- prefix for name of
        *Block`s (and name of weight if params is `None*).

      * **params** (*Parameter* or *None*) -- Container for weight
        sharing between cells. Created if *None*.

**class gluon.rnn.HybridRecurrentCell(prefix=None,
params=None)**

   HybridRecurrentCell supports hybridize.

**class gluon.rnn.LSTM(hidden_size, num_layers=1, layout='TNC',
dropout=0, bidirectional=False, input_size=0,
i2h_weight_initializer=None, h2h_weight_initializer=None,
i2h_bias_initializer='zeros', h2h_bias_initializer='zeros',
**kwargs)**

   Applies a multi-layer long short-term memory (LSTM) RNN to an input
   sequence.

   For each element in the input sequence, each layer computes the
   following function:

      \begin{array}{ll} i_t = sigmoid(W_{ii} x_t + b_{ii} + W_{hi}
      h_{(t-1)} + b_{hi}) \\ f_t = sigmoid(W_{if} x_t + b_{if} +
      W_{hf} h_{(t-1)} + b_{hf}) \\ g_t = \tanh(W_{ig} x_t + b_{ig} +
      W_{hc} h_{(t-1)} + b_{hg}) \\ o_t = sigmoid(W_{io} x_t + b_{io}
      + W_{ho} h_{(t-1)} + b_{ho}) \\ c_t = f_t * c_{(t-1)} + i_t *
      g_t \\ h_t = o_t * \tanh(c_t) \end{array}

   where h_t is the hidden state at time *t*, c_t is the cell state at
   time *t*, x_t is the hidden state of the previous layer at time *t*
   or input_t for the first layer, and i_t, f_t, g_t, o_t are the
   input, forget, cell, and out gates, respectively.

   :Parameters:
      * **hidden_size** (*int*) -- The number of features in the
        hidden state h.

      * **num_layers** (*int*, *default 1*) -- Number of recurrent
        layers.

      * **layout** (*str*, *default 'TNC'*) -- The format of input
        and output tensors. T, N and C stand for sequence length,
        batch size, and feature dimensions respectively.

      * **dropout** (*float*, *default 0*) -- If non-zero,
        introduces a dropout layer on the outputs of each RNN layer
        except the last layer.

      * **bidirectional** (*bool*, *default False*) -- If *True*,
        becomes a bidirectional RNN.

      * **i2h_weight_initializer** (*str* or `Initializer
        <https://mxnet.incubator.apache.org/versions/master/api/python/optimization/optimization.html#mxnet.initializer.Initializer>`_)
        -- Initializer for the input weights matrix, used for the
        linear transformation of the inputs.

      * **h2h_weight_initializer** (*str* or `Initializer
        <https://mxnet.incubator.apache.org/versions/master/api/python/optimization/optimization.html#mxnet.initializer.Initializer>`_)
        -- Initializer for the recurrent weights matrix, used for the
        linear transformation of the recurrent state.

      * **i2h_bias_initializer** (*str* or *Initializer*,
        *default 'lstmbias'*) -- Initializer for the bias vector. By
        default, bias for the forget gate is initialized to 1 while
        all other biases are initialized to zero.

      * **h2h_bias_initializer** (*str* or `Initializer
        <https://mxnet.incubator.apache.org/versions/master/api/python/optimization/optimization.html#mxnet.initializer.Initializer>`_)
        -- Initializer for the bias vector.

      * **input_size** (*int*, *default 0*) -- The number of
        expected features in the input x. If not specified, it will be
        inferred from input.

      * **prefix** (*str* or *None*) -- Prefix of this *Block*.

      * **params** (*ParameterDict* or *None*) -- Shared Parameters
        for this *Block*.

   Input shapes:
      The input shape depends on *layout*. For *layout='TNC'*, the
      input has shape *(sequence_length, batch_size, input_size)*

   Output shape:
      The output shape depends on *layout*. For *layout='TNC'*, the
      output has shape *(sequence_length, batch_size, num_hidden)*. If
      *bidirectional* is True, output shape will instead be
      *(sequence_length, batch_size, 2*num_hidden)*

   Recurrent state:
      The recurrent state is a list of two NDArrays. Both has shape
      *(num_layers, batch_size, num_hidden)*. If *bidirectional* is
      True, each recurrent state will instead have shape
      *(2*num_layers, batch_size, num_hidden)*. If input recurrent
      state is None, zeros are used as default begin states, and the
      output recurrent state is omitted.

   Example:

.. code-block:: python
   layer =  gluon.rnn.LSTM(100, 3)
   layer.initialize()
   input = nd.random.uniform(shape=(5, 3, 10))
   # by default zeros are used as begin state
   output = layer(input)
   # manually specify begin state.
   h0 = nd.random.uniform(shape=(3, 3, 100))
   c0 = nd.random.uniform(shape=(3, 3, 100))
   output, hn = layer(input, [h0, c0])

**class gluon.rnn.LSTMCell(hidden_size,
i2h_weight_initializer=None, h2h_weight_initializer=None,
i2h_bias_initializer='zeros', h2h_bias_initializer='zeros',
input_size=0, prefix=None, params=None)**

   Long-Short Term Memory (LSTM) network cell.

   Each call computes the following function:

      \begin{array}{ll} i_t = sigmoid(W_{ii} x_t + b_{ii} + W_{hi}
      h_{(t-1)} + b_{hi}) \\ f_t = sigmoid(W_{if} x_t + b_{if} +
      W_{hf} h_{(t-1)} + b_{hf}) \\ g_t = \tanh(W_{ig} x_t + b_{ig} +
      W_{hc} h_{(t-1)} + b_{hg}) \\ o_t = sigmoid(W_{io} x_t + b_{io}
      + W_{ho} h_{(t-1)} + b_{ho}) \\ c_t = f_t * c_{(t-1)} + i_t *
      g_t \\ h_t = o_t * \tanh(c_t) \end{array}

   where h_t is the hidden state at time *t*, c_t is the cell state at
   time *t*, x_t is the hidden state of the previous layer at time *t*
   or input_t for the first layer, and i_t, f_t, g_t, o_t are the
   input, forget, cell, and out gates, respectively.

   :Parameters:
      * **hidden_size** (*int*) -- Number of units in output symbol.

      * **i2h_weight_initializer** (*str* or `Initializer
        <https://mxnet.incubator.apache.org/versions/master/api/python/optimization/optimization.html#mxnet.initializer.Initializer>`_)
        -- Initializer for the input weights matrix, used for the
        linear transformation of the inputs.

      * **h2h_weight_initializer** (*str* or `Initializer
        <https://mxnet.incubator.apache.org/versions/master/api/python/optimization/optimization.html#mxnet.initializer.Initializer>`_)
        -- Initializer for the recurrent weights matrix, used for the
        linear transformation of the recurrent state.

      * **i2h_bias_initializer** (*str* or `Initializer`,
        *default 'lstmbias'*) -- Initializer for the bias vector. By
        default, bias for the forget gate is initialized to 1 while
        all other biases are initialized to zero.

      * **h2h_bias_initializer** (*str* or `Initializer
        <https://mxnet.incubator.apache.org/versions/master/api/python/optimization/optimization.html#mxnet.initializer.Initializer>`_)
        -- Initializer for the bias vector.

      * **prefix** (str, default '>>lstm_<<') -- Prefix for name of
        *Block`s (and name of weight if params is `None*).

      * **params** (*Parameter* or *None*) -- Container for weight
        sharing between cells. Created if *None*.

**class gluon.rnn.ModifierCell(base_cell)**

   Base class for modifier cells. A modifier cell takes a base cell,
   apply modifications on it (e.g. Zoneout), and returns a new cell.

   After applying modifiers the base cell should no longer be called
   directly. The modifier cell should be used instead.

**class gluon.rnn.RNN(hidden_size, num_layers=1,
activation='relu', layout='TNC', dropout=0, bidirectional=False,
i2h_weight_initializer=None, h2h_weight_initializer=None,
i2h_bias_initializer='zeros', h2h_bias_initializer='zeros',
input_size=0, **kwargs)**

   Applies a multi-layer Elman RNN with *tanh* or *ReLU* non-linearity
   to an input sequence.

   For each element in the input sequence, each layer computes the
   following function:

      h_t = \tanh(w_{ih} * x_t + b_{ih}  +  w_{hh} * h_{(t-1)} +
      b_{hh})

   where h_t is the hidden state at time *t*, and x_t is the hidden
   state of the previous layer at time *t* or input_t for the first
   layer. If nonlinearity='relu', then *ReLU* is used instead of
   *tanh*.

   :Parameters:
      * **hidden_size** (*int*) -- The number of features in the
        hidden state h.

      * **num_layers** (*int*, *default 1*) -- Number of recurrent
        layers.

      * **activation** ({*'relu'* or *'tanh'*}, *default 'tanh'*)
        -- The activation function to use.

      * **layout** (*str*, *default 'TNC'*) -- The format of input
        and output tensors. T, N and C stand for sequence length,
        batch size, and feature dimensions respectively.

      * **dropout** (*float*, *default 0*) -- If non-zero,
        introduces a dropout layer on the outputs of each RNN layer
        except the last layer.

      * **bidirectional** (*bool*, *default False*) -- If *True*,
        becomes a bidirectional RNN.

      * **i2h_weight_initializer** (*str* or `Initializer
        <https://mxnet.incubator.apache.org/versions/master/api/python/optimization/optimization.html#mxnet.initializer.Initializer>`_)
        -- Initializer for the input weights matrix, used for the
        linear transformation of the inputs.

      * **h2h_weight_initializer** (*str* or `Initializer
        <https://mxnet.incubator.apache.org/versions/master/api/python/optimization/optimization.html#mxnet.initializer.Initializer>`_)
        -- Initializer for the recurrent weights matrix, used for the
        linear transformation of the recurrent state.

      * **i2h_bias_initializer** (*str* or `Initializer
        <https://mxnet.incubator.apache.org/versions/master/api/python/optimization/optimization.html#mxnet.initializer.Initializer>`_)
        -- Initializer for the bias vector.

      * **h2h_bias_initializer** (*str* or `Initializer
        <https://mxnet.incubator.apache.org/versions/master/api/python/optimization/optimization.html#mxnet.initializer.Initializer>`_)
        -- Initializer for the bias vector.

      * **input_size** (*int*, *default 0*) -- The number of
        expected features in the input x. If not specified, it will be
        inferred from input.

      * **prefix** (*str* or *None*) -- Prefix of this *Block*.

      * **params** (*ParameterDict* or *None*) -- Shared Parameters
        for this *Block*.

   Input shapes:
      The input shape depends on *layout*. For *layout='TNC'*, the
      input has shape *(sequence_length, batch_size, input_size)*

   Output shape:
      The output shape depends on *layout*. For *layout='TNC'*, the
      output has shape *(sequence_length, batch_size, num_hidden)*. If
      *bidirectional* is True, output shape will instead be
      *(sequence_length, batch_size, 2*num_hidden)*

   Recurrent state:
      The recurrent state is an NDArray with shape *(num_layers,
      batch_size, num_hidden)*. If *bidirectional* is True, the
      recurrent state shape will instead be *(2*num_layers,
      batch_size, num_hidden)* If input recurrent state is None, zeros
      are used as default begin states, and the output recurrent state
      is omitted.

   Example:

.. code-block:: python
   layer =  gluon.rnn.RNN(100, 3)
   layer.initialize()
   input = nd.random.uniform(shape=(5, 3, 10))
   # by default zeros are used as begin state
   output = layer(input)
   # manually specify begin state.
   h0 = nd.random.uniform(shape=(3, 3, 100))
   output, hn = layer(input, h0)

**class gluon.rnn.RNNCell(hidden_size, activation='tanh',
i2h_weight_initializer=None, h2h_weight_initializer=None,
i2h_bias_initializer='zeros', h2h_bias_initializer='zeros',
input_size=0, prefix=None, params=None)**

   Elman RNN recurrent neural network cell.

   Each call computes the following function:

      h_t = \tanh(w_{ih} * x_t + b_{ih}  +  w_{hh} * h_{(t-1)} +
      b_{hh})

   where h_t is the hidden state at time *t*, and x_t is the hidden
   state of the previous layer at time *t* or input_t for the first
   layer. If nonlinearity='relu', then *ReLU* is used instead of
   *tanh*.

   :Parameters:
      * **hidden_size** (*int*) -- Number of units in output symbol

      * **activation** (*str* or `Symbol
        <https://mxnet.incubator.apache.org/versions/master/api/python/symbol/symbol.html#mxnet.symbol.Symbol>`_, *default
        'tanh'*) -- Type of activation function.

      * **i2h_weight_initializer** (*str* or `Initializer
        <https://mxnet.incubator.apache.org/versions/master/api/python/optimization/optimization.html#mxnet.initializer.Initializer>`_)
        -- Initializer for the input weights matrix, used for the
        linear transformation of the inputs.

      * **h2h_weight_initializer** (*str* or `Initializer
        <https://mxnet.incubator.apache.org/versions/master/api/python/optimization/optimization.html#mxnet.initializer.Initializer>`_)
        -- Initializer for the recurrent weights matrix, used for the
        linear transformation of the recurrent state.

      * **i2h_bias_initializer** (*str* or `Initializer
        <https://mxnet.incubator.apache.org/versions/master/api/python/optimization/optimization.html#mxnet.initializer.Initializer>`_)
        -- Initializer for the bias vector.

      * **h2h_bias_initializer** (*str* or `Initializer
        <https://mxnet.incubator.apache.org/versions/master/api/python/optimization/optimization.html#mxnet.initializer.Initializer>`_)
        -- Initializer for the bias vector.

      * **prefix** (str, default '>>rnn_<<') -- Prefix for name of
        *Block`s (and name of weight if params is `None*).

      * **params** (*Parameter* or *None*) -- Container for weight
        sharing between cells. Created if *None*.

**class gluon.rnn.Recurrence(input, step_function, initial_states, return_full_state=False, direction_mode=’go_forward’)**

    :Parameters:
      * **input** (list of ndarray) – input to recurrence layer.
      * **step_function** (callable, ndarray.zeros, ndarray.ones, etc.) – A function that take intput and states, output the result and next states.
      * **initial_states** (list of ndarray) - the initial status for recurrence.
      * **return_full_state** (bool) – Boolen that defaults False, return full states or not.
      * **direction_mode** (str) – One of three values: go_forward, go_backward, or bidirection


    The signature of step_function:
      :Parameters:
         * **x** (*NDArray*) –- the input NDArray
         * **states** (*NDArray*) -– list of NDArray

      :Returns:
         * **outputs** (*NDArray*) –- the result of step function
         * **states**
         (*list of NDArray*) -– the states for next recurrent step.


    **while** (condition, body, loop_vars, name=’ ’)
      :Parameters:
         * **condition** –- A callable function that returns a Boolean.
         * **body** –- A callable function that maps from loop_vars to new loop_vars.
         * **loop_vars** –- list of variable that will be used to calculate condition and update in body.
         * **name** –- name for this loop.

    **fold** (fold_function, elems, initial_state=None, go_backwards=False, name=’ ’)
      :Parameters:
        * **fold_function** – A callable function that take inputs and states, output the result.
        * **elems** – the inputs to fold.
        * **initial_state** – the initial state for fold.
        * **go_backwards** ( default False ) – fold from left to right or in reversed order.
        * **name** – name for this layer.

    **map** ( map_function,  elems,  name=’ ’ )
      :Parameters:
        * **map_function** – A callable function that take input elements, output the result.
        * **elems** – the input to map.
        * **name** – name for this layer.

**class gluon.rnn.RecurrentCell(prefix=None, params=None)**

   Abstract base class for RNN cells

   :Parameters:
      * **prefix** (*str*, *optional*) -- Prefix for names of
        *Block`s (this prefix is also used for names of weights if
        `params* is *None* i.e. if *params* are being created and not
        reused)

      * **params** (*Parameter* or *None*, *optional*) --
        Container for weight sharing between cells. A new Parameter
        container is created if *params* is *None*.

   **begin_state(batch_size=0, func=<function zeros>, **kwargs)**

      Initial state for this cell.

      :Parameters:
         * **func** (*callable*, *default symbol.zeros*) --

           Function for creating initial state.

           For Symbol API, func can be *symbol.zeros*,
           *symbol.uniform*, *symbol.var etc*. Use *symbol.var* if you
           want to directly feed input as states.

           For NDArray API, func can be *ndarray.zeros*,
           *ndarray.ones*, etc.

         * **batch_size** (*int*, *default 0*) -- Only required for
           NDArray API. Size of the batch ('N' in layout) dimension of
           input.

         * ****kwargs** -- Additional keyword arguments passed to
           func. For example *mean*, *std*, *dtype*, etc.

      :Returns:
         **states** -- Starting states for the first RNN step.

      :Return type:
         nested list of Symbol

   **forward(inputs, states)**

      Unrolls the recurrent cell for one time step.

      :Parameters:
         * **inputs** (*sym.Variable*) -- Input symbol, 2D, of shape
           (batch_size * num_units).

         * **states** (*list of sym.Variable*) -- RNN state from
           previous step or the output of begin_state().

      :Returns:
         * **output** (*Symbol*) -- Symbol corresponding to the output
           from the RNN when unrolling for a single time step.

         * **states** (*list of Symbol*) -- The new state of this RNN
           after this unrolling. The type of this symbol is same as
           the output of *begin_state()*. This can be used as an input
           state to the next time step of this RNN.

      ``begin_state()``
            This function can provide the states for the first time
            step.

         ``unroll()``
            This function unrolls an RNN for a given number of (>=1)
            time steps.

   **reset()**

      Reset before re-using the cell for another graph.

   **state_info(batch_size=0)**

      shape and layout information of states

   **unroll(length, inputs, begin_state=None, layout='NTC',
   merge_outputs=None)**

      Unrolls an RNN cell across time steps.

      :Parameters:
         * **length** (*int*) -- Number of steps to unroll.

         * **inputs** (`Symbol
           <https://mxnet.incubator.apache.org/versions/master/api/python/symbol/symbol.html#mxnet.symbol.Symbol>`_, *list of
           Symbol*, or *None*) --

           If *inputs* is a single Symbol (usually the output of
           Embedding symbol), it should have shape (batch_size,
           length, ...) if *layout* is 'NTC', or (length, batch_size,
           ...) if *layout* is 'TNC'.

           If *inputs* is a list of symbols (usually output of
           previous unroll), they should all have shape (batch_size,
           ...).

         * **begin_state** (*nested list of Symbol*, *optional*) --
           Input states created by *begin_state()* or output state of
           another cell. Created from *begin_state()* if *None*.

         * **layout** (*str*, *optional*) -- *layout* of input
           symbol. Only used if inputs is a single Symbol.

         * **merge_outputs** (*bool*, *optional*) -- If *False*,
           returns outputs as a list of Symbols. If *True*,
           concatenates output across time steps and returns a single
           symbol with shape (batch_size, length, ...) if layout is
           'NTC', or (length, batch_size, ...) if layout is 'TNC'. If
           *None*, output whatever is faster.

      :Returns:
         * **outputs** (*list of Symbol or Symbol*) -- Symbol (if
           *merge_outputs* is True) or list of Symbols (if
           *merge_outputs* is False) corresponding to the output from
           the RNN from this unrolling.

         * **states** (*list of Symbol*) -- The new state of this RNN
           after this unrolling. The type of this symbol is same as
           the output of *begin_state()*.

**class gluon.rnn.ResidualCell(base_cell)**

   Adds residual connection as described in Wu et al, 2016
   (https://arxiv.org/abs/1609.08144). Output of the cell is output of
   the base cell plus input.

**class gluon.rnn.SequentialRNNCell(prefix=None, params=None)**

   Sequentially stacking multiple RNN cells.

   **add(cell)**

      Appends a cell into the stack.

      :Parameters:
         **cell** (*rnn cell*) --

**class gluon.rnn.ZoneoutCell(base_cell, zoneout_outputs=0.0,
zoneout_states=0.0)**

   Applies Zoneout on base cell.
