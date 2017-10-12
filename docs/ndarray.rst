
NDArray API
***********


Overview
========

The ``NDArray`` API, defined in the ``ndarray`` (or simply ``nd``)
package, provides imperative tensor operations on CPU/GPU. An
``NDArray`` represents a multi-dimensional, fixed-size homogenous
array.

Further details on NDArray and its API Reference can be found in the 
`MXNet NDArray API <https://mxnet.incubator.apache.org/versions/master/api/python/ndarray/ndarray.html>`_.

We will also support ragged ndarray:

**class gluon.ndarray.RaggedNDArray(handle, writable=True)**

    The Ragged NDArray is a special NDArray for handling ragged arrays, also known as jagged arrays. 
    Member arrays can be of different sizes which will be packed into a RaggedNDArray format. 
    The returned RaggedNDArray is formed as <V, I> as follows:
      V: values stored in a NDArray
      I: indices stored in a NDArray

    Example:
    
      Given a sequence with 3 samples with lengths of 10, 5, and 3 we can store these as a Ragged NDArray as follows:
          
      +--------------------------+---------+---+---+
      | Sample1                                    |
      +--------------------------+---------+---+---+
      | S _ a _ m _ p _ l _ e  2 | Sample3 |   |   |
      +--------------------------+---------+---+---+

      Additionally, we store metadata about the samples in a NDArray:
      
      +----+---+---+
      | 10 | 5 | 3 |
      +----+---+---+
          
      The returned RaggedNDArray (V, I) has the samples (V)  as an NDArray, and (I) is the NDArray containing the metadata.

    :Parameters:
      * **data** (*NDArray*) -– Values stored in NDArray or SparseNDArray format.
      * **indices** (*NDArray*) -– Metadata stored in NDArray format.

    :Returns:
      A RaggedNDArray.

    :Return type:
      RaggedNDArray

    **to_ndarray** (*padding_value*) 
      Convert to a dense NDArray with a given padding value.
      
    **gather** (*condition*) 
      Take a condition of the same length as the current RaggedNDArray and return a new RaggedNDArray.
      
    **scatter** (*condition*) 
      Inverse of the gather op.
      
    **slice** (*begin*, *end*, *step=1*) 
      Returns a sliced view of the array.
      
    **softmax** (*axis*)  
      Computes the softmax across the given axis, or the packed sequence.
      
    **first** ( )  
      Returns the first element of the sequence.
      
    **last** ( )  
      Returns the last element of the sequence.
      
    **past_value** (*initial_state=0*, *time_step=1*)  
      Create a new RaggedNDArray where all elements have shifted by time_step backwards to a previous value.

    **future_value** (*initial_state=0*, *time_step=1*)
      Inverse of the past_value op.

    **reduce_sum** (*axis*)  
      Computes the sum of the input elements across the given axis, or the packed sequence.

    **reduce_min** (*axis*)  
      Computes the min of the input elements across the given axis, or the packed sequence.

    **reduce_mean** (*axis*)  
      Computes the mean of the input elements across the given axis, or the packed sequence.

    **reduce_prod** (*axis*)  
      Computes the prod of the input elements across the given axis, or the packed sequence.

    **reduce_logsumexp** (*axis*)  
      Computes the logsumexp of the input elements across the given axis, or the packed sequence.
