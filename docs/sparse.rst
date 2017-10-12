
Sparse NDArray API
******************


Overview
========

This document lists the routines of the *n*-dimensional sparse array
package:

+-------------------------------------------------------+--------------------------------------------------------------------------------------------+
| gluon.ndarray.sparse                                  | Sparse NDArray API of gluon.                                                               |
+-------------------------------------------------------+--------------------------------------------------------------------------------------------+

The ``CSRNDArray`` and ``RowSparseNDArray`` API, defined in the
``ndarray.sparse`` package, provides imperative sparse tensor
operations on CPU.

An ``CSRNDArray`` inherits from ``NDArray``, and represents a
two-dimensional, fixed-size array in compressed sparse row format.


Further details on Sparse NDArray and its API Reference can be found in the
`MXNet Sparse NDArray API <https://mxnet.incubator.apache.org/versions/master/api/python/ndarray/sparse.html>`_.

**class gluon.ndarray.sparse.CSRNDArray(handle, writable=True)**

   A sparse representation of 2D NDArray in the Compressed Sparse Row
   format.

   A CSRNDArray represents an NDArray as three separate arrays:
   *data*, *indptr* and *indices*. It uses the standard CSR
   representation where the column indices for row i are stored in
   ``indices[indptr[i]:indptr[i+1]]`` and their corresponding values
   are stored in ``data[indptr[i]:indptr[i+1]]``.

   The column indices for a given row are expected to be sorted in
   ascending order. Duplicate column entries for the same row are not
   allowed.

   Examples:

   >>> a = gluon.nd.array([[0, 1, 0], [2, 0, 0], [0, 0, 0], [0, 0, 3]])
   >>> a = a.tostype('csr')
   >>> a.indices.asnumpy()
   array([1, 0, 2])
   >>> a.indptr.asnumpy()
   array([0, 1, 2, 2, 3])
   >>> a.data.asnumpy()
   array([ 1.,  2.,  3.], dtype=float32)

   **__getitem__(key)**

      x.__getitem__(i) <=> x[i]

      Returns a sliced view of this array.

      :Parameters:
         **key** (*slice*) --
         Indexing key.

      Examples:

      >>> indptr = np.array([0, 2, 3, 6])
      >>> indices = np.array([0, 2, 2, 0, 1, 2])
      >>> data = np.array([1, 2, 3, 4, 5, 6])
      >>> a = gluon.nd.sparse.csr_matrix(data, indptr, indices, (3, 3))
      >>> a.asnumpy()
      array([[1, 0, 2],
             [0, 0, 3],
             [4, 5, 6]])
      >>> a[1:2].asnumpy()
      array([[0, 0, 3]], dtype=float32)

   **__setitem__(key, value)**

      x.__setitem__(i, y) <=> x[i]=y

      Set self[key] to value. Only slice key [:] is supported.

      :Parameters:
         * **key** (*slice*) -- The
           indexing key.

         * **value** (*NDArray* or *CSRNDArray*
           or **numpy.ndarray*) -- The value to set.

      Examples:

      >>> src = gluon.nd.sparse.zeros('csr', (3,3))
      >>> src.asnumpy()
      array([[ 0.,  0.,  0.],
             [ 0.,  0.,  0.],
             [ 0.,  0.,  0.]], dtype=float32)
      >>> # assign CSRNDArray with same storage type
      >>> x = gluon.nd.ones('row_sparse', (3,3)).tostype('csr')
      >>> x[:] = src
      >>> x.asnumpy()
      array([[ 1.,  1.,  1.],
             [ 1.,  1.,  1.],
             [ 1.,  1.,  1.]], dtype=float32)
      >>> # assign NDArray to CSRNDArray
      >>> x[:] = gluon.nd.ones((3,3)) * 2
      >>> x.asnumpy()
      array([[ 2.,  2.,  2.],
             [ 2.,  2.,  2.],
             [ 2.,  2.,  2.]], dtype=float32)

   ``indices``

      A deep copy NDArray of the indices array of the CSRNDArray. This
      generates a deep copy of the column indices of the current *csr*
      matrix.

      :Returns:
         This CSRNDArray's indices array.

      :Return type:
         *NDArray*

   ``indptr``

      A deep copy NDArray of the indptr array of the CSRNDArray. This
      generates a deep copy of the *indptr* of the current *csr*
      matrix.

      :Returns:
         This CSRNDArray's indptr array.

      :Return type:
         *NDArray*

   ``data``

      A deep copy NDArray of the data array of the CSRNDArray. This
      generates a deep copy of the *data* of the current *csr* matrix.

      :Returns:
         This CSRNDArray's data array.

      :Return type:
         *NDArray*

   **tostype(stype)**

      Return a copy of the array with chosen storage type.

      :Returns:
         A copy of the array with the chosen storage stype

      :Return type:
         *NDArray* or
         *CSRNDArray*
         

   **copyto(other)**

      Copies the value of this array to another array.

      If ``other`` is a ``NDArray`` or ``CSRNDArray`` object, then
      ``other.shape`` and ``self.shape`` should be the same. This
      function copies the value from ``self`` to ``other``.

      If ``other`` is a context, a new ``CSRNDArray`` will be first
      created on the target context, and the value of ``self`` is
      copied.

      :Parameters:
         **other** (*NDArray* or
         *CSRNDArray* or *Context*) -- The destination array or context.

      :Returns:
         The copied array. If ``other`` is an ``NDArray`` or
         ``CSRNDArray``, then the return value and ``other`` will
         point to the same ``NDArray`` or ``CSRNDArray``.

      :Return type:
         *NDArray* or
         *CSRNDArray*
         

   **as_in_context(context)**

      Returns an array on the target device with the same value as
      this array.

      If the target context is the same as ``self.context``, then
      ``self`` is returned.  Otherwise, a copy is made.

      :Parameters:
         **context** (*Context*) -- The target context.

      :Returns:
         The target array.

      :Return type:
         *NDArray*,
         *CSRNDArray*
          or
         *RowSparseNDArray*

      Examples:

      >>> x = gluon.nd.ones((2,3))
      >>> y = x.as_in_context(gluon.cpu())
      >>> y is x
      True
      >>> z = x.as_in_context(gluon.gpu(0))
      >>> z is x
      False

   **asnumpy()**

      Return a dense ``numpy.ndarray`` object with value copied from
      this array

   **asscalar()**

      Returns a scalar whose value is copied from this array.

      This function is equivalent to ``self.asnumpy()[0]``. This
      NDArray must have shape (1,).

      Examples:

      >>> x = gluon.nd.ones((1,), dtype='int32')
      >>> x.asscalar()
      1
      >>> type(x.asscalar())
      <type 'numpy.int32'>

   **astype(dtype)**

      Returns a copy of the array after casting to a specified type.
      :param dtype: The type of the returned array. :type dtype:
      numpy.dtype or str

      Examples:

      >>> x = gluon.nd.sparse.zeros('row_sparse', (2,3), dtype='float32')
      >>> y = x.astype('int32')
      >>> y.dtype
      <type 'numpy.int32'>

   ``context``

      Device context of the array.

      Examples:

      >>> x = gluon.nd.array([1, 2, 3, 4])
      >>> x.context
      cpu(0)
      >>> type(x.context)
      <class 'gluon.context.Context'>
      >>> y = gluon.nd.zeros((2,3), gluon.gpu(0))
      >>> y.context
      gpu(0)

   **copy()**

      Makes a copy of this ``NDArray``, keeping the same context.

      :Returns:
         The copied array

      :Return type:
         *NDArray*,
         *CSRNDArray*
          or
         *RowSparseNDArray*

      Examples:

      >>> x = gluon.nd.ones((2,3))
      >>> y = x.copy()
      >>> y.asnumpy()
      array([[ 1.,  1.,  1.],
             [ 1.,  1.,  1.]], dtype=float32)

   ``dtype``

      Data-type of the array's elements.

      :Returns:
         This NDArray's data type.

      :Return type:
         numpy.dtype

      Examples:

      >>> x = gluon.nd.zeros((2,3))
      >>> x.dtype
      <type 'numpy.float32'>
      >>> y = gluon.nd.zeros((2,3), dtype='int32')
      >>> y.dtype
      <type 'numpy.int32'>

   ``shape``

      Tuple of array dimensions.

      Examples:

      >>> x = gluon.nd.array([1, 2, 3, 4])
      >>> x.shape
      (4L,)
      >>> y = gluon.nd.zeros((2, 3, 4))
      >>> y.shape
      (2L, 3L, 4L)

   **slice(*args, **kwargs)**

      Convenience fluent method for *slice()*.

      The arguments are the same as for *slice()*, with this
      array as data.

   ``stype``

      Storage-type of the array.

   **wait_to_read()**

      Waits until all previous write operations on the current array
      are finished.

      This method guarantees that all previous write operations that
      pushed into the backend engine for execution are actually
      finished.

      Examples:

      >>> import time
      >>> tic = time.time()
      >>> a = gluon.nd.ones((1000,1000))
      >>> b = gluon.nd.dot(a, a)
      >>> print(time.time() - tic) # doctest: +SKIP
      0.003854036331176758
      >>> b.wait_to_read()
      >>> print(time.time() - tic) # doctest: +SKIP
      0.0893700122833252

   **zeros_like(*args, **kwargs)**

      Convenience fluent method for *zeros_like()*.

      The arguments are the same as for *zeros_like()*, with
      this array as data.


**class gluon.ndarray.sparse.SparseArray(handle, writable=True)**
   A sparse matrix in COOrdinate format <V, I, S> as follows:
      V: values stored in a NDArray
      I: indices stored in a NDArray
      S: shape stored in a NDArray

   Example:
      TBD

   :Parameters:
      * **data** (*NDArray*) –- Values stored in NDArray format.
      * **indices** (*NDArray*) –- Indices stored in NDArray format.
      * **shape** (*int* or *tuple of int*) –- The shape of the matrix empty.

   :Returns:
      A SparseNDArray.

   :Return type:
      SparseNDArray
