// LIST OPERATORS
/*!
 * \brief list all the available operator names, include entries.
 * \param out_size the size of returned array
 * \param out_array the output operator name array.
 * \return 0 when success, -1 when failure happens
 */
int GLListAllOpNames(nn_uint *out_size,
                     const char*** out_array);
/*!
 * \brief Get operator handle given name.
 * \param op_name The name of the operator.
 * \param op_out The returnning op handle.
 */
int GLGetOpHandle(const char* op_name,
                  OpHandle* op_out);

// NDARRAY CREATION AND MANUPULATION
/*!
 * \brief create a NDArray with specified shape
 * \param shape the pointer to the shape
 * \param ndim the dimension of the shape
 * \param dev_type device type, specify device we want to take
 * \param dev_id the device id of the specific device
 * \param delay_alloc whether to delay allocation until
 *    the narray is first mutated
 * \param out the returning handle
 * \return 0 when success, -1 when failure happens
 */
int GLNDArrayCreate(const uint32_t *shape,
                    uint32_t ndim,
                    int dev_type,
                    int dev_id,
                    int delay_alloc,
                    NDArrayHandle *out);
/*!
 * \brief free the ndarray handle
 * \param handle the handle to be freed
 * \return 0 when success, -1 when failure happens
 */
int GLNDArrayFree(NDArrayHandle handle);
/*!
 * \brief invoke a nnvm op and imperative function
 * \param creator the op
 * \param num_inputs number of input NDArrays
 * \param inputs input NDArrays
 * \param num_outputs number of output NDArrays
 * \param outputs output NDArrays
 * \param num_params number of keyword parameters
 * \param param_keys keys for keyword parameters
 * \param param_vals values for keyword parameters
 * \return 0 when success, -1 when failure happens
 */
int GLImperativeInvoke(OpHandle creaor,
                       int num_inputs,
                       NDArrayHandle *inputs,
                       int *num_outputs,
                       NDArrayHandle **outputs,
                       int num_params,
                       const char **param_keys,
                       const char **param_vals);
/*!
 * \brief Perform a synchronize copy from a continugous CPU memory region.
 *
 *  This function will call WaitToWrite before the copy is performed.
 *  This is useful to copy data from existing memory region that are
 *  not wrapped by NDArray(thus dependency not being tracked).
 *
 * \param handle the NDArray handle
 * \param data the data source to copy from.
 * \param size the memory size we want to copy from.
 */
int GLNDArraySyncCopyFromCPU(NDArrayHandle handle,
                             const void *data,
                             size_t size);
/*!
 * \brief Perform a synchronize copyto a continugous CPU memory region.
 *
 *  This function will call WaitToRead before the copy is performed.
 *  This is useful to copy data from existing memory region that are
 *  not wrapped by NDArray(thus dependency not being tracked).
 *
 * \param handle the NDArray handle
 * \param data the data source to copy into.
 * \param size the memory size we want to copy into.
 */
int GLNDArraySyncCopyToCPU(NDArrayHandle handle,
                           void *data,
                           size_t size);
/*!
 * \brief get the shape of the array
 * \param handle the handle to the narray
 * \param out_dim the output dimension
 * \param out_pdata pointer holder to get data pointer of the shape
 * \return 0 when success, -1 when failure happens
 */
int GLNDArrayGetShape(NDArrayHandle handle,
                      uint32_t *out_dim,
                      const uint32_t **out_pdata);
/*!
 * \brief get the context of the NDArray
 * \param handle the handle to the narray
 * \param out_dev_type the output device type
 * \param out_dev_id the output device id
 * \return 0 when success, -1 when failure happens
 */
int GLNDArrayGetContext(NDArrayHandle handle,
                        int *out_dev_type,
                        int *out_dev_id);
/*!
 * \brief get the type of the data in NDArray
 * \param handle the handle to the narray
 * \param out_dtype pointer holder to get type of data
 * \return 0 when success, -1 when failure happens
 */
int GLNDArrayGetDType(NDArrayHandle handle,
                      int *out_dtype);
/*!
 * \brief get the storage type of the array. kDense, kSparse, kRagged, etc.
 */
int GLNDArrayGetStorageType(NDArrayHandle handle,
                            int *out_storage_type);
/*!
 * \brief detach and ndarray from computation graph by clearing entry_
 * \param handle NDArray handle
 * \return 0 when success, -1 when failure happens
 */
int GLNDArrayDetach(NDArrayHandle handle, NDArrayHandle *out);
/*!
 * \brief Wait until all the pending writes with respect NDArray are finished.
 *  Always call this before read data out synchronizely.
 * \param handle the NDArray handle
 * \return 0 when success, -1 when failure happens
 */
int GLNDArrayWaitToRead(NDArrayHandle handle);
/*!
 * \brief Wait until all the pending read/write with respect NDArray are finished.
 *  Always call this before write data into NDArray synchronizely.
 * \param handle the NDArray handle
 * \return 0 when success, -1 when failure happens
 */
int GLNDArrayWaitToWrite(NDArrayHandle handle);
/*!
 * \brief wait until all delayed operations in
 *   the system is completed
 * \return 0 when success, -1 when failure happens
 */
int GLNDArrayWaitAll();


// AUTOGRAD
/*!
 * \brief set whether to record operator for autograd
 * \param is_recording 1 when recording, 0 when not recording.
 * \param prev returns the previous status before this set.
 * \return 0 when success, -1 when failure happens
 */
int GLAutogradSetIsRecording(int is_recording, int* prev);
/*!
 * \brief set whether to record operator for autograd
 * \param is_training 1 when training, 0 when testing
 * \param prev returns the previous status before this set.
 * \return 0 when success, -1 when failure happens
 */
int GLAutogradSetIsTraining(int is_training, int* prev);
/*!
 * \brief get whether autograd recording is on
 * \param curr returns the current status.
 * \return 0 when success, -1 when failure happens
 */
int GLAutogradIsRecording(bool* curr);
/*!
 * \brief get whether training mode is on
 * \param curr returns the current status.
 * \return 0 when success, -1 when failure happens
 */
int GLAutogradIsTraining(bool* curr);
/*!
 * \brief mark NDArrays as variables to compute gradient for autograd
 * \param num_var number of variable NDArrays
 * \param var_handles variable NDArrays
 * \return 0 when success, -1 when failure happens
 */
int GLAutogradMarkVariables(uint32_t num_var,
                            NDArrayHandle *var_handles,
                            uint32_t *reqs_array,
                            NDArrayHandle *grad_handles);
/*!
 * \brief compute the gradient of outputs w.r.t variabels
 * \param num_output number of output NDArray
 * \param output_handles output NDArrays
 * \param ograd_handles head gradient for NDArrays
 * \param num_variables number of variables
 * \param
 * \param retain_graph whether to keep the graph after backward
 * \param is_train whether to do backward for training or inference
 * \return 0 when success, -1 when failure happens
 */
int GLAutogradBackward(uint32_t num_output,
                       NDArrayHandle *output_handles,
                       NDArrayHandle *ograd_handles,
                       uint32_t num_variables,
                       NDArrayHandle *var_handles,
                       int retain_graph,
                       int create_graph,
                       int is_train,
                       NDArrayHandle **grad_handles,
                       int **grad_stypes);


// SYMBOLIC GRAPH CREATION AND QUERY
/*!
 * \brief Create an AtomicSymbol.
 * \param creator the OpHandle
 * \param num_param the number of parameters
 * \param keys the keys to the params
 * \param vals the vals of the params
 * \param out pointer to the created symbol handle
 * \return 0 when success, -1 when failure happens
 */
int GLSymbolCreate(OpHandle creator,
                   uint32_t num_param,
                   const char **keys,
                   const char **vals,
                   SymbolHandle *out);
/*!
 * \brief Free the symbol handle.
 * \param symbol the symbol
 * \return 0 when success, -1 when failure happens
 */
int GLSymbolFree(SymbolHandle symbol);
/*!
 * \brief Compose the symbol on other symbols.
 *
 *  This function will change the sym hanlde.
 *  To achieve function apply behavior, copy the symbol first
 *  before apply.
 *
 * \param sym the symbol to apply
 * \param name the name of symbol
 * \param num_args number of arguments
 * \param keys the key of keyword args (optional)
 * \param args arguments to sym
 * \return 0 when success, -1 when failure happens
 */
int GLSymbolCompose(SymbolHandle sym,
                    const char *name,
                    uint32_t num_args,
                    const char** keys,
                    SymbolHandle* args);
/*!
 * \brief List arguments in the symbol.
 * \param symbol the symbol
 * \param out_size output size
 * \param out_str_array pointer to hold the output string array
 * \return 0 when success, -1 when failure happens
 */
int GLSymbolListArguments(SymbolHandle symbol,
                          uint32_t *out_size,
                          const char ***out_str_array);
/*!
 * \brief Get a symbol that contains all the internals.
 * \param symbol The symbol
 * \param out The output symbol whose outputs are all the internals.
 * \return 0 when success, -1 when failure happens
 */
int GLSymbolGetInternals(SymbolHandle symbol,
                         SymbolHandle *out);
/*!
 * \brief Get index-th outputs of the symbol.
 * \param symbol The symbol
 * \param index the Index of the output.
 * \param out The output symbol whose outputs are the index-th symbol.
 * \return 0 when success, -1 when failure happens
 */
int GLSymbolGetOutput(SymbolHandle symbol,
                      uint32_t index,
                      SymbolHandle *out);


// CACHED GRAPH EXECUTION
/*!
 * \brief create cached operator
 * \param handle symbolic graph
 * \param out created cached op
 */
int GLCreateCachedOp(SymbolHandle handle,
                     CachedOpHandle *out);

/*!
 * \brief free cached operator
 * \param handle cached op
 */
int GLFreeCachedOp(CachedOpHandle handle);

/*!
 * \brief invoke cached operator
 * \param handle cached op
 * \param num_inputs number of input ndarrays
 * \param inputs input ndarrays
 * \param num_outputs number of returned outputs
 * \param outputs returned output ndarrays
 */
int GLInvokeCachedOp(CachedOpHandle handle,
                     int num_inputs,
                     NDArrayHandle *inputs,
                     int *num_outputs,
                     NDArrayHandle **outputs);
