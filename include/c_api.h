#ifndef CMI_C_API_H
#define CMI_C_API_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

/**
 * @file c_api.h
 * @brief C API for CPP-ML-Interface.
 *
 * Provides procedural C99-compatible bindings for coupling HPC simulations with
 * ML inference providers (PhyDLL, SmartSim, AIxelerator, Dummy, and Generic Callbacks).
 *
 * All functions returning @ref cmi_status return @ref CMI_SUCCESS (0) on success,
 * or an error code > 0 on failure. The most recent error message on the calling
 * thread can be retrieved via @ref cmi_get_last_error().
 */

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup c_api C API Reference
 * @brief Complete C99 procedural interface for CPP-ML-Interface.
 * @{
 */

/* ========================================================================= */
/* Status / Error Codes                                                      */
/* ========================================================================= */

/**
 * @defgroup c_status Status & Error Handling
 * @ingroup c_api
 * @brief Error reporting functions and return codes.
 * @{
 */

/**
 * @brief Return status codes for C API operations.
 */
enum cmi_status {
    CMI_SUCCESS                = 0,  /**< Operation completed successfully. */
    CMI_ERROR_INVALID_ARGUMENT = 1,  /**< An invalid parameter value was supplied. */
    CMI_ERROR_NULL_POINTER     = 2,  /**< A required pointer parameter was NULL. */
    CMI_ERROR_OUT_OF_RANGE     = 3,  /**< Index or dimension out of valid range. */
    CMI_ERROR_TYPE_MISMATCH    = 4,  /**< Data type mismatch between components or arguments. */
    CMI_ERROR_RUNTIME          = 5,  /**< Internal C++ runtime error or exception. */
    CMI_ERROR_NOT_IMPLEMENTED  = 6,  /**< Requested functionality is not implemented. */
    CMI_ERROR_UNKNOWN          = 99  /**< Unspecified fatal error occurred. */
};

/**
 * @brief Retrieves the last thread-local error message.
 *
 * If no error has occurred, returns an empty string `""`. The returned string
 * is valid until the next C API call on the calling thread.
 *
 * @return Null-terminated string describing the last error.
 */
const char* cmi_get_last_error(void);

/**
 * @brief Converts a @ref cmi_status code into a static human-readable string.
 *
 * @param[in] status A @ref cmi_status code.
 * @return Static string constant representing the status.
 */
const char* cmi_status_string(int status);

/**
 * @brief Clears the thread-local error state.
 */
void cmi_clear_last_error(void);

/** @} */ /* end of c_status */

/* ========================================================================= */
/* Data Types (corresponds to MLCouplingDataType)                            */
/* ========================================================================= */

/**
 * @defgroup c_types Data Types and Layouts
 * @ingroup c_api
 * @brief Enumerations describing tensor primitive types and memory layouts.
 * @{
 */

/**
 * @brief Supported primitive scalar types for tensors and data containers.
 */
enum cmi_data_type {
    CMI_DTYPE_INVALID = 0, /**< Uninitialized or unsupported type. */
    CMI_DTYPE_DOUBLE  = 1, /**< 64-bit IEEE floating point (double). */
    CMI_DTYPE_FLOAT   = 2, /**< 32-bit IEEE floating point (float). */
    CMI_DTYPE_INT8    = 3, /**< 8-bit signed integer (int8_t). */
    CMI_DTYPE_INT16   = 4, /**< 16-bit signed integer (int16_t). */
    CMI_DTYPE_INT32   = 5, /**< 32-bit signed integer (int32_t). */
    CMI_DTYPE_INT64   = 6, /**< 64-bit signed integer (int64_t). */
    CMI_DTYPE_UINT8   = 7, /**< 8-bit unsigned integer (uint8_t). */
    CMI_DTYPE_UINT16  = 8  /**< 16-bit unsigned integer (uint16_t). */
};

/* ========================================================================= */
/* Memory Layouts (corresponds to MLCouplingMemoryLayout)                     */
/* ========================================================================= */

/**
 * @brief Memory ordering and storage layout for multi-dimensional tensors.
 */
enum cmi_memory_layout {
    CMI_LAYOUT_INVALID            = 0, /**< Uninitialized layout. */
    CMI_LAYOUT_NESTED             = 1, /**< Row-major pointer-tree (C/C++ array of pointers). */
    CMI_LAYOUT_CONTIGUOUS         = 2, /**< Row-major contiguous single-block buffer (C/C++ native). */
    CMI_LAYOUT_FORTRAN_NESTED     = 3, /**< Column-major pointer-tree. */
    CMI_LAYOUT_FORTRAN_CONTIGUOUS = 4  /**< Column-major contiguous single-block buffer (Fortran native). */
};

/* ========================================================================= */
/* Ownership (corresponds to MLCouplingOwnership)                            */
/* ========================================================================= */

/**
 * @brief Buffer memory ownership policy for tensors.
 */
enum cmi_ownership {
    CMI_OWNERSHIP_EXTERNAL = 0, /**< Tensor borrows buffer; caller retains free responsibility. */
    CMI_OWNERSHIP_OWNED    = 1  /**< Tensor owns buffer; destroyed when tensor is freed. */
};

/* ========================================================================= */
/* Merge Strategies (corresponds to MLCouplingMergeStrategy)                 */
/* ========================================================================= */

/**
 * @brief Merge strategies for flexible tiered inference inputs across ranks.
 */
enum cmi_merge_strategy {
    CMI_MERGE_LIST  = 0, /**< Concatenate tensors per batch item along feature dimension. */
    CMI_MERGE_STACK = 1, /**< Interleave tensors across a new dimension. */
    CMI_MERGE_AUTO  = 2, /**< Automatically determine optimal merge strategy. */
    CMI_MERGE_NONE  = 3  /**< Disable merging (single tensor expected). */
};

/* ========================================================================= */
/* Config Parameter Types for Name-Based Instantiation                       */
/* ========================================================================= */

/**
 * @brief Type descriptor for dynamic constructor arguments passed to name-based factories.
 */
enum cmi_config_param_type {
    CMI_PARAM_RAW    = 0, /**< Raw pointer passed as-is. */
    CMI_PARAM_INT64  = 1, /**< 64-bit integer value (int64_t). */
    CMI_PARAM_DOUBLE = 2, /**< 64-bit floating point value (double). */
    CMI_PARAM_STRING = 3, /**< Null-terminated string (const char*). */
    CMI_PARAM_BOOL   = 4  /**< Boolean value (0 = false, 1 = true). */
};

/** @} */ /* end of c_types */

/* ========================================================================= */
/* Opaque Handles                                                            */
/* ========================================================================= */

/**
 * @defgroup c_handles Opaque Handles
 * @ingroup c_api
 * @brief Type-safe opaque pointers representing C++ library objects.
 * @{
 */

typedef struct cmi_tensor_s*        cmi_tensor_t;        /**< Handle to an MLCouplingTensor. */
typedef struct cmi_data_s*          cmi_data_t;          /**< Handle to an MLCouplingData container. */
typedef struct cmi_behavior_s*      cmi_behavior_t;      /**< Handle to an MLCouplingBehavior instance. */
typedef struct cmi_normalization_s* cmi_normalization_t; /**< Handle to an MLCouplingNormalization instance. */
typedef struct cmi_library_s*       cmi_library_t;       /**< Handle to an MLCouplingLibrary backend. */
typedef struct cmi_application_s*   cmi_application_t;   /**< Handle to an MLCouplingApplication pipeline. */
typedef struct cmi_coupling_s*      cmi_coupling_t;      /**< Handle to an MLCoupling orchestrator. */

/** @} */ /* end of c_handles */

/* ========================================================================= */
/* Tensor API                                                                */
/* ========================================================================= */

/**
 * @defgroup c_tensor Tensor API
 * @ingroup c_api
 * @brief Creation, inspection, manipulation, and destruction of tensors.
 * @{
 */

/**
 * @brief Wraps an existing flat memory buffer into an MLCouplingTensor without copying.
 *
 * @param[out] out Receives the created tensor handle.
 * @param[in]  data Pointer to the existing raw data buffer.
 * @param[in]  dims Array of dimension sizes.
 * @param[in]  ndims Number of dimensions in @p dims.
 * @param[in]  data_type Value from @ref cmi_data_type indicating the scalar type.
 * @param[in]  layout Value from @ref cmi_memory_layout (e.g. @ref CMI_LAYOUT_CONTIGUOUS or @ref CMI_LAYOUT_FORTRAN_CONTIGUOUS).
 * @param[in]  ownership @ref CMI_OWNERSHIP_EXTERNAL or @ref CMI_OWNERSHIP_OWNED.
 * @return @ref CMI_SUCCESS on success, or an error status code on failure.
 */
int cmi_tensor_create_flat(cmi_tensor_t* out,
                           void* data,
                           const int* dims,
                           int ndims,
                           int data_type,
                           int layout,
                           int ownership);

/**
 * @brief Creates a new tensor by copying data from the given flat buffer into newly allocated memory.
 *
 * @param[out] out Receives the created tensor handle.
 * @param[in]  data Pointer to the source data buffer to copy.
 * @param[in]  dims Array of dimension sizes.
 * @param[in]  ndims Number of dimensions in @p dims.
 * @param[in]  data_type Value from @ref cmi_data_type.
 * @param[in]  layout Memory layout of the source and newly created tensor.
 * @return @ref CMI_SUCCESS on success, or an error status code on failure.
 */
int cmi_tensor_create_from_copy(cmi_tensor_t* out,
                                const void* data,
                                const int* dims,
                                int ndims,
                                int data_type,
                                int layout);

/**
 * @brief Destroys a tensor handle and releases its memory if owned.
 *
 * @param[in] tensor The tensor to destroy. Safe if NULL.
 * @return @ref CMI_SUCCESS on success.
 */
int cmi_tensor_destroy(cmi_tensor_t tensor);

/**
 * @brief Retrieves the pointer to the underlying raw buffer of a tensor.
 *
 * @param[in]  tensor The tensor handle.
 * @param[out] out_data Receives the raw data pointer.
 * @return @ref CMI_SUCCESS on success, or an error code.
 */
int cmi_tensor_get_data(cmi_tensor_t tensor, void** out_data);

/**
 * @brief Retrieves the shape (dimensions) of a tensor.
 *
 * @param[in]  tensor The tensor handle.
 * @param[out] out_dims Receives pointer to internal dimensions array.
 * @param[out] out_ndims Receives the number of dimensions.
 * @return @ref CMI_SUCCESS on success, or an error code.
 */
int cmi_tensor_get_dims(cmi_tensor_t tensor, const int** out_dims, int* out_ndims);

/**
 * @brief Retrieves the memory layout of a tensor.
 *
 * @param[in]  tensor The tensor handle.
 * @param[out] out_layout Receives the @ref cmi_memory_layout value.
 * @return @ref CMI_SUCCESS on success, or an error code.
 */
int cmi_tensor_get_layout(cmi_tensor_t tensor, int* out_layout);

/**
 * @brief Retrieves the data type of the tensor elements.
 *
 * @param[in]  tensor The tensor handle.
 * @param[out] out_data_type Receives the @ref cmi_data_type value.
 * @return @ref CMI_SUCCESS on success, or an error code.
 */
int cmi_tensor_get_data_type(cmi_tensor_t tensor, int* out_data_type);

/**
 * @brief Retrieves the total number of scalar elements in a tensor.
 *
 * @param[in]  tensor The tensor handle.
 * @param[out] out_numel Receives the total element count.
 * @return @ref CMI_SUCCESS on success, or an error code.
 */
int cmi_tensor_get_numel(cmi_tensor_t tensor, size_t* out_numel);

/**
 * @brief Retrieves the ownership status of a tensor's buffer.
 *
 * @param[in]  tensor The tensor handle.
 * @param[out] out_ownership Receives @ref cmi_ownership value.
 * @return @ref CMI_SUCCESS on success, or an error code.
 */
int cmi_tensor_get_ownership(cmi_tensor_t tensor, int* out_ownership);

/**
 * @brief Flattens a tensor into a contiguous buffer of the specified target layout.
 *
 * @param[in]  tensor Source tensor handle.
 * @param[in]  target_layout Target layout (@ref CMI_LAYOUT_CONTIGUOUS or @ref CMI_LAYOUT_FORTRAN_CONTIGUOUS).
 * @param[out] out Receives the new flattened tensor handle.
 * @return @ref CMI_SUCCESS on success, or an error code.
 */
int cmi_tensor_flatten(cmi_tensor_t tensor, int target_layout, cmi_tensor_t* out);

/**
 * @brief Creates an independent deep copy of a tensor.
 *
 * @param[in]  tensor Source tensor handle.
 * @param[out] out Receives the deep-copied tensor handle.
 * @return @ref CMI_SUCCESS on success, or an error code.
 */
int cmi_tensor_deep_copy(cmi_tensor_t tensor, cmi_tensor_t* out);

/** @} */ /* end of c_tensor */

/* ========================================================================= */
/* Data (Collection of Tensors) API                                          */
/* ========================================================================= */

/**
 * @defgroup c_data Data Container API
 * @ingroup c_api
 * @brief Creation, manipulation, and inspection of MLCouplingData containers.
 * @{
 */

/**
 * @brief Creates an empty MLCouplingData container for tensors of the given type.
 *
 * @param[out] out Receives the created data container handle.
 * @param[in]  data_type Value from @ref cmi_data_type.
 * @return @ref CMI_SUCCESS on success, or an error code.
 */
int cmi_data_create(cmi_data_t* out, int data_type);

/**
 * @brief Destroys an MLCouplingData container and releases all associated tensors.
 *
 * @param[in] data Container handle to destroy. Safe if NULL.
 * @return @ref CMI_SUCCESS on success.
 */
int cmi_data_destroy(cmi_data_t data);

/**
 * @brief Appends a tensor to an MLCouplingData container.
 *
 * @param[in] data The target data container.
 * @param[in] tensor The tensor to append.
 * @return @ref CMI_SUCCESS on success, or an error code.
 */
int cmi_data_add_tensor(cmi_data_t data, cmi_tensor_t tensor);

/**
 * @brief Retrieves a tensor by 0-based index from the container.
 *
 * @param[in]  data Container handle.
 * @param[in]  index Zero-based tensor index.
 * @param[out] out_tensor Receives the tensor handle.
 * @return @ref CMI_SUCCESS on success, or @ref CMI_ERROR_OUT_OF_RANGE if index is invalid.
 */
int cmi_data_get_tensor(cmi_data_t data, int index, cmi_tensor_t* out_tensor);

/**
 * @brief Retrieves the number of tensors contained in the container.
 *
 * @param[in]  data Container handle.
 * @param[out] out_size Receives the number of tensors.
 * @return @ref CMI_SUCCESS on success, or an error code.
 */
int cmi_data_size(cmi_data_t data, int* out_size);

/**
 * @brief Retrieves the element data type of the container.
 *
 * @param[in]  data Container handle.
 * @param[out] out_data_type Receives the @ref cmi_data_type value.
 * @return @ref CMI_SUCCESS on success, or an error code.
 */
int cmi_data_get_data_type(cmi_data_t data, int* out_data_type);

/**
 * @brief Creates an independent deep copy of an entire MLCouplingData container and all its tensors.
 *
 * @param[in]  data Source container handle.
 * @param[out] out Receives the deep-copied container handle.
 * @return @ref CMI_SUCCESS on success, or an error code.
 */
int cmi_data_deep_copy(cmi_data_t data, cmi_data_t* out);

/** @} */ /* end of c_data */

/* ========================================================================= */
/* Behavior API                                                              */
/* ========================================================================= */

/**
 * @defgroup c_behavior Behavior API
 * @ingroup c_api
 * @brief Controls inference scheduling, timestepping deltas, and data transmission.
 * @{
 */

/** @brief Callback signature to determine if ML inference should occur at the current step. */
typedef bool (*cmi_should_infer_fn)(void* user_data);

/** @brief Callback signature returning how many time steps the simulation should advance after inference. */
typedef int  (*cmi_time_step_delta_fn)(void* user_data);

/** @brief Callback signature to determine if simulation data should be sent to the database. */
typedef bool (*cmi_should_send_data_fn)(void* user_data);

/**
 * @brief Creates a generic behavior instance driven by custom user function pointers.
 *
 * @param[out] out Receives the created behavior handle.
 * @param[in]  should_infer Callback determining whether to infer.
 * @param[in]  time_step_delta Callback returning the step delta.
 * @param[in]  should_send_data Callback determining whether to send data.
 * @param[in]  user_data Arbitrary context pointer passed to all callbacks.
 * @return @ref CMI_SUCCESS on success, or an error code.
 */
int cmi_behavior_create_generic(cmi_behavior_t* out,
                                cmi_should_infer_fn should_infer,
                                cmi_time_step_delta_fn time_step_delta,
                                cmi_should_send_data_fn should_send_data,
                                void* user_data);

/**
 * @brief Creates a default behavior instance (infers every step, delta = 1, sends data every step).
 *
 * @param[out] out Receives the created behavior handle.
 * @return @ref CMI_SUCCESS on success, or an error code.
 */
int cmi_behavior_create_default(cmi_behavior_t* out);

/**
 * @brief Creates a periodic behavior instance for scheduled inference.
 *
 * @param[out] out Receives the created behavior handle.
 * @param[in]  inference_interval Timestep interval between ML inference calls.
 * @param[in]  coupled_steps_before_inference Number of physics steps before triggering inference.
 * @param[in]  coupled_steps_stride Step stride for coupled physics iterations.
 * @param[in]  step_increment_after_inference Timesteps skipped forward after successful inference.
 * @return @ref CMI_SUCCESS on success, or an error code.
 */
int cmi_behavior_create_periodic(cmi_behavior_t* out,
                                 int inference_interval,
                                 int coupled_steps_before_inference,
                                 int coupled_steps_stride,
                                 int step_increment_after_inference);

/**
 * @brief Instantiates a behavior subclass registered by name via the code-generated factory.
 *
 * @param[out] out Receives the created behavior handle.
 * @param[in]  name Registered class name or alias (e.g. "Periodic", "Default").
 * @param[in]  param_names Array of constructor parameter names.
 * @param[in]  param_values Array of pointers to parameter values.
 * @param[in]  param_types Array of @ref cmi_config_param_type describing parameter types.
 * @param[in]  param_count Number of parameters.
 * @return @ref CMI_SUCCESS on success, or an error code.
 */
int cmi_behavior_create_by_name(cmi_behavior_t* out,
                                const char* name,
                                const char** param_names,
                                const void** param_values,
                                const int* param_types,
                                int param_count);

/**
 * @brief Evaluates whether inference should be performed at the current step.
 *
 * @param[in]  behavior Behavior handle.
 * @param[out] out Receives true if inference should proceed, false otherwise.
 * @return @ref CMI_SUCCESS on success, or an error code.
 */
int cmi_behavior_should_perform_inference(cmi_behavior_t behavior, bool* out);

/**
 * @brief Retrieves the time step delta to advance after inference.
 *
 * @param[in]  behavior Behavior handle.
 * @param[out] out Receives the step delta.
 * @return @ref CMI_SUCCESS on success, or an error code.
 */
int cmi_behavior_time_step_delta(cmi_behavior_t behavior, int* out);

/**
 * @brief Evaluates whether simulation data should be sent to the backend.
 *
 * @param[in]  behavior Behavior handle.
 * @param[out] out Receives true if data should be transmitted, false otherwise.
 * @return @ref CMI_SUCCESS on success, or an error code.
 */
int cmi_behavior_should_send_data(cmi_behavior_t behavior, bool* out);

/**
 * @brief Destroys a behavior instance.
 *
 * @param[in] behavior Handle to destroy. Safe if NULL.
 * @return @ref CMI_SUCCESS on success.
 */
int cmi_behavior_destroy(cmi_behavior_t behavior);

/** @} */ /* end of c_behavior */

/* ========================================================================= */
/* Normalization API                                                         */
/* ========================================================================= */

/**
 * @defgroup c_normalization Normalization API
 * @ingroup c_api
 * @brief Feature scaling and normalization transformations for inference data.
 * @{
 */

/** @brief Callback signature to normalize input data before feeding into ML inference. */
typedef void (*cmi_normalize_fn)(cmi_data_t input_data, void* user_data);

/** @brief Callback signature to denormalize raw ML output back into simulation scales. */
typedef void (*cmi_denormalize_fn)(cmi_data_t output_data, void* user_data);

/**
 * @brief Creates a generic normalization component driven by custom function pointers.
 *
 * @param[out] out Receives the created normalization handle.
 * @param[in]  in_type Scalar type for input data (@ref cmi_data_type).
 * @param[in]  out_type Scalar type for output data (@ref cmi_data_type).
 * @param[in]  normalize_fn Function pointer to perform normalization.
 * @param[in]  denormalize_fn Function pointer to perform denormalization.
 * @param[in]  user_data Arbitrary context pointer passed to callbacks.
 * @return @ref CMI_SUCCESS on success, or an error code.
 */
int cmi_normalization_create_generic(cmi_normalization_t* out,
                                     int in_type,
                                     int out_type,
                                     cmi_normalize_fn normalize_fn,
                                     cmi_denormalize_fn denormalize_fn,
                                     void* user_data);

/**
 * @brief Creates a min-max normalization component with scalar bounding values.
 *
 * @param[out] out Receives the created normalization handle.
 * @param[in]  in_type Input scalar type (@ref cmi_data_type).
 * @param[in]  out_type Output scalar type (@ref cmi_data_type).
 * @param[in]  in_min Minimum expected value for input scaling.
 * @param[in]  in_max Maximum expected value for input scaling.
 * @param[in]  out_min Minimum expected value for output denormalization.
 * @param[in]  out_max Maximum expected value for output denormalization.
 * @return @ref CMI_SUCCESS on success, or an error code.
 */
int cmi_normalization_create_minmax(cmi_normalization_t* out,
                                    int in_type,
                                    int out_type,
                                    double in_min,
                                    double in_max,
                                    double out_min,
                                    double out_max);

/**
 * @brief Instantiates a normalization subclass registered by name via the code-generated factory.
 *
 * @param[out] out Receives the created normalization handle.
 * @param[in]  name Registered class name or alias (e.g. "MinMax").
 * @param[in]  in_type Input scalar type.
 * @param[in]  out_type Output scalar type.
 * @param[in]  param_names Array of constructor parameter names.
 * @param[in]  param_values Array of pointers to parameter values.
 * @param[in]  param_types Array of @ref cmi_config_param_type.
 * @param[in]  param_count Number of parameters.
 * @return @ref CMI_SUCCESS on success, or an error code.
 */
int cmi_normalization_create_by_name(cmi_normalization_t* out,
                                     const char* name,
                                     int in_type,
                                     int out_type,
                                     const char** param_names,
                                     const void** param_values,
                                     const int* param_types,
                                     int param_count);

/**
 * @brief Normalizes input data in-place using the configured normalization component.
 *
 * @param[in] norm Normalization handle.
 * @param[in] input_data Data container to normalize.
 * @return @ref CMI_SUCCESS on success, or an error code.
 */
int cmi_normalization_normalize_input(cmi_normalization_t norm, cmi_data_t input_data);

/**
 * @brief Denormalizes output data in-place using the configured normalization component.
 *
 * @param[in] norm Normalization handle.
 * @param[in] output_data Data container to denormalize.
 * @return @ref CMI_SUCCESS on success, or an error code.
 */
int cmi_normalization_denormalize_output(cmi_normalization_t norm, cmi_data_t output_data);

/**
 * @brief Destroys a normalization instance.
 *
 * @param[in] norm Handle to destroy. Safe if NULL.
 * @return @ref CMI_SUCCESS on success.
 */
int cmi_normalization_destroy(cmi_normalization_t norm);

/** @} */ /* end of c_normalization */

/* ========================================================================= */
/* Library API                                                               */
/* ========================================================================= */

/**
 * @defgroup c_library Library API
 * @ingroup c_api
 * @brief ML provider lifecycle, model inference execution, training, and tiered flexible staging.
 * @{
 */

/** @brief Callback signature for custom user inference function. */
typedef void (*cmi_inference_fn)(cmi_data_t input, cmi_data_t output, void* user_data);

/** @brief Callback signature for custom user training function. */
typedef void (*cmi_train_fn)(cmi_data_t input, cmi_data_t target, void* user_data);

/**
 * @brief Creates a generic ML library provider driven by custom inference and training callbacks.
 *
 * @param[out] out Receives the created library handle.
 * @param[in]  in_type Input scalar type (@ref cmi_data_type).
 * @param[in]  out_type Output scalar type (@ref cmi_data_type).
 * @param[in]  inference_fn User callback executing inference.
 * @param[in]  user_data Arbitrary context pointer passed to callbacks.
 * @return @ref CMI_SUCCESS on success, or an error code.
 */
int cmi_library_create_generic(cmi_library_t* out,
                               int in_type,
                               int out_type,
                               cmi_inference_fn inference_fn,
                               void* user_data);

/**
 * @brief Instantiates an ML library provider by name (e.g. "Phydll", "Smartsim", "Aixelerator", "Dummy").
 *
 * @param[out] out Receives the created library handle.
 * @param[in]  name Registered provider name or alias.
 * @param[in]  in_type Input scalar type.
 * @param[in]  out_type Output scalar type.
 * @param[in]  param_names Array of constructor parameter names.
 * @param[in]  param_values Array of pointers to parameter values.
 * @param[in]  param_types Array of @ref cmi_config_param_type.
 * @param[in]  param_count Number of parameters.
 * @return @ref CMI_SUCCESS on success, or an error code.
 */
int cmi_library_create_by_name(cmi_library_t* out,
                               const char* name,
                               int in_type,
                               int out_type,
                               const char** param_names,
                               const void** param_values,
                               const int* param_types,
                               int param_count);

/**
 * @brief Sets the MPI rank associated with the library instance.
 *
 * @param[in] lib Library handle.
 * @param[in] rank Integer MPI process rank.
 * @return @ref CMI_SUCCESS on success, or an error code.
 */
int cmi_library_set_rank(cmi_library_t lib, int rank);

/**
 * @brief Configures the merge strategy for tiered flexible inference inputs.
 *
 * @param[in] lib Library handle.
 * @param[in] strategy Value from @ref cmi_merge_strategy (e.g. @ref CMI_MERGE_LIST or @ref CMI_MERGE_STACK).
 * @return @ref CMI_SUCCESS on success, or an error code.
 */
int cmi_library_set_merge_strategy(cmi_library_t lib, int strategy);

/**
 * @brief Runs synchronous inference through the library backend.
 *
 * @param[in]  lib Library handle.
 * @param[in]  input Container holding inference input tensors.
 * @param[out] output Container to receive output tensors.
 * @return @ref CMI_SUCCESS on success, or an error code.
 */
int cmi_library_inference(cmi_library_t lib, cmi_data_t input, cmi_data_t output);

/**
 * @brief Runs synchronous in-situ training through the library backend.
 *
 * @param[in] lib Library handle.
 * @param[in] input Container holding input feature tensors.
 * @param[in] target Container holding target label/ground-truth tensors.
 * @return @ref CMI_SUCCESS on success, or an error code.
 */
int cmi_library_train(cmi_library_t lib, cmi_data_t input, cmi_data_t target);

/**
 * @brief Stages input data for ordered flexible inference (Tier 1).
 *
 * @param[in] lib Library handle.
 * @param[in] data Input data container to stage.
 * @return @ref CMI_SUCCESS on success, or an error code.
 */
int cmi_library_flex_ordered_set(cmi_library_t lib, cmi_data_t data);

/**
 * @brief Stages target data for ordered flexible training (Tier 1).
 *
 * @param[in] lib Library handle.
 * @param[in] data Target data container to stage.
 * @return @ref CMI_SUCCESS on success, or an error code.
 */
int cmi_library_flex_ordered_set_target(cmi_library_t lib, cmi_data_t data);

/**
 * @brief Triggers inference on all staged ordered data, with optional fallback output.
 *
 * @param[in]  lib Library handle.
 * @param[out] fallback_output Output container used if no staged data was present.
 * @return @ref CMI_SUCCESS on success, or an error code.
 */
int cmi_library_flex_ordered_inference(cmi_library_t lib, cmi_data_t fallback_output);

/**
 * @brief Stages input data with an associated string key for keyed flexible inference (Tier 2).
 *
 * @param[in] lib Library handle.
 * @param[in] key Identifying key name for this tensor slot.
 * @param[in] data Data container to stage.
 * @return @ref CMI_SUCCESS on success, or an error code.
 */
int cmi_library_flex_keyed_set(cmi_library_t lib, const char* key, cmi_data_t data);

/**
 * @brief Stages target data with an associated string key for keyed flexible training (Tier 2).
 *
 * @param[in] lib Library handle.
 * @param[in] key Identifying key name.
 * @param[in] data Target data container to stage.
 * @return @ref CMI_SUCCESS on success, or an error code.
 */
int cmi_library_flex_keyed_set_target(cmi_library_t lib, const char* key, cmi_data_t data);

/**
 * @brief Triggers inference on staged keyed inputs according to key order specifications.
 *
 * @param[in]  lib Library handle.
 * @param[in]  in_keys Array of input key names defining tensor order.
 * @param[in]  in_key_count Number of keys in @p in_keys.
 * @param[in]  out_keys Array of output key names defining output tensor order.
 * @param[in]  out_key_count Number of keys in @p out_keys.
 * @param[out] fallback_output Container receiving output if keys are empty.
 * @return @ref CMI_SUCCESS on success, or an error code.
 */
int cmi_library_flex_keyed_inference(cmi_library_t lib,
                                     const char** in_keys,
                                     int in_key_count,
                                     const char** out_keys,
                                     int out_key_count,
                                     cmi_data_t fallback_output);

/**
 * @brief Destroys an ML library instance and shuts down provider resources.
 *
 * @param[in] lib Handle to destroy. Safe if NULL.
 * @return @ref CMI_SUCCESS on success.
 */
int cmi_library_destroy(cmi_library_t lib);

/** @} */ /* end of c_library */

/* ========================================================================= */
/* Application API                                                           */
/* ========================================================================= */

/**
 * @defgroup c_application Application API
 * @ingroup c_api
 * @brief Preprocessing, postprocessing, and custom step orchestration.
 * @{
 */

/** @brief Callback signature for custom user preprocessing. */
typedef void (*cmi_preprocess_fn)(cmi_data_t input, cmi_data_t out_library_input, void* user_data);

/** @brief Callback signature for custom user postprocessing. */
typedef void (*cmi_postprocess_fn)(cmi_data_t library_output, cmi_data_t out_coupling_output, void* user_data);

/**
 * @brief Creates a generic application component with bound coupling input and output buffers.
 *
 * @param[out] out Receives the created application handle.
 * @param[in]  coupling_in_type Coupling input scalar type (@ref cmi_data_type).
 * @param[in]  coupling_out_type Coupling output scalar type (@ref cmi_data_type).
 * @param[in]  library_in_type Library input scalar type (@ref cmi_data_type).
 * @param[in]  library_out_type Library output scalar type (@ref cmi_data_type).
 * @param[in]  coupling_input Container holding coupling inputs.
 * @param[in]  coupling_output Container holding coupling outputs.
 * @param[in]  normalization Optional normalization handle (can be NULL).
 * @return @ref CMI_SUCCESS on success, or an error code.
 */
int cmi_application_create_generic(cmi_application_t* out,
                                   int coupling_in_type,
                                   int coupling_out_type,
                                   int library_in_type,
                                   int library_out_type,
                                   cmi_data_t coupling_input,
                                   cmi_data_t coupling_output,
                                   cmi_normalization_t normalization);

/**
 * @brief Instantiates an application component by name via the code-generated factory.
 *
 * @param[out] out Receives the created application handle.
 * @param[in]  name Registered application class name or alias (e.g. "TurbulenceClosure").
 * @param[in]  coupling_in_type Coupling input type.
 * @param[in]  coupling_out_type Coupling output type.
 * @param[in]  library_in_type Library input type.
 * @param[in]  library_out_type Library output type.
 * @param[in]  param_names Array of constructor parameter names.
 * @param[in]  param_values Array of pointers to parameter values.
 * @param[in]  param_types Array of @ref cmi_config_param_type.
 * @param[in]  param_count Number of parameters.
 * @return @ref CMI_SUCCESS on success, or an error code.
 */
int cmi_application_create_by_name(cmi_application_t* out,
                                   const char* name,
                                   int coupling_in_type,
                                   int coupling_out_type,
                                   int library_in_type,
                                   int library_out_type,
                                   const char** param_names,
                                   const void** param_values,
                                   const int* param_types,
                                   int param_count);

/**
 * @brief Executes a single ML step via the application pipeline (preprocess -> infer -> postprocess).
 *
 * @param[in]  app Application handle.
 * @param[in]  lib Library handle.
 * @param[in]  behavior Behavior handle.
 * @param[out] out_delta Receives simulation step delta to advance.
 * @return @ref CMI_SUCCESS on success, or an error code.
 */
int cmi_application_ml_step(cmi_application_t app,
                            cmi_library_t lib,
                            cmi_behavior_t behavior,
                            int* out_delta);

/**
 * @brief Destroys an application instance.
 *
 * @param[in] app Handle to destroy. Safe if NULL.
 * @return @ref CMI_SUCCESS on success.
 */
int cmi_application_destroy(cmi_application_t app);

/** @} */ /* end of c_application */

/* ========================================================================= */
/* MLCoupling Orchestrator API                                               */
/* ========================================================================= */

/**
 * @defgroup c_coupling Coupling Orchestrator API
 * @ingroup c_api
 * @brief Top-level pipeline coordination connecting Library, Application, and Behavior.
 * @{
 */

/**
 * @brief Creates a top-level MLCoupling orchestrator from existing components.
 *
 * @param[out] out Receives the created coupling handle.
 * @param[in]  coupling_in_type Coupling input scalar type (@ref cmi_data_type).
 * @param[in]  coupling_out_type Coupling output scalar type (@ref cmi_data_type).
 * @param[in]  library_in_type Library input scalar type (@ref cmi_data_type).
 * @param[in]  library_out_type Library output scalar type (@ref cmi_data_type).
 * @param[in]  library Library handle.
 * @param[in]  application Application handle.
 * @param[in]  behavior Behavior handle.
 * @return @ref CMI_SUCCESS on success, or an error code.
 */
int cmi_coupling_create(cmi_coupling_t* out,
                        int coupling_in_type,
                        int coupling_out_type,
                        int library_in_type,
                        int library_out_type,
                        cmi_library_t library,
                        cmi_application_t application,
                        cmi_behavior_t behavior);

/**
 * @brief Creates an MLCoupling orchestrator configured from a TOML configuration file.
 *
 * @param[out] out Receives the created coupling handle.
 * @param[in]  config_path Path to the TOML configuration file.
 * @param[in]  coupling_in_type Coupling input type.
 * @param[in]  coupling_out_type Coupling output type.
 * @param[in]  library_in_type Library input type.
 * @param[in]  library_out_type Library output type.
 * @return @ref CMI_SUCCESS on success, or an error code.
 */
int cmi_coupling_create_from_config(cmi_coupling_t* out,
                                    const char* config_path,
                                    int coupling_in_type,
                                    int coupling_out_type,
                                    int library_in_type,
                                    int library_out_type);

/**
 * @brief Executes a single iteration of the coupled pipeline.
 *
 * Checks behavior conditions, triggers preprocessing, inference, and postprocessing,
 * and updates time step advancement.
 *
 * @param[in]  coupling Coupling orchestrator handle.
 * @param[out] out_delta Receives step delta (e.g. 1 if advanced normally, or larger if accelerated).
 * @return @ref CMI_SUCCESS on success, or an error code.
 */
int cmi_coupling_step(cmi_coupling_t coupling, int* out_delta);

/**
 * @brief Executes a training step for in-situ model updating.
 *
 * @param[in] coupling Coupling orchestrator handle.
 * @param[in] step_id Simulation iteration identifier for training tracking.
 * @return @ref CMI_SUCCESS on success, or an error code.
 */
int cmi_coupling_train_step(cmi_coupling_t coupling, int64_t step_id);

/**
 * @brief Destroys an MLCoupling orchestrator.
 *
 * @param[in] coupling Handle to destroy. Safe if NULL.
 * @return @ref CMI_SUCCESS on success.
 */
int cmi_coupling_destroy(cmi_coupling_t coupling);

/** @} */ /* end of c_coupling */

/* ========================================================================= */
/* Introspection API                                                         */
/* ========================================================================= */

/**
 * @defgroup c_introspection Registry Introspection API
 * @ingroup c_api
 * @brief Query registered classes, aliases, and available implementations at runtime.
 * @{
 */

/**
 * @brief Queries the number of registered subclasses in a given category.
 *
 * @param[in] category Category string ("behavior", "library", "normalization", or "application").
 * @return Number of registered classes, or -1 if the category is invalid.
 */
int cmi_get_class_count(const char* category);

/**
 * @brief Retrieves the registered name of a class by index within a category.
 *
 * @param[in] category Category string ("behavior", "library", "normalization", or "application").
 * @param[in] index Zero-based class index.
 * @return Static string containing the class name, or empty string if invalid index/category.
 */
const char* cmi_get_class_name(const char* category, int index);

/**
 * @brief Checks if a specific class name or alias is registered in a category.
 *
 * @param[in] category Category string ("behavior", "library", "normalization", or "application").
 * @param[in] name Class name or alias to query.
 * @return True if registered, false otherwise.
 */
bool cmi_is_class_registered(const char* category, const char* name);

/** @} */ /* end of c_introspection */

/* ========================================================================= */
/* Backward Compatibility with Original C API                                */
/* ========================================================================= */

/**
 * @defgroup c_legacy Legacy C API
 * @ingroup c_api
 * @brief Backward-compatible functions for legacy callers.
 * @{
 */

/**
 * @brief Legacy factory function to create an ML library instance.
 *
 * @deprecated Use @ref cmi_library_create_by_name or @ref cmi_library_create_generic instead.
 * @param[in] name Provider name.
 * @param[in] in_selection Input type selector integer.
 * @param[in] out_selection Output type selector integer.
 * @param[in] param_names Array of parameter names.
 * @param[in] params Array of parameter value pointers.
 * @param[in] param_count Number of parameters.
 * @return Raw pointer handle to the created library, or NULL on error.
 */
void* create_library(const char* name,
                     int in_selection,
                     int out_selection,
                     char** param_names,
                     void** params,
                     int param_count);

/**
 * @brief Legacy function to destroy an ML library instance.
 *
 * @deprecated Use @ref cmi_library_destroy instead.
 * @param[in] handle Handle returned by @ref create_library. Safe if NULL.
 */
void destroy_library(void* handle);

/** @} */ /* end of c_legacy */

/** @} */ /* end of c_api */

#ifdef __cplusplus
}
#endif

#endif /* CMI_C_API_H */
