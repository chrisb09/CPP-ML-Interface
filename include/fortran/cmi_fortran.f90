! =============================================================================
! CPP-ML-Interface Fortran Bindings (cmi_fortran.f90)
! Provides modern Fortran (2003/2008) interfaces using iso_c_binding to the
! CPP-ML-Interface C API.
! =============================================================================

!> @file cmi_fortran.f90
!> @brief Fortran 2003/2008 bindings for CPP-ML-Interface.
!>
!> Offers zero-copy array wrapping with native column-major memory layout
!> (@ref CMI_LAYOUT_FORTRAN_CONTIGUOUS) and type-safe wrappers for coupling
!> Fortran numerical simulations with ML inference engines.

!> @defgroup fortran_api Fortran API Reference
!> @brief Fortran 2003/2008 bindings via `iso_c_binding`.
!> @{

!> @brief Main Fortran module for CPP-ML-Interface.
module cmi
  use, intrinsic :: iso_c_binding
  implicit none

  !> @name Status Return Codes
  !> @{
  integer(c_int), parameter :: CMI_SUCCESS                = 0  !< Success
  integer(c_int), parameter :: CMI_ERROR_INVALID_ARGUMENT = 1  !< Invalid argument supplied
  integer(c_int), parameter :: CMI_ERROR_NULL_POINTER     = 2  !< Unexpected null pointer
  integer(c_int), parameter :: CMI_ERROR_OUT_OF_RANGE     = 3  !< Index or dimension out of range
  integer(c_int), parameter :: CMI_ERROR_TYPE_MISMATCH    = 4  !< Data type mismatch
  integer(c_int), parameter :: CMI_ERROR_RUNTIME          = 5  !< Internal runtime error
  integer(c_int), parameter :: CMI_ERROR_NOT_IMPLEMENTED  = 6  !< Feature not implemented
  integer(c_int), parameter :: CMI_ERROR_UNKNOWN          = 99 !< Unknown fatal error
  !> @}

  !> @name Primitive Data Types
  !> @{
  integer(c_int), parameter :: CMI_DTYPE_INVALID = 0 !< Invalid / unspecified type
  integer(c_int), parameter :: CMI_DTYPE_DOUBLE  = 1 !< 64-bit IEEE double precision float
  integer(c_int), parameter :: CMI_DTYPE_FLOAT   = 2 !< 32-bit IEEE single precision float
  integer(c_int), parameter :: CMI_DTYPE_INT8    = 3 !< 8-bit signed integer
  integer(c_int), parameter :: CMI_DTYPE_INT16   = 4 !< 16-bit signed integer
  integer(c_int), parameter :: CMI_DTYPE_INT32   = 5 !< 32-bit signed integer
  integer(c_int), parameter :: CMI_DTYPE_INT64   = 6 !< 64-bit signed integer
  integer(c_int), parameter :: CMI_DTYPE_UINT8   = 7 !< 8-bit unsigned integer
  integer(c_int), parameter :: CMI_DTYPE_UINT16  = 8 !< 16-bit unsigned integer
  !> @}

  !> @name Memory Layouts
  !> @{
  integer(c_int), parameter :: CMI_LAYOUT_INVALID            = 0 !< Invalid layout
  integer(c_int), parameter :: CMI_LAYOUT_NESTED             = 1 !< Row-major pointer-tree (C/C++)
  integer(c_int), parameter :: CMI_LAYOUT_CONTIGUOUS         = 2 !< Row-major flat buffer (C/C++)
  integer(c_int), parameter :: CMI_LAYOUT_FORTRAN_NESTED     = 3 !< Column-major pointer-tree
  integer(c_int), parameter :: CMI_LAYOUT_FORTRAN_CONTIGUOUS = 4 !< Column-major flat buffer (Fortran native)
  !> @}

  !> @name Buffer Ownership
  !> @{
  integer(c_int), parameter :: CMI_OWNERSHIP_EXTERNAL = 0 !< Tensor borrows buffer (caller retains ownership)
  integer(c_int), parameter :: CMI_OWNERSHIP_OWNED    = 1 !< Tensor owns buffer and frees it upon destruction
  !> @}

  !> @name Merge Strategies
  !> @{
  integer(c_int), parameter :: CMI_MERGE_LIST  = 0 !< Concatenate along feature dimension
  integer(c_int), parameter :: CMI_MERGE_STACK = 1 !< Interleave across a new batch dimension
  integer(c_int), parameter :: CMI_MERGE_AUTO  = 2 !< Automatic strategy selection
  integer(c_int), parameter :: CMI_MERGE_NONE  = 3 !< No merging
  !> @}

  !> @name Name-Based Factory Parameter Types
  !> @{
  integer(c_int), parameter :: CMI_PARAM_RAW    = 0 !< Raw pointer passed as-is
  integer(c_int), parameter :: CMI_PARAM_INT64  = 1 !< 64-bit integer value
  integer(c_int), parameter :: CMI_PARAM_DOUBLE = 2 !< 64-bit floating point value
  integer(c_int), parameter :: CMI_PARAM_STRING = 3 !< Null-terminated string
  integer(c_int), parameter :: CMI_PARAM_BOOL   = 4 !< Boolean value (0/1)
  !> @}

  !> @name Opaque Derived Types
  !> @{

  !> @brief Opaque handle representing an MLCouplingTensor.
  type :: cmi_tensor
     type(c_ptr) :: ptr = c_null_ptr !< Underlying C pointer
  end type cmi_tensor

  !> @brief Opaque handle representing an MLCouplingData container.
  type :: cmi_data
     type(c_ptr) :: ptr = c_null_ptr !< Underlying C pointer
  end type cmi_data

  !> @brief Opaque handle representing an MLCouplingBehavior instance.
  type :: cmi_behavior
     type(c_ptr) :: ptr = c_null_ptr !< Underlying C pointer
  end type cmi_behavior

  !> @brief Opaque handle representing an MLCouplingNormalization instance.
  type :: cmi_normalization
     type(c_ptr) :: ptr = c_null_ptr !< Underlying C pointer
  end type cmi_normalization

  !> @brief Opaque handle representing an MLCouplingLibrary provider.
  type :: cmi_library
     type(c_ptr) :: ptr = c_null_ptr !< Underlying C pointer
  end type cmi_library

  !> @brief Opaque handle representing an MLCouplingApplication pipeline.
  type :: cmi_application
     type(c_ptr) :: ptr = c_null_ptr !< Underlying C pointer
  end type cmi_application

  !> @brief Opaque handle representing an MLCoupling orchestrator.
  type :: cmi_coupling
     type(c_ptr) :: ptr = c_null_ptr !< Underlying C pointer
  end type cmi_coupling

  !> @brief Opaque handle representing a TOML configuration override set.
  type :: cmi_config_overrides
     type(c_ptr) :: ptr = c_null_ptr !< Underlying C pointer
  end type cmi_config_overrides
  !> @}

  !> @brief Overloaded generic interface to wrap Fortran 1D, 2D, or 3D arrays into tensors without copying.
  interface cmi_tensor_wrap
     module procedure cmi_tensor_wrap_1d_float
     module procedure cmi_tensor_wrap_2d_float
     module procedure cmi_tensor_wrap_3d_float
     module procedure cmi_tensor_wrap_4d_float
     module procedure cmi_tensor_wrap_5d_float
     module procedure cmi_tensor_wrap_6d_float
     module procedure cmi_tensor_wrap_1d_double
     module procedure cmi_tensor_wrap_2d_double
     module procedure cmi_tensor_wrap_3d_double
     module procedure cmi_tensor_wrap_4d_double
     module procedure cmi_tensor_wrap_5d_double
     module procedure cmi_tensor_wrap_6d_double
  end interface cmi_tensor_wrap

  ! =========================================================================
  ! Low-level C API Interface Declarations
  ! =========================================================================
  interface
     !> @brief Returns pointer to thread-local error string.
     function c_cmi_get_last_error() bind(C, name="cmi_get_last_error") result(res)
       import :: c_ptr
       type(c_ptr) :: res
     end function c_cmi_get_last_error

     !> @brief Converts status integer into descriptive C string.
     function c_cmi_status_string(status) bind(C, name="cmi_status_string") result(res)
       import :: c_int, c_ptr
       integer(c_int), value :: status
       type(c_ptr) :: res
     end function c_cmi_status_string

     !> @brief Clears thread-local error state.
     subroutine c_cmi_clear_last_error() bind(C, name="cmi_clear_last_error")
     end subroutine c_cmi_clear_last_error

     !> @brief Queries registered class count by category string.
     function c_cmi_get_class_count(category) bind(C, name="cmi_get_class_count") result(res)
       import :: c_char, c_int
       character(kind=c_char), intent(in) :: category(*)
       integer(c_int) :: res
     end function c_cmi_get_class_count

     !> @brief Queries registered class name by category and index.
     function c_cmi_get_class_name(category, index) bind(C, name="cmi_get_class_name") result(res)
       import :: c_char, c_int, c_ptr
       character(kind=c_char), intent(in) :: category(*)
       integer(c_int), value :: index
       type(c_ptr) :: res
     end function c_cmi_get_class_name

     !> @brief Checks if a class or alias is registered in a category.
     function c_cmi_is_class_registered(category, name) bind(C, name="cmi_is_class_registered") result(res)
       import :: c_char, c_bool
       character(kind=c_char), intent(in) :: category(*)
       character(kind=c_char), intent(in) :: name(*)
       logical(c_bool) :: res
     end function c_cmi_is_class_registered

     !> @brief Creates a tensor wrapping a flat buffer.
     function c_cmi_tensor_create_flat(out_t, data_ptr, dims, ndims, data_type, layout, ownership) &
          bind(C, name="cmi_tensor_create_flat") result(st)
       import :: c_ptr, c_int
       type(c_ptr), intent(out) :: out_t
       type(c_ptr), value :: data_ptr
       integer(c_int), intent(in) :: dims(*)
       integer(c_int), value :: ndims
       integer(c_int), value :: data_type
       integer(c_int), value :: layout
       integer(c_int), value :: ownership
       integer(c_int) :: st
     end function c_cmi_tensor_create_flat

     !> @brief Creates a tensor by copying an existing flat buffer.
     function c_cmi_tensor_create_from_copy(out_t, data_ptr, dims, ndims, data_type, layout) &
          bind(C, name="cmi_tensor_create_from_copy") result(st)
       import :: c_ptr, c_int
       type(c_ptr), intent(out) :: out_t
       type(c_ptr), value :: data_ptr
       integer(c_int), intent(in) :: dims(*)
       integer(c_int), value :: ndims
       integer(c_int), value :: data_type
       integer(c_int), value :: layout
       integer(c_int) :: st
     end function c_cmi_tensor_create_from_copy

     !> @brief Destroys a tensor.
     function c_cmi_tensor_destroy(t) bind(C, name="cmi_tensor_destroy") result(st)
       import :: c_ptr, c_int
       type(c_ptr), value :: t
       integer(c_int) :: st
     end function c_cmi_tensor_destroy

     !> @brief Retrieves raw pointer from a tensor.
     function c_cmi_tensor_get_data(t, out_data) bind(C, name="cmi_tensor_get_data") result(st)
       import :: c_ptr, c_int
       type(c_ptr), value :: t
       type(c_ptr), intent(out) :: out_data
       integer(c_int) :: st
     end function c_cmi_tensor_get_data

     !> @brief Retrieves dimension shape array and rank from a tensor.
     function c_cmi_tensor_get_dims(t, out_dims, out_ndims) bind(C, name="cmi_tensor_get_dims") result(st)
       import :: c_ptr, c_int
       type(c_ptr), value :: t
       type(c_ptr), intent(out) :: out_dims
       integer(c_int), intent(out) :: out_ndims
       integer(c_int) :: st
     end function c_cmi_tensor_get_dims

     !> @brief Retrieves memory layout of a tensor.
     function c_cmi_tensor_get_layout(t, out_layout) bind(C, name="cmi_tensor_get_layout") result(st)
       import :: c_ptr, c_int
       type(c_ptr), value :: t
       integer(c_int), intent(out) :: out_layout
       integer(c_int) :: st
     end function c_cmi_tensor_get_layout

     !> @brief Retrieves primitive data type of a tensor.
     function c_cmi_tensor_get_data_type(t, out_dtype) bind(C, name="cmi_tensor_get_data_type") result(st)
       import :: c_ptr, c_int
       type(c_ptr), value :: t
       integer(c_int), intent(out) :: out_dtype
       integer(c_int) :: st
     end function c_cmi_tensor_get_data_type

     !> @brief Retrieves total number of elements in a tensor.
     function c_cmi_tensor_get_numel(t, out_numel) bind(C, name="cmi_tensor_get_numel") result(st)
       import :: c_ptr, c_int, c_size_t
       type(c_ptr), value :: t
       integer(c_size_t), intent(out) :: out_numel
       integer(c_int) :: st
     end function c_cmi_tensor_get_numel

     !> @brief Retrieves ownership policy of a tensor buffer.
     function c_cmi_tensor_get_ownership(t, out_ownership) bind(C, name="cmi_tensor_get_ownership") result(st)
       import :: c_ptr, c_int
       type(c_ptr), value :: t
       integer(c_int), intent(out) :: out_ownership
       integer(c_int) :: st
     end function c_cmi_tensor_get_ownership

     !> @brief Flattens a tensor to the specified layout.
     function c_cmi_tensor_flatten(t, target_layout, out_t) bind(C, name="cmi_tensor_flatten") result(st)
       import :: c_ptr, c_int
       type(c_ptr), value :: t
       integer(c_int), value :: target_layout
       type(c_ptr), intent(out) :: out_t
       integer(c_int) :: st
     end function c_cmi_tensor_flatten

     !> @brief Deep-copies a tensor.
     function c_cmi_tensor_deep_copy(t, out_t) bind(C, name="cmi_tensor_deep_copy") result(st)
       import :: c_ptr, c_int
       type(c_ptr), value :: t
       type(c_ptr), intent(out) :: out_t
       integer(c_int) :: st
     end function c_cmi_tensor_deep_copy

     !> @brief Creates an empty MLCouplingData container.
     function c_cmi_data_create(out_d, data_type) bind(C, name="cmi_data_create") result(st)
       import :: c_ptr, c_int
       type(c_ptr), intent(out) :: out_d
       integer(c_int), value :: data_type
       integer(c_int) :: st
     end function c_cmi_data_create

     !> @brief Destroys an MLCouplingData container.
     function c_cmi_data_destroy(d) bind(C, name="cmi_data_destroy") result(st)
       import :: c_ptr, c_int
       type(c_ptr), value :: d
       integer(c_int) :: st
     end function c_cmi_data_destroy

     !> @brief Appends a tensor to a container.
     function c_cmi_data_add_tensor(d, t) bind(C, name="cmi_data_add_tensor") result(st)
       import :: c_ptr, c_int
       type(c_ptr), value :: d
       type(c_ptr), value :: t
       integer(c_int) :: st
     end function c_cmi_data_add_tensor

     !> @brief Retrieves a tensor by index from a container.
     function c_cmi_data_get_tensor(d, idx, out_t) bind(C, name="cmi_data_get_tensor") result(st)
       import :: c_ptr, c_int
       type(c_ptr), value :: d
       integer(c_int), value :: idx
       type(c_ptr), intent(out) :: out_t
       integer(c_int) :: st
     end function c_cmi_data_get_tensor

     !> @brief Queries the number of tensors in a container.
     function c_cmi_data_size(d, out_size) bind(C, name="cmi_data_size") result(st)
       import :: c_ptr, c_int
       type(c_ptr), value :: d
       integer(c_int), intent(out) :: out_size
       integer(c_int) :: st
     end function c_cmi_data_size

     !> @brief Queries the element data type of a container.
     function c_cmi_data_get_data_type(d, out_dtype) bind(C, name="cmi_data_get_data_type") result(st)
       import :: c_ptr, c_int
       type(c_ptr), value :: d
       integer(c_int), intent(out) :: out_dtype
       integer(c_int) :: st
     end function c_cmi_data_get_data_type

     !> @brief Deep-copies an entire data container.
     function c_cmi_data_deep_copy(d, out_d) bind(C, name="cmi_data_deep_copy") result(st)
       import :: c_ptr, c_int
       type(c_ptr), value :: d
       type(c_ptr), intent(out) :: out_d
       integer(c_int) :: st
     end function c_cmi_data_deep_copy

     !> @brief Creates a default behavior component.
     function c_cmi_behavior_create_default(out_b) bind(C, name="cmi_behavior_create_default") result(st)
       import :: c_ptr, c_int
       type(c_ptr), intent(out) :: out_b
       integer(c_int) :: st
     end function c_cmi_behavior_create_default

     !> @brief Creates a periodic behavior component.
     function c_cmi_behavior_create_periodic(out_b, interval, before, stride, inc) &
          bind(C, name="cmi_behavior_create_periodic") result(st)
       import :: c_ptr, c_int
       type(c_ptr), intent(out) :: out_b
       integer(c_int), value :: interval, before, stride, inc
       integer(c_int) :: st
     end function c_cmi_behavior_create_periodic

     !> @brief Creates a generic behavior driven by callbacks.
     function c_cmi_behavior_create_generic(out_b, infer_fn, delta_fn, send_fn, user_data) &
          bind(C, name="cmi_behavior_create_generic") result(st)
       import :: c_ptr, c_int, c_funptr
       type(c_ptr), intent(out) :: out_b
       type(c_funptr), value :: infer_fn, delta_fn, send_fn
       type(c_ptr), value :: user_data
       integer(c_int) :: st
     end function c_cmi_behavior_create_generic

     !> @brief Instantiates a behavior subclass by registered name.
     function c_cmi_behavior_create_by_name(out_b, name, param_names, param_values, param_types, param_count) &
          bind(C, name="cmi_behavior_create_by_name") result(st)
       import :: c_ptr, c_int, c_char
       type(c_ptr), intent(out) :: out_b
       character(kind=c_char), intent(in) :: name(*)
       type(c_ptr), intent(in) :: param_names(*)
       type(c_ptr), intent(in) :: param_values(*)
       integer(c_int), intent(in) :: param_types(*)
       integer(c_int), value :: param_count
       integer(c_int) :: st
     end function c_cmi_behavior_create_by_name

     !> @brief Queries if inference should occur.
     function c_cmi_behavior_should_perform_inference(b, out_val) &
          bind(C, name="cmi_behavior_should_perform_inference") result(st)
       import :: c_ptr, c_int, c_bool
       type(c_ptr), value :: b
       logical(c_bool), intent(out) :: out_val
       integer(c_int) :: st
     end function c_cmi_behavior_should_perform_inference

     !> @brief Queries simulation timestep delta.
     function c_cmi_behavior_time_step_delta(b, out_val) &
          bind(C, name="cmi_behavior_time_step_delta") result(st)
       import :: c_ptr, c_int
       type(c_ptr), value :: b
       integer(c_int), intent(out) :: out_val
       integer(c_int) :: st
     end function c_cmi_behavior_time_step_delta

     !> @brief Queries if data should be transmitted.
     function c_cmi_behavior_should_send_data(b, out_val) &
          bind(C, name="cmi_behavior_should_send_data") result(st)
       import :: c_ptr, c_int, c_bool
       type(c_ptr), value :: b
       logical(c_bool), intent(out) :: out_val
       integer(c_int) :: st
     end function c_cmi_behavior_should_send_data

     !> @brief Destroys a behavior instance.
     function c_cmi_behavior_destroy(b) bind(C, name="cmi_behavior_destroy") result(st)
       import :: c_ptr, c_int
       type(c_ptr), value :: b
       integer(c_int) :: st
     end function c_cmi_behavior_destroy

     !> @brief Creates MinMax normalization.
     function c_cmi_normalization_create_minmax(out_n, in_t, out_t, in_min, in_max, out_min, out_max) &
          bind(C, name="cmi_normalization_create_minmax") result(st)
       import :: c_ptr, c_int, c_double
       type(c_ptr), intent(out) :: out_n
       integer(c_int), value :: in_t, out_t
       real(c_double), value :: in_min, in_max, out_min, out_max
       integer(c_int) :: st
     end function c_cmi_normalization_create_minmax

     !> @brief Creates a generic normalization driven by callbacks.
     function c_cmi_normalization_create_generic(out_n, in_t, out_t, normalize_fn, denormalize_fn, user_data) &
          bind(C, name="cmi_normalization_create_generic") result(st)
       import :: c_ptr, c_int, c_funptr
       type(c_ptr), intent(out) :: out_n
       integer(c_int), value :: in_t, out_t
       type(c_funptr), value :: normalize_fn, denormalize_fn
       type(c_ptr), value :: user_data
       integer(c_int) :: st
     end function c_cmi_normalization_create_generic

     !> @brief Instantiates a normalization subclass by registered name.
     function c_cmi_normalization_create_by_name(out_n, name, in_t, out_t, param_names, param_values, param_types, param_count) &
          bind(C, name="cmi_normalization_create_by_name") result(st)
       import :: c_ptr, c_int, c_char
       type(c_ptr), intent(out) :: out_n
       character(kind=c_char), intent(in) :: name(*)
       integer(c_int), value :: in_t, out_t
       type(c_ptr), intent(in) :: param_names(*)
       type(c_ptr), intent(in) :: param_values(*)
       integer(c_int), intent(in) :: param_types(*)
       integer(c_int), value :: param_count
       integer(c_int) :: st
     end function c_cmi_normalization_create_by_name

     !> @brief Normalizes input data.
     function c_cmi_normalization_normalize_input(norm, d) &
          bind(C, name="cmi_normalization_normalize_input") result(st)
       import :: c_ptr, c_int
       type(c_ptr), value :: norm, d
       integer(c_int) :: st
     end function c_cmi_normalization_normalize_input

     !> @brief Denormalizes output data.
     function c_cmi_normalization_denormalize_output(norm, d) &
          bind(C, name="cmi_normalization_denormalize_output") result(st)
       import :: c_ptr, c_int
       type(c_ptr), value :: norm, d
       integer(c_int) :: st
     end function c_cmi_normalization_denormalize_output

     !> @brief Destroys normalization component.
     function c_cmi_normalization_destroy(norm) bind(C, name="cmi_normalization_destroy") result(st)
       import :: c_ptr, c_int
       type(c_ptr), value :: norm
       integer(c_int) :: st
     end function c_cmi_normalization_destroy

     !> @brief Creates generic ML library provider.
     function c_cmi_library_create_generic(out_l, in_t, out_t, infer_fn, user_data) &
          bind(C, name="cmi_library_create_generic") result(st)
       import :: c_ptr, c_int, c_funptr
       type(c_ptr), intent(out) :: out_l
       integer(c_int), value :: in_t, out_t
       type(c_funptr), value :: infer_fn
       type(c_ptr), value :: user_data
       integer(c_int) :: st
     end function c_cmi_library_create_generic

     !> @brief Creates a generic ML library provider with training callback.
     function c_cmi_library_create_generic_with_training(out_l, in_t, out_t, infer_fn, train_fn, user_data) &
          bind(C, name="cmi_library_create_generic_with_training") result(st)
       import :: c_ptr, c_int, c_funptr
       type(c_ptr), intent(out) :: out_l
       integer(c_int), value :: in_t, out_t
       type(c_funptr), value :: infer_fn, train_fn
       type(c_ptr), value :: user_data
       integer(c_int) :: st
     end function c_cmi_library_create_generic_with_training

     !> @brief Instantiates a library provider by registered name.
     function c_cmi_library_create_by_name(out_l, name, in_t, out_t, param_names, param_values, param_types, param_count) &
          bind(C, name="cmi_library_create_by_name") result(st)
       import :: c_ptr, c_int, c_char
       type(c_ptr), intent(out) :: out_l
       character(kind=c_char), intent(in) :: name(*)
       integer(c_int), value :: in_t, out_t
       type(c_ptr), intent(in) :: param_names(*)
       type(c_ptr), intent(in) :: param_values(*)
       integer(c_int), intent(in) :: param_types(*)
       integer(c_int), value :: param_count
       integer(c_int) :: st
     end function c_cmi_library_create_by_name

     !> @brief Sets the MPI rank of a library instance.
     function c_cmi_library_set_rank(lib, rank) bind(C, name="cmi_library_set_rank") result(st)
       import :: c_ptr, c_int
       type(c_ptr), value :: lib
       integer(c_int), value :: rank
       integer(c_int) :: st
     end function c_cmi_library_set_rank

     !> @brief Sets the tiered flexible merge strategy.
     function c_cmi_library_set_merge_strategy(lib, strategy) bind(C, name="cmi_library_set_merge_strategy") result(st)
       import :: c_ptr, c_int
       type(c_ptr), value :: lib
       integer(c_int), value :: strategy
       integer(c_int) :: st
     end function c_cmi_library_set_merge_strategy

     !> @brief Runs inference through library provider.
     function c_cmi_library_inference(lib, in_d, out_d) bind(C, name="cmi_library_inference") result(st)
       import :: c_ptr, c_int
       type(c_ptr), value :: lib, in_d, out_d
       integer(c_int) :: st
     end function c_cmi_library_inference

     !> @brief Runs synchronous training through the library provider.
     function c_cmi_library_train(lib, in_d, target_d) bind(C, name="cmi_library_train") result(st)
       import :: c_ptr, c_int
       type(c_ptr), value :: lib, in_d, target_d
       integer(c_int) :: st
     end function c_cmi_library_train

     !> @brief Stages an ordered flexible input.
     function c_cmi_library_flex_ordered_set(lib, d) bind(C, name="cmi_library_flex_ordered_set") result(st)
       import :: c_ptr, c_int
       type(c_ptr), value :: lib, d
       integer(c_int) :: st
     end function c_cmi_library_flex_ordered_set

     !> @brief Stages an ordered flexible training target.
     function c_cmi_library_flex_ordered_set_target(lib, d) bind(C, name="cmi_library_flex_ordered_set_target") result(st)
       import :: c_ptr, c_int
       type(c_ptr), value :: lib, d
       integer(c_int) :: st
     end function c_cmi_library_flex_ordered_set_target

     !> @brief Runs inference on staged ordered inputs.
     function c_cmi_library_flex_ordered_inference(lib, fallback) bind(C, name="cmi_library_flex_ordered_inference") result(st)
       import :: c_ptr, c_int
       type(c_ptr), value :: lib, fallback
       integer(c_int) :: st
     end function c_cmi_library_flex_ordered_inference

     !> @brief Retrieves the ordered flexible result.
     function c_cmi_library_flex_ordered_get(lib, d) bind(C, name="cmi_library_flex_ordered_get") result(st)
       import :: c_ptr, c_int
       type(c_ptr), value :: lib, d
       integer(c_int) :: st
     end function c_cmi_library_flex_ordered_get

     !> @brief Runs ordered flexible training.
     function c_cmi_library_flex_ordered_train(lib, step_id) bind(C, name="cmi_library_flex_ordered_train") result(st)
       import :: c_ptr, c_int, c_int64_t
       type(c_ptr), value :: lib
       integer(c_int64_t), value :: step_id
       integer(c_int) :: st
     end function c_cmi_library_flex_ordered_train

     !> @brief Stages a keyed flexible input.
     function c_cmi_library_flex_keyed_set(lib, key, d) bind(C, name="cmi_library_flex_keyed_set") result(st)
       import :: c_ptr, c_int, c_char
       type(c_ptr), value :: lib, d
       character(kind=c_char), intent(in) :: key(*)
       integer(c_int) :: st
     end function c_cmi_library_flex_keyed_set

     !> @brief Stages a keyed flexible training target.
     function c_cmi_library_flex_keyed_set_target(lib, key, d) bind(C, name="cmi_library_flex_keyed_set_target") result(st)
       import :: c_ptr, c_int, c_char
       type(c_ptr), value :: lib, d
       character(kind=c_char), intent(in) :: key(*)
       integer(c_int) :: st
     end function c_cmi_library_flex_keyed_set_target

     !> @brief Runs inference on staged keyed inputs.
     function c_cmi_library_flex_keyed_inference(lib, in_keys, in_count, out_keys, out_count, fallback) &
          bind(C, name="cmi_library_flex_keyed_inference") result(st)
       import :: c_ptr, c_int
       type(c_ptr), value :: lib, fallback
       type(c_ptr), intent(in) :: in_keys(*), out_keys(*)
       integer(c_int), value :: in_count, out_count
       integer(c_int) :: st
     end function c_cmi_library_flex_keyed_inference

     !> @brief Retrieves a keyed flexible result.
     function c_cmi_library_flex_keyed_get(lib, key, d) bind(C, name="cmi_library_flex_keyed_get") result(st)
       import :: c_ptr, c_int, c_char
       type(c_ptr), value :: lib, d
       character(kind=c_char), intent(in) :: key(*)
       integer(c_int) :: st
     end function c_cmi_library_flex_keyed_get

     !> @brief Runs keyed flexible training.
     function c_cmi_library_flex_keyed_train(lib, step_id, in_keys, in_count, target_keys, target_count) &
          bind(C, name="cmi_library_flex_keyed_train") result(st)
       import :: c_ptr, c_int, c_int64_t
       type(c_ptr), value :: lib
       integer(c_int64_t), value :: step_id
       type(c_ptr), intent(in) :: in_keys(*), target_keys(*)
       integer(c_int), value :: in_count, target_count
       integer(c_int) :: st
     end function c_cmi_library_flex_keyed_train

     !> @brief Destroys library provider.
     function c_cmi_library_destroy(lib) bind(C, name="cmi_library_destroy") result(st)
       import :: c_ptr, c_int
       type(c_ptr), value :: lib
       integer(c_int) :: st
     end function c_cmi_library_destroy

     !> @brief Creates generic application component.
     function c_cmi_application_create_generic(out_a, ci, co, li, lo, cin, cout, norm) &
          bind(C, name="cmi_application_create_generic") result(st)
       import :: c_ptr, c_int
       type(c_ptr), intent(out) :: out_a
       integer(c_int), value :: ci, co, li, lo
       type(c_ptr), value :: cin, cout, norm
       integer(c_int) :: st
     end function c_cmi_application_create_generic

     !> @brief Creates a generic application with preprocessing/postprocessing callbacks.
     function c_cmi_application_create_generic_with_callbacks(out_a, ci, co, li, lo, cin, cout, &
          preprocess_fn, postprocess_fn, norm, user_data) &
          bind(C, name="cmi_application_create_generic_with_callbacks") result(st)
       import :: c_ptr, c_int, c_funptr
       type(c_ptr), intent(out) :: out_a
       integer(c_int), value :: ci, co, li, lo
       type(c_ptr), value :: cin, cout, norm, user_data
       type(c_funptr), value :: preprocess_fn, postprocess_fn
       integer(c_int) :: st
     end function c_cmi_application_create_generic_with_callbacks

     !> @brief Instantiates an application component by registered name.
     function c_cmi_application_create_by_name(out_a, name, ci, co, li, lo, param_names, param_values, param_types, param_count) &
          bind(C, name="cmi_application_create_by_name") result(st)
       import :: c_ptr, c_int, c_char
       type(c_ptr), intent(out) :: out_a
       character(kind=c_char), intent(in) :: name(*)
       integer(c_int), value :: ci, co, li, lo
       type(c_ptr), intent(in) :: param_names(*)
       type(c_ptr), intent(in) :: param_values(*)
       integer(c_int), intent(in) :: param_types(*)
       integer(c_int), value :: param_count
       integer(c_int) :: st
     end function c_cmi_application_create_by_name

     !> @brief Executes application step.
     function c_cmi_application_ml_step(app, lib, beh, out_delta) &
          bind(C, name="cmi_application_ml_step") result(st)
       import :: c_ptr, c_int
       type(c_ptr), value :: app, lib, beh
       integer(c_int), intent(out) :: out_delta
       integer(c_int) :: st
     end function c_cmi_application_ml_step

     !> @brief Destroys application component.
     function c_cmi_application_destroy(app) bind(C, name="cmi_application_destroy") result(st)
       import :: c_ptr, c_int
       type(c_ptr), value :: app
       integer(c_int) :: st
     end function c_cmi_application_destroy

     !> @brief Creates coupling orchestrator from components.
     function c_cmi_coupling_create(out_c, ci, co, li, lo, lib, app, beh) &
          bind(C, name="cmi_coupling_create") result(st)
       import :: c_ptr, c_int
       type(c_ptr), intent(out) :: out_c
       integer(c_int), value :: ci, co, li, lo
       type(c_ptr), value :: lib, app, beh
       integer(c_int) :: st
     end function c_cmi_coupling_create

     !> @brief Creates coupling orchestrator from TOML configuration file.
     function c_cmi_coupling_create_from_config(out_c, path, ci, co, li, lo) &
          bind(C, name="cmi_coupling_create_from_config") result(st)
       import :: c_ptr, c_int, c_char
       type(c_ptr), intent(out) :: out_c
       character(kind=c_char), intent(in) :: path(*)
       integer(c_int), value :: ci, co, li, lo
       integer(c_int) :: st
     end function c_cmi_coupling_create_from_config

     !> @brief Creates coupling from TOML configuration bound to caller buffers.
     function c_cmi_coupling_create_from_config_with_data(out_c, path, ci, co, li, lo, cin, cout) &
          bind(C, name="cmi_coupling_create_from_config_with_data") result(st)
       import :: c_ptr, c_int, c_char
       type(c_ptr), intent(out) :: out_c
       character(kind=c_char), intent(in) :: path(*)
       integer(c_int), value :: ci, co, li, lo
       type(c_ptr), value :: cin, cout
       integer(c_int) :: st
     end function c_cmi_coupling_create_from_config_with_data

     !> @brief Creates coupling from an inline TOML string bound to caller buffers.
     function c_cmi_coupling_create_from_config_string_with_data(out_c, config_str, ci, co, li, lo, cin, cout) &
          bind(C, name="cmi_coupling_create_from_config_string_with_data") result(st)
       import :: c_ptr, c_int, c_char
       type(c_ptr), intent(out) :: out_c
       character(kind=c_char), intent(in) :: config_str(*)
       integer(c_int), value :: ci, co, li, lo
       type(c_ptr), value :: cin, cout
       integer(c_int) :: st
     end function c_cmi_coupling_create_from_config_string_with_data

     !> @brief Creates a config-file-bound coupling with runtime overrides.
     function c_cmi_coupling_create_from_config_with_data_and_overrides(out_c, path, ci, co, li, lo, cin, cout, ov) &
          bind(C, name="cmi_coupling_create_from_config_with_data_and_overrides") result(st)
       import :: c_ptr, c_int, c_char
       type(c_ptr), intent(out) :: out_c
       character(kind=c_char), intent(in) :: path(*)
       integer(c_int), value :: ci, co, li, lo
       type(c_ptr), value :: cin, cout, ov
       integer(c_int) :: st
     end function c_cmi_coupling_create_from_config_with_data_and_overrides

     !> @brief Creates a config-string-bound coupling with runtime overrides.
     function c_cmi_coupling_create_from_config_string_overrides(out_c, config_str, ci, co, li, lo, &
          cin, cout, ov) &
          bind(C, name="cmi_coupling_create_from_config_string_with_data_and_overrides") result(st)
       import :: c_ptr, c_int, c_char
       type(c_ptr), intent(out) :: out_c
       character(kind=c_char), intent(in) :: config_str(*)
       integer(c_int), value :: ci, co, li, lo
       type(c_ptr), value :: cin, cout, ov
       integer(c_int) :: st
     end function c_cmi_coupling_create_from_config_string_overrides

     !> @brief Creates an empty configuration override set.
     function c_cmi_config_overrides_create(out_o) bind(C, name="cmi_config_overrides_create") result(st)
       import :: c_ptr, c_int
       type(c_ptr), intent(out) :: out_o
       integer(c_int) :: st
     end function c_cmi_config_overrides_create

     !> @brief Destroys a configuration override set.
     function c_cmi_config_overrides_destroy(o) bind(C, name="cmi_config_overrides_destroy") result(st)
       import :: c_ptr, c_int
       type(c_ptr), value :: o
       integer(c_int) :: st
     end function c_cmi_config_overrides_destroy

     !> @brief Sets a section-scoped override value.
     function c_cmi_config_overrides_set(o, section, key, value_type, value) &
          bind(C, name="cmi_config_overrides_set") result(st)
       import :: c_ptr, c_int, c_char
       type(c_ptr), value :: o, value
       character(kind=c_char), intent(in) :: section(*), key(*)
       integer(c_int), value :: value_type
       integer(c_int) :: st
     end function c_cmi_config_overrides_set

     !> @brief Sets a dotted-key override value.
     function c_cmi_config_overrides_set_dotted(o, dotted_key, value_type, value) &
          bind(C, name="cmi_config_overrides_set_dotted") result(st)
       import :: c_ptr, c_int, c_char
       type(c_ptr), value :: o, value
       character(kind=c_char), intent(in) :: dotted_key(*)
       integer(c_int), value :: value_type
       integer(c_int) :: st
     end function c_cmi_config_overrides_set_dotted

     !> @brief Sets a section-scoped string-array override.
     function c_cmi_config_overrides_set_string_array(o, section, key, values, count) &
          bind(C, name="cmi_config_overrides_set_string_array") result(st)
       import :: c_ptr, c_int, c_char
       type(c_ptr), value :: o
       character(kind=c_char), intent(in) :: section(*), key(*)
       type(c_ptr), intent(in) :: values(*)
       integer(c_int), value :: count
       integer(c_int) :: st
     end function c_cmi_config_overrides_set_string_array

     !> @brief Sets a dotted-key string-array override.
     function c_cmi_config_overrides_set_dotted_string_array(o, dotted_key, values, count) &
          bind(C, name="cmi_config_overrides_set_dotted_string_array") result(st)
       import :: c_ptr, c_int, c_char
       type(c_ptr), value :: o
       character(kind=c_char), intent(in) :: dotted_key(*)
       type(c_ptr), intent(in) :: values(*)
       integer(c_int), value :: count
       integer(c_int) :: st
     end function c_cmi_config_overrides_set_dotted_string_array

     !> @brief Advances coupled pipeline one step.
     function c_cmi_coupling_step(c, out_delta) bind(C, name="cmi_coupling_step") result(st)
       import :: c_ptr, c_int
       type(c_ptr), value :: c
       integer(c_int), intent(out) :: out_delta
       integer(c_int) :: st
     end function c_cmi_coupling_step

     !> @brief Sets the tiered flexible merge strategy on a coupling.
     function c_cmi_coupling_set_merge_strategy(c, strategy) bind(C, name="cmi_coupling_set_merge_strategy") result(st)
       import :: c_ptr, c_int
       type(c_ptr), value :: c
       integer(c_int), value :: strategy
       integer(c_int) :: st
     end function c_cmi_coupling_set_merge_strategy

     !> @brief Registers a training metric field for tracking.
     function c_cmi_coupling_track(c, field) bind(C, name="cmi_coupling_track") result(st)
       import :: c_ptr, c_int, c_char
       type(c_ptr), value :: c
       character(kind=c_char), intent(in) :: field(*)
       integer(c_int) :: st
     end function c_cmi_coupling_track

     !> @brief Retrieves the current value of a tracked metric.
     function c_cmi_coupling_get_current(c, field, out_value) bind(C, name="cmi_coupling_get_current") result(st)
       import :: c_ptr, c_int, c_char, c_double
       type(c_ptr), value :: c
       character(kind=c_char), intent(in) :: field(*)
       real(c_double), intent(out) :: out_value
       integer(c_int) :: st
     end function c_cmi_coupling_get_current

     !> @brief Retrieves the recorded history of a tracked metric.
     function c_cmi_coupling_get_history(c, field, out_values, capacity, out_count) &
          bind(C, name="cmi_coupling_get_history") result(st)
       import :: c_ptr, c_int, c_char, c_double
       type(c_ptr), value :: c
       character(kind=c_char), intent(in) :: field(*)
       real(c_double), intent(out) :: out_values(*)
       integer(c_int), value :: capacity
       integer(c_int), intent(out) :: out_count
       integer(c_int) :: st
     end function c_cmi_coupling_get_history

     !> @brief Logs externally computed training metrics.
     function c_cmi_coupling_log_training_metrics(c, step_id, fields, values, count) &
          bind(C, name="cmi_coupling_log_training_metrics") result(st)
       import :: c_ptr, c_int, c_int64_t, c_double
       type(c_ptr), value :: c
       integer(c_int64_t), value :: step_id
       type(c_ptr), intent(in) :: fields(*)
       real(c_double), intent(in) :: values(*)
       integer(c_int), value :: count
       integer(c_int) :: st
     end function c_cmi_coupling_log_training_metrics

     !> @brief Executes a training step for in-situ model updates.
     function c_cmi_coupling_train_step(c, step_id) bind(C, name="cmi_coupling_train_step") result(st)
       import :: c_ptr, c_int, c_int64_t
       type(c_ptr), value :: c
       integer(c_int64_t), value :: step_id
       integer(c_int) :: st
     end function c_cmi_coupling_train_step

     !> @brief Destroys coupling orchestrator.
     function c_cmi_coupling_destroy(c) bind(C, name="cmi_coupling_destroy") result(st)
       import :: c_ptr, c_int
       type(c_ptr), value :: c
       integer(c_int) :: st
     end function c_cmi_coupling_destroy

     !> @brief Legacy factory function for library creation (deprecated).
     function c_create_library(name, in_selection, out_selection, param_names, params, param_count) &
          bind(C, name="create_library") result(res)
       import :: c_char, c_int, c_ptr
       character(kind=c_char), intent(in) :: name(*)
       integer(c_int), value :: in_selection, out_selection, param_count
       type(c_ptr), intent(in) :: param_names(*)
       type(c_ptr), intent(in) :: params(*)
       type(c_ptr) :: res
     end function c_create_library

     !> @brief Legacy function to destroy a library instance (deprecated).
     subroutine c_destroy_library(handle) bind(C, name="destroy_library")
       import :: c_ptr
       type(c_ptr), value :: handle
     end subroutine c_destroy_library
   end interface

contains

  ! =========================================================================
  ! Fortran Wrapper Subroutines (Column-Major / Zero-Copy by Default)
  ! =========================================================================

  !> @brief Wraps a 1D single-precision Fortran array into an MLCouplingTensor without copying.
  !> @param[out] t Created tensor handle.
  !> @param[in]  array Target 1D float array.
  !> @param[out] ierr Status code (0 on success).
  subroutine cmi_tensor_wrap_1d_float(t, array, ierr)
    type(cmi_tensor), intent(out) :: t
    real(c_float), target, intent(in) :: array(:)
    integer, intent(out) :: ierr
    integer(c_int) :: dims(1)

    dims(1) = int(size(array), kind=c_int)
    ierr = c_cmi_tensor_create_flat(t%ptr, c_loc(array), dims, 1_c_int, &
                                   CMI_DTYPE_FLOAT, CMI_LAYOUT_FORTRAN_CONTIGUOUS, &
                                   CMI_OWNERSHIP_EXTERNAL)
  end subroutine cmi_tensor_wrap_1d_float

  !> @brief Wraps a 2D single-precision Fortran array into an MLCouplingTensor without copying.
  !> @param[out] t Created tensor handle.
  !> @param[in]  array Target 2D float array.
  !> @param[out] ierr Status code (0 on success).
  subroutine cmi_tensor_wrap_2d_float(t, array, ierr)
    type(cmi_tensor), intent(out) :: t
    real(c_float), target, intent(in) :: array(:,:)
    integer, intent(out) :: ierr
    integer(c_int) :: dims(2)

    dims(1) = int(size(array, 1), kind=c_int)
    dims(2) = int(size(array, 2), kind=c_int)
    ierr = c_cmi_tensor_create_flat(t%ptr, c_loc(array), dims, 2_c_int, &
                                   CMI_DTYPE_FLOAT, CMI_LAYOUT_FORTRAN_CONTIGUOUS, &
                                   CMI_OWNERSHIP_EXTERNAL)
  end subroutine cmi_tensor_wrap_2d_float

  !> @brief Wraps a 3D single-precision Fortran array into an MLCouplingTensor without copying.
  !> @param[out] t Created tensor handle.
  !> @param[in]  array Target 3D float array.
  !> @param[out] ierr Status code (0 on success).
  subroutine cmi_tensor_wrap_3d_float(t, array, ierr)
    type(cmi_tensor), intent(out) :: t
    real(c_float), target, intent(in) :: array(:,:,:)
    integer, intent(out) :: ierr
    integer(c_int) :: dims(3)

    dims(1) = int(size(array, 1), kind=c_int)
    dims(2) = int(size(array, 2), kind=c_int)
    dims(3) = int(size(array, 3), kind=c_int)
    ierr = c_cmi_tensor_create_flat(t%ptr, c_loc(array), dims, 3_c_int, &
                                   CMI_DTYPE_FLOAT, CMI_LAYOUT_FORTRAN_CONTIGUOUS, &
                                   CMI_OWNERSHIP_EXTERNAL)
  end subroutine cmi_tensor_wrap_3d_float

  !> @brief Wraps a 1D double-precision Fortran array into an MLCouplingTensor without copying.
  !> @param[out] t Created tensor handle.
  !> @param[in]  array Target 1D double array.
  !> @param[out] ierr Status code (0 on success).
  subroutine cmi_tensor_wrap_1d_double(t, array, ierr)
    type(cmi_tensor), intent(out) :: t
    real(c_double), target, intent(in) :: array(:)
    integer, intent(out) :: ierr
    integer(c_int) :: dims(1)

    dims(1) = int(size(array), kind=c_int)
    ierr = c_cmi_tensor_create_flat(t%ptr, c_loc(array), dims, 1_c_int, &
                                   CMI_DTYPE_DOUBLE, CMI_LAYOUT_FORTRAN_CONTIGUOUS, &
                                   CMI_OWNERSHIP_EXTERNAL)
  end subroutine cmi_tensor_wrap_1d_double

  !> @brief Wraps a 2D double-precision Fortran array into an MLCouplingTensor without copying.
  !> @param[out] t Created tensor handle.
  !> @param[in]  array Target 2D double array.
  !> @param[out] ierr Status code (0 on success).
  subroutine cmi_tensor_wrap_2d_double(t, array, ierr)
    type(cmi_tensor), intent(out) :: t
    real(c_double), target, intent(in) :: array(:,:)
    integer, intent(out) :: ierr
    integer(c_int) :: dims(2)

    dims(1) = int(size(array, 1), kind=c_int)
    dims(2) = int(size(array, 2), kind=c_int)
    ierr = c_cmi_tensor_create_flat(t%ptr, c_loc(array), dims, 2_c_int, &
                                   CMI_DTYPE_DOUBLE, CMI_LAYOUT_FORTRAN_CONTIGUOUS, &
                                   CMI_OWNERSHIP_EXTERNAL)
  end subroutine cmi_tensor_wrap_2d_double

  !> @brief Wraps a 3D double-precision Fortran array into an MLCouplingTensor without copying.
  !> @param[out] t Created tensor handle.
  !> @param[in]  array Target 3D double array.
  !> @param[out] ierr Status code (0 on success).
  subroutine cmi_tensor_wrap_3d_double(t, array, ierr)
    type(cmi_tensor), intent(out) :: t
    real(c_double), target, intent(in) :: array(:,:,:)
    integer, intent(out) :: ierr
    integer(c_int) :: dims(3)

    dims(1) = int(size(array, 1), kind=c_int)
    dims(2) = int(size(array, 2), kind=c_int)
    dims(3) = int(size(array, 3), kind=c_int)
    ierr = c_cmi_tensor_create_flat(t%ptr, c_loc(array), dims, 3_c_int, &
                                   CMI_DTYPE_DOUBLE, CMI_LAYOUT_FORTRAN_CONTIGUOUS, &
                                   CMI_OWNERSHIP_EXTERNAL)
  end subroutine cmi_tensor_wrap_3d_double

  !> @brief Wraps a 4D single-precision Fortran array into an MLCouplingTensor without copying.
  !> @param[out] t Created tensor handle.
  !> @param[in]  array Target 4D float array.
  !> @param[out] ierr Status code (0 on success).
  subroutine cmi_tensor_wrap_4d_float(t, array, ierr)
    type(cmi_tensor), intent(out) :: t
    real(c_float), target, intent(in) :: array(:,:,:,:)
    integer, intent(out) :: ierr
    integer(c_int) :: dims(4)

    dims(1) = int(size(array, 1), kind=c_int)
    dims(2) = int(size(array, 2), kind=c_int)
    dims(3) = int(size(array, 3), kind=c_int)
    dims(4) = int(size(array, 4), kind=c_int)
    ierr = c_cmi_tensor_create_flat(t%ptr, c_loc(array), dims, 4_c_int, &
                                   CMI_DTYPE_FLOAT, CMI_LAYOUT_FORTRAN_CONTIGUOUS, &
                                   CMI_OWNERSHIP_EXTERNAL)
  end subroutine cmi_tensor_wrap_4d_float

  !> @brief Wraps a 5D single-precision Fortran array into an MLCouplingTensor without copying.
  subroutine cmi_tensor_wrap_5d_float(t, array, ierr)
    type(cmi_tensor), intent(out) :: t
    real(c_float), target, intent(in) :: array(:,:,:,:,:)
    integer, intent(out) :: ierr
    integer(c_int) :: dims(5)

    dims(1) = int(size(array, 1), kind=c_int)
    dims(2) = int(size(array, 2), kind=c_int)
    dims(3) = int(size(array, 3), kind=c_int)
    dims(4) = int(size(array, 4), kind=c_int)
    dims(5) = int(size(array, 5), kind=c_int)
    ierr = c_cmi_tensor_create_flat(t%ptr, c_loc(array), dims, 5_c_int, &
                                   CMI_DTYPE_FLOAT, CMI_LAYOUT_FORTRAN_CONTIGUOUS, &
                                   CMI_OWNERSHIP_EXTERNAL)
  end subroutine cmi_tensor_wrap_5d_float

  !> @brief Wraps a 6D single-precision Fortran array into an MLCouplingTensor without copying.
  subroutine cmi_tensor_wrap_6d_float(t, array, ierr)
    type(cmi_tensor), intent(out) :: t
    real(c_float), target, intent(in) :: array(:,:,:,:,:,:)
    integer, intent(out) :: ierr
    integer(c_int) :: dims(6)

    dims(1) = int(size(array, 1), kind=c_int)
    dims(2) = int(size(array, 2), kind=c_int)
    dims(3) = int(size(array, 3), kind=c_int)
    dims(4) = int(size(array, 4), kind=c_int)
    dims(5) = int(size(array, 5), kind=c_int)
    dims(6) = int(size(array, 6), kind=c_int)
    ierr = c_cmi_tensor_create_flat(t%ptr, c_loc(array), dims, 6_c_int, &
                                   CMI_DTYPE_FLOAT, CMI_LAYOUT_FORTRAN_CONTIGUOUS, &
                                   CMI_OWNERSHIP_EXTERNAL)
  end subroutine cmi_tensor_wrap_6d_float

  !> @brief Wraps a 4D double-precision Fortran array into an MLCouplingTensor without copying.
  subroutine cmi_tensor_wrap_4d_double(t, array, ierr)
    type(cmi_tensor), intent(out) :: t
    real(c_double), target, intent(in) :: array(:,:,:,:)
    integer, intent(out) :: ierr
    integer(c_int) :: dims(4)

    dims(1) = int(size(array, 1), kind=c_int)
    dims(2) = int(size(array, 2), kind=c_int)
    dims(3) = int(size(array, 3), kind=c_int)
    dims(4) = int(size(array, 4), kind=c_int)
    ierr = c_cmi_tensor_create_flat(t%ptr, c_loc(array), dims, 4_c_int, &
                                   CMI_DTYPE_DOUBLE, CMI_LAYOUT_FORTRAN_CONTIGUOUS, &
                                   CMI_OWNERSHIP_EXTERNAL)
  end subroutine cmi_tensor_wrap_4d_double

  !> @brief Wraps a 5D double-precision Fortran array into an MLCouplingTensor without copying.
  subroutine cmi_tensor_wrap_5d_double(t, array, ierr)
    type(cmi_tensor), intent(out) :: t
    real(c_double), target, intent(in) :: array(:,:,:,:,:)
    integer, intent(out) :: ierr
    integer(c_int) :: dims(5)

    dims(1) = int(size(array, 1), kind=c_int)
    dims(2) = int(size(array, 2), kind=c_int)
    dims(3) = int(size(array, 3), kind=c_int)
    dims(4) = int(size(array, 4), kind=c_int)
    dims(5) = int(size(array, 5), kind=c_int)
    ierr = c_cmi_tensor_create_flat(t%ptr, c_loc(array), dims, 5_c_int, &
                                   CMI_DTYPE_DOUBLE, CMI_LAYOUT_FORTRAN_CONTIGUOUS, &
                                   CMI_OWNERSHIP_EXTERNAL)
  end subroutine cmi_tensor_wrap_5d_double

  !> @brief Wraps a 6D double-precision Fortran array into an MLCouplingTensor without copying.
  subroutine cmi_tensor_wrap_6d_double(t, array, ierr)
    type(cmi_tensor), intent(out) :: t
    real(c_double), target, intent(in) :: array(:,:,:,:,:,:)
    integer, intent(out) :: ierr
    integer(c_int) :: dims(6)

    dims(1) = int(size(array, 1), kind=c_int)
    dims(2) = int(size(array, 2), kind=c_int)
    dims(3) = int(size(array, 3), kind=c_int)
    dims(4) = int(size(array, 4), kind=c_int)
    dims(5) = int(size(array, 5), kind=c_int)
    dims(6) = int(size(array, 6), kind=c_int)
    ierr = c_cmi_tensor_create_flat(t%ptr, c_loc(array), dims, 6_c_int, &
                                   CMI_DTYPE_DOUBLE, CMI_LAYOUT_FORTRAN_CONTIGUOUS, &
                                   CMI_OWNERSHIP_EXTERNAL)
  end subroutine cmi_tensor_wrap_6d_double

  !> @brief Builds a C-compatible null-terminated string array for name-based factories.
  !>
  !> Allocates @p c_buffers holding the null-terminated characters and @p c_ptrs
  !> holding one pointer per string. Keep both arrays alive until after the C
  !> factory call, since the pointers reference @p c_buffers storage.
  !> @param[in]  strings Fortran strings to marshal.
  !> @param[out] c_ptrs Array of C pointers to the marshalled strings.
  !> @param[out] c_buffers Backing character storage (must outlive @p c_ptrs).
  !> @param[out] ierr Status code (0 on success).
  subroutine cmi_c_string_array(strings, c_ptrs, c_buffers, ierr)
    character(len=*), intent(in) :: strings(:)
    type(c_ptr), allocatable, intent(out) :: c_ptrs(:)
    character(kind=c_char), allocatable, target, intent(out) :: c_buffers(:,:)
    integer, intent(out) :: ierr
    integer :: n, i, j, max_len

    n = size(strings)
    ierr = CMI_SUCCESS
    allocate(c_ptrs(n))
    if (n == 0) then
       allocate(c_buffers(1, 0))
       return
    end if

    max_len = 1
    do i = 1, n
       max_len = max(max_len, len_trim(strings(i)) + 1)
    end do

    allocate(c_buffers(max_len, n))
    c_buffers = c_null_char
    do i = 1, n
       do j = 1, len_trim(strings(i))
          c_buffers(j, i) = strings(i)(j:j)
       end do
       c_ptrs(i) = c_loc(c_buffers(1, i))
    end do
  end subroutine cmi_c_string_array

  !> @brief Retrieves the last thread-local error message as a Fortran character string.
  !> @param[out] msg Target string variable to receive the error message.
  subroutine cmi_get_error(msg)
    character(len=*), intent(out) :: msg
    type(c_ptr) :: c_msg
    character(kind=c_char), pointer :: f_str(:)
    integer :: i

    msg = ""
    c_msg = c_cmi_get_last_error()
    if (c_associated(c_msg)) then
       call c_f_pointer(c_msg, f_str, [len(msg)])
       do i = 1, len(msg)
          if (f_str(i) == c_null_char) exit
          msg(i:i) = f_str(i)
       end do
    end if
  end subroutine cmi_get_error

end module cmi

!> @}
