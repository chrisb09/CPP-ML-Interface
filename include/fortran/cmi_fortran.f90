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
  !> @}

  !> @brief Overloaded generic interface to wrap Fortran 1D, 2D, or 3D arrays into tensors without copying.
  interface cmi_tensor_wrap
     module procedure cmi_tensor_wrap_1d_float
     module procedure cmi_tensor_wrap_2d_float
     module procedure cmi_tensor_wrap_3d_float
     module procedure cmi_tensor_wrap_1d_double
     module procedure cmi_tensor_wrap_2d_double
     module procedure cmi_tensor_wrap_3d_double
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

     !> @brief Runs inference through library provider.
     function c_cmi_library_inference(lib, in_d, out_d) bind(C, name="cmi_library_inference") result(st)
       import :: c_ptr, c_int
       type(c_ptr), value :: lib, in_d, out_d
       integer(c_int) :: st
     end function c_cmi_library_inference

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

     !> @brief Advances coupled pipeline one step.
     function c_cmi_coupling_step(c, out_delta) bind(C, name="cmi_coupling_step") result(st)
       import :: c_ptr, c_int
       type(c_ptr), value :: c
       integer(c_int), intent(out) :: out_delta
       integer(c_int) :: st
     end function c_cmi_coupling_step

     !> @brief Destroys coupling orchestrator.
     function c_cmi_coupling_destroy(c) bind(C, name="cmi_coupling_destroy") result(st)
       import :: c_ptr, c_int
       type(c_ptr), value :: c
       integer(c_int) :: st
     end function c_cmi_coupling_destroy
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
