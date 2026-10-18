! =============================================================================
! Test Fortran API (test_fortran_api.f90)
! Tests the cmi Fortran module and column-major array support.
! =============================================================================

module test_state_mod
  use, intrinsic :: iso_c_binding
  implicit none

  !> Context passed through user_data for application callback tests.
  type, bind(C) :: app_ctx_t
     type(c_ptr) :: out_ptr
     integer(c_int) :: pre_calls
     integer(c_int) :: post_calls
  end type app_ctx_t

  !> Last merged input observed by the inference callback.
  real(c_float), save, target :: g_seen(32)
  integer(c_int), save :: g_seen_n = 0
end module test_state_mod

program test_fortran_api
  use, intrinsic :: iso_c_binding
  use cmi
  use test_state_mod
  implicit none

  interface
     subroutine fortran_inference_callback(in_d, out_d, user_data) bind(C)
       import :: c_ptr
       type(c_ptr), value :: in_d, out_d, user_data
     end subroutine fortran_inference_callback

     subroutine fortran_train_callback(in_d, target_d, user_data) bind(C)
       import :: c_ptr
       type(c_ptr), value :: in_d, target_d, user_data
     end subroutine fortran_train_callback

     subroutine fortran_preprocess_callback(in_d, out_d, user_data) bind(C)
       import :: c_ptr
       type(c_ptr), value :: in_d, out_d, user_data
     end subroutine fortran_preprocess_callback

     subroutine fortran_postprocess_callback(in_d, out_d, user_data) bind(C)
       import :: c_ptr
       type(c_ptr), value :: in_d, out_d, user_data
     end subroutine fortran_postprocess_callback
  end interface

  call test_tensor_lifecycle_and_column_major()
  call test_tensor_copy_and_ownership()
  call test_tensor_wrap_4d()
  call test_data_lifecycle()
  call test_data_metadata_and_deep_copy()
  call test_behavior()
  call test_behavior_by_name()
  call test_normalization()
  call test_normalization_by_name()
  call test_library_and_inference()
  call test_library_metadata()
  call test_flex_get_and_train()
  call test_flex_fortran_layout_merge()
  call test_application_callbacks()
  call test_coupling_pipeline()
  call test_config_with_data()
  call test_config_string_and_overrides()
  call test_coupling_tracking()

  print *, ""
  print *, "All Fortran API tests PASSED."

contains

  subroutine test_tensor_lifecycle_and_column_major()
    real(c_float), target :: arr2d(3, 4)
    type(cmi_tensor) :: t
    integer :: ierr, layout, dtype, ndims
    integer(c_size_t) :: numel
    type(c_ptr) :: dims_ptr, data_ptr
    integer(c_int), pointer :: f_dims(:)
    integer :: i, j, count

    ! Initialize with distinct values
    count = 1
    do j = 1, 4
       do i = 1, 3
          arr2d(i, j) = real(count, kind=c_float)
          count = count + 1
       end do
    end do

    ! Wrap 2D Fortran array (column-major)
    call cmi_tensor_wrap(t, arr2d, ierr)
    if (ierr /= CMI_SUCCESS) error stop "Failed to wrap 2d tensor"

    ! Check layout is FortranContiguous
    ierr = c_cmi_tensor_get_layout(t%ptr, layout)
    if (layout /= CMI_LAYOUT_FORTRAN_CONTIGUOUS) error stop "Expected FortranContiguous layout"

    ! Check data type is float
    ierr = c_cmi_tensor_get_data_type(t%ptr, dtype)
    if (dtype /= CMI_DTYPE_FLOAT) error stop "Expected float dtype"

    ! Check numel is 12
    ierr = c_cmi_tensor_get_numel(t%ptr, numel)
    if (numel /= 12_c_size_t) error stop "Expected numel 12"

    ! Check dims are [3, 4]
    ierr = c_cmi_tensor_get_dims(t%ptr, dims_ptr, ndims)
    if (ndims /= 2) error stop "Expected 2 dimensions"
    call c_f_pointer(dims_ptr, f_dims, [2])
    if (f_dims(1) /= 3 .or. f_dims(2) /= 4) error stop "Expected dimensions [3, 4]"

    ! Check zero-copy: raw pointer equals c_loc(arr2d)
    ierr = c_cmi_tensor_get_data(t%ptr, data_ptr)
    if (.not. c_associated(data_ptr, c_loc(arr2d))) error stop "Expected zero-copy memory pointer match"

    ! Destroy tensor handle
    ierr = c_cmi_tensor_destroy(t%ptr)
    if (ierr /= CMI_SUCCESS) error stop "Failed to destroy tensor"

    print *, "[PASS] test_tensor_lifecycle_and_column_major"
  end subroutine test_tensor_lifecycle_and_column_major

  subroutine test_tensor_copy_and_ownership()
    real(c_float), target :: src(3)
    type(cmi_tensor) :: t
    integer :: ierr, ownership
    type(c_ptr) :: data_ptr

    src = [4.0_c_float, 5.0_c_float, 6.0_c_float]

    ierr = c_cmi_tensor_create_from_copy(t%ptr, c_loc(src), [3_c_int], 1_c_int, &
                                         CMI_DTYPE_FLOAT, CMI_LAYOUT_CONTIGUOUS)
    if (ierr /= CMI_SUCCESS) error stop "Failed to create tensor from copy"

    ierr = c_cmi_tensor_get_ownership(t%ptr, ownership)
    if (ownership /= CMI_OWNERSHIP_OWNED) error stop "Expected owned tensor"

    ierr = c_cmi_tensor_get_data(t%ptr, data_ptr)
    if (c_associated(data_ptr, c_loc(src))) error stop "create_from_copy must not alias source"

    ierr = c_cmi_tensor_destroy(t%ptr)
    print *, "[PASS] test_tensor_copy_and_ownership"
  end subroutine test_tensor_copy_and_ownership

  subroutine test_tensor_wrap_4d()
    real(c_float), target :: arr4(2, 3, 2, 2)
    type(cmi_tensor) :: t
    integer :: ierr, ndims, layout
    type(c_ptr) :: dims_ptr
    integer(c_int), pointer :: f_dims(:)

    arr4 = 1.0_c_float
    call cmi_tensor_wrap(t, arr4, ierr)
    if (ierr /= CMI_SUCCESS) error stop "Failed to wrap 4d tensor"

    ierr = c_cmi_tensor_get_dims(t%ptr, dims_ptr, ndims)
    if (ndims /= 4) error stop "Expected 4 dimensions"
    call c_f_pointer(dims_ptr, f_dims, [4])
    if (f_dims(1) /= 2 .or. f_dims(2) /= 3 .or. f_dims(3) /= 2 .or. f_dims(4) /= 2) then
       error stop "Unexpected 4d dimensions"
    end if

    ierr = c_cmi_tensor_get_layout(t%ptr, layout)
    if (layout /= CMI_LAYOUT_FORTRAN_CONTIGUOUS) error stop "Expected Fortran layout for 4d tensor"

    ierr = c_cmi_tensor_destroy(t%ptr)
    print *, "[PASS] test_tensor_wrap_4d"
  end subroutine test_tensor_wrap_4d

  subroutine test_data_lifecycle()
    real(c_float), target :: a1(2), a2(3)
    type(cmi_tensor) :: t1, t2
    type(cmi_data) :: d
    integer :: ierr, sz

    a1 = [1.0_c_float, 2.0_c_float]
    a2 = [10.0_c_float, 20.0_c_float, 30.0_c_float]

    call cmi_tensor_wrap(t1, a1, ierr)
    call cmi_tensor_wrap(t2, a2, ierr)

    ierr = c_cmi_data_create(d%ptr, CMI_DTYPE_FLOAT)
    if (ierr /= CMI_SUCCESS) error stop "Failed to create data"

    ierr = c_cmi_data_add_tensor(d%ptr, t1%ptr)
    ierr = c_cmi_data_add_tensor(d%ptr, t2%ptr)

    ierr = c_cmi_data_size(d%ptr, sz)
    if (sz /= 2) error stop "Expected data size 2"

    ierr = c_cmi_tensor_destroy(t1%ptr)
    ierr = c_cmi_tensor_destroy(t2%ptr)
    ierr = c_cmi_data_destroy(d%ptr)

    print *, "[PASS] test_data_lifecycle"
  end subroutine test_data_lifecycle

  subroutine test_data_metadata_and_deep_copy()
    real(c_float), target :: a(3)
    type(cmi_tensor) :: t, tcopy
    type(cmi_data) :: d, dcopy
    integer :: ierr, dtype, sz
    type(c_ptr) :: p_orig, p_copy

    a = [1.0_c_float, 2.0_c_float, 3.0_c_float]
    call cmi_tensor_wrap(t, a, ierr)

    ierr = c_cmi_data_create(d%ptr, CMI_DTYPE_FLOAT)
    ierr = c_cmi_data_add_tensor(d%ptr, t%ptr)

    ierr = c_cmi_data_get_data_type(d%ptr, dtype)
    if (dtype /= CMI_DTYPE_FLOAT) error stop "Unexpected data container dtype"

    ierr = c_cmi_data_deep_copy(d%ptr, dcopy%ptr)
    if (ierr /= CMI_SUCCESS) error stop "Failed to deep-copy data container"

    ierr = c_cmi_data_size(dcopy%ptr, sz)
    if (sz /= 1) error stop "Unexpected deep-copy data size"

    ierr = c_cmi_data_get_tensor(dcopy%ptr, 0_c_int, tcopy%ptr)
    ierr = c_cmi_tensor_get_data(tcopy%ptr, p_copy)
    ierr = c_cmi_tensor_get_data(t%ptr, p_orig)
    if (c_associated(p_copy, p_orig)) error stop "Deep copy must not share tensor memory"

    ierr = c_cmi_tensor_destroy(tcopy%ptr)
    ierr = c_cmi_data_destroy(dcopy%ptr)
    ierr = c_cmi_tensor_destroy(t%ptr)
    ierr = c_cmi_data_destroy(d%ptr)

    print *, "[PASS] test_data_metadata_and_deep_copy"
  end subroutine test_data_metadata_and_deep_copy

  subroutine test_behavior()
    type(cmi_behavior) :: b
    integer :: ierr, delta
    logical(c_bool) :: infer, send_data

    ierr = c_cmi_behavior_create_default(b%ptr)
    if (ierr /= CMI_SUCCESS) error stop "Failed to create default behavior"

    ierr = c_cmi_behavior_should_perform_inference(b%ptr, infer)
    if (.not. infer) error stop "Expected should_perform_inference = true"

    ierr = c_cmi_behavior_time_step_delta(b%ptr, delta)
    if (delta /= 0) error stop "Expected time_step_delta = 0"

    ierr = c_cmi_behavior_should_send_data(b%ptr, send_data)
    if (.not. send_data) error stop "Expected should_send_data = true"

    ierr = c_cmi_behavior_destroy(b%ptr)
    print *, "[PASS] test_behavior"
  end subroutine test_behavior

  subroutine test_behavior_by_name()
    type(cmi_behavior) :: b
    type(c_ptr) :: no_ptrs(1)
    integer(c_int) :: no_types(1)
    integer :: ierr
    logical(c_bool) :: infer

    no_ptrs(1) = c_null_ptr
    no_types(1) = 0_c_int

    ierr = c_cmi_behavior_create_by_name(b%ptr, "Default"//c_null_char, &
                                         no_ptrs, no_ptrs, no_types, 0_c_int)
    if (ierr /= CMI_SUCCESS) error stop "Failed to create behavior by name"

    ierr = c_cmi_behavior_should_perform_inference(b%ptr, infer)
    if (.not. infer) error stop "Behavior-by-name should infer"

    ierr = c_cmi_behavior_destroy(b%ptr)
    print *, "[PASS] test_behavior_by_name"
  end subroutine test_behavior_by_name

  subroutine test_normalization()
    real(c_float), target :: val(2)
    type(cmi_tensor) :: t
    type(cmi_data) :: d
    type(cmi_normalization) :: norm
    integer :: ierr

    val = [5.0_c_float, 10.0_c_float]
    call cmi_tensor_wrap(t, val, ierr)

    ierr = c_cmi_data_create(d%ptr, CMI_DTYPE_FLOAT)
    ierr = c_cmi_data_add_tensor(d%ptr, t%ptr)

    ! MinMax norm from [0, 10] to [0, 10]
    ierr = c_cmi_normalization_create_minmax(norm%ptr, CMI_DTYPE_FLOAT, CMI_DTYPE_FLOAT, &
                                            0.0_c_double, 10.0_c_double, &
                                            0.0_c_double, 10.0_c_double)
    if (ierr /= CMI_SUCCESS) error stop "Failed to create normalization"

    ! Normalize
    ierr = c_cmi_normalization_normalize_input(norm%ptr, d%ptr)
    if (abs(val(1) - 0.5_c_float) > 1.0e-5 .or. abs(val(2) - 1.0_c_float) > 1.0e-5) then
       error stop "Normalization values incorrect"
    end if

    ! Denormalize
    ierr = c_cmi_normalization_denormalize_output(norm%ptr, d%ptr)
    if (abs(val(1) - 5.0_c_float) > 1.0e-5 .or. abs(val(2) - 10.0_c_float) > 1.0e-5) then
       error stop "Denormalization values incorrect"
    end if

    ierr = c_cmi_tensor_destroy(t%ptr)
    ierr = c_cmi_data_destroy(d%ptr)
    ierr = c_cmi_normalization_destroy(norm%ptr)

    print *, "[PASS] test_normalization"
  end subroutine test_normalization

  subroutine test_normalization_by_name()
    real(c_float), target :: val(2)
    real(c_double), target :: p_in_min, p_in_max, p_out_min, p_out_max
    type(c_ptr) :: vals(4)
    type(c_ptr), allocatable :: name_ptrs(:)
    character(kind=c_char), allocatable, target :: name_bufs(:,:)
    character(len=16) :: pnames(4)
    integer(c_int) :: ptypes(4)
    type(cmi_tensor) :: t
    type(cmi_data) :: d
    type(cmi_normalization) :: norm
    integer :: ierr

    p_in_min = 0.0_c_double
    p_in_max = 10.0_c_double
    p_out_min = 0.0_c_double
    p_out_max = 10.0_c_double
    vals(1) = c_loc(p_in_min)
    vals(2) = c_loc(p_in_max)
    vals(3) = c_loc(p_out_min)
    vals(4) = c_loc(p_out_max)
    ptypes = CMI_PARAM_DOUBLE

    pnames(1) = "input_min"
    pnames(2) = "input_max"
    pnames(3) = "output_min"
    pnames(4) = "output_max"
    call cmi_c_string_array(pnames, name_ptrs, name_bufs, ierr)
    if (ierr /= CMI_SUCCESS) error stop "Failed to marshal parameter names"

    ierr = c_cmi_normalization_create_by_name(norm%ptr, "MinMax"//c_null_char, &
                                              CMI_DTYPE_FLOAT, CMI_DTYPE_FLOAT, &
                                              name_ptrs, vals, ptypes, 4_c_int)
    if (ierr /= CMI_SUCCESS) error stop "Failed to create normalization by name"

    val = [5.0_c_float, 10.0_c_float]
    call cmi_tensor_wrap(t, val, ierr)
    ierr = c_cmi_data_create(d%ptr, CMI_DTYPE_FLOAT)
    ierr = c_cmi_data_add_tensor(d%ptr, t%ptr)

    ierr = c_cmi_normalization_normalize_input(norm%ptr, d%ptr)
    if (abs(val(1) - 0.5_c_float) > 1.0e-5 .or. abs(val(2) - 1.0_c_float) > 1.0e-5) then
       error stop "Normalization-by-name values incorrect"
    end if

    ierr = c_cmi_tensor_destroy(t%ptr)
    ierr = c_cmi_data_destroy(d%ptr)
    ierr = c_cmi_normalization_destroy(norm%ptr)

    print *, "[PASS] test_normalization_by_name"
  end subroutine test_normalization_by_name

  subroutine test_library_and_inference()
    type(cmi_library) :: lib
    type(cmi_tensor) :: tin, tout
    type(cmi_data) :: din, dout
    real(c_float), target :: in_arr(3), out_arr(3)
    integer :: ierr

    in_arr = [1.0_c_float, 2.0_c_float, 3.0_c_float]
    out_arr = [0.0_c_float, 0.0_c_float, 0.0_c_float]

    call cmi_tensor_wrap(tin, in_arr, ierr)
    call cmi_tensor_wrap(tout, out_arr, ierr)

    ierr = c_cmi_data_create(din%ptr, CMI_DTYPE_FLOAT)
    ierr = c_cmi_data_add_tensor(din%ptr, tin%ptr)

    ierr = c_cmi_data_create(dout%ptr, CMI_DTYPE_FLOAT)
    ierr = c_cmi_data_add_tensor(dout%ptr, tout%ptr)

    ierr = c_cmi_library_create_generic(lib%ptr, CMI_DTYPE_FLOAT, CMI_DTYPE_FLOAT, &
                                       c_funloc(fortran_inference_callback), c_null_ptr)
    if (ierr /= CMI_SUCCESS) error stop "Failed to create generic library from Fortran"

    ierr = c_cmi_library_inference(lib%ptr, din%ptr, dout%ptr)
    if (ierr /= CMI_SUCCESS) error stop "Inference call failed"

    if (abs(out_arr(1) - 10.0_c_float) > 1.0e-5 .or. &
        abs(out_arr(2) - 20.0_c_float) > 1.0e-5 .or. &
        abs(out_arr(3) - 30.0_c_float) > 1.0e-5) then
       error stop "Inference output incorrect"
    end if

    ierr = c_cmi_tensor_destroy(tin%ptr)
    ierr = c_cmi_tensor_destroy(tout%ptr)
    ierr = c_cmi_data_destroy(din%ptr)
    ierr = c_cmi_data_destroy(dout%ptr)
    ierr = c_cmi_library_destroy(lib%ptr)

    print *, "[PASS] test_library_and_inference"
  end subroutine test_library_and_inference

  subroutine test_library_metadata()
    type(cmi_library) :: lib
    type(c_ptr) :: no_ptrs(1)
    integer(c_int) :: no_types(1)
    integer :: ierr

    no_ptrs(1) = c_null_ptr
    no_types(1) = 0_c_int

    ierr = c_cmi_library_create_by_name(lib%ptr, "Dummy"//c_null_char, &
                                        CMI_DTYPE_FLOAT, CMI_DTYPE_FLOAT, &
                                        no_ptrs, no_ptrs, no_types, 0_c_int)
    if (ierr /= CMI_SUCCESS) error stop "Failed to create library by name"

    ierr = c_cmi_library_set_rank(lib%ptr, 3_c_int)
    if (ierr /= CMI_SUCCESS) error stop "Failed to set library rank"

    ierr = c_cmi_library_set_merge_strategy(lib%ptr, CMI_MERGE_LIST)
    if (ierr /= CMI_SUCCESS) error stop "Failed to set merge strategy"

    ierr = c_cmi_library_destroy(lib%ptr)
    print *, "[PASS] test_library_metadata"
  end subroutine test_library_metadata

  subroutine test_flex_get_and_train()
    type(cmi_library) :: lib
    type(cmi_tensor) :: tin, tout, ttarget
    type(cmi_data) :: din, dout, dtarget
    real(c_float), target :: in_arr(2), out_arr(2), target_arr(2)
    integer(c_int), target :: train_calls
    character(kind=c_char), target :: k_input(6), k_target(7), k_output(7)
    type(c_ptr) :: in_keys(1), target_keys(1), out_keys(1)
    integer :: ierr

    in_arr = [1.0_c_float, 2.0_c_float]
    out_arr = [0.0_c_float, 0.0_c_float]
    target_arr = [0.0_c_float, 0.0_c_float]
    train_calls = 0_c_int

    k_input(1) = 'i'; k_input(2) = 'n'; k_input(3) = 'p'
    k_input(4) = 'u'; k_input(5) = 't'; k_input(6) = c_null_char
    k_target(1) = 't'; k_target(2) = 'a'; k_target(3) = 'r'
    k_target(4) = 'g'; k_target(5) = 'e'; k_target(6) = 't'
    k_target(7) = c_null_char
    k_output(1) = 'o'; k_output(2) = 'u'; k_output(3) = 't'
    k_output(4) = 'p'; k_output(5) = 'u'; k_output(6) = 't'
    k_output(7) = c_null_char
    in_keys(1) = c_loc(k_input)
    target_keys(1) = c_loc(k_target)
    out_keys(1) = c_loc(k_output)

    call cmi_tensor_wrap(tin, in_arr, ierr)
    call cmi_tensor_wrap(tout, out_arr, ierr)
    call cmi_tensor_wrap(ttarget, target_arr, ierr)

    ierr = c_cmi_data_create(din%ptr, CMI_DTYPE_FLOAT)
    ierr = c_cmi_data_add_tensor(din%ptr, tin%ptr)
    ierr = c_cmi_data_create(dout%ptr, CMI_DTYPE_FLOAT)
    ierr = c_cmi_data_add_tensor(dout%ptr, tout%ptr)
    ierr = c_cmi_data_create(dtarget%ptr, CMI_DTYPE_FLOAT)
    ierr = c_cmi_data_add_tensor(dtarget%ptr, ttarget%ptr)

    ierr = c_cmi_library_create_generic_with_training( &
        lib%ptr, CMI_DTYPE_FLOAT, CMI_DTYPE_FLOAT, &
        c_funloc(fortran_inference_callback), c_funloc(fortran_train_callback), &
        c_loc(train_calls))
    if (ierr /= CMI_SUCCESS) error stop "Failed to create training library"

    ! Ordered training
    ierr = c_cmi_library_flex_ordered_set(lib%ptr, din%ptr)
    ierr = c_cmi_library_flex_ordered_set_target(lib%ptr, dtarget%ptr)
    ierr = c_cmi_library_flex_ordered_train(lib%ptr, 1_c_int64_t)
    if (ierr /= CMI_SUCCESS) error stop "Ordered flexible training failed"
    if (train_calls /= 1_c_int) error stop "Expected one ordered training call"

    ! Ordered inference + result retrieval
    ierr = c_cmi_library_flex_ordered_set(lib%ptr, din%ptr)
    ierr = c_cmi_library_flex_ordered_inference(lib%ptr, dout%ptr)
    if (ierr /= CMI_SUCCESS) error stop "Ordered flexible inference failed"
    if (abs(out_arr(1) - 10.0_c_float) > 1.0e-5 .or. &
        abs(out_arr(2) - 20.0_c_float) > 1.0e-5) error stop "Ordered flex output incorrect"

    ierr = c_cmi_library_flex_ordered_get(lib%ptr, dout%ptr)
    if (ierr /= CMI_SUCCESS) error stop "Ordered flex get failed"

    ! Keyed training
    ierr = c_cmi_library_flex_keyed_set(lib%ptr, "input"//c_null_char, din%ptr)
    ierr = c_cmi_library_flex_keyed_set_target(lib%ptr, "target"//c_null_char, dtarget%ptr)
    ierr = c_cmi_library_flex_keyed_train(lib%ptr, 2_c_int64_t, in_keys, 1_c_int, &
                                          target_keys, 1_c_int)
    if (ierr /= CMI_SUCCESS) error stop "Keyed flexible training failed"
    if (train_calls /= 2_c_int) error stop "Expected two training calls"

    ! Keyed inference + result retrieval
    out_arr = [0.0_c_float, 0.0_c_float]
    ierr = c_cmi_library_flex_keyed_set(lib%ptr, "input"//c_null_char, din%ptr)
    ierr = c_cmi_library_flex_keyed_inference(lib%ptr, in_keys, 1_c_int, &
                                              out_keys, 1_c_int, dout%ptr)
    if (ierr /= CMI_SUCCESS) error stop "Keyed flexible inference failed"
    if (abs(out_arr(1) - 10.0_c_float) > 1.0e-5 .or. &
        abs(out_arr(2) - 20.0_c_float) > 1.0e-5) error stop "Keyed flex output incorrect"

    ierr = c_cmi_library_flex_keyed_get(lib%ptr, "output"//c_null_char, dout%ptr)
    if (ierr /= CMI_SUCCESS) error stop "Keyed flex get failed"

    ierr = c_cmi_tensor_destroy(tin%ptr)
    ierr = c_cmi_tensor_destroy(tout%ptr)
    ierr = c_cmi_tensor_destroy(ttarget%ptr)
    ierr = c_cmi_data_destroy(din%ptr)
    ierr = c_cmi_data_destroy(dout%ptr)
    ierr = c_cmi_data_destroy(dtarget%ptr)
    ierr = c_cmi_library_destroy(lib%ptr)

    print *, "[PASS] test_flex_get_and_train"
  end subroutine test_flex_get_and_train

  subroutine test_flex_fortran_layout_merge()
    type(cmi_library) :: lib
    type(cmi_tensor) :: ta, tb, tout
    type(cmi_data) :: da, db, dout
    real(c_float), target :: a(2, 3), b(2, 3), out_arr(12)
    integer :: i, j, ierr

    do j = 1, 3
       do i = 1, 2
          ! Logical tensor coordinates (i-1, j-1): row-major values 1..6 and 11..16.
          a(i, j) = real((i - 1) * 3 + j, kind=c_float)
          b(i, j) = real(10 + (i - 1) * 3 + j, kind=c_float)
       end do
    end do

    g_seen_n = 0_c_int
    out_arr = 0.0_c_float

    call cmi_tensor_wrap(ta, a, ierr)
    call cmi_tensor_wrap(tb, b, ierr)
    call cmi_tensor_wrap(tout, out_arr, ierr)

    ierr = c_cmi_data_create(da%ptr, CMI_DTYPE_FLOAT)
    ierr = c_cmi_data_add_tensor(da%ptr, ta%ptr)
    ierr = c_cmi_data_create(db%ptr, CMI_DTYPE_FLOAT)
    ierr = c_cmi_data_add_tensor(db%ptr, tb%ptr)
    ierr = c_cmi_data_create(dout%ptr, CMI_DTYPE_FLOAT)
    ierr = c_cmi_data_add_tensor(dout%ptr, tout%ptr)

    ierr = c_cmi_library_create_generic(lib%ptr, CMI_DTYPE_FLOAT, CMI_DTYPE_FLOAT, &
                                       c_funloc(fortran_inference_callback), c_null_ptr)
    if (ierr /= CMI_SUCCESS) error stop "Failed to create flex merge library"

    ierr = c_cmi_library_set_merge_strategy(lib%ptr, CMI_MERGE_STACK)
    if (ierr /= CMI_SUCCESS) error stop "Failed to set stack merge strategy"

    ierr = c_cmi_library_flex_ordered_set(lib%ptr, da%ptr)
    ierr = c_cmi_library_flex_ordered_set(lib%ptr, db%ptr)
    ierr = c_cmi_library_flex_ordered_inference(lib%ptr, dout%ptr)
    if (ierr /= CMI_SUCCESS) error stop "Fortran-layout flex merge failed"

    if (g_seen_n /= 12_c_int) error stop "Expected 12 merged input values"
    ! Row-major merged layout for dims [B=2, m=2, D=3]:
    ! [1,2,3, 11,12,13, 4,5,6, 14,15,16]
    if (abs(g_seen(1) - 1.0_c_float) > 1.0e-5) error stop "merged(1) incorrect"
    if (abs(g_seen(2) - 2.0_c_float) > 1.0e-5) error stop "merged(2) incorrect"
    if (abs(g_seen(3) - 3.0_c_float) > 1.0e-5) error stop "merged(3) incorrect"
    if (abs(g_seen(4) - 11.0_c_float) > 1.0e-5) error stop "merged(4) incorrect"
    if (abs(g_seen(5) - 12.0_c_float) > 1.0e-5) error stop "merged(5) incorrect"
    if (abs(g_seen(6) - 13.0_c_float) > 1.0e-5) error stop "merged(6) incorrect"
    if (abs(g_seen(7) - 4.0_c_float) > 1.0e-5) error stop "merged(7) incorrect"
    if (abs(g_seen(8) - 5.0_c_float) > 1.0e-5) error stop "merged(8) incorrect"
    if (abs(g_seen(9) - 6.0_c_float) > 1.0e-5) error stop "merged(9) incorrect"
    if (abs(g_seen(10) - 14.0_c_float) > 1.0e-5) error stop "merged(10) incorrect"
    if (abs(g_seen(11) - 15.0_c_float) > 1.0e-5) error stop "merged(11) incorrect"
    if (abs(g_seen(12) - 16.0_c_float) > 1.0e-5) error stop "merged(12) incorrect"

    ierr = c_cmi_tensor_destroy(ta%ptr)
    ierr = c_cmi_tensor_destroy(tb%ptr)
    ierr = c_cmi_tensor_destroy(tout%ptr)
    ierr = c_cmi_data_destroy(da%ptr)
    ierr = c_cmi_data_destroy(db%ptr)
    ierr = c_cmi_data_destroy(dout%ptr)
    ierr = c_cmi_library_destroy(lib%ptr)

    print *, "[PASS] test_flex_fortran_layout_merge"
  end subroutine test_flex_fortran_layout_merge

  subroutine test_application_callbacks()
    type(cmi_library) :: lib
    type(cmi_application) :: app
    type(cmi_behavior) :: beh
    type(cmi_coupling) :: coupling
    type(cmi_tensor) :: tin, tout
    type(cmi_data) :: din, dout
    type(app_ctx_t), target :: ctx
    real(c_float), target :: in_arr(2), out_arr(2)
    integer :: ierr, delta

    in_arr = [1.0_c_float, 2.0_c_float]
    out_arr = [0.0_c_float, 0.0_c_float]
    ctx%out_ptr = c_loc(out_arr)
    ctx%pre_calls = 0_c_int
    ctx%post_calls = 0_c_int

    call cmi_tensor_wrap(tin, in_arr, ierr)
    call cmi_tensor_wrap(tout, out_arr, ierr)

    ierr = c_cmi_data_create(din%ptr, CMI_DTYPE_FLOAT)
    ierr = c_cmi_data_add_tensor(din%ptr, tin%ptr)
    ierr = c_cmi_data_create(dout%ptr, CMI_DTYPE_FLOAT)
    ierr = c_cmi_data_add_tensor(dout%ptr, tout%ptr)

    ierr = c_cmi_library_create_generic(lib%ptr, CMI_DTYPE_FLOAT, CMI_DTYPE_FLOAT, &
                                       c_funloc(fortran_inference_callback), c_null_ptr)
    if (ierr /= CMI_SUCCESS) error stop "Failed to create callback library"

    ierr = c_cmi_application_create_generic_with_callbacks( &
        app%ptr, CMI_DTYPE_FLOAT, CMI_DTYPE_FLOAT, CMI_DTYPE_FLOAT, CMI_DTYPE_FLOAT, &
        din%ptr, dout%ptr, c_funloc(fortran_preprocess_callback), &
        c_funloc(fortran_postprocess_callback), c_null_ptr, c_loc(ctx))
    if (ierr /= CMI_SUCCESS) error stop "Failed to create callback application"

    ierr = c_cmi_behavior_create_default(beh%ptr)

    ierr = c_cmi_coupling_create(coupling%ptr, &
                                 CMI_DTYPE_FLOAT, CMI_DTYPE_FLOAT, &
                                 CMI_DTYPE_FLOAT, CMI_DTYPE_FLOAT, &
                                 lib%ptr, app%ptr, beh%ptr)
    if (ierr /= CMI_SUCCESS) error stop "Failed to create callback coupling"

    ierr = c_cmi_coupling_step(coupling%ptr, delta)
    if (ierr /= CMI_SUCCESS) error stop "Callback coupling step failed"

    if (ctx%pre_calls /= 1_c_int) error stop "Expected one preprocess call"
    if (ctx%post_calls /= 1_c_int) error stop "Expected one postprocess call"
    if (abs(out_arr(1) - 60.0_c_float) > 1.0e-5 .or. &
        abs(out_arr(2) - 120.0_c_float) > 1.0e-5) error stop "Callback pipeline output incorrect"

    ierr = c_cmi_coupling_destroy(coupling%ptr)
    ierr = c_cmi_library_destroy(lib%ptr)
    ierr = c_cmi_application_destroy(app%ptr)
    ierr = c_cmi_behavior_destroy(beh%ptr)
    ierr = c_cmi_tensor_destroy(tin%ptr)
    ierr = c_cmi_tensor_destroy(tout%ptr)
    ierr = c_cmi_data_destroy(din%ptr)
    ierr = c_cmi_data_destroy(dout%ptr)

    print *, "[PASS] test_application_callbacks"
  end subroutine test_application_callbacks

  subroutine test_coupling_pipeline()
    type(cmi_library) :: lib
    type(cmi_application) :: app
    type(cmi_behavior) :: beh
    type(cmi_coupling) :: coupling
    type(cmi_tensor) :: tin, tout
    type(cmi_data) :: din, dout
    real(c_float), target :: in_arr(2), out_arr(2)
    integer :: ierr, delta

    in_arr = [3.0_c_float, 4.0_c_float]
    out_arr = [0.0_c_float, 0.0_c_float]

    call cmi_tensor_wrap(tin, in_arr, ierr)
    call cmi_tensor_wrap(tout, out_arr, ierr)

    ierr = c_cmi_data_create(din%ptr, CMI_DTYPE_FLOAT)
    ierr = c_cmi_data_add_tensor(din%ptr, tin%ptr)

    ierr = c_cmi_data_create(dout%ptr, CMI_DTYPE_FLOAT)
    ierr = c_cmi_data_add_tensor(dout%ptr, tout%ptr)

    ierr = c_cmi_library_create_generic(lib%ptr, CMI_DTYPE_FLOAT, CMI_DTYPE_FLOAT, &
                                       c_funloc(fortran_inference_callback), c_null_ptr)

    ierr = c_cmi_application_create_generic(app%ptr, &
                                           CMI_DTYPE_FLOAT, CMI_DTYPE_FLOAT, &
                                           CMI_DTYPE_FLOAT, CMI_DTYPE_FLOAT, &
                                           din%ptr, dout%ptr, c_null_ptr)

    ierr = c_cmi_behavior_create_default(beh%ptr)

    ierr = c_cmi_coupling_create(coupling%ptr, &
                                CMI_DTYPE_FLOAT, CMI_DTYPE_FLOAT, &
                                CMI_DTYPE_FLOAT, CMI_DTYPE_FLOAT, &
                                lib%ptr, app%ptr, beh%ptr)
    if (ierr /= CMI_SUCCESS) error stop "Failed to create coupling"

    ierr = c_cmi_coupling_step(coupling%ptr, delta)
    if (ierr /= CMI_SUCCESS) error stop "Coupling step failed"

    if (abs(out_arr(1) - 30.0_c_float) > 1.0e-5 .or. abs(out_arr(2) - 40.0_c_float) > 1.0e-5) then
       error stop "Coupling output incorrect"
    end if

    ierr = c_cmi_coupling_destroy(coupling%ptr)
    ierr = c_cmi_tensor_destroy(tin%ptr)
    ierr = c_cmi_tensor_destroy(tout%ptr)
    ierr = c_cmi_data_destroy(din%ptr)
    ierr = c_cmi_data_destroy(dout%ptr)

    print *, "[PASS] test_coupling_pipeline"
  end subroutine test_coupling_pipeline

  subroutine test_config_with_data()
    integer :: unit
    type(cmi_tensor) :: tin, tout
    type(cmi_data) :: din, dout
    type(cmi_coupling) :: coupling
    real(c_float), target :: in_arr(4), out_arr(4)
    integer :: ierr

    open(newunit=unit, file='cmi_fortran_test_config.toml', status='replace', action='write')
    write(unit, '(a)') 'coupling_type = "STATIC"'
    write(unit, '(a)') ''
    write(unit, '(a)') '[library]'
    write(unit, '(a)') 'class = "Dummy"'
    write(unit, '(a)') ''
    write(unit, '(a)') '[application]'
    write(unit, '(a)') 'class = "MLCouplingApplicationGeneric"'
    close(unit)

    in_arr = [1.0_c_float, 2.0_c_float, 3.0_c_float, 4.0_c_float]
    out_arr = 0.0_c_float

    ierr = c_cmi_tensor_create_flat(tin%ptr, c_loc(in_arr), [2_c_int, 2_c_int], 2_c_int, &
                                    CMI_DTYPE_FLOAT, CMI_LAYOUT_FORTRAN_CONTIGUOUS, &
                                    CMI_OWNERSHIP_EXTERNAL)
    ierr = c_cmi_tensor_create_flat(tout%ptr, c_loc(out_arr), [2_c_int, 2_c_int], 2_c_int, &
                                    CMI_DTYPE_FLOAT, CMI_LAYOUT_FORTRAN_CONTIGUOUS, &
                                    CMI_OWNERSHIP_EXTERNAL)

    ierr = c_cmi_data_create(din%ptr, CMI_DTYPE_FLOAT)
    ierr = c_cmi_data_add_tensor(din%ptr, tin%ptr)
    ierr = c_cmi_data_create(dout%ptr, CMI_DTYPE_FLOAT)
    ierr = c_cmi_data_add_tensor(dout%ptr, tout%ptr)

    ierr = c_cmi_coupling_create_from_config_with_data( &
        coupling%ptr, 'cmi_fortran_test_config.toml'//c_null_char, &
        CMI_DTYPE_FLOAT, CMI_DTYPE_FLOAT, CMI_DTYPE_FLOAT, CMI_DTYPE_FLOAT, &
        din%ptr, dout%ptr)
    if (ierr /= CMI_SUCCESS) error stop "Failed to create config coupling with data"
    if (.not. c_associated(coupling%ptr)) error stop "Config coupling handle is null"

    ierr = c_cmi_coupling_destroy(coupling%ptr)
    ierr = c_cmi_tensor_destroy(tin%ptr)
    ierr = c_cmi_tensor_destroy(tout%ptr)
    ierr = c_cmi_data_destroy(din%ptr)
    ierr = c_cmi_data_destroy(dout%ptr)

    open(newunit=unit, file='cmi_fortran_test_config.toml', status='old')
    close(unit, status='delete')

    print *, "[PASS] test_config_with_data"
  end subroutine test_config_with_data

  subroutine test_config_string_and_overrides()
    character(len=*), parameter :: invalid_config = &
        'coupling_type = "STATIC"'//achar(10)// &
        '[library]'//achar(10)//'class = "MissingLibrary"'//achar(10)// &
        '[application]'//achar(10)//'class = "MLCouplingApplicationGeneric"'
    character(kind=c_char), target :: dummy_name(6)
    type(cmi_tensor) :: tin, tout
    type(cmi_data) :: din, dout
    type(cmi_config_overrides) :: ov
    type(cmi_coupling) :: coupling
    real(c_float), target :: in_arr(2), out_arr(2)
    integer :: ierr

    dummy_name(1) = 'D'; dummy_name(2) = 'u'; dummy_name(3) = 'm'
    dummy_name(4) = 'm'; dummy_name(5) = 'y'; dummy_name(6) = c_null_char

    in_arr = [1.0_c_float, 2.0_c_float]
    out_arr = 0.0_c_float

    call cmi_tensor_wrap(tin, in_arr, ierr)
    call cmi_tensor_wrap(tout, out_arr, ierr)
    ierr = c_cmi_data_create(din%ptr, CMI_DTYPE_FLOAT)
    ierr = c_cmi_data_add_tensor(din%ptr, tin%ptr)
    ierr = c_cmi_data_create(dout%ptr, CMI_DTYPE_FLOAT)
    ierr = c_cmi_data_add_tensor(dout%ptr, tout%ptr)

    ierr = c_cmi_config_overrides_create(ov%ptr)
    if (ierr /= CMI_SUCCESS) error stop "Failed to create override set"
    ierr = c_cmi_config_overrides_set(ov%ptr, "library"//c_null_char, &
                                      "class"//c_null_char, CMI_PARAM_STRING, &
                                      c_loc(dummy_name))
    if (ierr /= CMI_SUCCESS) error stop "Failed to set string override"

    ierr = c_cmi_coupling_create_from_config_string_overrides( &
        coupling%ptr, invalid_config//c_null_char, &
        CMI_DTYPE_FLOAT, CMI_DTYPE_FLOAT, CMI_DTYPE_FLOAT, CMI_DTYPE_FLOAT, &
        din%ptr, dout%ptr, ov%ptr)
    if (ierr /= CMI_SUCCESS) error stop "Config string with overrides failed"
    if (.not. c_associated(coupling%ptr)) error stop "Config string coupling is null"

    ierr = c_cmi_coupling_destroy(coupling%ptr)
    ierr = c_cmi_config_overrides_destroy(ov%ptr)
    ierr = c_cmi_tensor_destroy(tin%ptr)
    ierr = c_cmi_tensor_destroy(tout%ptr)
    ierr = c_cmi_data_destroy(din%ptr)
    ierr = c_cmi_data_destroy(dout%ptr)

    print *, "[PASS] test_config_string_and_overrides"
  end subroutine test_config_string_and_overrides

  subroutine test_coupling_tracking()
    type(cmi_library) :: lib
    type(cmi_application) :: app
    type(cmi_behavior) :: beh
    type(cmi_coupling) :: coupling
    type(cmi_tensor) :: tin, tout
    type(cmi_data) :: din, dout
    real(c_float), target :: in_arr(2), out_arr(2)
    real(c_double) :: current
    real(c_double) :: history(4)
    real(c_double) :: values(1)
    character(kind=c_char), target :: loss_name(5)
    type(c_ptr) :: field_ptrs(1)
    integer(c_int) :: count
    integer :: ierr

    loss_name(1) = 'l'; loss_name(2) = 'o'; loss_name(3) = 's'
    loss_name(4) = 's'; loss_name(5) = c_null_char
    field_ptrs(1) = c_loc(loss_name)

    in_arr = [1.0_c_float, 2.0_c_float]
    out_arr = 0.0_c_float
    call cmi_tensor_wrap(tin, in_arr, ierr)
    call cmi_tensor_wrap(tout, out_arr, ierr)
    ierr = c_cmi_data_create(din%ptr, CMI_DTYPE_FLOAT)
    ierr = c_cmi_data_add_tensor(din%ptr, tin%ptr)
    ierr = c_cmi_data_create(dout%ptr, CMI_DTYPE_FLOAT)
    ierr = c_cmi_data_add_tensor(dout%ptr, tout%ptr)

    ierr = c_cmi_library_create_generic(lib%ptr, CMI_DTYPE_FLOAT, CMI_DTYPE_FLOAT, &
                                       c_funloc(fortran_inference_callback), c_null_ptr)
    ierr = c_cmi_application_create_generic(app%ptr, CMI_DTYPE_FLOAT, CMI_DTYPE_FLOAT, &
                                           CMI_DTYPE_FLOAT, CMI_DTYPE_FLOAT, &
                                           din%ptr, dout%ptr, c_null_ptr)
    ierr = c_cmi_behavior_create_default(beh%ptr)
    ierr = c_cmi_coupling_create(coupling%ptr, &
                                 CMI_DTYPE_FLOAT, CMI_DTYPE_FLOAT, &
                                 CMI_DTYPE_FLOAT, CMI_DTYPE_FLOAT, &
                                 lib%ptr, app%ptr, beh%ptr)
    if (ierr /= CMI_SUCCESS) error stop "Failed to create tracking coupling"

    ierr = c_cmi_coupling_set_merge_strategy(coupling%ptr, CMI_MERGE_LIST)
    if (ierr /= CMI_SUCCESS) error stop "Failed to set coupling merge strategy"

    ierr = c_cmi_coupling_track(coupling%ptr, "loss"//c_null_char)
    if (ierr /= CMI_SUCCESS) error stop "Failed to track loss"

    values(1) = 0.5_c_double
    ierr = c_cmi_coupling_log_training_metrics(coupling%ptr, 1_c_int64_t, &
                                               field_ptrs, values, 1_c_int)
    if (ierr /= CMI_SUCCESS) error stop "Failed to log training metrics"

    ierr = c_cmi_coupling_get_current(coupling%ptr, "loss"//c_null_char, current)
    if (ierr /= CMI_SUCCESS) error stop "Failed to read current metric"
    if (abs(current - 0.5_c_double) > 1.0e-12) error stop "Unexpected current metric"

    values(1) = 0.25_c_double
    ierr = c_cmi_coupling_log_training_metrics(coupling%ptr, 2_c_int64_t, &
                                               field_ptrs, values, 1_c_int)

    count = -1_c_int
    ierr = c_cmi_coupling_get_history(coupling%ptr, "loss"//c_null_char, &
                                      history, 4_c_int, count)
    if (ierr /= CMI_SUCCESS) error stop "Failed to read metric history"
    if (count /= 2_c_int) error stop "Unexpected metric history count"
    if (abs(history(1) - 0.5_c_double) > 1.0e-12) error stop "Unexpected history(1)"
    if (abs(history(2) - 0.25_c_double) > 1.0e-12) error stop "Unexpected history(2)"

    ierr = c_cmi_coupling_destroy(coupling%ptr)
    ierr = c_cmi_library_destroy(lib%ptr)
    ierr = c_cmi_application_destroy(app%ptr)
    ierr = c_cmi_behavior_destroy(beh%ptr)
    ierr = c_cmi_tensor_destroy(tin%ptr)
    ierr = c_cmi_tensor_destroy(tout%ptr)
    ierr = c_cmi_data_destroy(din%ptr)
    ierr = c_cmi_data_destroy(dout%ptr)

    print *, "[PASS] test_coupling_tracking"
  end subroutine test_coupling_tracking

end program test_fortran_api

! Callback function for generic library inference in Fortran.
! Records the merged input and multiplies it by 10.
subroutine fortran_inference_callback(in_d, out_d, user_data) bind(C)
  use, intrinsic :: iso_c_binding
  use cmi
  use test_state_mod
  implicit none
  type(c_ptr), value :: in_d, out_d, user_data
  type(c_ptr) :: t_in, t_out, p_in, p_out
  real(c_float), pointer :: in_arr(:), out_arr(:)
  integer(c_size_t) :: n_in, n_out
  integer(c_int) :: ierr
  integer :: i

  ierr = c_cmi_data_get_tensor(in_d, 0_c_int, t_in)
  ierr = c_cmi_data_get_tensor(out_d, 0_c_int, t_out)

  ierr = c_cmi_tensor_get_data(t_in, p_in)
  ierr = c_cmi_tensor_get_data(t_out, p_out)
  ierr = c_cmi_tensor_get_numel(t_in, n_in)
  ierr = c_cmi_tensor_get_numel(t_out, n_out)

  call c_f_pointer(p_in, in_arr, [int(n_in)])
  call c_f_pointer(p_out, out_arr, [int(n_out)])

  g_seen_n = int(min(int(n_in), size(g_seen)), kind=c_int)
  do i = 1, int(g_seen_n)
     g_seen(i) = in_arr(i)
  end do

  do i = 1, int(min(n_in, n_out))
     out_arr(i) = in_arr(i) * 10.0_c_float
  end do

  ierr = c_cmi_tensor_destroy(t_in)
  ierr = c_cmi_tensor_destroy(t_out)
end subroutine fortran_inference_callback

! Callback function for generic training (counts invocations).
subroutine fortran_train_callback(in_d, target_d, user_data) bind(C)
  use, intrinsic :: iso_c_binding
  implicit none
  type(c_ptr), value :: in_d, target_d, user_data
  integer(c_int), pointer :: counter

  call c_f_pointer(user_data, counter)
  counter = counter + 1_c_int
end subroutine fortran_train_callback

! Preprocessing callback: scales coupling input by 2 into the library input.
subroutine fortran_preprocess_callback(in_d, out_d, user_data) bind(C)
  use, intrinsic :: iso_c_binding
  use cmi
  use test_state_mod
  implicit none
  type(c_ptr), value :: in_d, out_d, user_data
  type(app_ctx_t), pointer :: ctx
  type(c_ptr) :: t_in, t_out, p_in
  real(c_float), pointer :: in_arr(:)
  real(c_float), allocatable, target :: tmp(:)
  integer(c_size_t) :: n
  integer(c_int) :: ierr
  integer :: i

  call c_f_pointer(user_data, ctx)
  ctx%pre_calls = ctx%pre_calls + 1_c_int

  ierr = c_cmi_data_get_tensor(in_d, 0_c_int, t_in)
  ierr = c_cmi_tensor_get_data(t_in, p_in)
  ierr = c_cmi_tensor_get_numel(t_in, n)

  allocate(tmp(int(n)))
  call c_f_pointer(p_in, in_arr, [int(n)])
  do i = 1, int(n)
     tmp(i) = in_arr(i) * 2.0_c_float
  end do

  ierr = c_cmi_tensor_create_from_copy(t_out, c_loc(tmp), [int(n, c_int)], 1_c_int, &
                                       CMI_DTYPE_FLOAT, CMI_LAYOUT_CONTIGUOUS)
  ierr = c_cmi_data_add_tensor(out_d, t_out)
  ierr = c_cmi_tensor_destroy(t_out)
  ierr = c_cmi_tensor_destroy(t_in)
  deallocate(tmp)
end subroutine fortran_preprocess_callback

! Postprocessing callback: scales library output by 3 into the coupling output buffer.
subroutine fortran_postprocess_callback(in_d, out_d, user_data) bind(C)
  use, intrinsic :: iso_c_binding
  use cmi
  use test_state_mod
  implicit none
  type(c_ptr), value :: in_d, out_d, user_data
  type(app_ctx_t), pointer :: ctx
  type(c_ptr) :: t_in, t_out, p_in, p_out
  real(c_float), pointer :: in_arr(:), out_arr(:)
  integer(c_size_t) :: n
  integer(c_int) :: ierr
  integer :: i

  call c_f_pointer(user_data, ctx)
  ctx%post_calls = ctx%post_calls + 1_c_int

  ierr = c_cmi_data_get_tensor(in_d, 0_c_int, t_in)
  ierr = c_cmi_tensor_get_data(t_in, p_in)
  ierr = c_cmi_tensor_get_numel(t_in, n)

  p_out = ctx%out_ptr
  call c_f_pointer(p_in, in_arr, [int(n)])
  call c_f_pointer(p_out, out_arr, [int(n)])
  do i = 1, int(n)
     out_arr(i) = in_arr(i) * 3.0_c_float
  end do

  ierr = c_cmi_tensor_create_flat(t_out, p_out, [int(n, c_int)], 1_c_int, &
                                  CMI_DTYPE_FLOAT, CMI_LAYOUT_CONTIGUOUS, &
                                  CMI_OWNERSHIP_EXTERNAL)
  ierr = c_cmi_data_add_tensor(out_d, t_out)
  ierr = c_cmi_tensor_destroy(t_out)
  ierr = c_cmi_tensor_destroy(t_in)
end subroutine fortran_postprocess_callback
