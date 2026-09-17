! =============================================================================
! Test Fortran API (test_fortran_api.f90)
! Tests the cmi Fortran module and column-major array support.
! =============================================================================

program test_fortran_api
  use, intrinsic :: iso_c_binding
  use cmi
  implicit none

  interface
     subroutine fortran_inference_callback(in_d, out_d, user_data) bind(C)
       import :: c_ptr
       type(c_ptr), value :: in_d, out_d, user_data
     end subroutine fortran_inference_callback
  end interface

  call test_tensor_lifecycle_and_column_major()
  call test_data_lifecycle()
  call test_behavior()
  call test_normalization()
  call test_library_and_inference()
  call test_coupling_pipeline()

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
    real(c_float), pointer :: f_data(:)
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
    if (ierr /= CMI_SUCCESS) stop "Failed to wrap 2d tensor"

    ! Check layout is FortranContiguous
    ierr = c_cmi_tensor_get_layout(t%ptr, layout)
    if (layout /= CMI_LAYOUT_FORTRAN_CONTIGUOUS) stop "Expected FortranContiguous layout"

    ! Check data type is float
    ierr = c_cmi_tensor_get_data_type(t%ptr, dtype)
    if (dtype /= CMI_DTYPE_FLOAT) stop "Expected float dtype"

    ! Check numel is 12
    ierr = c_cmi_tensor_get_numel(t%ptr, numel)
    if (numel /= 12_c_size_t) stop "Expected numel 12"

    ! Check dims are [3, 4]
    ierr = c_cmi_tensor_get_dims(t%ptr, dims_ptr, ndims)
    if (ndims /= 2) stop "Expected 2 dimensions"
    call c_f_pointer(dims_ptr, f_dims, [2])
    if (f_dims(1) /= 3 .or. f_dims(2) /= 4) stop "Expected dimensions [3, 4]"

    ! Check zero-copy: raw pointer equals c_loc(arr2d)
    ierr = c_cmi_tensor_get_data(t%ptr, data_ptr)
    if (.not. c_associated(data_ptr, c_loc(arr2d))) stop "Expected zero-copy memory pointer match"

    ! Destroy tensor handle
    ierr = c_cmi_tensor_destroy(t%ptr)
    if (ierr /= CMI_SUCCESS) stop "Failed to destroy tensor"

    print *, "[PASS] test_tensor_lifecycle_and_column_major"
  end subroutine test_tensor_lifecycle_and_column_major

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
    if (ierr /= CMI_SUCCESS) stop "Failed to create data"

    ierr = c_cmi_data_add_tensor(d%ptr, t1%ptr)
    ierr = c_cmi_data_add_tensor(d%ptr, t2%ptr)

    ierr = c_cmi_data_size(d%ptr, sz)
    if (sz /= 2) stop "Expected data size 2"

    ierr = c_cmi_tensor_destroy(t1%ptr)
    ierr = c_cmi_tensor_destroy(t2%ptr)
    ierr = c_cmi_data_destroy(d%ptr)

    print *, "[PASS] test_data_lifecycle"
  end subroutine test_data_lifecycle

  subroutine test_behavior()
    type(cmi_behavior) :: b
    integer :: ierr, delta
    logical(c_bool) :: infer, send_data

    ierr = c_cmi_behavior_create_default(b%ptr)
    if (ierr /= CMI_SUCCESS) stop "Failed to create default behavior"

    ierr = c_cmi_behavior_should_perform_inference(b%ptr, infer)
    if (.not. infer) stop "Expected should_perform_inference = true"

    ierr = c_cmi_behavior_time_step_delta(b%ptr, delta)
    if (delta /= 0) stop "Expected time_step_delta = 0"

    ierr = c_cmi_behavior_should_send_data(b%ptr, send_data)
    if (.not. send_data) stop "Expected should_send_data = true"

    ierr = c_cmi_behavior_destroy(b%ptr)
    print *, "[PASS] test_behavior"
  end subroutine test_behavior

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
    if (ierr /= CMI_SUCCESS) stop "Failed to create normalization"

    ! Normalize
    ierr = c_cmi_normalization_normalize_input(norm%ptr, d%ptr)
    if (abs(val(1) - 0.5_c_float) > 1.0e-5 .or. abs(val(2) - 1.0_c_float) > 1.0e-5) then
       stop "Normalization values incorrect"
    end if

    ! Denormalize
    ierr = c_cmi_normalization_denormalize_output(norm%ptr, d%ptr)
    if (abs(val(1) - 5.0_c_float) > 1.0e-5 .or. abs(val(2) - 10.0_c_float) > 1.0e-5) then
       stop "Denormalization values incorrect"
    end if

    ierr = c_cmi_tensor_destroy(t%ptr)
    ierr = c_cmi_data_destroy(d%ptr)
    ierr = c_cmi_normalization_destroy(norm%ptr)

    print *, "[PASS] test_normalization"
  end subroutine test_normalization

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
    if (ierr /= CMI_SUCCESS) stop "Failed to create generic library from Fortran"

    ierr = c_cmi_library_inference(lib%ptr, din%ptr, dout%ptr)
    if (ierr /= CMI_SUCCESS) stop "Inference call failed"

    if (abs(out_arr(1) - 10.0_c_float) > 1.0e-5 .or. &
        abs(out_arr(2) - 20.0_c_float) > 1.0e-5 .or. &
        abs(out_arr(3) - 30.0_c_float) > 1.0e-5) then
       stop "Inference output incorrect"
    end if

    ierr = c_cmi_tensor_destroy(tin%ptr)
    ierr = c_cmi_tensor_destroy(tout%ptr)
    ierr = c_cmi_data_destroy(din%ptr)
    ierr = c_cmi_data_destroy(dout%ptr)
    ierr = c_cmi_library_destroy(lib%ptr)

    print *, "[PASS] test_library_and_inference"
  end subroutine test_library_and_inference

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
    if (ierr /= CMI_SUCCESS) stop "Failed to create coupling"

    ierr = c_cmi_coupling_step(coupling%ptr, delta)
    if (ierr /= CMI_SUCCESS) stop "Coupling step failed"

    if (abs(out_arr(1) - 30.0_c_float) > 1.0e-5 .or. abs(out_arr(2) - 40.0_c_float) > 1.0e-5) then
       stop "Coupling output incorrect"
    end if

    ierr = c_cmi_coupling_destroy(coupling%ptr)
    ierr = c_cmi_tensor_destroy(tin%ptr)
    ierr = c_cmi_tensor_destroy(tout%ptr)
    ierr = c_cmi_data_destroy(din%ptr)
    ierr = c_cmi_data_destroy(dout%ptr)

    print *, "[PASS] test_coupling_pipeline"
  end subroutine test_coupling_pipeline

end program test_fortran_api

! Callback function for generic library inference in Fortran
subroutine fortran_inference_callback(in_d, out_d, user_data) bind(C)
  use, intrinsic :: iso_c_binding
  use cmi
  implicit none
  type(c_ptr), value :: in_d, out_d, user_data
  type(c_ptr) :: t_in, t_out, p_in, p_out
  real(c_float), pointer :: in_arr(:), out_arr(:)
  integer(c_size_t) :: n
  integer(c_int) :: ierr
  integer :: i

  ierr = c_cmi_data_get_tensor(in_d, 0_c_int, t_in)
  ierr = c_cmi_data_get_tensor(out_d, 0_c_int, t_out)

  ierr = c_cmi_tensor_get_data(t_in, p_in)
  ierr = c_cmi_tensor_get_data(t_out, p_out)
  ierr = c_cmi_tensor_get_numel(t_in, n)

  call c_f_pointer(p_in, in_arr, [int(n)])
  call c_f_pointer(p_out, out_arr, [int(n)])

  do i = 1, int(n)
     out_arr(i) = in_arr(i) * 10.0_c_float
  end do

  ierr = c_cmi_tensor_destroy(t_in)
  ierr = c_cmi_tensor_destroy(t_out)
end subroutine fortran_inference_callback
