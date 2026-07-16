//! Literal port of kernels/portable/cpu/util/broadcast_util.cpp + kernels/portable/cpu/util/broadcast_util.h.

extern crate alloc;

use crate::kernels::portable::cpu::util::broadcast_indexes_range::BroadcastIndexesRange;
use crate::kernels::portable::cpu::util::repeat_util::repeat_tensor;
use crate::runtime::core::array_ref::ArrayRef;
use crate::runtime::core::error::Error;
use crate::runtime::core::exec_aten::util::tensor_util::{
    K_TENSOR_DIMENSION_LIMIT, resize_tensor_same_type,
};
use crate::runtime::core::portable_type::device::DeviceType;
use crate::runtime::core::portable_type::scalar_type::ScalarType;
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::core::portable_type::tensor_impl::{
    DimOrderType, SizesType, StridesType, TensorImpl, ssize_t,
};
use crate::runtime::core::span::Span;
use crate::runtime::core::tensor_shape_dynamism::TensorShapeDynamism;

// PORT-NOTE: `ET_CHECK` / `ET_CHECK_MSG` are C++ fatal checks; mirrored with a
// local `runtime_abort` on failure, matching tensor_util.rs / scalar_type_util.rs.
// Format arguments are dropped since a fatal abort follows.
macro_rules! et_check {
    ($cond:expr) => {
        if !($cond) {
            crate::runtime::platform::abort::runtime_abort();
        }
    };
}
macro_rules! et_check_msg {
    ($cond:expr, $($arg:tt)*) => {
        if !($cond) {
            crate::runtime::platform::abort::runtime_abort();
        }
    };
}

// [spec:et:def:broadcast-util.torch.executor.free-broadcast-tensor-fn]
// [spec:et:sem:broadcast-util.torch.executor.free-broadcast-tensor-fn]
//
// PORT-NOTE: the C++ `free()`s the raw malloc allocations backing a tensor
// produced by `make_tensor`. This port allocates those buffers with the global
// allocator (see `make_tensor`), so this frees them via `alloc::dealloc` using
// the same layouts. The order matches the C++ (data, sizes, dim_order,
// strides, impl).
pub fn free_broadcast_tensor(broadcast_tensor: &Tensor) {
    let impl_ = broadcast_tensor.unsafe_get_tensor_impl();
    let dim = broadcast_tensor.dim() as usize;
    let nbytes = broadcast_tensor.nbytes();
    unsafe {
        if nbytes != 0 {
            alloc::alloc::dealloc(
                broadcast_tensor.const_data_ptr::<u8>() as *mut u8,
                alloc::alloc::Layout::from_size_align(nbytes, 1).unwrap(),
            );
        }
        dealloc_array::<SizesType>(broadcast_tensor.sizes().data() as *mut SizesType, dim);
        dealloc_array::<DimOrderType>(
            broadcast_tensor.dim_order().data() as *mut DimOrderType,
            dim,
        );
        dealloc_array::<StridesType>(broadcast_tensor.strides().data() as *mut StridesType, dim);
        alloc::alloc::dealloc(impl_ as *mut u8, alloc::alloc::Layout::new::<TensorImpl>());
    }
}

unsafe fn alloc_array<T>(count: usize) -> *mut T {
    if count == 0 {
        return core::ptr::NonNull::<T>::dangling().as_ptr();
    }
    let layout = alloc::alloc::Layout::array::<T>(count).unwrap();
    alloc::alloc::alloc(layout) as *mut T
}

unsafe fn dealloc_array<T>(ptr: *mut T, count: usize) {
    if count == 0 {
        return;
    }
    let layout = alloc::alloc::Layout::array::<T>(count).unwrap();
    alloc::alloc::dealloc(ptr as *mut u8, layout);
}

// [spec:et:def:broadcast-util.torch.executor.make-tensor-fn]
// [spec:et:sem:broadcast-util.torch.executor.make-tensor-fn]
//
// PORT-NOTE: C++ uses raw `malloc`/`memcpy`/placement-new; this port uses the
// global allocator to obtain the buffers and copies the input arrays into them.
// The data buffer is left uninitialized (garbage) as in C++.
fn make_tensor(
    sizes: &ArrayRef<SizesType>,
    dim_order: &ArrayRef<DimOrderType>,
    strides: &ArrayRef<StridesType>,
    dtype: &ScalarType,
) -> Tensor<'static> {
    let dim = sizes.size();

    let size_data_ptr = unsafe { alloc_array::<SizesType>(dim) };
    et_check_msg!(!size_data_ptr.is_null(), "Failed to malloc for size bytes");
    unsafe {
        core::ptr::copy_nonoverlapping(sizes.data(), size_data_ptr, dim);
    }

    let dim_order_data_ptr = unsafe { alloc_array::<DimOrderType>(dim) };
    et_check_msg!(
        !dim_order_data_ptr.is_null(),
        "Failed to malloc for dim order bytes"
    );
    unsafe {
        core::ptr::copy_nonoverlapping(dim_order.data(), dim_order_data_ptr, dim);
    }

    let strides_data_ptr = unsafe { alloc_array::<StridesType>(dim) };
    et_check_msg!(
        !strides_data_ptr.is_null(),
        "Failed to malloc for strides bytes"
    );
    unsafe {
        core::ptr::copy_nonoverlapping(strides.data(), strides_data_ptr, dim);
    }

    let tensor_impl = unsafe {
        alloc::alloc::alloc(alloc::alloc::Layout::new::<TensorImpl>()) as *mut TensorImpl
    };
    et_check_msg!(
        !tensor_impl.is_null(),
        "Failed to malloc for data TensorImpl"
    );

    unsafe {
        core::ptr::write(
            tensor_impl,
            TensorImpl::new(
                *dtype,
                dim as ssize_t,
                size_data_ptr,
                core::ptr::null_mut(),
                dim_order_data_ptr,
                strides_data_ptr,
                TensorShapeDynamism::STATIC,
                DeviceType::CPU,
                0,
            ),
        );
    }

    let nbytes = unsafe { (*tensor_impl).nbytes() };
    let data_ptr = if nbytes == 0 {
        core::ptr::NonNull::<u8>::dangling().as_ptr()
    } else {
        unsafe { alloc::alloc::alloc(alloc::alloc::Layout::from_size_align(nbytes, 1).unwrap()) }
    };
    et_check_msg!(!data_ptr.is_null(), "Failed to malloc for data buffer");
    unsafe {
        (*tensor_impl).set_data(data_ptr as *mut core::ffi::c_void);
    }

    Tensor::new(tensor_impl)
}

// [spec:et:def:broadcast-util.torch.executor.tensor-is-broadcastable-to-fn]
// [spec:et:sem:broadcast-util.torch.executor.tensor-is-broadcastable-to-fn]
pub fn tensor_is_broadcastable_to(
    broadcast_from_shape: ArrayRef<SizesType>,
    broadcast_to_shape: ArrayRef<SizesType>,
) -> bool {
    let mut feasible_bcast = true;

    if broadcast_to_shape.size() < broadcast_from_shape.size() {
        return false;
    }

    let mut i = broadcast_to_shape.size() as isize - 1;
    let mut j = broadcast_from_shape.size() as isize - 1;
    while j >= 0 {
        let broadcast_to_s = *broadcast_to_shape.at(i as usize);
        let broadcast_from_s = *broadcast_from_shape.at(j as usize);
        feasible_bcast &= broadcast_to_s == broadcast_from_s || broadcast_from_s == 1;
        if !feasible_bcast {
            return false;
        }
        i -= 1;
        j -= 1;
    }

    feasible_bcast
}

pub fn tensor_is_broadcastable_to_tensors(broadcast_from: &Tensor, broadcast_to: &Tensor) -> bool {
    tensor_is_broadcastable_to(broadcast_from.sizes(), broadcast_to.sizes())
}

// [spec:et:def:broadcast-util.torch.executor.tensors-are-broadcastable-between-fn]
// [spec:et:sem:broadcast-util.torch.executor.tensors-are-broadcastable-between-fn]
pub fn tensors_are_broadcastable_between(
    a_shape: ArrayRef<SizesType>,
    b_shape: ArrayRef<SizesType>,
) -> bool {
    let a_dim = a_shape.size();
    let b_dim = b_shape.size();

    // Although the documentation says that tensor with 0-dim can not be
    // broadcasted, experiment shows that actually it can. So here we do not test
    // the dimension.

    let mut a_index = a_dim as isize - 1;
    let mut b_index = b_dim as isize - 1;
    while a_index >= 0 && b_index >= 0 {
        let a_s = *a_shape.at(a_index as usize);
        let b_s = *b_shape.at(b_index as usize);
        if a_s == b_s || a_s == 1 || b_s == 1 {
            a_index -= 1;
            b_index -= 1;
            continue;
        }
        return false;
    }

    true
}

pub fn tensors_are_broadcastable_between_tensors(a: &Tensor, b: &Tensor) -> bool {
    tensors_are_broadcastable_between(a.sizes(), b.sizes())
}

// Broadcast tensor broadcast_from to match broadcast_to's shape, and return the
// broadcasted tensor.
// [spec:et:def:broadcast-util.torch.executor.broadcast-tensor-fn]
// [spec:et:sem:broadcast-util.torch.executor.broadcast-tensor-fn]
// [spec:et:def:broadcast-util.torch.executor.executorch.aten.tensor-broadcast-tensor-fn]
// [spec:et:sem:broadcast-util.torch.executor.executorch.aten.tensor-broadcast-tensor-fn]
//
// PORT-NOTE: DEPRECATED; the returned tensor owns raw allocations and must be
// released with `free_broadcast_tensor`.
pub fn broadcast_tensor(broadcast_from: &Tensor, broadcast_to: &Tensor) -> Tensor<'static> {
    let broadcast_to_shape = broadcast_to.sizes();
    let broadcast_from_shape = broadcast_from.sizes();
    let broadcast_to_dim_order = broadcast_to.dim_order();
    let broadcast_to_strides = broadcast_to.strides();

    et_check_msg!(
        broadcast_from.numel() != 0 || !broadcast_from.sizes().empty(),
        "Input tensor must be non-empty"
    );
    et_check_msg!(
        !broadcast_to.sizes().empty(),
        "Input tensor must be non-empty"
    );
    et_check_msg!(
        broadcast_to_shape.size() >= broadcast_from_shape.size(),
        "For broadcast, tensor broadcast_to must be higher dimensional than tensor broadcast_from"
    );

    let feasible_bcast = tensor_is_broadcastable_to_tensors(broadcast_from, broadcast_to);

    et_check_msg!(
        feasible_bcast,
        "Cannot broadcast tensor broadcast_from into tensor broadcast_to along some dimensions"
    );

    let out = make_tensor(
        &broadcast_to_shape,
        &broadcast_to_dim_order,
        &broadcast_to_strides,
        &broadcast_from.scalar_type(),
    );

    // We need to pass IntArrayRef (i.e. ArrayRef<int64_t>) to repeat_tensor() but
    // .sizes() is ArrayRef<int32_t>
    let ndim = broadcast_to.dim();

    // repeat is int64_t* but broadcast_to_shape is ArrayRef<int32_t>
    let repeats = unsafe { alloc_array::<i64>(ndim as usize) };
    for i in 0..ndim {
        unsafe {
            *repeats.offset(i as isize) = *broadcast_to_shape.at(i as usize) as i64;
        }
    }

    // Compute the repeat factor along each dimension
    let mut i = broadcast_to_shape.size() as isize - 1;
    let mut j = broadcast_from_shape.size() as isize - 1;
    while j >= 0 {
        if *broadcast_to_shape.at(i as usize) == *broadcast_from_shape.at(j as usize) {
            unsafe {
                *repeats.offset(i) = 1;
            }
        }
        i -= 1;
        j -= 1;
    }

    et_check!(
        repeat_tensor(
            broadcast_from,
            ArrayRef::from_raw_parts(repeats, ndim as usize),
            &out,
        ) == Error::Ok
    );

    unsafe {
        dealloc_array::<i64>(repeats, ndim as usize);
    }

    out
}

// [spec:et:def:broadcast-util.torch.executor.get-broadcast-target-size-fn]
// [spec:et:sem:broadcast-util.torch.executor.get-broadcast-target-size-fn]
#[must_use]
pub fn get_broadcast_target_size(
    a_size: ArrayRef<SizesType>,
    b_size: ArrayRef<SizesType>,
    out_sizes: *mut SizesType,
    out_sizes_len: usize,
    out_dim: *mut usize,
) -> Error {
    if !tensors_are_broadcastable_between(a_size, b_size) {
        let a_size_span = Span::from_raw_parts(a_size.data() as *mut SizesType, a_size.size());
        let b_size_span = Span::from_raw_parts(b_size.data() as *mut SizesType, b_size.size());
        let _ = &a_size_span;
        let _ = &b_size_span;
        crate::et_log!(
            Error,
            "Two input tensors should be broadcastable but got shapes {:?} and {:?}.",
            crate::runtime::core::exec_aten::util::tensor_shape_to_c_string::tensor_shape_to_c_string(
                a_size_span
            ),
            crate::runtime::core::exec_aten::util::tensor_shape_to_c_string::tensor_shape_to_c_string(
                b_size_span
            )
        );
        return Error::InvalidArgument;
    }

    let a_dim = a_size.size();
    let b_dim = b_size.size();

    crate::et_check_or_return_error!(
        a_dim <= out_sizes_len && b_dim <= out_sizes_len,
        InvalidArgument,
        "Dim of input tensors should be smaller than the limitation, but find {}, {} and {}.",
        a_dim,
        b_dim,
        out_sizes_len
    );

    unsafe {
        *out_dim = if a_dim > b_dim { a_dim } else { b_dim };
    }

    let mut a_idx = a_dim as isize - 1;
    let mut b_idx = b_dim as isize - 1;
    let mut expected_target_idx = unsafe { *out_dim } as isize - 1;
    while expected_target_idx >= 0 {
        unsafe {
            if a_idx >= 0 && b_idx >= 0 {
                *out_sizes.offset(expected_target_idx) = if *b_size.at(b_idx as usize) == 1 {
                    *a_size.at(a_idx as usize)
                } else {
                    *b_size.at(b_idx as usize)
                };
            } else {
                *out_sizes.offset(expected_target_idx) = if a_idx >= 0 {
                    *a_size.at(a_idx as usize)
                } else {
                    *b_size.at(b_idx as usize)
                };
            }
        }
        a_idx -= 1;
        b_idx -= 1;
        expected_target_idx -= 1;
    }

    Error::Ok
}

#[must_use]
pub fn get_broadcast_target_size_tensors(
    a: &Tensor,
    b: &Tensor,
    out_sizes: *mut SizesType,
    out_sizes_len: usize,
    out_dim: *mut usize,
) -> Error {
    get_broadcast_target_size(a.sizes(), b.sizes(), out_sizes, out_sizes_len, out_dim)
}

// [spec:et:def:broadcast-util.torch.executor.resize-to-broadcast-target-size-fn]
// [spec:et:sem:broadcast-util.torch.executor.resize-to-broadcast-target-size-fn]
//
// PORT-NOTE: two-input inline overload; not separately annotated in C++.
#[must_use]
pub fn resize_to_broadcast_target_size(a: &Tensor, b: &Tensor, out: &Tensor) -> Error {
    let mut expected_output_size: [SizesType; K_TENSOR_DIMENSION_LIMIT] =
        [0; K_TENSOR_DIMENSION_LIMIT];
    let mut expected_output_dim: usize = 0;

    crate::et_check_ok_or_return_error!(
        get_broadcast_target_size_tensors(
            a,
            b,
            expected_output_size.as_mut_ptr(),
            K_TENSOR_DIMENSION_LIMIT,
            &mut expected_output_dim,
        ),
        "Failed to get broadcast target size"
    );

    resize_tensor_same_type(
        out,
        ArrayRef::from_raw_parts(expected_output_size.as_ptr(), expected_output_dim),
    )
}

// [spec:et:def:broadcast-util.torch.executor.resize-to-broadcast-target-size-fn]
// [spec:et:sem:broadcast-util.torch.executor.resize-to-broadcast-target-size-fn]
#[must_use]
pub fn resize_to_broadcast_target_size_3(
    a: &Tensor,
    b: &Tensor,
    c: &Tensor,
    out: &Tensor,
) -> Error {
    let mut interim_output_size: [SizesType; K_TENSOR_DIMENSION_LIMIT] =
        [0; K_TENSOR_DIMENSION_LIMIT];
    let mut interim_output_dim: usize = 0;

    // Obtain the broadcast size of the first two input tensors
    crate::et_check_ok_or_return_error!(
        get_broadcast_target_size_tensors(
            a,
            b,
            interim_output_size.as_mut_ptr(),
            K_TENSOR_DIMENSION_LIMIT,
            &mut interim_output_dim,
        ),
        "Failed to get broadcast target size"
    );

    let mut expected_output_size: [SizesType; K_TENSOR_DIMENSION_LIMIT] =
        [0; K_TENSOR_DIMENSION_LIMIT];
    let mut expected_output_dim: usize = 0;

    // Apply broadcasting to the intermediate broadcast size and the third input
    // tensor
    crate::et_check_ok_or_return_error!(
        get_broadcast_target_size(
            ArrayRef::from_raw_parts(interim_output_size.as_ptr(), interim_output_dim),
            c.sizes(),
            expected_output_size.as_mut_ptr(),
            K_TENSOR_DIMENSION_LIMIT,
            &mut expected_output_dim,
        ),
        "Failed to get broadcast target size"
    );

    resize_tensor_same_type(
        out,
        ArrayRef::from_raw_parts(expected_output_size.as_ptr(), expected_output_dim),
    )
}

// [spec:et:def:broadcast-util.torch.executor.linearize-access-indexes-fn]
// [spec:et:sem:broadcast-util.torch.executor.linearize-access-indexes-fn]
pub fn linearize_access_indexes(
    indexes_broadcast_to: ArrayRef<usize>,
    broadcast_to_ndim: ssize_t,
    broadcast_from_shape: ArrayRef<SizesType>,
    broadcast_from_strides: ArrayRef<StridesType>,
) -> usize {
    let num_skip_dims: usize = (broadcast_to_ndim as usize) - broadcast_from_shape.size();
    let indexes_broadcast_from =
        indexes_broadcast_to.slice(num_skip_dims, broadcast_to_ndim as usize - num_skip_dims);

    et_check!(indexes_broadcast_from.size() == broadcast_from_shape.size());

    let mut linear_index: usize = 0;
    for i in 0..indexes_broadcast_from.size() {
        // If this dimension is broadcasted, add zero to the linear address.
        if *indexes_broadcast_from.at(i) >= *broadcast_from_shape.at(i) as usize {
            et_check_msg!(
                *broadcast_from_shape.at(i) == 1,
                "Expected dim size == 1 if broadcasted, but actual dim size is {}",
                *broadcast_from_shape.at(i) as usize
            );
            continue;
        }
        linear_index += *indexes_broadcast_from.at(i) * *broadcast_from_strides.at(i) as usize;
    }
    linear_index
}

pub fn linearize_access_indexes_tensor(
    indexes_broadcast_to: ArrayRef<usize>,
    broadcast_to_ndim: ssize_t,
    broadcast_from: &Tensor,
) -> usize {
    linearize_access_indexes(
        indexes_broadcast_to,
        broadcast_to_ndim,
        broadcast_from.sizes(),
        broadcast_from.strides(),
    )
}

//
// Mapping with broadcasting
//

/// Useful for binary elementwise operators. For each element of the inputs,
/// perform a computation and write to the corresponding element of the output.
/// Tensor broadcasting is applied wherever it is required.
// [spec:et:def:broadcast-util.torch.executor.apply-binary-elementwise-fn-fn]
// [spec:et:sem:broadcast-util.torch.executor.apply-binary-elementwise-fn-fn]
pub fn apply_binary_elementwise_fn<CTYPE_A, CTYPE_B, CTYPE_OUT, Op>(
    compute_fun: Op,
    a: &Tensor,
    b: &Tensor,
    out: &Tensor,
) where
    CTYPE_A: Copy,
    CTYPE_B: Copy,
    Op: Fn(CTYPE_A, CTYPE_B) -> CTYPE_OUT,
{
    let data_a = a.const_data_ptr::<CTYPE_A>();
    let data_b = b.const_data_ptr::<CTYPE_B>();
    let data_out = out.mutable_data_ptr::<CTYPE_OUT>();

    for indexes in BroadcastIndexesRange::<3>::new(out, &[a, b]) {
        let out_index = indexes[0];
        let a_index = indexes[1];
        let b_index = indexes[2];
        unsafe {
            *data_out.offset(out_index as isize) = compute_fun(
                *data_a.offset(a_index as isize),
                *data_b.offset(b_index as isize),
            );
        }
    }
}

/// Useful for ternary elementwise operators. For each element of the inputs,
/// perform a computation and write to the corresponding element of the output.
/// Tensor broadcasting is applied wherever it is required.
// [spec:et:def:broadcast-util.torch.executor.apply-ternary-elementwise-fn-fn]
// [spec:et:sem:broadcast-util.torch.executor.apply-ternary-elementwise-fn-fn]
pub fn apply_ternary_elementwise_fn<CTYPE_A, CTYPE_B, CTYPE_C, CTYPE_OUT, Op>(
    compute_fun: Op,
    a: &Tensor,
    b: &Tensor,
    c: &Tensor,
    out: &Tensor,
) where
    CTYPE_A: Copy,
    CTYPE_B: Copy,
    CTYPE_C: Copy,
    Op: Fn(CTYPE_A, CTYPE_B, CTYPE_C) -> CTYPE_OUT,
{
    let data_a = a.const_data_ptr::<CTYPE_A>();
    let data_b = b.const_data_ptr::<CTYPE_B>();
    let data_c = c.const_data_ptr::<CTYPE_C>();
    let data_out = out.mutable_data_ptr::<CTYPE_OUT>();

    for indexes in BroadcastIndexesRange::<4>::new(out, &[a, b, c]) {
        let out_index = indexes[0];
        let a_index = indexes[1];
        let b_index = indexes[2];
        let c_index = indexes[3];
        unsafe {
            *data_out.offset(out_index as isize) = compute_fun(
                *data_a.offset(a_index as isize),
                *data_b.offset(b_index as isize),
                *data_c.offset(c_index as isize),
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assert_tensor_data_eq;
    use crate::kernels::portable::cpu::util::delinearize_index::delinearize_index_tensor;
    use crate::runtime::core::exec_aten::testing_util::tensor_factory::TensorFactory;
    use crate::runtime::core::exec_aten::util::tensor_util::K_TENSOR_DIMENSION_LIMIT;

    // broadcast_tensor calls make_tensor to allocate the output tensor's
    // sizes/dim_order/strides/data buffers; the shape+data assertions below would
    // fail if make_tensor set them up wrong.
    // [spec:et:sem:broadcast-util.torch.executor.broadcast-tensor-fn/test]
    // [spec:et:sem:broadcast-util.torch.executor.free-broadcast-tensor-fn/test]
    // [spec:et:sem:broadcast-util.torch.executor.make-tensor-fn/test]
    #[test]
    fn broadcast_util_test_broadcast_tensor() {
        let tf = TensorFactory::<i32>::new();

        let a = tf.make_default(vec![1], vec![2]);
        let b = tf.make_default(vec![2, 2], vec![2, 2, 2, 2]);
        let c = tf.zeros_default(vec![2, 2]);

        let d = broadcast_tensor(&a, &c);
        assert_tensor_data_eq!(d, tf.make_default(vec![2, 2], vec![2, 2, 2, 2]));
        free_broadcast_tensor(&d);

        let d = broadcast_tensor(&b, &c);
        assert_tensor_data_eq!(d, tf.make_default(vec![2, 2], vec![2, 2, 2, 2]));
        free_broadcast_tensor(&d);
    }

    // [spec:et:sem:broadcast-util.torch.executor.tensors-are-broadcastable-between-fn/test]
    #[test]
    fn broadcast_util_test_broadcastable_between() {
        let tf = TensorFactory::<i32>::new();

        let tensor_list = [
            tf.zeros_default(vec![1, 2]),
            tf.zeros_default(vec![2, 1]),
            tf.zeros_default(vec![1]),
            tf.zeros_default(vec![2, 2]),
        ];

        for i in 0..4 {
            for j in (i + 1)..4 {
                assert!(tensors_are_broadcastable_between_tensors(
                    &tensor_list[i],
                    &tensor_list[j]
                ));
            }
        }
    }

    // [spec:et:sem:broadcast-util.torch.executor.tensor-is-broadcastable-to-fn/test]
    // [spec:et:sem:broadcast-util.torch.executor.broadcast-tensor-fn/test]
    #[test]
    fn broadcast_util_test_broadcastable_to_from() {
        let tf = TensorFactory::<i32>::new();

        let a = tf.make_default(vec![1, 2], vec![2, 2]);
        let b = tf.make_default(vec![2, 1], vec![2, 2]);
        let c = tf.zeros_default(vec![2, 2]);

        assert!(tensor_is_broadcastable_to_tensors(&a, &c));
        let d = broadcast_tensor(&a, &c);
        assert_tensor_data_eq!(d, tf.make_default(vec![2, 2], vec![2, 2, 2, 2]));
        free_broadcast_tensor(&d);

        assert!(tensor_is_broadcastable_to_tensors(&b, &c));
        let d = broadcast_tensor(&b, &c);
        assert_tensor_data_eq!(d, tf.make_default(vec![2, 2], vec![2, 2, 2, 2]));
        free_broadcast_tensor(&d);
    }

    // [spec:et:sem:broadcast-util.torch.executor.tensor-is-broadcastable-to-fn/test]
    #[test]
    fn broadcast_util_test_not_broadcastable_to() {
        let tf = TensorFactory::<i32>::new();

        // Tensor a is broadcastable to tensor b means when tracing their sizes
        // from back to front, each pair of corresponding dimensions should meet
        // one of: (1) equal, (2) a's dim is 1, (3) one dim does not exist.
        let a = tf.make_default(vec![3], vec![2, 2, 2]);
        let b = tf.zeros_default(vec![2, 1]);
        let c = tf.zeros_default(vec![1, 2]);

        assert!(!tensor_is_broadcastable_to_tensors(&a, &b));
        // ET_EXPECT_DEATH(broadcast_tensor(a, b)) -> see
        // broadcast_util_test_not_broadcastable_to_death_ab.

        // Can not broadcast from b to c, though they are broadcastable.
        assert!(!tensor_is_broadcastable_to_tensors(&b, &c));
        // ET_EXPECT_DEATH(broadcast_tensor(b, c)) -> see
        // broadcast_util_test_not_broadcastable_to_death_bc.
    }

    // PORT-NOTE: `ET_EXPECT_DEATH` death test. `runtime_abort` calls
    // `libc::abort()`, which terminates the process rather than unwinding, so
    // `#[should_panic]` cannot catch it; ported and `#[ignore]`d per the
    // established convention (see kernels/quantized/cpu/op_add.rs).
    // [spec:et:sem:broadcast-util.torch.executor.broadcast-tensor-fn/test]
    #[test]
    #[should_panic]
    #[ignore]
    fn broadcast_util_test_not_broadcastable_to_death_ab() {
        let tf = TensorFactory::<i32>::new();
        let a = tf.make_default(vec![3], vec![2, 2, 2]);
        let b = tf.zeros_default(vec![2, 1]);
        let _ = broadcast_tensor(&a, &b);
    }

    // PORT-NOTE: death test; see note above.
    // [spec:et:sem:broadcast-util.torch.executor.broadcast-tensor-fn/test]
    #[test]
    #[should_panic]
    #[ignore]
    fn broadcast_util_test_not_broadcastable_to_death_bc() {
        let tf = TensorFactory::<i32>::new();
        let b = tf.zeros_default(vec![2, 1]);
        let c = tf.zeros_default(vec![1, 2]);
        let _ = broadcast_tensor(&b, &c);
    }

    // [spec:et:sem:broadcast-util.torch.executor.tensor-is-broadcastable-to-fn/test]
    #[test]
    fn broadcast_util_test_not_broadcastable_between() {
        let tf = TensorFactory::<i32>::new();

        let a = tf.make_default(vec![3], vec![2, 2, 2]);
        let b = tf.zeros_default(vec![2, 1]);

        assert!(!tensor_is_broadcastable_to_tensors(&a, &b));
    }

    // [spec:et:sem:broadcast-util.torch.executor.get-broadcast-target-size-fn/test]
    #[test]
    fn broadcast_util_test_get_broadcast_target_size() {
        let tf = TensorFactory::<i32>::new();
        let mut expected_output_size: [SizesType; K_TENSOR_DIMENSION_LIMIT] =
            [0; K_TENSOR_DIMENSION_LIMIT];
        let mut expected_output_dim: usize = 0;

        let a = tf.zeros_default(vec![2, 1]);
        let b = tf.zeros_default(vec![5, 1, 2]);

        let err = get_broadcast_target_size_tensors(
            &a,
            &b,
            expected_output_size.as_mut_ptr(),
            K_TENSOR_DIMENSION_LIMIT,
            &mut expected_output_dim,
        );
        assert_eq!(err, Error::Ok);

        let expected = [5, 2, 2];
        assert!(
            ArrayRef::from_raw_parts(expected_output_size.as_ptr(), expected_output_dim)
                .equals(ArrayRef::from_raw_parts(expected.as_ptr(), expected.len()))
        );

        let c = tf.zeros_default(vec![4, 5]);
        crate::runtime::platform::platform::pal_init();
        let err = get_broadcast_target_size_tensors(
            &a,
            &c,
            expected_output_size.as_mut_ptr(),
            K_TENSOR_DIMENSION_LIMIT,
            &mut expected_output_dim,
        );
        assert_eq!(err, Error::InvalidArgument);
    }

    // Test-local helper mirroring the C++ `linearize_indexes`.
    fn linearize_indexes(indexes: &[usize], indexes_len: usize, t: &Tensor) -> usize {
        let mut linear_index: usize = 0;
        let mut acc_loop_counts: usize = 1;
        let mut i = indexes_len as isize - 1;
        while i >= 0 {
            linear_index += indexes[i as usize] * acc_loop_counts;
            acc_loop_counts *= *t.sizes().at(i as usize) as usize;
            i -= 1;
        }
        linear_index
    }

    // [spec:et:sem:delinearize-index.torch.executor.delinearize-index-fn/test]
    #[test]
    fn broadcast_util_test_delinearize_index() {
        let tf = TensorFactory::<i32>::new();

        const DIMS: usize = 3;
        let t = tf.zeros_default(vec![4, 3, 5]);
        let sizes = t.sizes();

        for i0 in 0..(*sizes.at(0) as usize) {
            for i1 in 0..(*sizes.at(1) as usize) {
                for i2 in 0..(*sizes.at(2) as usize) {
                    let indexes = [i0, i1, i2];
                    let linear_index = linearize_indexes(&indexes, DIMS, &t);

                    let mut out_indexes = [0usize; DIMS];
                    delinearize_index_tensor(linear_index, &t, out_indexes.as_mut_ptr(), DIMS);

                    assert_eq!(linear_index, linearize_indexes(&out_indexes, DIMS, &t));
                }
            }
        }
    }

    // PORT-NOTE: `apply_ternary_elementwise_fn` has no in-port caller yet, so this
    // is a focused unit test written directly against the sem rule (three-input
    // broadcasting elementwise map). It broadcasts a full [2,2] `a`, a row [2] `b`
    // and a scalar [1] `c`, and checks the output equals `a*b + c` per element,
    // which pins that each input index is resolved through its own broadcast.
    // [spec:et:sem:broadcast-util.torch.executor.apply-ternary-elementwise-fn-fn/test]
    #[test]
    fn broadcast_util_test_apply_ternary_elementwise_fn() {
        let tf = TensorFactory::<i32>::new();

        let a = tf.make_default(vec![2, 2], vec![1, 2, 3, 4]);
        let b = tf.make_default(vec![2], vec![10, 20]);
        let c = tf.make_default(vec![1], vec![100]);
        let out = tf.zeros_default(vec![2, 2]);

        apply_ternary_elementwise_fn::<i32, i32, i32, i32, _>(
            |x, y, z| x * y + z,
            &a,
            &b,
            &c,
            &out,
        );

        // a*b broadcasts b=[10,20] across rows; +c=100.
        assert_tensor_data_eq!(out, tf.make_default(vec![2, 2], vec![110, 140, 130, 180]));
    }

    // [spec:et:sem:broadcast-util.torch.executor.linearize-access-indexes-fn/test]
    #[test]
    fn broadcast_util_test_linearize_index() {
        let tf = TensorFactory::<i32>::new();

        let broadcast_from = tf.zeros_default(vec![2, 1, 3, 1]);
        let broadcast_to = tf.zeros_default(vec![2, 2, 3, 4]);

        // The linear index for broadcast_from should be the same in the
        // broadcasted dimension.
        for i in 0..3 {
            let test_indexes: [usize; 4] = [0, 0, 0, i];
            let linear_index = linearize_access_indexes_tensor(
                ArrayRef::from_raw_parts(test_indexes.as_ptr(), test_indexes.len()),
                broadcast_to.dim(),
                &broadcast_from,
            );
            assert_eq!(linear_index, 0);
        }

        // The linear index for broadcast_from should be the same.
        for i in 0..=2 {
            let test_indexes: [usize; 4] = [0, i, 2, 3];
            let linear_index = linearize_access_indexes_tensor(
                ArrayRef::from_raw_parts(test_indexes.as_ptr(), test_indexes.len()),
                broadcast_to.dim(),
                &broadcast_from,
            );
            assert_eq!(linear_index, 2);
        }
    }

    // PORT-NOTE: no dedicated C++ test; focused unit test against the sem rule.
    // Covers both the two-input overload and the three-input overload (which
    // folds a x b first, then the interim size with c), plus error propagation
    // for non-broadcastable inputs.
    // [spec:et:sem:broadcast-util.torch.executor.resize-to-broadcast-target-size-fn/test]
    #[test]
    fn broadcast_util_test_resize_to_broadcast_target_size() {
        use crate::runtime::core::tensor_shape_dynamism::TensorShapeDynamism;
        let tf = TensorFactory::<i32>::new();
        crate::runtime::platform::platform::pal_init();

        let expected: [SizesType; 3] = [5, 2, 2];

        // Two-input overload: [2,1] x [5,1,2] broadcasts to [5,2,2]; out starts
        // at a different (equal-numel, same-rank) shape and must be resized.
        let a = tf.zeros_default(vec![2, 1]);
        let b = tf.zeros_default(vec![5, 1, 2]);
        let out = tf.zeros(vec![2, 2, 5], TensorShapeDynamism::DYNAMIC_BOUND);
        assert_eq!(resize_to_broadcast_target_size(&a, &b, &out), Error::Ok);
        assert!(
            out.sizes()
                .equals(ArrayRef::from_raw_parts(expected.as_ptr(), expected.len()))
        );

        // Three-input overload: ([2,1] x [1,2]) -> [2,2], then x [5,1,1] ->
        // [5,2,2].
        let c = tf.zeros_default(vec![1, 2]);
        let d = tf.zeros_default(vec![5, 1, 1]);
        let out3 = tf.zeros(vec![2, 2, 5], TensorShapeDynamism::DYNAMIC_BOUND);
        assert_eq!(
            resize_to_broadcast_target_size_3(&a, &c, &d, &out3),
            Error::Ok
        );
        assert!(
            out3.sizes()
                .equals(ArrayRef::from_raw_parts(expected.as_ptr(), expected.len()))
        );

        // Non-broadcastable inputs propagate the get_broadcast_target_size error.
        let e = tf.zeros_default(vec![4, 5]);
        assert_ne!(resize_to_broadcast_target_size(&a, &e, &out), Error::Ok);
        assert_ne!(
            resize_to_broadcast_target_size_3(&a, &e, &d, &out3),
            Error::Ok
        );
    }
}
