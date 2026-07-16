//! Literal port of runtime/core/exec_aten/util/tensor_util_portable.cpp.
//!
//! Implementation for ExecuTorch tensor util, should only be included in
//! an target with ATen mode turned off. Explicitly taking
//! torch::executor::Tensor (instead of executorch::aten::Tensor) to make sure it
//! fails at compile time if built incorrectly.

use crate::runtime::core::array_ref::ArrayRef;
use crate::runtime::core::error::Error;
use crate::runtime::core::exec_aten::util::dim_order_util::{
    is_channels_last_dim_order, is_contiguous_dim_order, validate_dim_order,
};
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::core::portable_type::tensor_impl::{DimOrderType, SizesType, TensorImpl};

// PORT-NOTE: local re-implementation of `ET_CHECK_OR_RETURN_FALSE` that forwards
// the caller's format args. See the same note in tensor_util.rs: the crate-level
// `et_check_or_return_false!` drops all args after the leading literal.
macro_rules! et_check_or_return_false {
    ($cond:expr, $fmt:literal $(, $($arg:tt)*)?) => {{
        if !($cond) {
            $crate::et_log!(
                Error,
                ::core::concat!("Check failed ({}): ", $fmt),
                ::core::stringify!($cond)
                $(, $($arg)*)?
            );
            return false;
        }
    }};
}

/// Get dim_order of a Tensor and write it to out_dim_order.
///
/// # Safety
/// `out_dim_order` must point to at least `out_dim_order_size` valid
/// `DimOrderType` elements.
// [spec:et:def:tensor-util-portable.executorch.runtime.get-dim-order-fn]
// [spec:et:sem:tensor-util-portable.executorch.runtime.get-dim-order-fn]
#[must_use]
pub unsafe fn get_dim_order(
    tensor: &Tensor,
    out_dim_order: *mut DimOrderType,
    out_dim_order_size: usize,
) -> Error {
    crate::et_check_or_return_error!(
        out_dim_order_size == tensor.dim_order().size(),
        InvalidArgument,
        "Size needs to be equal to the number of dimensions of the tensor size {}, tensor.dim() {}",
        out_dim_order_size,
        tensor.dim_order().size()
    );
    unsafe {
        core::ptr::copy_nonoverlapping(
            tensor.dim_order().data(),
            out_dim_order,
            tensor.dim_order().size(),
        );
    }
    Error::Ok
}

// [spec:et:def:tensor-util-portable.executorch.runtime.tensor-has-valid-dim-order-fn]
// [spec:et:sem:tensor-util-portable.executorch.runtime.tensor-has-valid-dim-order-fn]
pub fn tensor_has_valid_dim_order(t: &Tensor) -> bool {
    if !unsafe { validate_dim_order(t.dim_order().data(), t.dim_order().size()) } {
        crate::et_log!(Error, "Tensor dim order is not valid:");
        for d in 0..t.dim() {
            crate::et_log!(Error, "    dim_order({}): {}", d as usize, unsafe {
                *t.dim_order().data().add(d as usize)
            }
                as usize);
        }
        return false;
    }
    true
}

// [spec:et:def:tensor-util-portable.executorch.runtime.tensor-is-default-or-channels-last-dim-order-fn]
// [spec:et:sem:tensor-util-portable.executorch.runtime.tensor-is-default-or-channels-last-dim-order-fn]
pub fn tensor_is_default_or_channels_last_dim_order(t: &Tensor) -> bool {
    let ret_val: bool =
        unsafe { is_contiguous_dim_order(t.dim_order().data(), t.dim_order().size()) }
            || unsafe { is_channels_last_dim_order(t.dim_order().data(), t.dim_order().size()) };

    if !ret_val {
        crate::et_log!(
            Error,
            "Expected tensor to have default or channels last dim order, but got"
        );
        for d in 0..t.dim() {
            crate::et_log!(Error, "    dim_order({}): {}", d as usize, unsafe {
                *t.dim_order().data().add(d as usize)
            }
                as usize);
        }
    }
    ret_val
}

// [spec:et:def:tensor-util-portable.executorch.runtime.tensor-is-default-dim-order-fn]
// [spec:et:sem:tensor-util-portable.executorch.runtime.tensor-is-default-dim-order-fn]
pub fn tensor_is_default_dim_order(t: &Tensor) -> bool {
    let ret_val: bool =
        unsafe { is_contiguous_dim_order(t.dim_order().data(), t.dim_order().size()) };

    if !ret_val {
        crate::et_log!(Error, "Expected tensor to have default dim order, but got");
        for d in 0..t.dim() {
            crate::et_log!(Error, "    dim_order({}): {}", d as usize, unsafe {
                *t.dim_order().data().add(d as usize)
            }
                as usize);
        }
    }
    ret_val
}

// [spec:et:def:tensor-util-portable.executorch.runtime.tensor-is-channels-last-dim-order-fn]
// [spec:et:sem:tensor-util-portable.executorch.runtime.tensor-is-channels-last-dim-order-fn]
pub fn tensor_is_channels_last_dim_order(t: &Tensor) -> bool {
    let ret_val: bool =
        unsafe { is_channels_last_dim_order(t.dim_order().data(), t.dim_order().size()) };

    if !ret_val {
        crate::et_log!(
            Error,
            "Expected tensor to have channels last dim order, but got"
        );
        for d in 0..t.dim() {
            crate::et_log!(Error, "    dim_order({}): {}", d as usize, unsafe {
                *t.dim_order().data().add(d as usize)
            }
                as usize);
        }
    }
    ret_val
}

// [spec:et:def:tensor-util-portable.executorch.runtime.tensors-have-same-dim-order-fn]
// [spec:et:sem:tensor-util-portable.executorch.runtime.tensors-have-same-dim-order-fn]
pub fn tensors_have_same_dim_order(tensor_list: ArrayRef<Tensor>) -> bool {
    if tensor_list.size() < 2 {
        return true;
    }
    let mut all_contiguous: bool = true;
    let mut all_channels_last: bool = true;
    for i in 0..tensor_list.size() {
        all_contiguous = all_contiguous
            && unsafe {
                is_contiguous_dim_order(
                    tensor_list.at(i).dim_order().data(),
                    tensor_list.at(i).dim_order().size(),
                )
            };
        all_channels_last = all_channels_last
            && unsafe {
                is_channels_last_dim_order(
                    tensor_list.at(i).dim_order().data(),
                    tensor_list.at(i).dim_order().size(),
                )
            };
    }

    et_check_or_return_false!(
        all_contiguous || all_channels_last,
        "{} input tensors have different dim orders",
        tensor_list.size()
    );

    true
}

pub mod internal {
    use super::*;

    // [spec:et:def:tensor-util-portable.executorch.runtime.internal.share-tensor-data-fn]
    // [spec:et:sem:tensor-util-portable.executorch.runtime.internal.share-tensor-data-fn]
    pub fn share_tensor_data(t_dst: &Tensor, t_src: &Tensor) -> Error {
        crate::et_check_or_return_error!(
            t_dst.nbytes() == t_src.nbytes(),
            InvalidArgument,
            "t_dst.nbytes() {} != t_src.nbytes(). {}",
            t_dst.nbytes(),
            t_src.nbytes()
        );

        // Either the t_src is empty or contains valid data.
        crate::et_check_or_return_error!(
            !t_src.mutable_data_ptr_typed().is_null() || t_src.nbytes() == 0,
            InvalidArgument,
            "Source tensor should have data_ptr not being nullptr."
        );

        // Setting data_ptr to nullptr explicitly when t_src is empty.
        let t_src_data_ptr: *mut core::ffi::c_void = if t_src.numel() == 0 {
            core::ptr::null_mut()
        } else {
            t_src.mutable_data_ptr_typed()
        };
        // Assign internal data_ptr as the one in forwarded tensor
        unsafe { (*t_dst.unsafe_get_tensor_impl()).set_data(t_src_data_ptr) };

        Error::Ok
    }

    // [spec:et:def:tensor-util-portable.executorch.runtime.internal.copy-tensor-data-fn]
    // [spec:et:sem:tensor-util-portable.executorch.runtime.internal.copy-tensor-data-fn]
    pub fn copy_tensor_data(t_dst: &Tensor, t_src: &Tensor) -> Error {
        crate::et_check_or_return_error!(
            !t_dst.const_data_ptr_typed().is_null() || (t_dst.nbytes() == 0 && t_src.nbytes() == 0),
            InvalidArgument,
            "ExecutionPlan input supposed to preallocated but has nullptr for data"
        );
        // inputs with a size 0 dimension can be nullptr
        if !t_src.const_data_ptr_typed().is_null() {
            crate::et_check_or_return_error!(
                t_dst.nbytes() == t_src.nbytes(),
                InvalidArgument,
                "t_dst.nbytes() {} != t_src.nbytes(). {}",
                t_dst.nbytes(),
                t_src.nbytes()
            );
            unsafe {
                core::ptr::copy_nonoverlapping(
                    t_src.const_data_ptr_typed() as *const u8,
                    t_dst.mutable_data_ptr_typed() as *mut u8,
                    t_src.nbytes(),
                );
            }
        }
        Error::Ok
    }

    // [spec:et:def:tensor-util-portable.executorch.runtime.internal.set-tensor-data-fn]
    // [spec:et:sem:tensor-util-portable.executorch.runtime.internal.set-tensor-data-fn]
    #[must_use]
    pub fn set_tensor_data(
        t: &Tensor,
        buffer: *mut core::ffi::c_void,
        buffer_size: usize,
    ) -> Error {
        crate::et_check_or_return_error!(
            buffer_size >= t.nbytes(),
            InvalidArgument,
            "buffer_size {} is smaller than smaller than tensor nbytes {}",
            buffer_size,
            t.nbytes()
        );
        unsafe { (*t.unsafe_get_tensor_impl()).set_data(buffer) };
        Error::Ok
    }

    // [spec:et:def:tensor-util-portable.executorch.runtime.internal.reset-data-ptr-fn]
    // [spec:et:sem:tensor-util-portable.executorch.runtime.internal.reset-data-ptr-fn]
    pub fn reset_data_ptr(tensor: &Tensor) {
        // Lean mode doesn't deallocate the tensor data_ptr in the allocator
        unsafe { (*tensor.unsafe_get_tensor_impl()).set_data(core::ptr::null_mut()) };
    }

    // [spec:et:def:tensor-util-portable.executorch.runtime.internal.tensor-resizer-friend]
    // PORT-NOTE: `TensorResizerFriend` is a C++ friend class of `TensorImpl`
    // used only to reach the private `internal_resize_contiguous` entry point.
    // In the Rust port `internal_resize_contiguous` is already a public method on
    // `TensorImpl`, so this struct exists purely to preserve the symbol shape;
    // it forwards directly. Construct deviation (no `friend` in Rust).
    pub struct TensorResizerFriend;

    impl TensorResizerFriend {
        // [spec:et:def:tensor-util-portable.executorch.runtime.internal.tensor-resizer-friend.resize-tensor-impl-fn]
        // [spec:et:sem:tensor-util-portable.executorch.runtime.internal.tensor-resizer-friend.resize-tensor-impl-fn]
        #[must_use]
        pub fn resize_tensor_impl(impl_: *mut TensorImpl, new_sizes: ArrayRef<SizesType>) -> Error {
            unsafe { (*impl_).internal_resize_contiguous(new_sizes) }
        }
    }

    // [spec:et:def:tensor-util-portable.executorch.runtime.internal.resize-tensor-impl-fn]
    // [spec:et:sem:tensor-util-portable.executorch.runtime.internal.resize-tensor-impl-fn]
    #[must_use]
    pub fn resize_tensor_impl(impl_: *mut TensorImpl, new_sizes: ArrayRef<SizesType>) -> Error {
        TensorResizerFriend::resize_tensor_impl(impl_, new_sizes)
    }
}

// C++ has no dedicated portable-only test file for these; the dim-order
// predicates and the `internal::*` data-pointer/resize helpers are pinned here
// against the sem rules (dim_order copy-out, contiguous/channels-last
// classification, and the aliasing/copy/reset/resize semantics on TensorImpl).
#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::core::exec_aten::testing_util::tensor_factory::TensorFactory;
    use crate::runtime::core::tensor_shape_dynamism::TensorShapeDynamism;

    fn setup() {
        crate::runtime::platform::runtime::runtime_init();
    }

    // [spec:et:sem:tensor-util-portable.executorch.runtime.get-dim-order-fn/test]
    #[test]
    fn get_dim_order_default_and_channels_last() {
        setup();
        let tf = TensorFactory::<f32>::new();

        // Default (contiguous) 4D: dim order is 0,1,2,3.
        let a = tf.ones_default(vec![2, 3, 4, 5]);
        let mut out = [0u8; 4];
        assert_eq!(
            unsafe { get_dim_order(&a, out.as_mut_ptr(), out.len()) },
            Error::Ok
        );
        assert_eq!(out, [0, 1, 2, 3]);

        // Channels-last 4D: dim order is 0,2,3,1.
        let b = tf.full_channels_last(vec![2, 3, 4, 5], 1.0, TensorShapeDynamism::STATIC);
        let mut outb = [0u8; 4];
        assert_eq!(
            unsafe { get_dim_order(&b, outb.as_mut_ptr(), outb.len()) },
            Error::Ok
        );
        assert_eq!(outb, [0, 2, 3, 1]);

        // Wrong out size is rejected without writing.
        let mut wrong = [0u8; 3];
        assert_eq!(
            unsafe { get_dim_order(&a, wrong.as_mut_ptr(), wrong.len()) },
            Error::InvalidArgument
        );
    }

    // [spec:et:sem:tensor-util-portable.executorch.runtime.tensor-has-valid-dim-order-fn/test]
    #[test]
    fn tensor_has_valid_dim_order_true_for_permutations() {
        setup();
        let tf = TensorFactory::<f32>::new();
        let a = tf.ones_default(vec![2, 3, 4, 5]);
        assert!(tensor_has_valid_dim_order(&a));
        let b = tf.full_channels_last(vec![2, 3, 4, 5], 1.0, TensorShapeDynamism::STATIC);
        assert!(tensor_has_valid_dim_order(&b));
        // Any permutation is a valid dim order.
        let p = tf.make_with_dimorder(
            vec![2, 3, 4],
            vec![0.0; 24],
            vec![1, 2, 0],
            TensorShapeDynamism::STATIC,
        );
        assert!(tensor_has_valid_dim_order(&p));
    }

    // [spec:et:sem:tensor-util-portable.executorch.runtime.tensor-is-default-dim-order-fn/test]
    #[test]
    fn tensor_is_default_dim_order_classifies() {
        setup();
        let tf = TensorFactory::<f32>::new();
        let default_t = tf.ones_default(vec![2, 3, 4, 5]);
        assert!(tensor_is_default_dim_order(&default_t));

        let cl = tf.full_channels_last(vec![2, 3, 4, 5], 1.0, TensorShapeDynamism::STATIC);
        assert!(!tensor_is_default_dim_order(&cl));

        let permuted = tf.make_with_dimorder(
            vec![2, 3, 4],
            vec![0.0; 24],
            vec![1, 2, 0],
            TensorShapeDynamism::STATIC,
        );
        assert!(!tensor_is_default_dim_order(&permuted));
    }

    // [spec:et:sem:tensor-util-portable.executorch.runtime.tensor-is-channels-last-dim-order-fn/test]
    #[test]
    fn tensor_is_channels_last_dim_order_classifies() {
        setup();
        let tf = TensorFactory::<f32>::new();
        let cl = tf.full_channels_last(vec![2, 3, 4, 5], 1.0, TensorShapeDynamism::STATIC);
        assert!(tensor_is_channels_last_dim_order(&cl));

        let default_t = tf.ones_default(vec![2, 3, 4, 5]);
        assert!(!tensor_is_channels_last_dim_order(&default_t));
    }

    // [spec:et:sem:tensor-util-portable.executorch.runtime.tensor-is-default-or-channels-last-dim-order-fn/test]
    #[test]
    fn tensor_is_default_or_channels_last_dim_order_classifies() {
        setup();
        let tf = TensorFactory::<f32>::new();
        let default_t = tf.ones_default(vec![2, 3, 4, 5]);
        assert!(tensor_is_default_or_channels_last_dim_order(&default_t));

        let cl = tf.full_channels_last(vec![2, 3, 4, 5], 1.0, TensorShapeDynamism::STATIC);
        assert!(tensor_is_default_or_channels_last_dim_order(&cl));

        // A generic 3D permutation is neither default nor channels-last.
        let permuted = tf.make_with_dimorder(
            vec![2, 3, 4],
            vec![0.0; 24],
            vec![1, 2, 0],
            TensorShapeDynamism::STATIC,
        );
        assert!(!tensor_is_default_or_channels_last_dim_order(&permuted));
    }

    // [spec:et:sem:tensor-util-portable.executorch.runtime.internal.set-tensor-data-fn/test]
    // [spec:et:sem:tensor-util-portable.executorch.runtime.internal.reset-data-ptr-fn/test]
    #[test]
    fn internal_set_and_reset_data_ptr() {
        setup();
        let tf = TensorFactory::<u8>::new();
        let t = tf.zeros_default(vec![2, 3]); // nbytes == 6
        let mut buffer = [7u8; 8];

        // Too-small buffer is rejected.
        assert_eq!(
            internal::set_tensor_data(&t, buffer.as_mut_ptr() as *mut core::ffi::c_void, 3),
            Error::InvalidArgument
        );

        // A buffer >= nbytes is installed as the data pointer.
        assert_eq!(
            internal::set_tensor_data(&t, buffer.as_mut_ptr() as *mut core::ffi::c_void, 8),
            Error::Ok
        );
        assert_eq!(
            t.mutable_data_ptr_typed(),
            buffer.as_mut_ptr() as *mut core::ffi::c_void
        );

        internal::reset_data_ptr(&t);
        assert_eq!(t.mutable_data_ptr_typed(), core::ptr::null_mut());
    }

    // [spec:et:sem:tensor-util-portable.executorch.runtime.internal.share-tensor-data-fn/test]
    #[test]
    fn internal_share_tensor_data() {
        setup();
        let tf = TensorFactory::<u8>::new();
        let src = tf.make_default(vec![2, 3], vec![1, 2, 3, 4, 5, 6]);
        let dst = tf.zeros_default(vec![2, 3]);

        assert_eq!(internal::share_tensor_data(&dst, &src), Error::Ok);
        assert_eq!(dst.const_data_ptr_typed(), src.const_data_ptr_typed());

        // Mismatched nbytes is rejected.
        let src2 = tf.zeros_default(vec![2, 4]);
        assert_eq!(
            internal::share_tensor_data(&dst, &src2),
            Error::InvalidArgument
        );
    }

    // [spec:et:sem:tensor-util-portable.executorch.runtime.internal.copy-tensor-data-fn/test]
    #[test]
    fn internal_copy_tensor_data() {
        setup();
        let tf = TensorFactory::<u8>::new();
        let src = tf.make_default(vec![2, 3], vec![1, 2, 3, 4, 5, 6]);
        let dst = tf.zeros_default(vec![2, 3]);

        assert_eq!(internal::copy_tensor_data(&dst, &src), Error::Ok);
        let dp = dst.const_data_ptr_typed() as *const u8;
        for i in 0..6 {
            assert_eq!(unsafe { *dp.add(i) }, (i + 1) as u8);
        }
        // Data pointers stay distinct (copy, not share).
        assert_ne!(dst.const_data_ptr_typed(), src.const_data_ptr_typed());
    }

    // [spec:et:sem:tensor-util-portable.executorch.runtime.internal.resize-tensor-impl-fn/test]
    // [spec:et:sem:tensor-util-portable.executorch.runtime.internal.tensor-resizer-friend.resize-tensor-impl-fn/test]
    #[test]
    fn internal_resize_tensor_impl() {
        setup();
        let tf = TensorFactory::<f32>::new();
        let t = tf.zeros(vec![3, 5], TensorShapeDynamism::DYNAMIC_BOUND);
        let impl_ = t.unsafe_get_tensor_impl();

        let new_sizes: [SizesType; 2] = [2, 4];
        assert_eq!(
            internal::resize_tensor_impl(impl_, ArrayRef::from_raw_parts(new_sizes.as_ptr(), 2)),
            Error::Ok
        );
        assert_eq!(t.numel(), 8);
        assert_eq!(*t.sizes().at(0), 2);
        assert_eq!(*t.sizes().at(1), 4);

        // The friend forwarder produces identical behavior.
        let back: [SizesType; 2] = [3, 5];
        assert_eq!(
            internal::TensorResizerFriend::resize_tensor_impl(
                impl_,
                ArrayRef::from_raw_parts(back.as_ptr(), 2)
            ),
            Error::Ok
        );
        assert_eq!(t.numel(), 15);
    }
}
