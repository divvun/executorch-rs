//! Literal port of kernels/portable/cpu/op_flip.cpp.

use crate::kernels::portable::cpu::util::reduce_util::check_dim_list_is_valid;
use crate::runtime::core::array_ref::{ArrayRef, IntArrayRef};
use crate::runtime::core::error::Error;
use crate::runtime::core::exec_aten::util::tensor_util::{
    K_TENSOR_DIMENSION_LIMIT, coordinateToIndex, indexToCoordinate, nonzero_dim,
    resize_tensor_same_type, tensors_have_same_dim_order2, tensors_have_same_dtype2,
};
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

// PORT-NOTE: local `et_log_and_return_if_false!` mirroring the C++
// `ET_LOG_AND_RETURN_IF_FALSE(cond)` (the crate-level check macro drops format
// args); same reasoning as repeat_util.rs / op_roll.rs.
macro_rules! et_log_and_return_if_false {
    ($cond:expr) => {{
        if !($cond) {
            $crate::et_log!(Error, "Check failed ({}): ", ::core::stringify!($cond));
            return false;
        }
    }};
}

// PORT-NOTE: `check_dim_list_is_valid(in, dims)` takes a
// `std::optional<ArrayRef>` in C++; the `IntArrayRef dims` argument is implicitly
// wrapped into an engaged optional, so `&Some(dims)` is passed here.
// [spec:et:def:op-flip.torch.executor.native.check-flip-args-fn]
// [spec:et:sem:op-flip.torch.executor.native.check-flip-args-fn]
fn check_flip_args(in_: &Tensor, dims: IntArrayRef, out: &Tensor) -> bool {
    et_log_and_return_if_false!(tensors_have_same_dtype2(in_, out));
    check_dim_list_is_valid(in_, &Some(dims))
}

// [spec:et:def:op-flip.torch.executor.native.unflip-flat-ix-fn]
// [spec:et:sem:op-flip.torch.executor.native.unflip-flat-ix-fn]
fn unflip_flat_ix(ix: usize, in_: &Tensor, flip_dim: ArrayRef<bool>) -> usize {
    let mut ix_coord: [usize; K_TENSOR_DIMENSION_LIMIT] = [0; K_TENSOR_DIMENSION_LIMIT];
    unsafe {
        indexToCoordinate(in_, ix, ix_coord.as_mut_ptr());
    }

    let mut unflip_coord: [usize; K_TENSOR_DIMENSION_LIMIT] = [0; K_TENSOR_DIMENSION_LIMIT];
    for d in 0..in_.dim() {
        if *flip_dim.at(d as usize) {
            unflip_coord[d as usize] = in_.size(d) as usize - ix_coord[d as usize] - 1;
        } else {
            unflip_coord[d as usize] = ix_coord[d as usize];
        }
    }

    unsafe { coordinateToIndex(in_, unflip_coord.as_ptr()) }
}

// [spec:et:def:op-flip.torch.executor.native.flip-out-fn]
// [spec:et:sem:op-flip.torch.executor.native.flip-out-fn]
pub fn flip_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    in_: &Tensor,
    dims: IntArrayRef,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    crate::et_kernel_check!(
        ctx,
        resize_tensor_same_type(out, in_.sizes()) == Error::Ok,
        InvalidArgument,
        out
    );

    crate::et_kernel_check!(
        ctx,
        tensors_have_same_dim_order2(in_, out),
        InvalidArgument,
        out
    );

    crate::et_kernel_check!(ctx, check_flip_args(in_, dims, out), InvalidArgument, out);

    let mut flip_dim_data: [bool; K_TENSOR_DIMENSION_LIMIT] = [false; K_TENSOR_DIMENSION_LIMIT];
    for i in 0..in_.dim() {
        flip_dim_data[i as usize] = false;
    }
    for i in 0..dims.size() {
        let d = if *dims.at(i) < 0 {
            *dims.at(i) + nonzero_dim(in_) as i64
        } else {
            *dims.at(i)
        };
        flip_dim_data[d as usize] = true;
    }
    let flip_dim_length: usize = in_.dim() as usize;
    let flip_dim: ArrayRef<bool> =
        ArrayRef::from_raw_parts(flip_dim_data.as_ptr(), flip_dim_length);

    let op_name = "flip_out";

    crate::et_switch_realhbbf16_types!(in_.scalar_type(), ctx, op_name, CTYPE, {
        let in_data: *const CTYPE = in_.const_data_ptr::<CTYPE>();
        let out_data: *mut CTYPE = out.mutable_data_ptr::<CTYPE>();

        for ix in 0..in_.numel() {
            unsafe {
                *out_data.add(ix as usize) =
                    *in_data.add(unflip_flat_ix(ix as usize, in_, flip_dim));
            }
        }
    });

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::core::exec_aten::testing_util::tensor_factory::{
        FactoryValue, TensorFactory,
    };
    use crate::runtime::core::exec_aten::testing_util::tensor_util::{internal, tensors_are_close};
    use crate::runtime::core::exec_aten::util::scalar_type_util::CppTypeToScalarType;
    use crate::runtime::core::memory_allocator::{MemoryAllocator, MemoryAllocatorBase};
    use crate::runtime::core::portable_type::{BFloat16, Half};

    // PORT-NOTE: the C++ free function `op_flip_out` builds a fresh
    // `KernelRuntimeContext` per call (not the fixture's `context_`); mirrored here.
    fn op_flip_out<'a, 'b>(
        input: &Tensor,
        dims: IntArrayRef,
        out: &'a Tensor<'b>,
    ) -> &'a Tensor<'b> {
        crate::runtime::platform::platform::pal_init();
        let mut context = KernelRuntimeContext::new(
            crate::extension::module::module::null_event_tracer(),
            core::ptr::null_mut::<MemoryAllocator>() as *mut dyn MemoryAllocatorBase,
        );
        flip_out(&mut context, input, dims, out)
    }

    // PORT-NOTE: `static_cast<CTYPE>(int)` bridge for building integer literal data
    // in the REALHBF16 factory element types used by the 1d smoke test.
    trait FromI64: Copy {
        fn from_i64(v: i64) -> Self;
    }
    macro_rules! impl_from_i64_num {
        ($($t:ty),*) => {$(impl FromI64 for $t { fn from_i64(v: i64) -> Self { v as $t } })*};
    }
    impl_from_i64_num!(u8, i8, i16, i32, i64, f32, f64);
    impl FromI64 for Half {
        fn from_i64(v: i64) -> Self {
            Half::from_f32(v as f32)
        }
    }
    impl FromI64 for BFloat16 {
        fn from_i64(v: i64) -> Self {
            BFloat16::from_f32(v as f32)
        }
    }

    fn test_1d_dtype<T>()
    where
        T: CppTypeToScalarType + FactoryValue + FromI64,
    {
        let tf = TensorFactory::<T>::new();
        let d = |v: &[i64]| -> Vec<T> { v.iter().map(|&x| T::from_i64(x)).collect() };

        let input = tf.make_default(vec![4, 1, 3], d(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]));
        let dims_data: [i64; 1] = [-1];
        let dims = ArrayRef::from_raw_parts(dims_data.as_ptr(), 1);
        let out = tf.zeros_default(vec![4, 1, 3]);
        let out_expected =
            tf.make_default(vec![4, 1, 3], d(&[3, 2, 1, 6, 5, 4, 9, 8, 7, 12, 11, 10]));
        op_flip_out(&input, dims, &out);
        assert!(tensors_are_close(
            &out,
            &out_expected,
            internal::K_DEFAULT_RTOL,
            None
        ));
    }

    // Valid dtype/dim-list args flow through check_flip_args (returns true).
    // [spec:et:sem:op-flip.torch.executor.native.flip-out-fn/test]
    // [spec:et:sem:op-flip.torch.executor.native.check-flip-args-fn/test]
    // [spec:et:sem:op-flip.torch.executor.native.unflip-flat-ix-fn/test]
    #[test]
    fn op_flip_out_test_smoke_test_1_dim() {
        test_1d_dtype::<u8>();
        test_1d_dtype::<i8>();
        test_1d_dtype::<i16>();
        test_1d_dtype::<i32>();
        test_1d_dtype::<i64>();
        test_1d_dtype::<f32>();
        test_1d_dtype::<f64>();
        test_1d_dtype::<Half>();
        test_1d_dtype::<BFloat16>();
    }

    // Flipping two dims exercises unflip_flat_ix's multi-dim coordinate reversal;
    // a wrong index computation would reorder the output.
    // [spec:et:sem:op-flip.torch.executor.native.flip-out-fn/test]
    // [spec:et:sem:op-flip.torch.executor.native.unflip-flat-ix-fn/test]
    #[test]
    fn op_flip_out_test_smoke_test_2_dims() {
        let tf_float = TensorFactory::<f32>::new();

        let input = tf_float.make_default(
            vec![4, 1, 3],
            vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.],
        );
        let dims_data: [i64; 2] = [-1, 0];
        let dims = ArrayRef::from_raw_parts(dims_data.as_ptr(), 2);
        let out = tf_float.zeros_default(vec![4, 1, 3]);
        let out_expected = tf_float.make_default(
            vec![4, 1, 3],
            vec![12., 11., 10., 9., 8., 7., 6., 5., 4., 3., 2., 1.],
        );
        op_flip_out(&input, dims, &out);
        assert!(tensors_are_close(
            &out,
            &out_expected,
            internal::K_DEFAULT_RTOL,
            None
        ));
    }
}
