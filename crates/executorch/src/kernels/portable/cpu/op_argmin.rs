//! Literal port of kernels/portable/cpu/op_argmin.cpp.

use crate::kernels::portable::cpu::util::math_util::isnan_override;
#[cfg(not(feature = "aten"))]
use crate::kernels::portable::cpu::util::reduce_util::check_argmin_argmax_args;
use crate::kernels::portable::cpu::util::reduce_util::{
    parallel_for_each_reduce_over_dim_output_index, reduce_over_dim, resize_reduction_out_dim,
};
use crate::runtime::core::error::Error;
use crate::runtime::core::exec_aten::util::tensor_util::tensors_have_same_dim_order2;
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

// PORT-NOTE: C++ `Tensor& out` / returned `Tensor&` become `&'a Tensor` (interior
// mutation through the non-owning handle's raw pointer). `optional<int64_t> dim`
// maps to `Option<i64>`. The C++ single-dim `resize_reduction_out(in, dim, ...)`
// overload maps to the ported `resize_reduction_out_dim`.

// [spec:et:def:op-argmin.torch.executor.native.argmin-out-fn]
// [spec:et:sem:op-argmin.torch.executor.native.argmin-out-fn]
pub fn argmin_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    in_: &Tensor,
    dim: Option<i64>,
    keepdim: bool,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    // (void)ctx;

    // PORT-NOTE: `check_argmin_argmax_args` is under the C++ `#ifndef USE_ATEN_LIB`
    // block (the portable arg-checkers are absent in the ATen build), so gate to
    // match.
    #[cfg(not(feature = "aten"))]
    crate::et_kernel_check!(
        ctx,
        check_argmin_argmax_args(in_, dim, keepdim, out),
        InvalidArgument,
        out
    );

    crate::et_kernel_check!(
        ctx,
        resize_reduction_out_dim(in_, &dim, keepdim, out) == Error::Ok,
        InvalidArgument,
        out
    );

    crate::et_kernel_check!(
        ctx,
        tensors_have_same_dim_order2(in_, out),
        InvalidArgument,
        out
    );

    let op_name = "argmin.out";

    crate::et_switch_realhbf16_types!(in_.scalar_type(), ctx, op_name, CTYPE, {
        let out_data: *mut i64 = out.mutable_data_ptr::<i64>();

        let success = parallel_for_each_reduce_over_dim_output_index(
            in_,
            dim,
            out,
            &|begin: i64, end: i64| {
                for out_ix in begin..end {
                    let out_ix = out_ix as usize;
                    let acc: (CTYPE, i64) = reduce_over_dim::<CTYPE, _>(
                        |v: CTYPE, ix: i64, acc_val: CTYPE, acc_ix: i64| -> (CTYPE, i64) {
                            let mut acc_val = acc_val;
                            let mut acc_ix = acc_ix;
                            // the below condition as written is equivalent to
                            // !isnan(accval) && (isnan(v) || v < acc_val). cases:
                            // - if neither acc_val nor v is NaN, !(v >= acc_val) is
                            //   trivially equivalent to v < acc_val.
                            // - if acc_val is NaN, the whole thing is trivially false.
                            // - if acc_val is not NaN and v is NaN, then v >= acc_val
                            // - is false because all comparisons involving NaN are
                            // - false, so the result is true. The result is trivially
                            // - true for the above condition that uses isnan(v) as
                            // - well.
                            if !isnan_override(acc_val) && !(v >= acc_val) {
                                acc_val = v;
                                acc_ix = ix;
                            }
                            (acc_val, acc_ix)
                        },
                        in_,
                        &dim,
                        out_ix,
                    );
                    unsafe {
                        *out_data.add(out_ix) = acc.1;
                    }
                }
            },
        );
        crate::et_kernel_check_msg!(ctx, success, Internal, out, "parallel_for failed");
    });

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assert_tensor_eq;
    use crate::runtime::core::exec_aten::testing_util::tensor_factory::TensorFactory;
    use crate::runtime::core::memory_allocator::{MemoryAllocator, MemoryAllocatorBase};
    use crate::runtime::core::portable_type::{BFloat16, Half};

    fn context() -> KernelRuntimeContext<'static> {
        // Mirrors `OperatorTest::SetUp()`'s `runtime_init()` PAL initialization.
        crate::runtime::platform::platform::pal_init();
        KernelRuntimeContext::new(
            crate::extension::module::module::null_event_tracer(),
            core::ptr::null_mut::<MemoryAllocator>() as *mut dyn MemoryAllocatorBase,
        )
    }

    // PORT-NOTE: `test_argmin_dtype<DTYPE>` is templated over the element type and
    // dispatched via `ET_FORALL_REALHBF16_TYPES` in `SanityCheck`; each dtype is
    // expanded here as a separate helper call.
    trait FromI64 {
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

    fn test_argmin_dtype<T>()
    where
        T: crate::runtime::core::exec_aten::util::scalar_type_util::CppTypeToScalarType
            + crate::runtime::core::exec_aten::testing_util::tensor_factory::FactoryValue
            + FromI64,
    {
        let tfl = TensorFactory::<i64>::new();
        let tf_dtype = TensorFactory::<T>::new();

        let in_ = tf_dtype.make_default(
            vec![2, 3, 4],
            [
                1, 4, 1, 6, 5, 8, 5, 6, 5, 3, 9, 2, 3, 9, 1, 4, 9, 7, 5, 5, 7, 7, 6, 3,
            ]
            .iter()
            .map(|&v| T::from_i64(v))
            .collect(),
        );

        let out = tfl.zeros_default(vec![2, 4]);
        let expected = tfl.make_default(vec![2, 4], vec![0, 2, 0, 2, 0, 1, 0, 2]);
        let mut ctx = context();
        let ret = argmin_out(&mut ctx, &in_, Some(1), false, &out);

        assert_tensor_eq!(out, *ret);
        assert_tensor_eq!(out, expected);
    }

    // [spec:et:sem:op-argmin.torch.executor.native.argmin-out-fn/test]
    #[test]
    fn op_argmin_test_sanity_check() {
        test_argmin_dtype::<u8>();
        test_argmin_dtype::<i8>();
        test_argmin_dtype::<i16>();
        test_argmin_dtype::<i32>();
        test_argmin_dtype::<i64>();
        test_argmin_dtype::<f32>();
        test_argmin_dtype::<f64>();
        test_argmin_dtype::<Half>();
        test_argmin_dtype::<BFloat16>();
    }

    // [spec:et:sem:op-argmin.torch.executor.native.argmin-out-fn/test]
    #[test]
    fn op_argmin_test_sanity_check_null_dim() {
        let tf = TensorFactory::<i64>::new();

        let in_ = tf.make_default(
            vec![2, 3, 4],
            vec![
                9, 4, 1, 6, 5, 8, 5, 6, 5, 3, 9, 2, 3, 9, 1, 4, 9, 7, 5, 5, 7, 7, 6, 3,
            ],
        );

        let out = tf.zeros_default(vec![]);
        let expected = tf.make_default(vec![], vec![2]);

        let dim: Option<i64> = None;
        let mut ctx = context();
        let ret = argmin_out(&mut ctx, &in_, dim, false, &out);

        assert_tensor_eq!(out, *ret);
        assert_tensor_eq!(out, expected);
    }

    // [spec:et:sem:op-argmin.torch.executor.native.argmin-out-fn/test]
    #[test]
    fn op_argmin_test_first_nan_wins() {
        let tf_float = TensorFactory::<f32>::new();
        let in_ = tf_float.make_default(vec![4], vec![1.0, f32::NAN, -4.0, f32::NAN]);

        let tf_long = TensorFactory::<i64>::new();
        let out = tf_long.zeros_default(vec![]);
        let expected = tf_long.make_default(vec![], vec![1]);

        let mut ctx = context();
        let ret = argmin_out(&mut ctx, &in_, None, false, &out);
        assert_tensor_eq!(out, *ret);
        assert_tensor_eq!(out, expected);
    }
}
