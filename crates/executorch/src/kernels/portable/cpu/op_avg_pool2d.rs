//! Literal port of kernels/portable/cpu/op_avg_pool2d.cpp.

use crate::kernels::portable::cpu::util::dtype_util::internal::StaticCast;
use crate::kernels::portable::cpu::util::kernel_ops_util::{
    apply_kernel_2d_reduce_then_map_fn, check_avg_pool2d_args, get_avg_pool2d_out_target_size,
    output_size_is_valid,
};
use crate::runtime::core::array_ref::{ArrayRef, IntArrayRef};
use crate::runtime::core::error::Error;
use crate::runtime::core::exec_aten::util::tensor_util::{
    K_TENSOR_DIMENSION_LIMIT, resize_tensor_same_type, tensor_is_default_dim_order,
    tensors_have_same_dim_order2,
};
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::core::portable_type::tensor_impl::SizesType;
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

// PORT-NOTE: `Tensor& out` / returned `Tensor&` become `&'a Tensor` (interior
// mutation through `*mut TensorImpl`).
//
// PORT-NOTE: `std::optional<int64_t> divisor_override` -> `Option<i64>`; the C++
// `divisor_override.has_value()` / `.value()` become `is_some()` / `.unwrap()`.
// `check_avg_pool2d_args` takes it by reference (`&divisor_override`).
//
// PORT-NOTE: the reduce/map closures use CTYPE `+` and `/`, and
// `static_cast<CTYPE>(count|divisor)` via the dtype_util `StaticCast<i64>` trait
// (mirrors the C++ `static_cast<CTYPE>(int64_t)`; Half/BFloat16 route via f32).
// The empty dilation `{}` becomes `IntArrayRef::new()`.

// [spec:et:def:op-avg-pool2d.torch.executor.native.avg-pool2d-out-fn]
// [spec:et:sem:op-avg-pool2d.torch.executor.native.avg-pool2d-out-fn]
#[allow(clippy::too_many_arguments)]
pub fn avg_pool2d_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    in_: &Tensor,
    kernel_size: IntArrayRef,
    stride: IntArrayRef,
    padding: IntArrayRef,
    ceil_mode: bool,
    count_include_pad: bool,
    divisor_override: Option<i64>,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    crate::et_kernel_check!(
        ctx,
        check_avg_pool2d_args(
            in_,
            kernel_size,
            stride,
            padding,
            ceil_mode,
            count_include_pad,
            &divisor_override,
            out,
        ),
        InvalidArgument,
        out
    );

    crate::et_kernel_check!(
        ctx,
        tensors_have_same_dim_order2(in_, out),
        InvalidArgument,
        out
    );

    crate::et_kernel_check!(ctx, tensor_is_default_dim_order(in_), InvalidArgument, out);

    let mut output_ndim: usize = 0;
    let mut output_sizes: [SizesType; K_TENSOR_DIMENSION_LIMIT] = [0; K_TENSOR_DIMENSION_LIMIT];
    unsafe {
        get_avg_pool2d_out_target_size(
            in_,
            kernel_size,
            stride,
            padding,
            ceil_mode,
            output_sizes.as_mut_ptr(),
            &mut output_ndim,
        );
    }

    crate::et_kernel_check!(
        ctx,
        output_size_is_valid(
            ArrayRef::from_raw_parts(output_sizes.as_ptr(), output_ndim),
            2
        ),
        InvalidArgument,
        out
    );

    crate::et_kernel_check!(
        ctx,
        resize_tensor_same_type(
            out,
            ArrayRef::from_raw_parts(output_sizes.as_ptr(), output_ndim)
        ) == Error::Ok,
        InvalidArgument,
        out
    );

    let in_type = in_.scalar_type();

    crate::et_switch_floathbf16_types_and!(Long, in_type, ctx, "avg_pool2d.out", CTYPE, {
        if divisor_override.is_some() {
            let divisor: i64 = divisor_override.unwrap();
            // If divisor_override is specified, then we don't need to use `count`
            // in the calculation. Simply sum x / divisor to get the output.
            apply_kernel_2d_reduce_then_map_fn::<CTYPE, _, _>(
                &|in_val: CTYPE, _in_idx: i64, accum: CTYPE, _accum_idx: i64| -> (CTYPE, i64) {
                    // Average pooling does not track indexes, so return 0 for
                    // accum_idx
                    (in_val + accum, 0)
                },
                &|_count: i64, accum: CTYPE| -> CTYPE {
                    accum / <CTYPE as StaticCast<i64>>::static_cast(divisor)
                },
                count_include_pad,
                in_,
                kernel_size,
                stride,
                padding,
                IntArrayRef::new(),
                out,
                None,
            );
        } else {
            apply_kernel_2d_reduce_then_map_fn::<CTYPE, _, _>(
                &|in_val: CTYPE, _in_idx: i64, accum: CTYPE, _accum_idx: i64| -> (CTYPE, i64) {
                    // Average pooling does not track indexes, so return 0 for
                    // accum_idx
                    (in_val + accum, 0)
                },
                &|count: i64, accum: CTYPE| -> CTYPE {
                    accum / <CTYPE as StaticCast<i64>>::static_cast(count)
                },
                count_include_pad,
                in_,
                kernel_size,
                stride,
                padding,
                IntArrayRef::new(),
                out,
                None,
            );
        }
    });

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assert_tensor_close;
    use crate::runtime::core::exec_aten::testing_util::tensor_factory::{
        FactoryValue, TensorFactory,
    };
    use crate::runtime::core::exec_aten::testing_util::tensor_util::internal::K_DEFAULT_ATOL;
    use crate::runtime::core::exec_aten::testing_util::tensor_util::tensors_are_close;
    use crate::runtime::core::exec_aten::util::scalar_type_util::CppTypeToScalarType;
    use crate::runtime::core::memory_allocator::{MemoryAllocator, MemoryAllocatorBase};
    use crate::runtime::core::portable_type::scalar_type::ScalarType;
    use crate::runtime::core::portable_type::{BFloat16, Half};

    fn context() -> KernelRuntimeContext<'static> {
        crate::runtime::platform::platform::pal_init();
        KernelRuntimeContext::new(
            crate::extension::module::module::null_event_tracer(),
            core::ptr::null_mut::<MemoryAllocator>() as *mut dyn MemoryAllocatorBase,
        )
    }

    // Local `f64 -> element` bridge for the FLOATHBF16 element types.
    trait FromF64: Copy {
        fn from_f64(v: f64) -> Self;
    }
    impl FromF64 for f32 {
        fn from_f64(v: f64) -> Self {
            v as f32
        }
    }
    impl FromF64 for f64 {
        fn from_f64(v: f64) -> Self {
            v
        }
    }
    impl FromF64 for Half {
        fn from_f64(v: f64) -> Self {
            Half::from_f32(v as f32)
        }
    }
    impl FromF64 for BFloat16 {
        fn from_f64(v: f64) -> Self {
            BFloat16::from_f32(v as f32)
        }
    }

    fn iarr(v: &[i64]) -> IntArrayRef {
        IntArrayRef::from_raw_parts(v.as_ptr(), v.len())
    }

    // PORT-NOTE: `test_4d_dtype<DTYPE>` uses `tf_dtype` for self/out/expected and
    // widens tolerance for Half/BFloat16.
    fn test_4d_dtype<T>()
    where
        T: CppTypeToScalarType + FactoryValue + FromF64,
    {
        let tf = TensorFactory::<T>::new();

        let self_data: Vec<f64> = SELF_4D.to_vec();
        let self_ = tf.make_default(
            vec![2, 3, 8, 8],
            self_data.iter().map(|&v| T::from_f64(v)).collect(),
        );
        let kernel_size_vec: [i64; 2] = [2, 3];
        let stride_vec: [i64; 2] = [3, 2];
        let padding_vec: [i64; 2] = [1, 1];
        let ceil_mode = false;
        let count_include_pad = true;
        let divisor_override: Option<i64> = None;
        let out = tf.zeros_default(vec![2, 3, 3, 4]);
        let out_expected = tf.make_default(
            vec![2, 3, 3, 4],
            OUT_4D.iter().map(|&v| T::from_f64(v)).collect(),
        );
        let mut ctx = context();
        avg_pool2d_out(
            &mut ctx,
            &self_,
            iarr(&kernel_size_vec),
            iarr(&stride_vec),
            iarr(&padding_vec),
            ceil_mode,
            count_include_pad,
            divisor_override,
            &out,
        );
        if T::VALUE == ScalarType::Half || T::VALUE == ScalarType::BFloat16 {
            // op requires wide tolerance to pass test, but at least we verify
            // that it supports these dtypes.
            assert!(tensors_are_close(
                &out,
                &out_expected,
                5e-2,
                Some(K_DEFAULT_ATOL)
            ));
        } else {
            assert_tensor_close!(out, out_expected);
        }
    }

    // PORT-NOTE: `test_4d_divisor_override_dtype<DTYPE>` ignores DTYPE and always
    // uses the Float factory in the C++ body; ported faithfully (the dtype loop
    // therefore runs this identical float computation four times).
    fn test_4d_divisor_override_dtype() {
        let tf_float = TensorFactory::<f32>::new();

        let self_ = tf_float.make_default(
            vec![2, 3, 8, 8],
            SELF_4D_DIVISOR.iter().map(|&v| v as f32).collect(),
        );
        let kernel_size_vec: [i64; 2] = [2, 3];
        let stride_vec: [i64; 2] = [3, 2];
        let padding_vec: [i64; 2] = [1, 1];
        let ceil_mode = false;
        let count_include_pad = true;
        let divisor_override: Option<i64> = Some(10);
        let out = tf_float.zeros_default(vec![2, 3, 3, 4]);
        let out_expected = tf_float.make_default(
            vec![2, 3, 3, 4],
            OUT_4D_DIVISOR.iter().map(|&v| v as f32).collect(),
        );
        let mut ctx = context();
        avg_pool2d_out(
            &mut ctx,
            &self_,
            iarr(&kernel_size_vec),
            iarr(&stride_vec),
            iarr(&padding_vec),
            ceil_mode,
            count_include_pad,
            divisor_override,
            &out,
        );
        assert_tensor_close!(out, out_expected);
    }

    fn test_4d_ceil_mode_no_include_padding_dtype() {
        let tf_float = TensorFactory::<f32>::new();

        let self_ = tf_float.make_default(
            vec![2, 3, 14, 12],
            SELF_4D_CEIL.iter().map(|&v| v as f32).collect(),
        );
        let kernel_size_vec: [i64; 2] = [4, 2];
        let stride_vec: [i64; 2] = [1, 2];
        let padding_vec: [i64; 2] = [1, 1];
        let ceil_mode = true;
        let count_include_pad = false;
        let divisor_override: Option<i64> = None;
        let out = tf_float.zeros_default(vec![2, 3, 13, 7]);
        let out_expected = tf_float.make_default(
            vec![2, 3, 13, 7],
            OUT_4D_CEIL.iter().map(|&v| v as f32).collect(),
        );
        let mut ctx = context();
        avg_pool2d_out(
            &mut ctx,
            &self_,
            iarr(&kernel_size_vec),
            iarr(&stride_vec),
            iarr(&padding_vec),
            ceil_mode,
            count_include_pad,
            divisor_override,
            &out,
        );
        assert_tensor_close!(out, out_expected);
    }

    // [spec:et:sem:op-avg-pool2d.torch.executor.native.avg-pool2d-out-fn/test]
    // Numeric 4D avg-pool spans the full kernel_ops_util avg-pool pipeline; the
    // expected-tensor comparison would fail if any of these were wrong.
    // [spec:et:sem:kernel-ops-util.torch.executor.check-avg-pool2d-args-fn/test]
    // [spec:et:sem:kernel-ops-util.torch.executor.get-avg-pool2d-out-target-size-fn/test]
    // [spec:et:sem:kernel-ops-util.torch.executor.output-size-is-valid-fn/test]
    // [spec:et:sem:kernel-ops-util.torch.executor.calculate-kernel-output-sizes-fn/test]
    // [spec:et:sem:kernel-ops-util.torch.executor.kernel-output-size-helper-fn/test]
    // [spec:et:sem:kernel-ops-util.torch.executor.apply-kernel-2d-reduce-then-map-fn-fn/test]
    // [spec:et:sem:kernel-ops-util.torch.executor.kernel-reduction-then-map-2d-fn/test]
    // [spec:et:sem:kernel-ops-util.torch.executor.kernel-size-is-valid-fn/test]
    // [spec:et:sem:kernel-ops-util.torch.executor.stride-is-valid-fn/test]
    // [spec:et:sem:kernel-ops-util.torch.executor.padding-is-valid-fn/test]
    // [spec:et:sem:kernel-ops-util.torch.executor.param-array-is-valid-fn/test]
    // [spec:et:sem:kernel-ops-util.torch.executor.int-array-all-ge-fn/test]
    // [spec:et:sem:kernel-ops-util.torch.executor.val-at-fn/test]
    #[test]
    fn op_avg_pool2d_out_test_sanity_check_4d() {
        test_4d_dtype::<f32>();
        test_4d_dtype::<f64>();
        test_4d_dtype::<Half>();
        test_4d_dtype::<BFloat16>();
    }

    // [spec:et:sem:op-avg-pool2d.torch.executor.native.avg-pool2d-out-fn/test]
    #[test]
    fn op_avg_pool2d_out_test_sanity_check_4d_divisor_override() {
        // ET_FORALL_FLOATHBF16_TYPES: body is dtype-independent (uses tfFloat).
        test_4d_divisor_override_dtype();
        test_4d_divisor_override_dtype();
        test_4d_divisor_override_dtype();
        test_4d_divisor_override_dtype();
    }

    // [spec:et:sem:op-avg-pool2d.torch.executor.native.avg-pool2d-out-fn/test]
    #[test]
    fn op_avg_pool2d_out_test_sanity_check_4d_ceil_mode_no_include_padding() {
        test_4d_ceil_mode_no_include_padding_dtype();
        test_4d_ceil_mode_no_include_padding_dtype();
        test_4d_ceil_mode_no_include_padding_dtype();
        test_4d_ceil_mode_no_include_padding_dtype();
    }

    include!("op_avg_pool2d_test_data.rs");
}
