//! Literal port of kernels/quantized/cpu/op_mixed_mm.cpp.

use crate::kernels::portable::cpu::vec_ops::{FromI8, VecScalar, vec_quantized_matmul_int8};
use crate::runtime::core::array_ref::ArrayRef;
use crate::runtime::core::error::Error;
use crate::runtime::core::exec_aten::util::tensor_util::{
    K_TENSOR_DIMENSION_LIMIT, resize_tensor, tensor_is_rank, tensors_have_same_dtype,
    tensors_have_same_dtype2, tensors_have_same_shape2, tensors_have_same_size_at_dims,
};
use crate::runtime::core::memory_allocator::{MemoryAllocator, MemoryAllocatorBase};
use crate::runtime::core::portable_type::scalar_type::ScalarType;
use crate::runtime::core::portable_type::tensor::{Tensor, TensorSizesType};
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

// PORT-NOTE: header inline convenience overloads that omit `ctx` construct a
// fresh default-constructed `KernelRuntimeContext ctx;`. Mirrors the established
// `embeddingxb.rs` pattern.
fn default_context() -> KernelRuntimeContext<'static> {
    KernelRuntimeContext::new(
        crate::extension::module::module::null_event_tracer(),
        core::ptr::null_mut::<MemoryAllocator>() as *mut dyn MemoryAllocatorBase,
    )
}

// PORT-NOTE: `ET_CHECK` is a C++ fatal check; mirrored with a local abort.
macro_rules! et_check {
    ($cond:expr) => {
        if !($cond) {
            crate::runtime::platform::abort::runtime_abort();
        }
    };
}

// PORT-NOTE: `ET_LOG_AND_RETURN_IF_FALSE(cond)` = `ET_CHECK_OR_RETURN_FALSE(cond,
// "")`; the crate-level `et_check_or_return_false!` requires a leading format
// literal, so this module defines its own arg-free variant as matmul_ops_util.rs
// does.
macro_rules! et_log_and_return_if_false {
    ($cond:expr) => {{
        if !($cond) {
            $crate::et_log!(Error, "Check failed ({}): ", ::core::stringify!($cond));
            return false;
        }
    }};
}

// PORT-NOTE: `VecMm` bounds `CTYPE` to the numeric operations
// `vec_quantized_matmul_int8` needs (`VecScalar + FromI8`), reproducing the C++
// template instantiation over {Float, Half}.
trait VecMm: VecScalar + FromI8 {}
impl VecMm for f32 {}
impl VecMm for crate::runtime::core::portable_type::Half {}

// [spec:et:def:op-mixed-mm.torch.executor.native.check-quantized-mixed-mm-args-fn]
// [spec:et:sem:op-mixed-mm.torch.executor.native.check-quantized-mixed-mm-args-fn]
fn check_quantized_mixed_mm_args(
    in_: &Tensor,
    weight: &Tensor,
    weight_scales: &Tensor,
    opt_weight_zero_points: &Option<&Tensor>,
    out: &Tensor,
) -> bool {
    et_log_and_return_if_false!(tensor_is_rank(in_, 2));
    et_log_and_return_if_false!(tensor_is_rank(weight, 2));
    et_log_and_return_if_false!(tensor_is_rank(weight_scales, 1));
    et_log_and_return_if_false!(tensor_is_rank(out, 2));

    et_log_and_return_if_false!(tensors_have_same_size_at_dims(in_, 1, weight, 0));
    et_log_and_return_if_false!(tensors_have_same_size_at_dims(weight_scales, 0, weight, 0));

    et_log_and_return_if_false!(tensors_have_same_dtype(in_, weight_scales, out));
    crate::et_check_or_return_false!(
        weight.scalar_type() == ScalarType::Char,
        "weight dtype must be int8"
    );
    crate::et_check_or_return_false!(
        in_.scalar_type() == ScalarType::Float || in_.scalar_type() == ScalarType::Half,
        "input dtype must be Float or Half"
    );

    if let Some(zp) = opt_weight_zero_points {
        et_log_and_return_if_false!(tensors_have_same_shape2(zp, weight_scales));
        et_log_and_return_if_false!(tensors_have_same_dtype2(zp, in_));
    }

    // Support for non-null zero points is not implemented yet.
    crate::et_check_or_return_false!(
        opt_weight_zero_points.is_none(),
        "zero points not supported yet."
    );
    true
}

// [spec:et:def:op-mixed-mm.torch.executor.native.quantized-mixed-mm-out-fn]
// [spec:et:sem:op-mixed-mm.torch.executor.native.quantized-mixed-mm-out-fn]
pub fn quantized_mixed_mm_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    in_: &Tensor,
    weight: &Tensor,
    weight_scales: &Tensor,
    opt_weight_zero_points: &Option<&Tensor>,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    crate::et_kernel_check!(
        ctx,
        check_quantized_mixed_mm_args(in_, weight, weight_scales, opt_weight_zero_points, out),
        InvalidArgument,
        out
    );

    let output_ndim: usize = 2;
    let mut output_sizes: [TensorSizesType; K_TENSOR_DIMENSION_LIMIT] =
        [0; K_TENSOR_DIMENSION_LIMIT];
    output_sizes[0] = in_.size(0) as TensorSizesType;
    output_sizes[1] = weight.size(1) as TensorSizesType;

    crate::et_kernel_check!(
        ctx,
        resize_tensor(
            out,
            ArrayRef::<TensorSizesType>::from_raw_parts(output_sizes.as_ptr(), output_ndim)
        ) == Error::Ok,
        InvalidArgument,
        out
    );

    let name = "quantized_decomposed::mixed_mm.out";

    fn mm<CTYPE: VecMm>(in_: &Tensor, weight: &Tensor, weight_scales: &Tensor, out: &Tensor) {
        let m: i64 = in_.size(0) as i64;
        let n: i64 = in_.size(1) as i64;
        let p: i64 = weight.size(1) as i64;

        unsafe {
            vec_quantized_matmul_int8::<CTYPE>(
                out.mutable_data_ptr::<CTYPE>(),
                in_.const_data_ptr::<CTYPE>(),
                weight.const_data_ptr::<i8>(),
                weight_scales.const_data_ptr::<CTYPE>(),
                m,
                n,
                p,
            );
        }
    }

    crate::et_switch_two_types!(Float, Half, in_.scalar_type(), ctx, name, CTYPE, {
        mm::<CTYPE>(in_, weight, weight_scales, out);
    });

    out
}

pub fn quantized_mixed_mm_out_nocontext<'a, 'b>(
    in_: &Tensor,
    weight: &Tensor,
    weight_scales: &Tensor,
    opt_weight_zero_points: &Option<&Tensor>,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    // TODO(mcandales): Remove the need for this wrapper
    let mut context = default_context();
    let res = quantized_mixed_mm_out(
        &mut context,
        in_,
        weight,
        weight_scales,
        opt_weight_zero_points,
        out,
    );
    et_check!(context.failure_state() == Error::Ok);
    res
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::core::exec_aten::testing_util::tensor_factory::{
        FactoryValue, TensorFactory,
    };
    use crate::runtime::core::exec_aten::testing_util::tensor_util::tensors_are_close;
    use crate::runtime::core::exec_aten::util::scalar_type_util::CppTypeToScalarType;
    use crate::runtime::core::memory_allocator::{MemoryAllocator, MemoryAllocatorBase};
    use crate::runtime::core::portable_type::Half;
    use crate::runtime::core::tensor_shape_dynamism::TensorShapeDynamism::STATIC;

    // Mirrors `OpQuantizedMixedMMTest::SetUp()`'s `runtime_init()`.
    fn ctx() -> KernelRuntimeContext<'static> {
        crate::runtime::platform::runtime::runtime_init();
        KernelRuntimeContext::new(
            crate::extension::module::module::null_event_tracer(),
            core::ptr::null_mut::<MemoryAllocator>() as *mut dyn MemoryAllocatorBase,
        )
    }

    macro_rules! assert_tensor_close {
        ($t1:expr, $t2:expr) => {
            assert!(
                tensors_are_close(
                    &$t1,
                    &$t2,
                    crate::runtime::core::exec_aten::testing_util::tensor_util::internal::K_DEFAULT_RTOL,
                    None
                ),
                "tensors are not close"
            )
        };
    }

    trait FromF64Ctor: Copy {
        fn ctor(v: f64) -> Self;
    }
    impl FromF64Ctor for f32 {
        fn ctor(v: f64) -> Self {
            v as f32
        }
    }
    impl FromF64Ctor for Half {
        fn ctor(v: f64) -> Self {
            Half::from_f32(v as f32)
        }
    }

    // quantized_mixed_mm_out gates on check_quantized_mixed_mm_args (via
    // et_kernel_check!); a false return would surface as a kernel failure and
    // leave `out` unwritten, so the exact expected values pin the arg check.
    // [spec:et:sem:op-mixed-mm.torch.executor.native.quantized-mixed-mm-out-fn/test]
    // [spec:et:sem:op-mixed-mm.torch.executor.native.check-quantized-mixed-mm-args-fn/test]
    fn test_dtype<T>()
    where
        T: CppTypeToScalarType + FactoryValue + FromF64Ctor,
    {
        let tf = TensorFactory::<T>::new();
        let tf_char = TensorFactory::<i8>::new();

        let input = tf.make_default(vec![1, 3], vec![T::ctor(1.0), T::ctor(1.5), T::ctor(2.0)]);
        let weight = tf_char.make_default(vec![3, 2], vec![5, 4, 3, 2, 1, 1]);
        let weight_scales =
            tf.make_default(vec![3], vec![T::ctor(0.2), T::ctor(0.4), T::ctor(0.5)]);
        let opt_weight_zp: Option<&Tensor> = None;

        let out = tf.zeros(vec![1, 2], STATIC);

        let expected = tf.make_default(vec![1, 2], vec![T::ctor(3.8), T::ctor(3.0)]);

        let mut context = ctx();

        quantized_mixed_mm_out(
            &mut context,
            &input,
            &weight,
            &weight_scales,
            &opt_weight_zp,
            &out,
        );

        assert_tensor_close!(out, expected);
    }

    #[test]
    fn op_quantized_mixed_mm_test_float_input() {
        test_dtype::<f32>();
    }

    #[test]
    fn op_quantized_mixed_mm_test_half_input() {
        test_dtype::<Half>();
    }
}
