//! Literal port of kernels/quantized/cpu/op_mixed_linear.cpp.

use crate::kernels::portable::cpu::vec_ops::{FromI8, VecScalar, vec_quantized_matmul_transb_int8};
use crate::runtime::core::array_ref::ArrayRef;
use crate::runtime::core::error::Error;
use crate::runtime::core::exec_aten::util::tensor_util::{
    K_TENSOR_DIMENSION_LIMIT, resize_tensor, tensor_is_rank, tensors_have_same_dtype2,
    tensors_have_same_shape2, tensors_have_same_size_at_dims,
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

// PORT-NOTE: `VecMm` bounds the matmul element type to `VecScalar + FromI8`,
// reproducing the C++ template instantiation. See the PORT-NOTE at the
// `vec_quantized_matmul_transb_int8` call site regarding the collapsed type
// parameter.
trait VecMm: VecScalar + FromI8 {}
impl VecMm for f32 {}
impl VecMm for crate::runtime::core::portable_type::Half {}

// [spec:et:def:op-mixed-linear.torch.executor.native.check-quantized-mixed-linear-args-fn]
// [spec:et:sem:op-mixed-linear.torch.executor.native.check-quantized-mixed-linear-args-fn]
fn check_quantized_mixed_linear_args(
    in_: &Tensor,
    weight: &Tensor,
    weight_scales: &Tensor,
    opt_weight_zero_points: &Option<&Tensor>,
    dtype: Option<ScalarType>,
    out: &Tensor,
) -> bool {
    et_log_and_return_if_false!(tensor_is_rank(in_, 2));
    et_log_and_return_if_false!(tensor_is_rank(weight, 2));
    et_log_and_return_if_false!(
        tensor_is_rank(weight_scales, 1) || tensor_is_rank(weight_scales, 2)
    );
    et_log_and_return_if_false!(tensor_is_rank(out, 2));

    et_log_and_return_if_false!(tensors_have_same_size_at_dims(in_, 1, weight, 1));
    et_log_and_return_if_false!(tensors_have_same_size_at_dims(weight_scales, 0, weight, 0));
    et_log_and_return_if_false!(tensors_have_same_size_at_dims(in_, 1, weight, 1));

    et_log_and_return_if_false!(tensors_have_same_dtype2(in_, weight_scales));
    if let Some(d) = dtype {
        et_log_and_return_if_false!(out.scalar_type() == d);
        crate::et_check_or_return_false!(
            d == ScalarType::Float || d == ScalarType::Half,
            "dtype must be Float or Half"
        );
    }
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

// [spec:et:def:op-mixed-linear.torch.executor.native.quantized-mixed-linear-out-fn]
// [spec:et:sem:op-mixed-linear.torch.executor.native.quantized-mixed-linear-out-fn]
pub fn quantized_mixed_linear_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    in_: &Tensor,
    weight: &Tensor,
    weight_scales: &Tensor,
    opt_weight_zero_points: &Option<&Tensor>,
    dtype: Option<ScalarType>,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    crate::et_kernel_check!(
        ctx,
        check_quantized_mixed_linear_args(
            in_,
            weight,
            weight_scales,
            opt_weight_zero_points,
            dtype,
            out
        ),
        InvalidArgument,
        out
    );

    let out_dtype: ScalarType = if let Some(d) = dtype {
        d
    } else {
        out.scalar_type()
    };

    let output_ndim: usize = 2;
    let mut output_sizes: [TensorSizesType; K_TENSOR_DIMENSION_LIMIT] =
        [0; K_TENSOR_DIMENSION_LIMIT];
    output_sizes[0] = in_.size(0) as TensorSizesType;
    output_sizes[1] = weight.size(0) as TensorSizesType;

    crate::et_kernel_check!(
        ctx,
        resize_tensor(
            out,
            ArrayRef::<TensorSizesType>::from_raw_parts(output_sizes.as_ptr(), output_ndim)
        ) == Error::Ok,
        InvalidArgument,
        out
    );

    let name = "quantized_decomposed::mixed_linear.out";

    // PORT-NOTE: the C++ instantiates
    // `vec_quantized_matmul_transb_int8<CTYPE_OUT, CTYPE>` where the accumulator/
    // output type `T=CTYPE_OUT` (Float/Double/Half via ET_SWITCH_FLOAT_TYPES_AND
    // (Half)) differs from the input/scale operand type `U=V=CTYPE` (Float/Half).
    // The ported `vec_quantized_matmul_transb_int8` in vec_ops.rs currently
    // collapses these into a single generic `T` (z, x, s all `*T`). This port
    // therefore instantiates it with the INPUT/scale ctype `CTYPE`, which is
    // bit-exact for the common `out_dtype == in.scalar_type()` case but diverges
    // from the C++ when they differ (the C++ itself carries a
    // "FIXME: this currently ignores dtype" comment on this call, so accumulation
    // narrows to CTYPE there too — only the final stored result type differs).
    // CROSS-MODULE: vec_ops.rs `vec_quantized_matmul_transb_int8` needs a second
    // (and third) type parameter `<T, U, V=U>` to fully mirror the C++ signature;
    // flagged for the fixer.
    #[allow(clippy::too_many_arguments)]
    fn matmul<CTYPE: VecMm>(in_: &Tensor, weight: &Tensor, weight_scales: &Tensor, out: &Tensor) {
        let m: i64 = in_.size(0) as i64;
        let n: i64 = in_.size(1) as i64;
        let p: i64 = weight.size(0) as i64;
        let mut g: i64 = n;

        if weight_scales.dim() == 2 {
            g = (n + weight_scales.size(1) as i64 - 1) / weight_scales.size(1) as i64;
        }

        // FIXME: this currently ignores dtype
        unsafe {
            vec_quantized_matmul_transb_int8::<CTYPE>(
                out.mutable_data_ptr::<CTYPE>(),
                in_.const_data_ptr::<CTYPE>(),
                weight.const_data_ptr::<i8>(),
                weight_scales.const_data_ptr::<CTYPE>(),
                m,
                n,
                p,
                g,
            );
        }
    }

    crate::et_switch_two_types!(Float, Half, in_.scalar_type(), ctx, name, CTYPE, {
        crate::et_switch_float_types_and!(Half, out_dtype, ctx, name, CTYPE_OUT, {
            // CTYPE_OUT is dispatched to mirror the C++ ET_SWITCH; see the
            // PORT-NOTE above on why the call is instantiated with CTYPE.
            let _ = ::core::marker::PhantomData::<CTYPE_OUT>;
            matmul::<CTYPE>(in_, weight, weight_scales, out);
        });
    });

    out
}

pub fn quantized_mixed_linear_out_nocontext<'a, 'b>(
    in_: &Tensor,
    weight: &Tensor,
    weight_scales: &Tensor,
    opt_weight_zero_points: &Option<&Tensor>,
    dtype: Option<ScalarType>,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    // TODO(mcandales): Remove the need for this wrapper
    // TODO(mkg): add support for dtype
    let mut context = default_context();
    let res = quantized_mixed_linear_out(
        &mut context,
        in_,
        weight,
        weight_scales,
        opt_weight_zero_points,
        dtype,
        out,
    );
    et_check!(context.failure_state() == Error::Ok);
    res
}

#[cfg(test)]
mod tests {
    // PORT-NOTE: the C++ suite has four additional TEST_F cases each guarded by
    // `#if 0` (FloatInputHalfOutput, HalfInputFloatOutput, HalfInputHalfOutput and
    // their `_Partials` variants), commented "need <<" / "need to relax
    // tolerance". Those are compiled out in C++ (never run), so they are not ported
    // as `#[test]`s here — only the two enabled Float/Float cases are.
    use super::*;
    use crate::runtime::core::exec_aten::testing_util::tensor_factory::TensorFactory;
    use crate::runtime::core::exec_aten::testing_util::tensor_util::tensors_are_close;
    use crate::runtime::core::memory_allocator::{MemoryAllocator, MemoryAllocatorBase};
    use crate::runtime::core::tensor_shape_dynamism::TensorShapeDynamism::STATIC;

    // Mirrors `OpQuantizedMixedDtypeLinearTest::SetUp()`'s `runtime_init()`.
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

    // Float input, Float output. (The generic C++ helper is templated over DTYPE
    // and DTYPE_OUT; only the Float/Float instantiation is enabled.)
    //
    // quantized_mixed_linear_out gates on check_quantized_mixed_linear_args
    // (via et_kernel_check!); a false return would surface as a kernel failure
    // and leave `out` unwritten, so the exact expected values pin the arg check.
    // [spec:et:sem:op-mixed-linear.torch.executor.native.quantized-mixed-linear-out-fn/test]
    // [spec:et:sem:op-mixed-linear.torch.executor.native.check-quantized-mixed-linear-args-fn/test]
    fn test_dtype() {
        let tf = TensorFactory::<f32>::new();
        let tf_char = TensorFactory::<i8>::new();
        let tf_out = TensorFactory::<f32>::new();

        let input = tf.make_default(vec![1, 3], vec![1.0, 1.5, 2.0]);
        let weight = tf_char.make_default(vec![2, 3], vec![5, 3, 1, 4, 2, 1]);
        let weight_scales = tf.make_default(vec![2], vec![0.2, 0.4]);
        let opt_weight_zp: Option<&Tensor> = None;
        let opt_dtype_out: Option<ScalarType> = None;

        let out = tf_out.zeros(vec![1, 2], STATIC);

        let expected = tf_out.make_default(vec![1, 2], vec![2.3, 3.6]);

        let mut context = ctx();

        quantized_mixed_linear_out(
            &mut context,
            &input,
            &weight,
            &weight_scales,
            &opt_weight_zp,
            opt_dtype_out,
            &out,
        );

        assert_tensor_close!(out, expected);
    }

    #[test]
    fn op_quantized_mixed_dtype_linear_test_float_input_float_output() {
        test_dtype();
    }

    // Float input, Float output, per-partial scales.
    // [spec:et:sem:op-mixed-linear.torch.executor.native.quantized-mixed-linear-out-fn/test]
    fn test_dtype_partials() {
        let tf = TensorFactory::<f32>::new();
        let tf_char = TensorFactory::<i8>::new();
        let tf_out = TensorFactory::<f32>::new();

        let input = tf.make_default(vec![1, 3], vec![1.0, 1.5, 2.0]);
        let weight = tf_char.make_default(vec![2, 3], vec![5, 3, 1, 4, 2, 1]);
        let weight_scales = tf.make_default(vec![2, 2], vec![0.2, 1.0, 0.4, 0.5]);
        let opt_weight_zp: Option<&Tensor> = None;
        let opt_dtype_out: Option<ScalarType> = None;

        let out = tf_out.zeros(vec![1, 2], STATIC);

        let expected = tf_out.make_default(
            vec![1, 2],
            vec![
                ((1.0 * 5.0 + 1.5 * 3.0) * 0.2 + 2.0 * 1.0 * 1.0) as f32,
                ((1.0 * 4.0 + 1.5 * 2.0) * 0.4 + 2.0 * 1.0 * 0.5) as f32,
            ],
        );

        let mut context = ctx();

        quantized_mixed_linear_out(
            &mut context,
            &input,
            &weight,
            &weight_scales,
            &opt_weight_zp,
            opt_dtype_out,
            &out,
        );

        assert_tensor_close!(out, expected);
    }

    #[test]
    fn op_quantized_mixed_dtype_linear_test_float_input_float_output_partials() {
        test_dtype_partials();
    }
}
