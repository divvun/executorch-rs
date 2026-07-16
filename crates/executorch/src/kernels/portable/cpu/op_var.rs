//! Literal port of kernels/portable/cpu/op_var.cpp.

use crate::kernels::portable::cpu::scalar_utils::scalar_to;
use crate::kernels::portable::cpu::util::dtype_util::internal::StaticCast;
#[cfg(not(feature = "aten"))]
use crate::kernels::portable::cpu::util::reduce_util::check_reduction_args;
use crate::kernels::portable::cpu::util::reduce_util::{
    MapReduceOverDimListPlan, get_reduced_dim_product,
    parallel_for_each_reduce_over_dim_list_output_index, resize_reduction_out,
};
use crate::runtime::core::array_ref::ArrayRef;
use crate::runtime::core::error::Error;
use crate::runtime::core::exec_aten::util::tensor_util::{
    tensor_is_default_dim_order, tensor_is_floating_type, tensors_have_same_dim_order2,
};
use crate::runtime::core::portable_type::scalar::Scalar;
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

// PORT-NOTE: `Tensor& out` / returned `Tensor&` become `&'a Tensor` (interior
// mutation through `*mut TensorImpl`). `optional<ArrayRef<int64_t>> dim_list`
// maps to `Option<ArrayRef<i64>>`.
//
// PORT-NOTE: the C++ two-pass variance accumulates entirely in `CTYPE_OUT`
// (FLOATHBF16). c10::Half/BFloat16 arithmetic follows the same-type operator
// overloads (`Half op Half -> Half`), while `sum2 / denominator` is
// `CTYPE_OUT / double -> double` narrowed back to `CTYPE_OUT`, and `NAN`
// assigns the cast constant. The `VarArith` trait (shared with op_var_mean)
// reproduces each of these operator semantics per CTYPE_OUT.

pub(crate) trait VarArith: Copy + StaticCast<Self> {
    // static_cast<CTYPE_OUT>(NAN)
    fn nan() -> Self;
    // static_cast<CTYPE_OUT>(num) for the count / cast operands.
    fn from_usize(v: usize) -> Self;
    // CTYPE_OUT op CTYPE_OUT -> CTYPE_OUT (same-type operator overloads).
    fn add(a: Self, b: Self) -> Self;
    fn sub(a: Self, b: Self) -> Self;
    fn mul(a: Self, b: Self) -> Self;
    fn div(a: Self, b: Self) -> Self;
    // CTYPE_OUT / double -> double, narrowed back to CTYPE_OUT.
    fn div_by_double(a: Self, d: f64) -> Self;
    // static_cast<Self>(f64) narrowing.
    fn from_f64(v: f64) -> Self;
}

macro_rules! impl_var_arith_prim {
    ($t:ty) => {
        impl VarArith for $t {
            #[inline]
            fn nan() -> Self {
                f64::NAN as $t
            }
            #[inline]
            fn from_usize(v: usize) -> Self {
                v as $t
            }
            #[inline]
            fn add(a: Self, b: Self) -> Self {
                a + b
            }
            #[inline]
            fn sub(a: Self, b: Self) -> Self {
                a - b
            }
            #[inline]
            fn mul(a: Self, b: Self) -> Self {
                a * b
            }
            #[inline]
            fn div(a: Self, b: Self) -> Self {
                a / b
            }
            #[inline]
            fn div_by_double(a: Self, d: f64) -> Self {
                (a as f64 / d) as $t
            }
            #[inline]
            fn from_f64(v: f64) -> Self {
                v as $t
            }
        }
    };
}
impl_var_arith_prim!(f32);
impl_var_arith_prim!(f64);

macro_rules! impl_var_arith_half {
    ($t:ty) => {
        impl VarArith for $t {
            #[inline]
            fn nan() -> Self {
                <$t>::from_f64(f64::NAN)
            }
            #[inline]
            fn from_usize(v: usize) -> Self {
                <$t>::from_f32(v as f32)
            }
            #[inline]
            fn add(a: Self, b: Self) -> Self {
                <$t>::from_f32(a.to_f32() + b.to_f32())
            }
            #[inline]
            fn sub(a: Self, b: Self) -> Self {
                <$t>::from_f32(a.to_f32() - b.to_f32())
            }
            #[inline]
            fn mul(a: Self, b: Self) -> Self {
                <$t>::from_f32(a.to_f32() * b.to_f32())
            }
            #[inline]
            fn div(a: Self, b: Self) -> Self {
                <$t>::from_f32(a.to_f32() / b.to_f32())
            }
            #[inline]
            fn div_by_double(a: Self, d: f64) -> Self {
                <$t>::from_f64(a.to_f64() / d)
            }
            #[inline]
            fn from_f64(v: f64) -> Self {
                <$t>::from_f64(v)
            }
        }
    };
}
impl_var_arith_half!(crate::runtime::core::portable_type::Half);
impl_var_arith_half!(crate::runtime::core::portable_type::BFloat16);

// [spec:et:def:op-var.torch.executor.native.compute-variance-fn]
// [spec:et:sem:op-var.torch.executor.native.compute-variance-fn]
#[allow(non_camel_case_types)]
pub(crate) fn compute_variance<CTYPE_IN, CTYPE_OUT>(
    ctx: &mut KernelRuntimeContext,
    in_: &Tensor,
    out: &Tensor,
    dim_list: Option<ArrayRef<i64>>,
    num: usize,
    denominator: f64,
) where
    CTYPE_IN: Copy,
    CTYPE_OUT: VarArith + StaticCast<CTYPE_IN>,
{
    let out_data: *mut CTYPE_OUT = out.mutable_data_ptr::<CTYPE_OUT>();
    if num == 0 || denominator <= 0.0 {
        for out_ix in 0..out.numel() {
            unsafe {
                *out_data.add(out_ix as usize) = CTYPE_OUT::nan();
            }
        }
    } else if in_.numel() > 0 {
        let plan = MapReduceOverDimListPlan::new(in_, &dim_list);
        let success = parallel_for_each_reduce_over_dim_list_output_index(
            in_,
            dim_list,
            out,
            &|begin: i64, end: i64| {
                for out_ix in begin..end {
                    let out_ix = out_ix as usize;
                    let sum: CTYPE_OUT = plan.execute::<CTYPE_IN, CTYPE_OUT, _, _>(
                        |v: CTYPE_IN| <CTYPE_OUT as StaticCast<CTYPE_IN>>::static_cast(v),
                        |outv: CTYPE_OUT, acc: CTYPE_OUT| CTYPE_OUT::add(acc, outv),
                        out_ix,
                    );
                    let mean: CTYPE_OUT = CTYPE_OUT::div(sum, CTYPE_OUT::from_usize(num));
                    let sum2: CTYPE_OUT = plan.execute::<CTYPE_IN, CTYPE_OUT, _, _>(
                        |v: CTYPE_IN| {
                            let d = CTYPE_OUT::sub(
                                <CTYPE_OUT as StaticCast<CTYPE_IN>>::static_cast(v),
                                mean,
                            );
                            CTYPE_OUT::mul(d, d)
                        },
                        |outv: CTYPE_OUT, acc: CTYPE_OUT| CTYPE_OUT::add(acc, outv),
                        out_ix,
                    );
                    unsafe {
                        *out_data.add(out_ix) = CTYPE_OUT::div_by_double(sum2, denominator);
                    }
                }
            },
        );
        crate::et_kernel_check_msg!(ctx, success, Internal, (), "parallel_for failed");
    }
}

// [spec:et:def:op-var.torch.executor.native.var-out-fn]
// [spec:et:sem:op-var.torch.executor.native.var-out-fn]
pub fn var_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    in_: &Tensor,
    dim_list: Option<ArrayRef<i64>>,
    unbiased: bool,
    keepdim: bool,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    // (void)ctx;

    // PORT-NOTE: C++ calls `check_reduction_args` unconditionally, but the ported
    // reduce_util gates the portable arg-checkers behind `#[cfg(not(aten))]`
    // (absent in the ATen build); gated here to match the ported util. Unresolved
    // cross-module reference for the fixer: the C++ call was not `#ifndef`-guarded.
    #[cfg(not(feature = "aten"))]
    crate::et_kernel_check!(
        ctx,
        check_reduction_args(in_, &dim_list, keepdim, None, out),
        InvalidArgument,
        out
    );

    crate::et_kernel_check!(ctx, tensor_is_floating_type(in_), InvalidArgument, out);
    crate::et_kernel_check!(ctx, tensor_is_floating_type(out), InvalidArgument, out);

    crate::et_kernel_check!(
        ctx,
        tensors_have_same_dim_order2(in_, out),
        InvalidArgument,
        out
    );

    crate::et_kernel_check!(ctx, tensor_is_default_dim_order(in_), InvalidArgument, out);

    crate::et_kernel_check!(
        ctx,
        resize_reduction_out(in_, &dim_list, keepdim, out) == Error::Ok,
        InvalidArgument,
        out
    );

    let num: usize = get_reduced_dim_product(in_, &dim_list);
    // PORT-NOTE: C++ `num - 1` on `size_t`; when `num == 0` this wraps to
    // SIZE_MAX (the degenerate `num == 0` path in `compute_variance` still
    // short-circuits to NAN before it is used as a divisor). `wrapping_sub`
    // reproduces the wrap instead of panicking in debug.
    let denom: usize = if unbiased { num.wrapping_sub(1) } else { num };

    let name = "var.out";

    crate::et_switch_floathbf16_types!(in_.scalar_type(), ctx, name, CTYPE_IN, {
        crate::et_switch_floathbf16_types!(out.scalar_type(), ctx, name, CTYPE_OUT, {
            compute_variance::<CTYPE_IN, CTYPE_OUT>(ctx, in_, out, dim_list, num, denom as f64);
        });
    });

    out
}

// [spec:et:def:op-var.torch.executor.native.var-correction-out-fn]
// [spec:et:sem:op-var.torch.executor.native.var-correction-out-fn]
pub fn var_correction_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    in_: &Tensor,
    dim_list: Option<ArrayRef<i64>>,
    correction: &Option<Scalar>,
    keepdim: bool,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    // (void)ctx;

    #[cfg(not(feature = "aten"))]
    crate::et_kernel_check!(
        ctx,
        check_reduction_args(in_, &dim_list, keepdim, None, out),
        InvalidArgument,
        out
    );

    crate::et_kernel_check!(
        ctx,
        resize_reduction_out(in_, &dim_list, keepdim, out) == Error::Ok,
        InvalidArgument,
        out
    );

    let name = "var.correction_out";

    let mut correction_val: f64 = 1.0;
    if let Some(correction) = correction {
        correction_val = scalar_to::<f64>(correction);
    }

    let num: usize = get_reduced_dim_product(in_, &dim_list);
    let denom: f64 = num as f64 - correction_val;

    crate::et_switch_floathbf16_types!(in_.scalar_type(), ctx, name, CTYPE_IN, {
        crate::et_switch_floathbf16_types!(out.scalar_type(), ctx, name, CTYPE_OUT, {
            compute_variance::<CTYPE_IN, CTYPE_OUT>(ctx, in_, out, dim_list, num, denom);
        });
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
    use crate::runtime::core::portable_type::scalar_type::ScalarType;
    use crate::runtime::core::portable_type::{BFloat16, Half};
    use crate::runtime::core::tensor_shape_dynamism::TensorShapeDynamism;

    fn context() -> KernelRuntimeContext<'static> {
        crate::runtime::platform::platform::pal_init();
        KernelRuntimeContext::new(
            crate::extension::module::module::null_event_tracer(),
            core::ptr::null_mut::<MemoryAllocator>() as *mut dyn MemoryAllocatorBase,
        )
    }

    macro_rules! et_expect_kernel_failure {
        ($ctx:expr, $stmt:expr) => {{
            let _ = $stmt;
            assert_ne!(
                $ctx.failure_state(),
                Error::Ok,
                "Expected kernel failure but found success."
            );
        }};
    }

    fn op_var_out<'a, 'b>(
        ctx: &mut KernelRuntimeContext,
        self_: &Tensor,
        dim: Option<ArrayRef<i64>>,
        unbiased: bool,
        keepdim: bool,
        out: &'a Tensor<'b>,
    ) -> &'a Tensor<'b> {
        var_out(ctx, self_, dim, unbiased, keepdim, out)
    }

    fn op_var_correction_out<'a, 'b>(
        ctx: &mut KernelRuntimeContext,
        self_: &Tensor,
        dim: Option<ArrayRef<i64>>,
        correction: &Option<Scalar>,
        keepdim: bool,
        out: &'a Tensor<'b>,
    ) -> &'a Tensor<'b> {
        var_correction_out(ctx, self_, dim, correction, keepdim, out)
    }

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

    fn make_i64<T: FromI64>(vals: &[i64]) -> Vec<T> {
        vals.iter().map(|&v| T::from_i64(v)).collect()
    }
    fn make_f64<T: FromF64>(vals: &[f64]) -> Vec<T> {
        vals.iter().map(|&v| T::from_f64(v)).collect()
    }

    fn ar(v: &[i64]) -> Option<ArrayRef<i64>> {
        Some(ArrayRef::from_raw_parts(v.as_ptr(), v.len()))
    }

    fn expect_tensor_close_with_increased_tol(actual: &Tensor, expected: &Tensor) {
        if actual.scalar_type() == ScalarType::BFloat16 || actual.scalar_type() == ScalarType::Half
        {
            assert!(tensors_are_close(expected, actual, 1e-2, Some(1e-2)));
        } else {
            assert!(tensors_are_close(
                expected,
                actual,
                internal::K_DEFAULT_RTOL,
                None
            ));
        }
    }

    // The {0..23} data reused across many tests.
    fn make_input_23<IN>(tf_in: &TensorFactory<IN>) -> Tensor<'_>
    where
        IN: CppTypeToScalarType + FactoryValue + FromI64,
    {
        tf_in.make_default(
            vec![2, 3, 4],
            make_i64(&[
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                23,
            ]),
        )
    }

    fn test_var_out_invalid_dimensions<IN, OUT>()
    where
        IN: CppTypeToScalarType + FactoryValue + FromI64,
        OUT: CppTypeToScalarType + FactoryValue + FromF64,
    {
        let tf_in = TensorFactory::<IN>::new();
        let tf_out = TensorFactory::<OUT>::new();

        let self_ = make_input_23(&tf_in);
        let out = tf_out.zeros_default(vec![2, 3, 1]);

        // out-of-bound dim in dim list
        let dims_1: [i64; 1] = [3];
        let mut ctx = context();
        et_expect_kernel_failure!(
            ctx,
            op_var_out(&mut ctx, &self_, ar(&dims_1), true, true, &out)
        );

        // the same dim appears multiple times in list of dims
        let dims_2: [i64; 2] = [2, 2];
        let mut ctx = context();
        et_expect_kernel_failure!(
            ctx,
            op_var_out(&mut ctx, &self_, ar(&dims_2), true, true, &out)
        );
    }

    fn test_var_out_invalid_shape<IN, OUT>()
    where
        IN: CppTypeToScalarType + FactoryValue + FromI64,
        OUT: CppTypeToScalarType + FactoryValue + FromF64,
    {
        let tf_in = TensorFactory::<IN>::new();
        let tf_out = TensorFactory::<OUT>::new();

        let self_ = make_input_23(&tf_in);

        // dimension size mismatch when keepdim is true
        let out = tf_out.zeros_default(vec![2, 4]);
        let dims_1: [i64; 1] = [1];
        let mut ctx = context();
        et_expect_kernel_failure!(
            ctx,
            op_var_out(&mut ctx, &self_, ar(&dims_1), true, true, &out)
        );

        // dimension size mismatch when keepdim is false
        let out = tf_out.zeros_default(vec![2, 1, 4]);
        let mut ctx = context();
        et_expect_kernel_failure!(
            ctx,
            op_var_out(&mut ctx, &self_, ar(&dims_1), true, false, &out)
        );
    }

    fn test_var_out_dtype<IN, OUT>()
    where
        IN: CppTypeToScalarType + FactoryValue + FromI64,
        OUT: CppTypeToScalarType + FactoryValue + FromF64,
    {
        let tf_in = TensorFactory::<IN>::new();
        let tf_out = TensorFactory::<OUT>::new();
        let self_ = make_input_23(&tf_in);

        // keepdim=true should work
        let out = tf_out.zeros_default(vec![2, 3, 1]);
        let dims_1: [i64; 1] = [2];
        let mut ctx = context();
        op_var_out(&mut ctx, &self_, ar(&dims_1), true, true, &out);
        expect_tensor_close_with_increased_tol(
            &out,
            &tf_out.make_default(
                vec![2, 3, 1],
                make_f64(&[1.666667, 1.666667, 1.666667, 1.666667, 1.666667, 1.666667]),
            ),
        );

        // keepdim=false should work
        let out = tf_out.zeros_default(vec![2, 3]);
        op_var_out(&mut ctx, &self_, ar(&dims_1), true, false, &out);
        expect_tensor_close_with_increased_tol(
            &out,
            &tf_out.make_default(
                vec![2, 3],
                make_f64(&[1.666667, 1.666667, 1.666667, 1.666667, 1.666667, 1.666667]),
            ),
        );

        // dim list with multiple dimensions should work
        let out = tf_out.zeros_default(vec![1, 1, 4]);
        let dims_2: [i64; 2] = [0, 1];
        op_var_out(&mut ctx, &self_, ar(&dims_2), true, true, &out);
        expect_tensor_close_with_increased_tol(
            &out,
            &tf_out.make_default(vec![1, 1, 4], make_f64(&[56.0, 56.0, 56.0, 56.0])),
        );

        let out = tf_out.zeros_default(vec![4]);
        op_var_out(&mut ctx, &self_, ar(&dims_2), true, false, &out);
        expect_tensor_close_with_increased_tol(
            &out,
            &tf_out.make_default(vec![4], make_f64(&[56.0, 56.0, 56.0, 56.0])),
        );

        // dim list with negative dimensions should work
        let out = tf_out.zeros_default(vec![2, 1, 4]);
        let dims_3: [i64; 1] = [-2];
        op_var_out(&mut ctx, &self_, ar(&dims_3), false, true, &out);
        expect_tensor_close_with_increased_tol(
            &out,
            &tf_out.make_default(
                vec![2, 1, 4],
                make_f64(&[
                    10.666667, 10.666667, 10.666667, 10.666667, 10.666667, 10.666667, 10.666667,
                    10.666667,
                ]),
            ),
        );

        // empty/null dim list should work
        let out = tf_out.zeros_default(vec![1, 1, 1]);
        op_var_out(&mut ctx, &self_, None, true, true, &out);
        expect_tensor_close_with_increased_tol(
            &out,
            &tf_out.make_default(vec![1, 1, 1], make_f64(&[50.0])),
        );

        let empty: [i64; 0] = [];
        op_var_out(&mut ctx, &self_, ar(&empty), false, true, &out);
        expect_tensor_close_with_increased_tol(
            &out,
            &tf_out.make_default(vec![1, 1, 1], make_f64(&[47.916668])),
        );

        let out = tf_out.zeros_default(vec![]);
        op_var_out(&mut ctx, &self_, None, false, false, &out);
        expect_tensor_close_with_increased_tol(
            &out,
            &tf_out.make_default(vec![], make_f64(&[47.916668])),
        );

        op_var_out(&mut ctx, &self_, ar(&empty), true, false, &out);
        expect_tensor_close_with_increased_tol(
            &out,
            &tf_out.make_default(vec![], make_f64(&[50.0])),
        );
    }

    fn test_correction_dtype<T>()
    where
        T: CppTypeToScalarType + FactoryValue + FromF64,
    {
        let tf = TensorFactory::<T>::new();

        let x = tf.make_default(vec![2, 3], make_f64(&[4.9, 4.0, 5.6, 3.8, 4.9, 5.6]));
        let expected = tf.make_default(vec![2], make_f64(&[0.72693, 0.93032]));
        let correction: Option<Scalar> = Some(Scalar::from_double(1.23));
        let out = tf.zeros_default(vec![2]);

        let dims: [i64; 1] = [1];
        let mut ctx = context();
        op_var_correction_out(&mut ctx, &x, ar(&dims), &correction, false, &out);
        expect_tensor_close_with_increased_tol(&out, &expected);
    }

    // ---- OpVarOutTest ----

    // PORT-NOTE: `ET_SKIP_IF(is_aten, ...)`: never ATen, so the body runs.
    // [spec:et:sem:op-var.torch.executor.native.var-out-fn/test]
    #[test]
    fn op_var_out_test_invalid_dimension_list_dies() {
        // ET_FORALL_FLOAT_TYPES x ET_FORALL_FLOAT_TYPES
        test_var_out_invalid_dimensions::<f32, f32>();
        test_var_out_invalid_dimensions::<f32, f64>();
        test_var_out_invalid_dimensions::<f64, f32>();
        test_var_out_invalid_dimensions::<f64, f64>();
    }

    // [spec:et:sem:op-var.torch.executor.native.var-out-fn/test]
    #[test]
    fn op_var_out_test_invalid_shape_dies() {
        test_var_out_invalid_shape::<f32, f32>();
        test_var_out_invalid_shape::<f32, f64>();
        test_var_out_invalid_shape::<f64, f32>();
        test_var_out_invalid_shape::<f64, f64>();
    }

    // [spec:et:sem:op-var.torch.executor.native.var-out-fn/test]
    #[test]
    fn op_var_out_test_invalid_d_type_dies() {
        let tf_float = TensorFactory::<f32>::new();
        let tf_int = TensorFactory::<i32>::new();

        let self_ = make_input_23(&tf_int);

        let out = tf_float.zeros_default(vec![2, 3, 1]);
        let dims_1: [i64; 1] = [2];

        let mut ctx = context();
        et_expect_kernel_failure!(
            ctx,
            op_var_out(&mut ctx, &self_, ar(&dims_1), true, true, &out)
        );
    }

    // [spec:et:sem:op-var.torch.executor.native.var-out-fn/test]
    // also verifies compute_variance: the two-pass mean/sum-of-squares reduction
    // over the dim list produces the checked 1.666667/56.0/10.666667/50.0/47.916668
    // values (num>0, denominator>0 path) across every CTYPE_IN x CTYPE_OUT pair.
    // [spec:et:sem:op-var.torch.executor.native.compute-variance-fn/test]
    #[test]
    fn op_var_out_test_all_float_input_float_output_passes() {
        // ET_FORALL_FLOATHBF16_TYPES x ET_FORALL_FLOATHBF16_TYPES
        macro_rules! row {
            ($in:ty) => {{
                test_var_out_dtype::<$in, f32>();
                test_var_out_dtype::<$in, f64>();
                test_var_out_dtype::<$in, Half>();
                test_var_out_dtype::<$in, BFloat16>();
            }};
        }
        row!(f32);
        row!(f64);
        row!(Half);
        row!(BFloat16);
    }

    // PORT-NOTE: `AllFloatInputFloatOutputPasses_Aten` is `ET_SKIP_IF(!is_aten)`;
    // in the portable build it is always skipped.
    // [spec:et:sem:op-var.torch.executor.native.var-out-fn/test]
    #[test]
    fn op_var_out_test_all_float_input_float_output_passes_aten() {
        println!("Skipping: ATen-specific variant of test case");
    }

    // [spec:et:sem:op-var.torch.executor.native.var-out-fn/test]
    #[test]
    fn op_var_out_test_infinity_and_nan_test() {
        let tf_float = TensorFactory::<f32>::new();
        let self_ = tf_float.make_default(
            vec![2, 3, 4],
            vec![
                0.0,
                1.0,
                2.0,
                f32::INFINITY,
                f32::INFINITY,
                f32::NEG_INFINITY,
                1.0,
                0.0,
                f32::NAN,
                f32::INFINITY,
                f32::NEG_INFINITY,
                2.0,
                f32::NAN,
                f32::NAN,
                1.0,
                0.0,
                0.0,
                f32::INFINITY,
                f32::NAN,
                4.0,
                1.0,
                f32::NAN,
                3.14,
                2.0,
            ],
        );

        let out = tf_float.zeros_default(vec![2, 3, 1]);
        let dims: [i64; 1] = [-1];
        let mut ctx = context();
        op_var_out(&mut ctx, &self_, ar(&dims), true, true, &out);
        assert!(tensors_are_close(
            &out,
            &tf_float.make_default(
                vec![2, 3, 1],
                vec![f32::NAN, f32::NAN, f32::NAN, f32::NAN, f32::NAN, f32::NAN],
            ),
            internal::K_DEFAULT_RTOL,
            None
        ));
    }

    // [spec:et:sem:op-var.torch.executor.native.var-out-fn/test]
    #[test]
    fn op_var_out_test_dynamic_shape_upper_bound_same_as_expected() {
        let tf = TensorFactory::<f32>::new();

        let x = tf.make_default(vec![3, 2], vec![0.49, 0.40, 0.56, 0.38, 0.49, 0.56]);
        let expected_result = tf.make_default(vec![3], vec![0.004050, 0.016200, 0.002450]);

        let out = tf.zeros(vec![3], TensorShapeDynamism::DYNAMIC_BOUND);
        let dims: [i64; 1] = [1];
        let mut ctx = context();
        op_var_out(&mut ctx, &x, ar(&dims), true, false, &out);
        assert!(tensors_are_close(
            &out,
            &expected_result,
            internal::K_DEFAULT_RTOL,
            None
        ));
    }

    // [spec:et:sem:op-var.torch.executor.native.var-out-fn/test]
    #[test]
    fn op_var_out_test_dynamic_shape_upper_bound_larger_than_expected() {
        let tf = TensorFactory::<f32>::new();

        let x = tf.make_default(vec![3, 2], vec![0.49, 0.40, 0.56, 0.38, 0.49, 0.56]);
        let expected_result = tf.make_default(vec![3], vec![0.004050, 0.016200, 0.002450]);

        let out = tf.zeros(vec![10], TensorShapeDynamism::DYNAMIC_BOUND);
        let dims: [i64; 1] = [1];
        let mut ctx = context();
        op_var_out(&mut ctx, &x, ar(&dims), true, false, &out);
        assert!(tensors_are_close(
            &out,
            &expected_result,
            internal::K_DEFAULT_RTOL,
            None
        ));
    }

    // DISABLED: Dynamic shape unbound not supported
    // [spec:et:sem:op-var.torch.executor.native.var-out-fn/test]
    #[test]
    #[ignore = "DISABLED_DynamicShapeUnbound: dynamic shape unbound not supported"]
    fn op_var_out_test_dynamic_shape_unbound() {
        let tf = TensorFactory::<f32>::new();

        let x = tf.make_default(vec![3, 2], vec![0.49, 0.40, 0.56, 0.38, 0.49, 0.56]);
        let expected_result = tf.make_default(vec![3], vec![0.004050, 0.016200, 0.002450]);

        let out = tf.zeros(vec![1], TensorShapeDynamism::DYNAMIC_UNBOUND);
        let dims: [i64; 1] = [1];
        let mut ctx = context();
        op_var_out(&mut ctx, &x, ar(&dims), true, false, &out);
        assert!(tensors_are_close(
            &out,
            &expected_result,
            internal::K_DEFAULT_RTOL,
            None
        ));
    }

    // ---- OpVarCorrectionOutTest ----

    // [spec:et:sem:op-var.torch.executor.native.var-correction-out-fn/test]
    #[test]
    fn op_var_correction_out_test_smoke_test() {
        // ET_FORALL_FLOATHBF16_TYPES
        test_correction_dtype::<f32>();
        test_correction_dtype::<f64>();
        test_correction_dtype::<Half>();
        test_correction_dtype::<BFloat16>();
    }

    // ---- OpVarOutTest::EmptyInput ----

    // [spec:et:sem:op-var.torch.executor.native.var-out-fn/test]
    #[test]
    fn op_var_out_test_empty_input() {
        let tf = TensorFactory::<f32>::new();

        let x = tf.make_default(vec![2, 0, 3], vec![]);
        let unbiased = true;
        let empty: [i64; 0] = [];
        let out = tf.zeros_default(vec![1, 1, 1]);
        let mut ctx = context();
        op_var_out(&mut ctx, &x, ar(&empty), unbiased, true, &out);
        assert!(tensors_are_close(
            &out,
            &tf.make_default(vec![1, 1, 1], vec![f32::NAN]),
            internal::K_DEFAULT_RTOL,
            None
        ));

        let out = tf.zeros_default(vec![]);
        op_var_out(&mut ctx, &x, ar(&empty), unbiased, false, &out);
        assert!(tensors_are_close(
            &out,
            &tf.make_default(vec![], vec![f32::NAN]),
            internal::K_DEFAULT_RTOL,
            None
        ));

        let dims1: [i64; 1] = [1];
        let out = tf.zeros_default(vec![2, 3]);
        op_var_out(&mut ctx, &x, ar(&dims1), unbiased, false, &out);
        assert!(tensors_are_close(
            &out,
            &tf.make_default(
                vec![2, 3],
                vec![f32::NAN, f32::NAN, f32::NAN, f32::NAN, f32::NAN, f32::NAN],
            ),
            internal::K_DEFAULT_RTOL,
            None
        ));

        let dims2: [i64; 1] = [2];
        let out = tf.make_default(vec![2, 0, 1], vec![]);
        op_var_out(&mut ctx, &x, ar(&dims2), unbiased, true, &out);
        assert!(tensors_are_close(
            &out,
            &tf.make_default(vec![2, 0, 1], vec![]),
            internal::K_DEFAULT_RTOL,
            None
        ));
    }
}
