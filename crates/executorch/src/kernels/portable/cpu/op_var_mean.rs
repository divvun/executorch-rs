//! Literal port of kernels/portable/cpu/op_var_mean.cpp.

use crate::kernels::portable::cpu::op_var::VarArith;
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
use crate::runtime::core::exec_aten::util::tensor_util::tensor_is_contiguous;
use crate::runtime::core::portable_type::scalar::Scalar;
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

// PORT-NOTE: `Tensor& var_out/mean_out` become `&Tensor` (interior mutation).
// The two-pass accumulation is in `CTYPE_OUT` (FLOATHBF16); operator semantics
// are supplied by op_var's `VarArith`. The fast path divides variance by the
// CTYPE_OUT-cast denominator (`cdenom`), whereas the general path divides by the
// raw `double` denominator — this discrepancy is preserved.

// [spec:et:def:op-var-mean.torch.executor.native.compute-var-mean-fn]
// [spec:et:sem:op-var-mean.torch.executor.native.compute-var-mean-fn]
#[allow(non_camel_case_types)]
fn compute_var_mean<CTYPE_IN, CTYPE_OUT>(
    ctx: &mut KernelRuntimeContext,
    in_: &Tensor,
    var_out: &Tensor,
    mean_out: &Tensor,
    dim_list: Option<ArrayRef<i64>>,
    num: usize,
    denominator: f64,
) where
    CTYPE_IN: Copy,
    CTYPE_OUT: VarArith + StaticCast<CTYPE_IN>,
{
    let var_data: *mut CTYPE_OUT = var_out.mutable_data_ptr::<CTYPE_OUT>();
    let mean_data: *mut CTYPE_OUT = mean_out.mutable_data_ptr::<CTYPE_OUT>();
    if num == 0 || denominator <= 0.0 {
        for out_ix in 0..var_out.numel() {
            unsafe {
                *var_data.add(out_ix as usize) = CTYPE_OUT::nan();
                *mean_data.add(out_ix as usize) = CTYPE_OUT::nan();
            }
        }
    } else if in_.numel() > 0 {
        // Fast path: contiguous tensor, single innermost dim reduction, same dtype.
        let mut used_fast_path = false;
        if dim_list.is_some()
            && dim_list.unwrap().size() == 1
            && in_.scalar_type() == var_out.scalar_type()
        {
            let d: i64 = if *dim_list.unwrap().at(0) < 0 {
                *dim_list.unwrap().at(0) + in_.dim() as i64
            } else {
                *dim_list.unwrap().at(0)
            };
            if d >= 0
                && d < in_.dim() as i64
                && d == in_.dim() as i64 - 1
                && tensor_is_contiguous(in_)
            {
                used_fast_path = true;
                let reduce_size: i64 = in_.size(d as isize) as i64;
                let outer_size: i64 = in_.numel() as i64 / reduce_size;
                let cnum: CTYPE_OUT = CTYPE_OUT::from_usize(num);
                let cdenom: CTYPE_OUT = CTYPE_OUT::from_f64(denominator);
                let in_data: *const CTYPE_IN = in_.const_data_ptr::<CTYPE_IN>();
                for i in 0..outer_size {
                    let row: *const CTYPE_IN =
                        unsafe { in_data.offset((i * reduce_size) as isize) };
                    // Pass 1: compute mean
                    let mut sum: CTYPE_OUT = CTYPE_OUT::from_usize(0);
                    for j in 0..reduce_size {
                        sum = CTYPE_OUT::add(
                            sum,
                            <CTYPE_OUT as StaticCast<CTYPE_IN>>::static_cast(unsafe {
                                *row.offset(j as isize)
                            }),
                        );
                    }
                    let mean: CTYPE_OUT = CTYPE_OUT::div(sum, cnum);
                    unsafe {
                        *mean_data.offset(i as isize) = mean;
                    }
                    // Pass 2: compute variance
                    let mut sum2: CTYPE_OUT = CTYPE_OUT::from_usize(0);
                    for j in 0..reduce_size {
                        let diff: CTYPE_OUT = CTYPE_OUT::sub(
                            <CTYPE_OUT as StaticCast<CTYPE_IN>>::static_cast(unsafe {
                                *row.offset(j as isize)
                            }),
                            mean,
                        );
                        sum2 = CTYPE_OUT::add(sum2, CTYPE_OUT::mul(diff, diff));
                    }
                    unsafe {
                        *var_data.offset(i as isize) = CTYPE_OUT::div(sum2, cdenom);
                    }
                }
            }
        }
        if !used_fast_path {
            let plan = MapReduceOverDimListPlan::new(in_, &dim_list);
            let success = parallel_for_each_reduce_over_dim_list_output_index(
                in_,
                dim_list,
                var_out,
                &|begin: i64, end: i64| {
                    for out_ix in begin..end {
                        let out_ix = out_ix as usize;
                        // Pass 1: compute sum -> mean
                        let sum: CTYPE_OUT = plan.execute::<CTYPE_IN, CTYPE_OUT, _, _>(
                            |v: CTYPE_IN| <CTYPE_OUT as StaticCast<CTYPE_IN>>::static_cast(v),
                            |outv: CTYPE_OUT, acc: CTYPE_OUT| CTYPE_OUT::add(acc, outv),
                            out_ix,
                        );
                        let mean: CTYPE_OUT = CTYPE_OUT::div(sum, CTYPE_OUT::from_usize(num));
                        unsafe {
                            *mean_data.add(out_ix) = mean;
                        }
                        // Pass 2: compute sum of squared deviations
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
                            *var_data.add(out_ix) = CTYPE_OUT::div_by_double(sum2, denominator);
                        }
                    }
                },
            );
            crate::et_kernel_check_msg!(ctx, success, Internal, (), "parallel_for failed");
        } // !used_fast_path
    }
}

// [spec:et:def:op-var-mean.torch.executor.native.var-mean-correction-out-fn]
// [spec:et:sem:op-var-mean.torch.executor.native.var-mean-correction-out-fn]
pub fn var_mean_correction_out<'a, 'b, 'c, 'd>(
    ctx: &mut KernelRuntimeContext,
    in_: &Tensor,
    dim_list: Option<ArrayRef<i64>>,
    correction: &Option<Scalar>,
    keepdim: bool,
    out0: &'a Tensor<'b>,
    out1: &'c Tensor<'d>,
) -> (&'a Tensor<'b>, &'c Tensor<'d>) {
    // (void)ctx;

    let ret_val = (out0, out1);

    #[cfg(not(feature = "aten"))]
    crate::et_kernel_check!(
        ctx,
        check_reduction_args(in_, &dim_list, keepdim, None, out0),
        InvalidArgument,
        ret_val
    );

    #[cfg(not(feature = "aten"))]
    crate::et_kernel_check!(
        ctx,
        check_reduction_args(in_, &dim_list, keepdim, None, out1),
        InvalidArgument,
        ret_val
    );

    crate::et_kernel_check!(
        ctx,
        resize_reduction_out(in_, &dim_list, keepdim, out0) == Error::Ok,
        InvalidArgument,
        ret_val
    );

    crate::et_kernel_check!(
        ctx,
        resize_reduction_out(in_, &dim_list, keepdim, out1) == Error::Ok,
        InvalidArgument,
        ret_val
    );

    let name = "var_mean.correction_out";

    let mut correction_val: f64 = 1.0;
    if let Some(correction) = correction {
        correction_val = scalar_to::<f64>(correction);
    }

    let num: usize = get_reduced_dim_product(in_, &dim_list);
    let denom: f64 = num as f64 - correction_val;

    crate::et_switch_floathbf16_types!(in_.scalar_type(), ctx, name, CTYPE_IN, {
        crate::et_switch_floathbf16_types!(out0.scalar_type(), ctx, name, CTYPE_OUT, {
            compute_var_mean::<CTYPE_IN, CTYPE_OUT>(ctx, in_, out0, out1, dim_list, num, denom);
        });
    });

    ret_val
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

    #[allow(clippy::too_many_arguments)]
    fn op_var_mean_correction_out<'a, 'b, 'c, 'd>(
        ctx: &mut KernelRuntimeContext,
        self_: &Tensor,
        dim: Option<ArrayRef<i64>>,
        correction: &Option<Scalar>,
        keepdim: bool,
        out0: &'a Tensor<'b>,
        out1: &'c Tensor<'d>,
    ) -> (&'a Tensor<'b>, &'c Tensor<'d>) {
        var_mean_correction_out(ctx, self_, dim, correction, keepdim, out0, out1)
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

    fn test_dtype<T>()
    where
        T: CppTypeToScalarType + FactoryValue + FromF64,
    {
        let tf = TensorFactory::<T>::new();

        let x = tf.make_default(vec![2, 3], make_f64(&[4.9, 4.0, 5.6, 3.8, 4.9, 5.6]));
        let expected_var = tf.make_default(vec![2], make_f64(&[0.72693, 0.93032]));
        let expected_mean = tf.make_default(vec![2], make_f64(&[4.833333, 4.766667]));
        let correction: Option<Scalar> = Some(Scalar::from_double(1.23));
        let var_out = tf.zeros_default(vec![2]);
        let mean_out = tf.zeros_default(vec![2]);

        let dims: [i64; 1] = [1];
        let mut ctx = context();
        op_var_mean_correction_out(
            &mut ctx,
            &x,
            ar(&dims),
            &correction,
            false,
            &var_out,
            &mean_out,
        );
        expect_tensor_close_with_increased_tol(&var_out, &expected_var);
        expect_tensor_close_with_increased_tol(&mean_out, &expected_mean);
    }

    fn test_keepdim<IN, OUT>()
    where
        IN: CppTypeToScalarType + FactoryValue + FromI64,
        OUT: CppTypeToScalarType + FactoryValue + FromF64,
    {
        let tf_in = TensorFactory::<IN>::new();
        let tf_out = TensorFactory::<OUT>::new();
        let self_ = make_input_23(&tf_in);

        // keepdim=true
        let var_out = tf_out.zeros_default(vec![2, 3, 1]);
        let mean_out = tf_out.zeros_default(vec![2, 3, 1]);
        let dims_1: [i64; 1] = [2];
        let correction: Option<Scalar> = Some(Scalar::from_i64(1));
        let mut ctx = context();
        op_var_mean_correction_out(
            &mut ctx,
            &self_,
            ar(&dims_1),
            &correction,
            true,
            &var_out,
            &mean_out,
        );
        expect_tensor_close_with_increased_tol(
            &var_out,
            &tf_out.make_default(
                vec![2, 3, 1],
                make_f64(&[1.666667, 1.666667, 1.666667, 1.666667, 1.666667, 1.666667]),
            ),
        );
        expect_tensor_close_with_increased_tol(
            &mean_out,
            &tf_out.make_default(vec![2, 3, 1], make_f64(&[1.5, 5.5, 9.5, 13.5, 17.5, 21.5])),
        );

        // keepdim=false
        let var_out = tf_out.zeros_default(vec![2, 3]);
        let mean_out = tf_out.zeros_default(vec![2, 3]);
        op_var_mean_correction_out(
            &mut ctx,
            &self_,
            ar(&dims_1),
            &correction,
            false,
            &var_out,
            &mean_out,
        );
        expect_tensor_close_with_increased_tol(
            &var_out,
            &tf_out.make_default(
                vec![2, 3],
                make_f64(&[1.666667, 1.666667, 1.666667, 1.666667, 1.666667, 1.666667]),
            ),
        );
        expect_tensor_close_with_increased_tol(
            &mean_out,
            &tf_out.make_default(vec![2, 3], make_f64(&[1.5, 5.5, 9.5, 13.5, 17.5, 21.5])),
        );
    }

    fn test_multiple_dims<IN, OUT>()
    where
        IN: CppTypeToScalarType + FactoryValue + FromI64,
        OUT: CppTypeToScalarType + FactoryValue + FromF64,
    {
        let tf_in = TensorFactory::<IN>::new();
        let tf_out = TensorFactory::<OUT>::new();
        let self_ = make_input_23(&tf_in);

        let var_out = tf_out.zeros_default(vec![1, 1, 4]);
        let mean_out = tf_out.zeros_default(vec![1, 1, 4]);
        let dims: [i64; 2] = [0, 1];
        let correction: Option<Scalar> = Some(Scalar::from_i64(1));
        let mut ctx = context();
        op_var_mean_correction_out(
            &mut ctx,
            &self_,
            ar(&dims),
            &correction,
            true,
            &var_out,
            &mean_out,
        );
        expect_tensor_close_with_increased_tol(
            &var_out,
            &tf_out.make_default(vec![1, 1, 4], make_f64(&[56.0, 56.0, 56.0, 56.0])),
        );
        expect_tensor_close_with_increased_tol(
            &mean_out,
            &tf_out.make_default(vec![1, 1, 4], make_f64(&[10.0, 11.0, 12.0, 13.0])),
        );

        let var_out = tf_out.zeros_default(vec![4]);
        let mean_out = tf_out.zeros_default(vec![4]);
        op_var_mean_correction_out(
            &mut ctx,
            &self_,
            ar(&dims),
            &correction,
            false,
            &var_out,
            &mean_out,
        );
        expect_tensor_close_with_increased_tol(
            &var_out,
            &tf_out.make_default(vec![4], make_f64(&[56.0, 56.0, 56.0, 56.0])),
        );
        expect_tensor_close_with_increased_tol(
            &mean_out,
            &tf_out.make_default(vec![4], make_f64(&[10.0, 11.0, 12.0, 13.0])),
        );
    }

    fn test_negative_dim<IN, OUT>()
    where
        IN: CppTypeToScalarType + FactoryValue + FromI64,
        OUT: CppTypeToScalarType + FactoryValue + FromF64,
    {
        let tf_in = TensorFactory::<IN>::new();
        let tf_out = TensorFactory::<OUT>::new();
        let self_ = make_input_23(&tf_in);

        let var_out = tf_out.zeros_default(vec![2, 1, 4]);
        let mean_out = tf_out.zeros_default(vec![2, 1, 4]);
        let dims: [i64; 1] = [-2];
        let correction: Option<Scalar> = Some(Scalar::from_i64(0));
        let mut ctx = context();
        op_var_mean_correction_out(
            &mut ctx,
            &self_,
            ar(&dims),
            &correction,
            true,
            &var_out,
            &mean_out,
        );
        expect_tensor_close_with_increased_tol(
            &var_out,
            &tf_out.make_default(
                vec![2, 1, 4],
                make_f64(&[
                    10.666667, 10.666667, 10.666667, 10.666667, 10.666667, 10.666667, 10.666667,
                    10.666667,
                ]),
            ),
        );
        expect_tensor_close_with_increased_tol(
            &mean_out,
            &tf_out.make_default(
                vec![2, 1, 4],
                make_f64(&[4.0, 5.0, 6.0, 7.0, 16.0, 17.0, 18.0, 19.0]),
            ),
        );
    }

    fn test_null_and_empty_dim_list<IN, OUT>()
    where
        IN: CppTypeToScalarType + FactoryValue + FromI64,
        OUT: CppTypeToScalarType + FactoryValue + FromF64,
    {
        let tf_in = TensorFactory::<IN>::new();
        let tf_out = TensorFactory::<OUT>::new();
        let self_ = make_input_23(&tf_in);

        // null dim list, correction=1 (unbiased), keepdim=true
        let var_out = tf_out.zeros_default(vec![1, 1, 1]);
        let mean_out = tf_out.zeros_default(vec![1, 1, 1]);
        let correction: Option<Scalar> = Some(Scalar::from_i64(1));
        let mut ctx = context();
        op_var_mean_correction_out(
            &mut ctx,
            &self_,
            None,
            &correction,
            true,
            &var_out,
            &mean_out,
        );
        expect_tensor_close_with_increased_tol(
            &var_out,
            &tf_out.make_default(vec![1, 1, 1], make_f64(&[50.0])),
        );
        expect_tensor_close_with_increased_tol(
            &mean_out,
            &tf_out.make_default(vec![1, 1, 1], make_f64(&[11.5])),
        );

        // empty dim list, correction=0 (population), keepdim=true
        let empty: [i64; 0] = [];
        let correction_zero: Option<Scalar> = Some(Scalar::from_i64(0));
        op_var_mean_correction_out(
            &mut ctx,
            &self_,
            ar(&empty),
            &correction_zero,
            true,
            &var_out,
            &mean_out,
        );
        expect_tensor_close_with_increased_tol(
            &var_out,
            &tf_out.make_default(vec![1, 1, 1], make_f64(&[47.916668])),
        );
        expect_tensor_close_with_increased_tol(
            &mean_out,
            &tf_out.make_default(vec![1, 1, 1], make_f64(&[11.5])),
        );

        // null dim list, correction=0, keepdim=false
        let var_out = tf_out.zeros_default(vec![]);
        let mean_out = tf_out.zeros_default(vec![]);
        op_var_mean_correction_out(
            &mut ctx,
            &self_,
            None,
            &correction_zero,
            false,
            &var_out,
            &mean_out,
        );
        expect_tensor_close_with_increased_tol(
            &var_out,
            &tf_out.make_default(vec![], make_f64(&[47.916668])),
        );
        expect_tensor_close_with_increased_tol(
            &mean_out,
            &tf_out.make_default(vec![], make_f64(&[11.5])),
        );

        // empty dim list, correction=1, keepdim=false
        op_var_mean_correction_out(
            &mut ctx,
            &self_,
            ar(&empty),
            &correction,
            false,
            &var_out,
            &mean_out,
        );
        expect_tensor_close_with_increased_tol(
            &var_out,
            &tf_out.make_default(vec![], make_f64(&[50.0])),
        );
        expect_tensor_close_with_increased_tol(
            &mean_out,
            &tf_out.make_default(vec![], make_f64(&[11.5])),
        );
    }

    fn test_invalid_dimensions<IN, OUT>()
    where
        IN: CppTypeToScalarType + FactoryValue + FromI64,
        OUT: CppTypeToScalarType + FactoryValue + FromF64,
    {
        let tf_in = TensorFactory::<IN>::new();
        let tf_out = TensorFactory::<OUT>::new();
        let self_ = make_input_23(&tf_in);
        let var_out = tf_out.zeros_default(vec![2, 3, 1]);
        let mean_out = tf_out.zeros_default(vec![2, 3, 1]);
        let correction: Option<Scalar> = Some(Scalar::from_i64(1));

        // out-of-bound dim
        let dims_1: [i64; 1] = [3];
        let mut ctx = context();
        et_expect_kernel_failure!(
            ctx,
            op_var_mean_correction_out(
                &mut ctx,
                &self_,
                ar(&dims_1),
                &correction,
                true,
                &var_out,
                &mean_out
            )
        );

        // duplicate dim
        let dims_2: [i64; 2] = [2, 2];
        let mut ctx = context();
        et_expect_kernel_failure!(
            ctx,
            op_var_mean_correction_out(
                &mut ctx,
                &self_,
                ar(&dims_2),
                &correction,
                true,
                &var_out,
                &mean_out
            )
        );
    }

    // ---- OpVarMeanCorrectionOutTest ----

    // [spec:et:sem:op-var-mean.torch.executor.native.var-mean-correction-out-fn/test]
    #[test]
    fn op_var_mean_correction_out_test_smoke_test() {
        // ET_FORALL_FLOATHBF16_TYPES
        test_dtype::<f32>();
        test_dtype::<f64>();
        test_dtype::<Half>();
        test_dtype::<BFloat16>();
    }

    macro_rules! forall_floathbf16_x2 {
        ($f:ident) => {{
            macro_rules! row {
                ($in:ty) => {{
                    $f::<$in, f32>();
                    $f::<$in, f64>();
                    $f::<$in, Half>();
                    $f::<$in, BFloat16>();
                }};
            }
            row!(f32);
            row!(f64);
            row!(Half);
            row!(BFloat16);
        }};
    }

    // PORT-NOTE: `ET_SKIP_IF(is_aten, ...)`: never ATen, so the body runs.
    // [spec:et:sem:op-var-mean.torch.executor.native.var-mean-correction-out-fn/test]
    // also verifies compute_var_mean: reducing dim 2 (the innermost) with in==out
    // dtype on the T,T rows drives the contiguous fast path, while the mixed-dtype
    // rows drive the general dim-list path; both pin the checked mean (1.5/5.5/...)
    // and variance (1.666667) two-pass results.
    // [spec:et:sem:op-var-mean.torch.executor.native.compute-var-mean-fn/test]
    #[test]
    fn op_var_mean_correction_out_test_keep_dim() {
        forall_floathbf16_x2!(test_keepdim);
    }

    // PORT-NOTE: `KeepDim_Aten` is `ET_SKIP_IF(!is_aten)`; always skipped in portable.
    // [spec:et:sem:op-var-mean.torch.executor.native.var-mean-correction-out-fn/test]
    #[test]
    fn op_var_mean_correction_out_test_keep_dim_aten() {
        println!("Skipping: ATen-specific variant of test case");
    }

    // [spec:et:sem:op-var-mean.torch.executor.native.var-mean-correction-out-fn/test]
    #[test]
    fn op_var_mean_correction_out_test_multiple_dims() {
        forall_floathbf16_x2!(test_multiple_dims);
    }

    // [spec:et:sem:op-var-mean.torch.executor.native.var-mean-correction-out-fn/test]
    #[test]
    fn op_var_mean_correction_out_test_multiple_dims_aten() {
        println!("Skipping: ATen-specific variant of test case");
    }

    // [spec:et:sem:op-var-mean.torch.executor.native.var-mean-correction-out-fn/test]
    #[test]
    fn op_var_mean_correction_out_test_negative_dim() {
        forall_floathbf16_x2!(test_negative_dim);
    }

    // [spec:et:sem:op-var-mean.torch.executor.native.var-mean-correction-out-fn/test]
    #[test]
    fn op_var_mean_correction_out_test_negative_dim_aten() {
        println!("Skipping: ATen-specific variant of test case");
    }

    // [spec:et:sem:op-var-mean.torch.executor.native.var-mean-correction-out-fn/test]
    #[test]
    fn op_var_mean_correction_out_test_null_and_empty_dim_list() {
        forall_floathbf16_x2!(test_null_and_empty_dim_list);
    }

    // [spec:et:sem:op-var-mean.torch.executor.native.var-mean-correction-out-fn/test]
    #[test]
    fn op_var_mean_correction_out_test_null_and_empty_dim_list_aten() {
        println!("Skipping: ATen-specific variant of test case");
    }

    // [spec:et:sem:op-var-mean.torch.executor.native.var-mean-correction-out-fn/test]
    #[test]
    fn op_var_mean_correction_out_test_invalid_dimension_list_dies() {
        // ET_FORALL_FLOAT_TYPES x ET_FORALL_FLOAT_TYPES
        test_invalid_dimensions::<f32, f32>();
        test_invalid_dimensions::<f32, f64>();
        test_invalid_dimensions::<f64, f32>();
        test_invalid_dimensions::<f64, f64>();
    }

    // [spec:et:sem:op-var-mean.torch.executor.native.var-mean-correction-out-fn/test]
    #[test]
    fn op_var_mean_correction_out_test_invalid_d_type_dies() {
        let tf_float = TensorFactory::<f32>::new();
        let tf_int = TensorFactory::<i32>::new();

        let self_ = make_input_23(&tf_int);

        let var_out = tf_float.zeros_default(vec![2, 3, 1]);
        let mean_out = tf_float.zeros_default(vec![2, 3, 1]);
        let dims_1: [i64; 1] = [2];
        let correction: Option<Scalar> = Some(Scalar::from_i64(1));

        let mut ctx = context();
        et_expect_kernel_failure!(
            ctx,
            op_var_mean_correction_out(
                &mut ctx,
                &self_,
                ar(&dims_1),
                &correction,
                true,
                &var_out,
                &mean_out
            )
        );
    }

    // [spec:et:sem:op-var-mean.torch.executor.native.var-mean-correction-out-fn/test]
    #[test]
    fn op_var_mean_correction_out_test_empty_input() {
        let tf = TensorFactory::<f32>::new();

        let x = tf.make_default(vec![2, 0, 3], vec![]);
        let correction: Option<Scalar> = Some(Scalar::from_i64(1));

        // empty dim list, correction=1, keepdim=true
        let empty: [i64; 0] = [];
        let var_out = tf.zeros_default(vec![1, 1, 1]);
        let mean_out = tf.zeros_default(vec![1, 1, 1]);
        let mut ctx = context();
        op_var_mean_correction_out(
            &mut ctx,
            &x,
            ar(&empty),
            &correction,
            true,
            &var_out,
            &mean_out,
        );
        assert!(tensors_are_close(
            &var_out,
            &tf.make_default(vec![1, 1, 1], vec![f32::NAN]),
            internal::K_DEFAULT_RTOL,
            None
        ));
        assert!(tensors_are_close(
            &mean_out,
            &tf.make_default(vec![1, 1, 1], vec![f32::NAN]),
            internal::K_DEFAULT_RTOL,
            None
        ));

        // empty dim list, correction=1, keepdim=false
        let var_out = tf.zeros_default(vec![]);
        let mean_out = tf.zeros_default(vec![]);
        op_var_mean_correction_out(
            &mut ctx,
            &x,
            ar(&empty),
            &correction,
            false,
            &var_out,
            &mean_out,
        );
        assert!(tensors_are_close(
            &var_out,
            &tf.make_default(vec![], vec![f32::NAN]),
            internal::K_DEFAULT_RTOL,
            None
        ));
        assert!(tensors_are_close(
            &mean_out,
            &tf.make_default(vec![], vec![f32::NAN]),
            internal::K_DEFAULT_RTOL,
            None
        ));

        // reduce along the empty dim
        let dims1: [i64; 1] = [1];
        let var_out = tf.zeros_default(vec![2, 3]);
        let mean_out = tf.zeros_default(vec![2, 3]);
        op_var_mean_correction_out(
            &mut ctx,
            &x,
            ar(&dims1),
            &correction,
            false,
            &var_out,
            &mean_out,
        );
        assert!(tensors_are_close(
            &var_out,
            &tf.make_default(
                vec![2, 3],
                vec![f32::NAN, f32::NAN, f32::NAN, f32::NAN, f32::NAN, f32::NAN]
            ),
            internal::K_DEFAULT_RTOL,
            None
        ));
        assert!(tensors_are_close(
            &mean_out,
            &tf.make_default(
                vec![2, 3],
                vec![f32::NAN, f32::NAN, f32::NAN, f32::NAN, f32::NAN, f32::NAN]
            ),
            internal::K_DEFAULT_RTOL,
            None
        ));

        // reduce along a non-empty dim of an empty tensor
        let dims2: [i64; 1] = [2];
        let var_out = tf.make_default(vec![2, 0, 1], vec![]);
        let mean_out = tf.make_default(vec![2, 0, 1], vec![]);
        op_var_mean_correction_out(
            &mut ctx,
            &x,
            ar(&dims2),
            &correction,
            true,
            &var_out,
            &mean_out,
        );
        assert!(tensors_are_close(
            &var_out,
            &tf.make_default(vec![2, 0, 1], vec![]),
            internal::K_DEFAULT_RTOL,
            None
        ));
        assert!(tensors_are_close(
            &mean_out,
            &tf.make_default(vec![2, 0, 1], vec![]),
            internal::K_DEFAULT_RTOL,
            None
        ));
    }

    // [spec:et:sem:op-var-mean.torch.executor.native.var-mean-correction-out-fn/test]
    #[test]
    fn op_var_mean_correction_out_test_dynamic_shape_upper_bound_same_as_expected() {
        let tf = TensorFactory::<f32>::new();

        let x = tf.make_default(vec![3, 2], vec![0.49, 0.40, 0.56, 0.38, 0.49, 0.56]);
        let expected_var = tf.make_default(vec![3], vec![0.004050, 0.016200, 0.002450]);
        let expected_mean = tf.make_default(vec![3], vec![0.445, 0.47, 0.525]);
        let correction: Option<Scalar> = Some(Scalar::from_i64(1));

        let var_out = tf.zeros(vec![3], TensorShapeDynamism::DYNAMIC_BOUND);
        let mean_out = tf.zeros(vec![3], TensorShapeDynamism::DYNAMIC_BOUND);
        let dims: [i64; 1] = [1];
        let mut ctx = context();
        op_var_mean_correction_out(
            &mut ctx,
            &x,
            ar(&dims),
            &correction,
            false,
            &var_out,
            &mean_out,
        );
        assert!(tensors_are_close(
            &var_out,
            &expected_var,
            internal::K_DEFAULT_RTOL,
            None
        ));
        assert!(tensors_are_close(
            &mean_out,
            &expected_mean,
            internal::K_DEFAULT_RTOL,
            None
        ));
    }

    // [spec:et:sem:op-var-mean.torch.executor.native.var-mean-correction-out-fn/test]
    #[test]
    fn op_var_mean_correction_out_test_dynamic_shape_upper_bound_larger_than_expected() {
        let tf = TensorFactory::<f32>::new();

        let x = tf.make_default(vec![3, 2], vec![0.49, 0.40, 0.56, 0.38, 0.49, 0.56]);
        let expected_var = tf.make_default(vec![3], vec![0.004050, 0.016200, 0.002450]);
        let expected_mean = tf.make_default(vec![3], vec![0.445, 0.47, 0.525]);
        let correction: Option<Scalar> = Some(Scalar::from_i64(1));

        let var_out = tf.zeros(vec![10], TensorShapeDynamism::DYNAMIC_BOUND);
        let mean_out = tf.zeros(vec![10], TensorShapeDynamism::DYNAMIC_BOUND);
        let dims: [i64; 1] = [1];
        let mut ctx = context();
        op_var_mean_correction_out(
            &mut ctx,
            &x,
            ar(&dims),
            &correction,
            false,
            &var_out,
            &mean_out,
        );
        assert!(tensors_are_close(
            &var_out,
            &expected_var,
            internal::K_DEFAULT_RTOL,
            None
        ));
        assert!(tensors_are_close(
            &mean_out,
            &expected_mean,
            internal::K_DEFAULT_RTOL,
            None
        ));
    }

    // DISABLED: Dynamic shape unbound not supported
    // [spec:et:sem:op-var-mean.torch.executor.native.var-mean-correction-out-fn/test]
    #[test]
    #[ignore = "DISABLED_DynamicShapeUnbound: dynamic shape unbound not supported"]
    fn op_var_mean_correction_out_test_dynamic_shape_unbound() {
        let tf = TensorFactory::<f32>::new();

        let x = tf.make_default(vec![3, 2], vec![0.49, 0.40, 0.56, 0.38, 0.49, 0.56]);
        let expected_var = tf.make_default(vec![3], vec![0.004050, 0.016200, 0.002450]);
        let expected_mean = tf.make_default(vec![3], vec![0.445, 0.47, 0.525]);
        let correction: Option<Scalar> = Some(Scalar::from_i64(1));

        let var_out = tf.zeros(vec![1], TensorShapeDynamism::DYNAMIC_UNBOUND);
        let mean_out = tf.zeros(vec![1], TensorShapeDynamism::DYNAMIC_UNBOUND);
        let dims: [i64; 1] = [1];
        let mut ctx = context();
        op_var_mean_correction_out(
            &mut ctx,
            &x,
            ar(&dims),
            &correction,
            false,
            &var_out,
            &mean_out,
        );
        assert!(tensors_are_close(
            &var_out,
            &expected_var,
            internal::K_DEFAULT_RTOL,
            None
        ));
        assert!(tensors_are_close(
            &mean_out,
            &expected_mean,
            internal::K_DEFAULT_RTOL,
            None
        ));
    }

    // [spec:et:sem:op-var-mean.torch.executor.native.var-mean-correction-out-fn/test]
    #[test]
    fn op_var_mean_correction_out_test_infinity_and_nan_test() {
        let tf = TensorFactory::<f32>::new();
        let self_ = tf.make_default(
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

        let var_out = tf.zeros_default(vec![2, 3, 1]);
        let mean_out = tf.zeros_default(vec![2, 3, 1]);
        let dims: [i64; 1] = [-1];
        let correction: Option<Scalar> = Some(Scalar::from_i64(1));
        let mut ctx = context();
        op_var_mean_correction_out(
            &mut ctx,
            &self_,
            ar(&dims),
            &correction,
            true,
            &var_out,
            &mean_out,
        );
        // Only check var (all rows contain INFINITY or NAN -> var is NAN).
        assert!(tensors_are_close(
            &var_out,
            &tf.make_default(
                vec![2, 3, 1],
                vec![f32::NAN, f32::NAN, f32::NAN, f32::NAN, f32::NAN, f32::NAN],
            ),
            internal::K_DEFAULT_RTOL,
            None
        ));
    }
}
