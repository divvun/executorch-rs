//! Literal port of kernels/portable/cpu/op_max.cpp.

use crate::kernels::portable::cpu::util::dtype_util::StaticCast;
use crate::kernels::portable::cpu::util::math_util::isnan_override;
#[cfg(not(feature = "aten"))]
use crate::kernels::portable::cpu::util::reduce_util::check_min_max_args;
use crate::kernels::portable::cpu::util::reduce_util::{
    parallel_for_each_reduce_over_dim_output_index, reduce_over_dim, resize_reduction_out_dim,
};
use crate::runtime::core::array_ref::ArrayRef;
use crate::runtime::core::error::Error;
use crate::runtime::core::exec_aten::util::scalar_type_util::can_cast;
use crate::runtime::core::exec_aten::util::tensor_util::{
    resize_tensor_same_type, tensor_is_default_dim_order, tensors_have_same_dim_order2,
};
use crate::runtime::core::portable_type::scalar_type::ScalarType;
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::core::portable_type::{BFloat16, Half};
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

// PORT-NOTE: the C++ `lower_bound<CTYPE>()` is a `constexpr` template returning
// `lim::has_infinity ? -lim::infinity() : lim::lowest()`. It is reproduced here
// as a `LowerBound` trait (one impl per REALHBBF16 ctype): floating types
// (`f32`/`f64`/`Half`/`BFloat16`) return `-infinity`; integer types return
// `MIN` (== `numeric_limits::lowest`); `bool` (no infinity) returns `false`
// (== `lowest`).
// [spec:et:def:op-max.torch.executor.native.lower-bound-fn]
// [spec:et:sem:op-max.torch.executor.native.lower-bound-fn]
trait LowerBound {
    fn lower_bound() -> Self;
}
macro_rules! impl_lower_bound_int {
    ($t:ty) => {
        impl LowerBound for $t {
            fn lower_bound() -> Self {
                <$t>::MIN
            }
        }
    };
}
impl_lower_bound_int!(u8);
impl_lower_bound_int!(i8);
impl_lower_bound_int!(i16);
impl_lower_bound_int!(i32);
impl_lower_bound_int!(i64);
macro_rules! impl_lower_bound_float {
    ($t:ty) => {
        impl LowerBound for $t {
            fn lower_bound() -> Self {
                <$t>::NEG_INFINITY
            }
        }
    };
}
impl_lower_bound_float!(f32);
impl_lower_bound_float!(f64);
impl_lower_bound_float!(Half);
impl_lower_bound_float!(BFloat16);
impl LowerBound for bool {
    fn lower_bound() -> Self {
        false
    }
}

// PORT-NOTE: `Tensor& out` / returned `Tensor&` become `&Tensor`; the C++
// `std::tuple<Tensor&, Tensor&>` result is a Rust `(&Tensor, &Tensor)` tuple
// (interior mutation through `*mut TensorImpl`). `check_min_max_args`,
// `resize_reduction_out`, `reduce_over_dim`, and
// `parallel_for_each_reduce_over_dim_output_index` come from reduce_util.

// [spec:et:def:op-max.torch.executor.native.max-out-fn]
// [spec:et:sem:op-max.torch.executor.native.max-out-fn]
#[allow(clippy::too_many_arguments)]
pub fn max_out<'a, 'b, 'c>(
    ctx: &mut KernelRuntimeContext,
    in_: &Tensor,
    mut dim: i64,
    keepdim: bool,
    max: &'a Tensor<'b>,
    max_indices: &'a Tensor<'c>,
) -> (&'a Tensor<'b>, &'a Tensor<'c>) {
    // PORT-NOTE: `check_min_max_args` lives under the C++ `#ifndef USE_ATEN_LIB`
    // block (the portable arg-checkers are absent in the ATen build, which
    // validates elsewhere), so the call is gated to match.
    #[cfg(not(feature = "aten"))]
    crate::et_kernel_check!(
        ctx,
        check_min_max_args(in_, dim, keepdim, max, max_indices),
        InvalidArgument,
        (max, max_indices)
    );

    crate::et_kernel_check!(
        ctx,
        resize_reduction_out_dim(in_, &Some(dim), keepdim, max) == Error::Ok,
        InvalidArgument,
        (max, max_indices)
    );

    crate::et_kernel_check!(
        ctx,
        resize_tensor_same_type(max_indices, max.sizes()) == Error::Ok,
        InvalidArgument,
        (max, max_indices)
    );

    crate::et_kernel_check!(
        ctx,
        tensors_have_same_dim_order2(in_, max),
        InvalidArgument,
        (max, max_indices)
    );

    crate::et_kernel_check!(
        ctx,
        tensor_is_default_dim_order(max_indices),
        InvalidArgument,
        (max, max_indices)
    );

    crate::et_kernel_check!(
        ctx,
        tensor_is_default_dim_order(in_),
        InvalidArgument,
        (max, max_indices)
    );

    dim = if dim < 0 { dim + in_.dim() as i64 } else { dim };

    crate::et_switch_realhbbf16_types!(in_.scalar_type(), ctx, "max.dim_max", CTYPE, {
        let max_data: *mut CTYPE = max.mutable_data_ptr::<CTYPE>();
        let max_indices_data: *mut i64 = max_indices.mutable_data_ptr::<i64>();

        let success: bool = parallel_for_each_reduce_over_dim_output_index(
            in_,
            Some(dim),
            max,
            &|begin: i64, end: i64| {
                for out_ix in begin..end {
                    let acc: (CTYPE, i64) = reduce_over_dim::<CTYPE, _>(
                        |v: CTYPE, ix: i64, acc_val: CTYPE, acc_ix: i64| -> (CTYPE, i64) {
                            let mut acc_val = acc_val;
                            let mut acc_ix = acc_ix;
                            if !isnan_override(acc_val) && (isnan_override(v) || v > acc_val) {
                                acc_val = v;
                                acc_ix = ix;
                            }
                            (acc_val, acc_ix)
                        },
                        in_,
                        &Some(dim),
                        out_ix as usize,
                    );
                    unsafe {
                        *max_data.offset(out_ix as isize) = acc.0;
                        *max_indices_data.offset(out_ix as isize) = acc.1;
                    }
                }
            },
        );
        crate::et_kernel_check_msg!(
            ctx,
            success,
            Internal,
            (max, max_indices),
            "parallel_for failed"
        );
    });

    (max, max_indices)
}

// PORT-NOTE: `Tensor& out` / returned `Tensor&` become `&'a Tensor`.

// [spec:et:def:op-max.torch.executor.native.max-unary-out-fn]
// [spec:et:sem:op-max.torch.executor.native.max-unary-out-fn]
pub fn max_unary_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    in_: &Tensor,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    crate::et_kernel_check!(
        ctx,
        resize_tensor_same_type(out, ArrayRef::<i32>::new()) == Error::Ok,
        InvalidArgument,
        out
    );

    crate::et_kernel_check!(
        ctx,
        tensors_have_same_dim_order2(in_, out),
        InvalidArgument,
        out
    );

    let in_type: ScalarType = in_.scalar_type();
    let out_type: ScalarType = out.scalar_type();

    crate::et_kernel_check!(ctx, can_cast(in_type, out_type), InvalidArgument, out);

    let name = "max.unary_out";

    crate::et_switch_realhbbf16_types!(in_type, ctx, name, CTYPE_IN, {
        crate::et_switch_realhbbf16_types!(out_type, ctx, name, CTYPE_OUT, {
            let data_in: *const CTYPE_IN = in_.const_data_ptr::<CTYPE_IN>();
            let data_out: *mut CTYPE_OUT = out.mutable_data_ptr::<CTYPE_OUT>();
            unsafe {
                *data_out = <CTYPE_OUT as LowerBound>::lower_bound();
            }
            for i in 0..in_.numel() {
                let val: CTYPE_OUT = <CTYPE_OUT as StaticCast<CTYPE_IN>>::static_cast(unsafe {
                    *data_in.offset(i as isize)
                });
                if isnan_override(val) {
                    unsafe {
                        *data_out = val;
                    }
                    break;
                }
                if val > unsafe { *data_out } {
                    unsafe {
                        *data_out = val;
                    }
                }
            }
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
    use crate::runtime::core::exec_aten::testing_util::tensor_util::tensors_are_close;
    use crate::runtime::core::exec_aten::util::scalar_type_util::CppTypeToScalarType;
    use crate::runtime::core::memory_allocator::{MemoryAllocator, MemoryAllocatorBase};
    use crate::runtime::core::tensor_shape_dynamism::TensorShapeDynamism;
    use crate::{assert_tensor_close, assert_tensor_eq};

    fn context() -> KernelRuntimeContext<'static> {
        crate::runtime::platform::platform::pal_init();
        KernelRuntimeContext::new(
            crate::extension::module::module::null_event_tracer(),
            core::ptr::null_mut::<MemoryAllocator>() as *mut dyn MemoryAllocatorBase,
        )
    }

    fn op_max_dim_max<'a, 'b, 'c>(
        self_: &Tensor,
        dim: i64,
        keepdim: bool,
        max: &'a Tensor<'b>,
        max_indices: &'a Tensor<'c>,
    ) -> (&'a Tensor<'b>, &'a Tensor<'c>) {
        let mut ctx = context();
        max_out(&mut ctx, self_, dim, keepdim, max, max_indices)
    }

    fn op_max_unary_out<'a, 'b>(self_: &Tensor, out: &'a Tensor<'b>) -> &'a Tensor<'b> {
        let mut ctx = context();
        max_unary_out(&mut ctx, self_, out)
    }

    trait FromF64Elem: Copy {
        fn from_f64(v: f64) -> Self;
    }
    macro_rules! impl_from_f64_num {
        ($($t:ty),*) => {$(impl FromF64Elem for $t { fn from_f64(v: f64) -> Self { v as $t } })*};
    }
    impl_from_f64_num!(u8, i8, i16, i32, i64, f32, f64);
    impl FromF64Elem for Half {
        fn from_f64(v: f64) -> Self {
            Half::from_f32(v as f32)
        }
    }
    impl FromF64Elem for BFloat16 {
        fn from_f64(v: f64) -> Self {
            BFloat16::from_f32(v as f32)
        }
    }
    impl FromF64Elem for bool {
        fn from_f64(v: f64) -> Self {
            v != 0.0
        }
    }

    fn test_max_out_invalid_dimensions<IN>()
    where
        IN: CppTypeToScalarType + FactoryValue,
    {
        let tf_in = TensorFactory::<IN>::new();
        let self_ = tf_in.ones_default(vec![2, 3, 4]);

        // output tensor dim mismatch
        let mut max = tf_in.zeros_default(vec![2, 3, 2]);
        let mut max_indices = tf_in.zeros_default(vec![2, 3]);
        let mut ctx = context();
        max_out(&mut ctx, &self_, -1, true, &max, &max_indices);
        assert_ne!(ctx.failure_state(), Error::Ok);

        // output tensor shape incorrect: size of dimension dim should be 1
        max = tf_in.zeros_default(vec![2, 3, 2]);
        max_indices = tf_in.zeros_default(vec![2, 3, 2]);
        let mut ctx = context();
        max_out(&mut ctx, &self_, -1, true, &max, &max_indices);
        assert_ne!(ctx.failure_state(), Error::Ok);

        // output tensor shape should be squeezed when keepdim is false
        max = tf_in.zeros_default(vec![2, 3, 1]);
        max_indices = tf_in.zeros_default(vec![2, 3, 1]);
        let mut ctx = context();
        max_out(&mut ctx, &self_, -1, false, &max, &max_indices);
        assert_ne!(ctx.failure_state(), Error::Ok);

        // invalid dim
        max = tf_in.zeros_default(vec![2, 3, 1]);
        max_indices = tf_in.zeros_default(vec![2, 3, 1]);
        let mut ctx = context();
        max_out(&mut ctx, &self_, 3, true, &max, &max_indices);
        assert_ne!(ctx.failure_state(), Error::Ok);
    }

    fn test_dynamic_shape(out_shape: Vec<i32>, dynamism: TensorShapeDynamism) {
        let tf = TensorFactory::<f32>::new();
        let tfl = TensorFactory::<i64>::new();

        let input = tf.make_default(
            vec![2, 3, 4],
            vec![
                0.49625658988952637,
                0.7682217955589294,
                0.08847743272781372,
                0.13203048706054688,
                0.30742281675338745,
                0.6340786814689636,
                0.4900934100151062,
                0.8964447379112244,
                0.455627977848053,
                0.6323062777519226,
                0.3488934636116028,
                0.40171730518341064,
                0.022325754165649414,
                0.16885894536972046,
                0.2938884496688843,
                0.518521785736084,
                0.6976675987243652,
                0.800011396408081,
                0.16102945804595947,
                0.28226858377456665,
                0.6816085577011108,
                0.9151939749717712,
                0.39709991216659546,
                0.8741558790206909,
            ],
        );
        let expected_max = tf.make_default(
            vec![2, 4],
            vec![
                0.49625658988952637,
                0.7682217955589294,
                0.4900934100151062,
                0.8964447379112244,
                0.6976675987243652,
                0.9151939749717712,
                0.39709991216659546,
                0.8741558790206909,
            ],
        );
        let expected_max_indices = tfl.make_default(vec![2, 4], vec![0, 0, 1, 1, 1, 2, 2, 2]);
        let max = tf.zeros(out_shape.clone(), dynamism);
        let max_indices = tfl.zeros(out_shape, dynamism);

        op_max_dim_max(&input, 1, false, &max, &max_indices);
        assert_tensor_eq!(max, expected_max);
        assert_tensor_eq!(max_indices, expected_max_indices);
    }

    fn test_max_out_dtype<IN>()
    where
        IN: CppTypeToScalarType + FactoryValue + FromF64Elem,
    {
        let tf_in = TensorFactory::<IN>::new();
        let tf_long = TensorFactory::<i64>::new();
        let d = |v: &[f64]| -> Vec<IN> { v.iter().map(|&x| IN::from_f64(x)).collect() };
        let self_ = tf_in.make_default(
            vec![2, 3, 4],
            d(&[
                0., 1., 2., 4., 4., 2., 1., 0., 1., 0., 4., 2., 4., 2., 1., 0., 0., 1., 2., 4., 1.,
                0., 4., 2.,
            ]),
        );

        let max = tf_in.zeros_default(vec![2, 4]);
        let max_indices = tf_long.zeros_default(vec![2, 4]);
        op_max_dim_max(&self_, 1, false, &max, &max_indices);
        assert_tensor_close!(
            max,
            tf_in.make_default(vec![2, 4], d(&[4., 2., 4., 4., 4., 2., 4., 4.]))
        );
        assert_tensor_eq!(
            max_indices,
            tf_long.make_default(vec![2, 4], vec![1, 1, 2, 0, 0, 0, 2, 1])
        );

        // negative dim should work
        op_max_dim_max(&self_, -2, false, &max, &max_indices);
        assert_tensor_close!(
            max,
            tf_in.make_default(vec![2, 4], d(&[4., 2., 4., 4., 4., 2., 4., 4.]))
        );
        assert_tensor_eq!(
            max_indices,
            tf_long.make_default(vec![2, 4], vec![1, 1, 2, 0, 0, 0, 2, 1])
        );

        // keepdim should work
        let max = tf_in.zeros_default(vec![2, 3, 1]);
        let max_indices = tf_long.zeros_default(vec![2, 3, 1]);
        op_max_dim_max(&self_, -1, true, &max, &max_indices);
        assert_tensor_close!(
            max,
            tf_in.make_default(vec![2, 3, 1], d(&[4., 4., 4., 4., 4., 4.]))
        );
        assert_tensor_eq!(
            max_indices,
            tf_long.make_default(vec![2, 3, 1], vec![3, 0, 2, 0, 3, 2])
        );
    }

    fn test_max_out_dtype_bool() {
        let tf_bool = TensorFactory::<bool>::new();
        let tf_long = TensorFactory::<i64>::new();
        let self_ = tf_bool.make_default(
            vec![2, 3, 4],
            vec![
                true, false, true, false, false, false, false, false, false, true, true, false,
                false, false, true, false, false, false, false, true, true, true, true, true,
            ],
        );
        let max = tf_bool.zeros_default(vec![2, 3, 1]);
        let max_indices = tf_long.zeros_default(vec![2, 3, 1]);
        op_max_dim_max(&self_, -1, true, &max, &max_indices);
        assert_tensor_close!(
            max,
            tf_bool.make_default(vec![2, 3, 1], vec![true, false, true, true, true, true])
        );
        assert_tensor_eq!(
            max_indices,
            tf_long.make_default(vec![2, 3, 1], vec![0, 0, 1, 2, 3, 0])
        );
    }

    fn test_max_unary_out_dtype<IN>()
    where
        IN: CppTypeToScalarType + FactoryValue + FromF64Elem,
    {
        let tf_in = TensorFactory::<IN>::new();
        let tf_out = TensorFactory::<f32>::new();
        let d = |v: &[f64]| -> Vec<IN> { v.iter().map(|&x| IN::from_f64(x)).collect() };
        let input = tf_in.make_default(vec![2, 3], d(&[0., 1., 2., 4., 4., 2.]));
        let out = tf_out.zeros_default(vec![]);
        let expected = tf_out.make_default(vec![], vec![4.0]);
        op_max_unary_out(&input, &out);
        assert_tensor_close!(out, expected);
    }

    fn test_max_unary_out_empty_integer<IN>(lowest: IN)
    where
        IN: CppTypeToScalarType + FactoryValue,
    {
        let tf_in = TensorFactory::<IN>::new();
        let input = tf_in.make_default(vec![2, 0], vec![]);
        let out = tf_in.zeros_default(vec![]);
        let expected = tf_in.make_default(vec![], vec![lowest]);
        op_max_unary_out(&input, &out);
        assert_tensor_close!(out, expected);
    }

    fn test_max_unary_out_empty_floating<IN>(neg_inf: IN)
    where
        IN: CppTypeToScalarType + FactoryValue,
    {
        let tf_in = TensorFactory::<IN>::new();
        let input = tf_in.make_default(vec![2, 0], vec![]);
        let out = tf_in.zeros_default(vec![]);
        let expected = tf_in.make_default(vec![], vec![neg_inf]);
        op_max_unary_out(&input, &out);
        assert_tensor_close!(out, expected);
    }

    // ---- OpMaxUnaryOutTest ----

    // [spec:et:sem:op-max.torch.executor.native.max-unary-out-fn/test]
    #[test]
    fn op_max_unary_out_test_all_realhbf16_input_float_output_passes() {
        // ET_FORALL_REALHBF16_TYPES
        test_max_unary_out_dtype::<u8>();
        test_max_unary_out_dtype::<i8>();
        test_max_unary_out_dtype::<i16>();
        test_max_unary_out_dtype::<i32>();
        test_max_unary_out_dtype::<i64>();
        test_max_unary_out_dtype::<f32>();
        test_max_unary_out_dtype::<f64>();
        test_max_unary_out_dtype::<Half>();
        test_max_unary_out_dtype::<BFloat16>();
    }

    // [spec:et:sem:op-max.torch.executor.native.max-unary-out-fn/test]
    // also verifies lower_bound: max over an empty tensor leaves the accumulator at its
    // initial value, which must equal <T>::MIN for every integer ctype.
    // [spec:et:sem:op-max.torch.executor.native.lower-bound-fn/test]
    #[test]
    fn op_max_unary_out_test_empty_integer_input() {
        // ET_FORALL_INT_TYPES
        test_max_unary_out_empty_integer::<u8>(u8::MIN);
        test_max_unary_out_empty_integer::<i8>(i8::MIN);
        test_max_unary_out_empty_integer::<i16>(i16::MIN);
        test_max_unary_out_empty_integer::<i32>(i32::MIN);
        test_max_unary_out_empty_integer::<i64>(i64::MIN);
    }

    // [spec:et:sem:op-max.torch.executor.native.max-unary-out-fn/test]
    // also verifies lower_bound: max over an empty tensor leaves the accumulator at its
    // initial value, which must equal <T>::NEG_INFINITY for every floating ctype.
    // [spec:et:sem:op-max.torch.executor.native.lower-bound-fn/test]
    #[test]
    fn op_max_unary_out_test_empty_floating_input() {
        // ET_FORALL_FLOATHBF16_TYPES
        test_max_unary_out_empty_floating::<f32>(f32::NEG_INFINITY);
        test_max_unary_out_empty_floating::<f64>(f64::NEG_INFINITY);
        test_max_unary_out_empty_floating::<Half>(Half::from_f32(f32::NEG_INFINITY));
        test_max_unary_out_empty_floating::<BFloat16>(BFloat16::from_f32(f32::NEG_INFINITY));
    }

    // ---- OpMaxOutTest ----

    // [spec:et:sem:op-max.torch.executor.native.max-out-fn/test]
    #[test]
    fn op_max_out_test_mismatched_dimensions_dies() {
        // ET_FORALL_REAL_TYPES_AND(Bool, ...)
        test_max_out_invalid_dimensions::<u8>();
        test_max_out_invalid_dimensions::<i8>();
        test_max_out_invalid_dimensions::<i16>();
        test_max_out_invalid_dimensions::<i32>();
        test_max_out_invalid_dimensions::<i64>();
        test_max_out_invalid_dimensions::<f32>();
        test_max_out_invalid_dimensions::<f64>();
        test_max_out_invalid_dimensions::<bool>();
    }

    // [spec:et:sem:op-max.torch.executor.native.max-out-fn/test]
    #[test]
    fn op_max_out_test_mismatched_dtypes_dies() {
        let tf_float = TensorFactory::<f32>::new();
        let tf_long = TensorFactory::<i64>::new();
        let self_ = tf_float.ones_default(vec![2, 3, 4]);

        // dtype of self and max should match
        let max = tf_long.zeros_default(vec![2, 3, 1]);
        let max_indices = tf_long.zeros_default(vec![2, 3, 1]);
        let mut ctx = context();
        max_out(&mut ctx, &self_, -1, true, &max, &max_indices);
        assert_ne!(ctx.failure_state(), Error::Ok);

        // max_value tensor should have long as dtype
        let max = tf_float.zeros_default(vec![2, 3, 1]);
        let max_indices = tf_float.zeros_default(vec![2, 3, 1]);
        let mut ctx = context();
        max_out(&mut ctx, &self_, -1, true, &max, &max_indices);
        assert_ne!(ctx.failure_state(), Error::Ok);
    }

    // [spec:et:sem:op-max.torch.executor.native.max-out-fn/test]
    #[test]
    fn op_max_out_test_all_real_input_long_output_passes() {
        // ET_FORALL_REALHBBF16_TYPES
        test_max_out_dtype::<u8>();
        test_max_out_dtype::<i8>();
        test_max_out_dtype::<i16>();
        test_max_out_dtype::<i32>();
        test_max_out_dtype::<i64>();
        test_max_out_dtype::<f32>();
        test_max_out_dtype::<f64>();
        test_max_out_dtype::<Half>();
        test_max_out_dtype::<BFloat16>();
        test_max_out_dtype_bool();
    }

    // [spec:et:sem:op-max.torch.executor.native.max-out-fn/test]
    // also verifies isnan_override float overload (nan-containing rows reduce to nan)
    // [spec:et:sem:math-util.torch.executor.native.utils.isnan-override-fn/test]
    #[test]
    fn op_max_out_test_infinity_and_nan_test() {
        let tf_float = TensorFactory::<f32>::new();
        let tf_long = TensorFactory::<i64>::new();
        let inf = f32::INFINITY;
        let nan = f32::NAN;
        let self_ = tf_float.make_default(
            vec![2, 3, 4],
            vec![
                0., 1., 2., inf, inf, -inf, 1., 0., nan, inf, -inf, 2., nan, nan, 1., 0., 0., inf,
                nan, 4., 1., nan, 3.14, 2.,
            ],
        );
        let max = tf_float.zeros_default(vec![2, 3, 1]);
        let max_indices = tf_long.zeros_default(vec![2, 3, 1]);
        op_max_dim_max(&self_, -1, true, &max, &max_indices);
        assert_tensor_close!(
            max,
            tf_float.make_default(vec![2, 3, 1], vec![inf, inf, nan, nan, nan, nan])
        );
        assert_tensor_eq!(
            max_indices,
            tf_long.make_default(vec![2, 3, 1], vec![3, 0, 0, 0, 2, 1])
        );
    }

    // [spec:et:sem:op-max.torch.executor.native.max-out-fn/test]
    #[test]
    fn op_max_out_test_dynamic_shape_upper_bound_same_as_expected() {
        test_dynamic_shape(vec![2, 4], TensorShapeDynamism::DYNAMIC_BOUND);
    }

    // [spec:et:sem:op-max.torch.executor.native.max-out-fn/test]
    #[test]
    fn op_max_out_test_dynamic_shape_upper_bound_larger_than_expected() {
        test_dynamic_shape(vec![10, 10], TensorShapeDynamism::DYNAMIC_BOUND);
    }

    // PORT-NOTE: C++ DynamicShapeUnbound is ET_SKIP_IF-guarded on output_resize
    // support (unsupported in portable). Ported and #[ignore]d.
    // [spec:et:sem:op-max.torch.executor.native.max-out-fn/test]
    #[test]
    #[ignore]
    fn op_max_out_test_dynamic_shape_unbound() {
        test_dynamic_shape(vec![1, 1], TensorShapeDynamism::DYNAMIC_UNBOUND);
    }
}
