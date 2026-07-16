//! Literal port of kernels/portable/cpu/op_min.cpp.

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
    resize_tensor, tensor_is_default_dim_order, tensors_have_same_dim_order2,
};
use crate::runtime::core::portable_type::scalar_type::ScalarType;
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::core::portable_type::{BFloat16, Half};
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

// PORT-NOTE: the anonymous-namespace `upper_bound<CTYPE>()` (constexpr,
// `lim::has_infinity ? lim::infinity() : lim::max()`) is reproduced as an
// `UpperBound` trait (one impl per REALHBBF16 ctype): floating types
// (`f32`/`f64`/`Half`/`BFloat16`) return `+infinity`; integer types return `MAX`
// (== `numeric_limits::max`); `bool` (no infinity) returns `true` (== `max`).
// [spec:et:def:op-min.torch.executor.native.upper-bound-fn]
// [spec:et:sem:op-min.torch.executor.native.upper-bound-fn]
trait UpperBound {
    fn upper_bound() -> Self;
}
macro_rules! impl_upper_bound_int {
    ($t:ty) => {
        impl UpperBound for $t {
            fn upper_bound() -> Self {
                <$t>::MAX
            }
        }
    };
}
impl_upper_bound_int!(u8);
impl_upper_bound_int!(i8);
impl_upper_bound_int!(i16);
impl_upper_bound_int!(i32);
impl_upper_bound_int!(i64);
macro_rules! impl_upper_bound_float {
    ($t:ty) => {
        impl UpperBound for $t {
            fn upper_bound() -> Self {
                <$t>::INFINITY
            }
        }
    };
}
impl_upper_bound_float!(f32);
impl_upper_bound_float!(f64);
impl_upper_bound_float!(Half);
impl_upper_bound_float!(BFloat16);
impl UpperBound for bool {
    fn upper_bound() -> Self {
        true
    }
}

// PORT-NOTE: `Tensor& out` / returned `Tensor&` become `&Tensor`; the C++
// `std::tuple<Tensor&, Tensor&>` result is a Rust `(&Tensor, &Tensor)` tuple
// (interior mutation through `*mut TensorImpl`). Mirror of op_max.
// `check_min_max_args`, `resize_reduction_out` (single-dim overload =
// `resize_reduction_out_dim`), `reduce_over_dim`, and
// `parallel_for_each_reduce_over_dim_output_index` come from reduce_util.

// [spec:et:def:op-min.torch.executor.native.min-out-fn]
// [spec:et:sem:op-min.torch.executor.native.min-out-fn]
pub fn min_out<'a, 'b, 'c>(
    ctx: &mut KernelRuntimeContext,
    in_: &Tensor,
    mut dim: i64,
    keepdim: bool,
    min: &'a Tensor<'b>,
    min_indices: &'a Tensor<'c>,
) -> (&'a Tensor<'b>, &'a Tensor<'c>) {
    // (void)ctx;

    // PORT-NOTE: `check_min_max_args` lives under the C++ `#ifndef USE_ATEN_LIB`
    // block, so the call is gated to match.
    #[cfg(not(feature = "aten"))]
    crate::et_kernel_check!(
        ctx,
        check_min_max_args(in_, dim, keepdim, min, min_indices),
        InvalidArgument,
        (min, min_indices)
    );

    crate::et_kernel_check!(
        ctx,
        resize_reduction_out_dim(in_, &Some(dim), keepdim, min) == Error::Ok,
        InvalidArgument,
        (min, min_indices)
    );

    crate::et_kernel_check!(
        ctx,
        resize_tensor(min_indices, min.sizes()) == Error::Ok,
        InvalidArgument,
        (min, min_indices)
    );

    crate::et_kernel_check!(
        ctx,
        tensors_have_same_dim_order2(in_, min),
        InvalidArgument,
        (min, min_indices)
    );

    crate::et_kernel_check!(
        ctx,
        tensor_is_default_dim_order(min_indices),
        InvalidArgument,
        (min, min_indices)
    );

    crate::et_kernel_check!(
        ctx,
        tensor_is_default_dim_order(in_),
        InvalidArgument,
        (min, min_indices)
    );

    dim = if dim < 0 { dim + in_.dim() as i64 } else { dim };

    crate::et_switch_realhbbf16_types!(in_.scalar_type(), ctx, "min.dim_min", CTYPE, {
        let min_data: *mut CTYPE = min.mutable_data_ptr::<CTYPE>();
        let min_indices_data: *mut i64 = min_indices.mutable_data_ptr::<i64>();

        let success: bool = parallel_for_each_reduce_over_dim_output_index(
            in_,
            Some(dim),
            min,
            &|begin: i64, end: i64| {
                for out_ix in begin..end {
                    let acc: (CTYPE, i64) = reduce_over_dim::<CTYPE, _>(
                        |v: CTYPE, ix: i64, acc_val: CTYPE, acc_ix: i64| -> (CTYPE, i64) {
                            let mut acc_val = acc_val;
                            let mut acc_ix = acc_ix;
                            if !isnan_override(acc_val) && (isnan_override(v) || v < acc_val) {
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
                        *min_data.offset(out_ix as isize) = acc.0;
                        *min_indices_data.offset(out_ix as isize) = acc.1;
                    }
                }
            },
        );
        crate::et_kernel_check_msg!(
            ctx,
            success,
            Internal,
            (min, min_indices),
            "parallel_for failed"
        );
    });

    (min, min_indices)
}

// [spec:et:def:op-min.torch.executor.native.min-unary-out-fn]
// [spec:et:sem:op-min.torch.executor.native.min-unary-out-fn]
pub fn min_unary_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    in_: &Tensor,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    // (void)ctx;

    crate::et_kernel_check!(
        ctx,
        resize_tensor(out, ArrayRef::<i32>::new()) == Error::Ok,
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

    let name = "min.unary_out";

    crate::et_switch_realhbbf16_types!(in_type, ctx, name, CTYPE_IN, {
        crate::et_switch_realhbbf16_types!(out_type, ctx, name, CTYPE_OUT, {
            let data_in: *const CTYPE_IN = in_.const_data_ptr::<CTYPE_IN>();
            let data_out: *mut CTYPE_OUT = out.mutable_data_ptr::<CTYPE_OUT>();
            unsafe {
                *data_out = <CTYPE_OUT as UpperBound>::upper_bound();
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
                if val < unsafe { *data_out } {
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

    fn op_min_dim_min<'a, 'b, 'c>(
        in_: &Tensor,
        dim: i64,
        keepdim: bool,
        min: &'a Tensor<'b>,
        min_indices: &'a Tensor<'c>,
    ) -> (&'a Tensor<'b>, &'a Tensor<'c>) {
        let mut ctx = context();
        min_out(&mut ctx, in_, dim, keepdim, min, min_indices)
    }

    fn op_min_unary_out<'a, 'b>(self_: &Tensor, out: &'a Tensor<'b>) -> &'a Tensor<'b> {
        let mut ctx = context();
        min_unary_out(&mut ctx, self_, out)
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

    fn test_min_out_invalid_dimensions<IN>()
    where
        IN: CppTypeToScalarType + FactoryValue,
    {
        let tf_in = TensorFactory::<IN>::new();
        let in_ = tf_in.ones_default(vec![2, 3, 4]);

        let mut min = tf_in.zeros_default(vec![2, 3, 2]);
        let mut min_indices = tf_in.zeros_default(vec![2, 3]);
        let mut ctx = context();
        min_out(&mut ctx, &in_, -1, true, &min, &min_indices);
        assert_ne!(ctx.failure_state(), Error::Ok);

        min = tf_in.zeros_default(vec![2, 3, 2]);
        min_indices = tf_in.zeros_default(vec![2, 3, 2]);
        let mut ctx = context();
        min_out(&mut ctx, &in_, -1, true, &min, &min_indices);
        assert_ne!(ctx.failure_state(), Error::Ok);

        min = tf_in.zeros_default(vec![2, 3, 1]);
        min_indices = tf_in.zeros_default(vec![2, 3, 1]);
        let mut ctx = context();
        min_out(&mut ctx, &in_, -1, false, &min, &min_indices);
        assert_ne!(ctx.failure_state(), Error::Ok);

        min = tf_in.zeros_default(vec![2, 3, 1]);
        min_indices = tf_in.zeros_default(vec![2, 3, 1]);
        let mut ctx = context();
        min_out(&mut ctx, &in_, 3, true, &min, &min_indices);
        assert_ne!(ctx.failure_state(), Error::Ok);
    }

    fn test_dynamic_shape(out_shape: Vec<i32>, dynamism: TensorShapeDynamism) {
        let tf = TensorFactory::<f32>::new();
        let tfl = TensorFactory::<i64>::new();

        let input = tf.make_default(
            vec![2, 3, 4],
            vec![
                0.49, 0.76, 0.08, 0.13, 0.30, 0.63, 0.49, 0.89, 0.45, 0.63, 0.34, 0.40, 0.02, 0.16,
                0.29, 0.51, 0.69, 0.80, 0.16, 0.28, 0.68, 0.91, 0.39, 0.87,
            ],
        );
        let expected_min = tf.make_default(
            vec![2, 4],
            vec![0.30, 0.63, 0.08, 0.13, 0.02, 0.16, 0.16, 0.28],
        );
        let expected_min_indices = tfl.make_default(vec![2, 4], vec![1, 1, 0, 0, 0, 0, 1, 1]);
        let min = tf.zeros(out_shape.clone(), dynamism);
        let min_indices = tfl.zeros(out_shape, dynamism);

        op_min_dim_min(&input, 1, false, &min, &min_indices);
        assert_tensor_eq!(min, expected_min);
        assert_tensor_eq!(min_indices, expected_min_indices);
    }

    fn test_min_out_dtype<IN>()
    where
        IN: CppTypeToScalarType + FactoryValue + FromF64Elem,
    {
        let tf_in = TensorFactory::<IN>::new();
        let tf_long = TensorFactory::<i64>::new();
        let d = |v: &[f64]| -> Vec<IN> { v.iter().map(|&x| IN::from_f64(x)).collect() };
        let in_ = tf_in.make_default(
            vec![2, 3, 4],
            d(&[
                0., 1., 2., 4., 4., 2., 1., 0., 1., 0., 4., 2., 4., 2., 1., 0., 0., 1., 2., 4., 1.,
                0., 4., 2.,
            ]),
        );

        let min = tf_in.zeros_default(vec![2, 4]);
        let min_indices = tf_long.zeros_default(vec![2, 4]);
        op_min_dim_min(&in_, 1, false, &min, &min_indices);
        assert_tensor_close!(
            min,
            tf_in.make_default(vec![2, 4], d(&[0., 0., 1., 0., 0., 0., 1., 0.]))
        );
        assert_tensor_eq!(
            min_indices,
            tf_long.make_default(vec![2, 4], vec![0, 2, 1, 1, 1, 2, 0, 0])
        );

        op_min_dim_min(&in_, -2, false, &min, &min_indices);
        assert_tensor_close!(
            min,
            tf_in.make_default(vec![2, 4], d(&[0., 0., 1., 0., 0., 0., 1., 0.]))
        );
        assert_tensor_eq!(
            min_indices,
            tf_long.make_default(vec![2, 4], vec![0, 2, 1, 1, 1, 2, 0, 0])
        );

        let min = tf_in.zeros_default(vec![2, 3, 1]);
        let min_indices = tf_long.zeros_default(vec![2, 3, 1]);
        op_min_dim_min(&in_, -1, true, &min, &min_indices);
        assert_tensor_close!(
            min,
            tf_in.make_default(vec![2, 3, 1], d(&[0., 0., 0., 0., 0., 0.]))
        );
        assert_tensor_eq!(
            min_indices,
            tf_long.make_default(vec![2, 3, 1], vec![0, 3, 1, 3, 0, 1])
        );
    }

    fn test_min_out_dtype_bool() {
        let tf_bool = TensorFactory::<bool>::new();
        let tf_long = TensorFactory::<i64>::new();
        let in_ = tf_bool.make_default(
            vec![2, 3, 4],
            vec![
                true, false, true, false, false, false, false, false, false, true, true, false,
                false, false, true, false, false, false, false, true, true, true, true, true,
            ],
        );
        let min = tf_bool.zeros_default(vec![2, 3, 1]);
        let min_indices = tf_long.zeros_default(vec![2, 3, 1]);
        op_min_dim_min(&in_, -1, true, &min, &min_indices);
        assert_tensor_close!(
            min,
            tf_bool.make_default(vec![2, 3, 1], vec![false, false, false, false, false, true])
        );
        assert_tensor_eq!(
            min_indices,
            tf_long.make_default(vec![2, 3, 1], vec![1, 0, 0, 0, 0, 0])
        );
    }

    fn test_min_unary_out_dtype<IN>()
    where
        IN: CppTypeToScalarType + FactoryValue + FromF64Elem,
    {
        let tf_in = TensorFactory::<IN>::new();
        let tf_out = TensorFactory::<f32>::new();
        let d = |v: &[f64]| -> Vec<IN> { v.iter().map(|&x| IN::from_f64(x)).collect() };
        let input = tf_in.make_default(vec![2, 3], d(&[7., 1., 3., 4., 4., 2.]));
        let out = tf_out.zeros_default(vec![]);
        let expected = tf_out.make_default(vec![], vec![1.0]);
        op_min_unary_out(&input, &out);
        assert_tensor_close!(out, expected);
    }

    fn test_min_unary_out_empty_integer<IN>(max_val: IN)
    where
        IN: CppTypeToScalarType + FactoryValue,
    {
        let tf_in = TensorFactory::<IN>::new();
        let input = tf_in.make_default(vec![2, 0], vec![]);
        let out = tf_in.zeros_default(vec![]);
        let expected = tf_in.make_default(vec![], vec![max_val]);
        op_min_unary_out(&input, &out);
        assert_tensor_close!(out, expected);
    }

    fn test_min_unary_out_empty_floating<IN>(inf: IN)
    where
        IN: CppTypeToScalarType + FactoryValue,
    {
        let tf_in = TensorFactory::<IN>::new();
        let input = tf_in.make_default(vec![2, 0], vec![]);
        let out = tf_in.zeros_default(vec![]);
        let expected = tf_in.make_default(vec![], vec![inf]);
        op_min_unary_out(&input, &out);
        assert_tensor_close!(out, expected);
    }

    // ---- OpMinUnaryOutTest ----

    // [spec:et:sem:op-min.torch.executor.native.min-unary-out-fn/test]
    #[test]
    fn op_min_unary_out_test_all_realhbf16_input_float_output_passes() {
        // ET_FORALL_REALHBF16_TYPES
        test_min_unary_out_dtype::<u8>();
        test_min_unary_out_dtype::<i8>();
        test_min_unary_out_dtype::<i16>();
        test_min_unary_out_dtype::<i32>();
        test_min_unary_out_dtype::<i64>();
        test_min_unary_out_dtype::<f32>();
        test_min_unary_out_dtype::<f64>();
        test_min_unary_out_dtype::<Half>();
        test_min_unary_out_dtype::<BFloat16>();
    }

    // [spec:et:sem:op-min.torch.executor.native.min-unary-out-fn/test]
    // also verifies upper_bound: min over an empty tensor leaves the accumulator at its
    // initial value, which must equal <T>::MAX for every integer ctype.
    // [spec:et:sem:op-min.torch.executor.native.upper-bound-fn/test]
    #[test]
    fn op_min_unary_out_test_empty_integer_input() {
        // ET_FORALL_INT_TYPES
        test_min_unary_out_empty_integer::<u8>(u8::MAX);
        test_min_unary_out_empty_integer::<i8>(i8::MAX);
        test_min_unary_out_empty_integer::<i16>(i16::MAX);
        test_min_unary_out_empty_integer::<i32>(i32::MAX);
        test_min_unary_out_empty_integer::<i64>(i64::MAX);
    }

    // [spec:et:sem:op-min.torch.executor.native.min-unary-out-fn/test]
    // also verifies upper_bound: min over an empty tensor leaves the accumulator at its
    // initial value, which must equal <T>::INFINITY for every floating ctype.
    // [spec:et:sem:op-min.torch.executor.native.upper-bound-fn/test]
    #[test]
    fn op_min_unary_out_test_empty_floating_input() {
        // ET_FORALL_FLOATHBF16_TYPES
        test_min_unary_out_empty_floating::<f32>(f32::INFINITY);
        test_min_unary_out_empty_floating::<f64>(f64::INFINITY);
        test_min_unary_out_empty_floating::<Half>(Half::from_f32(f32::INFINITY));
        test_min_unary_out_empty_floating::<BFloat16>(BFloat16::from_f32(f32::INFINITY));
    }

    // ---- OpMinOutTest ----

    // [spec:et:sem:op-min.torch.executor.native.min-out-fn/test]
    #[test]
    fn op_min_out_test_mismatched_dimensions_dies() {
        // ET_FORALL_REAL_TYPES_AND(Bool, ...)
        test_min_out_invalid_dimensions::<u8>();
        test_min_out_invalid_dimensions::<i8>();
        test_min_out_invalid_dimensions::<i16>();
        test_min_out_invalid_dimensions::<i32>();
        test_min_out_invalid_dimensions::<i64>();
        test_min_out_invalid_dimensions::<f32>();
        test_min_out_invalid_dimensions::<f64>();
        test_min_out_invalid_dimensions::<bool>();
    }

    // [spec:et:sem:op-min.torch.executor.native.min-out-fn/test]
    #[test]
    fn op_min_out_test_mismatched_dtypes_dies() {
        let tf_float = TensorFactory::<f32>::new();
        let tf_long = TensorFactory::<i64>::new();
        let in_ = tf_float.ones_default(vec![2, 3, 4]);

        let min = tf_long.zeros_default(vec![2, 3, 1]);
        let min_indices = tf_long.zeros_default(vec![2, 3, 1]);
        let mut ctx = context();
        min_out(&mut ctx, &in_, -1, true, &min, &min_indices);
        assert_ne!(ctx.failure_state(), Error::Ok);

        let min = tf_float.zeros_default(vec![2, 3, 1]);
        let min_indices = tf_float.zeros_default(vec![2, 3, 1]);
        let mut ctx = context();
        min_out(&mut ctx, &in_, -1, true, &min, &min_indices);
        assert_ne!(ctx.failure_state(), Error::Ok);
    }

    // [spec:et:sem:op-min.torch.executor.native.min-out-fn/test]
    #[test]
    fn op_min_out_test_all_real_input_long_output_passes() {
        // ET_FORALL_REALHBBF16_TYPES
        test_min_out_dtype::<u8>();
        test_min_out_dtype::<i8>();
        test_min_out_dtype::<i16>();
        test_min_out_dtype::<i32>();
        test_min_out_dtype::<i64>();
        test_min_out_dtype::<f32>();
        test_min_out_dtype::<f64>();
        test_min_out_dtype::<Half>();
        test_min_out_dtype::<BFloat16>();
        test_min_out_dtype_bool();
    }

    // [spec:et:sem:op-min.torch.executor.native.min-out-fn/test]
    #[test]
    fn op_min_out_test_infinity_and_nan_test() {
        let tf_float = TensorFactory::<f32>::new();
        let tf_long = TensorFactory::<i64>::new();
        let inf = f32::INFINITY;
        let nan = f32::NAN;
        let in_ = tf_float.make_default(
            vec![2, 3, 4],
            vec![
                0., 1., 2., inf, inf, -inf, 1., 0., nan, inf, -inf, 2., nan, nan, 1., 0., 0., inf,
                nan, 4., 1., nan, 3.14, 2.,
            ],
        );
        let min = tf_float.zeros_default(vec![2, 3, 1]);
        let min_indices = tf_long.zeros_default(vec![2, 3, 1]);
        op_min_dim_min(&in_, -1, true, &min, &min_indices);
        assert_tensor_close!(
            min,
            tf_float.make_default(vec![2, 3, 1], vec![0., -inf, nan, nan, nan, nan])
        );
        assert_tensor_eq!(
            min_indices,
            tf_long.make_default(vec![2, 3, 1], vec![0, 1, 0, 0, 2, 1])
        );
    }

    // [spec:et:sem:op-min.torch.executor.native.min-out-fn/test]
    #[test]
    fn op_min_out_test_dynamic_shape_upper_bound_same_as_expected() {
        test_dynamic_shape(vec![2, 4], TensorShapeDynamism::DYNAMIC_BOUND);
    }

    // [spec:et:sem:op-min.torch.executor.native.min-out-fn/test]
    #[test]
    fn op_min_out_test_dynamic_shape_upper_bound_larger_than_expected() {
        test_dynamic_shape(vec![10, 10], TensorShapeDynamism::DYNAMIC_BOUND);
    }

    // PORT-NOTE: C++ DynamicShapeUnbound is ET_SKIP_IF-guarded on output_resize
    // support (unsupported in portable). Ported and #[ignore]d.
    // [spec:et:sem:op-min.torch.executor.native.min-out-fn/test]
    #[test]
    #[ignore]
    fn op_min_out_test_dynamic_shape_unbound() {
        test_dynamic_shape(vec![1, 1], TensorShapeDynamism::DYNAMIC_UNBOUND);
    }
}
