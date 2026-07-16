//! Literal port of kernels/portable/cpu/op_floor_divide.cpp.

use crate::kernels::portable::cpu::util::broadcast_util::resize_to_broadcast_target_size;
use crate::kernels::portable::cpu::util::dtype_util::SupportedTensorDtypes;
use crate::kernels::portable::cpu::util::elementwise_util::{
    apply_bitensor_elementwise_fn, get_compute_type,
};
use crate::kernels::portable::cpu::util::math_util::floor_divide;
use crate::runtime::core::error::Error;
use crate::runtime::core::exec_aten::util::scalar_type_util::{can_cast, promote_types};
use crate::runtime::core::exec_aten::util::tensor_util::tensors_have_same_dim_order3;
use crate::runtime::core::portable_type::scalar_type::ScalarType;
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

// PORT-NOTE: `Tensor& out` / returned `Tensor&` become `&'a Tensor` (interior
// mutation through `*mut TensorImpl`).
//
// PORT-NOTE: the closure needs the compile-time
// `is_integral_type<CTYPE_COMPUTE, /*includeBool=*/true>::value` test, a zero
// literal, and `val_b == 0`. Modeled by a `FloorDivCompute` trait (one impl per
// REAL ctype) so the generic closure carries the per-type body verbatim, mirroring
// op_div.rs's `RealCompute` strategy. `floor_divide` itself comes from math_util.
trait FloorDivCompute: Copy {
    const IS_INTEGRAL: bool;
    fn is_zero(self) -> bool;
    fn zero() -> Self;
}
macro_rules! impl_floor_div_compute_int {
    ($t:ty) => {
        impl FloorDivCompute for $t {
            const IS_INTEGRAL: bool = true;
            fn is_zero(self) -> bool {
                self == 0 as $t
            }
            fn zero() -> Self {
                0 as $t
            }
        }
    };
}
impl_floor_div_compute_int!(u8);
impl_floor_div_compute_int!(i8);
impl_floor_div_compute_int!(i16);
impl_floor_div_compute_int!(i32);
impl_floor_div_compute_int!(i64);
macro_rules! impl_floor_div_compute_float {
    ($t:ty) => {
        impl FloorDivCompute for $t {
            const IS_INTEGRAL: bool = false;
            fn is_zero(self) -> bool {
                self == 0 as $t
            }
            fn zero() -> Self {
                0 as $t
            }
        }
    };
}
impl_floor_div_compute_float!(f32);
impl_floor_div_compute_float!(f64);

// [spec:et:def:op-floor-divide.torch.executor.native.floor-divide-out-fn]
// [spec:et:sem:op-floor-divide.torch.executor.native.floor-divide-out-fn]
pub fn floor_divide_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    a: &Tensor,
    b: &Tensor,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    // Common Dtype
    let common_type = promote_types(a.scalar_type(), b.scalar_type(), false);

    // Check Common Dtype
    crate::et_kernel_check!(
        ctx,
        can_cast(common_type, out.scalar_type()) && common_type != ScalarType::Bool,
        InvalidArgument,
        out
    );

    // Check Dim Order
    crate::et_kernel_check!(
        ctx,
        tensors_have_same_dim_order3(a, b, out),
        InvalidArgument,
        out
    );

    // Resize
    crate::et_kernel_check!(
        ctx,
        resize_to_broadcast_target_size(a, b, out) == Error::Ok,
        InvalidArgument,
        out
    );

    // Compute Dtype
    let mut common_type_mut = common_type;
    let compute_type = get_compute_type(&mut common_type_mut);

    let op_name = "floor_divide.out";

    // PORT-NOTE: the C++ lambda captures `&div_by_zero_error` and writes through
    // the reference while the closure object itself stays const (`Fn`). A `Cell`
    // reproduces that interior mutation without widening the util's `Fn` bound
    // (same as op_div.rs).
    let div_by_zero_error = core::cell::Cell::new(false);

    crate::et_switch_real_types!(compute_type, ctx, op_name, CTYPE_COMPUTE, {
        apply_bitensor_elementwise_fn::<CTYPE_COMPUTE, _>(
            |vals: &[CTYPE_COMPUTE]| -> CTYPE_COMPUTE {
                let val_a: CTYPE_COMPUTE = vals[0];
                let val_b: CTYPE_COMPUTE = vals[1];
                // TODO: rewrite this to be vectorization-capable.
                if <CTYPE_COMPUTE as FloorDivCompute>::IS_INTEGRAL {
                    if val_b.is_zero() {
                        div_by_zero_error.set(true);
                        return <CTYPE_COMPUTE as FloorDivCompute>::zero();
                    }
                }
                floor_divide(val_a, val_b)
            },
            ctx,
            a,
            SupportedTensorDtypes::REALHBBF16,
            b,
            SupportedTensorDtypes::REALHBBF16,
            out,
            SupportedTensorDtypes::REALHBF16,
            /*support_noncontiguous*/ false,
        );
    });

    crate::et_kernel_check_msg!(
        ctx,
        !div_by_zero_error.get(),
        InvalidArgument,
        out,
        "Floor divide operation encountered integer division by zero"
    );

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

    fn setup() {
        crate::runtime::platform::platform::pal_init();
    }

    fn context() -> KernelRuntimeContext<'static> {
        setup();
        KernelRuntimeContext::new(
            crate::extension::module::module::null_event_tracer(),
            core::ptr::null_mut::<MemoryAllocator>() as *mut dyn MemoryAllocatorBase,
        )
    }

    fn op_floor_divide_out<'a, 'b>(
        self_: &Tensor,
        other: &Tensor,
        out: &'a Tensor<'b>,
    ) -> &'a Tensor<'b> {
        let mut ctx = context();
        floor_divide_out(&mut ctx, self_, other, out)
    }

    // PORT-NOTE: `static_cast<CTYPE>(int)`/`static_cast<CTYPE>(double)` bridges for
    // building literal data in the factory element types.
    trait FromI64: Copy {
        fn from_i64(v: i64) -> Self;
    }
    macro_rules! impl_from_i64_num {
        ($($t:ty),*) => {$(impl FromI64 for $t { fn from_i64(v: i64) -> Self { v as $t } })*};
    }
    impl_from_i64_num!(u8, i8, i16, i32, i64);

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

    fn test_integer_floor_divide<T>()
    where
        T: CppTypeToScalarType + FactoryValue + FromI64,
    {
        let tf = TensorFactory::<T>::new();
        let d = |v: &[i64]| -> Vec<T> { v.iter().map(|&x| T::from_i64(x)).collect() };

        let sizes = vec![3, 2];
        let out = tf.zeros_default(sizes.clone());

        // Integer division of -8 / 6 return -1, but -8 // 6 is -2
        op_floor_divide_out(
            &tf.make_default(sizes.clone(), d(&[-8, 1, 2, 4, 8, 3])),
            &tf.make_default(sizes.clone(), d(&[6, 2, 2, 2, 2, -5])),
            &out,
        );

        assert!(tensors_are_close(
            &out,
            &tf.make_default(sizes, d(&[-2, 0, 1, 2, 4, -1])),
            0.0,
            Some(0.0)
        ));
    }

    fn test_floating_point_floor_divide<T>()
    where
        T: CppTypeToScalarType + FactoryValue + FromF64,
    {
        let tf = TensorFactory::<T>::new();
        let d = |v: &[f64]| -> Vec<T> { v.iter().map(|&x| T::from_f64(x)).collect() };

        let sizes = vec![3, 2];
        let out = tf.zeros_default(sizes.clone());

        op_floor_divide_out(
            &tf.make_default(sizes.clone(), d(&[-5.3, 1.1, 2.2, 4.4, 6.8, -0.9])),
            &tf.make_default(sizes.clone(), d(&[2.7, 2.0, 2.0, 2.0, 2.0, -0.2])),
            &out,
        );

        assert!(tensors_are_close(
            &out,
            &tf.make_default(sizes, d(&[-2.0, 0.0, 1.0, 2.0, 3.0, 4.0])),
            internal::K_DEFAULT_RTOL,
            None
        ));
    }

    // [spec:et:sem:op-floor-divide.torch.executor.native.floor-divide-out-fn/test]
    #[test]
    fn op_floor_divide_test_byte_tensors() {
        let tf = TensorFactory::<u8>::new();
        let sizes = vec![2, 2];
        let out = tf.zeros_default(sizes.clone());

        op_floor_divide_out(
            &tf.make_default(sizes.clone(), vec![1, 2, 4, 8]),
            &tf.make_default(sizes.clone(), vec![2, 2, 2, 2]),
            &out,
        );

        assert!(tensors_are_close(
            &out,
            &tf.make_default(sizes, vec![0, 1, 2, 4]),
            0.0,
            Some(0.0)
        ));
    }

    // [spec:et:sem:op-floor-divide.torch.executor.native.floor-divide-out-fn/test]
    #[test]
    fn op_floor_divide_test_char_tensors() {
        test_integer_floor_divide::<i8>();
    }

    // [spec:et:sem:op-floor-divide.torch.executor.native.floor-divide-out-fn/test]
    #[test]
    fn op_floor_divide_test_short_tensors() {
        test_integer_floor_divide::<i16>();
    }

    // [spec:et:sem:op-floor-divide.torch.executor.native.floor-divide-out-fn/test]
    // also verifies floor_divide integral overload (signed adjustment: -8 // 6 == -2)
    // [spec:et:sem:math-util.torch.executor.native.utils.floor-divide-fn/test]
    #[test]
    fn op_floor_divide_test_int_tensors() {
        test_integer_floor_divide::<i32>();
    }

    // [spec:et:sem:op-floor-divide.torch.executor.native.floor-divide-out-fn/test]
    #[test]
    fn op_floor_divide_test_long_tensors() {
        test_integer_floor_divide::<i64>();
    }

    // [spec:et:sem:op-floor-divide.torch.executor.native.floor-divide-out-fn/test]
    // also verifies floor_divide floating-point overload (sign-aware floor: -5.3 // 2.7 == -2.0)
    // [spec:et:sem:math-util.torch.executor.native.utils.floor-divide-fn/test]
    #[test]
    fn op_floor_divide_test_float_tensors() {
        test_floating_point_floor_divide::<f32>();
    }

    // [spec:et:sem:op-floor-divide.torch.executor.native.floor-divide-out-fn/test]
    #[test]
    fn op_floor_divide_test_double_tensors() {
        test_floating_point_floor_divide::<f64>();
    }

    // [spec:et:sem:op-floor-divide.torch.executor.native.floor-divide-out-fn/test]
    #[test]
    fn op_floor_divide_test_half_tensors() {
        test_floating_point_floor_divide::<Half>();
    }

    // [spec:et:sem:op-floor-divide.torch.executor.native.floor-divide-out-fn/test]
    #[test]
    fn op_floor_divide_test_bfloat16_tensors() {
        test_floating_point_floor_divide::<BFloat16>();
    }

    // [spec:et:sem:op-floor-divide.torch.executor.native.floor-divide-out-fn/test]
    #[test]
    fn op_floor_divide_test_unhandled_dtype_dies() {
        // floor_divide() doesn't handle Bool.
        let tf = TensorFactory::<bool>::new();
        let sizes = vec![2, 2];

        let a = tf.make_default(sizes.clone(), vec![false, true, false, true]);
        let b = tf.make_default(sizes.clone(), vec![true, true, true, true]);
        let out = tf.zeros_default(sizes);

        let mut ctx = context();
        floor_divide_out(&mut ctx, &a, &b, &out);
        assert_ne!(ctx.failure_state(), Error::Ok);
    }

    // [spec:et:sem:op-floor-divide.torch.executor.native.floor-divide-out-fn/test]
    #[test]
    fn op_floor_divide_test_mismatched_input_shapes_dies() {
        let tf = TensorFactory::<i32>::new();

        let a = tf.ones_default(vec![4]);
        let b = tf.ones_default(vec![2, 2]);
        let out = tf.zeros_default(vec![4]);

        let mut ctx = context();
        floor_divide_out(&mut ctx, &a, &b, &out);
        assert_ne!(ctx.failure_state(), Error::Ok);
    }

    // PORT-NOTE: `ET_SKIP_IF(is_aten, ...)` — non-aten branch, so the body runs.
    // [spec:et:sem:op-floor-divide.torch.executor.native.floor-divide-out-fn/test]
    #[test]
    fn op_floor_divide_test_mismatched_output_shapes_dies() {
        let tf = TensorFactory::<i32>::new();
        let sizes = vec![2, 2];

        let a = tf.ones_default(sizes.clone());
        let b = tf.ones_default(sizes);
        let out = tf.zeros_default(vec![4]);

        let mut ctx = context();
        floor_divide_out(&mut ctx, &a, &b, &out);
        assert_ne!(ctx.failure_state(), Error::Ok);
    }

    // PORT-NOTE: `DISABLED_` gtest — dynamic (broadcast) shape not supported.
    // [spec:et:sem:op-floor-divide.torch.executor.native.floor-divide-out-fn/test]
    #[test]
    #[ignore]
    fn op_floor_divide_test_broadcast_dim_size_is_one_ab() {
        let tf = TensorFactory::<f32>::new();

        let x = tf.make_default(
            vec![3, 2],
            vec![
                0.6651028990745544,
                0.47241002321243286,
                0.15020078420639038,
                0.5280023813247681,
                0.9517974257469177,
                0.5294632911682129,
            ],
        );
        let y = tf.make_default(vec![1, 2], vec![0.522396445274353, 0.6753279566764832]);
        let expected_result = tf.make_default(vec![3, 2], vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);

        let out = tf.zeros_default(vec![3, 2]);
        op_floor_divide_out(&x, &y, &out);
        assert!(tensors_are_close(
            &out,
            &expected_result,
            internal::K_DEFAULT_RTOL,
            None
        ));
    }

    // PORT-NOTE: `DISABLED_` gtest — dynamic (broadcast) shape not supported.
    // [spec:et:sem:op-floor-divide.torch.executor.native.floor-divide-out-fn/test]
    #[test]
    #[ignore]
    fn op_floor_divide_test_broadcast_dim_size_missing_ab() {
        let tf = TensorFactory::<f32>::new();

        let x = tf.make_default(
            vec![3, 2],
            vec![
                0.6651028990745544,
                0.47241002321243286,
                0.15020078420639038,
                0.5280023813247681,
                0.9517974257469177,
                0.5294632911682129,
            ],
        );
        let y = tf.make_default(vec![2], vec![0.522396445274353, 0.6753279566764832]);
        let expected_result = tf.make_default(vec![3, 2], vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);

        let out = tf.zeros_default(vec![3, 2]);
        op_floor_divide_out(&x, &y, &out);
        assert!(tensors_are_close(
            &out,
            &expected_result,
            internal::K_DEFAULT_RTOL,
            None
        ));
    }

    // PORT-NOTE: `DISABLED_` gtest — dynamic (broadcast) shape not supported.
    // [spec:et:sem:op-floor-divide.torch.executor.native.floor-divide-out-fn/test]
    #[test]
    #[ignore]
    fn op_floor_divide_test_broadcast_dim_size_is_one_ba() {
        let tf = TensorFactory::<f32>::new();

        let x = tf.make_default(vec![1, 2], vec![0.522396445274353, 0.6753279566764832]);
        let y = tf.make_default(
            vec![3, 2],
            vec![
                0.6651028990745544,
                0.47241002321243286,
                0.15020078420639038,
                0.5280023813247681,
                0.9517974257469177,
                0.5294632911682129,
            ],
        );
        let expected_result = tf.make_default(vec![3, 2], vec![0.0, 1.0, 3.0, 1.0, 0.0, 1.0]);

        let out = tf.zeros_default(vec![3, 2]);
        op_floor_divide_out(&x, &y, &out);
        assert!(tensors_are_close(
            &out,
            &expected_result,
            internal::K_DEFAULT_RTOL,
            None
        ));
    }

    // PORT-NOTE: `DISABLED_` gtest — dynamic (broadcast) shape not supported.
    // [spec:et:sem:op-floor-divide.torch.executor.native.floor-divide-out-fn/test]
    #[test]
    #[ignore]
    fn op_floor_divide_test_broadcast_dim_size_missing_ba() {
        let tf = TensorFactory::<f32>::new();

        let x = tf.make_default(vec![1, 2], vec![0.522396445274353, 0.6753279566764832]);
        let y = tf.make_default(
            vec![3, 2],
            vec![
                0.6651028990745544,
                0.47241002321243286,
                0.15020078420639038,
                0.5280023813247681,
                0.9517974257469177,
                0.5294632911682129,
            ],
        );
        let expected_result = tf.make_default(vec![3, 2], vec![0.0, 1.0, 3.0, 1.0, 0.0, 1.0]);

        let out = tf.zeros_default(vec![3, 2]);
        op_floor_divide_out(&x, &y, &out);
        assert!(tensors_are_close(
            &out,
            &expected_result,
            internal::K_DEFAULT_RTOL,
            None
        ));
    }

    // PORT-NOTE: `DISABLED_` gtest — dynamic shape not supported.
    // [spec:et:sem:op-floor-divide.torch.executor.native.floor-divide-out-fn/test]
    #[test]
    #[ignore]
    fn op_floor_divide_test_dynamic_shape_upper_bound_same_as_expected() {
        use crate::runtime::core::tensor_shape_dynamism::TensorShapeDynamism;
        let tf = TensorFactory::<f32>::new();

        let x = tf.make_default(
            vec![3, 2],
            vec![
                0.34620773792266846,
                0.7118645310401917,
                0.028005361557006836,
                0.8868894577026367,
                0.38272881507873535,
                0.19501900672912598,
            ],
        );
        let y = tf.make_default(
            vec![3, 2],
            vec![
                0.3282443881034851,
                0.7458182573318481,
                0.1568273901939392,
                0.6325231194496155,
                0.2777167558670044,
                0.09974533319473267,
            ],
        );
        let expected_result = tf.make_default(vec![3, 2], vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0]);

        let out = tf.zeros(vec![3, 4], TensorShapeDynamism::DYNAMIC_BOUND);
        op_floor_divide_out(&x, &y, &out);
        assert!(tensors_are_close(
            &out,
            &expected_result,
            internal::K_DEFAULT_RTOL,
            None
        ));
    }

    // PORT-NOTE: `DISABLED_` gtest — dynamic shape not supported.
    // [spec:et:sem:op-floor-divide.torch.executor.native.floor-divide-out-fn/test]
    #[test]
    #[ignore]
    fn op_floor_divide_test_dynamic_shape_upper_bound_larger_than_expected() {
        use crate::runtime::core::tensor_shape_dynamism::TensorShapeDynamism;
        let tf = TensorFactory::<f32>::new();

        let x = tf.make_default(
            vec![3, 2],
            vec![
                0.34620773792266846,
                0.7118645310401917,
                0.028005361557006836,
                0.8868894577026367,
                0.38272881507873535,
                0.19501900672912598,
            ],
        );
        let y = tf.make_default(
            vec![3, 2],
            vec![
                0.3282443881034851,
                0.7458182573318481,
                0.1568273901939392,
                0.6325231194496155,
                0.2777167558670044,
                0.09974533319473267,
            ],
        );
        let expected_result = tf.make_default(vec![3, 2], vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0]);

        let out = tf.zeros(vec![10, 10], TensorShapeDynamism::DYNAMIC_BOUND);
        op_floor_divide_out(&x, &y, &out);
        assert!(tensors_are_close(
            &out,
            &expected_result,
            internal::K_DEFAULT_RTOL,
            None
        ));
    }

    // PORT-NOTE: `DISABLED_` gtest — dynamic shape not supported.
    // [spec:et:sem:op-floor-divide.torch.executor.native.floor-divide-out-fn/test]
    #[test]
    #[ignore]
    fn op_floor_divide_test_dynamic_shape_unbound() {
        use crate::runtime::core::tensor_shape_dynamism::TensorShapeDynamism;
        let tf = TensorFactory::<f32>::new();

        let x = tf.make_default(
            vec![3, 2],
            vec![
                0.34620773792266846,
                0.7118645310401917,
                0.028005361557006836,
                0.8868894577026367,
                0.38272881507873535,
                0.19501900672912598,
            ],
        );
        let y = tf.make_default(
            vec![3, 2],
            vec![
                0.3282443881034851,
                0.7458182573318481,
                0.1568273901939392,
                0.6325231194496155,
                0.2777167558670044,
                0.09974533319473267,
            ],
        );
        let expected_result = tf.make_default(vec![3, 2], vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0]);

        let out = tf.zeros(vec![1, 1], TensorShapeDynamism::DYNAMIC_UNBOUND);
        op_floor_divide_out(&x, &y, &out);
        assert!(tensors_are_close(
            &out,
            &expected_result,
            internal::K_DEFAULT_RTOL,
            None
        ));
    }

    // std::floor(0.5 / 0.1) == 5.0, but 0.5 // 0.1 yeilds 4.0
    // [spec:et:sem:op-floor-divide.torch.executor.native.floor-divide-out-fn/test]
    #[test]
    fn op_floor_divide_test_float_floor_divide_edge_case() {
        let tf = TensorFactory::<f32>::new();

        let x = tf.make_default(vec![1, 2], vec![0.5, -0.5]);
        let y = tf.make_default(vec![1, 2], vec![0.1, -0.1]);
        let expected_result = tf.make_default(vec![1, 2], vec![4.0, 4.0]);

        let out = tf.zeros_default(vec![1, 2]);
        let ret = op_floor_divide_out(&x, &y, &out);
        assert!(tensors_are_close(ret, &out, 0.0, Some(0.0)));
        assert!(tensors_are_close(
            &out,
            &expected_result,
            internal::K_DEFAULT_RTOL,
            None
        ));
    }
}
