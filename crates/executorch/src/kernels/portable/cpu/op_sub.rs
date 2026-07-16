//! Literal port of kernels/portable/cpu/op_sub.cpp.

use crate::kernels::portable::cpu::scalar_utils::{
    get_scalar_dtype, promote_type_with_scalar, scalar_to,
};
use crate::kernels::portable::cpu::util::broadcast_util::resize_to_broadcast_target_size;
use crate::kernels::portable::cpu::util::dtype_util::SupportedTensorDtypes;
use crate::kernels::portable::cpu::util::elementwise_util::{
    apply_bitensor_elementwise_fn, apply_unitensor_elementwise_fn, get_compute_type,
};
use crate::runtime::core::error::Error;
use crate::runtime::core::exec_aten::util::scalar_type_util::{can_cast, promote_types};
use crate::runtime::core::exec_aten::util::tensor_util::{
    resize_tensor, tensors_have_same_dim_order2, tensors_have_same_dim_order3,
};
use crate::runtime::core::portable_type::scalar::Scalar;
use crate::runtime::core::portable_type::scalar_type::ScalarType;
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

// PORT-NOTE: the C++ compute closures `val_a - (decltype(val_b))(val_alpha)*val_b`
// and `val_a - (decltype(val_a))(val_alpha_times_b)` operate over ET_SWITCH_REAL
// types (Byte, Char, Short, Int, Long, Float, Double) — all Rust primitives with
// native `-`/`*`, so the casts are identity and the arithmetic uses the plain
// operators. No Bool/Half/BFloat16 arm (REAL, not REALB), so no promotion trait
// is needed.

// [spec:et:def:op-sub.torch.executor.native.sub-out-fn]
// [spec:et:sem:op-sub.torch.executor.native.sub-out-fn]
pub fn sub_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    a: &Tensor,
    b: &Tensor,
    alpha: &Scalar,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    let alpha_type: ScalarType = get_scalar_dtype(*alpha);

    // Check alpha type
    crate::et_kernel_check!(ctx, alpha_type != ScalarType::Bool, InvalidArgument, out);

    // Common Dtype
    let common_type: ScalarType = promote_types(a.scalar_type(), b.scalar_type(), false);

    // Check Common Dtype
    crate::et_kernel_check!(
        ctx,
        can_cast(common_type, out.scalar_type()) && can_cast(alpha_type, common_type),
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
    let compute_type: ScalarType = get_compute_type(&mut common_type_mut);

    let op_name = "sub.out";

    crate::et_switch_real_types!(compute_type, ctx, op_name, CTYPE_COMPUTE, {
        let val_alpha: CTYPE_COMPUTE = scalar_to::<CTYPE_COMPUTE>(alpha);
        apply_bitensor_elementwise_fn::<CTYPE_COMPUTE, _>(
            move |vals: &[CTYPE_COMPUTE]| -> CTYPE_COMPUTE { vals[0] - val_alpha * vals[1] },
            ctx,
            a,
            SupportedTensorDtypes::REALHBF16,
            b,
            SupportedTensorDtypes::REALHBF16,
            out,
            SupportedTensorDtypes::REALHBF16,
            false,
        );
    });

    out
}

// [spec:et:def:op-sub.torch.executor.native.sub-scalar-out-fn]
// [spec:et:sem:op-sub.torch.executor.native.sub-scalar-out-fn]
pub fn sub_scalar_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    a: &Tensor,
    b: &Scalar,
    alpha: &Scalar,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    let alpha_type: ScalarType = get_scalar_dtype(*alpha);

    // Check alpha type
    crate::et_kernel_check!(ctx, alpha_type != ScalarType::Bool, InvalidArgument, out);

    // Common Dtype
    let common_type: ScalarType = promote_type_with_scalar(a.scalar_type(), *b, false);

    // Check Common Dtype
    crate::et_kernel_check!(
        ctx,
        common_type == out.scalar_type() && can_cast(alpha_type, common_type),
        InvalidArgument,
        out
    );

    // Check Dim Order
    crate::et_kernel_check!(
        ctx,
        tensors_have_same_dim_order2(a, out),
        InvalidArgument,
        out
    );

    // Resize
    crate::et_kernel_check!(
        ctx,
        resize_tensor(out, a.sizes()) == Error::Ok,
        InvalidArgument,
        out
    );

    // Compute Dtype
    let mut common_type_mut = common_type;
    let compute_type: ScalarType = get_compute_type(&mut common_type_mut);

    let op_name = "sub.Scalar_out";

    crate::et_switch_real_types!(compute_type, ctx, op_name, CTYPE_COMPUTE, {
        let val_b: CTYPE_COMPUTE = scalar_to::<CTYPE_COMPUTE>(b);
        let val_alpha: CTYPE_COMPUTE = scalar_to::<CTYPE_COMPUTE>(alpha);
        let val_alpha_times_b = val_alpha * val_b;
        apply_unitensor_elementwise_fn::<CTYPE_COMPUTE, _>(
            move |vals: &[CTYPE_COMPUTE]| -> CTYPE_COMPUTE { vals[0] - val_alpha_times_b },
            ctx,
            a,
            SupportedTensorDtypes::REALHBF16,
            out,
            SupportedTensorDtypes::SAME_AS_COMMON,
            false,
        );
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
    use crate::runtime::core::exec_aten::util::scalar_type_util::{
        CppTypeToScalarType, is_integral_type,
    };
    use crate::runtime::core::memory_allocator::{MemoryAllocator, MemoryAllocatorBase};
    use crate::runtime::core::portable_type::{BFloat16, Half};
    use crate::runtime::core::tensor_shape_dynamism::TensorShapeDynamism;
    use crate::{assert_tensor_close, assert_tensor_eq};

    fn context() -> KernelRuntimeContext<'static> {
        crate::runtime::platform::runtime::runtime_init();
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

    fn test_sub<A, B, OUT>()
    where
        A: CppTypeToScalarType + FactoryValue + FromF64Elem,
        B: CppTypeToScalarType + FactoryValue,
        OUT: CppTypeToScalarType + FactoryValue + FromF64Elem,
    {
        let tf_a = TensorFactory::<A>::new();
        let tf_b = TensorFactory::<B>::new();
        let tf_out = TensorFactory::<OUT>::new();

        let sizes = vec![2, 2];

        let out = tf_out.zeros_default(sizes.clone());

        let mut ctx = context();
        sub_out(
            &mut ctx,
            &tf_a.make_default(
                sizes.clone(),
                vec![
                    A::from_f64(1.0),
                    A::from_f64(2.0),
                    A::from_f64(4.0),
                    A::from_f64(8.0),
                ],
            ),
            &tf_b.ones_default(sizes.clone()),
            &Scalar::from_i64(1),
            &out,
        );

        assert_tensor_eq!(
            out,
            tf_out.make_default(
                sizes,
                vec![
                    OUT::from_f64(0.0),
                    OUT::from_f64(1.0),
                    OUT::from_f64(3.0),
                    OUT::from_f64(7.0)
                ],
            )
        );
    }

    fn test_sub_enumerate_out_types<A, B>()
    where
        A: CppTypeToScalarType + FactoryValue + FromF64Elem,
        B: CppTypeToScalarType + FactoryValue + FromF64Elem,
    {
        test_sub::<A, B, Half>();
        test_sub::<A, B, f32>();
        test_sub::<A, B, f64>();
        // Integral out type is only allowed if both inputs are integral types.
        if is_integral_type(A::VALUE, false) && is_integral_type(B::VALUE, false) {
            test_sub::<A, B, i32>();
            test_sub::<A, B, i64>();
        }
    }

    fn test_sub_enumerate_b_types<A>()
    where
        A: CppTypeToScalarType + FactoryValue + FromF64Elem,
    {
        // ET_FORALL_REALHBF16_TYPES
        test_sub_enumerate_out_types::<A, u8>();
        test_sub_enumerate_out_types::<A, i8>();
        test_sub_enumerate_out_types::<A, i16>();
        test_sub_enumerate_out_types::<A, i32>();
        test_sub_enumerate_out_types::<A, i64>();
        test_sub_enumerate_out_types::<A, f32>();
        test_sub_enumerate_out_types::<A, f64>();
        test_sub_enumerate_out_types::<A, Half>();
        test_sub_enumerate_out_types::<A, BFloat16>();
    }

    fn test_sub_enumerate_a_types() {
        // ET_FORALL_REALHBF16_TYPES
        test_sub_enumerate_b_types::<u8>();
        test_sub_enumerate_b_types::<i8>();
        test_sub_enumerate_b_types::<i16>();
        test_sub_enumerate_b_types::<i32>();
        test_sub_enumerate_b_types::<i64>();
        test_sub_enumerate_b_types::<f32>();
        test_sub_enumerate_b_types::<f64>();
        test_sub_enumerate_b_types::<Half>();
        test_sub_enumerate_b_types::<BFloat16>();
    }

    fn test_floating_point_sub_out<T>()
    where
        T: CppTypeToScalarType + FactoryValue + FromF64Elem,
    {
        let tf = TensorFactory::<T>::new();

        let sizes = vec![2, 2];

        let out = tf.zeros_default(sizes.clone());

        let mut ctx = context();
        sub_out(
            &mut ctx,
            &tf.make_default(
                sizes.clone(),
                vec![
                    T::from_f64(1.25),
                    T::from_f64(2.25),
                    T::from_f64(4.5),
                    T::from_f64(8.875),
                ],
            ),
            &tf.ones_default(sizes.clone()),
            &Scalar::from_i64(1),
            &out,
        );

        assert_tensor_close!(
            out,
            tf.make_default(
                sizes,
                vec![
                    T::from_f64(0.25),
                    T::from_f64(1.25),
                    T::from_f64(3.5),
                    T::from_f64(7.875),
                ],
            )
        );
    }

    fn test_broadcast_3d<T>()
    where
        T: CppTypeToScalarType + FactoryValue + FromF64Elem,
    {
        let tf_a = TensorFactory::<T>::new();

        let a = tf_a.make_default(
            vec![2, 2, 3],
            (1..=12).map(|v| T::from_f64(v as f64)).collect(),
        );
        let b = tf_a.make_default(
            vec![2, 1, 3],
            vec![
                T::from_f64(2.),
                T::from_f64(3.),
                T::from_f64(4.),
                T::from_f64(5.),
                T::from_f64(6.),
                T::from_f64(7.),
            ],
        );

        let out = tf_a.make_default(
            vec![2, 2, 3],
            (1..=12).map(|v| T::from_f64(v as f64)).collect(),
        );
        let expected = tf_a.make_default(
            vec![2, 2, 3],
            vec![-1., -1., -1., 2., 2., 2., 2., 2., 2., 5., 5., 5.]
                .into_iter()
                .map(T::from_f64)
                .collect(),
        );

        let mut ctx = context();
        assert_tensor_close!(
            *sub_out(&mut ctx, &a, &b, &Scalar::from_double(1.0), &out),
            expected
        );

        let expected = tf_a.make_default(
            vec![2, 2, 3],
            vec![
                0.5, 0.0, -0.5, -4.0, -4.5, -5.0, -5.5, -6.0, -6.5, -10.0, -10.5, -11.0,
            ]
            .into_iter()
            .map(T::from_f64)
            .collect(),
        );
        assert_tensor_close!(
            *sub_out(&mut ctx, &b, &a, &Scalar::from_double(1.5), &out),
            expected
        );
    }

    fn test_broadcast_4d<T>()
    where
        T: CppTypeToScalarType + FactoryValue + FromF64Elem,
    {
        let tf_a = TensorFactory::<T>::new();

        let a = tf_a.make_default(
            vec![2, 2, 3, 5],
            (1..=60).map(|v| T::from_f64(v as f64)).collect(),
        );
        let b = tf_a.make_default(
            vec![2, 1, 3, 5],
            (1..=30).map(|v| T::from_f64(v as f64)).collect(),
        );

        let out = tf_a.zeros_default(vec![2, 2, 3, 5]);
        let expected = tf_a.make_default(vec![2, 2, 3, 5], {
            let mut v = Vec::new();
            for _ in 0..15 {
                v.push(0.0);
            }
            for _ in 0..15 {
                v.push(15.0);
            }
            for _ in 0..15 {
                v.push(15.0);
            }
            for _ in 0..15 {
                v.push(30.0);
            }
            v.into_iter().map(T::from_f64).collect()
        });

        let mut ctx = context();
        assert_tensor_close!(
            *sub_out(&mut ctx, &a, &b, &Scalar::from_double(1.0), &out),
            expected
        );

        let expected = tf_a.make_default(vec![2, 2, 3, 5], {
            let mut v = Vec::new();
            for _ in 0..15 {
                v.push(0.0);
            }
            for _ in 0..15 {
                v.push(-15.0);
            }
            for _ in 0..15 {
                v.push(-15.0);
            }
            for _ in 0..15 {
                v.push(-30.0);
            }
            v.into_iter().map(T::from_f64).collect()
        });
        assert_tensor_close!(
            *sub_out(&mut ctx, &b, &a, &Scalar::from_double(1.0), &out),
            expected
        );

        let b = tf_a.make_default(
            vec![2, 2, 1, 5],
            (1..=20).map(|v| T::from_f64(v as f64)).collect(),
        );
        let out = tf_a.zeros_default(vec![2, 2, 3, 5]);
        let expected = tf_a.make_default(
            vec![2, 2, 3, 5],
            vec![
                0., 0., 0., 0., 0., 5., 5., 5., 5., 5., 10., 10., 10., 10., 10., 10., 10., 10.,
                10., 10., 15., 15., 15., 15., 15., 20., 20., 20., 20., 20., 20., 20., 20., 20.,
                20., 25., 25., 25., 25., 25., 30., 30., 30., 30., 30., 30., 30., 30., 30., 30.,
                35., 35., 35., 35., 35., 40., 40., 40., 40., 40.,
            ]
            .into_iter()
            .map(T::from_f64)
            .collect(),
        );

        assert_tensor_close!(
            *sub_out(&mut ctx, &a, &b, &Scalar::from_double(1.0), &out),
            expected
        );

        let expected = tf_a.make_default(
            vec![2, 2, 3, 5],
            vec![
                -0.5, -1.0, -1.5, -2.0, -2.5, -8.0, -8.5, -9.0, -9.5, -10.0, -15.5, -16.0, -16.5,
                -17.0, -17.5, -18.0, -18.5, -19.0, -19.5, -20.0, -25.5, -26.0, -26.5, -27.0, -27.5,
                -33.0, -33.5, -34.0, -34.5, -35.0, -35.5, -36.0, -36.5, -37.0, -37.5, -43.0, -43.5,
                -44.0, -44.5, -45.0, -50.5, -51.0, -51.5, -52.0, -52.5, -53.0, -53.5, -54.0, -54.5,
                -55.0, -60.5, -61.0, -61.5, -62.0, -62.5, -68.0, -68.5, -69.0, -69.5, -70.0,
            ]
            .into_iter()
            .map(T::from_f64)
            .collect(),
        );
        assert_tensor_close!(
            *sub_out(&mut ctx, &b, &a, &Scalar::from_double(1.5), &out),
            expected
        );
    }

    fn test_broadcast_rank1_scalar<T>()
    where
        T: CppTypeToScalarType + FactoryValue + FromF64Elem,
    {
        let tf = TensorFactory::<T>::new();

        let a = tf.make_default(
            vec![2, 1, 3],
            vec![2, 3, 4, 5, 6, 7]
                .into_iter()
                .map(|v| T::from_f64(v as f64))
                .collect(),
        );
        let b = tf.make_default(vec![1], vec![T::from_f64(2.0)]);

        let out = tf.zeros_default(vec![2, 1, 3]);

        let mut ctx = context();
        sub_out(&mut ctx, &a, &b, &Scalar::from_i64(1), &out);

        let ret = tf.make_default(
            vec![2, 1, 3],
            vec![0, 1, 2, 3, 4, 5]
                .into_iter()
                .map(|v| T::from_f64(v as f64))
                .collect(),
        );
        assert_tensor_eq!(out, ret);

        sub_out(&mut ctx, &b, &a, &Scalar::from_i64(1), &out);
        let ret = tf.make_default(
            vec![2, 1, 3],
            vec![0, -1, -2, -3, -4, -5]
                .into_iter()
                .map(|v| T::from_f64(v as f64))
                .collect(),
        );
        assert_tensor_eq!(out, ret);
    }

    // ---- OpSubOutTest ----

    // [spec:et:sem:op-sub.torch.executor.native.sub-out-fn/test]
    #[test]
    fn op_sub_out_test_all_real_dtypes_supported() {
        test_sub_enumerate_a_types();
    }

    // [spec:et:sem:op-sub.torch.executor.native.sub-out-fn/test]
    #[test]
    fn op_sub_out_test_float_tensors() {
        test_floating_point_sub_out::<f32>();
    }

    // [spec:et:sem:op-sub.torch.executor.native.sub-out-fn/test]
    #[test]
    fn op_sub_out_test_double_tensors() {
        test_floating_point_sub_out::<f64>();
    }

    // [spec:et:sem:op-sub.torch.executor.native.sub-out-fn/test]
    #[test]
    fn op_sub_out_test_half_tensors() {
        test_floating_point_sub_out::<Half>();
    }

    // [spec:et:sem:op-sub.torch.executor.native.sub-out-fn/test]
    #[test]
    fn op_sub_out_test_bfloat16_tensors() {
        test_floating_point_sub_out::<BFloat16>();
    }

    // [spec:et:sem:op-sub.torch.executor.native.sub-out-fn/test]
    #[test]
    fn op_sub_out_test_broadcast_supported() {
        let tf = TensorFactory::<f32>::new();

        let a = tf.make_default(vec![2, 1, 2, 1], vec![7., 8., 9., 10.]);
        let b = tf.make_default(vec![2, 1, 4], vec![1., 1., 1., 1., 2., 2., 2., 2.]);
        let refr = tf.make_default(
            vec![2, 2, 2, 4],
            vec![
                6., 6., 6., 6., 7., 7., 7., 7., 5., 5., 5., 5., 6., 6., 6., 6., 8., 8., 8., 8., 9.,
                9., 9., 9., 7., 7., 7., 7., 8., 8., 8., 8.,
            ],
        );

        let out = tf.zeros_default(vec![2, 2, 2, 4]);

        let mut ctx = context();
        sub_out(&mut ctx, &a, &b, &Scalar::from_i64(1), &out);

        assert_tensor_eq!(out, refr);
    }

    // [spec:et:sem:op-sub.torch.executor.native.sub-out-fn/test]
    #[test]
    fn op_sub_out_test_broadcast_supported2() {
        let tf = TensorFactory::<f32>::new();

        let a = tf.make_default(vec![3, 2, 1], vec![2., 3., 4., 5., 6., 7.]);
        let b = tf.make_default(vec![1, 2, 1], vec![2., 3.]);

        let out = tf.zeros_default(vec![3, 2, 1]);

        let mut ctx = context();
        sub_out(&mut ctx, &a, &b, &Scalar::from_i64(1), &out);

        let ret = tf.make_default(vec![3, 2, 1], vec![0., 0., 2., 2., 4., 4.]);
        assert_tensor_eq!(out, ret);
    }

    // [spec:et:sem:op-sub.torch.executor.native.sub-out-fn/test]
    #[test]
    fn op_sub_out_test_broadcast_scalar_supported1() {
        test_broadcast_rank1_scalar::<f32>();
        test_broadcast_rank1_scalar::<Half>();
    }

    // [spec:et:sem:op-sub.torch.executor.native.sub-out-fn/test]
    #[test]
    fn op_sub_out_test_broadcast_scalar_supported2() {
        let tf = TensorFactory::<f32>::new();

        let a = tf.make_default(vec![1, 1, 1], vec![8.]);
        let b = tf.make_default(vec![3, 1, 1], vec![2., 4., 8.]);

        let out = tf.zeros_default(vec![3, 1, 1]);

        let mut ctx = context();
        sub_out(&mut ctx, &a, &b, &Scalar::from_i64(1), &out);

        let ret = tf.make_default(vec![3, 1, 1], vec![6., 4., 0.]);
        assert_tensor_eq!(out, ret);

        // std::swap(a, b)
        let out = tf.zeros_default(vec![3, 1, 1]);
        sub_out(&mut ctx, &b, &a, &Scalar::from_i64(1), &out);
        let ret = tf.make_default(vec![3, 1, 1], vec![-6., -4., 0.]);
        assert_tensor_eq!(out, ret);
    }

    // [spec:et:sem:op-sub.torch.executor.native.sub-out-fn/test]
    #[test]
    fn op_sub_out_test_broadcast_scalar_rank0_supported() {
        let tf = TensorFactory::<f32>::new();

        let a = tf.make_default(vec![1], vec![5.]);
        let b = tf.make_default(vec![], vec![2.]);

        let out = tf.zeros_default(vec![1]);

        let mut ctx = context();
        sub_out(&mut ctx, &a, &b, &Scalar::from_i64(1), &out);

        let ret = tf.make_default(vec![1], vec![3.]);
        assert_tensor_eq!(out, ret);

        sub_out(&mut ctx, &b, &a, &Scalar::from_i64(1), &out);

        let ret = tf.make_default(vec![1], vec![-3.]);
        assert_tensor_eq!(out, ret);
    }

    // [spec:et:sem:op-sub.torch.executor.native.sub-out-fn/test]
    #[test]
    fn op_sub_out_test_broadcast_nd_test() {
        // Test 3D tensors
        test_broadcast_3d::<f32>();
        test_broadcast_3d::<Half>();
        // Sub doesnt yet support BFloat16

        // Test 4D tensors
        test_broadcast_4d::<f32>();
        test_broadcast_4d::<Half>();
    }

    // ---- Death Tests ----

    // [spec:et:sem:op-sub.torch.executor.native.sub-out-fn/test]
    #[test]
    fn op_sub_out_test_int_tensor_float_alpha_dies() {
        let tf = TensorFactory::<i32>::new();

        let sizes = vec![2, 2];

        let out = tf.zeros_default(sizes.clone());

        let mut ctx = context();
        et_expect_kernel_failure!(
            ctx,
            sub_out(
                &mut ctx,
                &tf.ones_default(sizes.clone()),
                &tf.ones_default(sizes),
                &Scalar::from_double(0.7),
                &out
            )
        );
    }

    // [spec:et:sem:op-sub.torch.executor.native.sub-out-fn/test]
    #[test]
    fn op_sub_out_test_bool_input_tensors_fail() {
        let tf = TensorFactory::<bool>::new();

        let sizes = vec![2, 2];

        let a = tf.make_default(sizes.clone(), vec![false, true, false, true]);
        let b = tf.make_default(sizes.clone(), vec![false, true, true, true]);

        let out = tf.zeros_default(sizes);

        let mut ctx = context();
        et_expect_kernel_failure!(ctx, sub_out(&mut ctx, &a, &b, &Scalar::from_i64(1), &out));
    }

    // [spec:et:sem:op-sub.torch.executor.native.sub-out-fn/test]
    #[test]
    fn op_sub_out_test_int_output_with_float_input_dies() {
        let tfi = TensorFactory::<i32>::new();
        let tff = TensorFactory::<f32>::new();

        let sizes = vec![2, 2];

        let a = tfi.make_default(sizes.clone(), vec![2, 4, 3, 3]);
        let b = tff.make_default(sizes.clone(), vec![2., 4., 3., 3.]);

        let out = tfi.zeros_default(sizes);

        let mut ctx = context();
        et_expect_kernel_failure!(ctx, sub_out(&mut ctx, &a, &b, &Scalar::from_i64(1), &out));
    }

    // [spec:et:sem:op-sub.torch.executor.native.sub-out-fn/test]
    #[test]
    fn op_sub_out_test_bool_output_with_integral_input() {
        let tf = TensorFactory::<bool>::new();
        let tfi = TensorFactory::<i32>::new();

        let sizes = vec![2, 2];

        let a = tfi.make_default(sizes.clone(), vec![0, 1, 1, 0]);
        let b = tfi.make_default(sizes.clone(), vec![2, 3, 4, 3]);

        let out = tf.zeros_default(sizes);

        let mut ctx = context();
        et_expect_kernel_failure!(ctx, sub_out(&mut ctx, &a, &b, &Scalar::from_i64(1), &out));
    }

    // [spec:et:sem:op-sub.torch.executor.native.sub-out-fn/test]
    #[test]
    fn op_sub_out_test_mismatched_non_broadcastable_input_shapes_dies() {
        let tf = TensorFactory::<i32>::new();

        let a = tf.ones_default(vec![4, 2]);
        let b = tf.ones_default(vec![2, 2]);

        let out = tf.zeros_default(vec![8]);

        let mut ctx = context();
        et_expect_kernel_failure!(ctx, sub_out(&mut ctx, &a, &b, &Scalar::from_i64(0), &out));
    }

    // PORT-NOTE: the C++ `ET_SKIP_IF(output_resize, ...)` skips only when the
    // kernel supports implicit resize; the portable kernel's `output_resize`
    // default is `false`, so the body runs here.
    // [spec:et:sem:op-sub.torch.executor.native.sub-out-fn/test]
    #[test]
    fn op_sub_out_test_mismatched_output_shapes_dies() {
        let tf = TensorFactory::<i32>::new();

        let sizes = vec![2, 2];

        let a = tf.ones_default(sizes.clone());
        let b = tf.ones_default(sizes);

        let out = tf.zeros_default(vec![4]);

        let mut ctx = context();
        et_expect_kernel_failure!(ctx, sub_out(&mut ctx, &a, &b, &Scalar::from_i64(0), &out));
    }

    // [spec:et:sem:op-sub.torch.executor.native.sub-out-fn/test]
    #[test]
    fn op_sub_out_test_broadcast_dim_size_is_one_ab() {
        let tf = TensorFactory::<f32>::new();

        let x = tf.make_default(
            vec![3, 2],
            vec![
                0.20342785120010376,
                0.8211539387702942,
                0.12307500839233398,
                0.8268751502037048,
                0.6484894752502441,
                0.8079752326011658,
            ],
        );
        let y = tf.make_default(vec![1, 2], vec![0.22279858589172363, 0.3636378049850464]);
        let expected_result = tf.make_default(
            vec![3, 2],
            vec![
                -0.019370734691619873,
                0.4575161337852478,
                -0.09972357749938965,
                0.46323734521865845,
                0.4256908893585205,
                0.4443374276161194,
            ],
        );

        let out = tf.zeros_default(vec![3, 2]);
        let mut ctx = context();
        sub_out(&mut ctx, &x, &y, &Scalar::from_i64(1), &out);
        assert_tensor_close!(out, expected_result);
    }

    // [spec:et:sem:op-sub.torch.executor.native.sub-out-fn/test]
    #[test]
    fn op_sub_out_test_broadcast_dim_size_missing_ab() {
        let tf = TensorFactory::<f32>::new();

        let x = tf.make_default(
            vec![3, 2],
            vec![
                0.20342785120010376,
                0.8211539387702942,
                0.12307500839233398,
                0.8268751502037048,
                0.6484894752502441,
                0.8079752326011658,
            ],
        );
        let y = tf.make_default(vec![2], vec![0.22279858589172363, 0.3636378049850464]);
        let expected_result = tf.make_default(
            vec![3, 2],
            vec![
                -0.019370734691619873,
                0.4575161337852478,
                -0.09972357749938965,
                0.46323734521865845,
                0.4256908893585205,
                0.4443374276161194,
            ],
        );

        let out = tf.zeros_default(vec![3, 2]);
        let mut ctx = context();
        sub_out(&mut ctx, &x, &y, &Scalar::from_i64(1), &out);
        assert_tensor_close!(out, expected_result);
    }

    // [spec:et:sem:op-sub.torch.executor.native.sub-out-fn/test]
    #[test]
    fn op_sub_out_test_broadcast_dim_size_is_one_ba() {
        let tf = TensorFactory::<f32>::new();

        let x = tf.make_default(vec![1, 2], vec![0.22279858589172363, 0.3636378049850464]);
        let y = tf.make_default(
            vec![3, 2],
            vec![
                0.20342785120010376,
                0.8211539387702942,
                0.12307500839233398,
                0.8268751502037048,
                0.6484894752502441,
                0.8079752326011658,
            ],
        );
        let expected_result = tf.make_default(
            vec![3, 2],
            vec![
                0.019370734691619873,
                -0.4575161337852478,
                0.09972357749938965,
                -0.46323734521865845,
                -0.4256908893585205,
                -0.4443374276161194,
            ],
        );

        let out = tf.zeros_default(vec![3, 2]);
        let mut ctx = context();
        sub_out(&mut ctx, &x, &y, &Scalar::from_i64(1), &out);
        assert_tensor_close!(out, expected_result);
    }

    // [spec:et:sem:op-sub.torch.executor.native.sub-out-fn/test]
    #[test]
    fn op_sub_out_test_broadcast_dim_size_missing_ba() {
        let tf = TensorFactory::<f32>::new();

        let x = tf.make_default(vec![1, 2], vec![0.22279858589172363, 0.3636378049850464]);
        let y = tf.make_default(
            vec![3, 2],
            vec![
                0.20342785120010376,
                0.8211539387702942,
                0.12307500839233398,
                0.8268751502037048,
                0.6484894752502441,
                0.8079752326011658,
            ],
        );
        let expected_result = tf.make_default(
            vec![3, 2],
            vec![
                0.019370734691619873,
                -0.4575161337852478,
                0.09972357749938965,
                -0.46323734521865845,
                -0.4256908893585205,
                -0.4443374276161194,
            ],
        );

        let out = tf.zeros_default(vec![3, 2]);
        let mut ctx = context();
        sub_out(&mut ctx, &x, &y, &Scalar::from_i64(1), &out);
        assert_tensor_close!(out, expected_result);
    }

    // [spec:et:sem:op-sub.torch.executor.native.sub-out-fn/test]
    #[test]
    fn op_sub_out_test_dynamic_shape_upper_bound_same_as_expected() {
        let tf = TensorFactory::<f32>::new();

        let x = tf.make_default(
            vec![3, 2],
            vec![
                0.44215160608291626,
                0.17627692222595215,
                0.46265703439712524,
                0.04357701539993286,
                0.838569700717926,
                0.06833052635192871,
            ],
        );
        let y = tf.make_default(
            vec![3, 2],
            vec![
                0.06382524967193604,
                0.18627053499221802,
                0.5863531231880188,
                0.12181782722473145,
                0.5662856698036194,
                0.930520236492157,
            ],
        );
        let expected_result = tf.make_default(
            vec![3, 2],
            vec![
                0.3783263564109802,
                -0.00999361276626587,
                -0.12369608879089355,
                -0.07824081182479858,
                0.27228403091430664,
                -0.8621897101402283,
            ],
        );

        let out = tf.zeros(vec![3, 2], TensorShapeDynamism::DYNAMIC_BOUND);
        let mut ctx = context();
        sub_out(&mut ctx, &x, &y, &Scalar::from_i64(1), &out);
        assert_tensor_close!(out, expected_result);
    }

    // [spec:et:sem:op-sub.torch.executor.native.sub-out-fn/test]
    #[test]
    fn op_sub_out_test_dynamic_shape_upper_bound_larger_than_expected() {
        let tf = TensorFactory::<f32>::new();

        let x = tf.make_default(
            vec![3, 2],
            vec![
                0.44215160608291626,
                0.17627692222595215,
                0.46265703439712524,
                0.04357701539993286,
                0.838569700717926,
                0.06833052635192871,
            ],
        );
        let y = tf.make_default(
            vec![3, 2],
            vec![
                0.06382524967193604,
                0.18627053499221802,
                0.5863531231880188,
                0.12181782722473145,
                0.5662856698036194,
                0.930520236492157,
            ],
        );
        let expected_result = tf.make_default(
            vec![3, 2],
            vec![
                0.3783263564109802,
                -0.00999361276626587,
                -0.12369608879089355,
                -0.07824081182479858,
                0.27228403091430664,
                -0.8621897101402283,
            ],
        );

        let out = tf.zeros(vec![10, 10], TensorShapeDynamism::DYNAMIC_BOUND);
        let mut ctx = context();
        sub_out(&mut ctx, &x, &y, &Scalar::from_i64(1), &out);
        assert_tensor_close!(out, expected_result);
    }

    // PORT-NOTE: C++ `DISABLED_DynamicShapeUnbound`. Ported and ignored.
    // [spec:et:sem:op-sub.torch.executor.native.sub-out-fn/test]
    #[test]
    #[ignore]
    fn op_sub_out_test_dynamic_shape_unbound() {
        let tf = TensorFactory::<f32>::new();

        let x = tf.make_default(
            vec![3, 2],
            vec![
                0.44215160608291626,
                0.17627692222595215,
                0.46265703439712524,
                0.04357701539993286,
                0.838569700717926,
                0.06833052635192871,
            ],
        );
        let y = tf.make_default(
            vec![3, 2],
            vec![
                0.06382524967193604,
                0.18627053499221802,
                0.5863531231880188,
                0.12181782722473145,
                0.5662856698036194,
                0.930520236492157,
            ],
        );
        let expected_result = tf.make_default(
            vec![3, 2],
            vec![
                0.3783263564109802,
                -0.00999361276626587,
                -0.12369608879089355,
                -0.07824081182479858,
                0.27228403091430664,
                -0.8621897101402283,
            ],
        );

        let out = tf.zeros(vec![1, 1], TensorShapeDynamism::DYNAMIC_UNBOUND);
        let mut ctx = context();
        sub_out(&mut ctx, &x, &y, &Scalar::from_i64(1), &out);
        assert_tensor_close!(out, expected_result);
    }

    // ---- OpSubScalarOutTest ----

    // [spec:et:sem:op-sub.torch.executor.native.sub-scalar-out-fn/test]
    #[test]
    fn op_sub_scalar_out_test_sanity_check() {
        let tf_a = TensorFactory::<i32>::new();
        let tf_out = TensorFactory::<f32>::new();

        let sizes = vec![2, 2];

        let out = tf_out.zeros_default(sizes.clone());

        let mut ctx = context();
        sub_scalar_out(
            &mut ctx,
            &tf_a.make_default(sizes.clone(), vec![1, 2, 4, 8]),
            &Scalar::from_double(0.5),
            &Scalar::from_double(1.5),
            &out,
        );

        assert_tensor_eq!(
            out,
            tf_out.make_default(sizes, vec![0.25, 1.25, 3.25, 7.25])
        );
    }

    // [spec:et:sem:op-sub.torch.executor.native.sub-scalar-out-fn/test]
    #[test]
    fn op_sub_scalar_out_test_optimized_sanity_check() {
        let tf = TensorFactory::<f32>::new();

        let sizes = vec![2, 2];

        let out = tf.zeros_default(sizes.clone());

        let mut ctx = context();
        sub_scalar_out(
            &mut ctx,
            &tf.make_default(sizes.clone(), vec![6.3, 2.1, 5.6, 8.2]),
            &Scalar::from_double(1.9),
            &Scalar::from_double(2.8),
            &out,
        );

        assert_tensor_close!(out, tf.make_default(sizes, vec![0.98, -3.22, 0.28, 2.88]));
    }

    // [spec:et:sem:op-sub.torch.executor.native.sub-scalar-out-fn/test]
    #[test]
    fn op_sub_scalar_out_test_dtype_test_float16_float_int_float16() {
        let tf_half = TensorFactory::<Half>::new();

        let self_ = tf_half.ones_default(vec![2, 2]);
        let other = Scalar::from_double(-1.0);
        let alpha = Scalar::from_i64(1);
        let out = tf_half.zeros_default(vec![2, 2]);
        let out_expected =
            tf_half.full(vec![2, 2], Half::from_f32(2.0), TensorShapeDynamism::STATIC);
        let mut ctx = context();
        sub_scalar_out(&mut ctx, &self_, &other, &alpha, &out);
        assert_tensor_close!(out, out_expected);
    }
}
