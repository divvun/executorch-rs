//! Literal port of kernels/optimized/cpu/op_div.cpp.
//!
//! DEVIATION: `at::vec::map` / `at::vec::map2` / `handle_broadcast_elementwise`
//! over `Vectorized<CTYPE>` collapse to scalar loops (PORTING.md optimized-
//! kernel substitution).

use crate::kernels::optimized::cpu::binary_ops::{
    ElementwiseOptimizedPath, handle_broadcast_elementwise, select_optimized_path,
};
use crate::kernels::portable::cpu::scalar_utils::{get_scalar_dtype, scalar_to};
use crate::kernels::portable::cpu::util::broadcast_util::resize_to_broadcast_target_size;
use crate::kernels::portable::cpu::util::dtype_util::{StaticCast, SupportedTensorDtypes};
use crate::kernels::portable::cpu::util::elementwise_util::{
    apply_bitensor_elementwise_fn, apply_unitensor_elementwise_fn, get_compute_type,
};
use crate::runtime::core::error::Error;
use crate::runtime::core::exec_aten::util::scalar_type_util::{
    is_bits_type, is_complex_type, is_floating_type, is_qint_type, promote_types,
};
use crate::runtime::core::exec_aten::util::tensor_util::{
    resize_tensor, tensors_have_same_dim_order2, tensors_have_same_dim_order3,
};
use crate::runtime::core::portable_type::scalar::Scalar;
use crate::runtime::core::portable_type::scalar_type::ScalarType;
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::core::portable_type::{ComplexDouble, ComplexFloat};
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

// PORT-NOTE: `ET_CHECK` fatal check; mirrored with `runtime_abort`.
macro_rules! et_check {
    ($cond:expr) => {
        if !($cond) {
            crate::runtime::platform::abort::runtime_abort();
        }
    };
}

// PORT-NOTE: complex division `a / b` — mirrors the `ComplexDiv` trait from
// kernels/portable/cpu/op_div.rs.
trait ComplexDiv: Copy {
    fn cdiv(self, b: Self) -> Self;
}
macro_rules! impl_complex_div {
    ($t:ty, $c:ty) => {
        impl ComplexDiv for $t {
            fn cdiv(self, b: Self) -> Self {
                let ar = self.real as f64;
                let ai = self.imag as f64;
                let br = b.real as f64;
                let bi = b.imag as f64;
                let denom = br * br + bi * bi;
                Self {
                    real: ((ar * br + ai * bi) / denom) as $c,
                    imag: ((ai * br - ar * bi) / denom) as $c,
                }
            }
        }
    };
}
impl_complex_div!(ComplexFloat, f32);
impl_complex_div!(ComplexDouble, f64);

// PORT-NOTE: REALB `x / y`, reciprocal `1 / s`, and `x * inv`. Bool promotes
// via i32 (matching the ET_SWITCH_REALB compute path).
trait RealbDiv: Copy {
    fn one() -> Self;
    fn rdiv(self, other: Self) -> Self;
    fn rmul(self, other: Self) -> Self;
}
macro_rules! impl_realb_div_prim {
    ($($t:ty),*) => {$(
        impl RealbDiv for $t {
            fn one() -> Self { 1 as $t }
            fn rdiv(self, other: Self) -> Self { self / other }
            fn rmul(self, other: Self) -> Self { self * other }
        }
    )*};
}
impl_realb_div_prim!(u8, i8, i16, i32, i64, f32, f64);
impl RealbDiv for bool {
    fn one() -> Self {
        true
    }
    fn rdiv(self, other: Self) -> Self {
        ((self as i32) / (other as i32)) != 0
    }
    fn rmul(self, other: Self) -> Self {
        ((self as i32) * (other as i32)) != 0
    }
}

// PORT-NOTE: local `ET_SWITCH_COMPLEX_TYPES` equivalent (ComplexFloat /
// ComplexDouble only — no ComplexHalf), mirroring op_div.rs (portable).
macro_rules! et_switch_complex_types {
    ($type:expr, $ctx:expr, $name:expr, $ctype_alias:ident, $body:block) => {{
        let _st = $type;
        match _st {
            ScalarType::ComplexFloat => {
                #[allow(non_camel_case_types, dead_code)]
                type $ctype_alias = ComplexFloat;
                $body
            }
            ScalarType::ComplexDouble => {
                #[allow(non_camel_case_types, dead_code)]
                type $ctype_alias = ComplexDouble;
                $body
            }
            _ => {
                $ctx.fail(Error::InvalidArgument);
                crate::et_log!(
                    Error,
                    "Unhandled dtype {} for {}",
                    crate::runtime::core::exec_aten::util::scalar_type_util::to_string(_st),
                    $name
                );
            }
        }
    }};
}

// [spec:et:def:op-div.torch.executor.native.get-common-type-fn]
// [spec:et:sem:op-div.torch.executor.native.get-common-type-fn]
fn get_common_type(a_type: ScalarType, b_type: ScalarType) -> ScalarType {
    if is_complex_type(a_type) || is_complex_type(b_type) {
        return promote_types(a_type, b_type, false);
    }
    et_check!(!is_qint_type(a_type) && !is_bits_type(a_type));
    et_check!(!is_qint_type(b_type) && !is_bits_type(b_type));

    if is_floating_type(a_type) && is_floating_type(b_type) {
        promote_types(a_type, b_type, false)
    } else if is_floating_type(a_type) {
        a_type
    } else if is_floating_type(b_type) {
        b_type
    } else {
        ScalarType::Float
    }
}

// [spec:et:def:op-div.torch.executor.native.opt-div-out-fn]
// [spec:et:sem:op-div.torch.executor.native.opt-div-out-fn]
pub fn opt_div_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    a: &'a Tensor<'b>,
    b: &'a Tensor<'b>,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
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

    let op_name = "div.out";

    let a_type = a.scalar_type();
    let b_type = b.scalar_type();
    let out_type = out.scalar_type();

    // Handle complex types
    if is_complex_type(a_type) || is_complex_type(b_type) {
        let common_type = get_common_type(a_type, b_type);
        et_switch_complex_types!(common_type, ctx, op_name, CTYPE, {
            let a_data = a.const_data_ptr::<CTYPE>();
            let b_data = b.const_data_ptr::<CTYPE>();
            let out_data = out.mutable_data_ptr::<CTYPE>();
            for i in 0..out.numel() {
                unsafe {
                    *out_data.offset(i) = (*a_data.offset(i)).cdiv(*b_data.offset(i));
                }
            }
        });
        return out;
    }

    if a.numel() == 1 || b.numel() == 1 {
        if a_type == b_type
            && a_type == out_type
            && a_type != ScalarType::Half
            && a_type != ScalarType::BFloat16
        {
            let tensor: &Tensor;
            let scalar: &Tensor;
            let tensor_type: ScalarType;
            let scalar_type: ScalarType;
            if a.numel() == 1 {
                tensor = b;
                tensor_type = b_type;
                scalar = a;
                scalar_type = a_type;
            } else {
                tensor = a;
                tensor_type = a_type;
                scalar = b;
                scalar_type = b_type;
            }
            crate::et_switch_realb_types!(tensor_type, ctx, op_name, CTYPE, {
                crate::et_switch_realb_types!(scalar_type, ctx, op_name, CTYPE_SCALAR, {
                    let scalar_val: CTYPE_SCALAR =
                        unsafe { *scalar.const_data_ptr::<CTYPE_SCALAR>() };
                    let scalar_casted: CTYPE =
                        <CTYPE as StaticCast<CTYPE_SCALAR>>::static_cast(scalar_val);

                    // DEVIATION: at::vec::map -> scalar loop.
                    let out_data = out.mutable_data_ptr::<CTYPE>();
                    let tensor_data = tensor.const_data_ptr::<CTYPE>();
                    if a.numel() == 1 {
                        for i in 0..out.numel() {
                            unsafe {
                                *out_data.offset(i) = scalar_casted.rdiv(*tensor_data.offset(i));
                            }
                        }
                    } else {
                        let inv_scalar_casted = <CTYPE as RealbDiv>::one().rdiv(scalar_casted);
                        for i in 0..out.numel() {
                            unsafe {
                                *out_data.offset(i) =
                                    (*tensor_data.offset(i)).rmul(inv_scalar_casted);
                            }
                        }
                    }
                });
            });
            return out;
        }
    }

    let selected_optimized_path = select_optimized_path(a, b, out);
    if selected_optimized_path == ElementwiseOptimizedPath::KTreatAs1d {
        crate::et_switch_realb_types!(out_type, ctx, op_name, CTYPE, {
            // DEVIATION: at::vec::map2 -> scalar loop; x / y.
            let out_data = out.mutable_data_ptr::<CTYPE>();
            let a_data = a.const_data_ptr::<CTYPE>();
            let b_data = b.const_data_ptr::<CTYPE>();
            for i in 0..out.numel() {
                unsafe {
                    *out_data.offset(i) = (*a_data.offset(i)).rdiv(*b_data.offset(i));
                }
            }
        });
    } else if selected_optimized_path != ElementwiseOptimizedPath::KNone {
        // Reason for using alpha is becasuse handle_broadcast_elementwise
        // is used for add and sub as well:
        crate::et_switch_realb_types!(out_type, ctx, op_name, CTYPE, {
            if selected_optimized_path == ElementwiseOptimizedPath::KBroadcast2dBy1dReverseArguments
                || selected_optimized_path
                    == ElementwiseOptimizedPath::KBroadcastLastDimReverseArguments
                || selected_optimized_path
                    == ElementwiseOptimizedPath::KBroadcastNdByNdReverseArguments
            {
                let div_lambda = |x: CTYPE, y: CTYPE| y.rdiv(x);
                handle_broadcast_elementwise::<CTYPE, _>(
                    ctx,
                    &div_lambda,
                    a,
                    b,
                    out,
                    selected_optimized_path,
                    None,
                );
            } else {
                let div_lambda = |x: CTYPE, y: CTYPE| x.rdiv(y);
                handle_broadcast_elementwise::<CTYPE, _>(
                    ctx,
                    &div_lambda,
                    a,
                    b,
                    out,
                    selected_optimized_path,
                    None,
                );
            }
        });
    } else {
        let common_type = get_common_type(a.scalar_type(), b.scalar_type());
        let mut common_type_mut = common_type;
        let compute_type = get_compute_type(&mut common_type_mut);

        crate::et_switch_float_types!(compute_type, ctx, op_name, CTYPE_COMPUTE, {
            apply_bitensor_elementwise_fn::<CTYPE_COMPUTE, _>(
                |vals: &[CTYPE_COMPUTE]| vals[0] / vals[1],
                ctx,
                a,
                SupportedTensorDtypes::REALHBBF16,
                b,
                SupportedTensorDtypes::REALHBBF16,
                out,
                SupportedTensorDtypes::FLOATHBF16,
                false,
            );
        });
    }

    out
}

// [spec:et:def:op-div.torch.executor.native.opt-div-scalar-out-fn]
// [spec:et:sem:op-div.torch.executor.native.opt-div-scalar-out-fn]
pub fn opt_div_scalar_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    a: &Tensor,
    b: &Scalar,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    let a_type = a.scalar_type();
    let b_type = get_scalar_dtype(*b);
    let common_type = if is_floating_type(a_type) {
        a_type
    } else {
        ScalarType::Float
    };
    let out_type = out.scalar_type();

    // Check Common Dtype
    crate::et_kernel_check!(ctx, common_type == out_type, InvalidArgument, out);

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

    let op_name = "div.Scalar_out";

    if a_type == common_type
        && a_type == out_type
        && a_type != ScalarType::Half
        && a_type != ScalarType::BFloat16
    {
        crate::et_switch_real_types!(a_type, ctx, op_name, CTYPE, {
            crate::et_switch_realb_types!(b_type, ctx, op_name, CTYPE_B, {
                let mut b_val: CTYPE_B = Default::default();
                crate::et_extract_scalar!(*b, b_val);
                let b_casted: CTYPE = <CTYPE as StaticCast<CTYPE_B>>::static_cast(b_val);

                // DEVIATION: at::vec::map -> scalar loop; x * (1 / b_casted).
                let inv_b_casted = <CTYPE as RealbDiv>::one().rdiv(b_casted);
                let out_data = out.mutable_data_ptr::<CTYPE>();
                let a_data = a.const_data_ptr::<CTYPE>();
                for i in 0..out.numel() {
                    unsafe {
                        *out_data.offset(i) = (*a_data.offset(i)).rmul(inv_b_casted);
                    }
                }
            });
        });
    } else {
        let mut common_type_mut = common_type;
        let compute_type = get_compute_type(&mut common_type_mut);

        crate::et_switch_float_types!(compute_type, ctx, op_name, CTYPE_COMPUTE, {
            let val_b: CTYPE_COMPUTE = scalar_to::<CTYPE_COMPUTE>(b);
            apply_unitensor_elementwise_fn::<CTYPE_COMPUTE, _>(
                move |vals: &[CTYPE_COMPUTE]| vals[0] / val_b,
                ctx,
                a,
                SupportedTensorDtypes::REALHBBF16,
                out,
                SupportedTensorDtypes::SAME_AS_COMMON,
                false,
            );
        });
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::core::exec_aten::testing_util::tensor_factory::{
        FactoryValue, TensorFactory,
    };
    use crate::runtime::core::exec_aten::util::scalar_type_util::CppTypeToScalarType;
    use crate::runtime::core::memory_allocator::{MemoryAllocator, MemoryAllocatorBase};
    use crate::runtime::core::portable_type::{BFloat16, Half};
    use crate::{assert_tensor_close, assert_tensor_close_with_tol, assert_tensor_eq};

    fn context() -> KernelRuntimeContext<'static> {
        crate::runtime::platform::platform::pal_init();
        KernelRuntimeContext::new(
            crate::extension::module::module::null_event_tracer(),
            core::ptr::null_mut::<MemoryAllocator>() as *mut dyn MemoryAllocatorBase,
        )
    }

    fn op_div_out<'a, 'b>(
        a: &'a Tensor<'b>,
        b: &'a Tensor<'b>,
        out: &'a Tensor<'b>,
    ) -> &'a Tensor<'b> {
        let mut ctx = context();
        opt_div_out(&mut ctx, a, b, out)
    }

    fn op_div_scalar_out<'a, 'b>(a: &Tensor, b: &Scalar, out: &'a Tensor<'b>) -> &'a Tensor<'b> {
        let mut ctx = context();
        opt_div_scalar_out(&mut ctx, a, b, out)
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
            Half::from_f64(v)
        }
    }
    impl FromF64Elem for BFloat16 {
        fn from_f64(v: f64) -> Self {
            BFloat16::from_f64(v)
        }
    }

    fn d<T: FromF64Elem>(v: &[f64]) -> Vec<T> {
        v.iter().map(|&x| T::from_f64(x)).collect()
    }

    // op_div_test.cpp test_div<DTYPE_A, DTYPE_B, DTYPE_OUT>.
    fn test_div<A, B, OUT>()
    where
        A: CppTypeToScalarType + FactoryValue + FromF64Elem,
        B: CppTypeToScalarType + FactoryValue + FromF64Elem,
        OUT: CppTypeToScalarType + FactoryValue + FromF64Elem,
    {
        let tf_a = TensorFactory::<A>::new();
        let tf_b = TensorFactory::<B>::new();
        let tf_out = TensorFactory::<OUT>::new();

        let sizes = vec![2, 2];
        let out = tf_out.zeros_default(sizes.clone());

        op_div_out(
            &tf_a.make_default(sizes.clone(), d(&[1.0, 2.0, 4.0, 8.0])),
            &tf_b.make_default(sizes.clone(), d(&[8.0, 4.0, 2.0, 1.0])),
            &out,
        );
        assert_tensor_close!(out, tf_out.make_default(sizes, d(&[0.125, 0.5, 2.0, 8.0])));
    }

    // op_div_test.cpp test_div<Float, Float, Float> specialization (division by
    // zero and inf/nan semantics on the same-dtype kTreatAs1d path).
    // [spec:et:sem:op-div.torch.executor.native.opt-div-out-fn/test]
    #[test]
    fn op_div_out_test_float_special_values() {
        let tf = TensorFactory::<f32>::new();
        let sizes = vec![2, 5];
        let out = tf.zeros_default(sizes.clone());

        op_div_out(
            &tf.make_default(
                sizes.clone(),
                vec![
                    1.0,
                    2.0,
                    4.0,
                    8.0,
                    f32::INFINITY,
                    f32::NEG_INFINITY,
                    f32::NAN,
                    1.0,
                    1.0,
                    1.0,
                ],
            ),
            &tf.make_default(
                sizes.clone(),
                vec![
                    8.0,
                    0.0,
                    2.0,
                    1.0,
                    f32::INFINITY,
                    f32::NEG_INFINITY,
                    f32::NAN,
                    f32::INFINITY,
                    f32::NEG_INFINITY,
                    f32::NAN,
                ],
            ),
            &out,
        );
        assert_tensor_close!(
            out,
            tf.make_default(
                sizes,
                vec![
                    0.125,
                    f32::INFINITY,
                    2.0,
                    8.0,
                    f32::NAN,
                    f32::NAN,
                    f32::NAN,
                    0.0,
                    0.0,
                    f32::NAN,
                ]
            )
        );
    }

    // Same-dtype float instances plus integer/bool inputs with float output
    // (integer inputs promote to the float compute type via the fallback path).
    // [spec:et:sem:op-div.torch.executor.native.opt-div-out-fn/test]
    #[test]
    fn op_div_out_test_real_dtypes_supported() {
        test_div::<f64, f64, f64>();
        test_div::<Half, Half, Half>();
        test_div::<BFloat16, BFloat16, BFloat16>();
        test_div::<i32, i32, f32>();
        test_div::<u8, i16, f32>();
        test_div::<i64, f32, f32>();
        test_div::<i32, i32, f64>();
    }

    // op_div_test.cpp test_div<Bool, Float, Float>.
    // [spec:et:sem:op-div.torch.executor.native.opt-div-out-fn/test]
    #[test]
    fn op_div_out_test_bool_input() {
        let tf_b = TensorFactory::<bool>::new();
        let tf = TensorFactory::<f32>::new();
        let sizes = vec![2, 2];
        let out = tf.zeros_default(sizes.clone());

        op_div_out(
            &tf_b.make_default(sizes.clone(), vec![true, true, true, true]),
            &tf.make_default(sizes.clone(), vec![4.0, 4.0, 2.0, 1.0]),
            &out,
        );
        assert_tensor_close!(out, tf.make_default(sizes, vec![0.25, 0.25, 0.5, 1.0]));
    }

    // [spec:et:sem:op-div.torch.executor.native.opt-div-out-fn/test]
    #[test]
    fn op_div_out_test_broadcast_supported_1() {
        let tf = TensorFactory::<f32>::new();

        let a = tf.make_default(vec![2, 1, 2, 1], vec![4., 8., 12., 16.]);
        let b = tf.make_default(vec![2, 1, 4], vec![1., 1., 1., 1., 2., 2., 2., 2.]);
        let out = tf.zeros_default(vec![2, 2, 2, 4]);

        op_div_out(&a, &b, &out);

        let ret = tf.make_default(
            vec![2, 2, 2, 4],
            vec![
                4., 4., 4., 4., 8., 8., 8., 8., 2., 2., 2., 2., 4., 4., 4., 4., 12., 12., 12., 12.,
                16., 16., 16., 16., 6., 6., 6., 6., 8., 8., 8., 8.,
            ],
        );
        assert_tensor_eq!(out, ret);
    }

    // [spec:et:sem:op-div.torch.executor.native.opt-div-out-fn/test]
    #[test]
    fn op_div_out_test_broadcast_supported_2() {
        let tf = TensorFactory::<f32>::new();

        let a = tf.make_default(vec![3, 2, 1], vec![2., 3., 4., 5., 6., 7.]);
        let b = tf.make_default(vec![1, 2, 1], vec![2., 2.]);
        let out = tf.zeros_default(vec![3, 2, 1]);

        op_div_out(&a, &b, &out);
        assert_tensor_eq!(
            out,
            tf.make_default(vec![3, 2, 1], vec![1., 1.5, 2., 2.5, 3., 3.5])
        );
    }

    // b.numel() == 1 same-dtype fast path (x * (1/b)) and the a.numel() == 1
    // form (b / x elementwise).
    // [spec:et:sem:op-div.torch.executor.native.opt-div-out-fn/test]
    #[test]
    fn op_div_out_test_broadcast_scalar_supported() {
        let tf = TensorFactory::<f32>::new();

        let a = tf.make_default(vec![2, 1, 3], vec![2., 3., 4., 5., 6., 7.]);
        let b = tf.make_default(vec![1], vec![2.]);
        let out = tf.zeros_default(vec![2, 1, 3]);

        op_div_out(&a, &b, &out);
        assert_tensor_eq!(
            out,
            tf.make_default(vec![2, 1, 3], vec![1., 1.5, 2., 2.5, 3., 3.5])
        );

        // a.numel() == 1: scalar / tensor.
        let a1 = tf.make_default(vec![1, 1, 1], vec![8.]);
        let b1 = tf.make_default(vec![3, 1, 1], vec![2., 4., 8.]);
        let out1 = tf.zeros_default(vec![3, 1, 1]);

        op_div_out(&a1, &b1, &out1);
        assert_tensor_eq!(out1, tf.make_default(vec![3, 1, 1], vec![4., 2., 1.]));

        // Swapped: tensor / scalar.
        let out2 = tf.zeros_default(vec![3, 1, 1]);
        op_div_out(&b1, &a1, &out2);
        assert_tensor_eq!(out2, tf.make_default(vec![3, 1, 1], vec![0.25, 0.5, 1.]));
    }

    // [spec:et:sem:op-div.torch.executor.native.opt-div-out-fn/test]
    #[test]
    fn op_div_out_test_broadcast_scalar_rank0_supported() {
        let tf = TensorFactory::<f32>::new();

        let a = tf.make_default(vec![1], vec![8.]);
        let b = tf.make_default(vec![], vec![2.]);
        let out = tf.zeros_default(vec![1]);

        op_div_out(&a, &b, &out);
        assert_tensor_eq!(out, tf.make_default(vec![1], vec![4.]));

        op_div_out(&b, &a, &out);
        assert_tensor_eq!(out, tf.make_default(vec![1], vec![0.25]));
    }

    // op_div_test.cpp test_broadcast_3D<Float> (kBroadcastNdByNd and reverse).
    // [spec:et:sem:op-div.torch.executor.native.opt-div-out-fn/test]
    #[test]
    fn op_div_out_test_broadcast_nd() {
        let tf = TensorFactory::<f32>::new();

        let a = tf.make_default(
            vec![2, 2, 3],
            vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.],
        );
        let b = tf.make_default(vec![2, 1, 3], vec![2., 3., 4., 5., 6., 7.]);

        let out = tf.zeros_default(vec![2, 2, 3]);
        let expected = tf.make_default(
            vec![2, 2, 3],
            vec![
                0.5000, 0.6667, 0.75002, 2.0000, 1.6667, 1.5000, 1.4000, 1.3333, 1.2857, 2.0000,
                1.8333, 1.7143,
            ],
        );
        assert_tensor_close_with_tol!(op_div_out(&a, &b, &out), expected, 1e-4, 1e-4);

        let expected = tf.make_default(
            vec![2, 2, 3],
            vec![
                2.0000, 1.5000, 1.3333, 0.5000, 0.6000, 0.6667, 0.7143, 0.7500, 0.7778, 0.5000,
                0.5455, 0.5833,
            ],
        );
        assert_tensor_close_with_tol!(op_div_out(&b, &a, &out), expected, 1e-4, 1e-4);
    }

    // [spec:et:sem:op-div.torch.executor.native.opt-div-out-fn/test]
    #[test]
    fn op_div_out_test_mismatched_shapes_dies() {
        let tf = TensorFactory::<f32>::new();

        let a = tf.ones_default(vec![2, 2]);
        let b = tf.ones_default(vec![3, 3]);
        let out = tf.zeros_default(vec![2, 2]);

        let mut ctx = context();
        opt_div_out(&mut ctx, &a, &b, &out);
        assert_ne!(ctx.failure_state(), Error::Ok);
    }

    // [spec:et:sem:op-div.torch.executor.native.opt-div-out-fn/test]
    #[test]
    fn op_div_out_test_non_float_output_dtype_dies() {
        let tf_float = TensorFactory::<f32>::new();
        let tf_out = TensorFactory::<i32>::new();

        let a = tf_float.ones_default(vec![2, 5]);
        let b = tf_float.ones_default(vec![2, 5]);
        let out = tf_out.zeros_default(vec![2, 5]);

        let mut ctx = context();
        opt_div_out(&mut ctx, &a, &b, &out);
        assert_ne!(ctx.failure_state(), Error::Ok);
    }

    // [spec:et:sem:op-div.torch.executor.native.opt-div-out-fn/test]
    #[test]
    fn op_div_out_test_complex_float_basic() {
        let tf = TensorFactory::<ComplexFloat>::new();
        let mk = |re: f32, im: f32| ComplexFloat { real: re, imag: im };

        let sizes = vec![2, 2];
        // (1+2i) / (1+0i) = (1+2i)
        // (4+4i) / (2+0i) = (2+2i)
        // (3+4i) / (1-1i) = (-0.5+3.5i)
        // (8+0i) / (2+2i) = (2-2i)
        let a = tf.make_default(
            sizes.clone(),
            vec![mk(1.0, 2.0), mk(4.0, 4.0), mk(3.0, 4.0), mk(8.0, 0.0)],
        );
        let b = tf.make_default(
            sizes.clone(),
            vec![mk(1.0, 0.0), mk(2.0, 0.0), mk(1.0, -1.0), mk(2.0, 2.0)],
        );
        let out = tf.zeros_default(sizes.clone());

        op_div_out(&a, &b, &out);
        assert_tensor_close!(
            out,
            tf.make_default(
                sizes,
                vec![mk(1.0, 2.0), mk(2.0, 2.0), mk(-0.5, 3.5), mk(2.0, -2.0)]
            )
        );
    }

    // [spec:et:sem:op-div.torch.executor.native.opt-div-out-fn/test]
    #[test]
    fn op_div_out_test_complex_double_basic() {
        let tf = TensorFactory::<ComplexDouble>::new();
        let mk = |re: f64, im: f64| ComplexDouble { real: re, imag: im };

        let a = tf.make_default(vec![2], vec![mk(6.0, 8.0), mk(4.0, 0.0)]);
        let b = tf.make_default(vec![2], vec![mk(2.0, 0.0), mk(0.0, 2.0)]);
        let out = tf.zeros_default(vec![2]);

        op_div_out(&a, &b, &out);
        assert_tensor_close!(
            out,
            tf.make_default(vec![2], vec![mk(3.0, 4.0), mk(0.0, -2.0)])
        );
    }

    // [spec:et:sem:op-div.torch.executor.native.opt-div-out-fn/test]
    #[test]
    fn op_div_out_test_complex_float_identity() {
        let tf = TensorFactory::<ComplexFloat>::new();
        let mk = |re: f32, im: f32| ComplexFloat { real: re, imag: im };

        let a = tf.make_default(vec![3], vec![mk(1.0, 2.0), mk(3.0, 4.0), mk(-5.0, 6.0)]);
        let one = tf.make_default(vec![3], vec![mk(1.0, 0.0), mk(1.0, 0.0), mk(1.0, 0.0)]);
        let out = tf.zeros_default(vec![3]);

        op_div_out(&a, &one, &out);
        assert_tensor_close!(out, a);
    }

    // ---- OpDivScalarOutTest ----

    // [spec:et:sem:op-div.torch.executor.native.opt-div-scalar-out-fn/test]
    #[test]
    fn op_div_scalar_out_test_sanity_check_int_scalar() {
        let tf_a = TensorFactory::<i32>::new();
        let tf_out = TensorFactory::<f32>::new();
        let sizes = vec![2, 2];
        let out = tf_out.zeros_default(sizes.clone());

        op_div_scalar_out(
            &tf_a.make_default(sizes.clone(), vec![1, 2, 4, -9]),
            &Scalar::from_i64(2),
            &out,
        );
        assert_tensor_eq!(out, tf_out.make_default(sizes, vec![0.5, 1.0, 2.0, -4.5]));
    }

    // [spec:et:sem:op-div.torch.executor.native.opt-div-scalar-out-fn/test]
    #[test]
    fn op_div_scalar_out_test_sanity_check_float_scalar() {
        let tf_a = TensorFactory::<i32>::new();
        let tf_out = TensorFactory::<f32>::new();
        let sizes = vec![2, 2];
        let out = tf_out.zeros_default(sizes.clone());

        op_div_scalar_out(
            &tf_a.make_default(sizes.clone(), vec![1, 2, 4, -9]),
            &Scalar::from_double(2.0),
            &out,
        );
        assert_tensor_eq!(out, tf_out.make_default(sizes, vec![0.5, 1.0, 2.0, -4.5]));
    }

    // Same-dtype float fast path (x * (1/b)).
    // [spec:et:sem:op-div.torch.executor.native.opt-div-scalar-out-fn/test]
    #[test]
    fn op_div_scalar_out_test_optimized_sanity_check() {
        let tf = TensorFactory::<f32>::new();
        let sizes = vec![2, 2];
        let out = tf.zeros_default(sizes.clone());

        op_div_scalar_out(
            &tf.make_default(sizes.clone(), vec![1.3, 2.1, 4.6, 8.2]),
            &Scalar::from_double(2.0),
            &out,
        );
        assert_tensor_close!(out, tf.make_default(sizes, vec![0.65, 1.05, 2.3, 4.1]));
    }

    // [spec:et:sem:op-div.torch.executor.native.opt-div-scalar-out-fn/test]
    #[test]
    fn op_div_scalar_out_test_mismatched_out_dtype_dies() {
        let tf_a = TensorFactory::<f32>::new();
        let tf_out = TensorFactory::<i32>::new();
        let out = tf_out.zeros_default(vec![2, 2]);

        let mut ctx = context();
        opt_div_scalar_out(
            &mut ctx,
            &tf_a.ones_default(vec![2, 2]),
            &Scalar::from_i64(2),
            &out,
        );
        assert_ne!(ctx.failure_state(), Error::Ok);
    }
}
