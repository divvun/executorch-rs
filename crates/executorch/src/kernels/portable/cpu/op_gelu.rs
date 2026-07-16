//! Literal port of kernels/portable/cpu/op_gelu.cpp.

use crate::kernels::portable::cpu::util::activation_ops_util::check_gelu_args;
use crate::kernels::portable::cpu::util::functional_util::apply_unary_map_fn;
use crate::runtime::core::error::Error;
use crate::runtime::core::exec_aten::util::tensor_util::{
    resize_tensor, tensors_have_same_dim_order2,
};
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::core::portable_type::{BFloat16, Half};
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

// math_constants.h
const M_SQRT2: f64 = 1.41421356237309504880; // sqrt(2)
const M_2_SQRTPI: f64 = 1.12837916709551257390; // 2/sqrt(pi)
const M_SQRT1_2: f64 = 0.70710678118654752440; // 1/sqrt(2)

// PORT-NOTE: the C++ closures do per-CTYPE math over FLOATHBF16 {Half, Float,
// Double, BFloat16}. Rust cannot express `std::tanh`/`std::erf`/`std::numeric_
// limits<CTYPE>::infinity()` and the `c10::Half` implicit float promotions
// generically, so — mirroring op_elu.rs / op_leaky_relu.rs's per-type-trait
// strategy — the two closures are modeled by a `GeluElem` trait, one impl per
// CTYPE, each reproducing the exact C++ arithmetic (including the double-literal
// usual-arithmetic-conversion promotions and, for the reduced-float types, the
// per-operation Half/BFloat16 rounding and the `std::tanh(float)` promotion).
trait GeluElem: Copy {
    fn gelu_tanh(self) -> Self;
    fn gelu_none(self) -> Self;
}

macro_rules! impl_gelu_elem_native {
    ($t:ty) => {
        impl GeluElem for $t {
            fn gelu_tanh(self) -> Self {
                let x = self;
                if x == <$t>::NEG_INFINITY {
                    return 0.0 as $t;
                } else if x == <$t>::INFINITY {
                    return <$t>::INFINITY;
                }
                let k_beta: $t = (M_SQRT2 * M_2_SQRTPI * 0.5) as $t;
                let k_kappa: $t = (0.044715f32) as $t;

                let x_cubed: $t = x * x * x;
                let inner: $t = k_beta * (x + k_kappa * x_cubed);
                // 0.5 * x * (1.0 + std::tanh(inner)); computed in f64 per the
                // usual arithmetic conversions, then narrowed to CTYPE.
                let ret: $t = (0.5 * (x as f64) * (1.0 + (inner.tanh() as f64))) as $t;

                ret
            }
            fn gelu_none(self) -> Self {
                let x = self;
                if x == <$t>::NEG_INFINITY {
                    return 0.0 as $t;
                } else if x == <$t>::INFINITY {
                    return <$t>::INFINITY;
                }
                (0.5 * (x as f64) * (1.0 + ((x * (M_SQRT1_2 as $t)).erf_impl() as f64))) as $t
            }
        }
    };
}

// PORT-NOTE: `std::erf` is not in Rust std; it maps to C `erf`/`erff` via libc,
// matching `std::erf` (same as vectorized_math.rs). A tiny local `ErfImpl`
// avoids widening `f32` arithmetic through `f64` before the erf call.
trait ErfImpl: Copy {
    fn erf_impl(self) -> Self;
}
unsafe extern "C" {
    fn erff(x: f32) -> f32;
    fn erf(x: f64) -> f64;
}
impl ErfImpl for f32 {
    fn erf_impl(self) -> Self {
        unsafe { erff(self) }
    }
}
impl ErfImpl for f64 {
    fn erf_impl(self) -> Self {
        unsafe { erf(self) }
    }
}

impl_gelu_elem_native!(f32);
impl_gelu_elem_native!(f64);

macro_rules! impl_gelu_elem_reduced {
    ($t:ty) => {
        impl GeluElem for $t {
            fn gelu_tanh(self) -> Self {
                let x = self;
                if x == <$t>::NEG_INFINITY {
                    return <$t>::from_f32(0.0);
                } else if x == <$t>::INFINITY {
                    return <$t>::INFINITY;
                }
                // `const CTYPE kBeta = double_product;` narrowed to the reduced type.
                let k_beta: $t = <$t>::from_f64(M_SQRT2 * M_2_SQRTPI * 0.5);
                let k_kappa: $t = <$t>::from_f32(0.044715f32);

                // c10::Half/BFloat16 `operator*`/`operator+` promote to float,
                // compute, and round back to the reduced type each step.
                let mul = |a: $t, b: $t| <$t>::from_f32(a.to_f32() * b.to_f32());
                let add = |a: $t, b: $t| <$t>::from_f32(a.to_f32() + b.to_f32());

                let x_cubed: $t = mul(mul(x, x), x);
                let inner: $t = mul(k_beta, add(x, mul(k_kappa, x_cubed)));
                // std::tanh(inner): inner promotes to float, tanh returns float.
                let tanh_f32: f32 = inner.to_f32().tanh();
                // 0.5 * x * (1.0 + tanh) in f64 per usual arithmetic conversions.
                let ret: $t = <$t>::from_f64(0.5 * (x.to_f64()) * (1.0 + (tanh_f32 as f64)));

                ret
            }
            fn gelu_none(self) -> Self {
                let x = self;
                if x == <$t>::NEG_INFINITY {
                    return <$t>::from_f32(0.0);
                } else if x == <$t>::INFINITY {
                    return <$t>::INFINITY;
                }
                // x * M_SQRT1_2: M_SQRT1_2 narrowed to CTYPE, multiply promotes
                // to float; std::erf promotes to float, returns float.
                let arg: $t = <$t>::from_f32(x.to_f32() * <$t>::from_f64(M_SQRT1_2).to_f32());
                let erf_f32: f32 = arg.to_f32().erf_impl();
                <$t>::from_f64(0.5 * (x.to_f64()) * (1.0 + (erf_f32 as f64)))
            }
        }
    };
}
impl_gelu_elem_reduced!(Half);
impl_gelu_elem_reduced!(BFloat16);

// PORT-NOTE: `(void)ctx;` dropped. `Tensor& out` / returned `Tensor&` become
// `&'a Tensor` (interior mutation through `*mut TensorImpl`). The C++
// `ET_CHECK_MSG(false, ...)` else-branch is unreachable (check_gelu_args already
// rejected any non-"tanh"/"none" value); it is reproduced as a fatal abort.

// [spec:et:def:op-gelu.torch.executor.native.gelu-out-fn]
// [spec:et:sem:op-gelu.torch.executor.native.gelu-out-fn]
pub fn gelu_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    in_: &Tensor,
    approximate: &str,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    // (void)ctx;

    crate::et_kernel_check!(
        ctx,
        check_gelu_args(in_, approximate, out),
        InvalidArgument,
        out
    );

    crate::et_kernel_check!(
        ctx,
        resize_tensor(out, in_.sizes()) == Error::Ok,
        InvalidArgument,
        out
    );

    crate::et_kernel_check!(
        ctx,
        tensors_have_same_dim_order2(in_, out),
        InvalidArgument,
        out
    );

    crate::et_switch_floathbf16_types!(in_.scalar_type(), ctx, "gelu.out", CTYPE, {
        if approximate == "tanh" {
            apply_unary_map_fn(
                |x: CTYPE| -> CTYPE { <CTYPE as GeluElem>::gelu_tanh(x) },
                in_.const_data_ptr::<CTYPE>(),
                out.mutable_data_ptr::<CTYPE>(),
                in_.numel() as i64,
                1,
            );
        } else if approximate == "none" {
            apply_unary_map_fn(
                |x: CTYPE| -> CTYPE { <CTYPE as GeluElem>::gelu_none(x) },
                in_.const_data_ptr::<CTYPE>(),
                out.mutable_data_ptr::<CTYPE>(),
                in_.numel() as i64,
                1,
            );
        } else {
            crate::runtime::platform::abort::runtime_abort();
        }
    });

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assert_tensor_close;
    use crate::runtime::core::error::Error;
    use crate::runtime::core::exec_aten::testing_util::tensor_factory::{
        FactoryValue, TensorFactory,
    };
    use crate::runtime::core::exec_aten::util::scalar_type_util::CppTypeToScalarType;
    use crate::runtime::core::memory_allocator::{MemoryAllocator, MemoryAllocatorBase};
    use crate::runtime::core::tensor_shape_dynamism::TensorShapeDynamism;

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
            Half::from_f64(v)
        }
    }
    impl FromF64 for BFloat16 {
        fn from_f64(v: f64) -> Self {
            BFloat16::from_f64(v)
        }
    }

    fn d<T: FromF64>(v: &[f64]) -> Vec<T> {
        v.iter().map(|&x| T::from_f64(x)).collect()
    }

    fn test_gelu_execution<T>()
    where
        T: CppTypeToScalarType + FactoryValue + FromF64,
    {
        let tf = TensorFactory::<T>::new();
        let sizes = vec![3, 2];

        let in_ = tf.make_default(
            sizes.clone(),
            d::<T>(&[-0.4775, 0.2948, -0.3984, 1.8690, -0.4048, -0.4848]),
        );

        let out = tf.zeros_default(sizes.clone());

        let mut ctx = context();
        gelu_out(&mut ctx, &in_, "none", &out);
        assert_tensor_close!(
            out,
            tf.make_default(
                sizes.clone(),
                d::<T>(&[-0.15113, 0.181575, -0.137515, 1.81141, -0.13877, -0.152183])
            )
        );

        gelu_out(&mut ctx, &in_, "tanh", &out);
        assert_tensor_close!(
            out,
            tf.make_default(
                sizes,
                d::<T>(&[-0.151145, 0.181573, -0.137522, 1.8114, -0.138778, -0.152199])
            )
        );
    }

    // [spec:et:sem:op-gelu.torch.executor.native.gelu-out-fn/test]
    #[test]
    fn op_gelu_out_test_float_tensors() {
        test_gelu_execution::<f32>();
    }

    // [spec:et:sem:op-gelu.torch.executor.native.gelu-out-fn/test]
    #[test]
    fn op_gelu_out_test_half_tensors() {
        test_gelu_execution::<Half>();
    }

    // [spec:et:sem:op-gelu.torch.executor.native.gelu-out-fn/test]
    #[test]
    fn op_gelu_out_test_bfloat16_tensors() {
        test_gelu_execution::<BFloat16>();
    }

    // PORT-NOTE: C++ guards with `ET_SKIP_IF(!op_gelu_dtype_double, "")`; the
    // portable kernel supports Double (FLOATHBF16), so this runs.
    // [spec:et:sem:op-gelu.torch.executor.native.gelu-out-fn/test]
    #[test]
    fn op_gelu_out_test_double_tensors() {
        test_gelu_execution::<f64>();
    }

    // [spec:et:sem:op-gelu.torch.executor.native.gelu-out-fn/test]
    #[test]
    fn op_gelu_out_test_unhandled_dtype_dies() {
        // gelu() doesn't handle Bool.
        let tf = TensorFactory::<bool>::new();
        let sizes = vec![2, 2];
        let a = tf.make_default(sizes.clone(), vec![false, true, false, true]);
        let out = tf.zeros_default(sizes);

        let mut ctx = context();
        et_expect_kernel_failure!(ctx, gelu_out(&mut ctx, &a, "none", &out));
    }

    // [spec:et:sem:op-gelu.torch.executor.native.gelu-out-fn/test]
    #[test]
    fn op_gelu_out_test_mismatched_output_dtype_dies() {
        let tf_float = TensorFactory::<f32>::new();
        let tf_double = TensorFactory::<f64>::new();
        let sizes = vec![2, 2];

        let a = tf_float.ones_default(sizes.clone());
        let out = tf_double.zeros_default(sizes);

        let mut ctx = context();
        et_expect_kernel_failure!(ctx, gelu_out(&mut ctx, &a, "none", &out));
    }

    // [spec:et:sem:op-gelu.torch.executor.native.gelu-out-fn/test]
    // also verifies check_gelu_args: the approximate-format branch rejects any
    // value other than "tanh"/"none".
    // [spec:et:sem:activation-ops-util.torch.executor.check-gelu-args-fn/test]
    #[test]
    fn op_gelu_out_test_invalid_appx_string_dies() {
        let tf = TensorFactory::<f32>::new();
        let a = tf.ones_default(vec![4]);
        let out = tf.zeros_default(vec![4]);

        let mut ctx = context();
        et_expect_kernel_failure!(ctx, gelu_out(&mut ctx, &a, "foo", &out));
    }

    // [spec:et:sem:op-gelu.torch.executor.native.gelu-out-fn/test]
    #[test]
    fn op_gelu_out_test_simple_generated_case() {
        let tf = TensorFactory::<f32>::new();

        let x = tf.make_default(vec![10, 10], vec![1.0f32; 100]);
        let expected_result = tf.make_default(vec![10, 10], vec![0.8411920070648193f32; 100]);

        let out = tf.zeros_default(vec![10, 10]);
        let mut ctx = context();
        let _ret = gelu_out(&mut ctx, &x, "tanh", &out);
        assert_tensor_close!(out, expected_result);
    }

    // [spec:et:sem:op-gelu.torch.executor.native.gelu-out-fn/test]
    #[test]
    fn op_gelu_out_test_dynamic_shape_upper_bound_same_as_expected() {
        let tf = TensorFactory::<f32>::new();

        let x = tf.make_default(
            vec![3, 2],
            vec![
                0.9769402146339417,
                0.4728269577026367,
                0.04416435956954956,
                0.7145527601242065,
                0.7109619975090027,
                0.36388522386550903,
            ],
        );
        let expected_result = tf.make_default(
            vec![3, 2],
            vec![
                0.8162848949432373,
                0.3223743438720703,
                0.022860059514641762,
                0.5448282957077026,
                0.5413010716438293,
                0.23361928761005402,
            ],
        );

        let out = tf.zeros(vec![3, 2], TensorShapeDynamism::DYNAMIC_BOUND);
        let mut ctx = context();
        let _ret = gelu_out(&mut ctx, &x, "tanh", &out);
        assert_tensor_close!(out, expected_result);
    }

    // DISABLED: Dynamic shape not supported
    // [spec:et:sem:op-gelu.torch.executor.native.gelu-out-fn/test]
    #[test]
    #[ignore]
    fn op_gelu_out_test_disabled_dynamic_shape_upper_bound_larger_than_expected() {
        let tf = TensorFactory::<f32>::new();

        let x = tf.make_default(
            vec![3, 2],
            vec![
                0.9769402146339417,
                0.4728269577026367,
                0.04416435956954956,
                0.7145527601242065,
                0.7109619975090027,
                0.36388522386550903,
            ],
        );
        let expected_result = tf.make_default(
            vec![3, 2],
            vec![
                0.8162848949432373,
                0.3223743438720703,
                0.022860059514641762,
                0.5448282957077026,
                0.5413010716438293,
                0.23361928761005402,
            ],
        );

        let out = tf.zeros(vec![10, 10], TensorShapeDynamism::DYNAMIC_BOUND);
        let mut ctx = context();
        let _ret = gelu_out(&mut ctx, &x, "tanh", &out);
        assert_tensor_close!(out, expected_result);
    }

    // DISABLED: Dynamic shape not supported
    // [spec:et:sem:op-gelu.torch.executor.native.gelu-out-fn/test]
    #[test]
    #[ignore]
    fn op_gelu_out_test_disabled_dynamic_shape_unbound() {
        let tf = TensorFactory::<f32>::new();

        let x = tf.make_default(
            vec![3, 2],
            vec![
                0.9769402146339417,
                0.4728269577026367,
                0.04416435956954956,
                0.7145527601242065,
                0.7109619975090027,
                0.36388522386550903,
            ],
        );
        let expected_result = tf.make_default(
            vec![3, 2],
            vec![
                0.8162848949432373,
                0.3223743438720703,
                0.022860059514641762,
                0.5448282957077026,
                0.5413010716438293,
                0.23361928761005402,
            ],
        );

        let out = tf.zeros(vec![1, 1], TensorShapeDynamism::DYNAMIC_UNBOUND);
        let mut ctx = context();
        let _ret = gelu_out(&mut ctx, &x, "tanh", &out);
        assert_tensor_close!(out, expected_result);
    }
}
