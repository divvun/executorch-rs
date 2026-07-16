//! Literal port of kernels/optimized/cpu/op_gelu.cpp.

use crate::kernels::portable::cpu::util::activation_ops_util::check_gelu_args;
use crate::runtime::core::error::Error;
use crate::runtime::core::exec_aten::util::tensor_util::resize_tensor;
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::core::portable_type::{BFloat16, Half};
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

// math_constants.h
const M_SQRT2: f64 = 1.41421356237309504880; // sqrt(2)
const M_2_SQRTPI: f64 = 1.12837916709551257390; // 2/sqrt(pi)
const M_SQRT1_2: f64 = 0.70710678118654752440; // 1/sqrt(2)

// DEVIATION: the C++ splits each loop into an `at::vec::Vectorized<CTYPE>`
// prefix calling `at::native::vectorized_gelu[_approximated_with_tanh]` and a
// scalar tail calling `at::native::scalar_gelu[_approximated_with_tanh]`. Per
// PORTING.md the SIMD lane type collapses to the scalar element type, so the
// whole range runs the scalar ATen gelu. `GeluElem` reproduces
// `scalar_gelu_approximated_with_tanh` (gelu_tanh) and `scalar_gelu`
// (gelu_none), one impl per CTYPE in the FLOATHBF16 dispatch set. The math and
// per-type promotions mirror kernels/portable/cpu/op_gelu.rs (same ATen
// Gelu.h source), so the two kernels agree bit-for-bit.
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
// matching `std::erf` (same seam as kernels/portable/cpu/op_gelu.rs).
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
                let k_beta: $t = <$t>::from_f64(M_SQRT2 * M_2_SQRTPI * 0.5);
                let k_kappa: $t = <$t>::from_f32(0.044715f32);

                let mul = |a: $t, b: $t| <$t>::from_f32(a.to_f32() * b.to_f32());
                let add = |a: $t, b: $t| <$t>::from_f32(a.to_f32() + b.to_f32());

                let x_cubed: $t = mul(mul(x, x), x);
                let inner: $t = mul(k_beta, add(x, mul(k_kappa, x_cubed)));
                let tanh_f32: f32 = inner.to_f32().tanh();
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
                let arg: $t = <$t>::from_f32(x.to_f32() * <$t>::from_f64(M_SQRT1_2).to_f32());
                let erf_f32: f32 = arg.to_f32().erf_impl();
                <$t>::from_f64(0.5 * (x.to_f64()) * (1.0 + (erf_f32 as f64)))
            }
        }
    };
}
impl_gelu_elem_reduced!(Half);
impl_gelu_elem_reduced!(BFloat16);

// [spec:et:def:op-gelu.torch.executor.native.gelu-fn]
// [spec:et:sem:op-gelu.torch.executor.native.gelu-fn]
fn gelu<CTYPE>(
    context: &mut KernelRuntimeContext,
    input: &Tensor,
    approximate: &str,
    output: &Tensor,
) where
    CTYPE: Copy + GeluElem,
{
    let in_data: *const CTYPE = input.const_data_ptr::<CTYPE>();
    let out_data: *mut CTYPE = output.mutable_data_ptr::<CTYPE>();
    let lim: usize = input.numel() as usize;

    if approximate == "tanh" {
        // DEVIATION: Vec::size() collapses to 1; the whole range is the scalar
        // tail loop applying scalar_gelu_approximated_with_tanh.
        for i in 0..lim {
            unsafe {
                *out_data.add(i) = <CTYPE as GeluElem>::gelu_tanh(*in_data.add(i));
            }
        }
    } else if approximate == "none" {
        // DEVIATION: Vec::size() collapses to 1; the whole range is the scalar
        // tail loop applying scalar_gelu.
        for i in 0..lim {
            unsafe {
                *out_data.add(i) = <CTYPE as GeluElem>::gelu_none(*in_data.add(i));
            }
        }
    } else {
        // PORT-NOTE: `et_kernel_check_msg!` keeps only the leading format
        // literal (drops the C++ `%.*s`/`approximate` args), so the message is
        // reproduced without the interpolated approximate string. This branch
        // is unreachable in practice: check_gelu_args already rejected any
        // value other than "tanh"/"none".
        crate::et_kernel_check_msg!(
            context,
            false,
            InvalidArgument,
            (),
            "Invalid approximation format for gelu"
        );
    }
}

// PORT-NOTE: `(void)context;` dropped. `Tensor& out` / returned `Tensor&`
// become `&'a Tensor` (interior mutation through `*mut TensorImpl`).
// [spec:et:def:op-gelu.torch.executor.native.opt-gelu-out-fn]
// [spec:et:sem:op-gelu.torch.executor.native.opt-gelu-out-fn]
pub fn opt_gelu_out<'a, 'b>(
    context: &mut KernelRuntimeContext,
    input: &Tensor,
    approximate: &str,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    // (void)context;
    crate::et_kernel_check!(
        context,
        check_gelu_args(input, approximate, out),
        InvalidArgument,
        out
    );

    crate::et_kernel_check!(
        context,
        resize_tensor(out, input.sizes()) == Error::Ok,
        InvalidArgument,
        out
    );

    crate::et_switch_floathbf16_types!(input.scalar_type(), context, "gelu.out", CTYPE, {
        gelu::<CTYPE>(context, input, approximate, out);
    });

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assert_tensor_close;
    use crate::runtime::core::exec_aten::testing_util::tensor_factory::{
        FactoryValue, TensorFactory,
    };
    use crate::runtime::core::exec_aten::util::scalar_type_util::CppTypeToScalarType;
    use crate::runtime::core::memory_allocator::{MemoryAllocator, MemoryAllocatorBase};
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
                crate::runtime::core::error::Error::Ok,
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

    // op_gelu_test.cpp test_gelu_execution<DTYPE> (both approximation modes).
    fn test_gelu_execution<T>()
    where
        T: CppTypeToScalarType + FactoryValue + FromF64 + GeluElem,
    {
        let tf = TensorFactory::<T>::new();
        let sizes = vec![3, 2];

        let in_ = tf.make_default(
            sizes.clone(),
            d::<T>(&[-0.4775, 0.2948, -0.3984, 1.8690, -0.4048, -0.4848]),
        );
        let out = tf.zeros_default(sizes.clone());

        let mut ctx = context();
        opt_gelu_out(&mut ctx, &in_, "none", &out);
        assert_tensor_close!(
            out,
            tf.make_default(
                sizes.clone(),
                d::<T>(&[-0.15113, 0.181575, -0.137515, 1.81141, -0.13877, -0.152183])
            )
        );

        opt_gelu_out(&mut ctx, &in_, "tanh", &out);
        assert_tensor_close!(
            out,
            tf.make_default(
                sizes,
                d::<T>(&[-0.151145, 0.181573, -0.137522, 1.8114, -0.138778, -0.152199])
            )
        );
    }

    // [spec:et:sem:op-gelu.torch.executor.native.opt-gelu-out-fn/test]
    // [spec:et:sem:op-gelu.torch.executor.native.gelu-fn/test]
    #[test]
    fn op_gelu_test_float_tensors() {
        test_gelu_execution::<f32>();
    }

    // [spec:et:sem:op-gelu.torch.executor.native.opt-gelu-out-fn/test]
    // [spec:et:sem:op-gelu.torch.executor.native.gelu-fn/test]
    #[test]
    fn op_gelu_test_half_tensors() {
        test_gelu_execution::<Half>();
    }

    // [spec:et:sem:op-gelu.torch.executor.native.opt-gelu-out-fn/test]
    // [spec:et:sem:op-gelu.torch.executor.native.gelu-fn/test]
    #[test]
    fn op_gelu_test_bfloat16_tensors() {
        test_gelu_execution::<BFloat16>();
    }

    // PORT-NOTE: the C++ optimized suite skips DoubleTensors via the
    // `op_gelu_dtype_double: false` supported-features entry, but the kernel's
    // FLOATHBF16 dispatch handles Double; the Rust port runs it.
    // [spec:et:sem:op-gelu.torch.executor.native.opt-gelu-out-fn/test]
    // [spec:et:sem:op-gelu.torch.executor.native.gelu-fn/test]
    #[test]
    fn op_gelu_test_double_tensors() {
        test_gelu_execution::<f64>();
    }

    // [spec:et:sem:op-gelu.torch.executor.native.opt-gelu-out-fn/test]
    #[test]
    fn op_gelu_test_unhandled_dtype_dies() {
        // gelu() doesn't handle Bool.
        let tf = TensorFactory::<bool>::new();
        let sizes = vec![2, 2];
        let a = tf.make_default(sizes.clone(), vec![false, true, false, true]);
        let out = tf.zeros_default(sizes);

        let mut ctx = context();
        et_expect_kernel_failure!(ctx, opt_gelu_out(&mut ctx, &a, "none", &out));
    }

    // [spec:et:sem:op-gelu.torch.executor.native.opt-gelu-out-fn/test]
    #[test]
    fn op_gelu_test_mismatched_output_dtype_dies() {
        let tf_float = TensorFactory::<f32>::new();
        let tf_double = TensorFactory::<f64>::new();
        let sizes = vec![2, 2];

        let a = tf_float.ones_default(sizes.clone());
        let out = tf_double.zeros_default(sizes);

        let mut ctx = context();
        et_expect_kernel_failure!(ctx, opt_gelu_out(&mut ctx, &a, "none", &out));
    }

    // [spec:et:sem:op-gelu.torch.executor.native.opt-gelu-out-fn/test]
    #[test]
    fn op_gelu_test_invalid_appx_string_dies() {
        let tf = TensorFactory::<f32>::new();
        let a = tf.ones_default(vec![4]);
        let out = tf.zeros_default(vec![4]);

        let mut ctx = context();
        et_expect_kernel_failure!(ctx, opt_gelu_out(&mut ctx, &a, "foo", &out));
    }

    // [spec:et:sem:op-gelu.torch.executor.native.opt-gelu-out-fn/test]
    // [spec:et:sem:op-gelu.torch.executor.native.gelu-fn/test]
    #[test]
    fn op_gelu_test_simple_generated_case() {
        let tf = TensorFactory::<f32>::new();

        let x = tf.make_default(vec![10, 10], vec![1.0f32; 100]);
        let expected_result = tf.make_default(vec![10, 10], vec![0.8411920070648193f32; 100]);
        let out = tf.zeros_default(vec![10, 10]);

        let mut ctx = context();
        opt_gelu_out(&mut ctx, &x, "tanh", &out);
        assert_tensor_close!(out, expected_result);
    }

    // [spec:et:sem:op-gelu.torch.executor.native.opt-gelu-out-fn/test]
    #[test]
    fn op_gelu_test_dynamic_shape_upper_bound_same_as_expected() {
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
        opt_gelu_out(&mut ctx, &x, "tanh", &out);
        assert_tensor_close!(out, expected_result);
    }
}
