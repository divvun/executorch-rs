//! Literal port of kernels/optimized/cpu/op_elu.cpp.

use crate::kernels::portable::cpu::scalar_utils::scalar_to;
use crate::runtime::core::error::Error;
use crate::runtime::core::exec_aten::util::tensor_util::{
    resize_tensor, tensor_is_floating_type, tensors_have_same_dim_order2, tensors_have_same_dtype2,
};
use crate::runtime::core::portable_type::scalar::Scalar;
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::core::portable_type::{BFloat16, Half};
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;
use crate::runtime::kernel::thread_parallel_interface::{internal::GRAIN_SIZE, parallel_for};

// PORT-NOTE: `using MathT = std::conditional_t<
// c10::is_reduced_floating_point_v<CTYPE>, float, CTYPE>` selects `float` for
// Half/BFloat16 and CTYPE itself for Float/Double. Rust has no `conditional_t`;
// the per-CTYPE `MathT` and its math are modeled by the `EluMath` trait — one
// impl per CTYPE in the FLOATHBF16 dispatch set — mirroring
// kernels/portable/cpu/op_elu.rs. `at::native::get_scalar_elu_elementwise_func`
// and `get_vectorized_elu_elementwise_func` both close over
// (negcoef = math_alpha * math_scale, math_scale, math_input_scale) and compute
// `x <= 0 ? expm1(x * input_scale) * negcoef : x * scale`.
trait EluMath: Copy {
    type MathT: Copy;
    fn scalar_to_math(s: &Scalar) -> Self::MathT;
    fn mul_math(a: Self::MathT, b: Self::MathT) -> Self::MathT;
    fn compute(
        self,
        negcoef: Self::MathT,
        math_scale: Self::MathT,
        math_input_scale: Self::MathT,
    ) -> Self;
}

macro_rules! impl_elu_math_direct {
    ($t:ty) => {
        impl EluMath for $t {
            type MathT = $t;
            fn scalar_to_math(s: &Scalar) -> Self::MathT {
                scalar_to::<$t>(s)
            }
            fn mul_math(a: Self::MathT, b: Self::MathT) -> Self::MathT {
                a * b
            }
            fn compute(
                self,
                negcoef: Self::MathT,
                math_scale: Self::MathT,
                math_input_scale: Self::MathT,
            ) -> Self {
                let x = self;
                if x <= 0 as $t {
                    (x * math_input_scale).exp_m1() * negcoef
                } else {
                    x * math_scale
                }
            }
        }
    };
}
impl_elu_math_direct!(f32);
impl_elu_math_direct!(f64);

macro_rules! impl_elu_math_reduced {
    ($t:ty) => {
        impl EluMath for $t {
            type MathT = f32;
            fn scalar_to_math(s: &Scalar) -> Self::MathT {
                scalar_to::<f32>(s)
            }
            fn mul_math(a: Self::MathT, b: Self::MathT) -> Self::MathT {
                a * b
            }
            fn compute(
                self,
                negcoef: Self::MathT,
                math_scale: Self::MathT,
                math_input_scale: Self::MathT,
            ) -> Self {
                let x = self.to_f32();
                let r = if x <= 0f32 {
                    (x * math_input_scale).exp_m1() * negcoef
                } else {
                    x * math_scale
                };
                <$t>::from_f32(r)
            }
        }
    };
}
impl_elu_math_reduced!(Half);
impl_elu_math_reduced!(BFloat16);

// DEVIATION: `at::vec::Vectorized<CTYPE>` collapses to the scalar element type,
// so `Vec::size()` is 1: the scalar prologue and epilogue are empty and the
// "vectorized" loop runs the scalar func over the whole [begin, end) chunk. The
// prologue/epilogue/main structure and the parallel_for bookkeeping are kept
// literal with Vec::size() == 1.
// [spec:et:def:op-elu.torch.executor.native.elu-fn]
// [spec:et:sem:op-elu.torch.executor.native.elu-fn]
fn elu<CTYPE>(
    _context: &mut KernelRuntimeContext,
    input: &Tensor,
    alpha: &Scalar,
    scale: &Scalar,
    input_scale: &Scalar,
    out: &Tensor,
) where
    CTYPE: Copy + EluMath,
{
    let in_data: *const CTYPE = input.const_data_ptr::<CTYPE>();
    let out_data: *mut CTYPE = out.mutable_data_ptr::<CTYPE>();
    let math_alpha = <CTYPE as EluMath>::scalar_to_math(alpha);
    let math_scale = <CTYPE as EluMath>::scalar_to_math(scale);
    let math_input_scale = <CTYPE as EluMath>::scalar_to_math(input_scale);
    let negcoef = <CTYPE as EluMath>::mul_math(math_alpha, math_scale);

    // DEVIATION: Vec::size() == 1.
    const VEC_SIZE: i64 = 1;

    parallel_for(
        0,
        out.numel() as i64,
        GRAIN_SIZE,
        &|begin: i64, end: i64| {
            let vectorized_begin = begin + (VEC_SIZE - begin % VEC_SIZE) % VEC_SIZE;
            let vectorized_end = end - (end % VEC_SIZE);
            // Scalar prologue.
            for idx in begin..vectorized_begin {
                unsafe {
                    *out_data.add(idx as usize) =
                        (*in_data.add(idx as usize)).compute(negcoef, math_scale, math_input_scale);
                }
            }

            // Main vectorized loop.
            let mut idx = vectorized_begin;
            while idx < vectorized_end {
                unsafe {
                    *out_data.add(idx as usize) =
                        (*in_data.add(idx as usize)).compute(negcoef, math_scale, math_input_scale);
                }
                idx += VEC_SIZE;
            }

            // Scalar epilogue.
            for idx in vectorized_end..end {
                unsafe {
                    *out_data.add(idx as usize) =
                        (*in_data.add(idx as usize)).compute(negcoef, math_scale, math_input_scale);
                }
            }
        },
    );
}

// [spec:et:def:op-elu.torch.executor.native.opt-elu-out-fn]
// [spec:et:sem:op-elu.torch.executor.native.opt-elu-out-fn]
pub fn opt_elu_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    in_: &Tensor,
    alpha: &Scalar,
    scale: &Scalar,
    input_scale: &Scalar,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    crate::et_kernel_check!(
        ctx,
        tensors_have_same_dtype2(in_, out),
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

    crate::et_kernel_check!(ctx, tensor_is_floating_type(in_), InvalidArgument, out);

    crate::et_kernel_check!(
        ctx,
        tensors_have_same_dtype2(in_, out),
        InvalidArgument,
        out
    );

    crate::et_switch_floathbf16_types!(in_.scalar_type(), ctx, "elu.out", CTYPE, {
        elu::<CTYPE>(ctx, in_, alpha, scale, input_scale, out);
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

    fn context() -> KernelRuntimeContext<'static> {
        crate::runtime::platform::platform::pal_init();
        KernelRuntimeContext::new(
            crate::extension::module::module::null_event_tracer(),
            core::ptr::null_mut::<MemoryAllocator>() as *mut dyn MemoryAllocatorBase,
        )
    }

    fn op_elu_out<'a, 'b>(
        self_: &Tensor,
        alpha: &Scalar,
        scale: &Scalar,
        input_scale: &Scalar,
        out: &'a Tensor<'b>,
    ) -> &'a Tensor<'b> {
        let mut ctx = context();
        opt_elu_out(&mut ctx, self_, alpha, scale, input_scale, out)
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

    // op_elu_test.cpp test_elu_execution<DTYPE>.
    fn test_elu_execution<T>()
    where
        T: CppTypeToScalarType + FactoryValue + FromF64 + EluMath,
    {
        let tf = TensorFactory::<T>::new();
        let sizes = vec![3, 2];

        let in_ = tf.make_default(
            sizes.clone(),
            vec![
                T::from_f64(-0.125),
                T::from_f64(-0.25),
                T::from_f64(-1.0),
                T::from_f64(0.0),
                T::from_f64(1.25),
                T::from_f64(100.0),
            ],
        );
        let out = tf.zeros_default(sizes.clone());

        // Run full elu.
        op_elu_out(
            &in_,
            &Scalar::from_double(1.25),
            &Scalar::from_i64(1),
            &Scalar::from_i64(1),
            &out,
        );

        assert_tensor_close!(
            out,
            tf.make_default(
                sizes,
                vec![
                    T::from_f64(-0.146879),
                    T::from_f64(-0.276499),
                    T::from_f64(-0.790151),
                    T::from_f64(0.0),
                    T::from_f64(1.25),
                    T::from_f64(100.0),
                ]
            )
        );
    }

    // op_elu_test.cpp test_integer_elu_dies<DTYPE>.
    fn test_integer_elu_dies<T>()
    where
        T: CppTypeToScalarType + FactoryValue,
    {
        let tf = TensorFactory::<T>::new();
        let in_ = tf.ones_default(vec![1]);
        let out = tf.ones_default(vec![1]);

        let mut ctx = context();
        opt_elu_out(
            &mut ctx,
            &in_,
            &Scalar::from_i64(1),
            &Scalar::from_i64(1),
            &Scalar::from_i64(1),
            &out,
        );
        assert_ne!(ctx.failure_state(), Error::Ok);
    }

    // [spec:et:sem:op-elu.torch.executor.native.opt-elu-out-fn/test]
    // [spec:et:sem:op-elu.torch.executor.native.elu-fn/test]
    #[test]
    fn op_elu_test_basic() {
        test_elu_execution::<f32>();
        test_elu_execution::<f64>();
        test_elu_execution::<Half>();
        test_elu_execution::<BFloat16>();
    }

    // [spec:et:sem:op-elu.torch.executor.native.opt-elu-out-fn/test]
    #[test]
    fn op_elu_test_unhandled_dtype_dies() {
        test_integer_elu_dies::<u8>();
        test_integer_elu_dies::<i8>();
        test_integer_elu_dies::<i16>();
        test_integer_elu_dies::<i32>();
        test_integer_elu_dies::<i64>();
    }

    // [spec:et:sem:op-elu.torch.executor.native.opt-elu-out-fn/test]
    #[test]
    fn op_elu_test_mismatched_output_dtype_dies() {
        let tf_float = TensorFactory::<f32>::new();
        let tf_double = TensorFactory::<f64>::new();
        let sizes = vec![2, 2];

        let a = tf_float.ones_default(sizes.clone());
        let out = tf_double.zeros_default(sizes);

        let mut ctx = context();
        opt_elu_out(
            &mut ctx,
            &a,
            &Scalar::from_i64(1),
            &Scalar::from_i64(1),
            &Scalar::from_i64(1),
            &out,
        );
        assert_ne!(ctx.failure_state(), Error::Ok);
    }

    // [spec:et:sem:op-elu.torch.executor.native.opt-elu-out-fn/test]
    // [spec:et:sem:op-elu.torch.executor.native.elu-fn/test]
    #[test]
    fn op_elu_test_mixed_scalar_types() {
        let tf_float = TensorFactory::<f32>::new();
        let sizes = vec![2, 2];

        let in_ = tf_float.ones_default(sizes.clone());
        let out = tf_float.zeros_default(sizes.clone());

        op_elu_out(
            &in_,
            &Scalar::from_bool(true),
            &Scalar::from_double(1.0),
            &Scalar::from_double(1.0),
            &out,
        );
        assert_tensor_close!(out, tf_float.ones_default(sizes.clone()));

        op_elu_out(
            &in_,
            &Scalar::from_bool(false),
            &Scalar::from_bool(true),
            &Scalar::from_i64(3),
            &out,
        );
        assert_tensor_close!(out, tf_float.ones_default(sizes));
    }
}
