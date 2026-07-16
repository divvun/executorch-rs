//! Literal port of kernels/portable/cpu/op_logical_not.cpp.

use crate::kernels::portable::cpu::util::dtype_util::StaticCast;
use crate::kernels::portable::cpu::util::functional_util::apply_unary_map_fn;
use crate::runtime::core::error::Error;
use crate::runtime::core::exec_aten::util::tensor_util::{
    resize_tensor_same_type, tensors_have_same_dim_order2, tensors_have_same_shape2,
};
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

// PORT-NOTE: `Tensor& out` / returned `Tensor&` become `&'a Tensor` — the ported
// `Tensor` handle mutates through an interior `*mut TensorImpl`.

// [spec:et:def:op-logical-not.torch.executor.native.logical-not-out-fn]
// [spec:et:sem:op-logical-not.torch.executor.native.logical-not-out-fn]
pub fn logical_not_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    in_: &Tensor,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    // Resize for dynamic shape
    crate::et_kernel_check_msg!(
        ctx,
        resize_tensor_same_type(out, in_.sizes()) == Error::Ok,
        InvalidArgument,
        out,
        "Failed to resize output tensor."
    );

    crate::et_kernel_check!(
        ctx,
        tensors_have_same_dim_order2(in_, out),
        InvalidArgument,
        out
    );

    crate::et_kernel_check!(
        ctx,
        tensors_have_same_shape2(in_, out),
        InvalidArgument,
        out
    );

    crate::et_switch_realhbbf16_types!(in_.scalar_type(), ctx, "logical_not.out", CTYPE_IN, {
        crate::et_switch_realhbbf16_types!(out.scalar_type(), ctx, "logical_not.out", CTYPE_OUT, {
            apply_unary_map_fn(
                |val_in: CTYPE_IN| -> CTYPE_OUT {
                    <CTYPE_OUT as StaticCast<bool>>::static_cast(
                        !<bool as StaticCast<CTYPE_IN>>::static_cast(val_in),
                    )
                },
                in_.const_data_ptr::<CTYPE_IN>(),
                out.mutable_data_ptr::<CTYPE_OUT>(),
                in_.numel() as i64,
                1,
            );
        });
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
    use crate::runtime::core::portable_type::{BFloat16, Half};

    fn context() -> KernelRuntimeContext<'static> {
        crate::runtime::platform::platform::pal_init();
        KernelRuntimeContext::new(
            crate::extension::module::module::null_event_tracer(),
            core::ptr::null_mut::<MemoryAllocator>() as *mut dyn MemoryAllocatorBase,
        )
    }

    // Element builders mirroring the integer / bool initializer lists coercing to
    // the factory element type.
    trait FromLn: Copy {
        fn from_i32(v: i32) -> Self;
        fn from_bool(v: bool) -> Self;
    }
    macro_rules! impl_from_ln_num {
        ($($t:ty),*) => {$(impl FromLn for $t {
            fn from_i32(v: i32) -> Self { v as $t }
            fn from_bool(v: bool) -> Self { v as i32 as $t }
        })*};
    }
    impl_from_ln_num!(u8, i8, i16, i32, i64, f32, f64);
    impl FromLn for Half {
        fn from_i32(v: i32) -> Self {
            Half::from_f32(v as f32)
        }
        fn from_bool(v: bool) -> Self {
            Half::from_f32(v as i32 as f32)
        }
    }
    impl FromLn for BFloat16 {
        fn from_i32(v: i32) -> Self {
            BFloat16::from_f32(v as f32)
        }
        fn from_bool(v: bool) -> Self {
            BFloat16::from_f32(v as i32 as f32)
        }
    }
    impl FromLn for bool {
        fn from_i32(v: i32) -> Self {
            v != 0
        }
        fn from_bool(v: bool) -> Self {
            v
        }
    }

    fn i<T: FromLn>(v: &[i32]) -> Vec<T> {
        v.iter().map(|&x| T::from_i32(x)).collect()
    }
    fn b<T: FromLn>(v: &[bool]) -> Vec<T> {
        v.iter().map(|&x| T::from_bool(x)).collect()
    }

    fn test_logical_not_out<IN, OUT>()
    where
        IN: CppTypeToScalarType + FactoryValue + FromLn,
        OUT: CppTypeToScalarType + FactoryValue + FromLn,
    {
        let tf_in = TensorFactory::<IN>::new();
        let tf_out = TensorFactory::<OUT>::new();

        let in_ = tf_in.make_default(vec![2, 4], i::<IN>(&[0, 1, 0, 1, 1, 0, 1, 0]));
        let bool_in = tf_in.make_default(
            vec![2, 4],
            b::<IN>(&[false, true, false, true, true, false, true, false]),
        );

        let out = tf_out.zeros_default(vec![2, 4]);
        let bool_out = tf_out.zeros_default(vec![2, 4]);

        let mut ctx = context();
        logical_not_out(&mut ctx, &in_, &out);
        assert_tensor_close!(
            out,
            tf_out.make_default(vec![2, 4], i::<OUT>(&[1, 0, 1, 0, 0, 1, 0, 1]))
        );

        let mut ctx = context();
        logical_not_out(&mut ctx, &bool_in, &out);
        assert_tensor_close!(
            out,
            tf_out.make_default(vec![2, 4], i::<OUT>(&[1, 0, 1, 0, 0, 1, 0, 1]))
        );

        let mut ctx = context();
        logical_not_out(&mut ctx, &in_, &bool_out);
        assert_tensor_close!(
            bool_out,
            tf_out.make_default(
                vec![2, 4],
                b::<OUT>(&[true, false, true, false, false, true, false, true])
            )
        );
    }

    fn test_logical_not_out_float<OUT>()
    where
        OUT: CppTypeToScalarType + FactoryValue + FromLn,
    {
        let tf_float = TensorFactory::<f32>::new();
        let tf_out = TensorFactory::<OUT>::new();

        let in_ = tf_float.make_default(
            vec![1, 4],
            vec![f32::INFINITY, f32::NAN, f32::NEG_INFINITY, 0.0],
        );
        let out = tf_out.zeros_default(vec![1, 4]);

        let mut ctx = context();
        logical_not_out(&mut ctx, &in_, &out);
        assert_tensor_close!(
            out,
            tf_out.make_default(vec![1, 4], i::<OUT>(&[0, 0, 0, 1]))
        );
    }

    // PORT-NOTE: guarded in C++ by `ET_SKIP_IF(is_aten, ...)`; non-ATen build runs.
    // [spec:et:sem:op-logical-not.torch.executor.native.logical-not-out-fn/test]
    #[test]
    fn op_logical_not_out_test_mismatched_dimensions_dies() {
        let tff = TensorFactory::<f32>::new();
        let size = vec![2, 2];

        let in_ = tff.make_default(size, vec![0.0, 0.0, 1.0, 0.0]);
        let out = tff.zeros_default(vec![4, 1]);

        let mut ctx = context();
        logical_not_out(&mut ctx, &in_, &out);
        assert_ne!(ctx.failure_state(), Error::Ok);
    }

    // ET_FORALL_REALHBBF16_TYPES x ET_FORALL_REALHBBF16_TYPES.
    // [spec:et:sem:op-logical-not.torch.executor.native.logical-not-out-fn/test]
    #[test]
    fn op_logical_not_out_test_all_type_passes() {
        fn forall_out<IN>()
        where
            IN: CppTypeToScalarType + FactoryValue + FromLn,
        {
            test_logical_not_out::<IN, u8>();
            test_logical_not_out::<IN, i8>();
            test_logical_not_out::<IN, i16>();
            test_logical_not_out::<IN, i32>();
            test_logical_not_out::<IN, i64>();
            test_logical_not_out::<IN, Half>();
            test_logical_not_out::<IN, BFloat16>();
            test_logical_not_out::<IN, bool>();
            test_logical_not_out::<IN, f32>();
            test_logical_not_out::<IN, f64>();
        }
        forall_out::<u8>();
        forall_out::<i8>();
        forall_out::<i16>();
        forall_out::<i32>();
        forall_out::<i64>();
        forall_out::<Half>();
        forall_out::<BFloat16>();
        forall_out::<bool>();
        forall_out::<f32>();
        forall_out::<f64>();
    }

    // ET_FORALL_FLOAT_TYPES: Float, Double.
    // [spec:et:sem:op-logical-not.torch.executor.native.logical-not-out-fn/test]
    #[test]
    fn op_logical_not_out_test_float_specific_test() {
        test_logical_not_out_float::<f32>();
        test_logical_not_out_float::<f64>();
    }
}
