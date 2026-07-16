//! Literal port of kernels/portable/cpu/op_native_dropout.cpp.

use crate::kernels::portable::cpu::op_rand::{Mt19937, UniformRealDistribution, random_device};
use crate::kernels::portable::cpu::util::dtype_util::SupportedTensorDtypes;
use crate::kernels::portable::cpu::util::elementwise_util::apply_bitensor_elementwise_fn;
use crate::runtime::core::error::Error;
use crate::runtime::core::exec_aten::util::tensor_util::{
    resize_tensor, tensor_is_bool_type, tensors_have_same_dim_order3, tensors_have_same_dtype2,
};
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::core::portable_type::{BFloat16, Half};
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

// PORT-NOTE: `Tensor& out/mask` and the returned `std::tuple<Tensor&, Tensor&>`
// become `&'a Tensor` handles and a Rust 2-tuple (interior mutation through
// `*mut TensorImpl`). `std::optional<bool> train` maps to `Option<bool>`.
//
// PORT-NOTE (nondeterminism): the C++ seeds `std::mt19937` from
// `std::random_device` every call, so `mask` is nondeterministic and not
// reproducible. The mt19937 engine, the `std::random_device` stand-in, and the
// `std::uniform_real_distribution<double>` (default [0,1) range) are the ones
// ported inline in op_rand.rs (reused here as `pub(crate)` items rather than
// re-porting); `dist(gen)` maps to `dist.sample(&mut gen)`. The exact bit
// sequence of a specific libstdc++/libc++ build is not (and cannot be)
// reproduced given the nondeterministic seed.
//
// PORT-NOTE: the compute closure's `mask_val` is loaded from the bool `mask`
// tensor into `CTYPE_COMPUTE` (a FLOATHBF16 float, 0.0 or 1.0); the C++
// `if (!mask_val)` therefore tests `mask_val == 0`. `DropoutElem` reproduces the
// per-CTYPE zero test and zero literal.

trait DropoutElem: Copy {
    fn is_zero(self) -> bool;
    fn zero() -> Self;
}
macro_rules! impl_dropout_elem_native {
    ($t:ty) => {
        impl DropoutElem for $t {
            fn is_zero(self) -> bool {
                self == 0 as $t
            }
            fn zero() -> Self {
                0 as $t
            }
        }
    };
}
impl_dropout_elem_native!(f32);
impl_dropout_elem_native!(f64);
impl DropoutElem for Half {
    fn is_zero(self) -> bool {
        self.to_f32() == 0.0
    }
    fn zero() -> Self {
        Half::from_f32(0.0)
    }
}
impl DropoutElem for BFloat16 {
    fn is_zero(self) -> bool {
        self.to_f32() == 0.0
    }
    fn zero() -> Self {
        BFloat16::from_f32(0.0)
    }
}

// [spec:et:def:op-native-dropout.torch.executor.native.native-dropout-out-fn]
// [spec:et:sem:op-native-dropout.torch.executor.native.native-dropout-out-fn]
pub fn native_dropout_out<'a, 'b, 'c>(
    ctx: &mut KernelRuntimeContext,
    input: &Tensor,
    prob: f64,
    train: Option<bool>,
    out: &'a Tensor<'b>,
    mask: &'a Tensor<'c>,
) -> (&'a Tensor<'b>, &'a Tensor<'c>) {
    let ret = (out, mask);
    crate::et_kernel_check!(
        ctx,
        tensors_have_same_dtype2(input, out),
        InvalidArgument,
        ret
    );
    crate::et_kernel_check!(
        ctx,
        tensors_have_same_dim_order3(input, out, mask),
        InvalidArgument,
        ret
    );
    crate::et_kernel_check!(
        ctx,
        resize_tensor(out, input.sizes()) == Error::Ok,
        InvalidArgument,
        ret
    );
    crate::et_kernel_check!(
        ctx,
        resize_tensor(mask, input.sizes()) == Error::Ok,
        InvalidArgument,
        ret
    );
    crate::et_kernel_check!(ctx, tensor_is_bool_type(mask), InvalidArgument, ret);
    // PORT-NOTE: the crate `et_kernel_check_msg!` keeps only the leading format
    // literal and drops trailing args, so the C++ `%f`/`prob` interpolation of the
    // offending value is dropped (mirroring the established pattern in op_rand/op_neg).
    crate::et_kernel_check_msg!(
        ctx,
        prob >= 0.0 && prob <= 1.0,
        InvalidArgument,
        ret,
        "dropout probability has to be between 0 and 1"
    );

    let op_name = "native_dropout.out";
    if (train.is_none() || train.unwrap()) && prob != 0.0 {
        {
            let mut gen_ = Mt19937::new(random_device());
            let dist = UniformRealDistribution::new(0.0, 1.0);
            let mask_data_ptr: *mut bool = mask.mutable_data_ptr::<bool>();
            for ii in 0..mask.numel() {
                unsafe {
                    *mask_data_ptr.add(ii as usize) = dist.sample(&mut gen_) >= prob;
                }
            }
        }
        crate::et_switch_floathbf16_types!(input.scalar_type(), ctx, op_name, CTYPE_COMPUTE, {
            apply_bitensor_elementwise_fn::<CTYPE_COMPUTE, _>(
                |vals: &[CTYPE_COMPUTE]| -> CTYPE_COMPUTE {
                    let val = vals[0];
                    let mask_val = vals[1];
                    if mask_val.is_zero() {
                        return <CTYPE_COMPUTE as DropoutElem>::zero();
                    }
                    val
                },
                ctx,
                input,
                SupportedTensorDtypes::FLOATHBF16,
                mask,
                // TODO: should really be just BOOL
                SupportedTensorDtypes::BOOL_OR_BYTE,
                out,
                SupportedTensorDtypes::SAME_AS_COMMON,
                false,
            );
        });
    } else if input.numel() > 0 {
        unsafe {
            core::ptr::copy_nonoverlapping(
                input.const_data_ptr::<u8>(),
                out.mutable_data_ptr::<u8>(),
                input.nbytes(),
            );
            core::ptr::write_bytes(mask.mutable_data_ptr::<u8>(), true as u8, mask.nbytes());
        }
    }
    ret
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

    trait FromI32Data: Copy + PartialEq {
        fn from_i32(v: i32) -> Self;
        fn ct_zero() -> Self;
    }
    impl FromI32Data for f32 {
        fn from_i32(v: i32) -> Self {
            v as f32
        }
        fn ct_zero() -> Self {
            0.0
        }
    }
    impl FromI32Data for f64 {
        fn from_i32(v: i32) -> Self {
            v as f64
        }
        fn ct_zero() -> Self {
            0.0
        }
    }
    impl FromI32Data for Half {
        fn from_i32(v: i32) -> Self {
            Half::from_f32(v as f32)
        }
        fn ct_zero() -> Self {
            Half::from_f32(0.0)
        }
    }
    impl FromI32Data for BFloat16 {
        fn from_i32(v: i32) -> Self {
            BFloat16::from_f32(v as f32)
        }
        fn ct_zero() -> Self {
            BFloat16::from_f32(0.0)
        }
    }

    fn test_dropout<T>()
    where
        T: CppTypeToScalarType + FactoryValue + FromI32Data,
    {
        let tf = TensorFactory::<T>::new();
        let tf_bool = TensorFactory::<bool>::new();
        let d = |v: &[i32]| -> Vec<T> { v.iter().map(|&x| T::from_i32(x)).collect() };
        let sizes = vec![3, 2];
        let in_ = tf.make_default(sizes.clone(), d(&[1, 2, 3, 4, 5, 6]));
        let out = tf.zeros_default(sizes.clone());
        let mask = tf_bool.zeros_default(sizes.clone());

        let mask_data: *mut bool = mask.mutable_data_ptr::<bool>();
        let expect_no_drops = |ctx_out: &Tensor| {
            assert_tensor_close!(*ctx_out, in_);
            for ii in 0..mask.numel() {
                unsafe {
                    assert!(*mask_data.add(ii as usize));
                    *mask_data.add(ii as usize) = false;
                }
            }
        };

        let mut ctx = context();
        native_dropout_out(&mut ctx, &in_, 0.0, Some(true), &out, &mask);
        expect_no_drops(&out);

        native_dropout_out(&mut ctx, &in_, 0.0, Some(false), &out, &mask);
        expect_no_drops(&out);

        native_dropout_out(&mut ctx, &in_, 1.0, Some(false), &out, &mask);
        expect_no_drops(&out);

        native_dropout_out(&mut ctx, &in_, 1.0, Some(true), &out, &mask);
        let out_data: *mut T = out.mutable_data_ptr::<T>();
        for ii in 0..out.numel() {
            assert!(unsafe { *out_data.add(ii as usize) } == T::ct_zero());
        }
        for ii in 0..mask.numel() {
            unsafe {
                assert!(!*mask_data.add(ii as usize));
                *mask_data.add(ii as usize) = false;
            }
        }
    }

    // [spec:et:sem:op-native-dropout.torch.executor.native.native-dropout-out-fn/test]
    #[test]
    fn op_native_dropout_test_basic() {
        // ET_FORALL_FLOATHBF16_TYPES
        test_dropout::<f32>();
        test_dropout::<f64>();
        test_dropout::<Half>();
        test_dropout::<BFloat16>();
    }

    // [spec:et:sem:op-native-dropout.torch.executor.native.native-dropout-out-fn/test]
    #[test]
    fn op_native_dropout_test_probability_range_check() {
        let tf_float = TensorFactory::<f32>::new();
        let tf_bool = TensorFactory::<bool>::new();
        let sizes = vec![2, 3];
        let a = tf_float.ones_default(sizes.clone());
        let out = tf_float.zeros_default(sizes.clone());
        let mask = tf_bool.zeros_default(sizes.clone());

        let mut ctx = context();
        native_dropout_out(&mut ctx, &a, -1.0, Some(true), &out, &mask);
        assert_ne!(ctx.failure_state(), Error::Ok);
    }

    // [spec:et:sem:op-native-dropout.torch.executor.native.native-dropout-out-fn/test]
    #[test]
    fn op_native_dropout_test_mask_bool_check() {
        let tf_float = TensorFactory::<f32>::new();
        let tf_byte = TensorFactory::<u8>::new();
        let sizes = vec![2, 3];
        let a = tf_float.ones_default(sizes.clone());
        let out = tf_float.zeros_default(sizes.clone());
        let mask_byte = tf_byte.zeros_default(sizes.clone());
        let mask_float = tf_float.zeros_default(sizes.clone());

        let mut ctx = context();
        native_dropout_out(&mut ctx, &a, 0.5, Some(true), &out, &mask_byte);
        assert_ne!(ctx.failure_state(), Error::Ok);

        let mut ctx = context();
        native_dropout_out(&mut ctx, &a, 0.5, Some(true), &out, &mask_float);
        assert_ne!(ctx.failure_state(), Error::Ok);
    }
}
