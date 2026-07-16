//! Literal port of kernels/portable/cpu/op_nonzero.cpp.

use crate::kernels::portable::cpu::util::index_util::check_nonzero_args;
use crate::runtime::core::array_ref::ArrayRef;
use crate::runtime::core::error::Error;
use crate::runtime::core::exec_aten::util::tensor_util::{
    K_TENSOR_DIMENSION_LIMIT, resize_tensor_same_type,
};
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::core::portable_type::tensor_impl::{SizesType, ssize_t};
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

// PORT-NOTE: C++ `Tensor& out` / returned `Tensor&` become `&'a Tensor` — the
// non-owning handle mutates its impl through the raw pointer, matching the
// established interior-mutation pattern.

// [spec:et:def:op-nonzero.torch.executor.native.increment-index-fn]
// [spec:et:sem:op-nonzero.torch.executor.native.increment-index-fn]
///
/// # Safety
/// `index` must point to at least `sizes.size()` valid `usize` elements.
unsafe fn increment_index(index: *mut usize, sizes: ArrayRef<SizesType>) {
    let mut i: ssize_t = sizes.size() as ssize_t - 1;
    while i >= 0 {
        unsafe {
            *index.add(i as usize) += 1;
            if *index.add(i as usize) as ssize_t == *sizes.at(i as usize) as ssize_t {
                *index.add(i as usize) = 0;
            } else {
                return;
            }
        }
        i -= 1;
    }
}

/// Two pass algorithm where we first count the number of non zeros, then resize
/// out to the appropriate size, and then loop again and properly write into out
// [spec:et:def:op-nonzero.torch.executor.native.nonzero-fn]
// [spec:et:sem:op-nonzero.torch.executor.native.nonzero-fn]
fn nonzero<CTYPE: PartialEq + Zero + Copy>(
    ctx: &mut KernelRuntimeContext,
    input: &Tensor,
    output: &Tensor,
) {
    let in_data: *const CTYPE = input.const_data_ptr::<CTYPE>();
    let lim: usize = input.numel() as usize;
    let mut num_nonzero: i32 = 0;

    // Count number of non zeros
    for i in 0..lim {
        if unsafe { *in_data.add(i) } != CTYPE::ZERO {
            num_nonzero += 1;
        }
    }

    // resize out
    let out_shape: [SizesType; 2] = [num_nonzero as SizesType, input.dim() as SizesType];
    crate::et_kernel_check!(
        ctx,
        resize_tensor_same_type(output, ArrayRef::from_raw_parts(out_shape.as_ptr(), 2))
            == Error::Ok,
        InvalidArgument,
    );

    let mut index: [usize; K_TENSOR_DIMENSION_LIMIT] = [0; K_TENSOR_DIMENSION_LIMIT];

    let out_data: *mut i64 = output.mutable_data_ptr::<i64>();
    let mut out_idx: usize = 0;

    // Loop again and this time write the proper indices into out
    for i in 0..lim {
        if unsafe { *in_data.add(i) } != CTYPE::ZERO {
            for j in 0..input.dim() {
                unsafe {
                    *out_data.add(out_idx) = index[j as usize] as i64;
                }
                out_idx += 1;
            }
        }
        unsafe {
            increment_index(index.as_mut_ptr(), input.sizes());
        }
    }
}

/// Determines the non zero indices of input.
/// Out is a 2-D tensor where every row is a non zero index of the input.
// [spec:et:def:op-nonzero.torch.executor.native.nonzero-out-fn]
// [spec:et:sem:op-nonzero.torch.executor.native.nonzero-out-fn]
pub fn nonzero_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    in_: &Tensor,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    // (void)ctx;

    crate::et_kernel_check!(ctx, check_nonzero_args(in_, out), InvalidArgument, out);

    crate::et_switch_realhbbf16_types!(in_.scalar_type(), ctx, "nonzero.out", CTYPE, {
        nonzero::<CTYPE>(ctx, in_, out);
    });

    out
}

// PORT-NOTE: the C++ `in_data[i] != 0` compares each ctype against the literal
// `0`. Rust has no single "zero literal" over the heterogeneous REALHBBF16 set,
// so a local `Zero` trait supplies `ZERO` per ctype (bool: `false`;
// Half/BFloat16 via their zero bit pattern). Float `-0.0 == 0.0` and `NaN != 0`
// follow Rust `PartialEq`, matching C++ `!=` IEEE-754 semantics.
trait Zero {
    const ZERO: Self;
}
macro_rules! impl_zero {
    ($($t:ty => $z:expr),* $(,)?) => {$(
        impl Zero for $t {
            const ZERO: Self = $z;
        }
    )*};
}
impl_zero!(u8 => 0, i8 => 0, i16 => 0, i32 => 0, i64 => 0, f32 => 0.0, f64 => 0.0, bool => false);
impl Zero for crate::runtime::core::portable_type::Half {
    const ZERO: Self = crate::runtime::core::portable_type::Half::ZERO;
}
impl Zero for crate::runtime::core::portable_type::BFloat16 {
    const ZERO: Self = crate::runtime::core::portable_type::BFloat16::ZERO;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assert_tensor_eq;
    use crate::runtime::core::exec_aten::testing_util::tensor_factory::{
        FactoryValue, TensorFactory,
    };
    use crate::runtime::core::exec_aten::util::scalar_type_util::CppTypeToScalarType;
    use crate::runtime::core::memory_allocator::{MemoryAllocator, MemoryAllocatorBase};
    use crate::runtime::core::portable_type::{BFloat16, Half};
    use crate::runtime::core::tensor_shape_dynamism::TensorShapeDynamism;

    fn context() -> KernelRuntimeContext<'static> {
        crate::runtime::platform::platform::pal_init();
        KernelRuntimeContext::new(
            crate::extension::module::module::null_event_tracer(),
            core::ptr::null_mut::<MemoryAllocator>() as *mut dyn MemoryAllocatorBase,
        )
    }

    // PORT-NOTE: `CTYPE(int)` bridge for building the input data in the REALHBBF16
    // factory element types used by test_dtype.
    trait FromI32Data: Copy {
        fn from_i32(v: i32) -> Self;
    }
    macro_rules! impl_from_i32_data_num {
        ($($t:ty),*) => {$(impl FromI32Data for $t { fn from_i32(v: i32) -> Self { v as $t } })*};
    }
    impl_from_i32_data_num!(u8, i8, i16, i32, i64, f32, f64);
    impl FromI32Data for bool {
        fn from_i32(v: i32) -> Self {
            v != 0
        }
    }
    impl FromI32Data for Half {
        fn from_i32(v: i32) -> Self {
            Half::from_f32(v as f32)
        }
    }
    impl FromI32Data for BFloat16 {
        fn from_i32(v: i32) -> Self {
            BFloat16::from_f32(v as f32)
        }
    }

    fn test_dtype<T>()
    where
        T: CppTypeToScalarType + FactoryValue + FromI32Data,
    {
        let tf_input = TensorFactory::<T>::new();
        let tf_long = TensorFactory::<i64>::new();
        let d = |v: &[i32]| -> Vec<T> { v.iter().map(|&x| T::from_i32(x)).collect() };

        let a = tf_input.make_default(vec![2, 2], d(&[2, 0, 2, 4]));
        let out = tf_long.zeros_default(vec![3, 2]);

        let mut ctx = context();
        nonzero_out(&mut ctx, &a, &out);
        assert_tensor_eq!(
            out,
            tf_long.make_default(vec![3, 2], vec![0, 0, 1, 0, 1, 1])
        );
    }

    // [spec:et:sem:op-nonzero.torch.executor.native.nonzero-out-fn/test]
    // also verifies check_nonzero_args (valid path: 2D Long out passes; the op
    // aborts and leaves output unwritten if it wrongly returned false).
    // [spec:et:sem:index-util.torch.executor.check-nonzero-args-fn/test]
    // also verifies the nonzero two-pass helper and increment_index: the 2x2 input
    // [[2,0],[2,4]] yields exactly the rows [0,0],[1,0],[1,1], which pins both the
    // count/resize/write logic and the multi-dim row-major index carry.
    // [spec:et:sem:op-nonzero.torch.executor.native.nonzero-fn/test]
    // [spec:et:sem:op-nonzero.torch.executor.native.increment-index-fn/test]
    #[test]
    fn op_nonzero_test_all_dtypes_supported() {
        // ET_FORALL_REALHBBF16_TYPES
        test_dtype::<u8>();
        test_dtype::<i8>();
        test_dtype::<i16>();
        test_dtype::<i32>();
        test_dtype::<i64>();
        test_dtype::<f32>();
        test_dtype::<f64>();
        test_dtype::<bool>();
        test_dtype::<Half>();
        test_dtype::<BFloat16>();
    }

    // PORT-NOTE: gated by `#if !defined(USE_ATEN_LIB)` in the C++; the Rust port is
    // the non-aten branch, so these run.
    // [spec:et:sem:op-nonzero.torch.executor.native.nonzero-out-fn/test]
    #[test]
    fn op_nonzero_test_static_shape_inconsistent_size() {
        let tf_input = TensorFactory::<f32>::new();
        let tf_long = TensorFactory::<i64>::new();
        let a = tf_input.make_default(vec![2, 2], vec![2.0, 0.0, 2.0, 4.0]);
        // If we use static size here (by default), it won't work unless we know the
        // output size.
        let out = tf_long.zeros(vec![4, 2], TensorShapeDynamism::STATIC);

        let mut ctx = context();
        nonzero_out(&mut ctx, &a, &out);
        assert_ne!(ctx.failure_state(), Error::Ok);
    }

    // [spec:et:sem:op-nonzero.torch.executor.native.nonzero-out-fn/test]
    #[test]
    fn op_nonzero_test_dynamic_shape() {
        let tf_input = TensorFactory::<f32>::new();
        let tf_long = TensorFactory::<i64>::new();
        let a = tf_input.make_default(vec![2, 2], vec![2.0, 0.0, 2.0, 4.0]);
        let out = tf_long.zeros(vec![4, 2], TensorShapeDynamism::DYNAMIC_BOUND);

        let mut ctx = context();
        nonzero_out(&mut ctx, &a, &out);
        assert_tensor_eq!(
            out,
            tf_long.make_default(vec![3, 2], vec![0, 0, 1, 0, 1, 1])
        );
    }

    // [spec:et:sem:op-nonzero.torch.executor.native.nonzero-out-fn/test]
    #[test]
    fn op_nonzero_test_dynamic_shape_insufficient_buffer() {
        let tf_input = TensorFactory::<f32>::new();
        let tf_long = TensorFactory::<i64>::new();
        let a = tf_input.make_default(vec![2, 2], vec![2.0, 0.0, 2.0, 4.0]);
        let out = tf_long.zeros(vec![2, 2], TensorShapeDynamism::DYNAMIC_BOUND);

        let mut ctx = context();
        nonzero_out(&mut ctx, &a, &out);
        assert_ne!(ctx.failure_state(), Error::Ok);
    }
}
