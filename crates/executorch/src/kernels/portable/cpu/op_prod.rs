//! Literal port of kernels/portable/cpu/op_prod.cpp.

use crate::kernels::portable::cpu::util::dtype_util::internal::StaticCast;
#[cfg(not(feature = "aten"))]
use crate::kernels::portable::cpu::util::reduce_util::{
    check_prod_out_args, check_reduction_args_single_dim,
};
use crate::kernels::portable::cpu::util::reduce_util::{
    map_reduce_over_dim, parallel_for_each_reduce_over_dim_output_index, resize_reduction_out_dim,
};
use crate::runtime::core::array_ref::ArrayRef;
use crate::runtime::core::error::Error;
use crate::runtime::core::exec_aten::util::tensor_util::resize_tensor_same_type;
use crate::runtime::core::portable_type::scalar_type::ScalarType;
use crate::runtime::core::portable_type::tensor::{Tensor, TensorSizesType};
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

// PORT-NOTE: C++ `Tensor& out` / returned `Tensor&` become `&'a Tensor` (interior
// mutation through the non-owning handle's raw pointer).

// [spec:et:def:op-prod.torch.executor.native.prod-out-fn]
// [spec:et:sem:op-prod.torch.executor.native.prod-out-fn]
pub fn prod_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    in_: &Tensor,
    dtype: Option<ScalarType>,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    // (void)ctx;

    // PORT-NOTE: `check_prod_out_args` is under the C++ `#ifndef USE_ATEN_LIB`
    // block (portable arg-checkers are absent in the ATen build), so gate to match.
    #[cfg(not(feature = "aten"))]
    crate::et_kernel_check!(
        ctx,
        check_prod_out_args(in_, dtype, out),
        InvalidArgument,
        out
    );

    crate::et_kernel_check!(
        ctx,
        resize_tensor_same_type(out, ArrayRef::<TensorSizesType>::new()) == Error::Ok,
        InvalidArgument,
        out
    );

    let in_type: ScalarType = in_.scalar_type();
    let out_type: ScalarType = out.scalar_type();
    let name = "prod.int_out";

    crate::et_switch_realhbbf16_types!(in_type, ctx, name, CTYPE_IN, {
        crate::et_switch_realhbbf16_types!(out_type, ctx, name, CTYPE_OUT, {
            let data_in: *const CTYPE_IN = in_.const_data_ptr::<CTYPE_IN>();
            let data_out: *mut CTYPE_OUT = out.mutable_data_ptr::<CTYPE_OUT>();
            unsafe {
                *data_out = <CTYPE_OUT as StaticCast<i32>>::static_cast(1);
                for i in 0..in_.numel() {
                    *data_out = ProdMul::prod_mul(
                        *data_out,
                        <CTYPE_OUT as StaticCast<CTYPE_IN>>::static_cast(*data_in.add(i as usize)),
                    );
                }
            }
        });
    });

    out
}

// [spec:et:def:op-prod.torch.executor.native.prod-int-out-fn]
// [spec:et:sem:op-prod.torch.executor.native.prod-int-out-fn]
pub fn prod_int_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    in_: &Tensor,
    dim: i64,
    keepdim: bool,
    dtype: Option<ScalarType>,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    // (void)ctx;

    // PORT-NOTE: `check_reduction_args_single_dim` is under the C++
    // `#ifndef USE_ATEN_LIB` block (absent in the ATen build), so gate to match.
    #[cfg(not(feature = "aten"))]
    crate::et_kernel_check!(
        ctx,
        check_reduction_args_single_dim(
            in_,
            Some(dim),
            keepdim,
            dtype,
            out,
            /*allow_empty_dim=*/ true
        ),
        InvalidArgument,
        out
    );

    crate::et_kernel_check!(
        ctx,
        resize_reduction_out_dim(in_, &Some(dim), keepdim, out) == Error::Ok,
        InvalidArgument,
        out
    );

    let in_type: ScalarType = in_.scalar_type();
    let out_type: ScalarType = out.scalar_type();
    let name = "prod.int_out";

    crate::et_switch_realhbbf16_types!(in_type, ctx, name, CTYPE_IN, {
        crate::et_switch_realhbbf16_types!(out_type, ctx, name, CTYPE_OUT, {
            let out_data: *mut CTYPE_OUT = out.mutable_data_ptr::<CTYPE_OUT>();
            let success = parallel_for_each_reduce_over_dim_output_index(
                in_,
                Some(dim),
                out,
                &|begin: i64, end: i64| {
                    for out_ix in begin..end {
                        let out_ix = out_ix as usize;
                        let mut prod: CTYPE_OUT = <CTYPE_OUT as StaticCast<i32>>::static_cast(1);
                        if in_.numel() > 0 {
                            let acc: (CTYPE_OUT, i64) =
                                map_reduce_over_dim::<CTYPE_IN, CTYPE_OUT, _, _>(
                                    |v: CTYPE_IN| -> CTYPE_OUT {
                                        <CTYPE_OUT as StaticCast<CTYPE_IN>>::static_cast(v)
                                    },
                                    |outv: CTYPE_OUT,
                                     _: i64,
                                     acc: CTYPE_OUT,
                                     _: i64|
                                     -> (CTYPE_OUT, i64) {
                                        (ProdMul::prod_mul(acc, outv), 0)
                                    },
                                    in_,
                                    &Some(dim),
                                    out_ix,
                                );
                            prod = acc.0;
                        }
                        unsafe {
                            *out_data.add(out_ix) = prod;
                        }
                    }
                },
            );
            crate::et_kernel_check_msg!(ctx, success, Internal, out, "parallel_for failed");
        });
    });

    out
}

// PORT-NOTE: C++ accumulates the product with `operator*=` on `CTYPE_OUT`. The
// `half` crate's f16/bf16 do not implement `Mul`, and Rust's `bool` has no
// `*=`. This local trait reproduces the C++ `x * y` per ctype: primitives use
// wrapping-free `*` (two's-complement wrap on integer overflow matches C++),
// Bool uses the integer-promoted product (`b1 * b2 != 0`), and Half/BFloat16
// multiply in float then narrow (mirroring c10::Half::operator*=).
trait ProdMul {
    fn prod_mul(a: Self, b: Self) -> Self;
}
macro_rules! impl_prod_mul_int {
    ($($t:ty),*) => {$(
        impl ProdMul for $t {
            fn prod_mul(a: Self, b: Self) -> Self { a.wrapping_mul(b) }
        }
    )*};
}
impl_prod_mul_int!(u8, i8, i16, i32, i64);
impl ProdMul for f32 {
    fn prod_mul(a: Self, b: Self) -> Self {
        a * b
    }
}
impl ProdMul for f64 {
    fn prod_mul(a: Self, b: Self) -> Self {
        a * b
    }
}
impl ProdMul for bool {
    fn prod_mul(a: Self, b: Self) -> Self {
        (a as i32) * (b as i32) != 0
    }
}
impl ProdMul for crate::runtime::core::portable_type::Half {
    fn prod_mul(a: Self, b: Self) -> Self {
        crate::runtime::core::portable_type::Half::from_f32(a.to_f32() * b.to_f32())
    }
}
impl ProdMul for crate::runtime::core::portable_type::BFloat16 {
    fn prod_mul(a: Self, b: Self) -> Self {
        crate::runtime::core::portable_type::BFloat16::from_f32(a.to_f32() * b.to_f32())
    }
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

    fn op_prod_out<'a, 'b>(
        self_: &Tensor,
        dtype: Option<ScalarType>,
        out: &'a Tensor<'b>,
    ) -> &'a Tensor<'b> {
        let mut ctx = context();
        prod_out(&mut ctx, self_, dtype, out)
    }

    fn op_prod_int_out<'a, 'b>(
        self_: &Tensor,
        dim: i64,
        keepdim: bool,
        dtype: Option<ScalarType>,
        out: &'a Tensor<'b>,
    ) -> &'a Tensor<'b> {
        let mut ctx = context();
        prod_int_out(&mut ctx, self_, dim, keepdim, dtype, out)
    }

    trait FromI64: Copy {
        fn from_i64(v: i64) -> Self;
    }
    macro_rules! impl_from_i64_num {
        ($($t:ty),*) => {$(impl FromI64 for $t { fn from_i64(v: i64) -> Self { v as $t } })*};
    }
    impl_from_i64_num!(u8, i8, i16, i32, i64, f32, f64);
    impl FromI64 for Half {
        fn from_i64(v: i64) -> Self {
            Half::from_f32(v as f32)
        }
    }
    impl FromI64 for BFloat16 {
        fn from_i64(v: i64) -> Self {
            BFloat16::from_f32(v as f32)
        }
    }

    // PORT-NOTE: the C++ `test_dtype<DTYPE>()` derives the output factory dtype as
    // `isIntegralType(DTYPE, /*includeBool*/ true) ? Long : DTYPE`. Rust cannot
    // pick a type parameter at compile time from a runtime predicate, so the
    // output element type `O` is passed explicitly at each call site to match.
    fn test_prod_out_dtype<I, O>()
    where
        I: CppTypeToScalarType + FactoryValue + FromI64,
        O: CppTypeToScalarType + FactoryValue + FromI64,
    {
        let tf = TensorFactory::<I>::new();
        let tf_out = TensorFactory::<O>::new();

        let self_ = tf.make_default(vec![2, 3], (1..=6).map(I::from_i64).collect());
        let dtype: Option<ScalarType> = None;
        let out = tf_out.zeros_default(vec![]);
        // ET_FORALL_REALHBBF16_TYPES excludes Bool, so the expected value is 720.
        let out_expected = tf_out.make_default(vec![], vec![O::from_i64(720)]);
        op_prod_out(&self_, dtype, &out);
        assert_tensor_close!(out, out_expected);
    }

    fn test_prod_int_out_dtype<I>()
    where
        I: CppTypeToScalarType + FactoryValue + FromI64,
    {
        let tf = TensorFactory::<I>::new();

        let self_ = tf.make_default(vec![2, 3], (1..=6).map(I::from_i64).collect());
        let dim: i64 = 0;
        let keepdim = false;
        let dtype: Option<ScalarType> = None;
        let out = tf.zeros_default(vec![3]);
        let out_expected = tf.make_default(
            vec![3],
            [4, 10, 18].iter().map(|&v| I::from_i64(v)).collect(),
        );
        op_prod_int_out(&self_, dim, keepdim, dtype, &out);
        assert_tensor_close!(out, out_expected);
    }

    // [spec:et:sem:op-prod.torch.executor.native.prod-out-fn/test]
    #[test]
    fn op_prod_out_test_smoke_test() {
        // ET_FORALL_REALHBBF16_TYPES: Byte,Char,Short,Int,Long,Float,Double,Half,BFloat16.
        // Integral inputs (incl. bool) map the output dtype to Long (i64).
        test_prod_out_dtype::<u8, i64>();
        test_prod_out_dtype::<i8, i64>();
        test_prod_out_dtype::<i16, i64>();
        test_prod_out_dtype::<i32, i64>();
        test_prod_out_dtype::<i64, i64>();
        test_prod_out_dtype::<f32, f32>();
        test_prod_out_dtype::<f64, f64>();
        test_prod_out_dtype::<Half, Half>();
        test_prod_out_dtype::<BFloat16, BFloat16>();
    }

    // [spec:et:sem:op-prod.torch.executor.native.prod-int-out-fn/test]
    #[test]
    fn op_prod_int_out_test_smoke_test() {
        // ET_FORALL_REALHBBF16_TYPES: Byte,Char,Short,Int,Long,Float,Double,Half,BFloat16.
        test_prod_int_out_dtype::<u8>();
        test_prod_int_out_dtype::<i8>();
        test_prod_int_out_dtype::<i16>();
        test_prod_int_out_dtype::<i32>();
        test_prod_int_out_dtype::<i64>();
        test_prod_int_out_dtype::<f32>();
        test_prod_int_out_dtype::<f64>();
        test_prod_int_out_dtype::<Half>();
        test_prod_int_out_dtype::<BFloat16>();
    }

    // [spec:et:sem:op-prod.torch.executor.native.prod-int-out-fn/test]
    #[test]
    fn op_prod_int_out_test_smoke_test_keepdim() {
        let tf_float = TensorFactory::<f32>::new();

        let self_ = tf_float.make_default(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let dim: i64 = 0;
        let keepdim = true;
        let dtype: Option<ScalarType> = None;
        let out = tf_float.zeros_default(vec![1, 3]);
        let out_expected = tf_float.make_default(vec![1, 3], vec![4.0, 10.0, 18.0]);
        op_prod_int_out(&self_, dim, keepdim, dtype, &out);
        assert_tensor_close!(out, out_expected);
    }
}
