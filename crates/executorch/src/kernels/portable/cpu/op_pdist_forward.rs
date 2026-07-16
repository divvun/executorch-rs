//! Literal port of kernels/portable/cpu/op_pdist_forward.cpp.

use crate::kernels::portable::cpu::util::distance_util::{
    check_pdist_args, get_pdist_out_target_size, pdist,
};
use crate::runtime::core::array_ref::ArrayRef;
use crate::runtime::core::error::Error;
use crate::runtime::core::exec_aten::util::tensor_util::{
    K_TENSOR_DIMENSION_LIMIT, resize_tensor_same_type, tensor_is_default_dim_order,
    tensors_have_same_dim_order2,
};
use crate::runtime::core::portable_type::scalar_type::ScalarType;
use crate::runtime::core::portable_type::tensor::{Tensor, TensorSizesType};
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

// PORT-NOTE: C++ `Tensor& out` / returned `Tensor&` become `&'a Tensor` (interior
// mutation through the non-owning handle's raw pointer).

// [spec:et:def:op-pdist-forward.torch.executor.native.pdist-forward-out-fn]
// [spec:et:sem:op-pdist-forward.torch.executor.native.pdist-forward-out-fn]
pub fn _pdist_forward_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    in_: &Tensor,
    p: f64,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    // (void)ctx;

    crate::et_kernel_check!(ctx, check_pdist_args(in_, p, out), InvalidArgument, out);

    crate::et_kernel_check!(
        ctx,
        tensors_have_same_dim_order2(in_, out),
        InvalidArgument,
        out
    );

    crate::et_kernel_check!(ctx, tensor_is_default_dim_order(in_), InvalidArgument, out);

    let mut target_sizes: [TensorSizesType; K_TENSOR_DIMENSION_LIMIT] =
        [0; K_TENSOR_DIMENSION_LIMIT];
    let mut target_ndim: usize = 0;
    unsafe {
        get_pdist_out_target_size(in_, target_sizes.as_mut_ptr(), &mut target_ndim);
    }

    crate::et_kernel_check!(
        ctx,
        resize_tensor_same_type(
            out,
            ArrayRef::from_raw_parts(target_sizes.as_ptr(), target_ndim)
        ) == Error::Ok,
        InvalidArgument,
        out
    );

    let in_type: ScalarType = in_.scalar_type();
    let name = "_pdist_forward.out";

    crate::et_switch_floathbf16_types!(in_type, ctx, name, CTYPE, {
        pdist::<CTYPE>(in_, out, p);
    });

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::core::exec_aten::testing_util::tensor_factory::{
        FactoryValue, TensorFactory,
    };
    use crate::runtime::core::exec_aten::testing_util::tensor_util::internal::K_DEFAULT_ATOL;
    use crate::runtime::core::exec_aten::util::scalar_type_util::CppTypeToScalarType;
    use crate::runtime::core::memory_allocator::{MemoryAllocator, MemoryAllocatorBase};
    use crate::runtime::core::portable_type::{BFloat16, Half};
    use crate::{assert_tensor_close, assert_tensor_close_with_tol};

    fn context() -> KernelRuntimeContext<'static> {
        crate::runtime::platform::runtime::runtime_init();
        KernelRuntimeContext::new(
            crate::extension::module::module::null_event_tracer(),
            core::ptr::null_mut::<MemoryAllocator>() as *mut dyn MemoryAllocatorBase,
        )
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
            Half::from_f32(v as f32)
        }
    }
    impl FromF64 for BFloat16 {
        fn from_f64(v: f64) -> Self {
            BFloat16::from_f32(v as f32)
        }
    }

    fn test_dtype<T>()
    where
        T: CppTypeToScalarType + FactoryValue + FromF64,
    {
        let tf = TensorFactory::<T>::new();
        let vv = |vals: &[f64]| -> Vec<T> { vals.iter().map(|&x| T::from_f64(x)).collect() };

        // PORT-NOTE: the C++ `is_aten` early-return for Half/BFloat16 is dropped;
        // this is the non-ATen build, which supports those dtypes.
        let is_half_or_bf16 = T::VALUE == ScalarType::Half || T::VALUE == ScalarType::BFloat16;

        let in_ = tf.make_default(
            vec![4, 5],
            vv(&[
                0., 1., 2., 3., 5., 4., 3., 2., -1., 5., 1., 1., -2., 1., 5., 4., 3., 2., -1., 5.,
            ]),
        );
        let out = tf.zeros_default(vec![6]);

        let l0 = tf.make_default(vec![6], vv(&[3., 3., 3., 4., 0., 4.]));
        let mut ctx = context();
        _pdist_forward_out(&mut ctx, &in_, 0.0, &out);
        assert_tensor_close!(out, l0);

        let l0p5 = tf.make_default(
            vec![6],
            vv(&[
                29.31370926,
                19.48528290,
                29.31370926,
                43.03986740,
                0.0,
                43.03986740,
            ]),
        );
        _pdist_forward_out(&mut ctx, &in_, 0.5, &out);
        if is_half_or_bf16 {
            assert_tensor_close_with_tol!(out, l0p5, 1e-2, K_DEFAULT_ATOL);
        } else {
            assert_tensor_close!(out, l0p5);
        }

        let l1 = tf.make_default(vec![6], vv(&[10., 7., 10., 11., 0., 11.]));
        _pdist_forward_out(&mut ctx, &in_, 1.0, &out);
        assert_tensor_close!(out, l1);

        let l1p5 = tf.make_default(
            vec![6],
            vv(&[
                7.07743692, 5.19140196, 7.07743692, 7.08359480, 0.0, 7.08359480,
            ]),
        );
        _pdist_forward_out(&mut ctx, &in_, 1.5, &out);
        if is_half_or_bf16 {
            assert_tensor_close_with_tol!(out, l1p5, 1e-2, K_DEFAULT_ATOL);
        } else {
            assert_tensor_close!(out, l1p5);
        }

        let l2 = tf.make_default(
            vec![6],
            vv(&[6.0, 4.58257580, 6.0, 5.74456263, 0.0, 5.74456263]),
        );
        _pdist_forward_out(&mut ctx, &in_, 2.0, &out);
        assert_tensor_close!(out, l2);

        let l3 = tf.make_default(
            vec![6],
            vv(&[
                5.14256334, 4.17933941, 5.14256334, 4.74745941, 0.0, 4.74745941,
            ]),
        );
        _pdist_forward_out(&mut ctx, &in_, 3.0, &out);
        assert_tensor_close!(out, l3);

        let linf = tf.make_default(vec![6], vv(&[4., 4., 4., 4., 0., 4.]));
        _pdist_forward_out(&mut ctx, &in_, f64::INFINITY, &out);
        assert_tensor_close!(out, linf);
    }

    // [spec:et:sem:op-pdist-forward.torch.executor.native.pdist-forward-out-fn/test]
    // Integration test spanning the distance_util helpers this op drives; each
    // norm's map/reduce/finish is pinned by a distinct expected tensor.
    // [spec:et:sem:distance-util.torch.executor.check-pdist-args-fn/test]
    // [spec:et:sem:distance-util.torch.executor.get-pdist-out-target-size-fn/test]
    // [spec:et:sem:distance-util.torch.executor.pdist-fn/test]
    // [spec:et:sem:distance-util.torch.executor.l0.map-fn/test]
    // [spec:et:sem:distance-util.torch.executor.l0.reduce-fn/test]
    // [spec:et:sem:distance-util.torch.executor.l0.finish-fn/test]
    // [spec:et:sem:distance-util.torch.executor.l1.map-fn/test]
    // [spec:et:sem:distance-util.torch.executor.l1.reduce-fn/test]
    // [spec:et:sem:distance-util.torch.executor.l1.finish-fn/test]
    // [spec:et:sem:distance-util.torch.executor.l2.map-fn/test]
    // [spec:et:sem:distance-util.torch.executor.l2.reduce-fn/test]
    // [spec:et:sem:distance-util.torch.executor.l2.finish-fn/test]
    // [spec:et:sem:distance-util.torch.executor.lp.map-fn/test]
    // [spec:et:sem:distance-util.torch.executor.lp.reduce-fn/test]
    // [spec:et:sem:distance-util.torch.executor.lp.finish-fn/test]
    // [spec:et:sem:distance-util.torch.executor.linf.map-fn/test]
    // [spec:et:sem:distance-util.torch.executor.linf.reduce-fn/test]
    // [spec:et:sem:distance-util.torch.executor.linf.finish-fn/test]
    #[test]
    fn op_pdist_forward_out_test_smoke_test() {
        // ET_FORALL_FLOATHBF16_TYPES
        test_dtype::<f32>();
        test_dtype::<f64>();
        test_dtype::<Half>();
        test_dtype::<BFloat16>();
    }
}
