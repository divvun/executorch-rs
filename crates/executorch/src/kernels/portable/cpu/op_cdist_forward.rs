//! Literal port of kernels/portable/cpu/op_cdist_forward.cpp.

use crate::kernels::portable::cpu::util::broadcast_util::{
    get_broadcast_target_size, linearize_access_indexes_tensor,
};
use crate::kernels::portable::cpu::util::delinearize_index::delinearize_index_tensor;
use crate::kernels::portable::cpu::util::distance_util::{
    L0, L1, L2, Linf, Lp, Norm, NormCtype, check_cdist_args,
};
use crate::runtime::core::array_ref::ArrayRef;
use crate::runtime::core::error::Error;
use crate::runtime::core::exec_aten::util::tensor_util::{
    K_TENSOR_DIMENSION_LIMIT, resize_tensor_same_type, tensor_is_default_dim_order,
    tensors_have_same_dim_order3,
};
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::core::portable_type::tensor_impl::{SizesType, ssize_t};
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

// PORT-NOTE: `Tensor& out` / returned `Tensor&` become `&'a Tensor` (interior
// mutation through `*mut TensorImpl`). `std::optional<int64_t>` -> `Option<i64>`.
//
// PORT-NOTE (cross-module): the C++ switch is `ET_SWITCH_FLOATHBF16_TYPES`
// (Float/Double/Half/BFloat16), but `distance_util::NormCtype` is implemented
// only for `f32`/`f64` — the Half/BFloat16 arms of the switch cannot satisfy the
// `CTYPE: NormCtype` bound `cdist` requires (same gap the sibling `op_pdist`
// port hits). Unresolved cross-module reference: extend `NormCtype` (and the
// norm policies' math) to Half/BFloat16 in distance_util (fixer).

// [spec:et:def:op-cdist-forward.torch.executor.native.get-batch-sizes-fn]
// [spec:et:sem:op-cdist-forward.torch.executor.native.get-batch-sizes-fn]
//
// PORT-NOTE: returns an `ArrayRef<SizesType>` view over the tensor's sizes
// minus the trailing two dims (matches `{sizes().data(), sizes().size() - 2}`).
fn get_batch_sizes(tensor: &Tensor) -> ArrayRef<SizesType> {
    ArrayRef::from_raw_parts(tensor.sizes().data(), tensor.sizes().size() - 2)
}

// [spec:et:def:op-cdist-forward.torch.executor.native.cdist-fn]
// [spec:et:sem:op-cdist-forward.torch.executor.native.cdist-fn]
//
// PORT-NOTE: the C++ `template <typename CTYPE, typename Norm>` becomes a fn
// generic over `CTYPE: NormCtype` and the `Norm<CTYPE>` policy `N`.
fn cdist_norm<CTYPE: NormCtype, N: Norm<CTYPE>>(x1: &Tensor, x2: &Tensor, out: &Tensor, p: f64) {
    if out.numel() == 0 {
        return;
    }

    let out_data: *mut CTYPE = out.mutable_data_ptr::<CTYPE>();

    // If the last dimension of x1 (which is equal to the last dimension of x2)
    // has size 0, then the output is filled with 0s.
    if x1.numel() == 0 {
        for out_ix in 0..(out.numel() as usize) {
            unsafe {
                *out_data.add(out_ix) = CTYPE::ZERO;
            }
        }
        return;
    }

    let x1_data: *const CTYPE = x1.const_data_ptr::<CTYPE>();
    let x2_data: *const CTYPE = x2.const_data_ptr::<CTYPE>();

    let x1_batch_sizes: ArrayRef<SizesType> = get_batch_sizes(x1);
    let x2_batch_sizes: ArrayRef<SizesType> = get_batch_sizes(x2);
    let out_batch_sizes: ArrayRef<SizesType> = get_batch_sizes(out);

    let x1_is_broadcasted: bool = !out_batch_sizes.equals(x1_batch_sizes);
    let x2_is_broadcasted: bool = !out_batch_sizes.equals(x2_batch_sizes);
    let any_is_broadcasted: bool = x1_is_broadcasted || x2_is_broadcasted;

    let mut out_batch_numel: usize = 1;
    for i in 0..out_batch_sizes.size() {
        out_batch_numel *= *out_batch_sizes.at(i) as usize;
    }

    let big_p: usize = x1.size(x1.dim() - 2) as usize;
    let r: usize = x2.size(x2.dim() - 2) as usize;
    let m: usize = x1.size(x1.dim() - 1) as usize;

    let x1_inner_size: usize = big_p * m;
    let x2_inner_size: usize = r * m;
    let out_inner_size: usize = big_p * r;

    let p_ct: CTYPE = CTYPE::from_f64(p);

    for b in 0..out_batch_numel {
        let mut x1_base_ix: usize = b * x1_inner_size;
        let mut x2_base_ix: usize = b * x2_inner_size;
        let out_base_ix: usize = b * out_inner_size;

        if any_is_broadcasted {
            let mut out_base_coord: [usize; K_TENSOR_DIMENSION_LIMIT] =
                [0; K_TENSOR_DIMENSION_LIMIT];
            delinearize_index_tensor(
                out_base_ix,
                out,
                out_base_coord.as_mut_ptr(),
                K_TENSOR_DIMENSION_LIMIT,
            );

            if x1_is_broadcasted {
                x1_base_ix = linearize_access_indexes_tensor(
                    ArrayRef::from_raw_parts(out_base_coord.as_ptr(), K_TENSOR_DIMENSION_LIMIT),
                    out.dim() as ssize_t,
                    x1,
                );
            }
            if x2_is_broadcasted {
                x2_base_ix = linearize_access_indexes_tensor(
                    ArrayRef::from_raw_parts(out_base_coord.as_ptr(), K_TENSOR_DIMENSION_LIMIT),
                    out.dim() as ssize_t,
                    x2,
                );
            }
        }
        let mut out_ix: usize = 0;
        for i in 0..big_p {
            let row_i: *const CTYPE = unsafe { x1_data.add(x1_base_ix + i * m) };
            for j in 0..r {
                let row_j: *const CTYPE = unsafe { x2_data.add(x2_base_ix + j * m) };
                let mut agg: CTYPE = CTYPE::ZERO;
                for k in 0..m {
                    let diff: CTYPE = (unsafe { *row_i.add(k) } - unsafe { *row_j.add(k) }).abs();
                    agg = N::reduce(agg, N::map(diff, p_ct));
                }
                unsafe {
                    *out_data.add(out_base_ix + out_ix) = N::finish(agg, p_ct);
                }
                out_ix += 1;
            }
        }
    }
}

// PORT-NOTE: the C++ `template <typename CTYPE> void cdist(...)` norm-selecting
// overload; dispatches on the runtime `p` (exact floating-point equality).
fn cdist<CTYPE: NormCtype>(x1: &Tensor, x2: &Tensor, out: &Tensor, p: f64) {
    if p == 0.0 {
        cdist_norm::<CTYPE, L0>(x1, x2, out, p);
    } else if p == 1.0 {
        cdist_norm::<CTYPE, L1>(x1, x2, out, p);
    } else if p == 2.0 {
        cdist_norm::<CTYPE, L2>(x1, x2, out, p);
    } else if p == f64::INFINITY {
        cdist_norm::<CTYPE, Linf>(x1, x2, out, p);
    } else {
        cdist_norm::<CTYPE, Lp>(x1, x2, out, p);
    }
}

// [spec:et:def:op-cdist-forward.torch.executor.native.cdist-forward-out-fn]
// [spec:et:sem:op-cdist-forward.torch.executor.native.cdist-forward-out-fn]
pub fn cdist_forward_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    x1: &Tensor,
    x2: &Tensor,
    p: f64,
    compute_mode: Option<i64>,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    crate::et_kernel_check!(
        ctx,
        tensors_have_same_dim_order3(x1, x2, out),
        InvalidArgument,
        out
    );

    crate::et_kernel_check!(ctx, tensor_is_default_dim_order(x1), InvalidArgument, out);

    crate::et_kernel_check!(
        ctx,
        check_cdist_args(x1, x2, p, compute_mode, out),
        InvalidArgument,
        out
    );

    let mut target_sizes: [SizesType; K_TENSOR_DIMENSION_LIMIT] = [0; K_TENSOR_DIMENSION_LIMIT];
    let mut target_ndim: usize = 0;

    crate::et_kernel_check!(
        ctx,
        get_broadcast_target_size(
            ArrayRef::from_raw_parts(x1.sizes().data(), x1.sizes().size() - 2),
            ArrayRef::from_raw_parts(x2.sizes().data(), x2.sizes().size() - 2),
            target_sizes.as_mut_ptr(),
            K_TENSOR_DIMENSION_LIMIT,
            &mut target_ndim,
        ) == Error::Ok,
        InvalidArgument,
        out
    );

    target_ndim += 2;
    target_sizes[target_ndim - 2] = x1.size(x1.dim() - 2) as SizesType;
    target_sizes[target_ndim - 1] = x2.size(x2.dim() - 2) as SizesType;

    crate::et_kernel_check!(
        ctx,
        resize_tensor_same_type(
            out,
            ArrayRef::from_raw_parts(target_sizes.as_ptr(), target_ndim)
        ) == Error::Ok,
        InvalidArgument,
        out
    );

    let out_type = out.scalar_type();

    crate::et_switch_floathbf16_types!(out_type, ctx, "_cdist_forward.out", CTYPE, {
        cdist::<CTYPE>(x1, x2, out, p);
    });

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::core::exec_aten::testing_util::tensor_factory::{
        FactoryValue, TensorFactory,
    };
    use crate::runtime::core::exec_aten::testing_util::tensor_util::internal::K_DEFAULT_BFLOAT16_ATOL;
    use crate::runtime::core::exec_aten::util::scalar_type_util::CppTypeToScalarType;
    use crate::runtime::core::memory_allocator::{MemoryAllocator, MemoryAllocatorBase};
    use crate::runtime::core::portable_type::scalar_type::ScalarType;
    use crate::runtime::core::portable_type::{BFloat16, Half};
    use crate::{assert_tensor_close, assert_tensor_close_with_tol};

    fn context() -> KernelRuntimeContext<'static> {
        crate::runtime::platform::runtime::runtime_init();
        KernelRuntimeContext::new(
            crate::extension::module::module::null_event_tracer(),
            core::ptr::null_mut::<MemoryAllocator>() as *mut dyn MemoryAllocatorBase,
        )
    }

    trait FromF64 {
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

    fn v<T: FromF64>(vals: &[f64]) -> Vec<T> {
        vals.iter().map(|&x| T::from_f64(x)).collect()
    }

    fn test_dtype<T>()
    where
        T: CppTypeToScalarType + FactoryValue + FromF64,
    {
        let tf = TensorFactory::<T>::new();

        // PORT-NOTE: the C++ `is_aten` early-return for Half/BFloat16 is dropped;
        // this is the non-ATen build, which supports those dtypes.
        let _ = T::VALUE == ScalarType::Half || T::VALUE == ScalarType::BFloat16;

        let x1 = tf.make_default(
            vec![2, 1, 4, 3],
            v(&[
                0., 1., 2., 3., 5., 4., 3., -3., 7., 1., 6., 2., -1., 5., 1., 1., -2., 1., 5., 4.,
                3., 2., -1., 5.,
            ]),
        );
        let x2 = tf.make_default(
            vec![1, 2, 5, 3],
            v(&[
                0., 1., 2., 3., 5., -3., 7., 1., 6., 2., -1., 5., 1., 1., -2., 4., 3., 2., -1., 5.,
                1., 1., -2., 1., 5., 4., 3., 2., -1., 5.,
            ]),
        );
        let compute_mode: Option<i64> = None;

        let out = tf.zeros_default(vec![2, 2, 4, 5]);

        let l0 = tf.make_default(
            vec![2, 2, 4, 5],
            v(&[
                0., 3., 2., 3., 2., 3., 1., 3., 3., 3., 3., 2., 3., 3., 3., 2., 3., 3., 3., 2., 2.,
                3., 3., 3., 3., 3., 2., 3., 3., 3., 3., 3., 3., 3., 3., 2., 3., 2., 3., 3., 3., 2.,
                3., 3., 3., 3., 3., 3., 3., 2., 3., 3., 3., 3., 3., 3., 3., 3., 0., 3., 3., 0., 2.,
                3., 3., 3., 2., 0., 3., 3., 3., 3., 3., 0., 3., 3., 3., 3., 3., 0.,
            ]),
        );
        let mut ctx = context();
        cdist_forward_out(&mut ctx, &x1, &x2, 0.0, compute_mode, &out);
        assert_tensor_close!(out, l0);

        let l1 = tf.make_default(
            vec![2, 2, 4, 5],
            v(&[
                0., 12., 11., 7., 5., 9., 7., 10., 8., 12., 12., 18., 9., 5., 15., 6., 8., 15.,
                11., 9., 6., 6., 5., 9., 7., 5., 7., 12., 4., 8., 12., 18., 9., 13., 5., 6., 4.,
                9., 7., 11., 6., 8., 17., 13., 9., 5., 13., 14., 6., 6., 9., 9., 8., 10., 12., 7.,
                15., 8., 0., 10., 8., 0., 9., 9., 13., 9., 9., 0., 12., 6., 3., 9., 12., 0., 10.,
                9., 13., 6., 10., 0.,
            ]),
        );
        cdist_forward_out(&mut ctx, &x1, &x2, 1.0, compute_mode, &out);
        assert_tensor_close!(out, l1);

        let l2 = tf.make_default(
            vec![2, 2, 4, 5],
            v(&[
                0.00000000,
                7.07106781,
                8.06225777,
                4.12310553,
                4.12310553,
                5.38516474,
                7.00000000,
                6.00000000,
                6.16441393,
                7.48331499,
                7.07106781,
                12.80624866,
                5.74456263,
                3.00000000,
                10.04987526,
                5.09901953,
                5.47722578,
                8.77496433,
                7.68114567,
                6.40312433,
                4.47213602,
                4.24264050,
                3.31662488,
                5.91608000,
                4.12310553,
                3.00000000,
                5.00000000,
                7.87400770,
                2.44948983,
                6.16441393,
                7.87400770,
                10.77032948,
                6.40312433,
                8.30662346,
                3.00000000,
                4.24264050,
                2.44948983,
                8.06225777,
                4.58257580,
                7.68114567,
                4.24264050,
                5.65685415,
                10.24695110,
                7.81024981,
                5.38516474,
                3.31662488,
                8.30662346,
                8.36660004,
                4.24264050,
                4.24264050,
                5.91608000,
                6.40312433,
                4.69041586,
                6.16441393,
                7.07106781,
                4.12310553,
                10.04987526,
                5.47722578,
                0.00000000,
                7.34846926,
                5.47722578,
                0.00000000,
                7.28010988,
                6.40312433,
                7.81024981,
                5.91608000,
                7.28010988,
                0.00000000,
                7.48331499,
                4.24264050,
                1.73205078,
                6.40312433,
                7.48331499,
                0.00000000,
                6.16441393,
                5.38516474,
                7.81024981,
                4.24264050,
                6.16441393,
                0.00000000,
            ]),
        );
        cdist_forward_out(&mut ctx, &x1, &x2, 2.0, compute_mode, &out);
        assert_tensor_close!(out, l2);

        let l3 = tf.make_default(
            vec![2, 2, 4, 5],
            v(&[
                0.00000000,
                6.00000000,
                7.41079521,
                3.50339794,
                4.02072573,
                4.62606478,
                7.00000000,
                5.14256334,
                6.01846170,
                6.60385466,
                6.00000000,
                11.47758675,
                5.05277443,
                2.57128167,
                9.28704357,
                5.01329803,
                5.11722994,
                7.39863634,
                7.18551636,
                5.73879337,
                4.16016769,
                4.04124022,
                3.07231688,
                5.34848118,
                3.50339794,
                2.57128167,
                4.49794149,
                7.23042679,
                2.15443468,
                6.01846170,
                6.99319077,
                9.25212955,
                6.08220196,
                7.45903587,
                2.57128167,
                3.77976322,
                2.15443468,
                8.00520515,
                4.17933941,
                7.18551636,
                4.04124022,
                5.03968430,
                8.88326645,
                6.74599648,
                4.62606478,
                3.07231688,
                7.45903587,
                7.16609573,
                4.04124022,
                3.77976322,
                5.34848118,
                6.08220196,
                3.95789170,
                5.42883539,
                6.00000000,
                3.50339794,
                9.00000000,
                5.11722994,
                0.00000000,
                7.06069660,
                5.11722994,
                0.00000000,
                7.05400419,
                6.08220196,
                6.74599648,
                5.34848118,
                7.05400419,
                0.00000000,
                6.60385466,
                4.04124022,
                1.44224954,
                6.08220196,
                6.60385466,
                0.00000000,
                5.42883539,
                4.62606478,
                6.74599648,
                4.04124022,
                5.42883539,
                0.00000000,
            ]),
        );
        cdist_forward_out(&mut ctx, &x1, &x2, 3.0, compute_mode, &out);
        if T::VALUE == ScalarType::BFloat16 {
            assert_tensor_close_with_tol!(out, l3, 1e-2, K_DEFAULT_BFLOAT16_ATOL);
        } else {
            assert_tensor_close!(out, l3);
        }

        let linf = tf.make_default(
            vec![2, 2, 4, 5],
            v(&[
                0., 5., 7., 3., 4., 4., 7., 4., 6., 6., 5., 10., 4., 2., 9., 5., 5., 6., 7., 5.,
                4., 4., 3., 5., 3., 2., 4., 7., 2., 6., 6., 8., 6., 7., 2., 3., 2., 8., 4., 7., 4.,
                4., 8., 6., 4., 3., 7., 6., 4., 3., 5., 6., 3., 5., 5., 3., 8., 5., 0., 7., 5., 0.,
                7., 6., 6., 5., 7., 0., 6., 4., 1., 6., 6., 0., 5., 4., 6., 4., 5., 0.,
            ]),
        );
        cdist_forward_out(&mut ctx, &x1, &x2, f64::INFINITY, compute_mode, &out);
        assert_tensor_close!(out, linf);
    }

    // [spec:et:sem:op-cdist-forward.torch.executor.native.cdist-forward-out-fn/test]
    // Also exercises check_cdist_args on the valid (accepting) path: rank-4
    // inputs, matching dtypes and trailing dim, p>=0, compute_mode=None.
    // The p=0/1/2/3/inf sweep drives the `cdist` norm dispatcher, and the
    // broadcasted batch dims exercise get_batch_sizes' broadcast detection.
    // [spec:et:sem:distance-util.torch.executor.check-cdist-args-fn/test]
    // [spec:et:sem:op-cdist-forward.torch.executor.native.cdist-fn/test]
    // [spec:et:sem:op-cdist-forward.torch.executor.native.get-batch-sizes-fn/test]
    #[test]
    fn op_cdist_forward_out_test_smoke_test() {
        test_dtype::<f32>();
        test_dtype::<f64>();
        test_dtype::<Half>();
        test_dtype::<BFloat16>();
    }
}
