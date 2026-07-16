//! Literal port of kernels/portable/cpu/op_any.cpp.

use crate::kernels::portable::cpu::util::dtype_util::StaticCast;
use crate::kernels::portable::cpu::util::reduce_util::{
    MapReduceOverDimListPlan, map_reduce_over_dim,
    parallel_for_each_reduce_over_dim_list_output_index,
    parallel_for_each_reduce_over_dim_output_index, resize_reduction_out, resize_reduction_out_dim,
};
#[cfg(not(feature = "aten"))]
use crate::kernels::portable::cpu::util::reduce_util::{
    check_reduction_args, check_reduction_args_single_dim,
};
use crate::runtime::core::array_ref::ArrayRef;
use crate::runtime::core::error::Error;
use crate::runtime::core::exec_aten::util::tensor_util::{
    resize_tensor, tensors_have_same_dim_order2,
};
use crate::runtime::core::portable_type::scalar_type::ScalarType;
use crate::runtime::core::portable_type::tensor::{Tensor, TensorSizesType};
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

// PORT-NOTE: C++ `Tensor& out` / returned `Tensor&` become `&'a Tensor` (interior
// mutation through the non-owning handle's raw pointer). `optional<ArrayRef<
// int64_t>> dim_list` maps to `Option<ArrayRef<i64>>`. The `ET_SWITCH_TWO_TYPES(
// Bool, Byte, ...)` output-dtype switch is provided by `et_switch_two_types!`.

// [spec:et:def:op-any.torch.executor.native.any-all-out-fn]
// [spec:et:sem:op-any.torch.executor.native.any-all-out-fn]
pub fn any_all_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    in_: &Tensor,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    // (void)ctx;

    crate::et_kernel_check!(
        ctx,
        resize_tensor(out, ArrayRef::<TensorSizesType>::new()) == Error::Ok,
        InvalidArgument,
        out
    );

    crate::et_kernel_check!(
        ctx,
        tensors_have_same_dim_order2(in_, out),
        InvalidArgument,
        out
    );

    let in_type: ScalarType = in_.scalar_type();
    let out_type: ScalarType = out.scalar_type();

    let op_name = "any.all_out";

    crate::et_switch_realhbbf16_types!(in_type, ctx, op_name, CTYPE_IN, {
        crate::et_switch_two_types!(Bool, Byte, out_type, ctx, op_name, CTYPE_OUT, {
            let data_in: *const CTYPE_IN = in_.const_data_ptr::<CTYPE_IN>();
            let data_out: *mut CTYPE_OUT = out.mutable_data_ptr::<CTYPE_OUT>();
            unsafe {
                *data_out = <CTYPE_OUT as StaticCast<bool>>::static_cast(false);
                for i in 0..in_.numel() {
                    if <bool as StaticCast<CTYPE_IN>>::static_cast(*data_in.add(i as usize)) {
                        *data_out = <CTYPE_OUT as StaticCast<bool>>::static_cast(true);
                        break;
                    }
                }
            }
        });
    });

    out
}

// [spec:et:def:op-any.torch.executor.native.any-dims-out-fn]
// [spec:et:sem:op-any.torch.executor.native.any-dims-out-fn]
pub fn any_dims_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    in_: &Tensor,
    dim_list: Option<ArrayRef<i64>>,
    keepdim: bool,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    // (void)ctx;

    // PORT-NOTE: `check_reduction_args` is under the C++ `#ifndef USE_ATEN_LIB`
    // block (the portable arg-checkers are absent in the ATen build), so gate to
    // match.
    #[cfg(not(feature = "aten"))]
    crate::et_kernel_check!(
        ctx,
        check_reduction_args(in_, &dim_list, keepdim, None, out),
        InvalidArgument,
        out
    );

    if dim_list.is_some() && dim_list.as_ref().unwrap().empty() {
        crate::et_kernel_check!(
            ctx,
            resize_tensor(out, in_.sizes()) == Error::Ok,
            InvalidArgument,
            out
        );
    } else {
        crate::et_kernel_check!(
            ctx,
            resize_reduction_out(in_, &dim_list, keepdim, out) == Error::Ok,
            InvalidArgument,
            out
        );
    }

    crate::et_kernel_check!(
        ctx,
        tensors_have_same_dim_order2(in_, out),
        InvalidArgument,
        out
    );

    let in_type: ScalarType = in_.scalar_type();
    let out_type: ScalarType = out.scalar_type();

    let op_name = "any.dims_out";

    let in_not_empty: bool = in_.numel() > 0;
    let plan: Option<MapReduceOverDimListPlan> =
        if (dim_list.is_none() || !dim_list.as_ref().unwrap().empty()) && in_not_empty {
            Some(MapReduceOverDimListPlan::new(in_, &dim_list))
        } else {
            None
        };
    crate::et_switch_realhbbf16_types!(in_type, ctx, op_name, CTYPE_IN, {
        crate::et_switch_two_types!(Bool, Byte, out_type, ctx, op_name, CTYPE_OUT, {
            let out_data: *mut CTYPE_OUT = out.mutable_data_ptr::<CTYPE_OUT>();
            if dim_list.is_some() && dim_list.as_ref().unwrap().empty() {
                let in_data: *const CTYPE_IN = in_.const_data_ptr::<CTYPE_IN>();
                for out_ix in 0..out.numel() {
                    let out_ix = out_ix as usize;
                    unsafe {
                        *out_data.add(out_ix) =
                            <CTYPE_OUT as StaticCast<bool>>::static_cast(<bool as StaticCast<
                                CTYPE_IN,
                            >>::static_cast(
                                *in_data.add(out_ix)
                            ));
                    }
                }
            } else {
                let success = parallel_for_each_reduce_over_dim_list_output_index(
                    in_,
                    dim_list,
                    out,
                    &|begin: i64, end: i64| {
                        for out_ix in begin..end {
                            let out_ix = out_ix as usize;
                            let mut any: bool = false;
                            if let Some(plan) = plan.as_ref() {
                                any = plan.execute::<CTYPE_IN, bool, _, _>(
                                    |v: CTYPE_IN| -> bool {
                                        <bool as StaticCast<CTYPE_IN>>::static_cast(v)
                                    },
                                    |outv: bool, acc: bool| -> bool { acc || outv },
                                    out_ix,
                                );
                            }
                            unsafe {
                                *out_data.add(out_ix) =
                                    <CTYPE_OUT as StaticCast<bool>>::static_cast(any);
                            }
                        }
                    },
                );
                crate::et_kernel_check_msg!(ctx, success, Internal, out, "parallel_for failed");
            }
        });
    });

    out
}

// PORT-NOTE: the C++ `map_reduce_over_dim<CTYPE_IN, CTYPE_OUT>` map lambda returns
// `bool` and the reduce lambda takes `bool` params but is instantiated with
// `CTYPE_OUT`, relying on implicit `bool <-> CTYPE_OUT` conversions in the util
// (`CTYPE_OUT acc_val = map_fun(...)`, `std::tuple<CTYPE_OUT,long> res =
// reduce_fun(...)`). The ported `map_reduce_over_dim<CTYPE_IN, CTYPE_OUT>` is
// strictly typed over `CTYPE_OUT`, so those implicit conversions are made
// explicit here via `StaticCast`: map produces `static_cast<CTYPE_OUT>(
// static_cast<bool>(v))` and reduce converts its `CTYPE_OUT` operands to `bool`,
// ORs them, and casts back — reproducing the C++ conversion chain.

// [spec:et:def:op-any.torch.executor.native.any-out-fn]
// [spec:et:sem:op-any.torch.executor.native.any-out-fn]
pub fn any_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    in_: &Tensor,
    dim: i64,
    keepdim: bool,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    // (void)ctx;

    // PORT-NOTE: `check_reduction_args_single_dim` is under the C++ `#ifndef
    // USE_ATEN_LIB` block (the portable arg-checkers are absent in the ATen
    // build), so gate to match.
    #[cfg(not(feature = "aten"))]
    crate::et_kernel_check!(
        ctx,
        check_reduction_args_single_dim(
            in_,
            Some(dim),
            keepdim,
            None,
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

    crate::et_kernel_check!(
        ctx,
        tensors_have_same_dim_order2(in_, out),
        InvalidArgument,
        out
    );

    let in_type: ScalarType = in_.scalar_type();
    let out_type: ScalarType = out.scalar_type();

    let op_name = "any.out";

    crate::et_switch_realhbbf16_types!(in_type, ctx, op_name, CTYPE_IN, {
        crate::et_switch_two_types!(Bool, Byte, out_type, ctx, op_name, CTYPE_OUT, {
            let out_data: *mut CTYPE_OUT = out.mutable_data_ptr::<CTYPE_OUT>();
            let success = parallel_for_each_reduce_over_dim_output_index(
                in_,
                Some(dim),
                out,
                &|begin: i64, end: i64| {
                    for out_ix in begin..end {
                        let out_ix = out_ix as usize;
                        let mut any: CTYPE_OUT =
                            <CTYPE_OUT as StaticCast<bool>>::static_cast(false);
                        if in_.numel() > 0 {
                            let acc: (CTYPE_OUT, i64) =
                                map_reduce_over_dim::<CTYPE_IN, CTYPE_OUT, _, _>(
                                    |v: CTYPE_IN| -> CTYPE_OUT {
                                        <CTYPE_OUT as StaticCast<bool>>::static_cast(
                                            <bool as StaticCast<CTYPE_IN>>::static_cast(v),
                                        )
                                    },
                                    |outv: CTYPE_OUT,
                                     _: i64,
                                     acc: CTYPE_OUT,
                                     _: i64|
                                     -> (CTYPE_OUT, i64) {
                                        let outv_b: bool =
                                            <bool as StaticCast<CTYPE_OUT>>::static_cast(outv);
                                        let acc_b: bool =
                                            <bool as StaticCast<CTYPE_OUT>>::static_cast(acc);
                                        (
                                            <CTYPE_OUT as StaticCast<bool>>::static_cast(
                                                acc_b || outv_b,
                                            ),
                                            0,
                                        )
                                    },
                                    in_,
                                    &Some(dim),
                                    out_ix,
                                );
                            any = acc.0;
                        }
                        unsafe {
                            *out_data.add(out_ix) = any;
                        }
                    }
                },
            );
            crate::et_kernel_check_msg!(ctx, success, Internal, out, "parallel_for failed");
        });
    });

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assert_tensor_close;
    use crate::assert_tensor_eq;
    use crate::runtime::core::exec_aten::testing_util::tensor_factory::TensorFactory;
    use crate::runtime::core::memory_allocator::{MemoryAllocator, MemoryAllocatorBase};

    fn context() -> KernelRuntimeContext<'static> {
        // Mirrors `OperatorTest::SetUp()`'s `runtime_init()` PAL initialization.
        crate::runtime::platform::platform::pal_init();
        KernelRuntimeContext::new(
            crate::extension::module::module::null_event_tracer(),
            core::ptr::null_mut::<MemoryAllocator>() as *mut dyn MemoryAllocatorBase,
        )
    }

    // PORT-NOTE: `test_any_all_out_invalid_type<OUT_DTYPE>` dispatched via
    // `ET_FORALL_FLOAT_TYPES` (Float, Double) in `InvalidDtypeDies`; each is a
    // separate helper call. `ET_EXPECT_KERNEL_FAILURE` runs the expr and then
    // asserts the context recorded a non-Ok failure state.
    fn test_any_all_out_invalid_type<T>()
    where
        T: crate::runtime::core::exec_aten::util::scalar_type_util::CppTypeToScalarType
            + crate::runtime::core::exec_aten::testing_util::tensor_factory::FactoryValue,
    {
        let tf_float = TensorFactory::<f32>::new();
        let tf_out = TensorFactory::<T>::new();

        let in_ = tf_float.make_default(vec![1, 4], vec![0.0, 0.0, 1.0, 0.0]);
        let out = tf_out.zeros_default(vec![0]);

        let mut ctx = context();
        any_all_out(&mut ctx, &in_, &out);
        assert_ne!(ctx.failure_state(), Error::Ok);
    }

    // PORT-NOTE: `test_any_all_out<IN_DTYPE>` dispatched via
    // `ET_FORALL_REALHBBF16_TYPES` in `AllRealInputTypePasses`; each is a separate
    // helper call. Reproduced via `FromI64` for integer-literal make() values.
    trait FromI64 {
        fn from_i64(v: i64) -> Self;
    }
    macro_rules! impl_from_i64_num {
        ($($t:ty),*) => {$(impl FromI64 for $t { fn from_i64(v: i64) -> Self { v as $t } })*};
    }
    impl_from_i64_num!(u8, i8, i16, i32, i64, f32, f64);
    impl FromI64 for crate::runtime::core::portable_type::Half {
        fn from_i64(v: i64) -> Self {
            crate::runtime::core::portable_type::Half::from_f32(v as f32)
        }
    }
    impl FromI64 for crate::runtime::core::portable_type::BFloat16 {
        fn from_i64(v: i64) -> Self {
            crate::runtime::core::portable_type::BFloat16::from_f32(v as f32)
        }
    }

    fn test_any_all_out<T>()
    where
        T: crate::runtime::core::exec_aten::util::scalar_type_util::CppTypeToScalarType
            + crate::runtime::core::exec_aten::testing_util::tensor_factory::FactoryValue
            + FromI64,
    {
        let tf_in = TensorFactory::<T>::new();
        let tf_bool = TensorFactory::<bool>::new();

        let in_ = tf_in.make_default(
            vec![2, 4],
            [0, 1, 0, 1, 1, 0, 1, 0]
                .iter()
                .map(|&v| T::from_i64(v))
                .collect(),
        );
        let bool_false_in = tf_bool.make_default(vec![2, 4], vec![false; 8]);
        let bool_true_in = tf_bool.make_default(vec![2, 4], vec![true; 8]);

        let out = tf_bool.make_default(vec![], vec![false]);

        let mut ctx = context();
        any_all_out(&mut ctx, &in_, &out);
        assert_tensor_eq!(out, tf_bool.make_default(vec![], vec![true]));

        any_all_out(&mut ctx, &bool_false_in, &out);
        assert_tensor_eq!(out, tf_bool.make_default(vec![], vec![false]));

        any_all_out(&mut ctx, &bool_true_in, &out);
        assert_tensor_eq!(out, tf_bool.make_default(vec![], vec![true]));
    }

    // [spec:et:sem:op-any.torch.executor.native.any-all-out-fn/test]
    #[test]
    fn op_any_test_mismatched_dimensions_dies() {
        let tff = TensorFactory::<f32>::new();

        let in_ = tff.make_default(vec![2, 2], vec![0.0, 0.0, 1.0, 0.0]);
        let out = tff.ones_default(vec![1, 1]);

        let mut ctx = context();
        any_all_out(&mut ctx, &in_, &out);
        assert_ne!(ctx.failure_state(), Error::Ok);
    }

    // [spec:et:sem:op-any.torch.executor.native.any-all-out-fn/test]
    #[test]
    fn op_any_test_invalid_dtype_dies() {
        test_any_all_out_invalid_type::<f32>();
        test_any_all_out_invalid_type::<f64>();
    }

    // [spec:et:sem:op-any.torch.executor.native.any-all-out-fn/test]
    #[test]
    fn op_any_test_all_real_input_type_passes() {
        test_any_all_out::<u8>();
        test_any_all_out::<i8>();
        test_any_all_out::<i16>();
        test_any_all_out::<i32>();
        test_any_all_out::<i64>();
        test_any_all_out::<f32>();
        test_any_all_out::<f64>();
        test_any_all_out::<crate::runtime::core::portable_type::Half>();
        test_any_all_out::<crate::runtime::core::portable_type::BFloat16>();
    }

    // [spec:et:sem:op-any.torch.executor.native.any-dims-out-fn/test]
    #[test]
    fn op_any_test_smoke_test_dims() {
        let tf_bool = TensorFactory::<bool>::new();

        let self_ =
            tf_bool.make_default(vec![2, 3, 1], vec![true, false, true, true, false, false]);
        let dims: [i64; 3] = [0, 2, 0];
        let opt_dim_list: Option<ArrayRef<i64>> = Some(ArrayRef::from_raw_parts(dims.as_ptr(), 2));
        let keepdim = true;
        let out = tf_bool.zeros_default(vec![1, 3, 1]);
        let out_expected = tf_bool.make_default(vec![1, 3, 1], vec![true, false, true]);
        let mut ctx = context();
        any_dims_out(&mut ctx, &self_, opt_dim_list, keepdim, &out);
        assert_tensor_close!(out, out_expected);
    }

    // [spec:et:sem:op-any.torch.executor.native.any-out-fn/test]
    #[test]
    fn op_any_test_smoke_test() {
        let tf_bool = TensorFactory::<bool>::new();

        let self_ =
            tf_bool.make_default(vec![2, 3, 1], vec![true, false, true, true, false, false]);
        let dim: i64 = 0;
        let keepdim = false;
        let out = tf_bool.zeros_default(vec![3, 1]);
        let out_expected = tf_bool.make_default(vec![3, 1], vec![true, false, true]);
        let mut ctx = context();
        any_out(&mut ctx, &self_, dim, keepdim, &out);
        assert_tensor_close!(out, out_expected);
    }

    // [spec:et:sem:op-any.torch.executor.native.any-dims-out-fn/test]
    #[test]
    fn op_any_test_empty_input() {
        let tf = TensorFactory::<f32>::new();
        let tf_bool = TensorFactory::<bool>::new();

        let x = tf.make_default(vec![2, 0, 3], vec![]);
        let empty: [i64; 0] = [];
        let dim_list: Option<ArrayRef<i64>> = Some(ArrayRef::from_raw_parts(empty.as_ptr(), 0));
        let out = tf_bool.make_default(vec![2, 0, 3], vec![]);

        let mut ctx = context();
        any_dims_out(&mut ctx, &x, dim_list, true, &out);
        assert_tensor_close!(out, tf_bool.zeros_default(vec![2, 0, 3]));

        let out = tf_bool.ones_default(vec![2, 0, 3]);
        any_dims_out(&mut ctx, &x, dim_list, false, &out);
        assert_tensor_close!(out, tf_bool.zeros_default(vec![2, 0, 3]));

        let dims1: [i64; 1] = [1];
        let dim_list: Option<ArrayRef<i64>> = Some(ArrayRef::from_raw_parts(dims1.as_ptr(), 1));
        let out = tf_bool.ones_default(vec![2, 3]);
        any_dims_out(&mut ctx, &x, dim_list, false, &out);
        assert_tensor_close!(out, tf_bool.zeros_default(vec![2, 3]));

        let dims2: [i64; 1] = [2];
        let dim_list: Option<ArrayRef<i64>> = Some(ArrayRef::from_raw_parts(dims2.as_ptr(), 1));
        let out = tf_bool.make_default(vec![2, 0, 1], vec![]);
        any_dims_out(&mut ctx, &x, dim_list, true, &out);
        assert_tensor_close!(out, tf_bool.make_default(vec![2, 0, 1], vec![]));
    }

    // [spec:et:sem:op-any.torch.executor.native.any-dims-out-fn/test]
    #[test]
    fn op_any_test_any_dims_out_null_dim_list() {
        let tf_int = TensorFactory::<i32>::new();
        let tf_bool = TensorFactory::<bool>::new();

        let self_ = tf_int.make_default(vec![2, 6], vec![0, 2, 0, 3, 0, 1, 5, 0, 2, 0, 4, 0]);
        let opt_dim_list: Option<ArrayRef<i64>> = None;
        let keepdim = false;
        let out = tf_bool.zeros_default(vec![]);
        let out_expected = tf_bool.make_default(vec![], vec![true]);

        let mut ctx = context();
        any_dims_out(&mut ctx, &self_, opt_dim_list, keepdim, &out);
        assert_tensor_close!(out, out_expected);
    }

    // [spec:et:sem:op-any.torch.executor.native.any-dims-out-fn/test]
    #[test]
    fn op_any_test_any_dims_out_empty_dim_list() {
        let tf_int = TensorFactory::<i32>::new();
        let tf_bool = TensorFactory::<bool>::new();

        let self_ = tf_int.make_default(vec![2, 3], vec![0, 2, 0, 0, 1, 5]);
        let dims: [i64; 0] = [];
        let opt_dim_list: Option<ArrayRef<i64>> = Some(ArrayRef::from_raw_parts(dims.as_ptr(), 0));
        let keepdim = false;
        let out = tf_bool.zeros_default(vec![2, 3]);
        let out_expected =
            tf_bool.make_default(vec![2, 3], vec![false, true, false, false, true, true]);

        let mut ctx = context();
        any_dims_out(&mut ctx, &self_, opt_dim_list, keepdim, &out);
        assert_tensor_close!(out, out_expected);
    }
}
