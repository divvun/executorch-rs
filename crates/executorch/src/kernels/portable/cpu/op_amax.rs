//! Literal port of kernels/portable/cpu/op_amax.cpp.

use crate::kernels::portable::cpu::util::math_util::isnan_override;
#[cfg(not(feature = "aten"))]
use crate::kernels::portable::cpu::util::reduce_util::check_amin_amax_args;
use crate::kernels::portable::cpu::util::reduce_util::{
    ReduceOverDimListPlan, parallel_for_each_reduce_over_dim_list_output_index,
    resize_reduction_out,
};
use crate::runtime::core::array_ref::ArrayRef;
use crate::runtime::core::error::Error;
use crate::runtime::core::exec_aten::util::tensor_util::tensors_have_same_dim_order2;
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

// PORT-NOTE: C++ `Tensor& out` / returned `Tensor&` become `&'a Tensor` (interior
// mutation through the non-owning handle's raw pointer). `ArrayRef<int64_t>
// dim_list` maps to `ArrayRef<i64>`; the ported `resize_reduction_out`,
// `ReduceOverDimListPlan`, and `parallel_for_each_reduce_over_dim_list_output_index`
// take `Option<ArrayRef<i64>>`, so the plain `dim_list` is wrapped in `Some(...)`
// at those call sites (`ArrayRef` is `Copy`).

// [spec:et:def:op-amax.torch.executor.native.amax-out-fn]
// [spec:et:sem:op-amax.torch.executor.native.amax-out-fn]
pub fn amax_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    in_: &Tensor,
    dim_list: ArrayRef<i64>,
    keepdim: bool,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    // (void)ctx;

    // PORT-NOTE: `check_amin_amax_args` is under the C++ `#ifndef USE_ATEN_LIB`
    // block (the portable arg-checkers are absent in the ATen build), so gate to
    // match.
    #[cfg(not(feature = "aten"))]
    crate::et_kernel_check!(
        ctx,
        check_amin_amax_args(in_, dim_list, keepdim, out),
        InvalidArgument,
        out
    );

    crate::et_kernel_check!(
        ctx,
        resize_reduction_out(in_, &Some(dim_list), keepdim, out) == Error::Ok,
        InvalidArgument,
        out
    );

    crate::et_kernel_check!(
        ctx,
        tensors_have_same_dim_order2(in_, out),
        InvalidArgument,
        out
    );

    let plan = ReduceOverDimListPlan::new(in_, &Some(dim_list));

    let op_name = "amax.out";

    crate::et_switch_realhbbf16_types!(in_.scalar_type(), ctx, op_name, CTYPE, {
        let out_data: *mut CTYPE = out.mutable_data_ptr::<CTYPE>();
        let success = parallel_for_each_reduce_over_dim_list_output_index(
            in_,
            Some(dim_list),
            out,
            &|begin: i64, end: i64| {
                for out_ix in begin..end {
                    let out_ix = out_ix as usize;
                    let val: CTYPE = plan.execute::<CTYPE, _>(
                        |v: CTYPE, max_v: CTYPE| -> CTYPE {
                            if isnan_override(v) || v > max_v {
                                v
                            } else {
                                max_v
                            }
                        },
                        out_ix,
                    );
                    unsafe {
                        *out_data.add(out_ix) = val;
                    }
                }
            },
        );
        crate::et_kernel_check_msg!(ctx, success, Internal, out, "parallel_for failed");
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
        // Mirrors `OperatorTest::SetUp()`'s `runtime_init()` PAL initialization.
        crate::runtime::platform::platform::pal_init();
        KernelRuntimeContext::new(
            crate::extension::module::module::null_event_tracer(),
            core::ptr::null_mut::<MemoryAllocator>() as *mut dyn MemoryAllocatorBase,
        )
    }

    trait FromI64 {
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
    impl FromI64 for bool {
        fn from_i64(v: i64) -> Self {
            v != 0
        }
    }

    trait FromF64: Copy {
        fn from_f64(v: f64) -> Self;
        fn inf() -> Self;
        fn neg_inf() -> Self;
        fn nan() -> Self;
    }
    impl FromF64 for f32 {
        fn from_f64(v: f64) -> Self {
            v as f32
        }
        fn inf() -> Self {
            f32::INFINITY
        }
        fn neg_inf() -> Self {
            f32::NEG_INFINITY
        }
        fn nan() -> Self {
            f32::NAN
        }
    }
    impl FromF64 for f64 {
        fn from_f64(v: f64) -> Self {
            v
        }
        fn inf() -> Self {
            f64::INFINITY
        }
        fn neg_inf() -> Self {
            f64::NEG_INFINITY
        }
        fn nan() -> Self {
            f64::NAN
        }
    }
    impl FromF64 for Half {
        fn from_f64(v: f64) -> Self {
            Half::from_f32(v as f32)
        }
        fn inf() -> Self {
            Half::from_f32(f32::INFINITY)
        }
        fn neg_inf() -> Self {
            Half::from_f32(f32::NEG_INFINITY)
        }
        fn nan() -> Self {
            Half::from_f32(f32::NAN)
        }
    }
    impl FromF64 for BFloat16 {
        fn from_f64(v: f64) -> Self {
            BFloat16::from_f32(v as f32)
        }
        fn inf() -> Self {
            BFloat16::from_f32(f32::INFINITY)
        }
        fn neg_inf() -> Self {
            BFloat16::from_f32(f32::NEG_INFINITY)
        }
        fn nan() -> Self {
            BFloat16::from_f32(f32::NAN)
        }
    }

    fn make_i64<T: FromI64>(vals: &[i64]) -> Vec<T> {
        vals.iter().map(|&v| T::from_i64(v)).collect()
    }

    fn test_amax_out_invalid_dimensions<T>()
    where
        T: CppTypeToScalarType + FactoryValue + FromI64,
    {
        let tf = TensorFactory::<T>::new();
        let in_ = tf.make_default(
            vec![2, 3, 4],
            make_i64(&[
                0, 1, 2, 4, 4, 2, 1, 0, 1, 0, 4, 2, 4, 2, 1, 0, 0, 1, 2, 4, 1, 0, 4, 2,
            ]),
        );
        let out = tf.zeros_default(vec![2, 3, 1]);

        let dims_1: [i64; 1] = [3];
        let dim_list = ArrayRef::from_raw_parts(dims_1.as_ptr(), 1);
        let mut ctx = context();
        amax_out(&mut ctx, &in_, dim_list, true, &out);
        assert_ne!(ctx.failure_state(), Error::Ok);

        let dims_2: [i64; 2] = [2, 2];
        let dim_list = ArrayRef::from_raw_parts(dims_2.as_ptr(), 2);
        let mut ctx = context();
        amax_out(&mut ctx, &in_, dim_list, true, &out);
        assert_ne!(ctx.failure_state(), Error::Ok);
    }

    fn test_amax_out_invalid_shape<T>()
    where
        T: CppTypeToScalarType + FactoryValue + FromI64,
    {
        let tf = TensorFactory::<T>::new();
        let in_ = tf.make_default(
            vec![2, 3, 4],
            make_i64(&[
                0, 1, 2, 4, 4, 2, 1, 0, 1, 0, 4, 2, 4, 2, 1, 0, 0, 1, 2, 4, 1, 0, 4, 2,
            ]),
        );

        let out = tf.zeros_default(vec![2, 4]);
        let dims_1: [i64; 1] = [1];
        let dim_list = ArrayRef::from_raw_parts(dims_1.as_ptr(), 1);
        let mut ctx = context();
        amax_out(&mut ctx, &in_, dim_list, true, &out);
        assert_ne!(ctx.failure_state(), Error::Ok);

        let out = tf.zeros_default(vec![2, 1, 4]);
        let mut ctx = context();
        amax_out(&mut ctx, &in_, dim_list, false, &out);
        assert_ne!(ctx.failure_state(), Error::Ok);
    }

    fn test_amax_out_dtype<T>()
    where
        T: CppTypeToScalarType + FactoryValue + FromI64,
    {
        let tf = TensorFactory::<T>::new();
        let in_ = tf.make_default(
            vec![2, 3, 4],
            make_i64(&[
                0, 1, 2, 4, 4, 2, 1, 0, 1, 5, 4, 2, 4, 2, 1, 0, 5, 1, 2, 4, 7, 5, 4, 2,
            ]),
        );

        let out = tf.zeros_default(vec![2, 3, 1]);
        let dims_1: [i64; 1] = [2];
        let dim_list = ArrayRef::from_raw_parts(dims_1.as_ptr(), 1);
        let mut ctx = context();
        amax_out(&mut ctx, &in_, dim_list, true, &out);
        assert_tensor_close!(
            out,
            tf.make_default(vec![2, 3, 1], make_i64(&[4, 4, 5, 4, 5, 7]))
        );

        let out = tf.zeros_default(vec![2, 3]);
        amax_out(&mut ctx, &in_, dim_list, false, &out);
        assert_tensor_close!(
            out,
            tf.make_default(vec![2, 3], make_i64(&[4, 4, 5, 4, 5, 7]))
        );

        let out = tf.zeros_default(vec![1, 1, 4]);
        let dims_2: [i64; 2] = [0, 1];
        let dim_list = ArrayRef::from_raw_parts(dims_2.as_ptr(), 2);
        amax_out(&mut ctx, &in_, dim_list, true, &out);
        assert_tensor_close!(out, tf.make_default(vec![1, 1, 4], make_i64(&[7, 5, 4, 4])));

        let out = tf.zeros_default(vec![4]);
        amax_out(&mut ctx, &in_, dim_list, false, &out);
        assert_tensor_close!(out, tf.make_default(vec![4], make_i64(&[7, 5, 4, 4])));

        let out = tf.zeros_default(vec![2, 1, 4]);
        let dims_3: [i64; 1] = [-2];
        let dim_list = ArrayRef::from_raw_parts(dims_3.as_ptr(), 1);
        amax_out(&mut ctx, &in_, dim_list, true, &out);
        assert_tensor_close!(
            out,
            tf.make_default(vec![2, 1, 4], make_i64(&[4, 5, 4, 4, 7, 5, 4, 4]))
        );

        let in_ = tf.make_default(
            vec![2, 2, 4],
            make_i64(&[8, 7, 5, 4, 4, 3, 7, 9, 4, 2, 6, 8, 8, 7, 3, 4]),
        );
        let out = tf.zeros_default(vec![1, 1, 1]);
        let null_dim_list: ArrayRef<i64> = ArrayRef::new();
        amax_out(&mut ctx, &in_, null_dim_list, true, &out);
        assert_tensor_close!(out, tf.make_default(vec![1, 1, 1], make_i64(&[9])));

        let empty: [i64; 0] = [];
        let empty_dim_list = ArrayRef::from_raw_parts(empty.as_ptr(), 0);
        amax_out(&mut ctx, &in_, empty_dim_list, true, &out);
        assert_tensor_close!(out, tf.make_default(vec![1, 1, 1], make_i64(&[9])));

        let out = tf.zeros_default(vec![]);
        amax_out(&mut ctx, &in_, null_dim_list, false, &out);
        assert_tensor_close!(out, tf.make_default(vec![], make_i64(&[9])));

        amax_out(&mut ctx, &in_, empty_dim_list, false, &out);
        assert_tensor_close!(out, tf.make_default(vec![], make_i64(&[9])));
    }

    // PORT-NOTE: the C++ `test_amax_out_dtype<ScalarType::Bool>` specialization is
    // defined but never invoked by any TEST_F in this file (`AllRealInputOutputPasses`
    // dispatches over `ET_FORALL_REALHBBF16_TYPES`, which excludes Bool). Ported for
    // fidelity but not called, matching the C++.
    #[allow(dead_code)]
    fn test_amax_out_dtype_bool() {
        let tf_bool = TensorFactory::<bool>::new();
        let in_ = tf_bool.make_default(
            vec![2, 3, 4],
            vec![
                true, false, true, false, false, false, false, false, false, true, true, false,
                false, false, true, false, false, false, false, true, true, true, true, true,
            ],
        );

        let out = tf_bool.zeros_default(vec![2, 3, 1]);
        let dims: [i64; 1] = [-1];
        let dim_list = ArrayRef::from_raw_parts(dims.as_ptr(), 1);
        let mut ctx = context();
        amax_out(&mut ctx, &in_, dim_list, true, &out);
        assert_tensor_close!(
            out,
            tf_bool.make_default(vec![2, 3, 1], vec![true, false, true, true, true, true])
        );
    }

    fn test_amax_out_infinity_nan<T>()
    where
        T: CppTypeToScalarType + FactoryValue + FromF64,
    {
        let tf = TensorFactory::<T>::new();
        let inf = T::inf();
        let ninf = T::neg_inf();
        let nan = T::nan();
        let f = |v: f64| T::from_f64(v);
        let in_ = tf.make_default(
            vec![2, 3, 4],
            vec![
                f(0.0),
                f(1.0),
                f(2.0),
                inf,
                inf,
                ninf,
                f(1.0),
                f(0.0),
                nan,
                inf,
                ninf,
                f(2.0),
                nan,
                nan,
                f(1.0),
                f(0.0),
                f(0.0),
                inf,
                nan,
                f(4.0),
                f(1.0),
                nan,
                f(3.14),
                f(2.0),
            ],
        );

        let out = tf.zeros_default(vec![2, 3, 1]);
        let dims: [i64; 1] = [-1];
        let dim_list = ArrayRef::from_raw_parts(dims.as_ptr(), 1);
        let mut ctx = context();
        amax_out(&mut ctx, &in_, dim_list, true, &out);
        assert_tensor_close!(
            out,
            tf.make_default(vec![2, 3, 1], vec![inf, inf, nan, nan, nan, nan])
        );
    }

    // [spec:et:sem:op-amax.torch.executor.native.amax-out-fn/test]
    #[test]
    fn op_amax_test_invalid_dimension_list_dies() {
        test_amax_out_invalid_dimensions::<u8>();
        test_amax_out_invalid_dimensions::<i8>();
        test_amax_out_invalid_dimensions::<i16>();
        test_amax_out_invalid_dimensions::<i32>();
        test_amax_out_invalid_dimensions::<i64>();
        test_amax_out_invalid_dimensions::<f32>();
        test_amax_out_invalid_dimensions::<f64>();
        test_amax_out_invalid_dimensions::<bool>();
    }

    // [spec:et:sem:op-amax.torch.executor.native.amax-out-fn/test]
    #[test]
    fn op_amax_test_invalid_shape_dies() {
        test_amax_out_invalid_shape::<u8>();
        test_amax_out_invalid_shape::<i8>();
        test_amax_out_invalid_shape::<i16>();
        test_amax_out_invalid_shape::<i32>();
        test_amax_out_invalid_shape::<i64>();
        test_amax_out_invalid_shape::<f32>();
        test_amax_out_invalid_shape::<f64>();
        test_amax_out_invalid_shape::<bool>();
    }

    // [spec:et:sem:op-amax.torch.executor.native.amax-out-fn/test]
    #[test]
    fn op_amax_test_mismatched_dtypes_dies() {
        let tf_float = TensorFactory::<f32>::new();
        let tf_int = TensorFactory::<i32>::new();

        let in_ = tf_int.make_default(
            vec![2, 3, 4],
            vec![
                0, 1, 2, 4, 4, 2, 1, 0, 1, 0, 4, 2, 4, 2, 1, 0, 0, 1, 2, 4, 1, 0, 4, 2,
            ],
        );

        let out = tf_float.zeros_default(vec![2, 3, 1]);
        let dims_1: [i64; 1] = [2];
        let dim_list = ArrayRef::from_raw_parts(dims_1.as_ptr(), 1);

        let mut ctx = context();
        amax_out(&mut ctx, &in_, dim_list, true, &out);
        assert_ne!(ctx.failure_state(), Error::Ok);
    }

    // [spec:et:sem:op-amax.torch.executor.native.amax-out-fn/test]
    #[test]
    fn op_amax_test_all_real_input_output_passes() {
        test_amax_out_dtype::<u8>();
        test_amax_out_dtype::<i8>();
        test_amax_out_dtype::<i16>();
        test_amax_out_dtype::<i32>();
        test_amax_out_dtype::<i64>();
        test_amax_out_dtype::<f32>();
        test_amax_out_dtype::<f64>();
        test_amax_out_dtype::<Half>();
        test_amax_out_dtype::<BFloat16>();
    }

    // [spec:et:sem:op-amax.torch.executor.native.amax-out-fn/test]
    #[test]
    fn op_amax_test_infinity_and_nan_test() {
        test_amax_out_infinity_nan::<f32>();
        test_amax_out_infinity_nan::<f64>();
        test_amax_out_infinity_nan::<Half>();
        test_amax_out_infinity_nan::<BFloat16>();
    }
}
