//! Literal port of kernels/portable/cpu/op_cumsum.cpp.

use crate::kernels::portable::cpu::util::dtype_util::SupportedTensorDtypes;
use crate::kernels::portable::cpu::util::dtype_util::internal::{
    ComputeCast, LoadToComputeFn, get_load_to_compute_fn,
};
use crate::kernels::portable::cpu::util::kernel_ops_util::check_cumsum_args;
use crate::runtime::core::error::Error;
use crate::runtime::core::exec_aten::util::scalar_type_util::CppTypeToScalarType;
use crate::runtime::core::exec_aten::util::tensor_util::{
    getLeadingDims, getTrailingDims, resize_tensor_same_type, tensors_have_same_dim_order2,
};
use crate::runtime::core::portable_type::scalar_type::ScalarType;
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::core::portable_type::{BFloat16, Half};
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

// PORT-NOTE: the C++ accumulates with `CTYPE_OUT + CTYPE_OUT`, storing the result
// back into the `CTYPE_OUT` output on every step. For the integer types the two
// operands integer-promote to `int`, add, then the result narrows back to
// `CTYPE_OUT` on store — a well-defined truncation for unsigned and the usual
// two's-complement wrap for signed. `wrapping_add` reproduces exactly that
// promote-add-then-truncate-on-store behaviour (Rust's `+` would instead panic on
// overflow in debug builds). Bool relies on the same `(int)a + (int)b != 0`
// promotion. The reduced/standard floats use the native `+`. Modeled as a
// `CumsumAdd` trait so the body is generic over REALHBBF16.
trait CumsumAdd: Copy {
    fn cumsum_add(self, b: Self) -> Self;
}
macro_rules! impl_cumsum_add_wrapping {
    ($t:ty) => {
        impl CumsumAdd for $t {
            fn cumsum_add(self, b: Self) -> Self {
                self.wrapping_add(b)
            }
        }
    };
}
macro_rules! impl_cumsum_add_native {
    ($t:ty) => {
        impl CumsumAdd for $t {
            fn cumsum_add(self, b: Self) -> Self {
                self + b
            }
        }
    };
}
impl_cumsum_add_wrapping!(u8);
impl_cumsum_add_wrapping!(i8);
impl_cumsum_add_wrapping!(i16);
impl_cumsum_add_wrapping!(i32);
impl_cumsum_add_wrapping!(i64);
impl_cumsum_add_native!(f32);
impl_cumsum_add_native!(f64);
impl CumsumAdd for Half {
    fn cumsum_add(self, b: Self) -> Self {
        self + b
    }
}
impl CumsumAdd for BFloat16 {
    fn cumsum_add(self, b: Self) -> Self {
        self + b
    }
}
impl CumsumAdd for bool {
    fn cumsum_add(self, b: Self) -> Self {
        (self as i32 + b as i32) != 0
    }
}

/// Returns the cumulative sum of elements of input in the dimension dim.
// [spec:et:def:op-cumsum.torch.executor.native.cumsum-tensors-fn]
// [spec:et:sem:op-cumsum.torch.executor.native.cumsum-tensors-fn]
fn cumsum_tensors<CTYPE_OUT: CumsumAdd>(
    self_: &Tensor,
    load_self: LoadToComputeFn<CTYPE_OUT>,
    dim: i64,
    out: &Tensor,
) {
    if self_.numel() == 0 {
        return;
    }

    let input_data_base: *const u8 = self_.const_data_ptr::<u8>();
    let output_data_base: *mut CTYPE_OUT = out.mutable_data_ptr::<CTYPE_OUT>();

    if self_.dim() == 0 {
        unsafe {
            *output_data_base.add(0) =
                load_self(input_data_base.add(0) as *const core::ffi::c_void);
        }
        return;
    }

    let dim_size: usize = self_.size(dim as isize) as usize;
    let leading_dims: usize = getLeadingDims(self_, dim);
    let trailing_dims: usize = getTrailingDims(self_, dim);
    let element_size: usize = self_.element_size() as usize;

    for i in 0..leading_dims {
        let start_loc: usize = i * (trailing_dims * dim_size);

        for idx in 0..trailing_dims {
            unsafe {
                *output_data_base.add(start_loc + idx) =
                    load_self(input_data_base.add((start_loc + idx) * element_size)
                        as *const core::ffi::c_void);
            }
        }

        for j in 1..dim_size {
            let cur_round_base: usize = start_loc + j * trailing_dims;
            let prev_round_base: usize = start_loc + (j - 1) * trailing_dims;
            for idx in 0..trailing_dims {
                unsafe {
                    let loaded =
                        load_self(input_data_base.add((cur_round_base + idx) * element_size)
                            as *const core::ffi::c_void);
                    *output_data_base.add(cur_round_base + idx) =
                        loaded.cumsum_add(*output_data_base.add(prev_round_base + idx));
                }
            }
        }
    }
}

/// Returns the cumulative sum of elements of input in the dimension dim.
// [spec:et:def:op-cumsum.torch.executor.native.cumsum-out-fn]
// [spec:et:sem:op-cumsum.torch.executor.native.cumsum-out-fn]
#[executorch_macros::et_kernel("aten::cumsum.out")]
pub fn cumsum_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    self_: &Tensor,
    mut dim: i64,
    enforced_dtype: Option<ScalarType>,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    crate::et_kernel_check!(
        ctx,
        check_cumsum_args(self_, dim, enforced_dtype, out),
        InvalidArgument,
        out
    );

    crate::et_kernel_check!(
        ctx,
        tensors_have_same_dim_order2(self_, out),
        InvalidArgument,
        out
    );

    crate::et_kernel_check!(
        ctx,
        resize_tensor_same_type(out, self_.sizes()) == Error::Ok,
        InvalidArgument,
        out
    );

    dim = if self_.dim() == 0 {
        0
    } else if dim < 0 {
        dim + self_.dim() as i64
    } else {
        dim
    };

    let op_name = "cumsum.out";

    crate::et_switch_realhbbf16_types!(out.scalar_type(), ctx, op_name, CTYPE_OUT, {
        let load_self = get_load_to_compute_fn::<CTYPE_OUT>(
            ctx,
            self_,
            SupportedTensorDtypes::REALHBBF16,
            op_name,
        );
        cumsum_tensors::<CTYPE_OUT>(self_, load_self, dim, out);
    });

    out
}

// PORT-NOTE: `get_load_to_compute_fn::<CTYPE_OUT>` requires `CTYPE_OUT:
// ComputeCast + CppTypeToScalarType`. Those bounds hold for every REALHBBF16
// ctype the switch instantiates, so they are added to the local trait use sites
// via the switch body rather than the `cumsum_tensors` signature (which only
// needs `CumsumAdd`).
#[allow(dead_code)]
fn _assert_bounds<T: ComputeCast + CppTypeToScalarType>() {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assert_tensor_close;
    use crate::runtime::core::exec_aten::testing_util::tensor_factory::{
        FactoryValue, TensorFactory,
    };
    use crate::runtime::core::memory_allocator::{MemoryAllocator, MemoryAllocatorBase};
    use crate::runtime::core::tensor_shape_dynamism::TensorShapeDynamism;

    fn setup() {
        crate::runtime::platform::platform::pal_init();
    }

    fn context() -> KernelRuntimeContext<'static> {
        setup();
        KernelRuntimeContext::new(
            crate::extension::module::module::null_event_tracer(),
            core::ptr::null_mut::<MemoryAllocator>() as *mut dyn MemoryAllocatorBase,
        )
    }

    fn op_cumsum_out<'a, 'b>(
        self_: &Tensor,
        dim: i64,
        enforced_dtype: Option<ScalarType>,
        out: &'a Tensor<'b>,
    ) -> &'a Tensor<'b> {
        let mut ctx = context();
        cumsum_out(&mut ctx, self_, dim, enforced_dtype, out)
    }

    // Bridge from small integer literals to any real ctype used in the tests.
    trait FromI64: Copy {
        fn from_i64(v: i64) -> Self;
    }
    macro_rules! impl_from_i64_num {
        ($($t:ty),*) => {$(impl FromI64 for $t { fn from_i64(v: i64) -> Self { v as $t } })*};
    }
    impl_from_i64_num!(u8, i8, i16, i32, i64, f32, f64);

    fn test_cumsum_out_dtype<IN, OUT>()
    where
        IN: CppTypeToScalarType + FactoryValue + FromI64,
        OUT: CppTypeToScalarType + FactoryValue + FromI64,
    {
        let tf_in = TensorFactory::<IN>::new();
        let tf_out = TensorFactory::<OUT>::new();
        let in_ = tf_in.make_default(
            vec![2, 4],
            [0, 1, 2, 4, 8, 16, 32, 64]
                .iter()
                .map(|&v| IN::from_i64(v))
                .collect(),
        );

        let out = tf_out.zeros_default(vec![2, 4]);
        let enforced_dtype = Some(OUT::VALUE);
        op_cumsum_out(&in_, 1, enforced_dtype, &out);

        let expected = tf_out.make_default(
            vec![2, 4],
            [0, 1, 3, 7, 8, 24, 56, 120]
                .iter()
                .map(|&v| OUT::from_i64(v))
                .collect(),
        );

        assert_tensor_close!(out, expected);

        // negative dim should work
        op_cumsum_out(&in_, -1, enforced_dtype, &out);
        assert_tensor_close!(out, expected);

        op_cumsum_out(&in_, 0, enforced_dtype, &out);
        let expected = tf_out.make_default(
            vec![2, 4],
            [0, 1, 2, 4, 8, 17, 34, 68]
                .iter()
                .map(|&v| OUT::from_i64(v))
                .collect(),
        );
        assert_tensor_close!(out, expected);
    }

    fn test_cumsum_out_float<OUT>()
    where
        OUT: CppTypeToScalarType + FactoryValue + FromI64 + FromF64,
    {
        let tf_float = TensorFactory::<f32>::new();
        let tf_out = TensorFactory::<OUT>::new();

        let enforced_dtype = Some(OUT::VALUE);

        let in_ = tf_float.make_default(vec![1, 2], vec![1.0, f32::INFINITY]);
        let out = tf_out.zeros_default(vec![1, 2]);
        op_cumsum_out(&in_, 1, enforced_dtype, &out);
        assert_tensor_close!(
            out,
            tf_out.make_default(
                vec![1, 2],
                vec![OUT::from_i64(1), OUT::from_f64(f64::INFINITY)]
            )
        );

        let in_ = tf_float.make_default(vec![1, 2], vec![1.0, f32::NEG_INFINITY]);
        op_cumsum_out(&in_, 1, enforced_dtype, &out);
        assert_tensor_close!(
            out,
            tf_out.make_default(
                vec![1, 2],
                vec![OUT::from_i64(1), OUT::from_f64(f64::NEG_INFINITY)]
            )
        );

        let in_ = tf_float.make_default(vec![1, 2], vec![1.0, f32::NAN]);
        op_cumsum_out(&in_, 1, enforced_dtype, &out);
        assert_tensor_close!(
            out,
            tf_out.make_default(vec![1, 2], vec![OUT::from_i64(1), OUT::from_f64(f64::NAN)])
        );

        let in_ = tf_float.make_default(vec![1, 2], vec![f32::NEG_INFINITY, f32::INFINITY]);
        op_cumsum_out(&in_, 1, enforced_dtype, &out);
        assert_tensor_close!(
            out,
            tf_out.make_default(
                vec![1, 2],
                vec![OUT::from_f64(f64::NEG_INFINITY), OUT::from_f64(f64::NAN)]
            )
        );
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

    // PORT-NOTE: C++ `ET_SKIP_IF(is_aten, ...)`: never ATen, so the failure runs.
    // [spec:et:sem:op-cumsum.torch.executor.native.cumsum-out-fn/test]
    #[test]
    fn op_cumsum_out_test_mismatched_dimensions_dies() {
        let tff = TensorFactory::<f32>::new();
        let in_ = tff.make_default(vec![1, 3], vec![0.0, 1.0, 2.0]);
        let out = tff.zeros_default(vec![1, 3]);

        // Dim out of bounds
        let mut ctx = context();
        cumsum_out(&mut ctx, &in_, 3, None, &out);
        assert_ne!(ctx.failure_state(), Error::Ok);

        // wrong_out has incompatible dim
        let wrong_out = tff.zeros_default(vec![2, 10, 4]);
        let mut ctx = context();
        cumsum_out(&mut ctx, &in_, 1, None, &wrong_out);
        assert_ne!(ctx.failure_state(), Error::Ok);
    }

    // [spec:et:sem:op-cumsum.torch.executor.native.cumsum-out-fn/test]
    // Exercises check_cumsum_args: valid dim plus the enforced-dtype == out.dtype branch.
    // The prefix-sum values along dims 0/1/-1 pin cumsum_tensors' accumulation.
    // [spec:et:sem:kernel-ops-util.torch.executor.check-cumsum-args-fn/test]
    // [spec:et:sem:op-cumsum.torch.executor.native.cumsum-tensors-fn/test]
    #[test]
    fn op_cumsum_out_test_enforced_dtype_passes() {
        // ET_FORALL_REAL_TYPES cross ET_FORALL_REAL_TYPES
        macro_rules! for_out {
            ($in:ty) => {
                test_cumsum_out_dtype::<$in, u8>();
                test_cumsum_out_dtype::<$in, i8>();
                test_cumsum_out_dtype::<$in, i16>();
                test_cumsum_out_dtype::<$in, i32>();
                test_cumsum_out_dtype::<$in, i64>();
                test_cumsum_out_dtype::<$in, f32>();
                test_cumsum_out_dtype::<$in, f64>();
            };
        }
        for_out!(u8);
        for_out!(i8);
        for_out!(i16);
        for_out!(i32);
        for_out!(i64);
        for_out!(f32);
        for_out!(f64);
    }

    // [spec:et:sem:op-cumsum.torch.executor.native.cumsum-out-fn/test]
    #[test]
    fn op_cumsum_out_test_type_cast_corner_cases() {
        let tf_int = TensorFactory::<i32>::new();
        let tf_float = TensorFactory::<f32>::new();
        let tf_byte = TensorFactory::<u8>::new();

        // Cast floating point to int
        let in_ = tf_float.make_default(vec![1, 2], vec![1.1, 2.2]);
        let out = tf_int.zeros_default(vec![1, 2]);
        op_cumsum_out(&in_, 1, Some(ScalarType::Int), &out);
        assert_tensor_close!(out, tf_int.make_default(vec![1, 2], vec![1, 3]));

        // Cast negative values to unsigned type
        let in_ = tf_int.make_default(vec![1, 2], vec![-1, -2]);
        let out = tf_byte.zeros_default(vec![1, 2]);
        op_cumsum_out(&in_, 1, Some(ScalarType::Byte), &out);
        assert_tensor_close!(out, tf_byte.make_default(vec![1, 2], vec![255, 253]));

        // Cast negative float values to int, float should rounding toward zero
        let in_ = tf_float.make_default(vec![1, 2], vec![-1.9, -2.9]);
        let out = tf_int.zeros_default(vec![1, 2]);
        op_cumsum_out(&in_, 1, Some(ScalarType::Int), &out);
        assert_tensor_close!(out, tf_int.make_default(vec![1, 2], vec![-1, -3]));
    }

    // [spec:et:sem:op-cumsum.torch.executor.native.cumsum-out-fn/test]
    #[test]
    fn op_cumsum_out_test_float_specific_test() {
        test_cumsum_out_float::<f32>();
        test_cumsum_out_float::<f64>();
    }

    // [spec:et:sem:op-cumsum.torch.executor.native.cumsum-out-fn/test]
    #[test]
    fn op_cumsum_out_test_simple_generated_case() {
        let tf = TensorFactory::<f32>::new();
        let x = tf.make_default(vec![10, 10], vec![1.0f32; 100]);
        let mut expected_data: Vec<f32> = Vec::with_capacity(100);
        for _ in 0..10 {
            for j in 1..=10 {
                expected_data.push(j as f32);
            }
        }
        let expected_result = tf.make_default(vec![10, 10], expected_data);
        let out = tf.zeros_default(vec![10, 10]);
        op_cumsum_out(&x, 1, Some(ScalarType::Float), &out);
        assert_tensor_close!(out, expected_result);
    }

    // [spec:et:sem:op-cumsum.torch.executor.native.cumsum-out-fn/test]
    #[test]
    fn op_cumsum_out_test_dynamic_shape_upper_bound_same_as_expected() {
        let tf = TensorFactory::<f32>::new();
        let x = tf.make_default(
            vec![3, 2],
            vec![
                0.04876953363418579,
                0.816348671913147,
                0.44230276346206665,
                0.2767965793609619,
                0.8998266458511353,
                0.09595239162445068,
            ],
        );
        let expected_result = tf.make_default(
            vec![3, 2],
            vec![
                0.04876953363418579,
                0.8651182055473328,
                0.44230276346206665,
                0.7190993428230286,
                0.8998266458511353,
                0.9957790374755859,
            ],
        );
        let out = tf.zeros(vec![3, 2], TensorShapeDynamism::DYNAMIC_BOUND);
        op_cumsum_out(&x, 1, Some(ScalarType::Float), &out);
        assert_tensor_close!(out, expected_result);
    }

    // [spec:et:sem:op-cumsum.torch.executor.native.cumsum-out-fn/test]
    #[test]
    fn op_cumsum_out_test_dynamic_shape_upper_bound_larger_than_expected() {
        let tf = TensorFactory::<f32>::new();
        let x = tf.make_default(
            vec![3, 2],
            vec![
                0.04876953363418579,
                0.816348671913147,
                0.44230276346206665,
                0.2767965793609619,
                0.8998266458511353,
                0.09595239162445068,
            ],
        );
        let expected_result = tf.make_default(
            vec![3, 2],
            vec![
                0.04876953363418579,
                0.8651182055473328,
                0.44230276346206665,
                0.7190993428230286,
                0.8998266458511353,
                0.9957790374755859,
            ],
        );
        let out = tf.zeros(vec![10, 10], TensorShapeDynamism::DYNAMIC_BOUND);
        op_cumsum_out(&x, 1, Some(ScalarType::Float), &out);
        assert_tensor_close!(out, expected_result);
    }

    // DISABLED: Dynamic shape unbound not supported
    // [spec:et:sem:op-cumsum.torch.executor.native.cumsum-out-fn/test]
    #[test]
    #[ignore = "DISABLED_DynamicShapeUnbound: dynamic shape unbound not supported"]
    fn op_cumsum_out_test_disabled_dynamic_shape_unbound() {
        let tf = TensorFactory::<f32>::new();
        let x = tf.make_default(
            vec![3, 2],
            vec![
                0.04876953363418579,
                0.816348671913147,
                0.44230276346206665,
                0.2767965793609619,
                0.8998266458511353,
                0.09595239162445068,
            ],
        );
        let expected_result = tf.make_default(
            vec![3, 2],
            vec![
                0.04876953363418579,
                0.8651182055473328,
                0.44230276346206665,
                0.7190993428230286,
                0.8998266458511353,
                0.9957790374755859,
            ],
        );
        let out = tf.zeros(vec![1, 1], TensorShapeDynamism::DYNAMIC_UNBOUND);
        op_cumsum_out(&x, 1, Some(ScalarType::Float), &out);
        assert_tensor_close!(out, expected_result);
    }
}
