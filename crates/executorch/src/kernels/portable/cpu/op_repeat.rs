//! Literal port of kernels/portable/cpu/op_repeat.cpp.

use crate::kernels::portable::cpu::util::repeat_util::repeat_tensor;
use crate::runtime::core::array_ref::ArrayRef;
use crate::runtime::core::error::Error;
use crate::runtime::core::exec_aten::util::tensor_util::{
    K_TENSOR_DIMENSION_LIMIT, resize_tensor_same_type, tensor_is_default_dim_order,
    tensors_have_same_dim_order2,
};
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::core::portable_type::tensor_impl::SizesType;
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

// PORT-NOTE: the crate-level `et_check_or_return_false!` (runtime/core/error.rs)
// drops all caller-supplied format arguments after the leading literal. This
// local override mirrors the C++ `ET_CHECK_OR_RETURN_FALSE` faithfully (same as
// repeat_util.rs).
macro_rules! et_check_or_return_false {
    ($cond:expr, $fmt:literal $(, $($arg:tt)*)?) => {{
        if !($cond) {
            $crate::et_log!(
                Error,
                ::core::concat!("Check failed ({}): ", $fmt),
                ::core::stringify!($cond)
                $(, $($arg)*)?
            );
            return false;
        }
    }};
}

// PORT-NOTE: `ET_LOG_AND_RETURN_IF_FALSE(cond)` expands to
// `ET_CHECK_OR_RETURN_FALSE(cond, "")`.
macro_rules! et_log_and_return_if_false {
    ($cond:expr) => {
        et_check_or_return_false!($cond, "")
    };
}

// [spec:et:def:op-repeat.torch.executor.native.calculate-output-size-fn]
// [spec:et:sem:op-repeat.torch.executor.native.calculate-output-size-fn]
///
/// # Safety
/// `out_sizes_ptr` must point to at least `repeats.size()` valid `SizesType`
/// elements.
unsafe fn calculate_output_size(
    self_sizes: &ArrayRef<SizesType>,
    repeats: &ArrayRef<i64>,
    out_sizes_ptr: *mut SizesType,
) -> bool {
    et_log_and_return_if_false!(repeats.size() < K_TENSOR_DIMENSION_LIMIT);

    et_check_or_return_false!(
        repeats.size() >= self_sizes.size(),
        "Repeats vector size is {} must be >= self_sizes {}.",
        repeats.size(),
        self_sizes.size()
    );

    let mut i: usize = 0;
    while i < (repeats.size() - self_sizes.size()) {
        unsafe {
            *out_sizes_ptr.add(i) = *repeats.at(i) as SizesType;
        }
        i += 1;
    }
    let mut j: usize = 0;
    while i < repeats.size() {
        unsafe {
            *out_sizes_ptr.add(i) = *repeats.at(i) as SizesType * *self_sizes.at(j);
        }
        j += 1;
        i += 1;
    }

    true
}

// repeat.out(Tensor self, int[] repeats, *, Tensor(a!) out) -> Tensor(a!)
// [spec:et:def:op-repeat.torch.executor.native.repeat-out-fn]
// [spec:et:sem:op-repeat.torch.executor.native.repeat-out-fn]
#[executorch_macros::et_kernel("aten::repeat.out")]
pub fn repeat_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    self_: &Tensor,
    repeats: ArrayRef<i64>,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    let mut expected_output_size: [SizesType; K_TENSOR_DIMENSION_LIMIT] =
        [0; K_TENSOR_DIMENSION_LIMIT];

    crate::et_kernel_check!(
        ctx,
        unsafe {
            calculate_output_size(&self_.sizes(), &repeats, expected_output_size.as_mut_ptr())
        },
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
        tensor_is_default_dim_order(self_),
        InvalidArgument,
        out
    );

    // Resize for dynamic shape
    crate::et_kernel_check_msg!(
        ctx,
        resize_tensor_same_type(
            out,
            ArrayRef::from_raw_parts(expected_output_size.as_ptr(), repeats.size())
        ) == Error::Ok,
        InvalidArgument,
        out,
        "Failed to resize output tensor."
    );

    crate::et_kernel_check!(
        ctx,
        repeat_tensor(self_, repeats, out) == Error::Ok,
        InvalidArgument,
        out
    );

    out
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
        crate::runtime::platform::runtime::runtime_init();
        KernelRuntimeContext::new(
            crate::extension::module::module::null_event_tracer(),
            core::ptr::null_mut::<MemoryAllocator>() as *mut dyn MemoryAllocatorBase,
        )
    }

    fn op_repeat_out<'a, 'b>(
        self_: &Tensor,
        repeats: ArrayRef<i64>,
        out: &'a Tensor<'b>,
    ) -> &'a Tensor<'b> {
        let mut ctx = context();
        repeat_out(&mut ctx, self_, repeats, out)
    }

    macro_rules! et_expect_kernel_failure {
        ($ctx:expr, $stmt:expr) => {{
            let _ = $stmt;
            assert_ne!(
                $ctx.failure_state(),
                Error::Ok,
                "Expected kernel failure but found success."
            );
        }};
    }

    fn ir(v: &[i64]) -> ArrayRef<i64> {
        ArrayRef::from_raw_parts(v.as_ptr(), v.len())
    }

    // PORT-NOTE: local `from_i32` bridge for the element types used across the
    // repeat suites (mirrors the op_constant_pad_nd.rs test helper).
    trait FromI32: Copy {
        fn from_i32(v: i32) -> Self;
    }
    macro_rules! impl_from_i32 {
        ($($t:ty),*) => {$(impl FromI32 for $t { fn from_i32(v: i32) -> Self { v as $t } })*};
    }
    impl_from_i32!(u8, i8, i16, i32, i64, f32, f64);
    impl FromI32 for Half {
        fn from_i32(v: i32) -> Self {
            Half::from_f32(v as f32)
        }
    }
    impl FromI32 for BFloat16 {
        fn from_i32(v: i32) -> Self {
            BFloat16::from_f32(v as f32)
        }
    }
    impl FromI32 for bool {
        fn from_i32(v: i32) -> Self {
            v != 0
        }
    }

    fn run_dtype_tests<T>()
    where
        T: CppTypeToScalarType + FactoryValue + FromI32,
    {
        let tf = TensorFactory::<T>::new();
        #[rustfmt::skip]
        let x = tf.make_default(
            vec![2, 2],
            [
                0, 1,
                2, 3,
            ]
            .iter()
            .map(|&v| T::from_i32(v))
            .collect(),
        );
        let repeats_vec: Vec<i64> = vec![3, 3, 3];
        let repeats = ir(&repeats_vec);

        let out = tf.zeros_default(vec![3, 6, 6]);

        #[rustfmt::skip]
        let expected = tf.make_default(
            vec![3, 6, 6],
            [
                // [0, :, :]
                0, 1, 0, 1, 0, 1,
                2, 3, 2, 3, 2, 3,
                0, 1, 0, 1, 0, 1,
                2, 3, 2, 3, 2, 3,
                0, 1, 0, 1, 0, 1,
                2, 3, 2, 3, 2, 3,

                // [1, :, :]
                0, 1, 0, 1, 0, 1,
                2, 3, 2, 3, 2, 3,
                0, 1, 0, 1, 0, 1,
                2, 3, 2, 3, 2, 3,
                0, 1, 0, 1, 0, 1,
                2, 3, 2, 3, 2, 3,

                // [2, :, :]
                0, 1, 0, 1, 0, 1,
                2, 3, 2, 3, 2, 3,
                0, 1, 0, 1, 0, 1,
                2, 3, 2, 3, 2, 3,
                0, 1, 0, 1, 0, 1,
                2, 3, 2, 3, 2, 3,
            ]
            .iter()
            .map(|&v| T::from_i32(v))
            .collect(),
        );

        let ret = op_repeat_out(&x, repeats, &out);
        assert_tensor_eq!(*ret, out);
        assert_tensor_eq!(*ret, expected);
    }

    // PORT-NOTE: C++ `ET_SKIP_IF(is_aten, ...)`; the port is never the ATen
    // kernel, so the body always runs.
    // [spec:et:sem:op-repeat.torch.executor.native.repeat-out-fn/test]
    // [spec:et:sem:op-repeat.torch.executor.native.calculate-output-size-fn/test]
    // also verifies the repeat_util path end-to-end: [2,2] tiled [3,3,3] -> [3,6,6]
    // exercises check_repeat_args (valid), repeat_tensor (stride/limit setup and
    // trailing m>n memcpy), repeat_internal (multi-slot copy), and
    // compute_access_offset (per-dim offset accumulation).
    // [spec:et:sem:repeat-util.torch.executor.check-repeat-args-fn/test]
    // [spec:et:sem:repeat-util.torch.executor.repeat-tensor-fn/test]
    // [spec:et:sem:repeat-util.torch.executor.repeat-internal-fn/test]
    // [spec:et:sem:repeat-util.torch.executor.compute-access-offset-fn/test]
    #[test]
    fn all_dtypes_supported() {
        run_dtype_tests::<u8>();
        run_dtype_tests::<i8>();
        run_dtype_tests::<i16>();
        run_dtype_tests::<i32>();
        run_dtype_tests::<i64>();
        run_dtype_tests::<f32>();
        run_dtype_tests::<f64>();
        run_dtype_tests::<bool>();
    }

    // [spec:et:sem:op-repeat.torch.executor.native.repeat-out-fn/test]
    // [spec:et:sem:op-repeat.torch.executor.native.calculate-output-size-fn/test]
    #[test]
    fn empty_input_supported() {
        let tf = TensorFactory::<i32>::new();

        let x = tf.make_default(vec![3, 0, 2], vec![]);

        let repeats_vec: Vec<i64> = vec![3, 4, 5, 6];
        let repeats = ir(&repeats_vec);

        let out = tf.ones_default(vec![3, 12, 0, 12]);
        let expected = tf.make_default(vec![3, 12, 0, 12], vec![]);

        let ret = op_repeat_out(&x, repeats, &out);
        assert_tensor_eq!(*ret, out);
        assert_tensor_eq!(*ret, expected);
    }

    // [spec:et:sem:op-repeat.torch.executor.native.repeat-out-fn/test]
    // [spec:et:sem:op-repeat.torch.executor.native.calculate-output-size-fn/test]
    #[test]
    fn zero_dim_input_supported() {
        let tf = TensorFactory::<i32>::new();

        let x = tf.make_default(vec![], vec![5]);

        let repeats_vec: Vec<i64> = vec![3, 4];
        let repeats = ir(&repeats_vec);

        let out = tf.ones_default(vec![3, 4]);

        #[rustfmt::skip]
        let expected = tf.make_default(
            vec![3, 4],
            vec![
                5, 5, 5, 5,
                5, 5, 5, 5,
                5, 5, 5, 5,
            ],
        );

        let ret = op_repeat_out(&x, repeats, &out);
        assert_tensor_eq!(*ret, out);
        assert_tensor_eq!(*ret, expected);
    }

    // [spec:et:sem:op-repeat.torch.executor.native.repeat-out-fn/test]
    // [spec:et:sem:op-repeat.torch.executor.native.calculate-output-size-fn/test]
    #[test]
    fn zero_repeat_regular_input_supported() {
        let tf = TensorFactory::<i32>::new();
        let x = tf.make_default(vec![3, 2], vec![0, 1, 2, 3, 4, 5]);

        let repeats_vec: Vec<i64> = vec![3, 0, 6];
        let repeats = ir(&repeats_vec);

        let out = tf.ones_default(vec![3, 0, 12]);
        let expected = tf.make_default(vec![3, 0, 12], vec![]);

        let ret = op_repeat_out(&x, repeats, &out);
        assert_tensor_eq!(*ret, out);
        assert_tensor_eq!(*ret, expected);
    }

    // [spec:et:sem:op-repeat.torch.executor.native.repeat-out-fn/test]
    // [spec:et:sem:op-repeat.torch.executor.native.calculate-output-size-fn/test]
    #[test]
    fn zero_repeat_zero_dim_input_supported() {
        let tf = TensorFactory::<i32>::new();

        let x = tf.make_default(vec![], vec![5]);

        let repeats_vec: Vec<i64> = vec![3, 0, 6];
        let repeats = ir(&repeats_vec);

        let out = tf.ones_default(vec![3, 0, 6]);
        let expected = tf.make_default(vec![3, 0, 6], vec![]);

        let ret = op_repeat_out(&x, repeats, &out);
        assert_tensor_eq!(*ret, out);
        assert_tensor_eq!(*ret, expected);
    }

    // [spec:et:sem:op-repeat.torch.executor.native.repeat-out-fn/test]
    // [spec:et:sem:op-repeat.torch.executor.native.calculate-output-size-fn/test]
    #[test]
    fn repeat_too_short_die() {
        let tf = TensorFactory::<i32>::new();

        let x = tf.make_default(vec![3, 2], vec![0, 1, 2, 3, 4, 5]);

        // The length of repeat vector shall not be shorter than x.dim().
        let repeats_vec: Vec<i64> = vec![3];
        let repeats = ir(&repeats_vec);

        let out = tf.ones_default(vec![3, 0, 12]);

        let mut ctx = context();
        et_expect_kernel_failure!(ctx, repeat_out(&mut ctx, &x, repeats, &out));
    }

    // [spec:et:sem:op-repeat.torch.executor.native.repeat-out-fn/test]
    // [spec:et:sem:op-repeat.torch.executor.native.calculate-output-size-fn/test]
    #[test]
    fn negative_repeat_die() {
        let tf = TensorFactory::<i32>::new();

        let x = tf.make_default(vec![3, 2], vec![0, 1, 2, 3, 4, 5]);

        // Try to create tensor with negative shape, die.
        let repeats_vec: Vec<i64> = vec![3, -1];
        let repeats = ir(&repeats_vec);

        let out = tf.ones_default(vec![3, 1]);

        let mut ctx = context();
        et_expect_kernel_failure!(ctx, repeat_out(&mut ctx, &x, repeats, &out));
    }

    // PORT-NOTE: C++ `ET_SKIP_IF(is_aten, ...)`; the port is never the ATen kernel.
    // [spec:et:sem:op-repeat.torch.executor.native.repeat-out-fn/test]
    #[test]
    fn wrong_output_shape_die() {
        let tf = TensorFactory::<i32>::new();

        let x = tf.ones_default(vec![3, 2]);

        let repeats_vec: Vec<i64> = vec![3, 5, 6];
        let repeats = ir(&repeats_vec);

        // The size of output shall be [3, 15, 12].
        let out = tf.ones_default(vec![3, 5, 12]);

        let mut ctx = context();
        et_expect_kernel_failure!(ctx, repeat_out(&mut ctx, &x, repeats, &out));
    }

    // [spec:et:sem:op-repeat.torch.executor.native.repeat-out-fn/test]
    #[test]
    fn output_dtype_mismatched_die() {
        let tf_in = TensorFactory::<i32>::new();
        let tf_out = TensorFactory::<f32>::new();

        let x = tf_in.ones_default(vec![3, 3]);

        let repeats_vec: Vec<i64> = vec![7, 5, 6];
        let repeats = ir(&repeats_vec);

        let out = tf_out.ones_default(vec![7, 15, 18]);

        let mut ctx = context();
        et_expect_kernel_failure!(ctx, repeat_out(&mut ctx, &x, repeats, &out));
    }

    // Right now we only support the dimension of input and output no larger
    // than 16.
    // PORT-NOTE: C++ `ET_SKIP_IF(is_aten, ...)`; the port is never the ATen kernel.
    // [spec:et:sem:op-repeat.torch.executor.native.repeat-out-fn/test]
    // [spec:et:sem:op-repeat.torch.executor.native.calculate-output-size-fn/test]
    #[test]
    fn too_many_dimensions_dies() {
        let tf = TensorFactory::<i32>::new();

        let x = tf.ones_default(vec![3, 2]);

        let repeats_vec: Vec<i64> = vec![1; 17];
        let repeats = ir(&repeats_vec);

        // The size of output shall be [1, 1, .. total 15 * 1 .. , 1, 3, 2].
        let mut output_shape: Vec<SizesType> = vec![1; 15];
        output_shape.push(3);
        output_shape.push(2);
        let out = tf.ones_default(output_shape);

        let mut ctx = context();
        et_expect_kernel_failure!(ctx, repeat_out(&mut ctx, &x, repeats, &out));
    }

    // [spec:et:sem:op-repeat.torch.executor.native.repeat-out-fn/test]
    // [spec:et:sem:op-repeat.torch.executor.native.calculate-output-size-fn/test]
    #[test]
    fn upper_bound_out_tensor() {
        let tf = TensorFactory::<f32>::new();
        #[rustfmt::skip]
        let x = tf.make_default(
            vec![2, 2],
            vec![
                0.0, 1.0,
                2.0, 3.0,
            ],
        );
        let repeats_vec: Vec<i64> = vec![3, 3, 3];
        let repeats = ir(&repeats_vec);

        let out = tf.zeros(vec![5, 9, 9], TensorShapeDynamism::DYNAMIC_BOUND);

        #[rustfmt::skip]
        let expected = tf.make_default(
            vec![3, 6, 6],
            vec![
                // [0, :, :]
                0.0, 1.0, 0.0, 1.0, 0.0, 1.0,
                2.0, 3.0, 2.0, 3.0, 2.0, 3.0,
                0.0, 1.0, 0.0, 1.0, 0.0, 1.0,
                2.0, 3.0, 2.0, 3.0, 2.0, 3.0,
                0.0, 1.0, 0.0, 1.0, 0.0, 1.0,
                2.0, 3.0, 2.0, 3.0, 2.0, 3.0,

                // [1, :, :]
                0.0, 1.0, 0.0, 1.0, 0.0, 1.0,
                2.0, 3.0, 2.0, 3.0, 2.0, 3.0,
                0.0, 1.0, 0.0, 1.0, 0.0, 1.0,
                2.0, 3.0, 2.0, 3.0, 2.0, 3.0,
                0.0, 1.0, 0.0, 1.0, 0.0, 1.0,
                2.0, 3.0, 2.0, 3.0, 2.0, 3.0,

                // [2, :, :]
                0.0, 1.0, 0.0, 1.0, 0.0, 1.0,
                2.0, 3.0, 2.0, 3.0, 2.0, 3.0,
                0.0, 1.0, 0.0, 1.0, 0.0, 1.0,
                2.0, 3.0, 2.0, 3.0, 2.0, 3.0,
                0.0, 1.0, 0.0, 1.0, 0.0, 1.0,
                2.0, 3.0, 2.0, 3.0, 2.0, 3.0,
            ],
        );

        let ret = op_repeat_out(&x, repeats, &out);
        assert_tensor_eq!(*ret, out);
        assert_tensor_eq!(*ret, expected);
    }

    // [spec:et:sem:op-repeat.torch.executor.native.repeat-out-fn/test]
    // [spec:et:sem:op-repeat.torch.executor.native.calculate-output-size-fn/test]
    #[test]
    fn dynamic_shape_upper_bound_same_as_expected() {
        let tf = TensorFactory::<i32>::new();

        let x = tf.make_default(vec![1, 2], vec![4, 9]);
        let expected = tf.make_default(
            vec![4, 4],
            vec![4, 9, 4, 9, 4, 9, 4, 9, 4, 9, 4, 9, 4, 9, 4, 9],
        );

        let repeatsv: Vec<i64> = vec![4, 2];
        let repeats = ir(&repeatsv);

        let out = tf.zeros(vec![4, 4], TensorShapeDynamism::DYNAMIC_BOUND);
        op_repeat_out(&x, repeats, &out);
        assert_tensor_eq!(out, expected);
    }

    // [spec:et:sem:op-repeat.torch.executor.native.repeat-out-fn/test]
    // [spec:et:sem:op-repeat.torch.executor.native.calculate-output-size-fn/test]
    #[test]
    fn dynamic_shape_upper_bound_larger_than_expected() {
        let tf = TensorFactory::<i32>::new();

        let x = tf.make_default(vec![1, 2], vec![4, 9]);
        let expected = tf.make_default(
            vec![4, 4],
            vec![4, 9, 4, 9, 4, 9, 4, 9, 4, 9, 4, 9, 4, 9, 4, 9],
        );

        let repeatsv: Vec<i64> = vec![4, 2];
        let repeats = ir(&repeatsv);

        let out = tf.zeros(vec![10, 10], TensorShapeDynamism::DYNAMIC_BOUND);
        op_repeat_out(&x, repeats, &out);
        assert_tensor_eq!(out, expected);
    }

    // PORT-NOTE: C++ `ET_SKIP_IF(!output_resize, ...)`. Dynamic unbound resize is
    // not supported by the portable path; ported and `#[ignore]`d to preserve the
    // suite.
    // [spec:et:sem:op-repeat.torch.executor.native.repeat-out-fn/test]
    #[test]
    #[ignore]
    fn dynamic_shape_unbound() {
        let tf = TensorFactory::<i32>::new();

        let x = tf.make_default(vec![1, 2], vec![4, 9]);
        let expected = tf.make_default(
            vec![4, 4],
            vec![4, 9, 4, 9, 4, 9, 4, 9, 4, 9, 4, 9, 4, 9, 4, 9],
        );

        let repeatsv: Vec<i64> = vec![4, 2];
        let repeats = ir(&repeatsv);

        let out = tf.zeros(vec![1, 1], TensorShapeDynamism::DYNAMIC_UNBOUND);
        op_repeat_out(&x, repeats, &out);
        assert_tensor_eq!(out, expected);
    }
}
