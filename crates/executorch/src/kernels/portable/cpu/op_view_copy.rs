//! Literal port of kernels/portable/cpu/op_view_copy.cpp.

use crate::kernels::portable::cpu::util::copy_ops_util::{
    check_view_copy_args, get_view_copy_target_size,
};
use crate::runtime::core::array_ref::ArrayRef;
use crate::runtime::core::error::Error;
use crate::runtime::core::exec_aten::util::tensor_util::{
    resize_tensor, tensor_is_default_dim_order, tensors_have_same_dim_order2,
};
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::core::portable_type::tensor_impl::SizesType;
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

// view_copy.out(Tensor self, int[] size, *, Tensor(a!) out) -> Tensor(a!)
// [spec:et:def:op-view-copy.torch.executor.native.view-copy-out-fn]
// [spec:et:sem:op-view-copy.torch.executor.native.view-copy-out-fn]
pub fn view_copy_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    self_: &Tensor,
    size_int64_t: ArrayRef<i64>,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    let _ = &ctx;

    let mut expected_output_size: [SizesType; 16] = [0; 16];
    crate::et_kernel_check!(
        ctx,
        unsafe {
            get_view_copy_target_size(
                self_,
                size_int64_t,
                out.dim() as i64,
                expected_output_size.as_mut_ptr(),
            )
        },
        InvalidArgument,
        out
    );

    // Resize for dynamic shape
    crate::et_kernel_check_msg!(
        ctx,
        resize_tensor(
            out,
            ArrayRef::from_raw_parts(expected_output_size.as_ptr(), out.dim() as usize)
        ) == Error::Ok,
        InvalidArgument,
        out,
        "Failed to resize output tensor."
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

    crate::et_kernel_check!(
        ctx,
        check_view_copy_args(self_, size_int64_t, out),
        InvalidArgument,
        out
    );

    if self_.nbytes() > 0 {
        unsafe {
            core::ptr::copy_nonoverlapping(
                self_.const_data_ptr::<u8>(),
                out.mutable_data_ptr::<u8>(),
                self_.nbytes(),
            );
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::core::exec_aten::testing_util::tensor_factory::{
        FactoryValue, TensorFactory,
    };
    use crate::runtime::core::exec_aten::testing_util::tensor_util::{
        tensor_data_is_close, tensors_are_close,
    };
    use crate::runtime::core::exec_aten::util::scalar_type_util::CppTypeToScalarType;
    use crate::runtime::core::memory_allocator::{MemoryAllocator, MemoryAllocatorBase};
    use crate::runtime::core::tensor_shape_dynamism::TensorShapeDynamism;

    fn context() -> KernelRuntimeContext<'static> {
        crate::runtime::platform::platform::pal_init();
        KernelRuntimeContext::new(
            crate::extension::module::module::null_event_tracer(),
            core::ptr::null_mut::<MemoryAllocator>() as *mut dyn MemoryAllocatorBase,
        )
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

    fn op_view_copy_out<'a, 'b>(
        ctx: &mut KernelRuntimeContext,
        self_: &Tensor,
        size: &[i64],
        out: &'a Tensor<'b>,
    ) -> &'a Tensor<'b> {
        view_copy_out(
            ctx,
            self_,
            ArrayRef::from_raw_parts(size.as_ptr(), size.len()),
            out,
        )
    }

    trait FromI64 {
        fn from_i64(v: i64) -> Self;
    }
    macro_rules! impl_from_i64_num {
        ($($t:ty),*) => {$(impl FromI64 for $t { fn from_i64(v: i64) -> Self { v as $t } })*};
    }
    impl_from_i64_num!(u8, i8, i16, i32, i64, f32, f64);
    impl FromI64 for bool {
        fn from_i64(v: i64) -> Self {
            v != 0
        }
    }

    fn make_i64<T: FromI64>(vals: &[i64]) -> Vec<T> {
        vals.iter().map(|&v| T::from_i64(v)).collect()
    }

    fn vector_32_to_64(v: &[i32]) -> Vec<i64> {
        v.iter().map(|&x| x as i64).collect()
    }

    fn run_view_test_cases<T>(input: &Tensor, out_shapes: &[Vec<SizesType>])
    where
        T: CppTypeToScalarType + FactoryValue,
    {
        let tf = TensorFactory::<T>::new();
        for size in out_shapes {
            let out = tf.ones_default(size.clone());

            let size_int64_t: Vec<i64> = size.iter().map(|&x| x as i64).collect();

            let mut ctx = context();
            let ret = op_view_copy_out(&mut ctx, input, &size_int64_t, &out);
            assert!(tensors_are_close(&out, ret, 0.0, Some(0.0)));
            assert!(tensor_data_is_close(input, &out, 0.0, Some(0.0)));
        }
    }

    // Test if op_view_copy_out works well under all kinds of legal input type.
    fn test_dtype<T>()
    where
        T: CppTypeToScalarType + FactoryValue + FromI64,
    {
        let tf = TensorFactory::<T>::new();
        let input = tf.make_default(vec![2, 4], make_i64(&[0, 1, 1, 1, 0, 1, 0, 1]));

        let out_shapes: Vec<Vec<SizesType>> = vec![
            vec![8],
            vec![8, 1],
            vec![1, 8],
            vec![2, 4],
            vec![4, 2],
            vec![2, 2, 2],
            vec![1, 2, 1, 2, 1, 2, 1],
        ];

        run_view_test_cases::<T>(&input, &out_shapes);
    }

    fn test_empty_input<T>()
    where
        T: CppTypeToScalarType + FactoryValue + FromI64,
    {
        let tf = TensorFactory::<T>::new();
        let input = tf.make_default(vec![3, 0, 1, 2], vec![]);
        let out_shapes: Vec<Vec<SizesType>> = vec![
            vec![6, 0],
            vec![6, 0, 0],
            vec![3, 0, 1, 2],
            vec![1, 0, 2, 3],
        ];
        run_view_test_cases::<T>(&input, &out_shapes);
    }

    fn test_dynamic_shape(out_shape: Vec<SizesType>, dynamism: TensorShapeDynamism) {
        let tf = TensorFactory::<i32>::new();

        let x = tf.make_default(vec![3, 4], vec![4, 9, 3, 0, 3, 9, 7, 3, 7, 3, 1, 6]);
        let expected = tf.make_default(vec![2, 6], vec![4, 9, 3, 0, 3, 9, 7, 3, 7, 3, 1, 6]);

        let size: [i64; 2] = [2, 6];

        let out = tf.zeros(out_shape, dynamism);
        let mut ctx = context();
        op_view_copy_out(&mut ctx, &x, &size, &out);
        assert!(tensors_are_close(&out, &expected, 0.0, Some(0.0)));
    }

    macro_rules! forall_real_types_and_bool {
        ($f:ident) => {{
            $f::<u8>();
            $f::<i8>();
            $f::<i16>();
            $f::<i32>();
            $f::<i64>();
            $f::<f32>();
            $f::<f64>();
            $f::<bool>();
        }};
    }

    // [spec:et:sem:op-view-copy.torch.executor.native.view-copy-out-fn/test]
    // also verifies get_view_copy_target_size (writes each requested dim into
    // out_sizes, used to resize) and check_view_copy_args (numel/dtype/size gate);
    // multiple out_shapes incl. rank-7 pin the per-dim size copy
    // [spec:et:sem:copy-ops-util.torch.executor.get-view-copy-target-size-fn/test]
    // [spec:et:sem:copy-ops-util.torch.executor.check-view-copy-args-fn/test]
    #[test]
    fn op_view_test_all_dtypes_supported() {
        forall_real_types_and_bool!(test_dtype);
    }

    // [spec:et:sem:op-view-copy.torch.executor.native.view-copy-out-fn/test]
    #[test]
    fn op_view_test_empty_input_supported() {
        forall_real_types_and_bool!(test_empty_input);
    }

    // [spec:et:sem:op-view-copy.torch.executor.native.view-copy-out-fn/test]
    #[test]
    fn op_view_test_input_output_mismatched_sizes_die() {
        let tf = TensorFactory::<i32>::new();
        let size_in = vec![3, 1, 1, 2];
        let size_out = vec![3, 2, 1, 2];

        let input = tf.make_default(size_in, vec![1, 2, 3, 4, 5, 6]);
        let out = tf.ones_default(size_out.clone());

        let size_int64_t = vector_32_to_64(&size_out);

        let mut ctx = context();
        et_expect_kernel_failure!(ctx, op_view_copy_out(&mut ctx, &input, &size_int64_t, &out));
    }

    // [spec:et:sem:op-view-copy.torch.executor.native.view-copy-out-fn/test]
    #[test]
    fn op_view_test_size_output_mismatched_sizes_die() {
        let tf = TensorFactory::<i32>::new();
        let size = vec![3, 1, 1, 2];
        let size_target = vec![3, 2, 1, 2];
        let input = tf.make_default(size.clone(), vec![1, 2, 3, 4, 5, 6]);
        let out = tf.ones_default(size);

        let size_int64_t = vector_32_to_64(&size_target);

        let mut ctx = context();
        et_expect_kernel_failure!(ctx, op_view_copy_out(&mut ctx, &input, &size_int64_t, &out));
    }

    // [spec:et:sem:op-view-copy.torch.executor.native.view-copy-out-fn/test]
    #[test]
    fn op_view_test_mismatched_types_die() {
        let tf_in = TensorFactory::<i32>::new();
        let tf_out = TensorFactory::<f32>::new();
        let size = vec![3, 1, 1, 2];

        let input = tf_in.make_default(size.clone(), vec![1, 2, 3, 4, 5, 6]);
        let out = tf_out.ones_default(size.clone());

        let size_int64_t = vector_32_to_64(&size);

        let mut ctx = context();
        et_expect_kernel_failure!(ctx, op_view_copy_out(&mut ctx, &input, &size_int64_t, &out));
    }

    // [spec:et:sem:op-view-copy.torch.executor.native.view-copy-out-fn/test]
    #[test]
    fn op_view_test_size_infer() {
        let tf_in = TensorFactory::<f32>::new();
        let tf_out_valid = TensorFactory::<f32>::new();
        let in_size = vec![2, 2, 2];
        let out_size_view = vec![4, 2];
        let out_size_valid: [i32; 2] = [-1, 2];
        let out_size_invalid: [i32; 2] = [-1, -1];

        let input = tf_in.make_default(in_size, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let out = tf_out_valid.ones_default(out_size_view);

        let valid_size_int64_t = vector_32_to_64(&out_size_valid);
        let invalid_size_int64_t = vector_32_to_64(&out_size_invalid);

        // Inferring one dimension is valid.
        let mut ctx = context();
        op_view_copy_out(&mut ctx, &input, &valid_size_int64_t, &out);
        assert!(tensor_data_is_close(&input, &out, 0.0, Some(0.0)));
        // Inferring two dimensions is invalid.
        let mut ctx = context();
        et_expect_kernel_failure!(
            ctx,
            op_view_copy_out(&mut ctx, &input, &invalid_size_int64_t, &out)
        );
    }

    // #if !defined(USE_ATEN_LIB)
    // [spec:et:sem:op-view-copy.torch.executor.native.view-copy-out-fn/test]
    #[test]
    fn op_view_test_upper_bound_out_tensor() {
        let tf = TensorFactory::<f32>::new();
        let input = tf.make_default(vec![2, 4], vec![0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0]);
        let output = tf.zeros(vec![2, 2, 2], TensorShapeDynamism::DYNAMIC_BOUND);

        let size: [i32; 3] = [2, 2, 2];
        let ref_output =
            tf.make_default(vec![2, 2, 2], vec![0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0]);
        let size_int64_t = vector_32_to_64(&size);

        let mut ctx = context();
        op_view_copy_out(&mut ctx, &input, &size_int64_t, &output);
        assert!(tensors_are_close(&ref_output, &output, 0.0, Some(0.0)));

        let output = tf.zeros(vec![1, 4, 2], TensorShapeDynamism::DYNAMIC_BOUND);
        let size: [i32; 3] = [1, 4, 2];
        let ref_output =
            tf.make_default(vec![1, 4, 2], vec![0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0]);
        let mut size_int64_t = vector_32_to_64(&size);
        size_int64_t[1] = -1;

        op_view_copy_out(&mut ctx, &input, &size_int64_t, &output);
        assert!(tensors_are_close(&ref_output, &output, 0.0, Some(0.0)));
    }

    // [spec:et:sem:op-view-copy.torch.executor.native.view-copy-out-fn/test]
    #[test]
    fn op_view_test_dynamic_shape_upper_bound_same_as_expected() {
        test_dynamic_shape(vec![2, 6], TensorShapeDynamism::DYNAMIC_BOUND);
    }

    // PORT-NOTE: the C++ `ET_SKIP_IF(!output_resize, ...)` skips this test for the
    // portable kernel (SupportedFeatures::output_resize default false).
    // [spec:et:sem:op-view-copy.torch.executor.native.view-copy-out-fn/test]
    #[test]
    fn op_view_test_dynamic_shape_upper_bound_larger_than_expected() {
        println!("Skipping: portable kernel does not support output_resize (dynamic shape)");
    }

    // PORT-NOTE: the C++ `ET_SKIP_IF(!output_resize, ...)` skips this test for the
    // portable kernel (output_resize default false).
    // [spec:et:sem:op-view-copy.torch.executor.native.view-copy-out-fn/test]
    #[test]
    fn op_view_test_dynamic_shape_unbound() {
        println!("Skipping: portable kernel does not support output_resize (dynamic shape)");
    }
}
