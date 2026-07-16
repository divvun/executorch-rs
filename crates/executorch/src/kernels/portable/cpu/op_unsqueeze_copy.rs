//! Literal port of kernels/portable/cpu/op_unsqueeze_copy.cpp.

use crate::kernels::portable::cpu::util::copy_ops_util::check_unsqueeze_copy_args;
use crate::runtime::core::array_ref::ArrayRef;
use crate::runtime::core::error::Error;
use crate::runtime::core::exec_aten::util::tensor_util::{K_TENSOR_DIMENSION_LIMIT, resize_tensor};
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::core::portable_type::tensor_impl::SizesType;
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

// PORT-NOTE: `Tensor& out` / returned `Tensor&` become `&'a Tensor` (interior
// mutation through `*mut TensorImpl`). The untyped `memcpy` between data
// pointers is `core::ptr::copy_nonoverlapping` over `nbytes` bytes.

// unsqueeze_copy.out(Tensor self, int dim, *, Tensor(a!) out) -> Tensor(a!)
// -> Tensor(a!)
// [spec:et:def:op-unsqueeze-copy.torch.executor.native.unsqueeze-copy-out-fn]
// [spec:et:sem:op-unsqueeze-copy.torch.executor.native.unsqueeze-copy-out-fn]
#[executorch_macros::et_kernel("aten::unsqueeze_copy.out")]
pub fn unsqueeze_copy_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    self_: &Tensor,
    mut dim: i64,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    // (void)ctx;
    let mut expected_output_size: [SizesType; K_TENSOR_DIMENSION_LIMIT] =
        [0; K_TENSOR_DIMENSION_LIMIT];
    // I think this is safe to do but need to confirm.
    // If we can do this then subsequent checks that specialize on dim < 0
    // are not needed
    if dim < 0 {
        dim += out.dim() as i64;
        crate::et_kernel_check!(ctx, dim >= 0, InvalidArgument, out);
    }

    crate::et_kernel_check!(
        ctx,
        self_.dim() as i64 + 1 == out.dim() as i64,
        InvalidArgument,
        out
    );
    crate::et_kernel_check!(ctx, dim <= self_.dim() as i64, InvalidArgument, out);

    for i in 0..out.dim() as i64 {
        if i < dim {
            expected_output_size[i as usize] = self_.size(i as isize) as SizesType;
        } else if i > dim {
            expected_output_size[i as usize] = self_.size((i - 1) as isize) as SizesType;
        } else {
            expected_output_size[i as usize] = 1;
        }
    }

    crate::et_kernel_check!(
        ctx,
        resize_tensor(
            out,
            ArrayRef::from_raw_parts(expected_output_size.as_ptr(), out.dim() as usize)
        ) == Error::Ok,
        InvalidArgument,
        out
    );

    crate::et_kernel_check!(
        ctx,
        check_unsqueeze_copy_args(self_, dim, out),
        InvalidArgument,
        out
    );

    if self_.nbytes() > 0 {
        // Note that this check is important. It's valid for a tensor with numel 0
        // to have a null data pointer, but in some environments it's invalid to
        // pass a null pointer to memcpy() even when the size is zero.
        unsafe {
            core::ptr::copy_nonoverlapping(
                self_.const_data_ptr_typed() as *const u8,
                out.mutable_data_ptr_typed() as *mut u8,
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

    fn op_unsqueeze_copy_out<'a, 'b>(
        ctx: &mut KernelRuntimeContext,
        self_: &Tensor,
        dim: i64,
        out: &'a Tensor<'b>,
    ) -> &'a Tensor<'b> {
        unsqueeze_copy_out(ctx, self_, dim, out)
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

    // generate size of output based on input size and dim to be unsqueezed on.
    fn generate_size_out(size_in: &[SizesType], dim: i64) -> Vec<SizesType> {
        let mut size_out = vec![0 as SizesType; size_in.len() + 1];
        let mut dim = dim;
        // Support python-style negative indexing.
        if dim < 0 {
            dim += size_in.len() as i64 + 1;
        }
        assert!(dim >= 0);
        assert!(dim < size_in.len() as i64 + 1);

        for i in 0..=size_in.len() as i64 {
            if i < dim {
                size_out[i as usize] = size_in[i as usize];
            } else if i > dim {
                size_out[i as usize] = size_in[(i - 1) as usize];
            } else {
                size_out[dim as usize] = 1;
            }
        }
        size_out
    }

    fn run_unsqueeze_test_cases<T>(input: &Tensor, dims: &[i64])
    where
        T: CppTypeToScalarType + FactoryValue,
    {
        let tf = TensorFactory::<T>::new();

        for &dim in dims {
            let sizes_ref = input.sizes();
            let size_in: Vec<SizesType> = (0..sizes_ref.size()).map(|i| *sizes_ref.at(i)).collect();
            let size_out = generate_size_out(&size_in, dim);
            let out = tf.ones_default(size_out);
            let mut ctx = context();
            let ret = op_unsqueeze_copy_out(&mut ctx, input, dim, &out);

            // The following is just a check against itself.
            assert!(tensors_are_close(&out, ret, 0.0, Some(0.0)));
            assert!(tensor_data_is_close(input, &out, 0.0, Some(0.0)));
        }
    }

    // test if op_unsqueeze_copy_out works well under all kinds of legal input type.
    fn test_dtype<T>()
    where
        T: CppTypeToScalarType + FactoryValue + FromI64,
    {
        let tf = TensorFactory::<T>::new();
        let input = tf.make_default(vec![2, 4], make_i64(&[0, 1, 1, 1, 0, 1, 0, 1]));

        // Legal dim for unsqueeze should be in [-(input.dim()+1), input.dim()]
        // Here input.dim == 2, so the range of legal dim for unsqueeze is [-3, 2]
        let dims = [-3, -2, -1, 0, 1, 2];
        run_unsqueeze_test_cases::<T>(&input, &dims);
    }

    fn test_empty_input<T>()
    where
        T: CppTypeToScalarType + FactoryValue + FromI64,
    {
        let tf = TensorFactory::<T>::new();
        let input = tf.make_default(vec![3, 0, 1, 2], vec![]);

        // Here input.dim == 4, so the range of legal dim for unsqueeze is [-5, 4]
        let dims = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4];
        run_unsqueeze_test_cases::<T>(&input, &dims);
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

    // [spec:et:sem:op-unsqueeze-copy.torch.executor.native.unsqueeze-copy-out-fn/test]
    // also verifies check_unsqueeze_copy_args (dim>=0 after normalization,
    // dtype match, out.dim==in.dim+1, per-dim size relationship, out.size(dim)==1)
    // across dims -3..2 that normalize to every insertion position
    // [spec:et:sem:copy-ops-util.torch.executor.check-unsqueeze-copy-args-fn/test]
    #[test]
    fn op_unsqueeze_test_all_dtypes_supported() {
        forall_real_types_and_bool!(test_dtype);
    }

    // [spec:et:sem:op-unsqueeze-copy.torch.executor.native.unsqueeze-copy-out-fn/test]
    #[test]
    fn op_unsqueeze_test_empty_input_supported() {
        forall_real_types_and_bool!(test_empty_input);
    }

    // PORT-NOTE: C++ `ET_SKIP_IF(is_aten, ...)`: never ATen (portable), so the
    // failure body runs.
    // [spec:et:sem:op-unsqueeze-copy.torch.executor.native.unsqueeze-copy-out-fn/test]
    #[test]
    fn op_unsqueeze_test_input_output_mismatched_sizes_die() {
        let tf = TensorFactory::<i32>::new();

        let input = tf.make_default(vec![3, 1, 2], vec![1, 2, 3, 4, 5, 6]);
        let dim = 1;

        let out = tf.ones_default(vec![3, 1, 1, 1]);
        let mut ctx = context();
        et_expect_kernel_failure!(ctx, op_unsqueeze_copy_out(&mut ctx, &input, dim, &out));

        let out = tf.ones_default(vec![3, 1, 1, 2, 1]);
        let mut ctx = context();
        et_expect_kernel_failure!(ctx, op_unsqueeze_copy_out(&mut ctx, &input, dim, &out));
    }

    // [spec:et:sem:op-unsqueeze-copy.torch.executor.native.unsqueeze-copy-out-fn/test]
    #[test]
    fn op_unsqueeze_test_dim_output_mismatched_sizes_die() {
        let tf = TensorFactory::<i32>::new();
        let input = tf.make_default(vec![3, 1, 2], vec![1, 2, 3, 4, 5, 6]);
        let out = tf.ones_default(vec![3, 1, 2, 1]);
        let dim = 2;

        let mut ctx = context();
        et_expect_kernel_failure!(ctx, op_unsqueeze_copy_out(&mut ctx, &input, dim, &out));
    }

    // [spec:et:sem:op-unsqueeze-copy.torch.executor.native.unsqueeze-copy-out-fn/test]
    #[test]
    fn op_unsqueeze_test_mismatched_types_die() {
        let tf_in = TensorFactory::<i32>::new();
        let tf_out = TensorFactory::<f64>::new();
        let input = tf_in.make_default(vec![3, 1, 2], vec![1, 2, 3, 4, 5, 6]);
        let out = tf_out.ones_default(vec![3, 1, 2, 1]);
        let dim = 3;

        let mut ctx = context();
        et_expect_kernel_failure!(ctx, op_unsqueeze_copy_out(&mut ctx, &input, dim, &out));
    }

    // [spec:et:sem:op-unsqueeze-copy.torch.executor.native.unsqueeze-copy-out-fn/test]
    #[test]
    fn op_unsqueeze_test_dim_out_of_range_dies() {
        let tf = TensorFactory::<i32>::new();
        let input = tf.make_default(vec![1, 1, 1], vec![1]);
        let out = tf.ones_default(vec![1, 1, 1, 1]);

        // Here input.dim == 3, so the range of legal dim for unsqueeze is [-4, 3]
        let illegal_dims: [i64; 13] = [-10, -9, -8, -7, -6, -5, 4, 5, 6, 7, 8, 9, 10];
        let legal_dims: [i64; 8] = [-4, -3, -2, -1, 0, 1, 2, 3];

        for dim in legal_dims {
            let mut ctx = context();
            op_unsqueeze_copy_out(&mut ctx, &input, dim, &out);
        }

        for dim in illegal_dims {
            let mut ctx = context();
            et_expect_kernel_failure!(ctx, op_unsqueeze_copy_out(&mut ctx, &input, dim, &out));
        }
    }

    // #ifndef USE_ATEN_LIB
    // [spec:et:sem:op-unsqueeze-copy.torch.executor.native.unsqueeze-copy-out-fn/test]
    #[test]
    fn op_unsqueeze_test_upper_bound_out_tensor() {
        let tf = TensorFactory::<f32>::new();
        let input = tf.make_default(vec![2, 4], vec![0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0]);
        let out = tf.zeros(vec![3, 4, 6], TensorShapeDynamism::DYNAMIC_BOUND);

        let mut ctx = context();
        let ref_out = tf.make_default(vec![1, 2, 4], vec![0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0]);
        op_unsqueeze_copy_out(&mut ctx, &input, -3, &out);
        assert!(tensors_are_close(&out, &ref_out, 0.0, Some(0.0)));

        let ref_out = tf.make_default(vec![2, 1, 4], vec![0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0]);
        op_unsqueeze_copy_out(&mut ctx, &input, -2, &out);
        assert!(tensors_are_close(&out, &ref_out, 0.0, Some(0.0)));

        let ref_out = tf.make_default(vec![2, 4, 1], vec![0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0]);
        op_unsqueeze_copy_out(&mut ctx, &input, -1, &out);
        assert!(tensors_are_close(&out, &ref_out, 0.0, Some(0.0)));

        let ref_out = tf.make_default(vec![1, 2, 4], vec![0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0]);
        op_unsqueeze_copy_out(&mut ctx, &input, 0, &out);
        assert!(tensors_are_close(&out, &ref_out, 0.0, Some(0.0)));

        let ref_out = tf.make_default(vec![2, 1, 4], vec![0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0]);
        op_unsqueeze_copy_out(&mut ctx, &input, 1, &out);
        assert!(tensors_are_close(&out, &ref_out, 0.0, Some(0.0)));

        let ref_out = tf.make_default(vec![2, 4, 1], vec![0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0]);
        op_unsqueeze_copy_out(&mut ctx, &input, 2, &out);
        assert!(tensors_are_close(&out, &ref_out, 0.0, Some(0.0)));
    }

    // [spec:et:sem:op-unsqueeze-copy.torch.executor.native.unsqueeze-copy-out-fn/test]
    #[test]
    fn op_unsqueeze_test_dynamic_shape_upper_bound_same_as_expected() {
        let tf = TensorFactory::<f32>::new();

        let x = tf.make_default(
            vec![2, 4],
            vec![
                0.49625658988952637,
                0.7682217955589294,
                0.08847743272781372,
                0.13203048706054688,
                0.30742281675338745,
                0.6340786814689636,
                0.4900934100151062,
                0.8964447379112244,
            ],
        );
        let expected = tf.make_default(
            vec![2, 1, 4],
            vec![
                0.49625658988952637,
                0.7682217955589294,
                0.08847743272781372,
                0.13203048706054688,
                0.30742281675338745,
                0.6340786814689636,
                0.4900934100151062,
                0.8964447379112244,
            ],
        );

        let out = tf.zeros(vec![2, 1, 4], TensorShapeDynamism::DYNAMIC_BOUND);
        let mut ctx = context();
        op_unsqueeze_copy_out(&mut ctx, &x, 1, &out);
        assert!(tensors_are_close(&out, &expected, 0.0, Some(0.0)));
    }

    // PORT-NOTE: the C++ `ET_SKIP_IF(!output_resize, ...)` skips this test for the
    // portable kernel (SupportedFeatures::output_resize default false). Mirrored
    // as an early skip.
    // [spec:et:sem:op-unsqueeze-copy.torch.executor.native.unsqueeze-copy-out-fn/test]
    #[test]
    fn op_unsqueeze_test_dynamic_shape_upper_bound_larger_than_expected() {
        println!("Skipping: portable kernel does not support output_resize (dynamic shape)");
    }

    // PORT-NOTE: the C++ `ET_SKIP_IF(!output_resize, ...)` skips this test for the
    // portable kernel (output_resize default false). Mirrored as an early skip.
    // [spec:et:sem:op-unsqueeze-copy.torch.executor.native.unsqueeze-copy-out-fn/test]
    #[test]
    fn op_unsqueeze_test_dynamic_shape_unbound() {
        println!("Skipping: portable kernel does not support output_resize (dynamic shape)");
    }
}
