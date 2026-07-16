//! Literal port of kernels/portable/cpu/op__clone_dim_order.cpp.

use crate::kernels::portable::cpu::util::copy_ops_util::{
    _to_dim_order_copy_impl, check__to_dim_order_copy_args,
};
use crate::runtime::core::array_ref::ArrayRef;
use crate::runtime::core::error::Error;
use crate::runtime::core::exec_aten::util::tensor_util::resize_tensor;
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

type OptionalArrayRefI64 = Option<ArrayRef<i64>>;

/// Checks the conditions for fast path direct memcpy. This can be used
/// when the output dim order is unchanged.
// [spec:et:def:op-clone-dim-order.torch.executor.native.check-fast-path-conditions-fn]
// [spec:et:sem:op-clone-dim-order.torch.executor.native.check-fast-path-conditions-fn]
fn check_fast_path_conditions(in_: &Tensor, dim_order: OptionalArrayRefI64) -> bool {
    if !dim_order.is_some() {
        // No dim order means preserve input dim order.
        return true;
    }

    let input_dim_order = in_.dim_order();
    // PORT-NOTE: C++ `std::equal(a.begin(), a.end(), b.begin(), b.end())`
    // (4-iterator form) compares length first, then each element with implicit
    // widening of the u8 input dim-order value to the i64 requested value.
    let requested = dim_order.unwrap();
    if requested.size() != input_dim_order.size() {
        return false;
    }
    for i in 0..requested.size() {
        if *requested.at(i) != *input_dim_order.at(i) as i64 {
            return false;
        }
    }
    true
}

/// _clone_dim_order.out(Tensor self, *, bool non_blocking=False, int[]?
/// dim_order=None, Tensor(a!) out) -> Tensor(a!)
///
/// Clones via element-wise copy while preserving dim_order.
// [spec:et:def:op-clone-dim-order.torch.executor.native.clone-dim-order-out-fn]
// [spec:et:sem:op-clone-dim-order.torch.executor.native.clone-dim-order-out-fn]
pub fn _clone_dim_order_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    self_: &Tensor,
    non_blocking: bool,
    dim_order: OptionalArrayRefI64,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    let _ = &ctx;

    // Ensure input and output dtype match.
    crate::et_kernel_check!(
        ctx,
        self_.scalar_type() == out.scalar_type(),
        InvalidArgument,
        out
    );

    // Ensure output has the same layout as input or matches dim_order.
    crate::et_kernel_check!(
        ctx,
        check__to_dim_order_copy_args(self_, non_blocking, dim_order, out),
        InvalidArgument,
        out
    );

    // Ensure input and output shapes match, resizing if necessary.
    crate::et_kernel_check!(
        ctx,
        resize_tensor(out, self_.sizes()) == Error::Ok,
        InvalidArgument,
        out
    );

    if self_.numel() == 0 {
        return out;
    }

    // Dispatch to the fast path if we can use direct memcpy.
    if check_fast_path_conditions(self_, dim_order) {
        unsafe {
            core::ptr::copy_nonoverlapping(
                self_.const_data_ptr_typed() as *const u8,
                out.mutable_data_ptr_typed() as *mut u8,
                self_.nbytes(),
            );
        }
    } else {
        // Select the correct input dtype and copy the tensors.
        crate::et_switch_realhbbf16_types!(
            self_.scalar_type(),
            ctx,
            "dim_order_ops::_clone_dim_order.out",
            CTYPE,
            {
                _to_dim_order_copy_impl::<CTYPE, CTYPE>(self_, out);
            }
        );
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::core::exec_aten::testing_util::tensor_factory::{
        FactoryValue, TensorFactory,
    };
    use crate::runtime::core::exec_aten::testing_util::tensor_util::tensors_are_close;
    use crate::runtime::core::exec_aten::util::scalar_type_util::CppTypeToScalarType;
    use crate::runtime::core::memory_allocator::{MemoryAllocator, MemoryAllocatorBase};
    use crate::runtime::core::tensor_shape_dynamism::TensorShapeDynamism;

    fn context() -> KernelRuntimeContext<'static> {
        // Mirrors `OperatorTest::SetUp()`'s `runtime_init()` PAL initialization.
        crate::runtime::platform::platform::pal_init();
        KernelRuntimeContext::new(
            crate::extension::module::module::null_event_tracer(),
            core::ptr::null_mut::<MemoryAllocator>() as *mut dyn MemoryAllocatorBase,
        )
    }

    trait FromF64: Copy {
        fn from_f64(v: f64) -> Self;
    }
    macro_rules! impl_from_f64_num {
        ($($t:ty),*) => {$(impl FromF64 for $t { fn from_f64(v: f64) -> Self { v as $t } })*};
    }
    impl_from_f64_num!(u8, i8, i16, i32, i64, f32, f64);

    struct ToTestCase {
        sizes: Vec<i32>,
        data_in: Vec<f64>,
    }

    fn test_runner_clone<T>(test_cases: &[ToTestCase])
    where
        T: CppTypeToScalarType + FactoryValue + FromF64,
    {
        let tf_in = TensorFactory::<T>::new();
        let tf_out = TensorFactory::<T>::new();

        for test_case in test_cases {
            let data_in: Vec<T> = test_case.data_in.iter().map(|&x| T::from_f64(x)).collect();

            let input = tf_in.make_default(test_case.sizes.clone(), data_in.clone());
            let output = tf_out.zeros_like(&input, TensorShapeDynamism::STATIC);

            let mut dim_order_vec: Vec<i64> = Vec::new();
            for i in 0..input.dim() {
                dim_order_vec.push(i as i64);
            }
            let dim_order = ArrayRef::from_raw_parts(dim_order_vec.as_ptr(), dim_order_vec.len());

            let mut ctx = context();
            let ret = _clone_dim_order_out(&mut ctx, &input, false, Some(dim_order), &output);

            let expected = tf_out.make_default(test_case.sizes.clone(), data_in);

            assert!(tensors_are_close(ret, &output, 0.0, Some(0.0)));
            assert!(tensors_are_close(ret, &expected, 0.0, Some(0.0)));
        }
    }

    fn test_dynamic_shape(out_shape: Vec<i32>, dynamism: TensorShapeDynamism) {
        let tf = TensorFactory::<f32>::new();

        let x = tf.make_default(
            vec![2, 3],
            vec![
                0.49625658988952637,
                0.7682217955589294,
                0.08847743272781372,
                0.13203048706054688,
                0.30742281675338745,
                0.6340786814689636,
            ],
        );
        let expected = tf.make_default(
            vec![2, 3],
            vec![
                0.49625658988952637,
                0.7682217955589294,
                0.08847743272781372,
                0.13203048706054688,
                0.30742281675338745,
                0.6340786814689636,
            ],
        );

        let non_blocking = false;

        let out = tf.zeros(out_shape, dynamism);

        let mut dim_order_vec: Vec<i64> = Vec::new();
        for i in 0..x.dim() {
            dim_order_vec.push(i as i64);
        }
        let dim_order = ArrayRef::from_raw_parts(dim_order_vec.as_ptr(), dim_order_vec.len());

        let mut ctx = context();
        let ret = _clone_dim_order_out(&mut ctx, &x, non_blocking, Some(dim_order), &out);

        assert!(tensors_are_close(&out, &expected, 0.0, Some(0.0)));
        assert!(tensors_are_close(ret, &expected, 0.0, Some(0.0)));
    }

    fn all_dtypes_cases() -> Vec<ToTestCase> {
        vec![
            ToTestCase {
                sizes: vec![2, 4],
                data_in: vec![2.11, 3.2, 2.3, 4.0, 1.1, 5.2, 1.1, 6.3],
            },
            ToTestCase {
                sizes: vec![3, 4, 0, 5],
                data_in: vec![],
            },
            ToTestCase {
                sizes: vec![],
                data_in: vec![10.0],
            },
        ]
    }

    // ET_FORALL_REAL_TYPES: Byte,Char,Short,Int,Long,Float,Double
    // Default (identity) dim_order drives check_fast_path_conditions to true → memcpy path.
    // [spec:et:sem:op-clone-dim-order.torch.executor.native.clone-dim-order-out-fn/test]
    // [spec:et:sem:op-clone-dim-order.torch.executor.native.check-fast-path-conditions-fn/test]
    #[test]
    fn op_dim_order_clone_test_all_dtypes_supported() {
        let cases = all_dtypes_cases();
        test_runner_clone::<u8>(&cases);
        test_runner_clone::<i8>(&cases);
        test_runner_clone::<i16>(&cases);
        test_runner_clone::<i32>(&cases);
        test_runner_clone::<i64>(&cases);
        test_runner_clone::<f32>(&cases);
        test_runner_clone::<f64>(&cases);
    }

    // PORT-NOTE: C++ `ET_SKIP_IF(is_aten, ...)`: the ported runtime is never ATen,
    // so this failure path is always exercised.
    // [spec:et:sem:op-clone-dim-order.torch.executor.native.clone-dim-order-out-fn/test]
    #[test]
    fn op_dim_order_clone_test_mismatched_sizes_die() {
        let tf = TensorFactory::<i32>::new();
        let input = tf.make_default(vec![3, 1, 1, 2], vec![1, 2, 3, 4, 5, 6]);
        let out = tf.zeros_default(vec![3, 2, 1, 1]);
        let mut dim_order_vec: Vec<i64> = Vec::new();
        for i in 0..input.dim() {
            dim_order_vec.push(i as i64);
        }
        let dim_order = ArrayRef::from_raw_parts(dim_order_vec.as_ptr(), dim_order_vec.len());

        let mut ctx = context();
        _clone_dim_order_out(&mut ctx, &input, false, Some(dim_order), &out);
        assert_ne!(ctx.failure_state(), Error::Ok);
    }

    // [spec:et:sem:op-clone-dim-order.torch.executor.native.clone-dim-order-out-fn/test]
    #[test]
    fn op_dim_order_clone_test_mismatched_memory_format_dies() {
        let tf_in = TensorFactory::<f32>::new();
        let tf_out = TensorFactory::<f32>::new();
        let input = tf_in.make_default(vec![3, 1, 1, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let out = tf_out.zeros_default(vec![3, 1, 1, 2]);

        let mut dim_order_vec: Vec<i64> = Vec::new();
        for i in 0..input.dim() {
            dim_order_vec.push(i as i64);
        }
        // Mutate dim_order_vec to create an illegal dim_order.
        dim_order_vec[1] = 3;
        dim_order_vec[3] = 1;
        let dim_order = ArrayRef::from_raw_parts(dim_order_vec.as_ptr(), dim_order_vec.len());

        let mut ctx = context();
        _clone_dim_order_out(&mut ctx, &input, false, Some(dim_order), &out);
        assert_ne!(ctx.failure_state(), Error::Ok);
    }

    // [spec:et:sem:op-clone-dim-order.torch.executor.native.clone-dim-order-out-fn/test]
    #[test]
    fn op_dim_order_clone_test_mismatched_blocking_die() {
        let tf = TensorFactory::<i32>::new();
        let input = tf.make_default(vec![3, 1, 1, 2], vec![1, 2, 3, 4, 5, 6]);
        let out = tf.zeros_default(vec![3, 1, 1, 2]);

        let mut dim_order_vec: Vec<i64> = Vec::new();
        for i in 0..input.dim() {
            dim_order_vec.push(i as i64);
        }
        let dim_order = ArrayRef::from_raw_parts(dim_order_vec.as_ptr(), dim_order_vec.len());

        let mut ctx = context();
        _clone_dim_order_out(&mut ctx, &input, true, Some(dim_order), &out);
        assert_ne!(ctx.failure_state(), Error::Ok);
    }

    // [spec:et:sem:op-clone-dim-order.torch.executor.native.clone-dim-order-out-fn/test]
    #[test]
    fn op_dim_order_clone_test_dynamic_shape_upper_bound_same_as_expected() {
        test_dynamic_shape(vec![2, 3], TensorShapeDynamism::DYNAMIC_BOUND);
    }

    // [spec:et:sem:op-clone-dim-order.torch.executor.native.clone-dim-order-out-fn/test]
    #[test]
    fn op_dim_order_clone_test_dynamic_shape_upper_bound_larger_than_expected() {
        test_dynamic_shape(vec![10, 10], TensorShapeDynamism::DYNAMIC_BOUND);
    }

    // PORT-NOTE: C++ `ET_SKIP_IF(!output_resize, ...)`: the portable kernel's
    // `output_resize` SupportedFeature is false, so this test is skipped in the
    // portable build. Ported as `#[ignore]`.
    // [spec:et:sem:op-clone-dim-order.torch.executor.native.clone-dim-order-out-fn/test]
    #[test]
    #[ignore = "SKIP_IF(!output_resize): portable kernel does not support output resize"]
    fn op_dim_order_clone_test_dynamic_shape_unbound() {
        test_dynamic_shape(vec![1, 1], TensorShapeDynamism::DYNAMIC_UNBOUND);
    }

    const CL_CONTIG_DATA: [f64; 60] = [
        0.2432, 0.5248, 0.5361, 0.8513, 0.8184, 0.8206, 0.7357, 0.9655, 0.6138, 0.1112, 0.2799,
        0.1079, 0.9680, 0.2548, 0.0393, 0.6002, 0.2257, 0.8766, 0.2715, 0.1595, 0.2029, 0.7026,
        0.6982, 0.8529, 0.4405, 0.6560, 0.9217, 0.6372, 0.2446, 0.6590, 0.3866, 0.7185, 0.4439,
        0.5346, 0.3179, 0.4492, 0.3491, 0.6970, 0.8456, 0.2516, 0.2345, 0.2924, 0.7695, 0.0911,
        0.8530, 0.8560, 0.6909, 0.7719, 0.8923, 0.5546, 0.6978, 0.8151, 0.3007, 0.3961, 0.8416,
        0.4296, 0.7203, 0.8963, 0.3597, 0.5552,
    ];

    const CL_CHANNELS_LAST_DATA: [f64; 60] = [
        0.2432, 0.8184, 0.6138, 0.9680, 0.2257, 0.5248, 0.8206, 0.1112, 0.2548, 0.8766, 0.5361,
        0.7357, 0.2799, 0.0393, 0.2715, 0.8513, 0.9655, 0.1079, 0.6002, 0.1595, 0.2029, 0.4405,
        0.2446, 0.4439, 0.3491, 0.7026, 0.6560, 0.6590, 0.5346, 0.6970, 0.6982, 0.9217, 0.3866,
        0.3179, 0.8456, 0.8529, 0.6372, 0.7185, 0.4492, 0.2516, 0.2345, 0.8530, 0.8923, 0.3007,
        0.7203, 0.2924, 0.8560, 0.5546, 0.3961, 0.8963, 0.7695, 0.6909, 0.6978, 0.8416, 0.3597,
        0.0911, 0.7719, 0.8151, 0.4296, 0.5552,
    ];

    fn f32_vec(v: &[f64]) -> Vec<f32> {
        v.iter().map(|&x| x as f32).collect()
    }

    // Changing the dim_order (contiguous → channels_last) drives
    // check_fast_path_conditions to false → element-wise copy path.
    // [spec:et:sem:op-clone-dim-order.torch.executor.native.clone-dim-order-out-fn/test]
    // [spec:et:sem:op-clone-dim-order.torch.executor.native.check-fast-path-conditions-fn/test]
    #[test]
    fn op_dim_order_clone_test_contiguous_to_channels_last() {
        let tf = TensorFactory::<f32>::new();

        // x is in contiguous dim order {0, 1, 2, 3}.
        let x = tf.make_with_dimorder(
            vec![3, 5, 2, 2],
            f32_vec(&CL_CONTIG_DATA),
            vec![],
            TensorShapeDynamism::STATIC,
        );

        let out = tf.full_channels_last(vec![3, 5, 2, 2], 0.0, TensorShapeDynamism::STATIC);
        let expected = tf.make_with_dimorder(
            vec![3, 5, 2, 2],
            f32_vec(&CL_CHANNELS_LAST_DATA),
            vec![0, 2, 3, 1],
            TensorShapeDynamism::STATIC,
        );

        let dim_order_vec: [i64; 4] = [0, 2, 3, 1];
        let dim_order = ArrayRef::from_raw_parts(dim_order_vec.as_ptr(), 4);
        let mut ctx = context();
        let ret = _clone_dim_order_out(&mut ctx, &x, false, Some(dim_order), &out);

        assert!(tensors_are_close(&out, &expected, 0.0, Some(0.0)));
        assert!(tensors_are_close(ret, &expected, 0.0, Some(0.0)));
    }

    // [spec:et:sem:op-clone-dim-order.torch.executor.native.clone-dim-order-out-fn/test]
    #[test]
    fn op_dim_order_clone_test_channels_last_to_contiguous() {
        let tf = TensorFactory::<f32>::new();

        let out = tf.full(vec![3, 5, 2, 2], 0.0, TensorShapeDynamism::STATIC);

        // x is in channels_last dim order {0, 2, 3, 1}.
        let x = tf.make_with_dimorder(
            vec![3, 5, 2, 2],
            f32_vec(&CL_CHANNELS_LAST_DATA),
            vec![0, 2, 3, 1],
            TensorShapeDynamism::STATIC,
        );

        let expected = tf.make_with_dimorder(
            vec![3, 5, 2, 2],
            f32_vec(&CL_CONTIG_DATA),
            vec![],
            TensorShapeDynamism::STATIC,
        );

        let dim_order_vec: [i64; 4] = [0, 1, 2, 3];
        let dim_order = ArrayRef::from_raw_parts(dim_order_vec.as_ptr(), 4);
        let mut ctx = context();
        let ret = _clone_dim_order_out(&mut ctx, &x, false, Some(dim_order), &out);

        assert!(tensors_are_close(&out, &expected, 0.0, Some(0.0)));
        assert!(tensors_are_close(ret, &expected, 0.0, Some(0.0)));
    }

    // [spec:et:sem:op-clone-dim-order.torch.executor.native.clone-dim-order-out-fn/test]
    #[test]
    fn op_dim_order_clone_test_preserve_channels_last() {
        let tf = TensorFactory::<f32>::new();

        let out = tf.full_channels_last(vec![3, 5, 2, 2], 0.0, TensorShapeDynamism::STATIC);
        let x = tf.make_with_dimorder(
            vec![3, 5, 2, 2],
            f32_vec(&CL_CHANNELS_LAST_DATA),
            vec![0, 2, 3, 1],
            TensorShapeDynamism::STATIC,
        );

        let expected = tf.make_with_dimorder(
            vec![3, 5, 2, 2],
            f32_vec(&CL_CHANNELS_LAST_DATA),
            vec![0, 2, 3, 1],
            TensorShapeDynamism::STATIC,
        );

        let mut ctx = context();
        let ret = _clone_dim_order_out(&mut ctx, &x, false, None, &out);

        assert!(tensors_are_close(&out, &expected, 0.0, Some(0.0)));
        assert!(tensors_are_close(ret, &expected, 0.0, Some(0.0)));
    }
}
