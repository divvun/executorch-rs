//! Literal port of kernels/portable/cpu/op_cat.cpp.

use crate::kernels::portable::cpu::util::copy_ops_util::{check_cat_args, get_cat_out_target_size};
use crate::kernels::portable::cpu::util::dtype_util::internal::StaticCast;
use crate::runtime::core::array_ref::ArrayRef;
use crate::runtime::core::error::Error;
use crate::runtime::core::exec_aten::util::scalar_type_util::is_complex_type;
use crate::runtime::core::exec_aten::util::tensor_util::{
    K_TENSOR_DIMENSION_LIMIT, getLeadingDims, getTrailingDims, resize_tensor_same_type,
};
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::core::portable_type::tensor_impl::{SizesType, ssize_t};
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

// PORT-NOTE: `Tensor& out` / returned `Tensor&` become `&'a Tensor` (interior
// mutation through `*mut TensorImpl`).
//
// PORT-NOTE: the C++ `memcpy` becomes `core::ptr::copy_nonoverlapping` over the
// byte-equal element region; the elementwise `static_cast<CTYPE_OUT>` uses the
// dtype_util `StaticCast` trait (as bitwise/dtype ports do).
//
// PORT-NOTE: the C++ inner-lambda `return` (real branch) skips just the current
// input; ported as `continue` on the `j` loop (the ported switch expands inline
// to a `match`, not a closure). The C++ outer-lambda `return` (complex branch)
// exits the whole copy; ported as a labeled `break 'copy` out of the loops.

// [spec:et:def:op-cat.torch.executor.native.cat-out-fn]
// [spec:et:sem:op-cat.torch.executor.native.cat-out-fn]
#[executorch_macros::et_kernel("aten::cat.out")]
pub fn cat_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    tensors: ArrayRef<Tensor>,
    mut dim: i64,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    if dim < 0 {
        dim += out.dim() as i64;
    }

    crate::et_kernel_check!(ctx, check_cat_args(tensors, dim, out), InvalidArgument, out);

    let mut expected_out_size: [SizesType; K_TENSOR_DIMENSION_LIMIT] =
        [0; K_TENSOR_DIMENSION_LIMIT];
    let mut expected_out_dim: usize = 0;
    unsafe {
        get_cat_out_target_size(
            tensors,
            dim,
            expected_out_size.as_mut_ptr(),
            &mut expected_out_dim,
        );
    }

    crate::et_kernel_check!(
        ctx,
        resize_tensor_same_type(
            out,
            ArrayRef::from_raw_parts(expected_out_size.as_ptr(), expected_out_dim)
        ) == Error::Ok,
        InvalidArgument,
        out
    );

    // Special handling when all inputs are 1D-empty tensors for aten consistency
    // In that case, just return an 1D-empty tensor without checking dim
    let mut all_1d_empty = true;
    for i in 0..tensors.size() {
        if tensors.at(i).numel() != 0 || tensors.at(i).dim() != 1 {
            all_1d_empty = false;
            break;
        }
    }
    if all_1d_empty {
        return out;
    }

    let outer: usize = getLeadingDims(out, dim);
    let dim_stride: usize = getTrailingDims(out, dim);
    let ninputs: usize = tensors.size();

    let out_type = out.scalar_type();
    let out_is_complex: bool = is_complex_type(out.scalar_type());

    if out_is_complex {
        // TODO: The current support for complex dtype enforces that input and
        // output tensors have the same dtype. Support mixed dtypes in the future.
        for i in 0..ninputs {
            let in_type = tensors.at(i).scalar_type();
            crate::et_kernel_check!(ctx, out_type == in_type, InvalidArgument, out);
        }
        crate::et_switch_complexh_types!(out_type, ctx, "cat.out", CTYPE, {
            let mut out_ptr: *mut CTYPE = out.mutable_data_ptr::<CTYPE>();
            'copy: for i in 0..outer {
                for j in 0..ninputs {
                    if tensors.at(j).numel() == 0 {
                        break 'copy;
                    }
                    let inner: usize = tensors.at(j).size(dim as ssize_t) as usize * dim_stride;
                    let in_ptr: *const CTYPE =
                        unsafe { tensors.at(j).const_data_ptr::<CTYPE>().add(i * inner) };
                    unsafe {
                        core::ptr::copy_nonoverlapping(
                            in_ptr as *const u8,
                            out_ptr as *mut u8,
                            inner * core::mem::size_of::<CTYPE>(),
                        );
                    }
                    out_ptr = unsafe { out_ptr.add(inner) };
                }
            }
        });
    } else {
        crate::et_switch_realhbbf16_types!(out_type, ctx, "cat.out", CTYPE_OUT, {
            let mut out_ptr: *mut CTYPE_OUT = out.mutable_data_ptr::<CTYPE_OUT>();
            for i in 0..outer {
                for j in 0..ninputs {
                    let in_type = tensors.at(j).scalar_type();
                    crate::et_switch_realhbbf16_types!(in_type, ctx, "cat.out", CTYPE_IN, {
                        if tensors.at(j).numel() == 0 {
                            continue;
                        }
                        let inner: usize = tensors.at(j).size(dim as ssize_t) as usize * dim_stride;
                        let in_ptr: *const CTYPE_IN =
                            unsafe { tensors.at(j).const_data_ptr::<CTYPE_IN>().add(i * inner) };

                        if core::mem::size_of::<CTYPE_IN>() == core::mem::size_of::<CTYPE_OUT>() {
                            unsafe {
                                core::ptr::copy_nonoverlapping(
                                    in_ptr as *const u8,
                                    out_ptr as *mut u8,
                                    inner * core::mem::size_of::<CTYPE_IN>(),
                                );
                            }
                        } else {
                            for k in 0..inner {
                                unsafe {
                                    *out_ptr.add(k) =
                                        <CTYPE_OUT as StaticCast<CTYPE_IN>>::static_cast(
                                            *in_ptr.add(k),
                                        );
                                }
                            }
                        }
                        out_ptr = unsafe { out_ptr.add(inner) };
                    });
                }
            }
        });
    }

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

    // Builds an `ArrayRef<Tensor>` view over a slice of Tensors (mirrors
    // `ArrayRef<Tensor>(inputs.data(), inputs.size())`).
    fn list<'t>(v: &'t [Tensor]) -> ArrayRef<Tensor<'t>> {
        ArrayRef::from_raw_parts(v.as_ptr(), v.len())
    }

    fn test_dtype<T>()
    where
        T: CppTypeToScalarType + FactoryValue + FromI32,
    {
        let tf = TensorFactory::<T>::new();

        let x = tf.ones_default(vec![2, 1]);
        let y = tf.zeros_default(vec![2, 1]);
        let inputs = vec![x, y];

        let out = tf.ones_default(vec![2, 2]);
        let mut ctx = context();
        cat_out(&mut ctx, list(&inputs), 1, &out);

        #[rustfmt::skip]
        let expected = tf.make_default(
            vec![2, 2],
            vec![
                T::from_i32(1), T::from_i32(0),
                T::from_i32(1), T::from_i32(0),
            ],
        );

        assert_tensor_eq!(out, expected);
    }

    fn test_16bit_dtype<T>()
    where
        T: CppTypeToScalarType + FactoryValue + FromF64,
    {
        let tf = TensorFactory::<T>::new();

        let x = tf.make_default(
            vec![2, 3],
            vec![
                T::from_f64(1.5),
                T::from_f64(-2.0),
                T::from_f64(3.25),
                T::from_f64(4.0),
                T::from_f64(-5.5),
                T::from_f64(6.5),
            ],
        );
        let y = tf.make_default(vec![2, 1], vec![T::from_f64(10.0), T::from_f64(20.0)]);

        let inputs = vec![x, y];

        let out = tf.zeros_default(vec![2, 4]);

        let mut ctx = context();
        let _ret = cat_out(&mut ctx, list(&inputs), 1, &out);

        let expected = tf.make_default(
            vec![2, 4],
            vec![
                T::from_f64(1.5),
                T::from_f64(-2.0),
                T::from_f64(3.25),
                T::from_f64(10.0),
                T::from_f64(4.0),
                T::from_f64(-5.5),
                T::from_f64(6.5),
                T::from_f64(20.0),
            ],
        );
        assert_tensor_eq!(out, expected);
    }

    trait FromI32 {
        fn from_i32(v: i32) -> Self;
    }
    macro_rules! impl_from_i32 {
        ($($t:ty),*) => {$(impl FromI32 for $t { fn from_i32(v: i32) -> Self { v as $t } })*};
    }
    impl_from_i32!(u8, i8, i16, i32, i64, f32, f64);
    impl FromI32 for bool {
        fn from_i32(v: i32) -> Self {
            v != 0
        }
    }

    trait FromF64 {
        fn from_f64(v: f64) -> Self;
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

    // [spec:et:sem:op-cat.torch.executor.native.cat-out-fn/test]
    // also verifies check_cat_args (arg validation gate) and
    // get_cat_out_target_size (out-shape computation asserted below)
    // [spec:et:sem:copy-ops-util.torch.executor.check-cat-args-fn/test]
    // [spec:et:sem:copy-ops-util.torch.executor.get-cat-out-target-size-fn/test]
    #[test]
    fn op_cat_out_test_smoke_dim1() {
        let tf = TensorFactory::<i32>::new();

        #[rustfmt::skip]
        let x = tf.make_default(
            vec![2, 3],
            vec![
                1, 2, 3,
                4, 5, 6,
            ],
        );
        #[rustfmt::skip]
        let y = tf.make_default(
            vec![2, 1],
            vec![
                10,
                20,
            ],
        );

        let inputs = vec![x, y];

        let out = tf.zeros_default(vec![2, 4]);

        let mut ctx = context();
        let ret = cat_out(&mut ctx, list(&inputs), 1, &out);

        assert_tensor_eq!(*ret, out);

        #[rustfmt::skip]
        let expected = tf.make_default(
            vec![2, 4],
            vec![
                1, 2, 3, 10,
                4, 5, 6, 20,
            ],
        );

        assert_tensor_eq!(out, expected);
    }

    // PORT-NOTE: C++ guards with `ET_SKIP_IF(is_aten, ...)`; non-ATen build runs.
    // [spec:et:sem:op-cat.torch.executor.native.cat-out-fn/test]
    #[test]
    fn op_cat_out_test_sixteen_bit_float_support() {
        test_16bit_dtype::<Half>();
        test_16bit_dtype::<BFloat16>();
    }

    // PORT-NOTE: `ComplexSupport` exercises complex element dtypes. The Rust
    // `TensorFactory` has no complex element type; the complex branch of `cat_out`
    // is reachable but cannot be built via the factory. Ported and ignored.
    // [spec:et:sem:op-cat.torch.executor.native.cat-out-fn/test]
    #[test]
    #[ignore = "complex dtypes unsupported by TensorFactory"]
    fn op_cat_out_test_complex_support() {}

    // [spec:et:sem:op-cat.torch.executor.native.cat-out-fn/test]
    #[test]
    fn op_cat_out_test_negative_dims() {
        let tf = TensorFactory::<i32>::new();

        #[rustfmt::skip]
        let x = tf.make_default(
            vec![2, 2],
            vec![
                1, 2,
                3, 4,
            ],
        );
        #[rustfmt::skip]
        let y = tf.make_default(
            vec![2, 2],
            vec![
                10, 20,
                30, 40,
            ],
        );

        let inputs = vec![x, y];

        let out_neg1 = tf.zeros_default(vec![2, 4]);
        let mut ctx = context();
        cat_out(&mut ctx, list(&inputs), -1, &out_neg1);

        let out_1 = tf.zeros_default(vec![2, 4]);
        cat_out(&mut ctx, list(&inputs), 1, &out_1);

        assert_tensor_eq!(out_neg1, out_1);

        let out_neg2 = tf.zeros_default(vec![4, 2]);
        cat_out(&mut ctx, list(&inputs), -2, &out_neg2);

        let out_0 = tf.zeros_default(vec![4, 2]);
        cat_out(&mut ctx, list(&inputs), 0, &out_0);

        assert_tensor_eq!(out_neg2, out_0);
    }

    // [spec:et:sem:op-cat.torch.executor.native.cat-out-fn/test]
    #[test]
    fn op_cat_out_test_all_dtypes_supported() {
        test_dtype::<u8>();
        test_dtype::<i8>();
        test_dtype::<i16>();
        test_dtype::<i32>();
        test_dtype::<i64>();
        test_dtype::<f32>();
        test_dtype::<f64>();
        test_dtype::<bool>();
    }

    // PORT-NOTE: C++ guards with `ET_SKIP_IF(is_aten, ...)`; non-ATen build runs.
    // [spec:et:sem:op-cat.torch.executor.native.cat-out-fn/test]
    #[test]
    fn op_cat_out_test_empty_input_tensor_shape_ignored() {
        let tf = TensorFactory::<i32>::new();

        let empty = tf.make_default(vec![0, 10, 3], vec![]);
        assert_eq!(empty.numel(), 0);

        let x = tf.ones_default(vec![2, 2]);

        let inputs = vec![
            tf.ones_default(vec![2, 2]),
            empty,
            tf.ones_default(vec![2, 2]),
        ];
        let _ = &x;

        let out = tf.zeros_default(vec![4, 2]);

        let mut ctx = context();
        cat_out(&mut ctx, list(&inputs), 0, &out);
    }

    // [spec:et:sem:op-cat.torch.executor.native.cat-out-fn/test]
    #[test]
    fn op_cat_out_test_dim_bounds() {
        let tf = TensorFactory::<i32>::new();

        let x = tf.ones_default(vec![2, 2]);
        let inputs = [x];

        let out = tf.zeros_default(vec![2, 2]);

        let valid_dims: [i64; 4] = [0, 1, -1, -2];
        let mut ctx = context();
        for dim in valid_dims {
            cat_out(&mut ctx, list(&inputs), dim, &out);
        }

        let invalid_dims: [i64; 2] = [2, -3];
        for dim in invalid_dims {
            let mut ctx = context();
            et_expect_kernel_failure!(ctx, cat_out(&mut ctx, list(&inputs), dim, &out));
        }
    }

    // [spec:et:sem:op-cat.torch.executor.native.cat-out-fn/test]
    #[test]
    fn op_cat_out_test_no_input_tensors_with_non_empty_output_dies() {
        let tf = TensorFactory::<i32>::new();
        let out = tf.ones_default(vec![1]);

        let mut ctx = context();
        et_expect_kernel_failure!(ctx, cat_out(&mut ctx, ArrayRef::new(), 0, &out));
    }

    // [spec:et:sem:op-cat.torch.executor.native.cat-out-fn/test]
    #[test]
    fn op_cat_out_test_no_input_tensors_with_empty_output_dies() {
        let tf = TensorFactory::<i32>::new();

        let out = tf.make_default(vec![0], vec![]);
        assert_eq!(out.numel(), 0);

        let mut ctx = context();
        et_expect_kernel_failure!(ctx, cat_out(&mut ctx, ArrayRef::new(), 0, &out));
    }

    // [spec:et:sem:op-cat.torch.executor.native.cat-out-fn/test]
    #[test]
    fn op_cat_out_test_mismatched_dtypes_dies() {
        let tf_int = TensorFactory::<i32>::new();
        let tf_float = TensorFactory::<f32>::new();
        let out = tf_int.zeros_default(vec![4, 2]);

        let inputs = vec![tf_float.ones_default(vec![2, 2])];

        let mut ctx = context();
        et_expect_kernel_failure!(ctx, cat_out(&mut ctx, list(&inputs), 0, &out));
    }

    // PORT-NOTE: C++ guards with `ET_SKIP_IF(is_aten, ...)`; non-ATen build runs.
    // [spec:et:sem:op-cat.torch.executor.native.cat-out-fn/test]
    #[test]
    fn op_cat_out_test_mismatched_dimensions_dies() {
        let tf = TensorFactory::<i32>::new();
        let out = tf.zeros_default(vec![2, 2]);

        let inputs = vec![tf.ones_default(vec![1, 1, 1, 1])];

        let mut ctx = context();
        et_expect_kernel_failure!(ctx, cat_out(&mut ctx, list(&inputs), 0, &out));
    }

    // PORT-NOTE: C++ guards with `ET_SKIP_IF(is_aten, ...)`; non-ATen build runs.
    // [spec:et:sem:op-cat.torch.executor.native.cat-out-fn/test]
    #[test]
    fn op_cat_out_test_mismatched_dimension_size_dies() {
        let tf = TensorFactory::<i32>::new();
        let out = tf.zeros_default(vec![2, 2]);

        let inputs = vec![tf.ones_default(vec![2, 3])];

        let mut ctx = context();
        et_expect_kernel_failure!(ctx, cat_out(&mut ctx, list(&inputs), 0, &out));
    }

    // PORT-NOTE: C++ guards with `ET_SKIP_IF(is_aten, ...)`; non-ATen build runs.
    // [spec:et:sem:op-cat.torch.executor.native.cat-out-fn/test]
    #[test]
    fn op_cat_out_test_wrong_out_shape_dies() {
        let tf = TensorFactory::<i32>::new();

        let out = tf.zeros_default(vec![4, 5]);

        let inputs = vec![tf.ones_default(vec![2, 3]), tf.ones_default(vec![2, 3])];

        let mut ctx = context();
        et_expect_kernel_failure!(ctx, cat_out(&mut ctx, list(&inputs), 0, &out));
    }

    fn dynamic_shape_inputs(tf: &TensorFactory<i32>) -> (Vec<Tensor<'_>>, Tensor<'_>) {
        let xv = vec![
            tf.make_default(vec![2, 3], vec![4, 9, 3, 0, 3, 9]),
            tf.make_default(vec![2, 3], vec![7, 3, 7, 3, 1, 6]),
            tf.make_default(vec![2, 3], vec![6, 9, 8, 6, 6, 8]),
            tf.make_default(vec![2, 3], vec![4, 3, 6, 9, 1, 4]),
        ];
        let expected = tf.make_default(
            vec![8, 3],
            vec![
                4, 9, 3, 0, 3, 9, 7, 3, 7, 3, 1, 6, 6, 9, 8, 6, 6, 8, 4, 3, 6, 9, 1, 4,
            ],
        );
        (xv, expected)
    }

    // [spec:et:sem:op-cat.torch.executor.native.cat-out-fn/test]
    #[test]
    fn op_cat_out_test_dynamic_shape_upper_bound_same_as_expected() {
        let tf = TensorFactory::<i32>::new();
        let (xv, expected) = dynamic_shape_inputs(&tf);

        let out = tf.zeros(vec![8, 3], TensorShapeDynamism::DYNAMIC_BOUND);
        let mut ctx = context();
        cat_out(&mut ctx, list(&xv), 0, &out);
        assert_tensor_eq!(out, expected);
    }

    // [spec:et:sem:op-cat.torch.executor.native.cat-out-fn/test]
    #[test]
    fn op_cat_out_test_dynamic_shape_upper_bound_larger_than_expected() {
        let tf = TensorFactory::<i32>::new();
        let (xv, expected) = dynamic_shape_inputs(&tf);

        let out = tf.zeros(vec![10, 10], TensorShapeDynamism::DYNAMIC_BOUND);
        let mut ctx = context();
        cat_out(&mut ctx, list(&xv), 0, &out);
        assert_tensor_eq!(out, expected);
    }

    // PORT-NOTE: C++ guards with `ET_SKIP_IF(!output_resize, ...)`. The portable
    // (non-ATen) build sets `output_resize: false` (supported_features_def_example.yaml),
    // so this test is skipped there. Ignored to match; DYNAMIC_UNBOUND resize from
    // {1,1} to {8,3} is unsupported by the portable resize path.
    // [spec:et:sem:op-cat.torch.executor.native.cat-out-fn/test]
    #[test]
    #[ignore = "output_resize unsupported in portable build (ET_SKIP_IF !output_resize)"]
    fn op_cat_out_test_dynamic_shape_unbound() {
        let tf = TensorFactory::<i32>::new();
        let (xv, expected) = dynamic_shape_inputs(&tf);

        let out = tf.zeros(vec![1, 1], TensorShapeDynamism::DYNAMIC_UNBOUND);
        let mut ctx = context();
        cat_out(&mut ctx, list(&xv), 0, &out);
        assert_tensor_eq!(out, expected);
    }
}
