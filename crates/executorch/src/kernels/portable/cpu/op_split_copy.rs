//! Literal port of kernels/portable/cpu/op_split_copy.cpp.

use crate::kernels::portable::cpu::util::copy_ops_util::check_split_copy_args;
use crate::kernels::portable::cpu::util::dtype_util::internal::StaticCast;
use crate::runtime::core::array_ref::ArrayRef;
use crate::runtime::core::exec_aten::util::tensor_util::{
    getLeadingDims, getTrailingDims, tensors_have_same_dim_order2,
};
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::core::portable_type::tensor_impl::ssize_t;
use crate::runtime::core::portable_type::{BFloat16, Half};
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

// PORT-NOTE: `TensorList` (kernel_includes.h) is
// `executorch::aten::ArrayRef<Tensor>`.
type TensorList<'a> = ArrayRef<Tensor<'a>>;

// PORT-NOTE: local port of the scalar_type_util `convert<To, From>` template.
// Two C++ overloads: floating `From` -> integral `To` goes through
// `static_cast<int64_t>` first; every other pair is a plain `static_cast<To>`.
// `StaticCast` reproduces `static_cast` for the ET scalar ctypes. (Mirrors
// op_unbind_copy's `Convert` scaffolding.)
trait Convert<From> {
    fn convert(val: From) -> Self;
}

macro_rules! impl_convert_plain {
    ($to:ty, $from:ty) => {
        impl Convert<$from> for $to {
            #[inline]
            fn convert(val: $from) -> Self {
                <$to as StaticCast<$from>>::static_cast(val)
            }
        }
    };
}
macro_rules! impl_convert_float_to_int {
    ($to:ty, $from:ty) => {
        impl Convert<$from> for $to {
            #[inline]
            fn convert(val: $from) -> Self {
                <$to as StaticCast<i64>>::static_cast(<i64 as StaticCast<$from>>::static_cast(val))
            }
        }
    };
}

macro_rules! impl_convert_row_plain {
    ($to:ty) => {
        impl_convert_plain!($to, u8);
        impl_convert_plain!($to, i8);
        impl_convert_plain!($to, i16);
        impl_convert_plain!($to, i32);
        impl_convert_plain!($to, i64);
        impl_convert_plain!($to, bool);
        impl_convert_plain!($to, f32);
        impl_convert_plain!($to, f64);
        impl_convert_plain!($to, Half);
        impl_convert_plain!($to, BFloat16);
    };
}
impl_convert_row_plain!(f32);
impl_convert_row_plain!(f64);
impl_convert_row_plain!(Half);
impl_convert_row_plain!(BFloat16);

macro_rules! impl_convert_row_int {
    ($to:ty) => {
        impl_convert_plain!($to, u8);
        impl_convert_plain!($to, i8);
        impl_convert_plain!($to, i16);
        impl_convert_plain!($to, i32);
        impl_convert_plain!($to, i64);
        impl_convert_plain!($to, bool);
        impl_convert_float_to_int!($to, f32);
        impl_convert_float_to_int!($to, f64);
        impl_convert_float_to_int!($to, Half);
        impl_convert_float_to_int!($to, BFloat16);
    };
}
impl_convert_row_int!(u8);
impl_convert_row_int!(i8);
impl_convert_row_int!(i16);
impl_convert_row_int!(i32);
impl_convert_row_int!(i64);
impl_convert_row_int!(bool);

/// Splits the tensor into chunks of size `split_size` along the specified
/// dimension.
///
/// The last chunk will be smaller if the tensor size along the given dimension
/// dim is not evenly divisible by `split_size`.
///
/// split_copy.Tensor_out(Tensor input, int split_size, int dim=0, *,
/// Tensor(a!)[] out) -> ()
// [spec:et:def:op-split-copy.torch.executor.native.split-copy-tensor-out-fn]
// [spec:et:sem:op-split-copy.torch.executor.native.split-copy-tensor-out-fn]
pub fn split_copy_Tensor_out(
    ctx: &mut KernelRuntimeContext,
    input: &Tensor,
    split_size: i64,
    mut dim: i64,
    out: TensorList,
) {
    let _ = &ctx;
    // Support python-style negative indexing.
    if dim < 0 {
        dim += input.dim() as i64;
    }

    crate::et_kernel_check!(
        ctx,
        check_split_copy_args(input, split_size, dim, out),
        InvalidArgument
    );

    for i in 0..out.size() {
        crate::et_kernel_check!(
            ctx,
            tensors_have_same_dim_order2(input, out.at(i)),
            InvalidArgument
        );
    }

    let leading_dims: usize = getLeadingDims(input, dim);
    let trailing_dims: usize = getTrailingDims(input, dim);
    let step: usize = input.size(dim as ssize_t) as usize * trailing_dims;

    let in_type = input.scalar_type();
    let out_type = out.at(0).scalar_type();

    crate::et_switch_realhbbf16_types!(in_type, ctx, "split_copy.Tensor_out", CTYPE_IN, {
        crate::et_switch_realhbbf16_types!(out_type, ctx, "split_copy.Tensor_out", CTYPE_OUT, {
            let mut input_data: *const CTYPE_IN = input.const_data_ptr::<CTYPE_IN>();
            let e = out.size();
            for i in 0..e {
                let out_step: usize = out.at(i).size(dim as ssize_t) as usize * trailing_dims;
                if out_step == 0 {
                    continue;
                }
                let mut src: *const CTYPE_IN = input_data;
                let mut dest: *mut CTYPE_OUT = out.at(i).mutable_data_ptr::<CTYPE_OUT>();
                for _j in 0..leading_dims {
                    for k in 0..out_step {
                        unsafe {
                            *dest.add(k) = <CTYPE_OUT as Convert<CTYPE_IN>>::convert(*src.add(k));
                        }
                    }
                    src = unsafe { src.add(step) };
                    dest = unsafe { dest.add(out_step) };
                }
                input_data = unsafe { input_data.add(out_step) };
            }
        });
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assert_tensor_lists_eq;
    use crate::runtime::core::error::Error;
    use crate::runtime::core::exec_aten::testing_util::tensor_factory::{
        FactoryValue, TensorFactory,
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

    // Wraps a `&[Tensor]` in the non-owning `ArrayRef<Tensor>` the kernel expects.
    // Mirrors the C++ `TensorList out` handle over the output vector.
    fn tensor_list<'t>(v: &'t [Tensor<'t>]) -> ArrayRef<Tensor<'t>> {
        ArrayRef::from_raw_parts(v.as_ptr(), v.len())
    }

    // PORT-NOTE: `static_cast<CTYPE>(int)` bridge for building integer literal data
    // in the REALHBBF16 factory element types used by the templated test helpers.
    trait FromI32Data: Copy {
        fn from_i32(v: i32) -> Self;
    }
    macro_rules! impl_from_i32_data_num {
        ($($t:ty),*) => {$(impl FromI32Data for $t { fn from_i32(v: i32) -> Self { v as $t } })*};
    }
    impl_from_i32_data_num!(u8, i8, i16, i32, i64, f32, f64);
    impl FromI32Data for Half {
        fn from_i32(v: i32) -> Self {
            Half::from_f32(v as f32)
        }
    }
    impl FromI32Data for BFloat16 {
        fn from_i32(v: i32) -> Self {
            BFloat16::from_f32(v as f32)
        }
    }
    impl FromI32Data for bool {
        fn from_i32(v: i32) -> Self {
            v != 0
        }
    }

    trait SplitElem: CppTypeToScalarType + FactoryValue + FromI32Data {}
    impl<T: CppTypeToScalarType + FactoryValue + FromI32Data> SplitElem for T {}

    fn d<T: FromI32Data>(v: &[i32]) -> Vec<T> {
        v.iter().map(|&x| T::from_i32(x)).collect()
    }

    fn make3x3x3<T: SplitElem>(tf: &TensorFactory<T>) -> Tensor<'_> {
        #[rustfmt::skip]
        let data = d::<T>(&[
             0,  1,  2,
             3,  4,  5,
             6,  7,  8,
             9, 10, 11,
            12, 13, 14,
            15, 16, 17,
            18, 19, 20,
            21, 22, 23,
            24, 25, 26,
        ]);
        tf.make_default(vec![3, 3, 3], data)
    }

    // A simple successful test case that will work for any real dtype and bool.
    fn test_dtype<T: SplitElem>() {
        let tf = TensorFactory::<T>::new();

        let input = tf.make_default(vec![2, 2], d::<T>(&[1, 0, 0, 1]));

        let expected_out = vec![
            tf.make_default(vec![1, 2], d::<T>(&[1, 0])),
            tf.make_default(vec![1, 2], d::<T>(&[0, 1])),
        ];
        let out = vec![tf.zeros_default(vec![1, 2]), tf.zeros_default(vec![1, 2])];

        let mut ctx = context();
        split_copy_Tensor_out(&mut ctx, &input, 1, 0, tensor_list(&out));

        assert_tensor_lists_eq!(out, expected_out);
    }

    fn test_dynamic_shape(out_shape: Vec<i32>, dynamism: TensorShapeDynamism) {
        let tf = TensorFactory::<i32>::new();

        let x = tf.make_default(
            vec![2, 9],
            vec![4, 9, 3, 0, 3, 9, 7, 3, 7, 3, 1, 6, 6, 9, 8, 6, 6, 8],
        );
        let expected = vec![
            tf.make_default(vec![2, 3], vec![4, 9, 3, 3, 1, 6]),
            tf.make_default(vec![2, 3], vec![0, 3, 9, 6, 9, 8]),
            tf.make_default(vec![2, 3], vec![7, 3, 7, 6, 6, 8]),
        ];

        let out = vec![
            tf.zeros(out_shape.clone(), dynamism),
            tf.zeros(out_shape.clone(), dynamism),
            tf.zeros(out_shape, dynamism),
        ];
        let mut ctx = context();
        split_copy_Tensor_out(&mut ctx, &x, 3, 1, tensor_list(&out));
        assert_tensor_lists_eq!(out, expected);
    }

    // [spec:et:sem:op-split-copy.torch.executor.native.split-copy-tensor-out-fn/test]
    // also verifies check_split_copy_args (split_size=2 < dim_size=3 -> ceil
    // out-count=2, remainder=1 for the final chunk, per-output shape validation)
    // [spec:et:sem:copy-ops-util.torch.executor.check-split-copy-args-fn/test]
    #[test]
    fn op_split_copy_tensor_out_test_split3x3x3_on_dim0() {
        let tf = TensorFactory::<i32>::new();

        let expected_out = vec![
            tf.make_default(
                vec![2, 3, 3],
                vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
            ),
            tf.make_default(vec![1, 3, 3], vec![18, 19, 20, 21, 22, 23, 24, 25, 26]),
        ];

        let input = make3x3x3(&tf);

        let out = vec![
            tf.zeros_default(vec![2, 3, 3]),
            tf.zeros_default(vec![1, 3, 3]),
        ];
        let mut ctx = context();
        split_copy_Tensor_out(&mut ctx, &input, 2, 0, tensor_list(&out));
        assert_tensor_lists_eq!(expected_out, out);

        // Also show that python negative indexing works for this case.
        let out2 = vec![
            tf.zeros_default(vec![2, 3, 3]),
            tf.zeros_default(vec![1, 3, 3]),
        ];
        split_copy_Tensor_out(&mut ctx, &input, 2, -3, tensor_list(&out2));
        assert_tensor_lists_eq!(expected_out, out2);
    }

    // [spec:et:sem:op-split-copy.torch.executor.native.split-copy-tensor-out-fn/test]
    #[test]
    fn op_split_copy_tensor_out_test_split3x3x3_on_dim1() {
        let tf = TensorFactory::<i32>::new();

        let expected_out = vec![
            tf.make_default(
                vec![3, 2, 3],
                vec![
                    0, 1, 2, 3, 4, 5, 9, 10, 11, 12, 13, 14, 18, 19, 20, 21, 22, 23,
                ],
            ),
            tf.make_default(vec![3, 1, 3], vec![6, 7, 8, 15, 16, 17, 24, 25, 26]),
        ];

        let input = make3x3x3(&tf);

        let out = vec![
            tf.zeros_default(vec![3, 2, 3]),
            tf.zeros_default(vec![3, 1, 3]),
        ];
        let mut ctx = context();
        split_copy_Tensor_out(&mut ctx, &input, 2, 1, tensor_list(&out));
        assert_tensor_lists_eq!(expected_out, out);

        // Also show that python negative indexing works for this case.
        let out2 = vec![
            tf.zeros_default(vec![3, 2, 3]),
            tf.zeros_default(vec![3, 1, 3]),
        ];
        split_copy_Tensor_out(&mut ctx, &input, 2, -2, tensor_list(&out2));
        assert_tensor_lists_eq!(expected_out, out2);
    }

    // [spec:et:sem:op-split-copy.torch.executor.native.split-copy-tensor-out-fn/test]
    #[test]
    fn op_split_copy_tensor_out_test_split3x3x3_on_dim2() {
        let tf = TensorFactory::<i32>::new();

        let expected_out = vec![
            tf.make_default(
                vec![3, 3, 2],
                vec![
                    0, 1, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16, 18, 19, 21, 22, 24, 25,
                ],
            ),
            tf.make_default(vec![3, 3, 1], vec![2, 5, 8, 11, 14, 17, 20, 23, 26]),
        ];

        let input = make3x3x3(&tf);

        let out = vec![
            tf.zeros_default(vec![3, 3, 2]),
            tf.zeros_default(vec![3, 3, 1]),
        ];
        let mut ctx = context();
        split_copy_Tensor_out(&mut ctx, &input, 2, 2, tensor_list(&out));
        assert_tensor_lists_eq!(expected_out, out);

        // Also show that python negative indexing works for this case.
        let out2 = vec![
            tf.zeros_default(vec![3, 3, 2]),
            tf.zeros_default(vec![3, 3, 1]),
        ];
        split_copy_Tensor_out(&mut ctx, &input, 2, -1, tensor_list(&out2));
        assert_tensor_lists_eq!(expected_out, out2);
    }

    // [spec:et:sem:op-split-copy.torch.executor.native.split-copy-tensor-out-fn/test]
    #[test]
    fn op_split_copy_tensor_out_test_larger_split_size_does_nothing() {
        let tf = TensorFactory::<i32>::new();

        let input = make3x3x3(&tf);

        // Since split_size will be >= the largest dimension, slicing along any
        // dimension should return the unmodified input as the only output entry.
        let expected_out = vec![make3x3x3(&tf)];

        let mut ctx = context();
        for split_size in 3..6i64 {
            for dim in 0..input.dim() as i64 {
                let out = vec![tf.zeros_default(vec![3, 3, 3])];
                split_copy_Tensor_out(&mut ctx, &input, split_size, dim, tensor_list(&out));
                assert_tensor_lists_eq!(out, expected_out);
            }
        }
    }

    // [spec:et:sem:op-split-copy.torch.executor.native.split-copy-tensor-out-fn/test]
    #[test]
    fn op_split_copy_tensor_out_test_all_dtypes_supported() {
        // ET_FORALL_REALHBBF16_TYPES
        test_dtype::<u8>();
        test_dtype::<i8>();
        test_dtype::<i16>();
        test_dtype::<i32>();
        test_dtype::<i64>();
        test_dtype::<f32>();
        test_dtype::<f64>();
        test_dtype::<Half>();
        test_dtype::<BFloat16>();
        test_dtype::<bool>();
    }

    // [spec:et:sem:op-split-copy.torch.executor.native.split-copy-tensor-out-fn/test]
    #[test]
    fn op_split_copy_tensor_out_test_empty_input_tensor() {
        let tf = TensorFactory::<i32>::new();

        let input = tf.ones_default(vec![0]);
        assert_eq!(input.numel(), 0);

        let expected_out = vec![tf.ones_default(vec![0])];

        // Splitting a zero-size tensor succeeds, even for split_size zero.
        let out = vec![tf.zeros_default(vec![0])];
        let mut ctx = context();
        for split_size in 0..3i64 {
            split_copy_Tensor_out(&mut ctx, &input, split_size, 0, tensor_list(&out));
            assert_tensor_lists_eq!(out, expected_out);
        }
    }

    // [spec:et:sem:op-split-copy.torch.executor.native.split-copy-tensor-out-fn/test]
    #[test]
    fn op_split_copy_tensor_out_test_zero_dimensional_input_tensor_dies() {
        let tf = TensorFactory::<i32>::new();

        let input = tf.ones_default(vec![]);
        // Arbitrary output shape since this input can't be split.
        let out = vec![tf.zeros_default(vec![])];

        let mut ctx = context();
        et_expect_kernel_failure!(
            ctx,
            split_copy_Tensor_out(&mut ctx, &input, 1, 0, tensor_list(&out))
        );
    }

    // [spec:et:sem:op-split-copy.torch.executor.native.split-copy-tensor-out-fn/test]
    #[test]
    fn op_split_copy_tensor_out_test_zero_split_size_only_works_for_zero_size_dims() {
        let tf = TensorFactory::<i32>::new();

        let input = tf.ones_default(vec![1, 0, 2]);
        assert_eq!(input.numel(), 0);

        let expected_out = vec![tf.ones_default(vec![1, 0, 2])];

        let out = vec![tf.zeros_default(vec![1, 0, 2])];

        let mut ctx = context();
        // Fails when trying to split with size zero on a dim with size > 0.
        et_expect_kernel_failure!(
            ctx,
            split_copy_Tensor_out(&mut ctx, &input, 0, 0, tensor_list(&out))
        );

        // Successfully splits with size zero on a dim with size == 0.
        let mut ctx = context();
        split_copy_Tensor_out(&mut ctx, &input, 0, 1, tensor_list(&out));
        assert_tensor_lists_eq!(out, expected_out);

        // Fails again when trying to split with size zero on a dim with size > 0.
        et_expect_kernel_failure!(
            ctx,
            split_copy_Tensor_out(&mut ctx, &input, 0, 2, tensor_list(&out))
        );
    }

    // [spec:et:sem:op-split-copy.torch.executor.native.split-copy-tensor-out-fn/test]
    #[test]
    fn op_split_copy_tensor_out_test_negative_split_size_fails() {
        let tf = TensorFactory::<i32>::new();

        let input = tf.ones_default(vec![2, 2]);
        // Arbitrary output shape since there's no actual valid size.
        let out = vec![tf.zeros_default(vec![2, 2])];

        let mut ctx = context();
        et_expect_kernel_failure!(
            ctx,
            split_copy_Tensor_out(&mut ctx, &input, -1, 0, tensor_list(&out))
        );
    }

    // [spec:et:sem:op-split-copy.torch.executor.native.split-copy-tensor-out-fn/test]
    #[test]
    fn op_split_copy_tensor_out_test_out_of_range_dims_die() {
        let tf = TensorFactory::<i32>::new();

        let input = tf.ones_default(vec![2, 2]);

        let good_dims: Vec<i64> = vec![-2, -1, 0, 1];
        let bad_dims: Vec<i64> = vec![-4, -3, 2, 3];

        // Since split_size is >= the largest dimension, slicing along any
        // dimension should return the unmodified input as the only output entry.
        let split_size: i64 = 2;
        let expected_out = vec![tf.ones_default(vec![2, 2])];

        for dim in good_dims {
            let out = vec![tf.zeros_default(vec![2, 2])];
            let mut ctx = context();
            split_copy_Tensor_out(&mut ctx, &input, split_size, dim, tensor_list(&out));
            assert_tensor_lists_eq!(out, expected_out);
        }

        for dim in bad_dims {
            let out = vec![tf.zeros_default(vec![2, 2])];
            let mut ctx = context();
            et_expect_kernel_failure!(
                ctx,
                split_copy_Tensor_out(&mut ctx, &input, split_size, dim, tensor_list(&out))
            );
        }
    }

    // PORT-NOTE: C++ `DtypeMismatchDies` is `ET_SKIP_IF(true, ...)` (always
    // skipped). Ported and ignored.
    // [spec:et:sem:op-split-copy.torch.executor.native.split-copy-tensor-out-fn/test]
    #[test]
    #[ignore]
    fn op_split_copy_tensor_out_test_dtype_mismatch_dies() {
        let tf_int = TensorFactory::<i32>::new();
        let tf_float = TensorFactory::<f32>::new();

        let input = tf_int.ones_default(vec![2, 2]);

        let split_size: i64 = 2;
        let dim: i64 = 0;

        // Demonstrate that this setup works when the dtypes are the same.
        {
            let out = vec![tf_int.zeros_default(vec![2, 2])];
            let mut ctx = context();
            split_copy_Tensor_out(&mut ctx, &input, split_size, dim, tensor_list(&out));
            let expected = vec![tf_int.ones_default(vec![2, 2])];
            assert_tensor_lists_eq!(out, expected);
        }

        // Dies with the same setup but the output dtype is different.
        {
            let out = vec![tf_float.zeros_default(vec![2, 2])];
            let mut ctx = context();
            et_expect_kernel_failure!(
                ctx,
                split_copy_Tensor_out(&mut ctx, &input, split_size, dim, tensor_list(&out))
            );
        }
    }

    // [spec:et:sem:op-split-copy.torch.executor.native.split-copy-tensor-out-fn/test]
    #[test]
    fn op_split_copy_tensor_out_test_wrong_num_output_entries_dies() {
        let tf = TensorFactory::<i32>::new();

        let input = tf.ones_default(vec![3]);

        // Use a split_size that produces two output entries on success.
        let split_size: i64 = 2;
        let dim: i64 = 0;

        // Demonstrate that splitting the input should produce two output entries.
        {
            let expected_out = vec![tf.ones_default(vec![2]), tf.ones_default(vec![1])];
            let out = vec![tf.zeros_default(vec![2]), tf.zeros_default(vec![1])];
            let mut ctx = context();
            split_copy_Tensor_out(&mut ctx, &input, split_size, dim, tensor_list(&out));
            assert_tensor_lists_eq!(out, expected_out);
        }

        // Dies with the same setup but the output has one fewer entry than it should.
        {
            let out = vec![tf.zeros_default(vec![2])];
            let mut ctx = context();
            et_expect_kernel_failure!(
                ctx,
                split_copy_Tensor_out(&mut ctx, &input, split_size, dim, tensor_list(&out))
            );
        }

        // Dies with the same setup but the output has one more entry than it should.
        {
            let out = vec![
                tf.zeros_default(vec![2]),
                tf.zeros_default(vec![1]),
                tf.zeros_default(vec![1]),
            ];
            let mut ctx = context();
            et_expect_kernel_failure!(
                ctx,
                split_copy_Tensor_out(&mut ctx, &input, split_size, dim, tensor_list(&out))
            );
        }
    }

    // [spec:et:sem:op-split-copy.torch.executor.native.split-copy-tensor-out-fn/test]
    #[test]
    fn op_split_copy_tensor_out_test_wrong_output_shape_dies() {
        let tf = TensorFactory::<i32>::new();

        let input = tf.ones_default(vec![5, 3, 4]);

        // Use a split_size that produces two output entries on success.
        let split_size: i64 = 2;
        let dim: i64 = 1;

        // Demonstrate the shapes that this split should produce.
        {
            let expected_out = vec![
                tf.ones_default(vec![5, 2, 4]),
                tf.ones_default(vec![5, 1, 4]),
            ];
            let out = vec![
                tf.zeros_default(vec![5, 2, 4]),
                tf.zeros_default(vec![5, 1, 4]),
            ];
            let mut ctx = context();
            split_copy_Tensor_out(&mut ctx, &input, split_size, dim, tensor_list(&out));
            assert_tensor_lists_eq!(out, expected_out);
        }

        // Make each of the dimensions of the final element incorrect.
        {
            let out = vec![
                tf.zeros_default(vec![5, 2, 4]),
                tf.zeros_default(vec![6, 1, 4]),
            ];
            let mut ctx = context();
            et_expect_kernel_failure!(
                ctx,
                split_copy_Tensor_out(&mut ctx, &input, split_size, dim, tensor_list(&out))
            );
        }
        {
            let out = vec![
                tf.zeros_default(vec![5, 2, 4]),
                tf.zeros_default(vec![5, 2, 4]),
            ];
            let mut ctx = context();
            et_expect_kernel_failure!(
                ctx,
                split_copy_Tensor_out(&mut ctx, &input, split_size, dim, tensor_list(&out))
            );
        }
        {
            let out = vec![
                tf.zeros_default(vec![5, 2, 4]),
                tf.zeros_default(vec![5, 1, 5]),
            ];
            let mut ctx = context();
            et_expect_kernel_failure!(
                ctx,
                split_copy_Tensor_out(&mut ctx, &input, split_size, dim, tensor_list(&out))
            );
        }

        // Wrong size of the split dimension for the non-last output element.
        {
            let out = vec![
                tf.zeros_default(vec![5, 3, 4]),
                tf.zeros_default(vec![5, 1, 4]),
            ];
            let mut ctx = context();
            et_expect_kernel_failure!(
                ctx,
                split_copy_Tensor_out(&mut ctx, &input, split_size, dim, tensor_list(&out))
            );
        }

        // Wrong number of output dimensions.
        {
            let out = vec![
                tf.zeros_default(vec![5, 2, 4]),
                tf.zeros_default(vec![5, 1, 4, 2]),
            ];
            let mut ctx = context();
            et_expect_kernel_failure!(
                ctx,
                split_copy_Tensor_out(&mut ctx, &input, split_size, dim, tensor_list(&out))
            );
        }
        {
            let out = vec![
                tf.zeros_default(vec![5, 2, 4]),
                tf.zeros_default(vec![5, 1]),
            ];
            let mut ctx = context();
            et_expect_kernel_failure!(
                ctx,
                split_copy_Tensor_out(&mut ctx, &input, split_size, dim, tensor_list(&out))
            );
        }
    }

    // [spec:et:sem:op-split-copy.torch.executor.native.split-copy-tensor-out-fn/test]
    #[test]
    fn op_split_copy_tensor_out_test_dynamic_shape_upper_bound_same_as_expected() {
        test_dynamic_shape(vec![2, 3], TensorShapeDynamism::DYNAMIC_BOUND);
    }

    // PORT-NOTE: C++ `DISABLED_DynamicShapeUpperBoundLargerThanExpected` (dynamic
    // shape not supported). Ported and ignored.
    // [spec:et:sem:op-split-copy.torch.executor.native.split-copy-tensor-out-fn/test]
    #[test]
    #[ignore]
    fn op_split_copy_tensor_out_test_dynamic_shape_upper_bound_larger_than_expected() {
        test_dynamic_shape(vec![10, 10], TensorShapeDynamism::DYNAMIC_BOUND);
    }

    // PORT-NOTE: C++ `DISABLED_DynamicShapeUnbound` (dynamic shape not supported).
    // Ported and ignored.
    // [spec:et:sem:op-split-copy.torch.executor.native.split-copy-tensor-out-fn/test]
    #[test]
    #[ignore]
    fn op_split_copy_tensor_out_test_dynamic_shape_unbound() {
        test_dynamic_shape(vec![1, 1], TensorShapeDynamism::DYNAMIC_UNBOUND);
    }
}
