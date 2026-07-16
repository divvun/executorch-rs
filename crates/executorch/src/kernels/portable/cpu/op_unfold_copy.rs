//! Literal port of kernels/portable/cpu/op_unfold_copy.cpp.

use crate::kernels::portable::cpu::util::copy_ops_util::{
    check_unfold_copy_args, get_unfold_copy_out_target_size,
};
use crate::kernels::portable::cpu::util::dtype_util::internal::StaticCast;
use crate::runtime::core::array_ref::ArrayRef;
use crate::runtime::core::error::Error;
use crate::runtime::core::exec_aten::util::tensor_util::{
    K_TENSOR_DIMENSION_LIMIT, getLeadingDims, getTrailingDims, nonzero_dim, resize_tensor,
};
use crate::runtime::core::portable_type::scalar_type::ScalarType;
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::core::portable_type::tensor_impl::{SizesType, ssize_t};
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

// PORT-NOTE: `Tensor& out` / returned `Tensor&` become `&'a Tensor` (interior
// mutation through `*mut TensorImpl`).
// PORT-NOTE: local port of the scalar_type_util `convert<To, From>` template.
// Two C++ overloads: floating `From` -> integral `To` (`std::is_integral`
// includes `bool`) goes through `static_cast<int64_t>` first; every other pair
// is a plain `static_cast<To>`. `StaticCast` reproduces `static_cast`.
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
use crate::runtime::core::portable_type::{BFloat16, Half};
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

// unfold_copy(Tensor self, int dimension, int size, int step, *, Tensor(a!)
// out) -> Tensor(a!)
// [spec:et:def:op-unfold-copy.torch.executor.native.unfold-copy-out-fn]
// [spec:et:sem:op-unfold-copy.torch.executor.native.unfold-copy-out-fn]
pub fn unfold_copy_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    self_: &Tensor,
    mut dim: i64,
    size: i64,
    step: i64,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    // (void)ctx;
    // Check if dimension is valid
    crate::et_kernel_check!(
        ctx,
        check_unfold_copy_args(self_, dim, size, step),
        InvalidArgument,
        out
    );
    if dim < 0 {
        dim += nonzero_dim(self_) as i64;
    }
    // Calculate output size
    let mut expected_output_size: [SizesType; K_TENSOR_DIMENSION_LIMIT] =
        [0; K_TENSOR_DIMENSION_LIMIT];
    let mut expected_out_dim: usize = 0;

    unsafe {
        get_unfold_copy_out_target_size(
            self_,
            dim,
            size,
            step,
            expected_output_size.as_mut_ptr(),
            &mut expected_out_dim,
        );
    }

    crate::et_kernel_check!(
        ctx,
        resize_tensor(
            out,
            ArrayRef::from_raw_parts(expected_output_size.as_ptr(), expected_out_dim)
        ) == Error::Ok,
        InvalidArgument,
        out
    );

    // Copy data
    let leading_dims: usize = getLeadingDims(self_, dim);
    let trailing_dims: usize = getTrailingDims(self_, dim);
    let in_type: ScalarType = self_.scalar_type();
    let out_type: ScalarType = out.scalar_type();

    crate::et_switch_realhbbf16_types!(in_type, ctx, "unfold_copy.out", CTYPE_IN, {
        let input_ptr: *const CTYPE_IN = self_.const_data_ptr::<CTYPE_IN>();
        crate::et_switch_realhbbf16_types!(out_type, ctx, "unfold_copy.out", CTYPE_OUT, {
            let mut out_ptr: *mut CTYPE_OUT = out.mutable_data_ptr::<CTYPE_OUT>();
            for i in 0..leading_dims {
                let src: *const CTYPE_IN = unsafe {
                    input_ptr.add(i * self_.size(dim as ssize_t) as usize * trailing_dims)
                };
                for j in 0..out.size(dim as ssize_t) {
                    let dim_src: *const CTYPE_IN =
                        unsafe { src.add(j as usize * step as usize * trailing_dims) };
                    for k in 0..trailing_dims {
                        for l in 0..size {
                            unsafe {
                                *out_ptr = <CTYPE_OUT as Convert<CTYPE_IN>>::convert(
                                    *dim_src.add(k + l as usize * trailing_dims),
                                );
                                out_ptr = out_ptr.add(1);
                            }
                        }
                    }
                }
            }
        });
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
    use crate::runtime::core::tensor_shape_dynamism::TensorShapeDynamism;

    fn context() -> KernelRuntimeContext<'static> {
        // Mirrors `OperatorTest::SetUp()`'s `runtime_init()` PAL initialization.
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

    fn op_unfold_copy_out<'a, 'b>(
        ctx: &mut KernelRuntimeContext,
        self_: &Tensor,
        dim: i64,
        size: i64,
        step: i64,
        out: &'a Tensor<'b>,
    ) -> &'a Tensor<'b> {
        unfold_copy_out(ctx, self_, dim, size, step, out)
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

    fn make_i64<T: FromI64>(vals: &[i64]) -> Vec<T> {
        vals.iter().map(|&v| T::from_i64(v)).collect()
    }

    // template test_unfold_copy_dtype<CTYPE, DTYPE>
    fn test_unfold_copy_dtype<T>()
    where
        T: CppTypeToScalarType + FactoryValue + FromI64,
    {
        let tf = TensorFactory::<T>::new();

        let input = tf.make_default(vec![3, 3], make_i64(&[1, 2, 3, 4, 5, 6, 7, 8, 9]));
        let expected = tf.make_default(
            vec![3, 2, 2],
            make_i64(&[1, 2, 2, 3, 4, 5, 5, 6, 7, 8, 8, 9]),
        );
        let actual_out = tf.zeros_like(&expected, TensorShapeDynamism::STATIC);
        let mut ctx = context();
        op_unfold_copy_out(&mut ctx, &input, 1, 2, 1, &actual_out);
        assert_tensor_close!(actual_out, expected);
    }

    macro_rules! forall_realhbf16 {
        ($f:ident) => {{
            $f::<u8>();
            $f::<i8>();
            $f::<i16>();
            $f::<i32>();
            $f::<i64>();
            $f::<f32>();
            $f::<f64>();
            $f::<Half>();
            $f::<BFloat16>();
        }};
    }

    // [spec:et:sem:op-unfold-copy.torch.executor.native.unfold-copy-out-fn/test]
    // also verifies check_unfold_copy_args (arg gate) and
    // get_unfold_copy_out_target_size ((dim_size-size+step)/step at dim + size
    // appended, out {3,1,2})
    // [spec:et:sem:copy-ops-util.torch.executor.check-unfold-copy-args-fn/test]
    // [spec:et:sem:copy-ops-util.torch.executor.get-unfold-copy-out-target-size-fn/test]
    #[test]
    fn op_unfold_test_smoke_test() {
        let tf = TensorFactory::<f32>::new();
        let input = tf.make_default(
            vec![3, 3],
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        );
        let expected = tf.make_default(vec![3, 1, 2], vec![1.0, 2.0, 4.0, 5.0, 7.0, 8.0]);
        let output = tf.zeros_like(&expected, TensorShapeDynamism::STATIC);

        let mut ctx = context();
        op_unfold_copy_out(&mut ctx, &input, 1, 2, 2, &output);
        assert_tensor_close!(output, expected);
    }

    // [spec:et:sem:op-unfold-copy.torch.executor.native.unfold-copy-out-fn/test]
    #[test]
    fn op_unfold_test_d_type() {
        forall_realhbf16!(test_unfold_copy_dtype);
    }

    // [spec:et:sem:op-unfold-copy.torch.executor.native.unfold-copy-out-fn/test]
    #[test]
    fn op_unfold_test_zero_dimension() {
        let tf = TensorFactory::<f32>::new();
        let input = tf.make_default(
            vec![3, 3],
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        );
        let expected = tf.make_default(
            vec![2, 3, 2],
            vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0, 4.0, 7.0, 5.0, 8.0, 6.0, 9.0],
        );
        let output = tf.zeros_like(&expected, TensorShapeDynamism::STATIC);

        let mut ctx = context();
        op_unfold_copy_out(&mut ctx, &input, 0, 2, 1, &output);
        assert_tensor_close!(output, expected);
    }

    // [spec:et:sem:op-unfold-copy.torch.executor.native.unfold-copy-out-fn/test]
    #[test]
    fn op_unfold_test_negative_dimension() {
        let tf = TensorFactory::<f32>::new();
        let input = tf.make_default(
            vec![3, 3],
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        );
        let expected = tf.make_default(vec![3, 1, 2], vec![1.0, 2.0, 4.0, 5.0, 7.0, 8.0]);
        let output = tf.zeros_like(&expected, TensorShapeDynamism::STATIC);

        let mut ctx = context();
        op_unfold_copy_out(&mut ctx, &input, -1, 2, 2, &output);
        assert_tensor_close!(output, expected);
    }

    // [spec:et:sem:op-unfold-copy.torch.executor.native.unfold-copy-out-fn/test]
    #[test]
    fn op_unfold_test_large_step() {
        let tf = TensorFactory::<f32>::new();
        let input = tf.make_default(
            vec![3, 3],
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        );
        let expected = tf.make_default(vec![3, 1, 2], vec![1.0, 2.0, 4.0, 5.0, 7.0, 8.0]);
        let output = tf.zeros_like(&expected, TensorShapeDynamism::STATIC);

        let mut ctx = context();
        op_unfold_copy_out(&mut ctx, &input, -1, 2, 5, &output);
        assert_tensor_close!(output, expected);
    }

    // [spec:et:sem:op-unfold-copy.torch.executor.native.unfold-copy-out-fn/test]
    #[test]
    fn op_unfold_test_zero_size() {
        let tf = TensorFactory::<f32>::new();
        let input = tf.make_default(
            vec![3, 3],
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        );
        let expected = tf.make_default(vec![3, 4, 0], vec![]);
        let output = tf.zeros_like(&expected, TensorShapeDynamism::STATIC);

        let mut ctx = context();
        op_unfold_copy_out(&mut ctx, &input, 1, 0, 1, &output);
        assert_tensor_close!(output, expected);
    }

    // [spec:et:sem:op-unfold-copy.torch.executor.native.unfold-copy-out-fn/test]
    #[test]
    fn op_unfold_test_negative_size_and_negative_step_dies() {
        let tf = TensorFactory::<f32>::new();
        let input = tf.make_default(
            vec![3, 3],
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        );
        let output = tf.zeros_default(vec![3, 1, 2]);

        let mut ctx = context();
        et_expect_kernel_failure!(ctx, op_unfold_copy_out(&mut ctx, &input, 1, -1, 1, &output));
        let mut ctx = context();
        et_expect_kernel_failure!(ctx, op_unfold_copy_out(&mut ctx, &input, 1, 1, -1, &output));
    }

    // [spec:et:sem:op-unfold-copy.torch.executor.native.unfold-copy-out-fn/test]
    #[test]
    fn op_unfold_test_invalid_dim_and_size_too_large_dies() {
        let tf = TensorFactory::<f32>::new();
        let input = tf.make_default(
            vec![3, 3],
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        );
        let output = tf.zeros_default(vec![3, 1, 2]);

        let mut ctx = context();
        et_expect_kernel_failure!(ctx, op_unfold_copy_out(&mut ctx, &input, 3, 2, 1, &output));
        let mut ctx = context();
        et_expect_kernel_failure!(ctx, op_unfold_copy_out(&mut ctx, &input, 1, 10, 1, &output));
    }
}
