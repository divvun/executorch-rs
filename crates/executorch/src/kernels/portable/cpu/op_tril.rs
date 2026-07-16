//! Literal port of kernels/portable/cpu/op_tril.cpp.

use crate::kernels::portable::cpu::util::copy_ops_util::check_tril_args;
use crate::runtime::core::array_ref::IntArrayRef;
use crate::runtime::core::error::Error;
use crate::runtime::core::exec_aten::util::tensor_util::{
    K_TENSOR_DIMENSION_LIMIT, getLeadingDims, getTrailingDims, resize_tensor,
    tensor_is_default_dim_order, tensors_have_same_dim_order2,
};
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

// PORT-NOTE: `Tensor& out` / returned `Tensor&` become `&'a Tensor` (interior
// mutation through `*mut TensorImpl`). `ET_RESTRICT` is dropped (no aliasing
// annotation in the ported raw-pointer arithmetic).

// PORT-NOTE: local `ET_KERNEL_CHECK_MSG` mirror that forwards the full message +
// args (the crate-level macro drops args after the leading literal).
macro_rules! et_kernel_check_msg_fmt {
    ($context:expr, $cond:expr, $error:ident, $retval:expr, $fmt:literal $(, $($arg:tt)*)?) => {{
        if !($cond) {
            $crate::et_log!(
                Error,
                ::core::concat!("Check failed ({}): ", $fmt),
                ::core::stringify!($cond)
                $(, $($arg)*)?
            );
            $context.fail($crate::runtime::core::error::Error::$error);
            return $retval;
        }
    }};
}

/// Clears `out` by setting all elements to 0.
// [spec:et:def:op-tril.torch.executor.native.clear-out-fn]
// [spec:et:sem:op-tril.torch.executor.native.clear-out-fn]
fn clear_out<'a, 'b>(out: &'a Tensor<'b>) -> &'a Tensor<'b> {
    let out_data: *mut u8 = out.mutable_data_ptr::<u8>();
    if !out_data.is_null() {
        unsafe {
            core::ptr::write_bytes(out_data, 0, out.nbytes());
        }
    }
    out
}

/// Applies lower-triangular part of `self` to `out` using parameters defined.
/// This function is agnostic to whether `self` is a 2D matrix or batch of
/// matrices.
#[allow(clippy::too_many_arguments)]
fn apply_tril<CTYPE: Copy>(
    self_: *mut CTYPE,
    out: *mut CTYPE,
    diagonal: i64,
    num_rows: i64,
    num_cols: i64,
    row_stride: i64,
    col_stride: i64,
) {
    for i in 0..num_rows {
        for j in 0..core::cmp::min(num_cols, i + diagonal + 1) {
            unsafe {
                *out.offset((i * row_stride + j * col_stride) as isize) =
                    *self_.offset((i * row_stride + j * col_stride) as isize);
            }
        }
    }
}

/// `tril_out` helper function.
// [spec:et:def:op-tril.torch.executor.native.tril-kernel-fn]
// [spec:et:sem:op-tril.torch.executor.native.tril-kernel-fn]
fn tril_kernel<CTYPE: Copy>(
    ctx: &mut KernelRuntimeContext,
    self_: &Tensor,
    diagonal: i64,
    out: &Tensor,
) {
    // Dynamically compute `self` sizes and strides.

    let ndim: usize = self_.dim() as usize;

    et_kernel_check_msg_fmt!(
        ctx,
        ndim < K_TENSOR_DIMENSION_LIMIT,
        InvalidArgument,
        (),
        "ndim {} >= {}",
        ndim,
        K_TENSOR_DIMENSION_LIMIT
    );

    let mut sizes: [i64; K_TENSOR_DIMENSION_LIMIT] = [0; K_TENSOR_DIMENSION_LIMIT];
    let mut strides: [i64; K_TENSOR_DIMENSION_LIMIT] = [0; K_TENSOR_DIMENSION_LIMIT];

    for i in 0..ndim {
        sizes[i] = self_.size(i as isize) as i64;
        strides[i] = getTrailingDims(self_, i as i64) as i64;
    }

    let sizes_ref: IntArrayRef = IntArrayRef::from_raw_parts(sizes.as_ptr(), ndim);
    let strides_ref: IntArrayRef = IntArrayRef::from_raw_parts(strides.as_ptr(), ndim);

    let num_rows: i64 = *sizes_ref.at(ndim - 2);
    let num_cols: i64 = *sizes_ref.at(ndim - 1);

    // Compute `tril` for a 2D matrix or a batch of matrices. For a batch of
    // matrices, `batch_size` will be >1, and `apply_tril` will be executed
    // multiple times, each referencing a multiple of `self_stride`.

    let batch_size: i64 = getLeadingDims(self_, (ndim - 2) as i64) as i64;
    let self_stride: i64 = if self_.dim() > 2 && *strides_ref.at(ndim - 3) > 0 {
        *strides_ref.at(ndim - 3)
    } else {
        1
    };

    let data_self = self_.mutable_data_ptr::<CTYPE>();
    let data_out = out.mutable_data_ptr::<CTYPE>();

    let row_stride: i64 = *strides_ref.at(ndim - 2);
    let col_stride: i64 = *strides_ref.at(ndim - 1);

    for i in 0..batch_size {
        let data_self_ptr: *mut CTYPE = unsafe { data_self.offset((i * self_stride) as isize) };
        let data_out_ptr: *mut CTYPE = unsafe { data_out.offset((i * self_stride) as isize) };

        apply_tril::<CTYPE>(
            data_self_ptr,
            data_out_ptr,
            diagonal,
            num_rows,
            num_cols,
            row_stride,
            col_stride,
        );
    }
}

/// `tril_out` implementation for all dtypes (real + bool). Returns the
/// lower-triangular part of a 2D matrix or batch of matrices in `out`, where all
/// other elements are set to 0, by default. Further, `diagonal` controls how the
/// lower-triangular subset is defined:
///    1. `diagonal = 0`: Elements on and below the main diagonal are retained.
///    2. `diagonal > 0`: Similar to case (1); additional diagonals above the
///       main one are also captured.
///    3. `diagonal < 0`: Similar to case (1); additional diagonals below the
///       main one are also captured.
// [spec:et:def:op-tril.torch.executor.native.tril-out-fn]
// [spec:et:sem:op-tril.torch.executor.native.tril-out-fn]
pub fn tril_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    self_: &Tensor,
    diagonal: i64,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    // (void)ctx;

    crate::et_kernel_check!(ctx, check_tril_args(self_, out), InvalidArgument, out);

    crate::et_kernel_check!(
        ctx,
        resize_tensor(out, self_.sizes()) == Error::Ok,
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

    if self_.numel() == 0 {
        return out;
    }

    // Fill `out` with 0s prior to executing tril.
    clear_out(out);

    let out_type = out.scalar_type();
    crate::et_switch_realhbbf16_types!(out_type, ctx, "tril.out", CTYPE, {
        tril_kernel::<CTYPE>(ctx, self_, diagonal, out);
    });

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

    fn context() -> KernelRuntimeContext<'static> {
        crate::runtime::platform::platform::pal_init();
        KernelRuntimeContext::new(
            crate::extension::module::module::null_event_tracer(),
            core::ptr::null_mut::<MemoryAllocator>() as *mut dyn MemoryAllocatorBase,
        )
    }

    fn op_tril_out<'a, 'b>(
        ctx: &mut KernelRuntimeContext,
        self_: &Tensor,
        diagonal: i64,
        out: &'a Tensor<'b>,
    ) -> &'a Tensor<'b> {
        tril_out(ctx, self_, diagonal, out)
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

    trait TrilElem: CppTypeToScalarType + FactoryValue + FromI32Data {}
    impl<T: CppTypeToScalarType + FactoryValue + FromI32Data> TrilElem for T {}

    fn d<T: FromI32Data>(v: &[i32]) -> Vec<T> {
        v.iter().map(|&x| T::from_i32(x)).collect()
    }

    fn test_tril_out_zeros<T: TrilElem>() {
        let tf = TensorFactory::<T>::new();
        let self_ = tf.make_default(vec![3, 3], d(&[0, 0, 0, 0, 0, 0, 0, 0, 0]));
        let out = tf.zeros_default(vec![3, 3]);
        let mut ctx = context();
        op_tril_out(&mut ctx, &self_, 0, &out);
        let result = tf.make_default(vec![3, 3], d(&[0, 0, 0, 0, 0, 0, 0, 0, 0]));
        assert_tensor_eq!(out, result);
    }

    fn test_tril_out_ones<T: TrilElem>() {
        let tf = TensorFactory::<T>::new();
        let self_ = tf.make_default(vec![3, 3], d(&[0, 0, 0, 0, 0, 0, 0, 0, 0]));
        let out = tf.ones_default(vec![3, 3]);
        let mut ctx = context();
        op_tril_out(&mut ctx, &self_, 0, &out);
        let result = tf.make_default(vec![3, 3], d(&[0, 0, 0, 0, 0, 0, 0, 0, 0]));
        assert_tensor_eq!(out, result);
    }

    fn test_tril_out_empty_dims<T: TrilElem>() {
        let tf = TensorFactory::<T>::new();
        let out = tf.zeros_default(vec![1, 1, 1, 1]);
        let self_ = tf.ones_default(vec![1, 1, 1, 1]);
        let mut ctx = context();
        op_tril_out(&mut ctx, &self_, 0, &out);
        let result = tf.ones_default(vec![1, 1, 1, 1]);
        assert_tensor_eq!(out, result);
    }

    fn test_tril_out_square<T: TrilElem>() {
        let tf = TensorFactory::<T>::new();
        let self_ = tf.make_default(vec![3, 3], d(&[1, 1, 1, 1, 1, 1, 1, 1, 1]));
        let out = tf.zeros_default(vec![3, 3]);
        let mut ctx = context();
        op_tril_out(&mut ctx, &self_, 0, &out);
        let result = tf.make_default(vec![3, 3], d(&[1, 0, 0, 1, 1, 0, 1, 1, 1]));
        assert_tensor_eq!(out, result);
    }

    fn test_tril_out_rectangle<T: TrilElem>() {
        let tf = TensorFactory::<T>::new();
        let self_ = tf.make_default(vec![3, 5], d(&[1; 15]));
        let out = tf.zeros_default(vec![3, 5]);
        let mut ctx = context();
        op_tril_out(&mut ctx, &self_, 0, &out);
        let result = tf.make_default(
            vec![3, 5],
            d(&[1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0]),
        );
        assert_tensor_eq!(out, result);
    }

    fn test_tril_out_pos_diag<T: TrilElem>() {
        let tf = TensorFactory::<T>::new();
        let self_ = tf.make_default(vec![3, 3], d(&[1, 1, 1, 1, 1, 1, 1, 1, 1]));
        let out = tf.zeros_default(vec![3, 3]);
        let mut ctx = context();
        op_tril_out(&mut ctx, &self_, 1, &out);
        let result = tf.make_default(vec![3, 3], d(&[1, 1, 0, 1, 1, 1, 1, 1, 1]));
        assert_tensor_eq!(out, result);
    }

    fn test_tril_out_neg_diag<T: TrilElem>() {
        let tf = TensorFactory::<T>::new();
        let self_ = tf.make_default(vec![3, 3], d(&[1, 1, 1, 1, 1, 1, 1, 1, 1]));
        let out = tf.zeros_default(vec![3, 3]);
        let mut ctx = context();
        op_tril_out(&mut ctx, &self_, -1, &out);
        let result = tf.make_default(vec![3, 3], d(&[0, 0, 0, 1, 0, 0, 1, 1, 0]));
        assert_tensor_eq!(out, result);
    }

    fn test_tril_out_multi_equal_dim<T: TrilElem>() {
        let tf = TensorFactory::<T>::new();
        let self_ = tf.make_default(vec![3, 3, 3], d(&[1; 27]));
        let out = tf.zeros_default(vec![3, 3, 3]);
        let mut ctx = context();
        op_tril_out(&mut ctx, &self_, 0, &out);
        let result = tf.make_default(
            vec![3, 3, 3],
            d(&[
                1, 0, 0, 1, 1, 0, 1, 1, 1, //
                1, 0, 0, 1, 1, 0, 1, 1, 1, //
                1, 0, 0, 1, 1, 0, 1, 1, 1,
            ]),
        );
        assert_tensor_eq!(out, result);
    }

    fn test_tril_out_multi_unequal_dim<T: TrilElem>() {
        let tf = TensorFactory::<T>::new();
        let self_ = tf.make_default(vec![3, 2, 3], d(&[1; 18]));
        let out = tf.zeros_default(vec![3, 2, 3]);
        let mut ctx = context();
        op_tril_out(&mut ctx, &self_, 0, &out);
        let result = tf.make_default(
            vec![3, 2, 3],
            d(&[
                1, 0, 0, 1, 1, 0, //
                1, 0, 0, 1, 1, 0, //
                1, 0, 0, 1, 1, 0,
            ]),
        );
        assert_tensor_eq!(out, result);
    }

    fn test_tril_out_arange_reg_diag<T: TrilElem>() {
        let tf = TensorFactory::<T>::new();
        let self_ = tf.make_default(vec![3, 3], d(&[1, 2, 3, 4, 5, 6, 7, 8, 9]));
        let out = tf.zeros_default(vec![3, 3]);
        let mut ctx = context();
        op_tril_out(&mut ctx, &self_, 0, &out);
        let result = tf.make_default(vec![3, 3], d(&[1, 0, 0, 4, 5, 0, 7, 8, 9]));
        assert_tensor_eq!(out, result);
    }

    fn test_tril_out_arange_pos_diag<T: TrilElem>() {
        let tf = TensorFactory::<T>::new();
        let self_ = tf.make_default(vec![3, 3], d(&[1, 2, 3, 4, 5, 6, 7, 8, 9]));

        let out1 = tf.zeros_default(vec![3, 3]);
        let mut ctx = context();
        op_tril_out(&mut ctx, &self_, 1, &out1);
        let result1 = tf.make_default(vec![3, 3], d(&[1, 2, 0, 4, 5, 6, 7, 8, 9]));
        assert_tensor_eq!(out1, result1);

        let out2 = tf.zeros_default(vec![3, 3]);
        op_tril_out(&mut ctx, &self_, 2, &out2);
        assert_tensor_eq!(out2, self_);

        let out3 = tf.zeros_default(vec![3, 3]);
        op_tril_out(&mut ctx, &self_, 10, &out3);
        assert_tensor_eq!(out3, self_);
    }

    fn test_tril_out_arange_neg_diag<T: TrilElem>() {
        let tf = TensorFactory::<T>::new();
        let self_ = tf.make_default(vec![3, 3], d(&[1, 2, 3, 4, 5, 6, 7, 8, 9]));

        let out1 = tf.zeros_default(vec![3, 3]);
        let mut ctx = context();
        op_tril_out(&mut ctx, &self_, -1, &out1);
        let result1 = tf.make_default(vec![3, 3], d(&[0, 0, 0, 4, 0, 0, 7, 8, 0]));
        assert_tensor_eq!(out1, result1);

        let out2 = tf.zeros_default(vec![3, 3]);
        op_tril_out(&mut ctx, &self_, -2, &out2);
        let result2 = tf.make_default(vec![3, 3], d(&[0, 0, 0, 0, 0, 0, 7, 0, 0]));
        assert_tensor_eq!(out2, result2);

        let out3 = tf.zeros_default(vec![3, 3]);
        op_tril_out(&mut ctx, &self_, -10, &out3);
        let result3 = tf.make_default(vec![3, 3], d(&[0, 0, 0, 0, 0, 0, 0, 0, 0]));
        assert_tensor_eq!(out3, result3);
    }

    fn test_tril_out_randint_multi_equal<T: TrilElem>() {
        let tf = TensorFactory::<T>::new();
        let self_ = tf.make_default(
            vec![3, 3, 3, 3],
            d(&[
                9, 5, 4, 3, 9, 6, 9, 9, 5, //
                7, 2, 6, 8, 5, 5, 9, 3, 9, //
                1, 2, 1, 6, 2, 6, 1, 1, 8, //
                3, 2, 5, 4, 4, 1, 7, 1, 1, //
                5, 7, 8, 1, 5, 7, 7, 6, 3, //
                3, 5, 9, 4, 2, 2, 9, 5, 2, //
                8, 4, 7, 8, 7, 5, 7, 3, 8, //
                9, 5, 5, 6, 1, 8, 8, 9, 7, //
                1, 2, 3, 7, 9, 1, 5, 2, 2,
            ]),
        );
        let out = tf.zeros_default(vec![3, 3, 3, 3]);
        let mut ctx = context();
        op_tril_out(&mut ctx, &self_, 0, &out);
        let result = tf.make_default(
            vec![3, 3, 3, 3],
            d(&[
                9, 0, 0, 3, 9, 0, 9, 9, 5, //
                7, 0, 0, 8, 5, 0, 9, 3, 9, //
                1, 0, 0, 6, 2, 0, 1, 1, 8, //
                3, 0, 0, 4, 4, 0, 7, 1, 1, //
                5, 0, 0, 1, 5, 0, 7, 6, 3, //
                3, 0, 0, 4, 2, 0, 9, 5, 2, //
                8, 0, 0, 8, 7, 0, 7, 3, 8, //
                9, 0, 0, 6, 1, 0, 8, 9, 7, //
                1, 0, 0, 7, 9, 0, 5, 2, 2,
            ]),
        );
        assert_tensor_eq!(out, result);
    }

    fn test_tril_out_randint_multi_unequal<T: TrilElem>() {
        let tf = TensorFactory::<T>::new();
        let self_ = tf.make_default(
            vec![3, 2, 3, 2],
            d(&[
                1, 1, 1, 1, 9, 1, //
                1, 6, 6, 2, 7, 2, //
                2, 4, 8, 3, 4, 2, //
                7, 6, 1, 8, 4, 3, //
                2, 2, 7, 4, 3, 7, //
                7, 8, 4, 9, 1, 6,
            ]),
        );
        let out = tf.zeros_default(vec![3, 2, 3, 2]);
        let mut ctx = context();
        op_tril_out(&mut ctx, &self_, 0, &out);
        let result = tf.make_default(
            vec![3, 2, 3, 2],
            d(&[
                1, 0, 1, 1, 9, 1, //
                1, 0, 6, 2, 7, 2, //
                2, 0, 8, 3, 4, 2, //
                7, 0, 1, 8, 4, 3, //
                2, 0, 7, 4, 3, 7, //
                7, 0, 4, 9, 1, 6,
            ]),
        );
        assert_tensor_eq!(out, result);
    }

    fn generic_test<T: TrilElem>() {
        test_tril_out_zeros::<T>();
        test_tril_out_ones::<T>();
        test_tril_out_empty_dims::<T>();
        test_tril_out_square::<T>();
        test_tril_out_rectangle::<T>();
        test_tril_out_pos_diag::<T>();
        test_tril_out_neg_diag::<T>();
        test_tril_out_multi_equal_dim::<T>();
        test_tril_out_multi_unequal_dim::<T>();
    }

    fn real_test<T: TrilElem>() {
        test_tril_out_arange_reg_diag::<T>();
        test_tril_out_arange_pos_diag::<T>();
        test_tril_out_arange_neg_diag::<T>();
        test_tril_out_randint_multi_equal::<T>();
        test_tril_out_randint_multi_unequal::<T>();
    }

    // ET_FORALL_REALHBBF16_TYPES(GENERATE_GENERIC_TEST): one TEST_F per dtype.
    // [spec:et:sem:op-tril.torch.executor.native.tril-out-fn/test]
    // also verifies check_tril_args (same-dtype + rank>=2 gate) via the 2D/3D
    // square/rectangle/multi-dim sub-cases
    // [spec:et:sem:copy-ops-util.torch.executor.check-tril-args-fn/test]
    // also verifies clear_out (test_tril_out_ones: an all-ones out is zeroed
    // before the lower-triangular fill, so the zeros above the diagonal prove
    // the clear) and tril_kernel (square/rectangle/multi-dim sub-cases pin the
    // per-batch lower-triangular copy with computed row/col strides)
    // [spec:et:sem:op-tril.torch.executor.native.clear-out-fn/test]
    // [spec:et:sem:op-tril.torch.executor.native.tril-kernel-fn/test]
    #[test]
    fn op_tril_test_byte_generic_test() {
        generic_test::<u8>();
    }
    // [spec:et:sem:op-tril.torch.executor.native.tril-out-fn/test]
    #[test]
    fn op_tril_test_char_generic_test() {
        generic_test::<i8>();
    }
    // [spec:et:sem:op-tril.torch.executor.native.tril-out-fn/test]
    #[test]
    fn op_tril_test_short_generic_test() {
        generic_test::<i16>();
    }
    // [spec:et:sem:op-tril.torch.executor.native.tril-out-fn/test]
    #[test]
    fn op_tril_test_int_generic_test() {
        generic_test::<i32>();
    }
    // [spec:et:sem:op-tril.torch.executor.native.tril-out-fn/test]
    #[test]
    fn op_tril_test_long_generic_test() {
        generic_test::<i64>();
    }
    // [spec:et:sem:op-tril.torch.executor.native.tril-out-fn/test]
    #[test]
    fn op_tril_test_float_generic_test() {
        generic_test::<f32>();
    }
    // [spec:et:sem:op-tril.torch.executor.native.tril-out-fn/test]
    #[test]
    fn op_tril_test_double_generic_test() {
        generic_test::<f64>();
    }
    // [spec:et:sem:op-tril.torch.executor.native.tril-out-fn/test]
    #[test]
    fn op_tril_test_half_generic_test() {
        generic_test::<Half>();
    }
    // [spec:et:sem:op-tril.torch.executor.native.tril-out-fn/test]
    #[test]
    fn op_tril_test_bfloat16_generic_test() {
        generic_test::<BFloat16>();
    }

    // ET_FORALL_REALHBBF16_TYPES(GENERATE_REAL_TEST): one TEST_F per dtype.
    // [spec:et:sem:op-tril.torch.executor.native.tril-out-fn/test]
    #[test]
    fn op_tril_test_byte_real_test() {
        real_test::<u8>();
    }
    // [spec:et:sem:op-tril.torch.executor.native.tril-out-fn/test]
    #[test]
    fn op_tril_test_char_real_test() {
        real_test::<i8>();
    }
    // [spec:et:sem:op-tril.torch.executor.native.tril-out-fn/test]
    #[test]
    fn op_tril_test_short_real_test() {
        real_test::<i16>();
    }
    // [spec:et:sem:op-tril.torch.executor.native.tril-out-fn/test]
    #[test]
    fn op_tril_test_int_real_test() {
        real_test::<i32>();
    }
    // [spec:et:sem:op-tril.torch.executor.native.tril-out-fn/test]
    #[test]
    fn op_tril_test_long_real_test() {
        real_test::<i64>();
    }
    // [spec:et:sem:op-tril.torch.executor.native.tril-out-fn/test]
    #[test]
    fn op_tril_test_float_real_test() {
        real_test::<f32>();
    }
    // [spec:et:sem:op-tril.torch.executor.native.tril-out-fn/test]
    #[test]
    fn op_tril_test_double_real_test() {
        real_test::<f64>();
    }
    // [spec:et:sem:op-tril.torch.executor.native.tril-out-fn/test]
    #[test]
    fn op_tril_test_half_real_test() {
        real_test::<Half>();
    }
    // [spec:et:sem:op-tril.torch.executor.native.tril-out-fn/test]
    #[test]
    fn op_tril_test_bfloat16_real_test() {
        real_test::<BFloat16>();
    }

    // [spec:et:sem:op-tril.torch.executor.native.tril-out-fn/test]
    #[test]
    fn op_tril_test_invalid_input_shapes_dies() {
        let tf = TensorFactory::<i32>::new();

        let self1 = tf.zeros_default(vec![]);
        let out1 = tf.zeros_default(vec![]);
        let mut ctx = context();
        op_tril_out(&mut ctx, &self1, 0, &out1);
        assert_ne!(ctx.failure_state(), Error::Ok);

        let self2 = tf.zeros_default(vec![1]);
        let out2 = tf.zeros_default(vec![1]);
        let mut ctx = context();
        op_tril_out(&mut ctx, &self2, 0, &out2);
        assert_ne!(ctx.failure_state(), Error::Ok);
    }

    // PORT-NOTE: C++ `ET_SKIP_IF(is_aten, ...)`; never ATen in the port.
    // [spec:et:sem:op-tril.torch.executor.native.tril-out-fn/test]
    #[test]
    fn op_tril_test_mismatched_output_shapes_dies() {
        let tf = TensorFactory::<i32>::new();

        let self_ = tf.zeros_default(vec![2, 1]);
        let out = tf.zeros_default(vec![2, 2]);
        let mut ctx = context();
        op_tril_out(&mut ctx, &self_, 0, &out);
        assert_ne!(ctx.failure_state(), Error::Ok);
    }

    // [spec:et:sem:op-tril.torch.executor.native.tril-out-fn/test]
    #[test]
    fn op_tril_test_mismatched_output_dtype_dies() {
        let tf_byte = TensorFactory::<u8>::new();
        let tf_float = TensorFactory::<f32>::new();

        let self_ = tf_byte.zeros_default(vec![2, 2]);
        let out = tf_float.zeros_default(vec![2, 2]);
        let mut ctx = context();
        op_tril_out(&mut ctx, &self_, 0, &out);
        assert_ne!(ctx.failure_state(), Error::Ok);
    }

    // PORT-NOTE: C++ `ET_SKIP_IF(is_aten, ...)`; never ATen in the port.
    // [spec:et:sem:op-tril.torch.executor.native.tril-out-fn/test]
    #[test]
    fn op_tril_test_invalid_tensor_dims() {
        let tf = TensorFactory::<i32>::new();

        let sizes = vec![1i32; 25];
        let self_ = tf.zeros_default(sizes.clone());
        let out = tf.zeros_default(sizes);
        let mut ctx = context();
        op_tril_out(&mut ctx, &self_, 0, &out);
        assert_ne!(ctx.failure_state(), Error::Ok);
    }
}
