//! Literal port of kernels/optimized/cpu/op_mm.cpp.

use crate::kernels::optimized::blas::CPUBlas::TransposeType;
use crate::kernels::optimized::cpu::opt_gemm::OptGemm;
use crate::kernels::portable::cpu::util::matmul_ops_util::{check_mm_args, get_mm_out_target_size};
use crate::runtime::core::array_ref::ArrayRef;
use crate::runtime::core::error::Error;
use crate::runtime::core::exec_aten::util::tensor_util::{K_TENSOR_DIMENSION_LIMIT, resize_tensor};
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::core::portable_type::tensor_impl::SizesType;
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

// PORT-NOTE: `RuntimeContext& ctx` / `Tensor& out` / returned `Tensor&` map to
// `&mut KernelRuntimeContext` / `&'a Tensor` (interior mutation through
// `*mut TensorImpl`), as in the portable op_mm port. `ET_SWITCH_REAL_TYPES_AND2(
// Half, BFloat16, ...)` maps to `et_switch_realhbf16_types!`.
//
// DEVIATION (rust/PORTING.md optimized-kernels): `executorch::cpublas::gemm`
// dispatches per element type in C++ via overload resolution. Rust
// monomorphization inside the dtype switch can't pick a per-type overload, so the
// dispatch goes through the `OptGemm` trait (opt_gemm.rs), which forwards each
// CTYPE to the matching CPUBlas `gemm_*` entry point. The column-major / transpose
// bookkeeping and the `(A @ B).t() = B.t() @ A.t()` identity are preserved.

// [spec:et:def:op-mm.torch.executor.native.opt-mm-out-fn]
// [spec:et:sem:op-mm.torch.executor.native.opt-mm-out-fn]
//
// Optimized matrix-multiply out-variant. Validates that `in`, `mat2`, `out` are
// rank-2 and shape-compatible (`check_mm_args`), resizes `out` to
// `[in.size(0), mat2.size(1)]`, and — for a non-empty output — computes
// `out = in @ mat2` for every real/Half/BFloat16 dtype via a single column-major
// GEMM. Because GEMM is column-major, the row-major product `in @ mat2` is
// obtained from the identity `(A @ B).t() = B.t() @ A.t()`: row-major `mat2` is
// `mat2.t()` in GEMM's column-major view and row-major `in` is `in.t()`, so a
// NoTranspose/NoTranspose GEMM of `(mat2, in)` with dims `(m, n, k)` writes the
// row-major result directly. Empty output returns early because GEMM on some
// platforms doesn't tolerate empty input. Returns `out`.
pub fn opt_mm_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    in_: &Tensor,
    mat2: &Tensor,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    crate::et_kernel_check!(ctx, check_mm_args(in_, mat2, out), InvalidArgument, out);

    let mut output_ndim: usize = 0;
    let mut output_sizes: [SizesType; K_TENSOR_DIMENSION_LIMIT] = [0; K_TENSOR_DIMENSION_LIMIT];
    unsafe {
        get_mm_out_target_size(in_, mat2, output_sizes.as_mut_ptr(), &mut output_ndim);
    }
    crate::et_kernel_check!(
        ctx,
        resize_tensor(
            out,
            ArrayRef::from_raw_parts(output_sizes.as_ptr(), output_ndim)
        ) == Error::Ok,
        InvalidArgument,
        out
    );

    if out.numel() == 0 {
        return out;
    }
    crate::et_switch_realhbf16_types!(in_.scalar_type(), ctx, "mm.out", CTYPE, {
        let n: usize = in_.size(0) as usize;
        let k: usize = in_.size(1) as usize;
        let m: usize = mat2.size(1) as usize;

        // gemm expects column-major inputs and produces column-major
        // output. So, we take advantage of the identity (A @ B).t()
        // = B.t() @ A.t() here; row-major B is B.t() from gemm's
        // column-major perspective, etc.
        unsafe {
            <CTYPE as OptGemm>::opt_gemm(
                TransposeType::NoTranspose,
                TransposeType::NoTranspose,
                m as i64,
                n as i64,
                k as i64,
                <CTYPE as OptGemm>::one(),
                mat2.const_data_ptr::<CTYPE>(),
                m as i64,
                in_.const_data_ptr::<CTYPE>(),
                k as i64,
                <CTYPE as OptGemm>::zero(),
                out.mutable_data_ptr::<CTYPE>(),
                m as i64,
            );
        }
    });

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::core::exec_aten::testing_util::tensor_factory::TensorFactory;
    use crate::runtime::core::memory_allocator::{MemoryAllocator, MemoryAllocatorBase};
    use crate::runtime::core::portable_type::{BFloat16, Half};
    use crate::runtime::core::tensor_shape_dynamism::TensorShapeDynamism;
    use crate::{assert_tensor_close, assert_tensor_eq};

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

    // OpMmOutTest.AllDtypesSupported: full(2) [3,4] @ full(3) [4,5] = full(24).
    macro_rules! test_mm_dtype {
        ($t:ty, $from:expr) => {{
            let tf = TensorFactory::<$t>::new();
            let x = tf.full(vec![3, 4], $from(2), TensorShapeDynamism::STATIC);
            let y = tf.full(vec![4, 5], $from(3), TensorShapeDynamism::STATIC);
            let out = tf.zeros_default(vec![3, 5]);

            let mut ctx = context();
            opt_mm_out(&mut ctx, &x, &y, &out);
            assert_eq!(ctx.failure_state(), Error::Ok);
            let expected = tf.full(vec![3, 5], $from(24), TensorShapeDynamism::STATIC);
            assert_tensor_eq!(out, expected);
        }};
    }

    // [spec:et:sem:op-mm.torch.executor.native.opt-mm-out-fn/test]
    #[test]
    fn opt_mm_out_all_dtypes_supported() {
        test_mm_dtype!(u8, |v: i32| v as u8);
        test_mm_dtype!(i8, |v: i32| v as i8);
        test_mm_dtype!(i16, |v: i32| v as i16);
        test_mm_dtype!(i32, |v: i32| v);
        test_mm_dtype!(i64, |v: i32| v as i64);
        test_mm_dtype!(f32, |v: i32| v as f32);
        test_mm_dtype!(f64, |v: i32| v as f64);
        test_mm_dtype!(Half, |v: i32| Half::from_f32(v as f32));
        test_mm_dtype!(BFloat16, |v: i32| BFloat16::from_f32(v as f32));
    }

    // Non-uniform values pin the column-major / (A@B).t() = B.t()@A.t()
    // bookkeeping: a uniform `full` GEMM cannot detect a swapped operand.
    // [spec:et:sem:op-mm.torch.executor.native.opt-mm-out-fn/test]
    #[test]
    fn opt_mm_out_hand_computed_values() {
        let tf = TensorFactory::<f32>::new();
        let a = tf.make_default(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = tf.make_default(vec![3, 2], vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
        let out = tf.zeros_default(vec![2, 2]);

        let mut ctx = context();
        opt_mm_out(&mut ctx, &a, &b, &out);
        // [[1*7+2*9+3*11, 1*8+2*10+3*12], [4*7+5*9+6*11, 4*8+5*10+6*12]]
        let expected = tf.make_default(vec![2, 2], vec![58.0, 64.0, 139.0, 154.0]);
        assert_tensor_close!(out, expected);
    }

    // OpMmOutTest.EmptyInputWithEmptyOutTensorPasses: numel == 0 returns early.
    // [spec:et:sem:op-mm.torch.executor.native.opt-mm-out-fn/test]
    #[test]
    fn opt_mm_out_empty_input_passes() {
        let tf = TensorFactory::<f32>::new();
        let x = tf.make_default(vec![0, 3], vec![]);
        let y = tf.make_default(vec![3, 0], vec![]);
        let out = tf.make_default(vec![0, 0], vec![]);

        let mut ctx = context();
        opt_mm_out(&mut ctx, &x, &y, &out);
        assert_eq!(ctx.failure_state(), Error::Ok);
        assert_tensor_eq!(out, tf.make_default(vec![0, 0], vec![]));
    }

    // OpMmOutTest.MismatchedDimensionSizeDies: inner dims disagree.
    // [spec:et:sem:op-mm.torch.executor.native.opt-mm-out-fn/test]
    #[test]
    fn opt_mm_out_mismatched_dimension_size_dies() {
        let tf = TensorFactory::<i32>::new();
        let x = tf.full(vec![2, 2], 3, TensorShapeDynamism::STATIC);
        let wrong_y = tf.full(vec![3, 1], 1, TensorShapeDynamism::STATIC);
        let out = tf.zeros_default(vec![2, 2]);

        let mut ctx = context();
        et_expect_kernel_failure!(ctx, opt_mm_out(&mut ctx, &x, &wrong_y, &out));
    }

    // OpMmOutTest.MismatchedDimensionsDies: 1-D operand is rejected.
    // [spec:et:sem:op-mm.torch.executor.native.opt-mm-out-fn/test]
    #[test]
    fn opt_mm_out_mismatched_dimensions_dies() {
        let tf = TensorFactory::<i32>::new();
        let x = tf.full(vec![2, 2], 3, TensorShapeDynamism::STATIC);
        let y = tf.full(vec![4], 1, TensorShapeDynamism::STATIC);
        let out = tf.zeros_default(vec![2, 2]);

        let mut ctx = context();
        et_expect_kernel_failure!(ctx, opt_mm_out(&mut ctx, &x, &y, &out));
    }
}
