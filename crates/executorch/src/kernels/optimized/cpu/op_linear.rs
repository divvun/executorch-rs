//! Literal port of kernels/optimized/cpu/op_linear.cpp.

use crate::kernels::optimized::blas::CPUBlas::TransposeType;
use crate::kernels::optimized::cpu::opt_gemm::OptGemm;
use crate::kernels::portable::cpu::util::matmul_ops_util::{
    check_linear_args, get_linear_out_target_size,
};
use crate::runtime::core::array_ref::ArrayRef;
use crate::runtime::core::error::Error;
use crate::runtime::core::exec_aten::util::tensor_util::{K_TENSOR_DIMENSION_LIMIT, resize_tensor};
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::core::portable_type::tensor_impl::{SizesType, ssize_t};
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

// Use vector store to initialize with scalar bias.
// [spec:et:def:op-linear.torch.executor.native.initialize-scalar-fn]
// [spec:et:sem:op-linear.torch.executor.native.initialize-scalar-fn]
//
// Fill the `out_numel`-element buffer `out` with the scalar `init`, one element
// at a time. The C++ builds a `Vectorized<scalar_t>(init)` and stores it in
// vector-length chunks, then a sub-vector-length store of the remainder. Because
// the result is identical to writing `init` into every slot, the port collapses
// the SIMD store into a plain scalar loop.
//
// DEVIATION (rust/PORTING.md optimized-kernels): `Vectorized<scalar_t>` /
// `Vec::store` → scalar loop. The blocked vector/remainder structure is folded to
// a single element-wise write; the written values are unchanged.
///
/// # Safety
/// `out` must point to at least `out_numel` writable `scalar_t` elements.
unsafe fn initialize_scalar<S: Copy>(out_numel: ssize_t, init: S, out: *mut S) {
    let mut d: ssize_t = 0;
    while d < out_numel {
        unsafe {
            *out.add(d as usize) = init;
        }
        d += 1;
    }
}

// Use std::memcpy to initialize with vector bias.
// [spec:et:def:op-linear.torch.executor.native.initialize-to-vector-fn]
// [spec:et:sem:op-linear.torch.executor.native.initialize-to-vector-fn]
//
// The output is an `n x m` matrix (row-major) of `scalar_t` and `bias` is an
// `m`-element vector. Copy `bias` into each of the `n` rows: for every column
// index `col` in `[0, n)`, memcpy `m * sizeof(scalar_t)` bytes from `bias` to
// `out + col * m`.
///
/// # Safety
/// `bias` must point to at least `m` readable `scalar_t` elements and `out` to at
/// least `n * m` writable `scalar_t` elements.
unsafe fn initialize_to_vector<S: Copy>(n: ssize_t, m: ssize_t, bias: *const S, out: *mut S) {
    // Output is a n x m x scalar_t, while bias is m x scalar_t.
    for col in 0..n {
        unsafe {
            // Point to Column `col` of the output tensor.
            core::ptr::copy_nonoverlapping(bias, out.add((col * m) as usize), m as usize);
        }
    }
}

// PORT-NOTE: `RuntimeContext& ctx` / `Tensor& out` / returned `Tensor&` map to
// `&mut KernelRuntimeContext` / `&'a Tensor` (interior mutation through
// `*mut TensorImpl`). `const optional<Tensor>& bias` -> `&Option<Tensor>` (the
// op_convolution convention). `ET_SWITCH_REAL_TYPES_AND2(Half, BFloat16, ...)`
// maps to `et_switch_realhbf16_types!`.
//
// DEVIATION (rust/PORTING.md optimized-kernels): `executorch::cpublas::gemm`
// overload resolution → the `OptGemm` trait (opt_gemm.rs). The transpose / alpha
// / beta / column-major bookkeeping is preserved: mat2 (weight) is passed
// Transposed and `in` NoTranspose so the `[*, K] x [N, K]^T -> [*, N]` linear
// layout is honored.
//
// PORT-NOTE (codegen): the C++ `ET_KERNEL_CHECK_MSG` carries printf-style detail
// (bias dtype / dimensionality). The crate `et_kernel_check_msg!` keeps only the
// leading format literal (`__et_first_fmt!` drops trailing args), so the detail
// arguments are not formatted; the check condition and failure behavior are
// identical.

// [spec:et:def:op-linear.torch.executor.native.opt-linear-out-fn]
// [spec:et:sem:op-linear.torch.executor.native.opt-linear-out-fn]
//
// Optimized `linear.out`: computes `out = in @ mat2^T (+ bias)` where `in` is
// `[*, K]`, `mat2` (the weight) is `[N, K]`, and `out` is `[*, N]`. Steps:
//  1. `check_linear_args(in, mat2, out)`; on failure record InvalidArgument and
//     return `out`.
//  2. Compute the target output sizes (`get_linear_out_target_size`) and resize
//     `out`; failure -> InvalidArgument.
//  3. Empty output returns early (GEMM doesn't tolerate empty input).
//  4. Flatten the leading dims of `in` into `n = prod(in.sizes[0..dim-1])`; set
//     `k = in.sizes[dim-1]`, `m = mat2.size(0)`.
//  5. If `bias` is present: its dtype must equal `out`'s dtype and it must be a
//     1-D tensor of size `m` or `1` (else InvalidArgument).
//  6. In the dtype switch: if bias has exactly one element, prefill `out` with
//     that scalar (`initialize_scalar`); else if bias is present, broadcast the
//     `m`-vector bias into every row (`initialize_to_vector`). `beta` is `1` when
//     bias was prefilled (GEMM accumulates onto it) and `0` otherwise (GEMM fully
//     overwrites). Then a single column-major GEMM with `transa=Transpose`
//     (weight), `transb=NoTranspose` (`in`), dims `(m, n, k)`, alpha `1`.
//  7. Return `out`.
pub fn opt_linear_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    in_: &Tensor,
    mat2: &Tensor,
    bias: &Option<Tensor>,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    crate::et_kernel_check!(ctx, check_linear_args(in_, mat2, out), InvalidArgument, out);

    let mut output_ndim: usize = 0;
    let mut output_sizes: [SizesType; K_TENSOR_DIMENSION_LIMIT] = [0; K_TENSOR_DIMENSION_LIMIT];
    unsafe {
        get_linear_out_target_size(in_, mat2, output_sizes.as_mut_ptr(), &mut output_ndim);
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

    // gemm on some platforms doesn't tolerate empty input.
    if out.numel() == 0 {
        return out;
    }

    let mut n: ssize_t = 1;
    let mut ii: ssize_t = 0;
    while ii < in_.dim() - 1 {
        n *= *in_.sizes().at(ii as usize) as ssize_t;
        ii += 1;
    }
    let k: ssize_t = *in_.sizes().at((in_.dim() - 1) as usize) as ssize_t;
    let m: ssize_t = mat2.size(0);

    if let Some(bias_t) = bias.as_ref() {
        crate::et_kernel_check_msg!(
            ctx,
            // Bias and output dtype must match.
            bias_t.dtype() == out.dtype(),
            InvalidArgument,
            out,
            // PORT-NOTE: C++ formats bias/out dtype via toString(); the crate
            // check macro drops trailing args, so the values are omitted.
            "Bias has wrong dtype! Expected bias dtype to be the same as out dtype"
        );

        crate::et_kernel_check_msg!(
            ctx,
            // Either no bias or bias is a 1D tensor of size m or 1.
            bias_t.dim() == 1 && (bias_t.size(0) == m || bias_t.size(0) == 1),
            InvalidArgument,
            out,
            // PORT-NOTE: C++ formats m / bias dim / bias numel; dropped as above.
            "Bias has wrong dimensionality! Expected 1-D tensor of size m or empty"
        );
    }

    crate::et_switch_realhbf16_types!(out.scalar_type(), ctx, "linear.out", CTYPE, {
        // Fill output with bias if it is provided.
        if let Some(bias_t) = bias.as_ref() {
            if bias_t.numel() == 1 {
                // Scalar version of initialization.
                unsafe {
                    initialize_scalar::<CTYPE>(
                        out.numel(),
                        *bias_t.const_data_ptr::<CTYPE>(),
                        out.mutable_data_ptr::<CTYPE>(),
                    );
                }
            } else {
                // Assume bias is a 1D tensor of size m.
                unsafe {
                    initialize_to_vector::<CTYPE>(
                        n,
                        m,
                        bias_t.const_data_ptr::<CTYPE>(),
                        out.mutable_data_ptr::<CTYPE>(),
                    );
                }
            }
        }

        // Set beta to 1 if bias was applied so that GEMM adds to the pre-filled
        // bias, otherwise beta remains 0 (i.e. the output is fully overwritten
        // by GEMM).
        let beta: CTYPE = if bias.is_some() {
            <CTYPE as OptGemm>::one()
        } else {
            <CTYPE as OptGemm>::zero()
        };

        unsafe {
            <CTYPE as OptGemm>::opt_gemm(
                /*transa=*/ TransposeType::Transpose,
                /*transb=*/ TransposeType::NoTranspose,
                m as i64,
                n as i64,
                k as i64,
                /*alpha=*/ <CTYPE as OptGemm>::one(),
                mat2.const_data_ptr::<CTYPE>(),
                k as i64,
                in_.const_data_ptr::<CTYPE>(),
                k as i64,
                beta,
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

    // Direct unit test: fill a raw buffer with a scalar.
    // [spec:et:sem:op-linear.torch.executor.native.initialize-scalar-fn/test]
    #[test]
    fn initialize_scalar_fills_buffer() {
        let mut buf = [0.0f32; 7];
        unsafe {
            initialize_scalar::<f32>(7, 3.5, buf.as_mut_ptr());
        }
        assert_eq!(buf, [3.5f32; 7]);

        let mut ibuf = [0i32; 3];
        unsafe {
            initialize_scalar::<i32>(3, -4, ibuf.as_mut_ptr());
        }
        assert_eq!(ibuf, [-4, -4, -4]);
    }

    // Direct unit test: replicate an m-vector bias into each of n rows.
    // [spec:et:sem:op-linear.torch.executor.native.initialize-to-vector-fn/test]
    #[test]
    fn initialize_to_vector_replicates_rows() {
        let bias = [10i32, 20];
        let mut out = [0i32; 6];
        unsafe {
            initialize_to_vector::<i32>(3, 2, bias.as_ptr(), out.as_mut_ptr());
        }
        assert_eq!(out, [10, 20, 10, 20, 10, 20]);
    }

    // OpLinearOutTest.AllDtypesSupported: full(2) [3,19] @ full(3) [5,19]^T =
    // full(114), no bias.
    macro_rules! test_linear_dtype {
        ($t:ty, $from:expr) => {{
            let tf = TensorFactory::<$t>::new();
            let x = tf.full(vec![3, 19], $from(2), TensorShapeDynamism::STATIC);
            let y = tf.full(vec![5, 19], $from(3), TensorShapeDynamism::STATIC);
            let out = tf.zeros_default(vec![3, 5]);

            let mut ctx = context();
            opt_linear_out(&mut ctx, &x, &y, &None, &out);
            assert_eq!(ctx.failure_state(), Error::Ok);
            let expected = tf.full(vec![3, 5], $from(114), TensorShapeDynamism::STATIC);
            assert_tensor_eq!(out, expected);
        }};
    }

    // [spec:et:sem:op-linear.torch.executor.native.opt-linear-out-fn/test]
    #[test]
    fn opt_linear_out_all_dtypes_supported() {
        test_linear_dtype!(u8, |v: i32| v as u8);
        test_linear_dtype!(i8, |v: i32| v as i8);
        test_linear_dtype!(i16, |v: i32| v as i16);
        test_linear_dtype!(i32, |v: i32| v);
        test_linear_dtype!(i64, |v: i32| v as i64);
        test_linear_dtype!(f32, |v: i32| v as f32);
        test_linear_dtype!(f64, |v: i32| v as f64);
        test_linear_dtype!(Half, |v: i32| Half::from_f32(v as f32));
        test_linear_dtype!(BFloat16, |v: i32| BFloat16::from_f32(v as f32));
    }

    // Non-uniform values pin `out = in @ mat2^T`: rows [a, b] against weights
    // [[1,0],[0,1],[1,1]] give [a, b, a+b], plus the m-vector bias per row.
    // [spec:et:sem:op-linear.torch.executor.native.opt-linear-out-fn/test]
    // [spec:et:sem:op-linear.torch.executor.native.initialize-to-vector-fn/test]
    #[test]
    fn opt_linear_out_hand_computed_with_vector_bias() {
        let tf = TensorFactory::<f32>::new();
        let x = tf.make_default(vec![3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let w = tf.make_default(vec![3, 2], vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
        let bias = tf.make_default(vec![3], vec![10.0, 20.0, 30.0]);
        let out = tf.zeros_default(vec![3, 3]);

        let mut ctx = context();
        opt_linear_out(&mut ctx, &x, &w, &Some(bias), &out);
        assert_eq!(ctx.failure_state(), Error::Ok);
        let expected = tf.make_default(
            vec![3, 3],
            vec![11.0, 22.0, 33.0, 13.0, 24.0, 37.0, 15.0, 26.0, 41.0],
        );
        assert_tensor_close!(out, expected);
    }

    // OpLinearOutTest.BiasTest: m-vector bias lands on the right output column.
    // [spec:et:sem:op-linear.torch.executor.native.opt-linear-out-fn/test]
    // [spec:et:sem:op-linear.torch.executor.native.initialize-to-vector-fn/test]
    #[test]
    fn opt_linear_out_bias_test() {
        let tf = TensorFactory::<i32>::new();
        let x = tf.full(vec![3, 4], 1, TensorShapeDynamism::STATIC);
        let y = tf.full(vec![2, 4], 2, TensorShapeDynamism::STATIC);
        let b = tf.make_default(vec![2], vec![4, 7]);
        let out = tf.zeros_default(vec![3, 2]);

        let mut ctx = context();
        opt_linear_out(&mut ctx, &x, &y, &Some(b), &out);
        // 1*2*4 + bias
        let expected = tf.make_default(vec![3, 2], vec![12, 15, 12, 15, 12, 15]);
        assert_tensor_eq!(out, expected);
    }

    // OpLinearOutTest.BiasBroadcastTest: single-element bias prefills the whole
    // output (initialize_scalar path, beta = 1).
    // [spec:et:sem:op-linear.torch.executor.native.opt-linear-out-fn/test]
    // [spec:et:sem:op-linear.torch.executor.native.initialize-scalar-fn/test]
    #[test]
    fn opt_linear_out_bias_broadcast_test() {
        let tf = TensorFactory::<i32>::new();
        let x = tf.full(vec![3, 4], 1, TensorShapeDynamism::STATIC);
        let y = tf.full(vec![5, 4], 2, TensorShapeDynamism::STATIC);
        let b = tf.full(vec![1], 4, TensorShapeDynamism::STATIC);
        let out = tf.zeros_default(vec![3, 5]);

        let mut ctx = context();
        opt_linear_out(&mut ctx, &x, &y, &Some(b), &out);
        // 1*2*4 + 4 = 12
        let expected = tf.full(vec![3, 5], 12, TensorShapeDynamism::STATIC);
        assert_tensor_eq!(out, expected);
    }

    // OpLinearOutTest.BiasDtypeMismatch: bias dtype must equal out dtype.
    // [spec:et:sem:op-linear.torch.executor.native.opt-linear-out-fn/test]
    #[test]
    fn opt_linear_out_bias_dtype_mismatch_dies() {
        let tf = TensorFactory::<i32>::new();
        let tf_bias = TensorFactory::<i16>::new();
        let x = tf.full(vec![3, 4], 1, TensorShapeDynamism::STATIC);
        let y = tf.full(vec![5, 4], 2, TensorShapeDynamism::STATIC);
        let b = tf_bias.full(vec![5], 4, TensorShapeDynamism::STATIC);
        let out = tf.zeros_default(vec![3, 5]);

        let mut ctx = context();
        et_expect_kernel_failure!(ctx, opt_linear_out(&mut ctx, &x, &y, &Some(b), &out));
    }

    // OpLinearOutTest.MismatchedDimensionSizeDies: reduce dims disagree.
    // [spec:et:sem:op-linear.torch.executor.native.opt-linear-out-fn/test]
    #[test]
    fn opt_linear_out_mismatched_dimension_size_dies() {
        let tf = TensorFactory::<i32>::new();
        let x = tf.full(vec![2, 2], 3, TensorShapeDynamism::STATIC);
        let wrong_y = tf.full(vec![5, 3], 1, TensorShapeDynamism::STATIC);
        let out = tf.zeros_default(vec![2, 5]);

        let mut ctx = context();
        et_expect_kernel_failure!(ctx, opt_linear_out(&mut ctx, &x, &wrong_y, &None, &out));
    }

    // OpLinearOutTest.EmptyInputWithEmptyOutTensorPasses.
    // [spec:et:sem:op-linear.torch.executor.native.opt-linear-out-fn/test]
    #[test]
    fn opt_linear_out_empty_input_passes() {
        let tf = TensorFactory::<f32>::new();
        let x = tf.make_default(vec![0, 3], vec![]);
        let y = tf.make_default(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let out = tf.make_default(vec![0, 2], vec![]);

        let mut ctx = context();
        opt_linear_out(&mut ctx, &x, &y, &None, &out);
        assert_eq!(ctx.failure_state(), Error::Ok);
        assert_tensor_eq!(out, tf.make_default(vec![0, 2], vec![]));
    }
}
