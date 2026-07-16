//! Literal port of kernels/optimized/cpu/op_bmm.cpp.
//!
//! Performs a batch matrix-matrix product of matrices stored in input and mat2.
//!
//! input and mat2 must be 3-D tensors each containing the same number of
//! matrices.
//!
//! If input is a (b × n × m) tensor, mat2 is a (b × m × p) tensor, out will be a
//! (b × n × p) tensor.
//!
//! Note: This function does not broadcast. For broadcasting matrix products, see
//! matmul().

use crate::kernels::optimized::blas::CPUBlas::TransposeType;
use crate::kernels::optimized::cpu::opt_gemm::OptGemm;
use crate::runtime::core::array_ref::ArrayRef;
use crate::runtime::core::error::Error;
use crate::runtime::core::exec_aten::util::scalar_type_util::is_complex_type;
use crate::runtime::core::exec_aten::util::tensor_util::{
    K_TENSOR_DIMENSION_LIMIT, resize_tensor, tensors_have_same_dtype,
};
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::core::portable_type::tensor_impl::SizesType;
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

// PORT-NOTE: the crate-level `et_check_or_return_false!` (runtime/core/error.rs)
// drops all caller-supplied format arguments after the leading literal. The C++
// `ET_LOG_AND_RETURN_IF_FALSE(cond)` expands to `ET_CHECK_OR_RETURN_FALSE(cond,
// "")`; matmul_ops_util.rs defines its own `et_log_and_return_if_false!` for the
// same reason, mirrored here.
macro_rules! et_log_and_return_if_false {
    ($cond:expr) => {{
        if !($cond) {
            $crate::et_log!(Error, "Check failed ({}): ", ::core::stringify!($cond));
            return false;
        }
    }};
}

// Verifies that the parameters are valid.
// [spec:et:def:op-bmm.torch.executor.native.check-bmm-out-args-fn]
// [spec:et:sem:op-bmm.torch.executor.native.check-bmm-out-args-fn]
//
// Returns true iff the bmm operands are self-consistent: `self`, `mat2`, `out`
// all have the same rank, that rank is 3, `self.size(0) >= 0`, the batch dims
// match across all three (`self.size(0) == mat2.size(0) == out.size(0)`), the
// output's trailing dims match (`mat2.size(2) == out.size(2)`,
// `self.size(1) == out.size(1)`), and all three share a dtype. Any failed check
// logs and returns false immediately.
//
// PORT-NOTE: `Tensor& out` -> `&Tensor` (matching the port's interior-mutation
// convention). The printf detail arguments on each `ET_CHECK_OR_RETURN_FALSE`
// are dropped by the crate check macro (see module note); the conditions and
// early-return behavior are identical.
pub fn check_bmm_out_args(self_: &Tensor, mat2: &Tensor, out: &Tensor) -> bool {
    // Ensure dimensions is 3 for all input and out
    et_log_and_return_if_false!(self_.dim() == mat2.dim());
    et_log_and_return_if_false!(self_.dim() == out.dim());
    et_log_and_return_if_false!(self_.dim() == 3);
    // Ensure batch larger than or equals to 0
    et_log_and_return_if_false!(self_.size(0) >= 0);
    // Ensure batches are the same
    et_log_and_return_if_false!(self_.size(0) == mat2.size(0));
    et_log_and_return_if_false!(self_.size(0) == out.size(0));
    // Ensure the out size is compatible with input tensors
    et_log_and_return_if_false!(mat2.size(2) == out.size(2));
    et_log_and_return_if_false!(self_.size(1) == out.size(1));

    // Ensure that all tensors share a dtype
    et_log_and_return_if_false!(tensors_have_same_dtype(self_, mat2, out));

    true
}

// [spec:et:def:op-bmm.torch.executor.native.bmm-kernel-fn]
// [spec:et:sem:op-bmm.torch.executor.native.bmm-kernel-fn]
//
// For each of the `batch_size = self.size(0)` matrix pairs, run a column-major
// GEMM producing one output matrix. `n = self.size(1)`, `k = self.size(2)`,
// `m = mat2.size(2)`. The B operand (`self`) rows are advanced by `k * n`, the A
// operand (`mat2`) by `m * k`, and the C output by `m * n` per batch. If any of
// `self`/`mat2`/`out` is empty, returns immediately. Each GEMM is
// NoTranspose/NoTranspose with `(m, n, k)`, alpha `1`, beta `0`, `lda = m`,
// `ldb = k`, `ldc = m` — the same `(A @ B).t()` column-major trick as mm.
//
// DEVIATION (rust/PORTING.md optimized-kernels): `executorch::cpublas::gemm`
// overload resolution → the `OptGemm` trait (opt_gemm.rs).
fn bmm_kernel<CTYPE: OptGemm>(self_: &Tensor, mat2: &Tensor, out: &Tensor) {
    if self_.numel() == 0 || mat2.numel() == 0 || out.numel() == 0 {
        return;
    }

    let b_data: *const CTYPE = self_.const_data_ptr::<CTYPE>();
    let a_data: *const CTYPE = mat2.const_data_ptr::<CTYPE>();
    let c_data: *mut CTYPE = out.mutable_data_ptr::<CTYPE>();

    let batch_size: i64 = self_.size(0) as i64;
    let n: i64 = self_.size(1) as i64;
    let k: i64 = self_.size(2) as i64;
    let m: i64 = mat2.size(2) as i64;

    for i in 0..batch_size {
        let a: *const CTYPE = unsafe { a_data.add((i * m * k) as usize) };
        let b: *const CTYPE = unsafe { b_data.add((i * k * n) as usize) };
        let c: *mut CTYPE = unsafe { c_data.add((i * m * n) as usize) };

        unsafe {
            <CTYPE as OptGemm>::opt_gemm(
                TransposeType::NoTranspose,
                TransposeType::NoTranspose,
                m,
                n,
                k,
                <CTYPE as OptGemm>::one(),
                a,
                m,
                b,
                k,
                <CTYPE as OptGemm>::zero(),
                c,
                m,
            );
        }
    }
}

// [spec:et:def:op-bmm.torch.executor.native.resize-out-tensor-fn]
// [spec:et:sem:op-bmm.torch.executor.native.resize-out-tensor-fn]
//
// Resize `out` to the bmm target shape. `m_dim = self.dim() - 2`,
// `n_dim = self.dim() - 1`. Copies `self.size(i)` into the output size for every
// leading dim `i < m_dim`, then — if `m_dim < self.dim()` and `n_dim < mat2.dim()`
// — sets the last two dims to `self.size(m_dim)` and `mat2.size(n_dim)`. If the
// dimensionality is incompatible (`m_dim >= self.dim()` or `n_dim >= mat2.dim()`),
// logs and returns `Error::InvalidArgument`. The `ArrayRef` handed to
// `resize_tensor` is `out.dim()` long. Returns the `resize_tensor` result.
fn resize_out_tensor(self_: &Tensor, mat2: &Tensor, out: &Tensor) -> Error {
    let mut expected_output_size: [SizesType; K_TENSOR_DIMENSION_LIMIT] =
        [0; K_TENSOR_DIMENSION_LIMIT];

    let m_dim: usize = (self_.dim() - 2) as usize;
    let n_dim: usize = (self_.dim() - 1) as usize;

    for i in 0..m_dim {
        expected_output_size[i] = self_.size(i as isize) as SizesType;
    }

    if m_dim >= self_.dim() as usize || n_dim >= mat2.dim() as usize {
        crate::et_log!(Error, "Incompatible matrix multiply dimensions.");
        return Error::InvalidArgument;
    }

    expected_output_size[m_dim] = self_.size(m_dim as isize) as SizesType;
    expected_output_size[n_dim] = mat2.size(n_dim as isize) as SizesType;

    let output_size: ArrayRef<SizesType> =
        ArrayRef::from_raw_parts(expected_output_size.as_ptr(), out.dim() as usize);

    resize_tensor(out, output_size)
}

// bmm.out(Tensor self, Tensor mat2, *, Tensor(a!) out) -> Tensor(a!)
// [spec:et:def:op-bmm.torch.executor.native.opt-bmm-out-fn]
// [spec:et:sem:op-bmm.torch.executor.native.opt-bmm-out-fn]
//
// Optimized `bmm.out`: resize `out` (`resize_out_tensor`) and validate operands
// (`check_bmm_out_args`); either failure records InvalidArgument and returns
// `out`. Then dispatch on `self`'s dtype: complex dtypes over the complex switch,
// real/Half/BFloat16 over the real switch, calling `bmm_kernel<CTYPE>` in both.
// Returns `out`.
//
// PORT-NOTE (cross-module): the C++ dispatches complex dtypes over
// `ET_SWITCH_COMPLEXH_TYPES` (ComplexHalf/ComplexFloat/ComplexDouble) and calls
// `bmm_kernel<CTYPE>`. That requires `cpublas::gemm` for each complex CTYPE; the
// ported CPUBlas provides `gemm_c32`/`gemm_c64` but NOT complex<Half> (see
// CPUBlas.rs — no GemmScalar for Complex<Half>). Because `et_switch_complexh_types!`
// monomorphizes the body for all three arms in one macro, `bmm_kernel::<CTYPE>`
// (bounded on `OptGemm`) cannot be instantiated for the ComplexHalf arm. Mirroring
// the portable op_bmm port, the complex branch therefore fails with
// InvalidArgument as a placeholder; the real branch (REALHBF16) is faithful.
// Unresolved cross-module reference: add a complex<Half> gemm overload (GemmScalar
// impl) + an `OptGemm` impl for the complex portable types, then restore the
// literal `et_switch_complexh_types!` dispatch here (fixer).
pub fn opt_bmm_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    self_: &Tensor,
    mat2: &Tensor,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    crate::et_kernel_check!(
        ctx,
        resize_out_tensor(self_, mat2, out) == Error::Ok,
        InvalidArgument,
        out
    );
    crate::et_kernel_check!(
        ctx,
        check_bmm_out_args(self_, mat2, out),
        InvalidArgument,
        out
    );

    let self_type = self_.scalar_type();

    if is_complex_type(self_type) {
        // PORT-NOTE: unrepresentable complex branch — see fn note.
        // C++: ET_SWITCH_COMPLEXH_TYPES(self_type, ..., bmm_kernel<CTYPE>(self, mat2, out));
        crate::et_kernel_check!(ctx, false, InvalidArgument, out);
    } else {
        crate::et_switch_realhbf16_types!(self_type, ctx, "bmm.out", CTYPE, {
            bmm_kernel::<CTYPE>(self_, mat2, out);
        });
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::core::exec_aten::testing_util::tensor_factory::{
        FactoryValue, TensorFactory,
    };
    use crate::runtime::core::exec_aten::util::scalar_type_util::CppTypeToScalarType;
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

    trait FromI32 {
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

    // op_bmm_test.cpp test_dtype<CTYPE, DTYPE>.
    fn test_dtype<T>()
    where
        T: CppTypeToScalarType + FactoryValue + FromI32 + OptGemm,
    {
        let tf = TensorFactory::<T>::new();

        // Gives 4 * 2 * 3 = 24, shape (10, 3, 5)
        let x = tf.full(vec![10, 3, 4], T::from_i32(2), TensorShapeDynamism::STATIC);
        let y = tf.full(vec![10, 4, 5], T::from_i32(3), TensorShapeDynamism::STATIC);

        let out = tf.zeros_default(vec![10, 3, 5]);
        let mut ctx = context();
        opt_bmm_out(&mut ctx, &x, &y, &out);

        let expected = tf.full(vec![10, 3, 5], T::from_i32(24), TensorShapeDynamism::STATIC);
        assert_tensor_close!(out, expected);
    }

    // ---- direct helper coverage ----

    // [spec:et:sem:op-bmm.torch.executor.native.check-bmm-out-args-fn/test]
    #[test]
    fn check_bmm_out_args_validates_operands() {
        let tf = TensorFactory::<f32>::new();
        let tfi = TensorFactory::<i32>::new();

        crate::runtime::platform::platform::pal_init();

        // Self-consistent 3-D operands pass.
        let x = tf.ones_default(vec![2, 3, 4]);
        let y = tf.ones_default(vec![2, 4, 5]);
        let out = tf.zeros_default(vec![2, 3, 5]);
        assert!(check_bmm_out_args(&x, &y, &out));

        // Rank mismatch between self and mat2.
        let y_2d = tf.ones_default(vec![4, 5]);
        assert!(!check_bmm_out_args(&x, &y_2d, &out));

        // Rank 2 everywhere (dim() != 3).
        let x_2d = tf.ones_default(vec![3, 4]);
        let out_2d = tf.zeros_default(vec![3, 5]);
        assert!(!check_bmm_out_args(&x_2d, &y_2d, &out_2d));

        // Batch mismatch between self and mat2.
        let y_wrong_batch = tf.ones_default(vec![3, 4, 5]);
        assert!(!check_bmm_out_args(&x, &y_wrong_batch, &out));

        // Batch mismatch between self and out.
        let out_wrong_batch = tf.zeros_default(vec![3, 3, 5]);
        assert!(!check_bmm_out_args(&x, &y, &out_wrong_batch));

        // out.size(2) must match mat2.size(2).
        let out_wrong_cols = tf.zeros_default(vec![2, 3, 6]);
        assert!(!check_bmm_out_args(&x, &y, &out_wrong_cols));

        // out.size(1) must match self.size(1).
        let out_wrong_rows = tf.zeros_default(vec![2, 4, 5]);
        assert!(!check_bmm_out_args(&x, &y, &out_wrong_rows));

        // dtype mismatch.
        let out_int = tfi.zeros_default(vec![2, 3, 5]);
        assert!(!check_bmm_out_args(&x, &y, &out_int));
    }

    // [spec:et:sem:op-bmm.torch.executor.native.resize-out-tensor-fn/test]
    #[test]
    fn resize_out_tensor_resizes_to_bmm_shape() {
        let tf = TensorFactory::<f32>::new();
        crate::runtime::platform::platform::pal_init();

        let x = tf.ones_default(vec![2, 3, 4]);
        let y = tf.ones_default(vec![2, 4, 5]);

        // Dynamic-bound out resizes from a smaller compatible bound.
        let out = tf.zeros(vec![2, 3, 5], TensorShapeDynamism::DYNAMIC_BOUND);
        assert_eq!(resize_out_tensor(&x, &y, &out), Error::Ok);
        assert_eq!(out.size(0), 2);
        assert_eq!(out.size(1), 3);
        assert_eq!(out.size(2), 5);

        // Static out with the exact target shape is a no-op success.
        let out_static = tf.zeros_default(vec![2, 3, 5]);
        assert_eq!(resize_out_tensor(&x, &y, &out_static), Error::Ok);

        // Static out with a different shape cannot be resized.
        let out_wrong = tf.zeros_default(vec![2, 3, 6]);
        assert_ne!(resize_out_tensor(&x, &y, &out_wrong), Error::Ok);

        // mat2 with too few dims: n_dim >= mat2.dim() -> InvalidArgument.
        let y_2d = tf.ones_default(vec![4, 5]);
        assert_eq!(
            resize_out_tensor(&x, &y_2d, &out_static),
            Error::InvalidArgument
        );
    }

    // ---- OpBmmOutTest ----

    // [spec:et:sem:op-bmm.torch.executor.native.opt-bmm-out-fn/test]
    #[test]
    fn op_bmm_out_test_output_dim() {
        let tf = TensorFactory::<i32>::new();

        let x = tf.ones_default(vec![10, 3, 4]);
        let y = tf.ones_default(vec![10, 4, 5]);
        let out = tf.zeros_default(vec![10, 3, 5]);

        let mut ctx = context();
        let ret = opt_bmm_out(&mut ctx, &x, &y, &out);
        assert_tensor_eq!(*ret, out);

        let expected = tf.full(vec![10, 3, 5], 4, TensorShapeDynamism::STATIC);
        assert_tensor_eq!(out, expected);
    }

    // [spec:et:sem:op-bmm.torch.executor.native.opt-bmm-out-fn/test]
    // [spec:et:sem:op-bmm.torch.executor.native.bmm-kernel-fn/test]
    #[test]
    fn op_bmm_out_test_output_dim_float() {
        let tf = TensorFactory::<f32>::new();

        #[rustfmt::skip]
        let x = tf.make_default(
            vec![2, 4, 5],
            vec![
                4., 3., 1., 1., 1.,
                3., 1., 4., 4., 2.,
                1., 1., 1., 3., 3.,
                4., 2., 2., 2., 3.,

                1., 3., 1., 4., 4.,
                1., 1., 2., 4., 3.,
                4., 3., 4., 1., 2.,
                1., 4., 4., 4., 4.,
            ],
        );

        #[rustfmt::skip]
        let y = tf.make_default(
            vec![2, 5, 3],
            vec![
                4., 4., 4.,
                2., 3., 1.,
                1., 4., 4.,
                3., 1., 2.,
                1., 4., 3.,

                1., 4., 4.,
                4., 4., 4.,
                2., 1., 4.,
                1., 4., 3.,
                1., 4., 4.,
            ],
        );

        let out = tf.zeros_default(vec![2, 4, 3]);

        let mut ctx = context();
        let ret = opt_bmm_out(&mut ctx, &x, &y, &out);
        assert_tensor_eq!(*ret, out);

        #[rustfmt::skip]
        let expected = tf.make_default(
            vec![2, 4, 3],
            vec![
                27., 34., 28.,
                32., 43., 43.,
                19., 26., 24.,
                31., 44., 39.,

                23., 49., 48.,
                16., 38., 40.,
                27., 44., 55.,
                33., 56., 64.,
            ],
        );
        assert_tensor_eq!(out, expected);
    }

    // [spec:et:sem:op-bmm.torch.executor.native.opt-bmm-out-fn/test]
    // [spec:et:sem:op-bmm.torch.executor.native.bmm-kernel-fn/test]
    #[test]
    fn op_bmm_out_test_all_real_dtypes_supported() {
        test_dtype::<u8>();
        test_dtype::<i8>();
        test_dtype::<i16>();
        test_dtype::<i32>();
        test_dtype::<i64>();
        test_dtype::<f32>();
        test_dtype::<f64>();
        test_dtype::<Half>();
        test_dtype::<BFloat16>();
    }

    // PORT-NOTE: the C++ AllComplexDtypesSupported exercises complex CTYPEs;
    // the ported CPUBlas lacks a complex<Half> gemm and the complex branch of
    // opt_bmm_out is an InvalidArgument placeholder (see fn note). Ported but
    // ignored until complex gemm support lands.
    // [spec:et:sem:op-bmm.torch.executor.native.opt-bmm-out-fn/test]
    #[test]
    #[ignore = "complex dtypes unsupported by the ported opt_bmm_out complex branch"]
    fn op_bmm_out_test_all_complex_dtypes_supported() {
        // Unrepresentable: complex CTYPE (2, 2, 3) x (2, 3, 2) -> (2, 2, 2).
    }

    // [spec:et:sem:op-bmm.torch.executor.native.opt-bmm-out-fn/test]
    // [spec:et:sem:op-bmm.torch.executor.native.bmm-kernel-fn/test]
    #[test]
    fn op_bmm_out_test_empty_input_with_empty_out_tensor_passes() {
        let tf = TensorFactory::<i32>::new();

        let x = tf.full(vec![2, 2, 2], 3, TensorShapeDynamism::STATIC);
        let y = tf.make_default(vec![2, 2, 0], vec![]);
        let out = tf.make_default(vec![2, 2, 0], vec![]);

        assert_eq!(out.numel(), 0);

        let mut ctx = context();
        opt_bmm_out(&mut ctx, &x, &y, &out);
        assert_eq!(ctx.failure_state(), Error::Ok);
        assert_eq!(out.numel(), 0);
    }

    // [spec:et:sem:op-bmm.torch.executor.native.opt-bmm-out-fn/test]
    // [spec:et:sem:op-bmm.torch.executor.native.check-bmm-out-args-fn/test]
    #[test]
    fn op_bmm_out_test_mismatched_dimensions_dies() {
        let tf = TensorFactory::<i32>::new();

        let x = tf.ones_default(vec![2, 10, 3]);
        let wrong_y = tf.ones_default(vec![3, 7, 4]);
        let right_y = tf.ones_default(vec![2, 3, 4]);
        let out = tf.ones_default(vec![2, 10, 4]);

        let mut ctx = context();
        et_expect_kernel_failure!(ctx, opt_bmm_out(&mut ctx, &x, &wrong_y, &out));

        let mut ctx = context();
        assert_tensor_eq!(
            *opt_bmm_out(&mut ctx, &x, &right_y, &out),
            tf.full(vec![2, 10, 4], 3, TensorShapeDynamism::STATIC)
        );
    }

    // [spec:et:sem:op-bmm.torch.executor.native.opt-bmm-out-fn/test]
    #[test]
    fn op_bmm_out_test_mismatched_dimension_size_dies() {
        let tf = TensorFactory::<i32>::new();

        let x = tf.ones_default(vec![2, 10, 3]);
        let wrong_y = tf.ones_default(vec![7, 4]);
        let right_y = tf.ones_default(vec![2, 3, 4]);
        let right_out = tf.ones_default(vec![2, 10, 4]);
        let wrong_out = tf.ones_default(vec![7, 5]);

        let mut ctx = context();
        et_expect_kernel_failure!(ctx, opt_bmm_out(&mut ctx, &x, &right_y, &wrong_out));

        let mut ctx = context();
        et_expect_kernel_failure!(ctx, opt_bmm_out(&mut ctx, &x, &wrong_y, &right_out));
    }

    // [spec:et:sem:op-bmm.torch.executor.native.opt-bmm-out-fn/test]
    #[test]
    fn op_bmm_out_test_wrong_out_shape_dies() {
        let tf = TensorFactory::<i32>::new();

        let x = tf.ones_default(vec![2, 10, 3]);
        let y = tf.ones_default(vec![2, 3, 4]);
        let right_out = tf.ones_default(vec![2, 10, 4]);
        let wrong_out = tf.ones_default(vec![3, 7, 5]);

        let mut ctx = context();
        et_expect_kernel_failure!(ctx, opt_bmm_out(&mut ctx, &x, &y, &wrong_out));

        let mut ctx = context();
        assert_tensor_eq!(
            *opt_bmm_out(&mut ctx, &x, &y, &right_out),
            tf.full(vec![2, 10, 4], 3, TensorShapeDynamism::STATIC)
        );
    }
}
