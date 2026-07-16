//! Literal port of kernels/portable/cpu/op_allclose.cpp.

use crate::runtime::core::exec_aten::util::tensor_util::{
    tensors_have_same_dim_order3, tensors_have_same_shape_and_dtype2,
};
use crate::runtime::core::portable_type::scalar_type::ScalarType;
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::core::portable_type::{BFloat16, Half};
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

// PORT-NOTE: `ET_CHECK_SAME_SHAPE_AND_DTYPE2` / `ET_CHECK_MSG` are C++ fatal
// checks; mirrored with a local abort on failure (message dropped since a fatal
// abort follows), matching the established pattern in tensor_util.rs /
// op_embedding.rs.
macro_rules! et_check_msg {
    ($cond:expr, $($arg:tt)*) => {
        if !($cond) {
            crate::runtime::platform::abort::runtime_abort();
        }
    };
}

// PORT-NOTE: `data_is_close<T>` is templated over the floating types {Float,
// Double, Half, BFloat16}. C++ performs the exact `a[i] != b[i]` comparison in
// `T`, computes `actual_error = std::fabs(a[i] - b[i])` in `T`'s arithmetic
// result type (float for Float/Half/BFloat16, double for Double) and
// `allowed_error = atol + std::fabs(rtol * b[i])` in double (`rtol` is double).
// This `FloatClose` trait reproduces those per-type promotions: `ne` is the
// exact `T` comparison; `actual_error` widens the native (f32/f64) difference to
// f64; `allowed_error` computes in f64. The `!std::isfinite(actual_error)` guard
// maps to `!actual_error.is_finite()`.
trait FloatClose: Copy {
    fn ne(a: Self, b: Self) -> bool;
    fn actual_error(a: Self, b: Self) -> f64;
    fn allowed_error(rtol: f64, atol: f64, b: Self) -> f64;
}

impl FloatClose for f32 {
    fn ne(a: Self, b: Self) -> bool {
        a != b
    }
    fn actual_error(a: Self, b: Self) -> f64 {
        (a - b).abs() as f64
    }
    fn allowed_error(rtol: f64, atol: f64, b: Self) -> f64 {
        atol + (rtol * b as f64).abs()
    }
}

impl FloatClose for f64 {
    fn ne(a: Self, b: Self) -> bool {
        a != b
    }
    fn actual_error(a: Self, b: Self) -> f64 {
        (a - b).abs()
    }
    fn allowed_error(rtol: f64, atol: f64, b: Self) -> f64 {
        atol + (rtol * b).abs()
    }
}

impl FloatClose for Half {
    fn ne(a: Self, b: Self) -> bool {
        a != b
    }
    fn actual_error(a: Self, b: Self) -> f64 {
        (a.to_f32() - b.to_f32()).abs() as f64
    }
    fn allowed_error(rtol: f64, atol: f64, b: Self) -> f64 {
        atol + (rtol * b.to_f64()).abs()
    }
}

impl FloatClose for BFloat16 {
    fn ne(a: Self, b: Self) -> bool {
        a != b
    }
    fn actual_error(a: Self, b: Self) -> f64 {
        (a.to_f32() - b.to_f32()).abs() as f64
    }
    fn allowed_error(rtol: f64, atol: f64, b: Self) -> f64 {
        atol + (rtol * b.to_f64()).abs()
    }
}

// [spec:et:def:op-allclose.torch.executor.native.data-is-close-fn]
// [spec:et:sem:op-allclose.torch.executor.native.data-is-close-fn]
fn data_is_close<T: FloatClose>(
    a: *const T,
    b: *const T,
    numel: usize,
    rtol: f64,
    atol: f64,
) -> bool {
    for i in 0..numel {
        let ai: T = unsafe { *a.add(i) };
        let bi: T = unsafe { *b.add(i) };
        if rtol == 0.0 && atol == 0.0 {
            // Exact comparison; avoid unnecessary math.
            if T::ne(ai, bi) {
                return false;
            }
        } else {
            let allowed_error: f64 = T::allowed_error(rtol, atol, bi);
            let actual_error: f64 = T::actual_error(ai, bi);
            if !actual_error.is_finite() || actual_error > allowed_error {
                return false;
            }
        }
    }
    true
}

// [spec:et:def:op-allclose.torch.executor.native.tensors-are-close-fn]
// [spec:et:sem:op-allclose.torch.executor.native.tensors-are-close-fn]
fn tensors_are_close(a: &Tensor, b: &Tensor, rtol: f64, atol: f64) -> bool {
    // TODO(dbort): Listen to strides instead of assuming that the data is
    // contiguous.

    if a.scalar_type() == ScalarType::Float {
        data_is_close::<f32>(
            a.const_data_ptr::<f32>(),
            b.const_data_ptr::<f32>(),
            a.numel() as usize,
            rtol,
            atol,
        )
    } else if a.scalar_type() == ScalarType::Double {
        data_is_close::<f64>(
            a.const_data_ptr::<f64>(),
            b.const_data_ptr::<f64>(),
            a.numel() as usize,
            rtol,
            atol,
        )
    } else if a.scalar_type() == ScalarType::Half {
        data_is_close::<Half>(
            a.const_data_ptr::<Half>(),
            b.const_data_ptr::<Half>(),
            a.numel() as usize,
            rtol,
            atol,
        )
    } else if a.scalar_type() == ScalarType::BFloat16 {
        data_is_close::<BFloat16>(
            a.const_data_ptr::<BFloat16>(),
            b.const_data_ptr::<BFloat16>(),
            a.numel() as usize,
            rtol,
            atol,
        )
    } else {
        // Non-floating-point types can be compared bitwise.
        let n: usize = a.nbytes();
        let pa: *const u8 = a.mutable_data_ptr_typed() as *const u8;
        let pb: *const u8 = b.mutable_data_ptr_typed() as *const u8;
        let sa: &[u8] = unsafe { core::slice::from_raw_parts(pa, n) };
        let sb: &[u8] = unsafe { core::slice::from_raw_parts(pb, n) };
        sa == sb
    }
}

// PORT-NOTE: C++ `Tensor& out` / returned `Tensor&` become `&'a Tensor` (interior
// mutation through the non-owning handle's raw pointer). `equal_nan` and
// `dummy_param` are unused (ET_UNUSED).

// [spec:et:def:op-allclose.torch.executor.native.allclose-out-fn]
// [spec:et:sem:op-allclose.torch.executor.native.allclose-out-fn]
#[allow(clippy::too_many_arguments)]
fn allclose_out_impl<'a, 'b>(
    self_: &Tensor,
    other: &Tensor,
    rtol: f64,
    atol: f64,
    _equal_nan: bool,
    _dummy_param: bool,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    et_check_msg!(
        tensors_have_same_shape_and_dtype2(self_, other),
        "self and other tensors should have same shape and dtype"
    );
    et_check_msg!(
        out.scalar_type() == ScalarType::Bool,
        "Out tensor must be type Bool; saw type {}",
        out.scalar_type() as i8
    );
    et_check_msg!(
        tensors_have_same_dim_order3(self_, other, out),
        "self, other and out tensors should have same dim order"
    );
    et_check_msg!(
        out.numel() == 1,
        "Out tensor must be a single element; saw {} elements",
        out.numel()
    );
    let out_data: *mut bool = out.mutable_data_ptr::<bool>();
    unsafe {
        *out_data = tensors_are_close(self_, other, rtol, atol);
    }
    out
}

// PORT-NOTE: the C++ `allclose_out` overload without `ctx` holds the real logic
// (ported as `allclose_out_impl` above, since Rust cannot overload on argument
// count); the `ctx`-taking overload discards `ctx` and delegates to it. This is
// the runtime-facing kernel entry point.

// [spec:et:def:op-allclose.torch.executor.native.allclose-out-fn]
// [spec:et:sem:op-allclose.torch.executor.native.allclose-out-fn]
#[allow(clippy::too_many_arguments)]
pub fn allclose_out<'a, 'b>(
    _ctx: &mut KernelRuntimeContext,
    self_: &Tensor,
    other: &Tensor,
    rtol: f64,
    atol: f64,
    equal_nan: bool,
    dummy_param: bool,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    // (void)ctx;
    // TODO(larryliu): Add a context arg to the real op function and remove this
    // wrapper
    allclose_out_impl(self_, other, rtol, atol, equal_nan, dummy_param, out)
}

// PORT-NOTE: the functional `allclose.Tensor` variant only exists for ATen-mode
// registration on the compiler side (`#ifdef USE_ATEN_LIB`), allocating a scalar
// Bool tensor via `torch::tensor(...)` and delegating to `allclose_out`. That
// path depends on the ATen `torch::` allocator, which is out of scope for the
// runtime port; the runtime (`#else`) branch is `ET_ASSERT_UNREACHABLE()`. Both
// the plain and `ctx`-taking overloads reduce to the unreachable abort here.

// [spec:et:def:op-allclose.torch.executor.native.allclose-tensor-fn]
// [spec:et:sem:op-allclose.torch.executor.native.allclose-tensor-fn]
#[allow(clippy::too_many_arguments)]
pub fn allclose_tensor(
    _ctx: &mut KernelRuntimeContext,
    _self_: &Tensor,
    _other: &Tensor,
    _rtol: f64,
    _atol: f64,
    _equal_nan: bool,
    _dummy_param: bool,
) -> ! {
    // TODO(larryliu): Add a context arg to the real op function and remove this
    // wrapper
    crate::runtime::platform::abort::runtime_abort()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::core::exec_aten::testing_util::tensor_factory::{
        FactoryValue, TensorFactory,
    };
    use crate::runtime::core::exec_aten::util::scalar_type_util::CppTypeToScalarType;
    use crate::runtime::core::memory_allocator::{MemoryAllocator, MemoryAllocatorBase};

    const DEFAULT_ATOL: f64 = 1e-08;
    const DEFAULT_RTOL: f64 = 1e-05;

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
    impl FromF64 for f32 {
        fn from_f64(v: f64) -> Self {
            v as f32
        }
    }
    impl FromF64 for f64 {
        fn from_f64(v: f64) -> Self {
            v
        }
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

    fn test_tensors_vary_tolerance<T>(
        rtol: f64,
        rdiff: f64,
        atol: f64,
        adiff: f64,
        should_match: bool,
    ) where
        T: CppTypeToScalarType + FactoryValue + FromF64 + ToF64,
    {
        let tf = TensorFactory::<T>::new();
        let a = tf.ones_default(vec![2, 2]);
        let b = tf.ones_default(vec![2, 2]);

        let a_data = a.mutable_data_ptr::<T>();
        let b_data = b.mutable_data_ptr::<T>();
        unsafe {
            let a0 = (*a_data).from_self();
            *b_data = T::from_f64(a0 + adiff + a0 * rdiff);
        }

        let tf_bool = TensorFactory::<bool>::new();
        let out = tf_bool.zeros_default(vec![1]);

        let mut ctx = context();
        allclose_out(&mut ctx, &a, &b, rtol, atol, false, false, &out);

        let out_data = out.const_data_ptr::<bool>();
        assert_eq!(unsafe { *out_data }, should_match);
    }

    // Helper so `test_tensors_vary_tolerance` can read `a_data[0]` back as f64.
    trait ToF64: Copy {
        fn from_self(self) -> f64;
    }
    impl ToF64 for f32 {
        fn from_self(self) -> f64 {
            self as f64
        }
    }
    impl ToF64 for f64 {
        fn from_self(self) -> f64 {
            self
        }
    }
    impl ToF64 for Half {
        fn from_self(self) -> f64 {
            self.to_f64()
        }
    }
    impl ToF64 for BFloat16 {
        fn from_self(self) -> f64 {
            self.to_f64()
        }
    }

    // allclose_out -> tensors_are_close (Float branch) -> data_is_close (exact path).
    // [spec:et:sem:op-allclose.torch.executor.native.allclose-out-fn/test]
    // [spec:et:sem:op-allclose.torch.executor.native.tensors-are-close-fn/test]
    // [spec:et:sem:op-allclose.torch.executor.native.data-is-close-fn/test]
    #[test]
    fn op_all_close_test_identical_float_tensors() {
        let tf_float = TensorFactory::<f32>::new();
        let a = tf_float.ones_default(vec![2, 2]);
        let b = tf_float.ones_default(vec![2, 2]);
        let tf_bool = TensorFactory::<bool>::new();
        let out = tf_bool.zeros_default(vec![1]);

        let mut ctx = context();
        allclose_out(
            &mut ctx,
            &a,
            &b,
            DEFAULT_RTOL,
            DEFAULT_ATOL,
            false,
            false,
            &out,
        );

        assert_eq!(unsafe { *out.const_data_ptr::<bool>() }, true);
    }

    // [spec:et:sem:op-allclose.torch.executor.native.allclose-out-fn/test]
    // [spec:et:sem:op-allclose.torch.executor.native.tensors-are-close-fn/test]
    // [spec:et:sem:op-allclose.torch.executor.native.data-is-close-fn/test]
    #[test]
    fn op_all_close_test_identical_double_tensors() {
        let tf_double = TensorFactory::<f64>::new();
        let a = tf_double.ones_default(vec![2, 2]);
        let b = tf_double.ones_default(vec![2, 2]);
        let tf_bool = TensorFactory::<bool>::new();
        let out = tf_bool.zeros_default(vec![1]);

        let mut ctx = context();
        allclose_out(
            &mut ctx,
            &a,
            &b,
            DEFAULT_RTOL,
            DEFAULT_ATOL,
            false,
            false,
            &out,
        );

        assert_eq!(unsafe { *out.const_data_ptr::<bool>() }, true);
    }

    // [spec:et:sem:op-allclose.torch.executor.native.allclose-out-fn/test]
    // [spec:et:sem:op-allclose.torch.executor.native.tensors-are-close-fn/test]
    // [spec:et:sem:op-allclose.torch.executor.native.data-is-close-fn/test]
    #[test]
    fn op_all_close_test_non_equal_float_tensors() {
        let tf_float = TensorFactory::<f32>::new();
        let a = tf_float.make_default(vec![2, 2], vec![1., 2., 3., 4.]);
        let b = tf_float.make_default(vec![2, 2], vec![5., 6., 7., 8.]);
        let tf_bool = TensorFactory::<bool>::new();
        let out = tf_bool.zeros_default(vec![1]);

        let mut ctx = context();
        allclose_out(
            &mut ctx,
            &a,
            &b,
            DEFAULT_RTOL,
            DEFAULT_ATOL,
            false,
            false,
            &out,
        );

        assert_eq!(unsafe { *out.const_data_ptr::<bool>() }, false);
    }

    // [spec:et:sem:op-allclose.torch.executor.native.allclose-out-fn/test]
    // [spec:et:sem:op-allclose.torch.executor.native.tensors-are-close-fn/test]
    // [spec:et:sem:op-allclose.torch.executor.native.data-is-close-fn/test]
    #[test]
    fn op_all_close_test_non_equal_double_tensors() {
        let tf_double = TensorFactory::<f64>::new();
        let a = tf_double.make_default(vec![2, 2], vec![1., 2., 3., 4.]);
        let b = tf_double.make_default(vec![2, 2], vec![5., 6., 7., 8.]);
        let tf_bool = TensorFactory::<bool>::new();
        let out = tf_bool.zeros_default(vec![1]);

        let mut ctx = context();
        allclose_out(
            &mut ctx,
            &a,
            &b,
            DEFAULT_RTOL,
            DEFAULT_ATOL,
            false,
            false,
            &out,
        );

        assert_eq!(unsafe { *out.const_data_ptr::<bool>() }, false);
    }

    // Non-floating dtype -> tensors_are_close bitwise-compare branch.
    // [spec:et:sem:op-allclose.torch.executor.native.allclose-out-fn/test]
    // [spec:et:sem:op-allclose.torch.executor.native.tensors-are-close-fn/test]
    #[test]
    fn op_all_close_test_identical_int_tensors() {
        let tf_int = TensorFactory::<i32>::new();
        let a = tf_int.ones_default(vec![2, 2]);
        let b = tf_int.ones_default(vec![2, 2]);
        let tf_bool = TensorFactory::<bool>::new();
        let out = tf_bool.zeros_default(vec![1]);

        let mut ctx = context();
        allclose_out(
            &mut ctx,
            &a,
            &b,
            DEFAULT_RTOL,
            DEFAULT_ATOL,
            false,
            false,
            &out,
        );

        assert_eq!(unsafe { *out.const_data_ptr::<bool>() }, true);
    }

    // [spec:et:sem:op-allclose.torch.executor.native.allclose-out-fn/test]
    // [spec:et:sem:op-allclose.torch.executor.native.tensors-are-close-fn/test]
    #[test]
    fn op_all_close_test_non_equal_int_tensors() {
        let tf_int = TensorFactory::<i32>::new();
        let a = tf_int.make_default(vec![2, 2], vec![1, 2, 3, 4]);
        let b = tf_int.make_default(vec![2, 2], vec![5, 6, 7, 8]);
        let tf_bool = TensorFactory::<bool>::new();
        let out = tf_bool.zeros_default(vec![1]);

        let mut ctx = context();
        allclose_out(
            &mut ctx,
            &a,
            &b,
            DEFAULT_RTOL,
            DEFAULT_ATOL,
            false,
            false,
            &out,
        );

        assert_eq!(unsafe { *out.const_data_ptr::<bool>() }, false);
    }

    // [spec:et:sem:op-allclose.torch.executor.native.allclose-out-fn/test]
    // [spec:et:sem:op-allclose.torch.executor.native.tensors-are-close-fn/test]
    #[test]
    fn op_all_close_test_identical_bool_tensors() {
        let tf_bool = TensorFactory::<bool>::new();
        let a = tf_bool.ones_default(vec![2, 2]);
        let b = tf_bool.ones_default(vec![2, 2]);
        let out = tf_bool.zeros_default(vec![1]);

        let mut ctx = context();
        allclose_out(
            &mut ctx,
            &a,
            &b,
            DEFAULT_RTOL,
            DEFAULT_ATOL,
            false,
            false,
            &out,
        );

        assert_eq!(unsafe { *out.const_data_ptr::<bool>() }, true);
    }

    // [spec:et:sem:op-allclose.torch.executor.native.allclose-out-fn/test]
    // [spec:et:sem:op-allclose.torch.executor.native.tensors-are-close-fn/test]
    #[test]
    fn op_all_close_test_non_equal_bool_tensors() {
        let tf_bool = TensorFactory::<bool>::new();
        let a = tf_bool.ones_default(vec![2, 2]);
        let b = tf_bool.ones_default(vec![2, 2]);
        unsafe {
            *b.mutable_data_ptr::<bool>() = false;
        }
        let out = tf_bool.zeros_default(vec![1]);

        let mut ctx = context();
        allclose_out(
            &mut ctx,
            &a,
            &b,
            DEFAULT_RTOL,
            DEFAULT_ATOL,
            false,
            false,
            &out,
        );

        assert_eq!(unsafe { *out.const_data_ptr::<bool>() }, false);
    }

    // PORT-NOTE: `ET_EXPECT_DEATH` death tests. The `et_check_msg!` failure path
    // calls `runtime_abort` -> `std::process::abort()`, which terminates the
    // process rather than unwinding, so `#[should_panic]` cannot catch it; ported
    // and `#[ignore]`d per the established convention (see broadcast_util.rs).
    // [spec:et:sem:op-allclose.torch.executor.native.allclose-out-fn/test]
    #[test]
    #[should_panic]
    #[ignore]
    fn op_all_close_test_mismatched_input_shapes_death() {
        let tf_int = TensorFactory::<i32>::new();
        let a = tf_int.ones_default(vec![2, 1]);
        let b = tf_int.ones_default(vec![2, 2]);
        let tf_bool = TensorFactory::<bool>::new();
        let out = tf_bool.zeros_default(vec![1]);
        let mut ctx = context();
        allclose_out(
            &mut ctx,
            &a,
            &b,
            DEFAULT_RTOL,
            DEFAULT_ATOL,
            false,
            false,
            &out,
        );
    }

    // [spec:et:sem:op-allclose.torch.executor.native.allclose-out-fn/test]
    #[test]
    #[should_panic]
    #[ignore]
    fn op_all_close_test_mismatched_input_dtypes_death() {
        let tf_float = TensorFactory::<f32>::new();
        let a = tf_float.ones_default(vec![2, 2]);
        let tf_int = TensorFactory::<i32>::new();
        let b = tf_int.ones_default(vec![2, 2]);
        let tf_bool = TensorFactory::<bool>::new();
        let out = tf_bool.zeros_default(vec![1]);
        let mut ctx = context();
        allclose_out(
            &mut ctx,
            &a,
            &b,
            DEFAULT_RTOL,
            DEFAULT_ATOL,
            false,
            false,
            &out,
        );
    }

    // [spec:et:sem:op-allclose.torch.executor.native.allclose-out-fn/test]
    #[test]
    #[should_panic]
    #[ignore]
    fn op_all_close_test_incorrect_output_dtype_death() {
        let tf_float = TensorFactory::<f32>::new();
        let a = tf_float.ones_default(vec![2, 2]);
        let b = tf_float.ones_default(vec![2, 2]);
        let out = tf_float.zeros_default(vec![1]);
        let mut ctx = context();
        allclose_out(
            &mut ctx,
            &a,
            &b,
            DEFAULT_RTOL,
            DEFAULT_ATOL,
            false,
            false,
            &out,
        );
    }

    // [spec:et:sem:op-allclose.torch.executor.native.allclose-out-fn/test]
    #[test]
    #[should_panic]
    #[ignore]
    fn op_all_close_test_incorrect_output_shape_death() {
        let tf_float = TensorFactory::<f32>::new();
        let a = tf_float.ones_default(vec![2, 2]);
        let b = tf_float.ones_default(vec![2, 2]);
        let tf_bool = TensorFactory::<bool>::new();
        let out = tf_bool.zeros_default(vec![2, 2]);
        let mut ctx = context();
        allclose_out(
            &mut ctx,
            &a,
            &b,
            DEFAULT_RTOL,
            DEFAULT_ATOL,
            false,
            false,
            &out,
        );
    }

    // ET_FORALL_FLOATHBF16_TYPES
    // [spec:et:sem:op-allclose.torch.executor.native.allclose-out-fn/test]
    // [spec:et:sem:op-allclose.torch.executor.native.tensors-are-close-fn/test]
    // [spec:et:sem:op-allclose.torch.executor.native.data-is-close-fn/test]
    #[test]
    fn op_all_close_test_tensors_vary_within_relative_tolerance() {
        test_tensors_vary_tolerance::<f32>(1e-01, 1e-02, 0., 0., true);
        test_tensors_vary_tolerance::<f64>(1e-01, 1e-02, 0., 0., true);
        test_tensors_vary_tolerance::<Half>(1e-01, 1e-02, 0., 0., true);
        test_tensors_vary_tolerance::<BFloat16>(1e-01, 1e-02, 0., 0., true);
    }

    // [spec:et:sem:op-allclose.torch.executor.native.allclose-out-fn/test]
    // [spec:et:sem:op-allclose.torch.executor.native.data-is-close-fn/test]
    #[test]
    fn op_all_close_test_tensors_vary_outside_relative_tolerance() {
        test_tensors_vary_tolerance::<f32>(1e-01, 1., 0., 0., false);
        test_tensors_vary_tolerance::<f64>(1e-01, 1., 0., 0., false);
        test_tensors_vary_tolerance::<Half>(1e-01, 1., 0., 0., false);
        test_tensors_vary_tolerance::<BFloat16>(1e-01, 1., 0., 0., false);
    }

    // [spec:et:sem:op-allclose.torch.executor.native.allclose-out-fn/test]
    // [spec:et:sem:op-allclose.torch.executor.native.data-is-close-fn/test]
    #[test]
    fn op_all_close_test_tensors_vary_within_absolute_tolerance() {
        test_tensors_vary_tolerance::<f32>(0., 0., 1e-01, 1e-02, true);
        test_tensors_vary_tolerance::<f64>(0., 0., 1e-01, 1e-02, true);
        test_tensors_vary_tolerance::<Half>(0., 0., 1e-01, 1e-02, true);
        test_tensors_vary_tolerance::<BFloat16>(0., 0., 1e-01, 1e-02, true);
    }

    // [spec:et:sem:op-allclose.torch.executor.native.allclose-out-fn/test]
    // [spec:et:sem:op-allclose.torch.executor.native.data-is-close-fn/test]
    #[test]
    fn op_all_close_test_tensors_vary_outside_absolute_tolerance() {
        test_tensors_vary_tolerance::<f32>(0., 0., 1e-01, 1., false);
        test_tensors_vary_tolerance::<f64>(0., 0., 1e-01, 1., false);
        test_tensors_vary_tolerance::<Half>(0., 0., 1e-01, 1., false);
        test_tensors_vary_tolerance::<BFloat16>(0., 0., 1e-01, 1., false);
    }

    // Exercises the exact-comparison branch (rtol == 0 && atol == 0) of data_is_close.
    // [spec:et:sem:op-allclose.torch.executor.native.allclose-out-fn/test]
    // [spec:et:sem:op-allclose.torch.executor.native.data-is-close-fn/test]
    #[test]
    fn op_all_close_test_tensors_vary_with_zero_tolerance() {
        test_tensors_vary_tolerance::<f32>(0., 0., 0., 1e-01, false);
        test_tensors_vary_tolerance::<f64>(0., 0., 0., 1e-01, false);
        test_tensors_vary_tolerance::<Half>(0., 0., 0., 1e-01, false);
        test_tensors_vary_tolerance::<BFloat16>(0., 0., 0., 1e-01, false);
    }

    // PORT-NOTE: `allclose_tensor` is the runtime-branch `ET_ASSERT_UNREACHABLE()`
    // functional variant; calling it aborts. `std::process::abort()` cannot be
    // caught by `#[should_panic]`, so `#[ignore]`d per the death-test convention.
    // [spec:et:sem:op-allclose.torch.executor.native.allclose-tensor-fn/test]
    #[test]
    #[should_panic]
    #[ignore]
    fn op_all_close_test_allclose_tensor_unreachable() {
        let tf_float = TensorFactory::<f32>::new();
        let a = tf_float.ones_default(vec![2, 2]);
        let b = tf_float.ones_default(vec![2, 2]);
        let mut ctx = context();
        allclose_tensor(&mut ctx, &a, &b, DEFAULT_RTOL, DEFAULT_ATOL, false, false);
    }
}
