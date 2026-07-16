//! Literal port of runtime/core/exec_aten/testing_util/tensor_util.{h,cpp}.
//!
//! This is the non-ATen (`!USE_ATEN_LIB`, `torch::executor`) branch. It ships as
//! a regular test-support library (not `cfg(test)`); the files are excluded from
//! the port manifest, so no `[spec:et:...]` annotations are carried.
//!
//! The gtest `MATCHER_P`/`EXPECT_THAT` machinery has no Rust analog. The
//! `EXPECT_TENSOR_*`/`ASSERT_TENSOR_*` gmock macros are re-expressed as the
//! `assert_tensor_eq!` / `assert_tensor_close!` (and friends) `macro_rules!` at
//! the bottom of this file, delegating to the ported `tensors_are_close` /
//! `tensor_data_is_close` predicates that preserve the exact comparison
//! semantics.

use crate::runtime::core::portable_type::scalar_type::ScalarType;
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::core::portable_type::{BFloat16, Half};
use crate::runtime::platform::abort::runtime_abort;

// PORT-NOTE: `ET_CHECK_MSG(cond, ...)` aborts via the PAL abort path when the
// condition is false; the assert module is not ported. Message formatting is
// dropped since the abort is unconditional.
macro_rules! et_check_msg {
    ($cond:expr $(,)?) => {
        if !($cond) {
            runtime_abort();
        }
    };
    ($cond:expr, $($arg:tt)*) => {
        if !($cond) {
            runtime_abort();
        }
    };
}

pub mod internal {
    pub const K_DEFAULT_RTOL: f64 = 1e-5;
    pub const K_DEFAULT_ATOL: f64 = 1e-8;
    // Per https://en.wikipedia.org/wiki/Half-precision_floating-point_format,
    // float16 has about 3.3 digits of precision.
    pub const K_DEFAULT_HALF_ATOL: f64 = 1e-3;

    // Following similar reasoning to float16, BFloat16 has
    // math.log10(2**8) = 2.4 digits of precision.
    pub const K_DEFAULT_BFLOAT16_ATOL: f64 = 1e-2;
}

/// PORT-NOTE: The C++ `element_is_close<T>` is templated over the element type
/// and, for reduced-floating-point types (Half/BFloat16), recurses after casting
/// to `float`. Here the comparison is performed entirely in `f64` (mirroring the
/// C++ math, which promotes to `double` for the tolerance computation). Callers
/// convert Half/BFloat16 to `f32` before invoking, matching the C++ recursion
/// that casts reduced types to `float` first.
///
/// T must be a floating point comparison performed in f64.
fn element_is_close(a: f64, b: f64, rtol: f64, atol: f64) -> bool {
    if a.is_nan() && b.is_nan() {
        // NaN == NaN
    } else if !a.is_finite() && !b.is_finite() && ((a > 0.0) == (b > 0.0)) {
        // -Inf == -Inf
        // +Inf == +Inf
    } else if rtol == 0.0 && atol == 0.0 {
        // Exact comparison; avoid unnecessary math.
        if a != b {
            return false;
        }
    } else {
        let allowed_error = atol + (rtol * b).abs();
        let actual_error = (a - b).abs();
        if !actual_error.is_finite() || actual_error > allowed_error {
            return false;
        }
    }
    true
}

/// Trait bridging an element type to the `f64` comparison performed by
/// `element_is_close`. Non-floating-point tensor data never reaches this path
/// (it is compared bitwise), so only the floating types are covered.
trait CloseElement: Copy {
    fn to_f64(self) -> f64;
}
impl CloseElement for f32 {
    fn to_f64(self) -> f64 {
        self as f64
    }
}
impl CloseElement for f64 {
    fn to_f64(self) -> f64 {
        self
    }
}
impl CloseElement for Half {
    fn to_f64(self) -> f64 {
        // C++ casts reduced-float types to `float` before comparing.
        self.to_f32() as f64
    }
}
impl CloseElement for BFloat16 {
    fn to_f64(self) -> f64 {
        self.to_f32() as f64
    }
}

/// Returns true if the two arrays are close according to the description on
/// `tensors_are_close()`.
///
/// # Safety
/// `a`/`b` must each point to at least `numel` valid `T` elements when
/// `numel > 0`.
unsafe fn data_is_close<T: CloseElement>(
    a: *const T,
    b: *const T,
    numel: usize,
    rtol: f64,
    atol: f64,
) -> bool {
    et_check_msg!(
        numel == 0 || (!a.is_null() && !b.is_null()),
        "Pointers must not be null when numel > 0"
    );
    if a == b {
        return true;
    }
    for i in 0..numel {
        let ai = unsafe { *a.add(i) };
        let bi = unsafe { *b.add(i) };
        if !element_is_close(ai.to_f64(), bi.to_f64(), rtol, atol) {
            return false;
        }
    }
    true
}

/// Mirrors `memcmp(a, b, nbytes) == 0` over the tensors' raw bytes.
///
/// # Safety
/// `a`/`b` must each point to at least `nbytes` valid bytes.
unsafe fn bytes_eq(
    a: *const core::ffi::c_void,
    b: *const core::ffi::c_void,
    nbytes: usize,
) -> bool {
    let a = a as *const u8;
    let b = b as *const u8;
    for i in 0..nbytes {
        if unsafe { *a.add(i) } != unsafe { *b.add(i) } {
            return false;
        }
    }
    true
}

fn default_atol_for_type(t: ScalarType) -> f64 {
    if t == ScalarType::Half {
        return internal::K_DEFAULT_HALF_ATOL;
    }
    if t == ScalarType::BFloat16 {
        return internal::K_DEFAULT_BFLOAT16_ATOL;
    }
    internal::K_DEFAULT_ATOL
}

/// Returns true if the tensors are of the same shape and dtype, and if all
/// elements are close to each other.
///
/// An element A is close to B when one is true:
///
/// (1) A is equal to B.
/// (2) A and B are both NaN, are both -infinity, or are both +infinity.
/// (3) The error abs(A - B) is finite and less than the max error
///     (atol + abs(rtol * B)).
///
/// If both rtol/atol are zero, this checks for exact equality. rtol/atol are
/// ignored and treated as zero for non-floating-point dtypes.
///
/// PORT-NOTE: the C++ default arguments (`rtol = kDefaultRtol`,
/// `opt_atol = nullopt`) are made explicit; pass `K_DEFAULT_RTOL` / `None` to
/// reproduce them.
pub fn tensors_are_close(a: &Tensor, b: &Tensor, rtol: f64, opt_atol: Option<f64>) -> bool {
    if a.scalar_type() != b.scalar_type() || !a.sizes().equals(b.sizes()) {
        return false;
    }

    // TODO(T132992348): support comparison between tensors of different strides
    et_check_msg!(
        a.strides().equals(b.strides()),
        "The two inputs of `tensors_are_close` function shall have same strides"
    );

    let atol = opt_atol.unwrap_or_else(|| default_atol_for_type(a.scalar_type()));

    if a.nbytes() == 0 {
        // Valid for a zero-size tensor to have a null data pointer.
        true
    } else if a.scalar_type() == ScalarType::Float {
        unsafe {
            data_is_close::<f32>(
                a.const_data_ptr::<f32>(),
                b.const_data_ptr::<f32>(),
                a.numel() as usize,
                rtol,
                atol,
            )
        }
    } else if a.scalar_type() == ScalarType::Double {
        unsafe {
            data_is_close::<f64>(
                a.const_data_ptr::<f64>(),
                b.const_data_ptr::<f64>(),
                a.numel() as usize,
                rtol,
                atol,
            )
        }
    } else if a.scalar_type() == ScalarType::Half {
        unsafe {
            data_is_close::<Half>(
                a.const_data_ptr::<Half>(),
                b.const_data_ptr::<Half>(),
                a.numel() as usize,
                rtol,
                atol,
            )
        }
    } else if a.scalar_type() == ScalarType::BFloat16 {
        unsafe {
            data_is_close::<BFloat16>(
                a.const_data_ptr::<BFloat16>(),
                b.const_data_ptr::<BFloat16>(),
                a.numel() as usize,
                rtol,
                atol,
            )
        }
    } else {
        // Non-floating-point types can be compared bitwise.
        unsafe {
            bytes_eq(
                a.const_data_ptr_typed(),
                b.const_data_ptr_typed(),
                a.nbytes(),
            )
        }
    }
}

/// Returns true if the tensors are of the same numel and dtype, and if all
/// elements are close to each other. The tensor shapes do not need to be same.
pub fn tensor_data_is_close(a: &Tensor, b: &Tensor, rtol: f64, opt_atol: Option<f64>) -> bool {
    if a.scalar_type() != b.scalar_type() || a.numel() != b.numel() {
        return false;
    }

    let atol = opt_atol.unwrap_or_else(|| default_atol_for_type(a.scalar_type()));
    if a.nbytes() == 0 {
        true
    } else if a.scalar_type() == ScalarType::Float {
        unsafe {
            data_is_close::<f32>(
                a.const_data_ptr::<f32>(),
                b.const_data_ptr::<f32>(),
                a.numel() as usize,
                rtol,
                atol,
            )
        }
    } else if a.scalar_type() == ScalarType::Double {
        unsafe {
            data_is_close::<f64>(
                a.const_data_ptr::<f64>(),
                b.const_data_ptr::<f64>(),
                a.numel() as usize,
                rtol,
                atol,
            )
        }
    } else {
        // Non-floating-point types can be compared bitwise.
        unsafe {
            bytes_eq(
                a.const_data_ptr_typed(),
                b.const_data_ptr_typed(),
                a.nbytes(),
            )
        }
    }
}

/// Returns true if the two lists are of the same length, and
/// tensor_data_is_close(tensors_a[i], tensors_b[i], rtol, atol) is true for all
/// i.
///
/// PORT-NOTE: C++ passes `opt_atol` straight through to `tensors_are_close`
/// (note: the header wraps element comparison in `tensors_are_close`, so this
/// mirrors that).
pub fn tensor_lists_are_close(
    tensors_a: &[Tensor],
    tensors_b: &[Tensor],
    rtol: f64,
    opt_atol: Option<f64>,
) -> bool {
    if tensors_a.len() != tensors_b.len() {
        return false;
    }
    for i in 0..tensors_a.len() {
        if !tensors_are_close(&tensors_a[i], &tensors_b[i], rtol, opt_atol) {
            return false;
        }
    }
    true
}

// Macro re-expressions of the gmock `EXPECT_TENSOR_*` / `ASSERT_TENSOR_*` family.
// gtest's `EXPECT_*` (non-fatal) vs `ASSERT_*` (fatal) distinction collapses in
// Rust: both variants panic on failure (like a failed `assert!`), so the two
// spellings are provided as aliases for source-level correspondence.

/// `EXPECT_TENSOR_EQ` / `ASSERT_TENSOR_EQ`: exact equality (rtol=0, atol=0).
#[macro_export]
macro_rules! assert_tensor_eq {
    ($t1:expr, $t2:expr $(,)?) => {
        assert!(
            $crate::runtime::core::exec_aten::testing_util::tensor_util::tensors_are_close(
                &$t1,
                &$t2,
                0.0,
                Some(0.0)
            ),
            "tensors are not equal"
        )
    };
}

/// `EXPECT_TENSOR_NE` / `ASSERT_TENSOR_NE`.
#[macro_export]
macro_rules! assert_tensor_ne {
    ($t1:expr, $t2:expr $(,)?) => {
        assert!(
            !$crate::runtime::core::exec_aten::testing_util::tensor_util::tensors_are_close(
                &$t1,
                &$t2,
                0.0,
                Some(0.0)
            ),
            "tensors are unexpectedly equal"
        )
    };
}

/// `EXPECT_TENSOR_CLOSE` / `ASSERT_TENSOR_CLOSE`: default rtol, dtype-derived
/// atol.
#[macro_export]
macro_rules! assert_tensor_close {
    ($t1:expr, $t2:expr $(,)?) => {
        assert!(
            $crate::runtime::core::exec_aten::testing_util::tensor_util::tensors_are_close(
                &$t1,
                &$t2,
                $crate::runtime::core::exec_aten::testing_util::tensor_util::internal::K_DEFAULT_RTOL,
                None
            ),
            "tensors are not close"
        )
    };
}

/// `EXPECT_TENSOR_NOT_CLOSE` / `ASSERT_TENSOR_NOT_CLOSE`.
#[macro_export]
macro_rules! assert_tensor_not_close {
    ($t1:expr, $t2:expr $(,)?) => {
        assert!(
            !$crate::runtime::core::exec_aten::testing_util::tensor_util::tensors_are_close(
                &$t1,
                &$t2,
                $crate::runtime::core::exec_aten::testing_util::tensor_util::internal::K_DEFAULT_RTOL,
                None
            ),
            "tensors are unexpectedly close"
        )
    };
}

/// `EXPECT_TENSOR_CLOSE_WITH_TOL` / `ASSERT_TENSOR_CLOSE_WITH_TOL`.
#[macro_export]
macro_rules! assert_tensor_close_with_tol {
    ($t1:expr, $t2:expr, $rtol:expr, $atol:expr $(,)?) => {
        assert!(
            $crate::runtime::core::exec_aten::testing_util::tensor_util::tensors_are_close(
                &$t1,
                &$t2,
                $rtol,
                Some($atol)
            ),
            "tensors are not close within tolerance"
        )
    };
}

/// `EXPECT_TENSOR_DATA_EQ` / `ASSERT_TENSOR_DATA_EQ`.
#[macro_export]
macro_rules! assert_tensor_data_eq {
    ($t1:expr, $t2:expr $(,)?) => {
        assert!(
            $crate::runtime::core::exec_aten::testing_util::tensor_util::tensor_data_is_close(
                &$t1,
                &$t2,
                0.0,
                Some(0.0)
            ),
            "tensor data is not equal"
        )
    };
}

/// `EXPECT_TENSOR_DATA_NE` / `ASSERT_TENSOR_DATA_NE`.
#[macro_export]
macro_rules! assert_tensor_data_ne {
    ($t1:expr, $t2:expr $(,)?) => {
        assert!(
            !$crate::runtime::core::exec_aten::testing_util::tensor_util::tensor_data_is_close(
                &$t1,
                &$t2,
                0.0,
                Some(0.0)
            ),
            "tensor data is unexpectedly equal"
        )
    };
}

/// `EXPECT_TENSOR_DATA_CLOSE` / `ASSERT_TENSOR_DATA_CLOSE`.
#[macro_export]
macro_rules! assert_tensor_data_close {
    ($t1:expr, $t2:expr $(,)?) => {
        assert!(
            $crate::runtime::core::exec_aten::testing_util::tensor_util::tensor_data_is_close(
                &$t1,
                &$t2,
                $crate::runtime::core::exec_aten::testing_util::tensor_util::internal::K_DEFAULT_RTOL,
                None
            ),
            "tensor data is not close"
        )
    };
}

/// `EXPECT_TENSOR_LISTS_EQ` / `ASSERT_TENSOR_LISTS_EQ`.
#[macro_export]
macro_rules! assert_tensor_lists_eq {
    ($t1:expr, $t2:expr $(,)?) => {
        assert!(
            $crate::runtime::core::exec_aten::testing_util::tensor_util::tensor_lists_are_close(
                &$t1,
                &$t2,
                0.0,
                Some(0.0)
            ),
            "tensor lists are not equal"
        )
    };
}

/// `EXPECT_TENSOR_LISTS_CLOSE` / `ASSERT_TENSOR_LISTS_CLOSE`.
#[macro_export]
macro_rules! assert_tensor_lists_close {
    ($t1:expr, $t2:expr $(,)?) => {
        assert!(
            $crate::runtime::core::exec_aten::testing_util::tensor_util::tensor_lists_are_close(
                &$t1,
                &$t2,
                $crate::runtime::core::exec_aten::testing_util::tensor_util::internal::K_DEFAULT_RTOL,
                None
            ),
            "tensor lists are not close"
        )
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::core::exec_aten::testing_util::tensor_factory::FactoryValue;
    use crate::runtime::core::exec_aten::testing_util::tensor_factory::TensorFactory;
    use crate::runtime::core::exec_aten::util::scalar_type_util::CppTypeToScalarType;
    use crate::runtime::core::portable_type::device::{DeviceIndex, DeviceType};
    use crate::runtime::core::portable_type::tensor_impl::{
        DimOrderType, SizesType, StridesType, TensorImpl,
    };
    use crate::runtime::core::tensor_shape_dynamism::TensorShapeDynamism;
    use crate::{
        assert_tensor_close, assert_tensor_close_with_tol, assert_tensor_data_close,
        assert_tensor_data_eq, assert_tensor_eq, assert_tensor_ne,
    };

    // PORT-NOTE: the gmock `IsCloseTo`/`IsEqualTo`/`IsDataCloseTo` matcher lines
    // in the C++ combining macros have no Rust analog (mirroring how this module
    // dropped the `MATCHER_P` machinery). Each combining macro is re-expressed
    // over the ported `tensors_are_close`/`tensor_data_is_close` predicates and
    // the `assert_tensor_*` macros, preserving the same commutative checks.
    //
    // The bare `tensors_are_close`/`tensor_data_is_close`/`tensor_lists_are_close`
    // calls in the C++ macros use the functions' default arguments
    // (`rtol=kDefaultRtol, opt_atol=nullopt`), so they map to
    // `K_DEFAULT_RTOL, None` here — NOT exact `0.0, Some(0.0)`.

    macro_rules! expect_tensors_close_and_equal {
        ($t1:expr, $t2:expr) => {{
            assert!(tensors_are_close(
                &$t1,
                &$t2,
                internal::K_DEFAULT_RTOL,
                None
            ));
            assert!(tensors_are_close(
                &$t2,
                &$t1,
                internal::K_DEFAULT_RTOL,
                None
            ));
            assert_tensor_close!($t1, $t2);
            assert_tensor_close!($t2, $t1);
            assert_tensor_eq!($t1, $t2);
            assert_tensor_eq!($t2, $t1);
        }};
    }

    macro_rules! expect_tensors_close_but_not_equal {
        ($t1:expr, $t2:expr) => {{
            assert!(tensors_are_close(
                &$t1,
                &$t2,
                internal::K_DEFAULT_RTOL,
                None
            ));
            assert!(tensors_are_close(
                &$t2,
                &$t1,
                internal::K_DEFAULT_RTOL,
                None
            ));
            assert_tensor_close!($t1, $t2);
            assert_tensor_close!($t2, $t1);
            assert_tensor_ne!($t1, $t2);
            assert_tensor_ne!($t2, $t1);
        }};
    }

    macro_rules! expect_tensors_not_close_or_equal {
        ($t1:expr, $t2:expr) => {{
            assert!(!tensors_are_close(
                &$t1,
                &$t2,
                internal::K_DEFAULT_RTOL,
                None
            ));
            assert!(!tensors_are_close(
                &$t2,
                &$t1,
                internal::K_DEFAULT_RTOL,
                None
            ));
            crate::assert_tensor_not_close!($t1, $t2);
            crate::assert_tensor_not_close!($t2, $t1);
            assert_tensor_ne!($t1, $t2);
            assert_tensor_ne!($t2, $t1);
        }};
    }

    macro_rules! expect_tensors_data_close_and_equal {
        ($t1:expr, $t2:expr) => {{
            assert!(tensor_data_is_close(
                &$t1,
                &$t2,
                internal::K_DEFAULT_RTOL,
                None
            ));
            assert!(tensor_data_is_close(
                &$t2,
                &$t1,
                internal::K_DEFAULT_RTOL,
                None
            ));
            assert_tensor_data_close!($t1, $t2);
            assert_tensor_data_close!($t2, $t1);
            assert_tensor_data_eq!($t1, $t2);
            assert_tensor_data_eq!($t2, $t1);
        }};
    }

    macro_rules! expect_tensors_data_not_close_or_equal {
        ($t1:expr, $t2:expr) => {{
            assert!(!tensor_data_is_close(
                &$t1,
                &$t2,
                internal::K_DEFAULT_RTOL,
                None
            ));
            assert!(!tensor_data_is_close(
                &$t2,
                &$t1,
                internal::K_DEFAULT_RTOL,
                None
            ));
            crate::assert_tensor_data_ne!($t1, $t2);
            crate::assert_tensor_data_ne!($t2, $t1);
        }};
    }

    macro_rules! expect_tensor_lists_close_and_equal {
        ($l1:expr, $l2:expr) => {{
            assert!(tensor_lists_are_close(
                &$l1,
                &$l2,
                internal::K_DEFAULT_RTOL,
                None
            ));
            assert!(tensor_lists_are_close(
                &$l2,
                &$l1,
                internal::K_DEFAULT_RTOL,
                None
            ));
            crate::assert_tensor_lists_close!($l1, $l2);
            crate::assert_tensor_lists_close!($l2, $l1);
            crate::assert_tensor_lists_eq!($l1, $l2);
            crate::assert_tensor_lists_eq!($l2, $l1);
        }};
    }

    macro_rules! expect_tensor_lists_close_but_not_equal {
        ($l1:expr, $l2:expr) => {{
            assert!(tensor_lists_are_close(
                &$l1,
                &$l2,
                internal::K_DEFAULT_RTOL,
                None
            ));
            assert!(tensor_lists_are_close(
                &$l2,
                &$l1,
                internal::K_DEFAULT_RTOL,
                None
            ));
            crate::assert_tensor_lists_close!($l1, $l2);
            crate::assert_tensor_lists_close!($l2, $l1);
            // PORT-NOTE: `EXPECT_TENSOR_LISTS_NE` (== `Not(IsListEqualTo)`, exact
            // rtol/atol = 0) was not ported as a macro; asserted directly.
            assert!(!tensor_lists_are_close(&$l1, &$l2, 0.0, Some(0.0)));
            assert!(!tensor_lists_are_close(&$l2, &$l1, 0.0, Some(0.0)));
        }};
    }

    macro_rules! expect_tensor_lists_not_close_or_equal {
        ($l1:expr, $l2:expr) => {{
            assert!(!tensor_lists_are_close(
                &$l1,
                &$l2,
                internal::K_DEFAULT_RTOL,
                None
            ));
            assert!(!tensor_lists_are_close(
                &$l2,
                &$l1,
                internal::K_DEFAULT_RTOL,
                None
            ));
            // `EXPECT_TENSOR_LISTS_NE` (exact) part.
            assert!(!tensor_lists_are_close(&$l1, &$l2, 0.0, Some(0.0)));
            assert!(!tensor_lists_are_close(&$l2, &$l1, 0.0, Some(0.0)));
        }};
    }

    // Mirrors the anonymous-namespace `size_to_numel` helper.
    fn size_to_numel(sizes: &[i32]) -> i32 {
        let mut numel: i32 = 1;
        for &size in sizes {
            numel *= size;
        }
        numel
    }

    // [spec:et:sem:tensor-util.executorch.runtime.tensors-are-close-fn/test]
    #[test]
    fn tensor_util_test_different_dtypes_are_not_close_or_equal() {
        let tf_int = TensorFactory::<i32>::new();
        let a = tf_int.make_default(vec![2, 2], vec![1, 2, 4, 8]);

        let tf_long = TensorFactory::<i64>::new();
        let b = tf_long.make_default(vec![2, 2], vec![1, 2, 4, 8]);

        expect_tensors_not_close_or_equal!(a, b);
    }

    // [spec:et:sem:tensor-util.executorch.runtime.tensors-are-close-fn/test]
    #[test]
    fn tensor_util_test_different_sizes_are_not_close_or_equal() {
        let tf = TensorFactory::<i32>::new();

        let a = tf.make_default(vec![2, 2], vec![1, 2, 4, 8]);
        let b = tf.make_default(vec![4], vec![1, 2, 4, 8]);

        expect_tensors_not_close_or_equal!(a, b);
    }

    // PORT-NOTE: `ET_EXPECT_DEATH` death test (different strides abort in
    // `tensors_are_close`). `runtime_abort` -> `libc::abort()` terminates the
    // process, so `#[should_panic]` cannot catch it; ported and `#[ignore]`d.
    // [spec:et:sem:tensor-util.executorch.runtime.tensors-are-close-fn/test]
    #[test]
    #[should_panic]
    #[ignore]
    fn tensor_util_test_different_layouts_dies() {
        let tf = TensorFactory::<i32>::new();
        let a = tf.make(
            vec![2, 2],
            vec![1, 2, 4, 8],
            vec![1, 2],
            TensorShapeDynamism::STATIC,
        );
        let b = tf.make(
            vec![2, 2],
            vec![1, 2, 4, 8],
            vec![2, 1],
            TensorShapeDynamism::STATIC,
        );
        let _ = tensors_are_close(&a, &b, 0.0, Some(0.0));
    }

    // [spec:et:sem:tensor-util.executorch.runtime.tensors-are-close-fn/test]
    #[test]
    fn tensor_util_test_int_tensor_is_close_and_equal_to_itself() {
        let tf = TensorFactory::<i32>::new();
        let t = tf.make_default(vec![2, 2], vec![1, 2, 4, 8]);
        expect_tensors_close_and_equal!(t, t);
    }

    // [spec:et:sem:tensor-util.executorch.runtime.tensors-are-close-fn/test]
    #[test]
    fn tensor_util_test_identical_int_tensors_are_close_and_equal() {
        let tf = TensorFactory::<i32>::new();
        let a = tf.make_default(vec![2, 2], vec![1, 2, 4, 8]);
        let b = tf.make_default(vec![2, 2], vec![1, 2, 4, 8]);
        expect_tensors_close_and_equal!(a, b);
    }

    // [spec:et:sem:tensor-util.executorch.runtime.tensors-are-close-fn/test]
    #[test]
    fn tensor_util_test_non_identical_int_tensors_are_not_close_or_equal() {
        let tf = TensorFactory::<i32>::new();
        let a = tf.make_default(vec![2, 2], vec![1, 2, 4, 8]);
        let b = tf.make_default(vec![2, 2], vec![99, 2, 4, 8]);
        expect_tensors_not_close_or_equal!(a, b);
    }

    // [spec:et:sem:tensor-util.executorch.runtime.tensors-are-close-fn/test]
    #[test]
    fn tensor_util_test_empty_tensors_are_close_and_equal() {
        let tf = TensorFactory::<i32>::new();
        let a = tf.make_default(vec![0, 2], vec![]);
        assert_eq!(a.numel(), 0);
        assert_eq!(a.nbytes(), 0);
        let b = tf.make_default(vec![0, 2], vec![]);
        assert_eq!(b.numel(), 0);
        assert_eq!(b.nbytes(), 0);
        expect_tensors_close_and_equal!(a, b);
    }

    // [spec:et:sem:tensor-util.executorch.runtime.tensors-are-close-fn/test]
    #[test]
    fn tensor_util_test_float_tensor_is_close_and_equal_to_itself() {
        let tf = TensorFactory::<f32>::new();
        let t = tf.make_default(vec![2, 2], vec![1.1, 2.2, 4.4, 8.8]);
        expect_tensors_close_and_equal!(t, t);
    }

    // [spec:et:sem:tensor-util.executorch.runtime.tensors-are-close-fn/test]
    #[test]
    fn tensor_util_test_identical_float_tensors_are_close_and_equal() {
        let tf = TensorFactory::<f32>::new();
        let a = tf.make_default(vec![2, 2], vec![1.1, 2.2, 4.4, 8.8]);
        let b = tf.make_default(vec![2, 2], vec![1.1, 2.2, 4.4, 8.8]);
        expect_tensors_close_and_equal!(a, b);
    }

    // [spec:et:sem:tensor-util.executorch.runtime.tensors-are-close-fn/test]
    #[test]
    fn tensor_util_test_nearly_identical_float_tensors_are_close_but_not_equal() {
        let tf = TensorFactory::<f32>::new();
        let a = tf.make_default(vec![2, 2], vec![1.1, 2.2, 4.4, 8.8]);
        let b = tf.make_default(
            vec![2, 2],
            // First data element is slightly larger.
            vec![1.1f32.next_up(), 2.2, 4.4, 8.8],
        );
        expect_tensors_close_but_not_equal!(a, b);
    }

    // [spec:et:sem:tensor-util.executorch.runtime.tensors-are-close-fn/test]
    #[test]
    fn tensor_util_test_non_identical_float_tensors_are_not_close_or_equal() {
        let tf = TensorFactory::<f32>::new();
        let a = tf.make_default(vec![2, 2], vec![1.1, 2.2, 4.4, 8.8]);
        let b = tf.make_default(vec![2, 2], vec![99.99, 2.2, 4.4, 8.8]);
        expect_tensors_not_close_or_equal!(a, b);
    }

    // [spec:et:sem:tensor-util.executorch.runtime.tensors-are-close-fn/test]
    #[test]
    fn tensor_util_test_float_nan_elements_are_close_and_equal() {
        let tf = TensorFactory::<f32>::new();
        let a = tf.make_default(vec![2, 2], vec![1.1, f32::NAN, 2.2, f32::NAN]);
        let b = tf.make_default(vec![2, 2], vec![1.1, f32::NAN, 2.2, f32::NAN]);
        expect_tensors_close_and_equal!(a, b);
    }

    // [spec:et:sem:tensor-util.executorch.runtime.tensors-are-close-fn/test]
    #[test]
    fn tensor_util_test_float_nan_elements_are_not_equal_to_non_nan() {
        let tf = TensorFactory::<f32>::new();
        let a = tf.make_default(vec![2, 2], vec![1.1, f32::NAN, 2.2, f32::NAN]);
        let b = tf.make_default(vec![2, 2], vec![1.1, 0.0, 2.2, 0.0]);
        expect_tensors_not_close_or_equal!(a, b);
    }

    // [spec:et:sem:tensor-util.executorch.runtime.tensors-are-close-fn/test]
    #[test]
    fn tensor_util_test_float_infinite_elements_are_close_and_equal() {
        let k_infinity = f32::INFINITY;
        let tf = TensorFactory::<f32>::new();
        let a = tf.make_default(vec![2, 2], vec![-k_infinity, 1.1, 2.2, k_infinity]);
        let b = tf.make_default(vec![2, 2], vec![-k_infinity, 1.1, 2.2, k_infinity]);
        expect_tensors_close_and_equal!(a, b);
    }

    // [spec:et:sem:tensor-util.executorch.runtime.tensors-are-close-fn/test]
    #[test]
    fn tensor_util_test_nearly_identical_double_tensors_are_close_but_not_equal() {
        let tf = TensorFactory::<f32>::new();
        let a = tf.make_default(vec![2, 2], vec![1.1, 2.2, 4.4, 8.8]);
        let b = tf.make_default(vec![2, 2], vec![1.1f32.next_up(), 2.2, 4.4, 8.8]);
        expect_tensors_close_but_not_equal!(a, b);
    }

    // [spec:et:sem:tensor-util.executorch.runtime.tensors-are-close-fn/test]
    #[test]
    fn tensor_util_test_double_and_infinit_nan_elements_are_close_and_equal() {
        let k_infinity = f64::INFINITY;
        let tf = TensorFactory::<f64>::new();
        let a = tf.make_default(vec![2, 2], vec![-k_infinity, f64::NAN, 1.1, k_infinity]);
        let b = tf.make_default(vec![2, 2], vec![-k_infinity, f64::NAN, 1.1, k_infinity]);
        expect_tensors_close_and_equal!(a, b);
    }

    // [spec:et:sem:tensor-util.executorch.runtime.tensors-are-close-fn/test]
    #[test]
    fn tensor_util_test_tensors_are_close_with_tol() {
        let tf = TensorFactory::<f32>::new();
        let td = TensorFactory::<f64>::new();

        let af = tf.make_default(vec![2, 2], vec![1.0, 2.099999, 0.0, -0.05]);
        let bf = tf.make_default(vec![2, 2], vec![1.099999, 2.0, 0.05, 0.0]);
        assert_tensor_close_with_tol!(af, bf, 0.0, 0.1);

        let ad = td.make_default(vec![2, 2], vec![1.099, 2.199, f64::NAN, -9.0]);
        let bd = td.make_default(vec![2, 2], vec![1.0, 2.0, f64::NAN, -10.0]);
        assert_tensor_close_with_tol!(ad, bd, 0.1, 0.0);
    }

    // [spec:et:sem:tensor-util.executorch.runtime.tensors-are-close-fn/test]
    #[test]
    fn tensor_util_test_tensors_are_not_close_with_tol() {
        let tf = TensorFactory::<f32>::new();
        let td = TensorFactory::<f64>::new();

        let af = tf.make_default(vec![3], vec![1.00, f32::NAN, -10.0]);
        let bf = tf.make_default(vec![3], vec![1.11, f32::NAN, -10.0]);
        assert!(!tensors_are_close(&af, &bf, 0.0, Some(0.1)));

        let ad = td.make_default(vec![3], vec![1.0, 0.0, -10.0]);
        let bd = td.make_default(vec![3], vec![1.0, 0.0, -9.0]);
        assert!(!tensors_are_close(&ad, &bd, 0.1, Some(0.0)));

        let ad = tf.make_default(vec![3], vec![1.0, 2.0, 0.00001]);
        let bd = tf.make_default(vec![3], vec![1.0, 2.0, 0.0]);
        assert!(!tensors_are_close(&ad, &bd, 0.1, Some(0.0)));
    }

    // PORT-NOTE: mirrors the templated `test_data_equal`. C++ fills with
    // `std::rand`; a deterministic incrementing pattern is used here since only
    // the two tensors sharing identical data matters.
    trait TestFill: CppTypeToScalarType + FactoryValue {
        fn fill(i: usize) -> Self;
    }
    macro_rules! impl_test_fill_int {
        ($($t:ty),*) => { $(impl TestFill for $t {
            fn fill(i: usize) -> Self { (i as i64 % 97) as $t }
        })* };
    }
    impl_test_fill_int!(u8, i8, i16, i32, i64);
    impl TestFill for f32 {
        fn fill(i: usize) -> Self {
            (i as f32) * 1.25 - 3.0
        }
    }
    impl TestFill for f64 {
        fn fill(i: usize) -> Self {
            (i as f64) * 1.25 - 3.0
        }
    }
    impl TestFill for bool {
        fn fill(i: usize) -> Self {
            i % 2 == 0
        }
    }

    fn test_data_equal<T: TestFill>(t1_sizes: Vec<i32>, t2_sizes: Vec<i32>) {
        let tf = TensorFactory::<T>::new();
        let numel = size_to_numel(&t1_sizes);
        assert_eq!(numel, size_to_numel(&t2_sizes));

        let data: Vec<T> = (0..numel as usize).map(T::fill).collect();
        let t1 = tf.make_default(t1_sizes, data.clone());
        let t2 = tf.make_default(t2_sizes, data);

        expect_tensors_data_close_and_equal!(t1, t2);
    }

    // [spec:et:sem:tensor-util.executorch.runtime.tensor-data-is-close-fn/test]
    #[test]
    fn tensor_util_test_tensor_data_equal_size_equal() {
        test_data_equal::<u8>(vec![3, 4, 5], vec![3, 4, 5]);
        test_data_equal::<i8>(vec![3, 4, 5], vec![3, 4, 5]);
        test_data_equal::<i16>(vec![3, 4, 5], vec![3, 4, 5]);
        test_data_equal::<i32>(vec![3, 4, 5], vec![3, 4, 5]);
        test_data_equal::<i64>(vec![3, 4, 5], vec![3, 4, 5]);
        test_data_equal::<f32>(vec![3, 4, 5], vec![3, 4, 5]);
        test_data_equal::<f64>(vec![3, 4, 5], vec![3, 4, 5]);
        test_data_equal::<bool>(vec![3, 4, 5], vec![3, 4, 5]);
    }

    // [spec:et:sem:tensor-util.executorch.runtime.tensor-data-is-close-fn/test]
    #[test]
    fn tensor_util_test_tensor_data_equal_size_unequal() {
        test_data_equal::<u8>(vec![3, 4, 5], vec![3, 5, 4]);
        test_data_equal::<i8>(vec![3, 4, 5], vec![3, 5, 4]);
        test_data_equal::<i16>(vec![3, 4, 5], vec![3, 5, 4]);
        test_data_equal::<i32>(vec![3, 4, 5], vec![3, 5, 4]);
        test_data_equal::<i64>(vec![3, 4, 5], vec![3, 5, 4]);
        test_data_equal::<f32>(vec![3, 4, 5], vec![3, 5, 4]);
        test_data_equal::<f64>(vec![3, 4, 5], vec![3, 5, 4]);
        test_data_equal::<bool>(vec![3, 4, 5], vec![3, 5, 4]);
    }

    // [spec:et:sem:tensor-util.executorch.runtime.tensor-data-is-close-fn/test]
    #[test]
    fn tensor_util_test_empty_tensors_supported() {
        test_data_equal::<u8>(vec![3, 4, 0, 5], vec![3, 4, 0, 5]);
        test_data_equal::<i8>(vec![3, 4, 0, 5], vec![3, 4, 0, 5]);
        test_data_equal::<i16>(vec![3, 4, 0, 5], vec![3, 4, 0, 5]);
        test_data_equal::<i32>(vec![3, 4, 0, 5], vec![3, 4, 0, 5]);
        test_data_equal::<i64>(vec![3, 4, 0, 5], vec![3, 4, 0, 5]);
        test_data_equal::<f32>(vec![3, 4, 0, 5], vec![3, 4, 0, 5]);
        test_data_equal::<f64>(vec![3, 4, 0, 5], vec![3, 4, 0, 5]);
        test_data_equal::<bool>(vec![3, 4, 0, 5], vec![3, 4, 0, 5]);
    }

    // [spec:et:sem:tensor-util.executorch.runtime.tensor-data-is-close-fn/test]
    #[test]
    fn tensor_util_test_zero_dim_tensors_supported() {
        test_data_equal::<u8>(vec![], vec![]);
        test_data_equal::<i8>(vec![], vec![]);
        test_data_equal::<i16>(vec![], vec![]);
        test_data_equal::<i32>(vec![], vec![]);
        test_data_equal::<i64>(vec![], vec![]);
        test_data_equal::<f32>(vec![], vec![]);
        test_data_equal::<f64>(vec![], vec![]);
        test_data_equal::<bool>(vec![], vec![]);
    }

    // PORT-NOTE: the C++ `TensorDataCloseNotEqualSizeEqual`/`...Unequal` bodies
    // call `test_data_equal` (not `test_data_close_but_not_equal`), matching the
    // literal source.
    // [spec:et:sem:tensor-util.executorch.runtime.tensor-data-is-close-fn/test]
    #[test]
    fn tensor_util_test_tensor_data_close_not_equal_size_equal() {
        test_data_equal::<f32>(vec![3, 4, 5], vec![3, 4, 5]);
        test_data_equal::<f64>(vec![3, 4, 5], vec![3, 4, 5]);
    }

    // [spec:et:sem:tensor-util.executorch.runtime.tensor-data-is-close-fn/test]
    #[test]
    fn tensor_util_test_tensor_data_close_not_equal_size_unequal() {
        test_data_equal::<f32>(vec![3, 4, 5], vec![3, 5, 4]);
        test_data_equal::<f64>(vec![3, 4, 5], vec![3, 5, 4]);
    }

    fn test_data_equal_but_size_or_dtype_mismatch<T1, T2>(t1_sizes: Vec<i32>, t2_sizes: Vec<i32>)
    where
        T1: CppTypeToScalarType + FactoryValue,
        T2: CppTypeToScalarType + FactoryValue,
    {
        let tf_t1 = TensorFactory::<T1>::new();
        let tf_t2 = TensorFactory::<T2>::new();

        let t1 = tf_t1.zeros_default(t1_sizes);
        let t2 = tf_t2.zeros_default(t2_sizes);

        expect_tensors_data_not_close_or_equal!(t1, t2);
    }

    // [spec:et:sem:tensor-util.executorch.runtime.tensor-data-is-close-fn/test]
    #[test]
    fn tensor_util_test_tensor_data_type_mismatched() {
        let sizes = vec![3, 4, 5, 6];
        test_data_equal_but_size_or_dtype_mismatch::<f32, f64>(sizes.clone(), sizes.clone());
        test_data_equal_but_size_or_dtype_mismatch::<i32, f64>(sizes.clone(), sizes);
    }

    // [spec:et:sem:tensor-util.executorch.runtime.tensor-data-is-close-fn/test]
    #[test]
    fn tensor_util_test_tensor_size_mismatched() {
        let sizes_t1 = vec![3, 4, 5, 6];
        let sizes_t2 = vec![3, 4, 5, 7];
        test_data_equal_but_size_or_dtype_mismatch::<f32, f32>(sizes_t1, sizes_t2);
    }

    // [spec:et:sem:tensor-util.executorch.runtime.tensor-data-is-close-fn/test]
    #[test]
    fn tensor_util_test_tensor_data_mismatched() {
        let tf = TensorFactory::<i32>::new();
        let t1 = tf.make_default(vec![3, 2], vec![1, 2, 3, 4, 5, 6]);
        let t2 = tf.make_default(vec![3, 2], vec![1, 2, 3, 1, 5, 6]);
        let t3 = tf.make_default(vec![2, 3], vec![1, 2, 3, 1, 5, 6]);
        expect_tensors_data_not_close_or_equal!(t1, t2);
        expect_tensors_data_not_close_or_equal!(t1, t3);

        let t_zero_dim = tf.make_default(vec![], vec![0]);
        let t_empty = tf.make_default(vec![0], vec![]);
        expect_tensors_data_not_close_or_equal!(t_zero_dim, t_empty);
    }

    // [spec:et:sem:tensor-util.executorch.runtime.tensor-data-is-close-fn/test]
    #[test]
    fn tensor_util_test_tensor_data_close_with_tol() {
        let tf = TensorFactory::<f32>::new();
        let td = TensorFactory::<f64>::new();

        let af = tf.make_default(vec![4, 1], vec![1.0, 2.099, 0.0, -0.05]);
        let bf = tf.make_default(vec![2, 2], vec![1.099, 2.0, 0.05, 0.0]);
        assert!(tensor_data_is_close(&af, &bf, 0.0, Some(0.1)));

        let ad = td.make_default(vec![2, 2], vec![1.099, 2.199, f64::NAN, -9.0]);
        let bd = td.make_default(vec![4], vec![1.0, 2.0, f64::NAN, -10.0]);
        assert!(tensor_data_is_close(&ad, &bd, 0.1, Some(0.0)));
    }

    // [spec:et:sem:tensor-util.executorch.runtime.tensor-data-is-close-fn/test]
    #[test]
    fn tensor_util_test_tensor_data_not_close_with_tol() {
        let tf = TensorFactory::<f32>::new();
        let td = TensorFactory::<f64>::new();

        let af = tf.make_default(vec![3], vec![1.00, 0.0, -10.0]);
        let bf = tf.make_default(vec![3, 1], vec![1.11, 0.0, -10.0]);
        assert!(!tensor_data_is_close(&af, &bf, 0.0, Some(0.1)));

        let ad = td.make_default(vec![2, 2], vec![1.0, 0.0, -10.0, 0.0]);
        let bd = td.make_default(vec![4], vec![1.0, 0.0, -9.0, 0.0]);
        assert!(!tensor_data_is_close(&ad, &bd, 0.1, Some(0.0)));

        let ad = tf.make_default(vec![1, 4], vec![1.0, 2.0, f32::NAN, 0.00001]);
        let bd = tf.make_default(vec![2, 2], vec![1.0, 2.0, f32::NAN, 0.0]);
        assert!(!tensor_data_is_close(&ad, &bd, 0.1, Some(0.0)));
    }

    // [spec:et:sem:tensor-util.executorch.runtime.tensor-lists-are-close-fn/test]
    #[test]
    fn tensor_util_test_tensor_lists_close_and_equal() {
        let tf_int = TensorFactory::<i32>::new();
        let tf_float = TensorFactory::<f32>::new();

        // PORT-NOTE: the C++ demonstrates comparing `TensorList` vs `std::vector`
        // interchangeably; the Rust `tensor_lists_are_close` takes `&[Tensor]`, so
        // both spellings collapse to slices of the same `Vec`.
        let list1 = vec![
            tf_int.zeros_default(vec![1, 2]),
            tf_float.ones_default(vec![2, 1]),
        ];
        let list2 = vec![
            tf_int.zeros_default(vec![1, 2]),
            tf_float.ones_default(vec![2, 1]),
        ];

        expect_tensor_lists_close_and_equal!(list1, list2);
    }

    // [spec:et:sem:tensor-util.executorch.runtime.tensor-lists-are-close-fn/test]
    #[test]
    fn tensor_util_test_empty_tensor_lists_are_close_and_equal() {
        let list1: Vec<Tensor> = vec![];
        assert_eq!(list1.len(), 0);
        let list2: Vec<Tensor> = vec![];
        assert_eq!(list2.len(), 0);
        expect_tensor_lists_close_and_equal!(list1, list2);
    }

    // [spec:et:sem:tensor-util.executorch.runtime.tensor-lists-are-close-fn/test]
    #[test]
    fn tensor_util_test_tensor_lists_close_but_not_equal() {
        let tf_int = TensorFactory::<i32>::new();
        let tf_float = TensorFactory::<f32>::new();

        let list1 = vec![
            tf_int.zeros_default(vec![1, 2]),
            tf_float.ones_default(vec![2, 1]),
        ];
        let list2 = vec![
            tf_int.zeros_default(vec![1, 2]),
            tf_float.ones_default(vec![2, 1]),
        ];

        // Tweak a float value slightly.
        unsafe {
            *list1[1].mutable_data_ptr::<f32>() = 1.0f32.next_up();
        }

        expect_tensor_lists_close_but_not_equal!(list1, list2);
    }

    // [spec:et:sem:tensor-util.executorch.runtime.tensor-lists-are-close-fn/test]
    #[test]
    fn tensor_util_test_tensor_lists_with_different_data_are_not_close_or_equal() {
        let tf_int = TensorFactory::<i32>::new();
        let tf_float = TensorFactory::<f32>::new();

        let list1 = vec![
            tf_int.zeros_default(vec![1, 2]),
            tf_float.ones_default(vec![2, 1]),
        ];
        let list2 = vec![
            tf_int.zeros_default(vec![1, 2]),
            tf_float.zeros_default(vec![2, 1]), // vs. ones() in the first list.
        ];

        expect_tensor_lists_not_close_or_equal!(list1, list2);
    }

    // [spec:et:sem:tensor-util.executorch.runtime.tensor-lists-are-close-fn/test]
    #[test]
    fn tensor_util_test_tensor_lists_with_different_lengths_are_not_close_or_equal() {
        let tf_int = TensorFactory::<i32>::new();
        let tf_float = TensorFactory::<f32>::new();

        let list1 = vec![
            tf_int.zeros_default(vec![1, 2]),
            tf_float.ones_default(vec![2, 1]),
        ];
        let list2 = vec![
            tf_int.zeros_default(vec![1, 2]),
            // Missing second element.
        ];

        expect_tensor_lists_not_close_or_equal!(list1, list2);
    }

    // PORT-NOTE: the C++ `operator<<` stream-formatting tests
    // (`ScalarTypeStreamSmokeTest`, `TensorStreamInt`, `TensorStreamDouble`,
    // `TensorStreamBool`) exercise the `!USE_ATEN_LIB` `operator<<(ScalarType)` /
    // `operator<<(Tensor)` overloads that produce the `ETensor(sizes=..., dtype=,
    // data=...)` string. Those stream operators are not part of this port (there
    // is no `Display`/stream surface for `Tensor`/`ScalarType` in the ported
    // tensor_util), so these four tests are not portable as written and are
    // recorded here rather than ported.

    // PORT-NOTE: `TestZeroShapeTensorEquality` builds `TensorImpl`s with a null
    // data pointer and later `set_data`s. The `EXPECT_TENSOR_EQ` over null-data
    // tensors is a death test (`data_is_close` aborts on null pointers with
    // numel > 0); split into an `#[ignore]`d `#[should_panic]` below and a live
    // equality test after `set_data`.
    // [spec:et:sem:tensor-util.executorch.runtime.tensors-are-close-fn/test]
    #[test]
    #[should_panic]
    #[ignore]
    fn tensor_test_test_zero_shape_tensor_equality_death() {
        let mut sizes: [SizesType; 2] = [2, 2];
        let mut strides: [StridesType; 2] = [2, 1];
        let mut dim_order: [DimOrderType; 2] = [0, 1];

        let mut t1 = TensorImpl::new(
            ScalarType::Float,
            2,
            sizes.as_mut_ptr(),
            core::ptr::null_mut(),
            dim_order.as_mut_ptr(),
            strides.as_mut_ptr(),
            TensorShapeDynamism::STATIC,
            DeviceType::CPU,
            0 as DeviceIndex,
        );
        let mut t2 = TensorImpl::new(
            ScalarType::Float,
            2,
            sizes.as_mut_ptr(),
            core::ptr::null_mut(),
            dim_order.as_mut_ptr(),
            strides.as_mut_ptr(),
            TensorShapeDynamism::STATIC,
            DeviceType::CPU,
            0 as DeviceIndex,
        );
        let a = Tensor::new(&mut t1 as *mut TensorImpl);
        let b = Tensor::new(&mut t2 as *mut TensorImpl);
        assert_tensor_eq!(a, b);
    }

    // [spec:et:sem:tensor-util.executorch.runtime.tensors-are-close-fn/test]
    #[test]
    fn tensor_test_test_zero_shape_tensor_equality() {
        let mut sizes: [SizesType; 2] = [2, 2];
        let mut strides: [StridesType; 2] = [2, 1];
        let mut dim_order: [DimOrderType; 2] = [0, 1];

        let mut t1 = TensorImpl::new(
            ScalarType::Float,
            2,
            sizes.as_mut_ptr(),
            core::ptr::null_mut(),
            dim_order.as_mut_ptr(),
            strides.as_mut_ptr(),
            TensorShapeDynamism::STATIC,
            DeviceType::CPU,
            0 as DeviceIndex,
        );
        let mut t2 = TensorImpl::new(
            ScalarType::Float,
            2,
            sizes.as_mut_ptr(),
            core::ptr::null_mut(),
            dim_order.as_mut_ptr(),
            strides.as_mut_ptr(),
            TensorShapeDynamism::STATIC,
            DeviceType::CPU,
            0 as DeviceIndex,
        );

        let mut data: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
        t1.set_data(data.as_mut_ptr() as *mut core::ffi::c_void);
        t2.set_data(data.as_mut_ptr() as *mut core::ffi::c_void);

        let a = Tensor::new(&mut t1 as *mut TensorImpl);
        let b = Tensor::new(&mut t2 as *mut TensorImpl);
        assert_tensor_eq!(a, b);
    }
}
