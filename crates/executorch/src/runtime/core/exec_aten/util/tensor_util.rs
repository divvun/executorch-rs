//! Literal port of runtime/core/exec_aten/util/tensor_util.h.

use crate::runtime::core::array_ref::ArrayRef;
use crate::runtime::core::error::Error;
use crate::runtime::core::exec_aten::util::scalar_type_util::{
    can_cast, is_bits_type, is_complex_type, is_floating_type, is_integral_type, is_real_h_type,
    is_real_hb_type, is_real_hbbf16_type, is_real_hbf16_type, is_real_type, to_string,
};
use crate::runtime::core::portable_type::scalar_type::ScalarType;
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::core::portable_type::tensor_impl::{
    DimOrderType, SizesType, StridesType, ssize_t,
};

/// The maximum number of dimensions a tensor may have.
// PORT-NOTE: `kTensorDimensionLimit` is defined in
// runtime/core/exec_aten/util/tensor_dimension_limit.h, which has no ported
// `tensor_dimension_limit.rs` target yet. Inlined here with the same value (16)
// until that module lands, matching dim_order_util.rs. Unresolved cross-module
// reference.
pub const K_TENSOR_DIMENSION_LIMIT: usize = 16;

// PORT-NOTE: `et_check_msg!` / `et_check!` mirror the C++ fatal `ET_CHECK_MSG` /
// `ET_CHECK`. There is no globally-exported version in the runtime port; each
// module defines its own local `macro_rules!` (see scalar_type_util.rs,
// tensor_impl.rs). Message formatting is dropped since a fatal abort follows,
// matching the established local definitions.
macro_rules! et_check_msg {
    ($cond:expr, $($arg:tt)*) => {
        if !($cond) {
            crate::runtime::platform::abort::runtime_abort();
        }
    };
}

macro_rules! et_check {
    ($cond:expr) => {
        if !($cond) {
            crate::runtime::platform::abort::runtime_abort();
        }
    };
}

// PORT-NOTE: the crate-level `et_check_or_return_false!`
// (runtime/core/error.rs) drops all caller-supplied format arguments after the
// leading literal — it only prepends the stringified condition and passes no
// further args, so any message with `{}` placeholders fails to compile
// ("N positional arguments in format string, but there is 1 argument"). That
// macro is a sibling module with no other callers; this local
// `et_check_or_return_false!` mirrors the C++ `ET_CHECK_OR_RETURN_FALSE`
// faithfully (prepend "Check failed (cond): " then forward the full message +
// args) so this module's ports stay literal. Unresolved cross-module reference.
macro_rules! et_check_or_return_false {
    ($cond:expr, $fmt:literal $(, $($arg:tt)*)?) => {{
        if !($cond) {
            $crate::et_log!(
                Error,
                ::core::concat!("Check failed ({}): ", $fmt),
                ::core::stringify!($cond)
                $(, $($arg)*)?
            );
            return false;
        }
    }};
}

/// All assertion messages should begin with this prefix.
// #define ET_TENSOR_CHECK_PREFIX__ "Tensors do not match"
pub const ET_TENSOR_CHECK_PREFIX__: &str = "Tensors do not match";

// PORT-NOTE: `ET_LOG_AND_RETURN_IF_FALSE(cond)` expands to
// `ET_CHECK_OR_RETURN_FALSE(cond, "")` in the C++ header. Ported as a local
// macro over the local `et_check_or_return_false!` above so
// `tensor_has_non_empty_dim` can use it verbatim.
macro_rules! et_log_and_return_if_false {
    ($cond:expr) => {
        et_check_or_return_false!($cond, "")
    };
}

// #define ET_MIN2(a, b) (std::min(a, b))
macro_rules! et_min2 {
    ($a:expr, $b:expr) => {
        core::cmp::min($a, $b)
    };
}

// #define ET_MIN3(a, b, c) (std::min(a, std::min(b, c)))
macro_rules! et_min3 {
    ($a:expr, $b:expr, $c:expr) => {
        core::cmp::min($a, core::cmp::min($b, $c))
    };
}

// #define ET_NORMALIZE_IX(IX, UPPER_BOUND) IX < 0 ? IX + UPPER_BOUND : IX
macro_rules! et_normalize_ix {
    ($ix:expr, $upper_bound:expr) => {
        if $ix < 0 { $ix + $upper_bound } else { $ix }
    };
}

// PORT-NOTE (wave-3 tensor_util tests): The `ET_CHECK_SAME_SHAPE2/3`,
// `ET_CHECK_SAME_DTYPE2/3`, `ET_CHECK_SAME_SHAPE_AND_DTYPE2/3`,
// `ET_CHECK_CONTIGUOUS`, and `ET_CHECK_SAME_STRIDES2/3` fatal assertion macros
// from tensor_util.h had no ported target (kernels use the `tensors_have_*`
// predicate functions directly). They are ported here as module-local
// `macro_rules!` mirroring the C++ macro bodies verbatim (on failure, abort via
// the local `et_check_msg!` == `ET_CHECK_MSG`), so `tensor_util_test.cpp` /
// `operator_impl_example_test.cpp` can port literally. Message formatting is
// dropped consistent with the local `et_check_msg!`.

// ET_CHECK_SAME_SHAPE2(a__, b__)
#[cfg(test)]
macro_rules! et_check_same_shape2 {
    ($a:expr, $b:expr) => {{
        let a_numel__: usize = $a.numel() as usize;
        let b_numel__: usize = $b.numel() as usize;
        let a_dim__: usize = $a.dim() as usize;
        let b_dim__: usize = $b.dim() as usize;
        et_check_msg!(
            a_numel__ == b_numel__ && ((a_numel__ == 1 && b_numel__ == 1) || (a_dim__ == b_dim__)),
            ""
        );
        let mut dim__: usize = 0;
        while dim__ < et_min2!(a_dim__, b_dim__) {
            let a_size__: usize = $a.size(dim__ as ssize_t) as usize;
            let b_size__: usize = $b.size(dim__ as ssize_t) as usize;
            et_check_msg!(a_size__ == b_size__, "");
            dim__ += 1;
        }
    }};
}

// ET_CHECK_SAME_SHAPE3(a__, b__, c__)
#[cfg(test)]
macro_rules! et_check_same_shape3 {
    ($a:expr, $b:expr, $c:expr) => {{
        let a_numel__: usize = $a.numel() as usize;
        let b_numel__: usize = $b.numel() as usize;
        let c_numel__: usize = $c.numel() as usize;
        let a_dim__: usize = $a.dim() as usize;
        let b_dim__: usize = $b.dim() as usize;
        let c_dim__: usize = $c.dim() as usize;
        et_check_msg!(
            a_numel__ == b_numel__
                && b_numel__ == c_numel__
                && ((a_numel__ == 1 && b_numel__ == 1 && c_numel__ == 1)
                    || (a_dim__ == b_dim__ && b_dim__ == c_dim__)),
            ""
        );
        let mut dim__: usize = 0;
        while dim__ < et_min3!(a_dim__, b_dim__, c_dim__) {
            let a_size__: usize = $a.size(dim__ as ssize_t) as usize;
            let b_size__: usize = $b.size(dim__ as ssize_t) as usize;
            let c_size__: usize = $c.size(dim__ as ssize_t) as usize;
            et_check_msg!(a_size__ == b_size__ && b_size__ == c_size__, "");
            dim__ += 1;
        }
    }};
}

// ET_CHECK_SAME_DTYPE2(a__, b__)
#[cfg(test)]
macro_rules! et_check_same_dtype2 {
    ($a:expr, $b:expr) => {{
        let a_type__ = $a.scalar_type();
        let b_type__ = $b.scalar_type();
        et_check_msg!(a_type__ == b_type__, "");
    }};
}

// ET_CHECK_SAME_DTYPE3(a__, b__, c__)
#[cfg(test)]
macro_rules! et_check_same_dtype3 {
    ($a:expr, $b:expr, $c:expr) => {{
        let a_type__ = $a.scalar_type();
        let b_type__ = $b.scalar_type();
        let c_type__ = $c.scalar_type();
        et_check_msg!(a_type__ == b_type__ && b_type__ == c_type__, "");
    }};
}

// ET_CHECK_SAME_SHAPE_AND_DTYPE2(a__, b__)
#[cfg(test)]
macro_rules! et_check_same_shape_and_dtype2 {
    ($a:expr, $b:expr) => {{
        let a_numel__: usize = $a.numel() as usize;
        let b_numel__: usize = $b.numel() as usize;
        let a_dim__: usize = $a.dim() as usize;
        let b_dim__: usize = $b.dim() as usize;
        let a_type__ = $a.scalar_type();
        let b_type__ = $b.scalar_type();
        et_check_msg!(
            a_numel__ == b_numel__
                && ((a_numel__ == 1 && b_numel__ == 1) || a_dim__ == b_dim__)
                && a_type__ == b_type__,
            ""
        );
        let mut dim__: usize = 0;
        while dim__ < et_min2!(a_dim__, b_dim__) {
            let a_size__: usize = $a.size(dim__ as ssize_t) as usize;
            let b_size__: usize = $b.size(dim__ as ssize_t) as usize;
            et_check_msg!(a_size__ == b_size__, "");
            dim__ += 1;
        }
    }};
}

// ET_CHECK_SAME_SHAPE_AND_DTYPE3(a__, b__, c__)
#[cfg(test)]
macro_rules! et_check_same_shape_and_dtype3 {
    ($a:expr, $b:expr, $c:expr) => {{
        let a_numel__: usize = $a.numel() as usize;
        let b_numel__: usize = $b.numel() as usize;
        let c_numel__: usize = $c.numel() as usize;
        let a_dim__: usize = $a.dim() as usize;
        let b_dim__: usize = $b.dim() as usize;
        let c_dim__: usize = $c.dim() as usize;
        let a_type__ = $a.scalar_type();
        let b_type__ = $b.scalar_type();
        let c_type__ = $c.scalar_type();
        et_check_msg!(
            a_numel__ == b_numel__
                && b_numel__ == c_numel__
                && ((a_numel__ == 1 && b_numel__ == 1 && c_numel__ == 1)
                    || (a_dim__ == b_dim__ && b_dim__ == c_dim__))
                && a_type__ == b_type__
                && b_type__ == c_type__,
            ""
        );
        let mut dim__: usize = 0;
        while dim__ < et_min3!(a_dim__, b_dim__, c_dim__) {
            let a_size__: usize = $a.size(dim__ as ssize_t) as usize;
            let b_size__: usize = $b.size(dim__ as ssize_t) as usize;
            let c_size__: usize = $c.size(dim__ as ssize_t) as usize;
            et_check_msg!(a_size__ == b_size__ && b_size__ == c_size__, "");
            dim__ += 1;
        }
    }};
}

// ET_CHECK_CONTIGUOUS(a__)
#[cfg(test)]
macro_rules! et_check_contiguous {
    ($a:expr) => {{
        let strides = $a.strides();
        let sizes = $a.sizes();
        et_check_msg!(*strides.at(strides.size() - 1) == 1, "");
        let mut i: usize = strides.size() - 1;
        while i > 0 {
            et_check_msg!(*strides.at(i - 1) == *strides.at(i) * *sizes.at(i), "");
            i -= 1;
        }
    }};
}

// ET_CHECK_SAME_STRIDES2(a__, b__)
#[cfg(test)]
macro_rules! et_check_same_strides2 {
    ($a:expr, $b:expr) => {{
        et_check_msg!($a.dim() == $b.dim(), "");
        let a_strides = $a.strides();
        let b_strides = $b.strides();
        let mut i: ssize_t = 0;
        while i < $a.dim() {
            et_check_msg!(*a_strides.at(i as usize) == *b_strides.at(i as usize), "");
            i += 1;
        }
    }};
}

// ET_CHECK_SAME_STRIDES3(a__, b__, c__)
#[cfg(test)]
macro_rules! et_check_same_strides3 {
    ($a:expr, $b:expr, $c:expr) => {{
        et_check_msg!($a.dim() == $b.dim() && $b.dim() == $c.dim(), "");
        let a_strides = $a.strides();
        let b_strides = $b.strides();
        let c_strides = $c.strides();
        let mut i: ssize_t = 0;
        while i < $a.dim() {
            et_check_msg!(
                *a_strides.at(i as usize) == *b_strides.at(i as usize)
                    && *b_strides.at(i as usize) == *c_strides.at(i as usize),
                ""
            );
            i += 1;
        }
    }};
}

// PORT-NOTE: `c10::mul_overflows(a, b, &out)` returns true on overflow and
// writes the wrapped product to `out`. Ported inline here (as in
// tensor_impl.rs) via `usize::checked_mul` which yields `None` on overflow.
fn mul_overflows(a: usize, b: usize, out: &mut usize) -> bool {
    match a.checked_mul(b) {
        Some(product) => {
            *out = product;
            false
        }
        None => {
            *out = a.wrapping_mul(b);
            true
        }
    }
}

//
// Utility functions for checking tensor attributes
//

/*
 * Returns true if the given dimension value is between -upper_bound and
 * upper_bound - 1, inclusive.
 */
// [spec:et:def:tensor-util.executorch.et-runtime-namespace.dim-is-valid-fn]
// [spec:et:sem:tensor-util.executorch.et-runtime-namespace.dim-is-valid-fn]
pub fn dim_is_valid(dim: i64, upper_bound: i64) -> bool {
    et_check_or_return_false!(
        dim >= -upper_bound && dim < upper_bound,
        "Dimension {} is out of range. Dimension should be between {} and {}, inclusive.",
        dim,
        -upper_bound,
        upper_bound - 1
    );

    true
}

/*
 * Returns the tensor's number of dimensions, except when the tensor is zero
 * dimensional. In this case, it returns 1.
 */
// [spec:et:def:tensor-util.executorch.et-runtime-namespace.nonzero-dim-fn]
// [spec:et:sem:tensor-util.executorch.et-runtime-namespace.nonzero-dim-fn]
pub fn nonzero_dim(tensor: &Tensor) -> ssize_t {
    if tensor.dim() == 0 { 1 } else { tensor.dim() }
}

/*
 * Returns the size along a dimension dim, except when the tensor is zero
 * dimensional. In this case, it returns 1.
 */
// [spec:et:def:tensor-util.executorch.et-runtime-namespace.nonempty-size-fn]
// [spec:et:sem:tensor-util.executorch.et-runtime-namespace.nonempty-size-fn]
pub fn nonempty_size(tensor: &Tensor, dim: ssize_t) -> ssize_t {
    if tensor.dim() == 0 {
        1
    } else {
        tensor.size(dim)
    }
}

// [spec:et:def:tensor-util.executorch.et-runtime-namespace.tensor-can-cast-to-fn]
// [spec:et:sem:tensor-util.executorch.et-runtime-namespace.tensor-can-cast-to-fn]
pub fn tensor_can_cast_to(a: &Tensor, dtype: ScalarType) -> bool {
    et_check_or_return_false!(
        can_cast(a.scalar_type(), dtype),
        "Tensor of dtype {} cannot cast to dtype {}",
        to_string(a.scalar_type()),
        to_string(dtype)
    );

    true
}

// [spec:et:def:tensor-util.executorch.et-runtime-namespace.tensor-is-bool-type-fn]
// [spec:et:sem:tensor-util.executorch.et-runtime-namespace.tensor-is-bool-type-fn]
pub fn tensor_is_bool_type(t: &Tensor) -> bool {
    et_check_or_return_false!(
        t.scalar_type() == ScalarType::Bool,
        "Expected to find bool type, but tensor has type {}",
        to_string(t.scalar_type())
    );

    true
}

pub fn tensor_is_type1(t: &Tensor, dtype: ScalarType) -> bool {
    et_check_or_return_false!(
        t.scalar_type() == dtype,
        "Expected to find {} type, but tensor has type {}",
        to_string(dtype),
        to_string(t.scalar_type())
    );

    true
}

pub fn tensor_is_type2(t: &Tensor, dtype: ScalarType, dtype2: ScalarType) -> bool {
    et_check_or_return_false!(
        t.scalar_type() == dtype || t.scalar_type() == dtype2,
        "Expected to find {} or {} type, but tensor has type {}",
        to_string(dtype),
        to_string(dtype2),
        to_string(t.scalar_type())
    );

    true
}

// [spec:et:def:tensor-util.executorch.et-runtime-namespace.tensor-is-type-fn]
// [spec:et:sem:tensor-util.executorch.et-runtime-namespace.tensor-is-type-fn]
pub fn tensor_is_type(
    t: &Tensor,
    dtype: ScalarType,
    dtype2: ScalarType,
    dtype3: ScalarType,
) -> bool {
    et_check_or_return_false!(
        t.scalar_type() == dtype || t.scalar_type() == dtype2 || t.scalar_type() == dtype3,
        "Expected to find {}, {}, or {} type, but tensor has type {}",
        to_string(dtype),
        to_string(dtype2),
        to_string(dtype3),
        to_string(t.scalar_type())
    );

    true
}

// [spec:et:def:tensor-util.executorch.et-runtime-namespace.tensor-is-integral-type-fn]
// [spec:et:sem:tensor-util.executorch.et-runtime-namespace.tensor-is-integral-type-fn]
pub fn tensor_is_integral_type(t: &Tensor, include_bool: bool) -> bool {
    et_check_or_return_false!(
        is_integral_type(t.scalar_type(), include_bool),
        "Expected to find a integral type, but tensor has type {}",
        to_string(t.scalar_type())
    );

    true
}

// [spec:et:def:tensor-util.executorch.et-runtime-namespace.tensor-is-floating-type-fn]
// [spec:et:sem:tensor-util.executorch.et-runtime-namespace.tensor-is-floating-type-fn]
pub fn tensor_is_floating_type(t: &Tensor) -> bool {
    et_check_or_return_false!(
        is_floating_type(t.scalar_type()),
        "Expected to find a floating type, but tensor has type {}",
        to_string(t.scalar_type())
    );

    true
}

// [spec:et:def:tensor-util.executorch.et-runtime-namespace.tensor-is-real-type-fn]
// [spec:et:sem:tensor-util.executorch.et-runtime-namespace.tensor-is-real-type-fn]
pub fn tensor_is_real_type(t: &Tensor) -> bool {
    et_check_or_return_false!(
        is_real_type(t.scalar_type()),
        "Expected to find a real type, but tensor has type {}",
        to_string(t.scalar_type())
    );

    true
}

// [spec:et:def:tensor-util.executorch.et-runtime-namespace.tensor-is-realh-type-fn]
// [spec:et:sem:tensor-util.executorch.et-runtime-namespace.tensor-is-realh-type-fn]
pub fn tensor_is_realh_type(t: &Tensor) -> bool {
    et_check_or_return_false!(
        is_real_h_type(t.scalar_type()),
        "Expected to find a real type, but tensor has type {}",
        to_string(t.scalar_type())
    );

    true
}

// [spec:et:def:tensor-util.executorch.et-runtime-namespace.tensor-is-realhbf16-type-fn]
// [spec:et:sem:tensor-util.executorch.et-runtime-namespace.tensor-is-realhbf16-type-fn]
pub fn tensor_is_realhbf16_type(t: &Tensor) -> bool {
    et_check_or_return_false!(
        is_real_hbf16_type(t.scalar_type()),
        "Expected to find a real type, but tensor has type {}",
        to_string(t.scalar_type())
    );

    true
}

// [spec:et:def:tensor-util.executorch.et-runtime-namespace.tensor-is-realhb-type-fn]
// [spec:et:sem:tensor-util.executorch.et-runtime-namespace.tensor-is-realhb-type-fn]
pub fn tensor_is_realhb_type(t: &Tensor) -> bool {
    et_check_or_return_false!(
        is_real_hb_type(t.scalar_type()),
        "Expected to find a real type, but tensor has type {}",
        to_string(t.scalar_type())
    );

    true
}

// [spec:et:def:tensor-util.executorch.et-runtime-namespace.tensor-is-realhbbf16-type-fn]
// [spec:et:sem:tensor-util.executorch.et-runtime-namespace.tensor-is-realhbbf16-type-fn]
pub fn tensor_is_realhbbf16_type(t: &Tensor) -> bool {
    et_check_or_return_false!(
        is_real_hbbf16_type(t.scalar_type()),
        "Expected to find a real type, but tensor has type {}",
        to_string(t.scalar_type())
    );

    true
}

// [spec:et:def:tensor-util.executorch.et-runtime-namespace.tensor-is-complex-type-fn]
// [spec:et:sem:tensor-util.executorch.et-runtime-namespace.tensor-is-complex-type-fn]
pub fn tensor_is_complex_type(t: &Tensor) -> bool {
    et_check_or_return_false!(
        is_complex_type(t.scalar_type()),
        "Expected to find a complex type, but tensor has type {}",
        to_string(t.scalar_type())
    );

    true
}

// [spec:et:def:tensor-util.executorch.et-runtime-namespace.tensor-is-bits-type-fn]
// [spec:et:sem:tensor-util.executorch.et-runtime-namespace.tensor-is-bits-type-fn]
pub fn tensor_is_bits_type(t: &Tensor) -> bool {
    et_check_or_return_false!(
        is_bits_type(t.scalar_type()),
        "Expected to find a bits type, but tensor has type {}",
        to_string(t.scalar_type())
    );

    true
}

pub fn tensors_have_same_dtype2(a: &Tensor, b: &Tensor) -> bool {
    et_check_or_return_false!(
        a.scalar_type() == b.scalar_type(),
        "{}: dtype={{{}, {}}}",
        ET_TENSOR_CHECK_PREFIX__,
        to_string(a.scalar_type()),
        to_string(b.scalar_type())
    );
    true
}

// [spec:et:def:tensor-util.executorch.et-runtime-namespace.tensors-have-same-dtype-fn]
// [spec:et:sem:tensor-util.executorch.et-runtime-namespace.tensors-have-same-dtype-fn]
pub fn tensors_have_same_dtype(a: &Tensor, b: &Tensor, c: &Tensor) -> bool {
    et_check_or_return_false!(
        a.scalar_type() == b.scalar_type() && b.scalar_type() == c.scalar_type(),
        "{}: dtype={{{}, {}, {}}}",
        ET_TENSOR_CHECK_PREFIX__,
        to_string(a.scalar_type()),
        to_string(b.scalar_type()),
        to_string(c.scalar_type())
    );
    true
}

// [spec:et:def:tensor-util.executorch.et-runtime-namespace.tensor-is-rank-fn]
// [spec:et:sem:tensor-util.executorch.et-runtime-namespace.tensor-is-rank-fn]
pub fn tensor_is_rank(t: &Tensor, rank: usize) -> bool {
    et_check_or_return_false!(
        t.dim() as usize == rank,
        "Expected tensor.dim() to be {}, but got {}",
        rank,
        t.dim() as usize
    );

    true
}

// [spec:et:def:tensor-util.executorch.et-runtime-namespace.tensor-has-rank-greater-or-equal-to-fn]
// [spec:et:sem:tensor-util.executorch.et-runtime-namespace.tensor-has-rank-greater-or-equal-to-fn]
pub fn tensor_has_rank_greater_or_equal_to(t: &Tensor, rank: usize) -> bool {
    et_check_or_return_false!(
        t.dim() as usize >= rank,
        "Expected tensor.dim() to be >= {}, but got {}",
        rank,
        t.dim() as usize
    );

    true
}

// [spec:et:def:tensor-util.executorch.et-runtime-namespace.tensor-has-rank-smaller-or-equal-to-fn]
// [spec:et:sem:tensor-util.executorch.et-runtime-namespace.tensor-has-rank-smaller-or-equal-to-fn]
pub fn tensor_has_rank_smaller_or_equal_to(t: &Tensor, rank: usize) -> bool {
    et_check_or_return_false!(
        t.dim() as usize <= rank,
        "Expected tensor.dim() to be <= {}, but got {}",
        rank,
        t.dim() as usize
    );

    true
}

// [spec:et:def:tensor-util.executorch.et-runtime-namespace.tensor-has-dim-fn]
// [spec:et:sem:tensor-util.executorch.et-runtime-namespace.tensor-has-dim-fn]
pub fn tensor_has_dim(t: &Tensor, d: i64) -> bool {
    if t.dim() == 0 {
        et_check_or_return_false!(
            d == 0 || d == -1,
            "dim must be 0 or -1 for 0-dim tensor, got {}",
            d
        );
    } else {
        et_check_or_return_false!(
            if d > 0 {
                d < t.dim() as i64
            } else {
                t.dim() as i64 + d >= 0
            },
            "{}-dim tensor does not have dim at index {}",
            t.dim() as usize,
            d as usize
        );
    }
    true
}

// [spec:et:def:tensor-util.executorch.et-runtime-namespace.tensor-has-non-empty-dim-fn]
// [spec:et:sem:tensor-util.executorch.et-runtime-namespace.tensor-has-non-empty-dim-fn]
pub fn tensor_has_non_empty_dim(t: &Tensor, d: i64) -> bool {
    let udim: usize = (et_normalize_ix!(d, t.dim() as i64)) as usize;
    et_log_and_return_if_false!(tensor_has_dim(t, d));
    et_log_and_return_if_false!(t.size(udim as ssize_t) != 0);
    true
}

// [spec:et:def:tensor-util.executorch.et-runtime-namespace.tensor-dim-has-index-fn]
// [spec:et:sem:tensor-util.executorch.et-runtime-namespace.tensor-dim-has-index-fn]
pub fn tensor_dim_has_index(t: &Tensor, mut d: i64, ix: i64) -> bool {
    // Indexing ops don't support zero-dim tensors
    et_check!(t.dim() != 0);
    if d < 0 {
        d += t.dim() as i64;
    }
    // Dimension must have been already checked by tensor_has_dim
    et_check!(d >= 0 && d < t.dim() as i64);

    et_check_or_return_false!(
        ix >= -(t.size(d as ssize_t) as i64) && ix < t.size(d as ssize_t) as i64,
        "index {} out of range [-{},{}) at dimension {})",
        ix,
        t.size(d as ssize_t) as usize,
        t.size(d as ssize_t) as usize,
        d
    );
    true
}

// [spec:et:def:tensor-util.executorch.et-runtime-namespace.tensors-have-same-size-at-dims-fn]
// [spec:et:sem:tensor-util.executorch.et-runtime-namespace.tensors-have-same-size-at-dims-fn]
pub fn tensors_have_same_size_at_dims(a: &Tensor, dim_a: usize, b: &Tensor, dim_b: usize) -> bool {
    et_check_or_return_false!(
        dim_a < a.dim() as usize,
        "Cannot retrieve dim {} from tensor with dim {}",
        dim_a,
        a.dim() as usize
    );
    et_check_or_return_false!(
        dim_b < b.dim() as usize,
        "Cannot retrieve dim {} from tensor with dim {}",
        dim_b,
        b.dim() as usize
    );
    et_check_or_return_false!(
        a.size(dim_a as ssize_t) == b.size(dim_b as ssize_t),
        "{}: a.size({}) = {} does not match b.size({}) = {}",
        ET_TENSOR_CHECK_PREFIX__,
        dim_a,
        a.size(dim_a as ssize_t) as usize,
        dim_b,
        b.size(dim_b as ssize_t) as usize
    );

    true
}

pub fn tensors_have_same_shape2(a: &Tensor, b: &Tensor) -> bool {
    if a.numel() == 1 && b.numel() == 1 {
        // PyTorch operators treat all scalar tensors as the same shape even if
        // they have different dims.
        return true;
    }
    if !(a.sizes().equals(b.sizes()) && a.numel() == b.numel()) {
        crate::et_log!(
            Error,
            "{}: numel=({},  {}), dim=({}, {})",
            ET_TENSOR_CHECK_PREFIX__,
            a.numel() as usize,
            b.numel() as usize,
            a.dim() as usize,
            b.dim() as usize
        );
        for d in 0..(et_min2!(a.dim(), b.dim())) {
            crate::et_log!(
                Error,
                "    size({}): ({}, {})",
                d as usize,
                a.size(d) as usize,
                b.size(d) as usize
            );
        }

        return false;
    }

    true
}

// [spec:et:def:tensor-util.executorch.et-runtime-namespace.tensors-have-same-shape-fn]
// [spec:et:sem:tensor-util.executorch.et-runtime-namespace.tensors-have-same-shape-fn]
pub fn tensors_have_same_shape(a: &Tensor, b: &Tensor, c: &Tensor) -> bool {
    if a.numel() == 1 && b.numel() == 1 && c.numel() == 1 {
        // PyTorch operators treat all scalar tensors as the same shape even if
        // they have different dims.
        return true;
    }
    let cond1: bool = a.sizes().equals(b.sizes()) && (a.numel() == b.numel());
    let cond2: bool = b.sizes().equals(c.sizes()) && (b.numel() == c.numel());

    if !(cond1 && cond2) {
        crate::et_log!(
            Error,
            "{}: numel=({}, {}, {}), dim=({}, {}, {})",
            ET_TENSOR_CHECK_PREFIX__,
            a.numel() as usize,
            b.numel() as usize,
            c.numel() as usize,
            a.dim() as usize,
            b.dim() as usize,
            c.dim() as usize
        );
        for d in 0..(et_min3!(a.dim(), b.dim(), c.dim())) {
            crate::et_log!(
                Error,
                "    size({}): ({}, {}, {})",
                d as usize,
                a.size(d) as usize,
                b.size(d) as usize,
                c.size(d) as usize
            );
        }

        return false;
    }

    true
}

pub fn tensors_have_same_shape_and_dtype2(a: &Tensor, b: &Tensor) -> bool {
    tensors_have_same_shape2(a, b) && tensors_have_same_dtype2(a, b)
}

// [spec:et:def:tensor-util.executorch.et-runtime-namespace.tensors-have-same-shape-and-dtype-fn]
// [spec:et:sem:tensor-util.executorch.et-runtime-namespace.tensors-have-same-shape-and-dtype-fn]
pub fn tensors_have_same_shape_and_dtype(a: &Tensor, b: &Tensor, c: &Tensor) -> bool {
    tensors_have_same_shape(a, b, c) && tensors_have_same_dtype(a, b, c)
}

// [spec:et:def:tensor-util.executorch.et-runtime-namespace.tensor-has-expected-size-fn]
// [spec:et:sem:tensor-util.executorch.et-runtime-namespace.tensor-has-expected-size-fn]
pub fn tensor_has_expected_size(a: &Tensor, expected_sizes: ArrayRef<SizesType>) -> bool {
    if !(a.sizes().equals(expected_sizes)) {
        crate::et_log!(
            Error,
            "{}: dim=({}, {})",
            ET_TENSOR_CHECK_PREFIX__,
            a.dim() as usize,
            expected_sizes.size()
        );
        let a_dim: usize = a.dim() as usize;
        let expected_dim: usize = expected_sizes.size();
        for d in 0..(et_min2!(a_dim, expected_dim)) {
            crate::et_log!(
                Error,
                "    size({}): ({}, {})",
                d,
                a.size(d as ssize_t) as usize,
                *expected_sizes.at(d) as usize
            );
        }

        return false;
    }
    true
}

pub fn tensors_have_same_strides2(a: &Tensor, b: &Tensor) -> bool {
    if !a.strides().equals(b.strides()) {
        crate::et_log!(
            Error,
            "{}: dim=({}, {})",
            ET_TENSOR_CHECK_PREFIX__,
            a.dim() as usize,
            b.dim() as usize
        );
        for d in 0..(et_min2!(a.dim(), b.dim())) {
            crate::et_log!(
                Error,
                "    stride({}): ({}, {})",
                d as usize,
                *a.strides().at(d as usize) as usize,
                *b.strides().at(d as usize) as usize
            );
        }

        return false;
    }
    true
}

// [spec:et:def:tensor-util.executorch.et-runtime-namespace.tensors-have-same-strides-fn]
// [spec:et:sem:tensor-util.executorch.et-runtime-namespace.tensors-have-same-strides-fn]
pub fn tensors_have_same_strides(a: &Tensor, b: &Tensor, c: &Tensor) -> bool {
    if !(a.strides().equals(b.strides()) && b.strides().equals(c.strides())) {
        crate::et_log!(
            Error,
            "{}: dim=({}, {}, {})",
            ET_TENSOR_CHECK_PREFIX__,
            a.dim() as usize,
            b.dim() as usize,
            c.dim() as usize
        );
        for d in 0..(et_min3!(a.dim(), b.dim(), c.dim())) {
            crate::et_log!(
                Error,
                "    stride({}): ({}, {}, {})",
                d as usize,
                *a.strides().at(d as usize) as usize,
                *b.strides().at(d as usize) as usize,
                *c.strides().at(d as usize) as usize
            );
        }

        return false;
    }
    true
}

// [spec:et:def:tensor-util.executorch.et-runtime-namespace.tensor-is-contiguous-fn]
// [spec:et:sem:tensor-util.executorch.et-runtime-namespace.tensor-is-contiguous-fn]
pub fn tensor_is_contiguous(t: &Tensor) -> bool {
    let strides = t.strides();
    let sizes = t.sizes();
    // If tensor is 0-dim (i.e. a scalar tensor) it is contiguous
    if strides.size() == 0 {
        return true;
    }
    et_check_or_return_false!(
        *strides.at(strides.size() - 1) == 1,
        "Tensor is not contiguous; the stride of the last dimension must be 1, but got {}",
        *strides.at(strides.size() - 1) as usize
    );
    let mut i: i32 = strides.size() as i32 - 1;
    while i > 0 {
        et_check_or_return_false!(
            *strides.at((i - 1) as usize) == *strides.at(i as usize) * *sizes.at(i as usize),
            "Tensor is not contiguous; the stride of dim {} should be equal to strides[{}] * sizes[{}] = {}, but found {}",
            (i - 1) as usize,
            i as usize,
            i as usize,
            (*strides.at(i as usize) * *sizes.at(i as usize)) as usize,
            *strides.at((i - 1) as usize) as usize
        );
        i -= 1;
    }
    true
}

// [spec:et:def:tensor-util.executorch.et-runtime-namespace.tensors-have-same-rank-fn]
// [spec:et:sem:tensor-util.executorch.et-runtime-namespace.tensors-have-same-rank-fn]
pub fn tensors_have_same_rank(a: &Tensor, b: &Tensor) -> bool {
    et_check_or_return_false!(
        a.dim() == b.dim(),
        "{}: rank={{{}, {}}}",
        ET_TENSOR_CHECK_PREFIX__,
        a.dim() as ssize_t,
        b.dim() as ssize_t
    );
    true
}

// [spec:et:def:tensor-util.executorch.et-runtime-namespace.tensor-is-scalar-fn]
// [spec:et:sem:tensor-util.executorch.et-runtime-namespace.tensor-is-scalar-fn]
pub fn tensor_is_scalar(t: &Tensor) -> bool {
    t.dim() == 0 && t.numel() == 1
}

/// Returns the product of dim[0:dim), not including dim.
// [spec:et:def:tensor-util.executorch.et-runtime-namespace.get-leading-dims-fn]
// [spec:et:sem:tensor-util.executorch.et-runtime-namespace.get-leading-dims-fn]
pub fn getLeadingDims(tensor: &Tensor, dim: i64) -> usize {
    et_check_msg!(
        dim >= 0 && dim <= tensor.dim() as i64,
        "Ending dimension {} should be in the range [0, tensor.dim() {}].",
        dim,
        tensor.dim() as ssize_t
    );
    let mut dims: usize = 1;
    for i in 0..dim {
        let mut next_dims: usize = 0;
        et_check_msg!(
            !mul_overflows(dims, tensor.size(i as ssize_t) as usize, &mut next_dims),
            "Overflow computing leading dims at dimension {}",
            i as ssize_t
        );
        dims = next_dims;
    }
    dims
}

/// Returns the product of dim[dim+1:].
// [spec:et:def:tensor-util.executorch.et-runtime-namespace.get-trailing-dims-fn]
// [spec:et:sem:tensor-util.executorch.et-runtime-namespace.get-trailing-dims-fn]
pub fn getTrailingDims(tensor: &Tensor, dim: i64) -> usize {
    et_check_msg!(
        dim >= -1 && dim < tensor.dim() as i64,
        "Starting dimension {} should be in the range [-1, tensor.dim() -1 {}).",
        dim,
        tensor.dim() as ssize_t
    );
    let mut dims: usize = 1;
    let mut i: usize = (dim + 1) as usize;
    while i < tensor.dim() as usize {
        let mut next_dims: usize = 0;
        et_check_msg!(
            !mul_overflows(dims, tensor.size(i as ssize_t) as usize, &mut next_dims),
            "Overflow computing trailing dims at dimension {}",
            i
        );
        dims = next_dims;
        i += 1;
    }
    dims
}

/// Given a N-dimensional tensor coordinate, return a linear index that can be
/// used to access the corresponding element in the tensor's data buffer.
///
/// # Safety
/// `coordinate` must point to at least `tensor.dim()` valid `size_t` elements.
// [spec:et:def:tensor-util.executorch.et-runtime-namespace.coordinate-to-index-fn]
// [spec:et:sem:tensor-util.executorch.et-runtime-namespace.coordinate-to-index-fn]
pub unsafe fn coordinateToIndex(tensor: &Tensor, coordinate: *const usize) -> usize {
    let mut index: usize = 0;
    for d in 0..tensor.dim() {
        index += unsafe { *coordinate.add(d as usize) } * getTrailingDims(tensor, d as i64);
    }
    index
}

/// Produce a memoized array for use with repeated calls to
/// coordinateToIndexWithTrailingDimsMemo.
///
/// # Safety
/// `trailing_dims_memo` must point to at least `kTensorDimensionLimit` valid
/// `size_t` elements.
// [spec:et:def:tensor-util.executorch.et-runtime-namespace.memoize-trailing-dims-fn]
// [spec:et:sem:tensor-util.executorch.et-runtime-namespace.memoize-trailing-dims-fn]
pub unsafe fn memoizeTrailingDims(tensor: &Tensor, trailing_dims_memo: *mut usize) {
    let tensor_dim = tensor.dim();
    let mut dims: usize = 1;
    let mut ii: i32 = tensor_dim as i32 - 1;
    while ii >= 0 {
        unsafe { *trailing_dims_memo.add(ii as usize) = dims };
        dims *= tensor.size(ii as ssize_t) as usize;
        ii -= 1;
    }
}

/// Like coordinateToIndex, but faster for repeated calls with the same tensor.
///
/// # Safety
/// `coordinate` and `trailing_dims_memo` must each point to valid elements as
/// documented for `coordinateToIndex` / `memoizeTrailingDims`.
// [spec:et:def:tensor-util.executorch.et-runtime-namespace.coordinate-to-index-with-trailing-dims-memo-fn]
// [spec:et:sem:tensor-util.executorch.et-runtime-namespace.coordinate-to-index-with-trailing-dims-memo-fn]
pub unsafe fn coordinateToIndexWithTrailingDimsMemo(
    tensor: &Tensor,
    coordinate: *const usize,
    trailing_dims_memo: *const usize,
) -> usize {
    let mut index: usize = 0;
    for d in 0..tensor.dim() {
        index +=
            unsafe { *coordinate.add(d as usize) } * unsafe { *trailing_dims_memo.add(d as usize) };
    }
    index
}

/// Given the linear index return the N-dimensional tensor coordinate. This is
/// the inverse operation of coordinateToIndex.
///
/// # Safety
/// `coordinate` must point to at least `tensor.dim()` valid `size_t` elements.
// [spec:et:def:tensor-util.executorch.et-runtime-namespace.index-to-coordinate-fn]
// [spec:et:sem:tensor-util.executorch.et-runtime-namespace.index-to-coordinate-fn]
pub unsafe fn indexToCoordinate(tensor: &Tensor, mut index: usize, coordinate: *mut usize) {
    et_check!(index < tensor.numel() as usize);
    for i in 0..tensor.dim() {
        let dim = tensor.dim() - 1 - i;
        let dim_size: usize = tensor.size(dim) as usize;
        unsafe { *coordinate.add(dim as usize) = index % dim_size };
        index /= dim_size;
    }
}

/// Extracts an integer value from a scalar Tensor.
// PORT-NOTE: the C++ `extract_scalar_tensor` is a family of SFINAE-selected
// overloads over the target type. Rust cannot express the overload set by name;
// each is ported as a distinct free function (`_int` / `_float` / `_bool`)
// generic over the corresponding numeric bound. The integer overload mirrors
// `ET_FORALL_INT_TYPES` (Byte, Char, Short, Int, Long). Construct deviation.
//
// `IntT` must be an integral type; the range-check bounds come from `min`/`max`.
// [spec:et:def:tensor-util.executorch.et-runtime-namespace.extract-scalar-tensor-fn]
// [spec:et:sem:tensor-util.executorch.et-runtime-namespace.extract-scalar-tensor-fn]
pub fn extract_scalar_tensor_int<IntT>(tensor: &Tensor, out_val: &mut IntT) -> bool
where
    IntT: IntBounds + Copy,
{
    if tensor.numel() != 1 {
        return false;
    }

    // CASE_INT_DTYPE reads element 0, range-checks the widened value against
    // [lowest(IntT), max(IntT)], then narrows.
    macro_rules! case_int_dtype {
        ($tensor_ctype:ty) => {{
            let val: $tensor_ctype = unsafe { *tensor.const_data_ptr::<$tensor_ctype>() };
            let val_i128: i128 = val as i128;
            if val_i128 < IntT::lowest_i128() || val_i128 > IntT::max_i128() {
                return false;
            }
            *out_val = IntT::from_i128_truncating(val_i128);
            return true;
        }};
    }

    match tensor.scalar_type() {
        ScalarType::Byte => case_int_dtype!(u8),
        ScalarType::Char => case_int_dtype!(i8),
        ScalarType::Short => case_int_dtype!(i16),
        ScalarType::Int => case_int_dtype!(i32),
        ScalarType::Long => case_int_dtype!(i64),
        _ => false,
    }
}

// PORT-NOTE: `std::numeric_limits<INT_T>::lowest()/max()` and the final
// `static_cast<INT_T>(val)` narrowing have no single core-Rust trait; modeled
// via `IntBounds`, which exposes the bounds widened to `i128` and a truncating
// `i128 -> Self` cast mirroring `static_cast`.
pub trait IntBounds {
    fn lowest_i128() -> i128;
    fn max_i128() -> i128;
    fn from_i128_truncating(v: i128) -> Self;
}
macro_rules! impl_int_bounds {
    ($($t:ty),*) => {$(
        impl IntBounds for $t {
            fn lowest_i128() -> i128 { <$t>::MIN as i128 }
            fn max_i128() -> i128 { <$t>::MAX as i128 }
            fn from_i128_truncating(v: i128) -> Self { v as $t }
        }
    )*};
}
impl_int_bounds!(i8, i16, i32, i64, u8, u16, u32, u64, isize, usize);

/// Extracts a floating point value from a scalar Tensor.
///
/// PORT-NOTE: floating overload; `ET_FORALL_REALHBF16_TYPES` set. Non-finite
/// values skip the range check. Modeled generically over `f64`-convertible
/// bounds; Half/BFloat16 element reads widen via `f64::from`.
pub fn extract_scalar_tensor_float(tensor: &Tensor, out_val: &mut f64) -> bool {
    if tensor.numel() != 1 {
        return false;
    }
    macro_rules! case_real_dtype {
        ($tensor_ctype:ty, $to_f64:expr) => {{
            let raw: $tensor_ctype = unsafe { *tensor.const_data_ptr::<$tensor_ctype>() };
            let val: f64 = $to_f64(raw);
            if val.is_finite() && (val < f64::MIN || val > f64::MAX) {
                return false;
            }
            *out_val = val;
            return true;
        }};
    }
    match tensor.scalar_type() {
        ScalarType::Byte => case_real_dtype!(u8, |v: u8| v as f64),
        ScalarType::Char => case_real_dtype!(i8, |v: i8| v as f64),
        ScalarType::Short => case_real_dtype!(i16, |v: i16| v as f64),
        ScalarType::Int => case_real_dtype!(i32, |v: i32| v as f64),
        ScalarType::Long => case_real_dtype!(i64, |v: i64| v as f64),
        ScalarType::Float => case_real_dtype!(f32, |v: f32| v as f64),
        ScalarType::Double => case_real_dtype!(f64, |v: f64| v),
        ScalarType::Half => {
            case_real_dtype!(
                crate::runtime::core::portable_type::Half,
                |v: crate::runtime::core::portable_type::Half| v.to_f64()
            )
        }
        ScalarType::BFloat16 => {
            case_real_dtype!(
                crate::runtime::core::portable_type::BFloat16,
                |v: crate::runtime::core::portable_type::BFloat16| v.to_f64()
            )
        }
        _ => false,
    }
}

/// Extracts a boolean value from a scalar Tensor.
// [spec:et:def:tensor-util.executorch.extract-scalar-tensor-fn]
// [spec:et:sem:tensor-util.executorch.extract-scalar-tensor-fn]
pub fn extract_scalar_tensor_bool(tensor: &Tensor, out_val: &mut bool) -> bool {
    if tensor.scalar_type() != ScalarType::Bool {
        return false;
    }
    if tensor.numel() != 1 {
        return false;
    }

    let val: bool = unsafe { *tensor.const_data_ptr::<bool>() };

    *out_val = val;

    true
}

/// These APIs should not be used outside of Executor.cpp.
pub mod internal {
    use super::*;

    // PORT-NOTE: The `internal::*` APIs are *declared* here in the C++ header and
    // *defined* out-of-line in tensor_util_portable.cpp / tensor_util_aten.cpp
    // per build mode. In the Rust port the portable definitions live in
    // tensor_util_portable.rs; these thin forwarders re-expose them under the
    // header module's `internal` namespace, matching the C++ symbol layout.

    // [spec:et:def:tensor-util.executorch.internal.share-tensor-data-fn]
    // [spec:et:sem:tensor-util.executorch.internal.share-tensor-data-fn]
    pub fn share_tensor_data(t_dst: &Tensor, t_src: &Tensor) -> Error {
        super::super::tensor_util_portable::internal::share_tensor_data(t_dst, t_src)
    }

    // [spec:et:def:tensor-util.executorch.internal.copy-tensor-data-fn]
    // [spec:et:sem:tensor-util.executorch.internal.copy-tensor-data-fn]
    pub fn copy_tensor_data(t_dst: &Tensor, t_src: &Tensor) -> Error {
        super::super::tensor_util_portable::internal::copy_tensor_data(t_dst, t_src)
    }

    // [spec:et:def:tensor-util.executorch.internal.set-tensor-data-fn]
    // [spec:et:sem:tensor-util.executorch.internal.set-tensor-data-fn]
    #[must_use]
    pub fn set_tensor_data(
        t: &Tensor,
        buffer: *mut core::ffi::c_void,
        buffer_size: usize,
    ) -> Error {
        super::super::tensor_util_portable::internal::set_tensor_data(t, buffer, buffer_size)
    }

    // [spec:et:def:tensor-util.executorch.internal.reset-data-ptr-fn]
    // [spec:et:sem:tensor-util.executorch.internal.reset-data-ptr-fn]
    pub fn reset_data_ptr(tensor: &Tensor) {
        super::super::tensor_util_portable::internal::reset_data_ptr(tensor)
    }

    // [spec:et:def:tensor-util.executorch.internal.resize-tensor-impl-fn]
    // [spec:et:sem:tensor-util.executorch.internal.resize-tensor-impl-fn]
    #[must_use]
    pub fn resize_tensor_impl(
        impl_: *mut crate::runtime::core::portable_type::tensor_impl::TensorImpl,
        new_sizes: ArrayRef<SizesType>,
    ) -> Error {
        super::super::tensor_util_portable::internal::resize_tensor_impl(impl_, new_sizes)
    }
}

/// Resize a tensor to new_sizes, rank must stay the same.
#[must_use]
pub fn resize_tensor_same_type(t: &Tensor, new_sizes: ArrayRef<SizesType>) -> Error {
    internal::resize_tensor_impl(t.unsafe_get_tensor_impl(), new_sizes)
}

/// Resize a tensor to new_sizes, rank must stay the same. Foreign-integer
/// overload (T != SizesType).
// [spec:et:def:tensor-util.executorch.resize-tensor-fn]
// [spec:et:sem:tensor-util.executorch.resize-tensor-fn]
#[must_use]
pub fn resize_tensor<T>(t: &Tensor, new_sizes: ArrayRef<T>) -> Error
where
    T: Copy,
    SizesType: TryFromLossy<T>,
{
    // Need to cast the input array to an array of Tensor::SizesType
    let mut new_sizes_casted: [SizesType; K_TENSOR_DIMENSION_LIMIT] = [0; K_TENSOR_DIMENSION_LIMIT];
    let new_sizes_ndim: usize = new_sizes.size();
    crate::et_check_or_return_error!(
        new_sizes_ndim <= K_TENSOR_DIMENSION_LIMIT,
        InvalidArgument,
        "new_sizes_ndim {} is greater than kTensorDimensionLimit {}",
        new_sizes_ndim,
        K_TENSOR_DIMENSION_LIMIT
    );
    for i in 0..new_sizes_ndim {
        new_sizes_casted[i] = SizesType::try_from_lossy(*new_sizes.at(i));
    }

    internal::resize_tensor_impl(
        t.unsafe_get_tensor_impl(),
        ArrayRef::from_raw_parts(new_sizes_casted.as_ptr(), new_sizes_ndim),
    )
}

// PORT-NOTE: models the C++ `static_cast<SizesType>(new_sizes[i])` narrowing for
// the SFINAE overload. `TryFromLossy` performs an `as`-style truncating cast to
// preserve the bug-for-bug narrowing semantics.
pub trait TryFromLossy<From> {
    fn try_from_lossy(v: From) -> Self;
}
impl TryFromLossy<i64> for i32 {
    fn try_from_lossy(v: i64) -> Self {
        v as i32
    }
}
// Identity narrowing for callers that already hold a `SizesType` (`i32`) array,
// e.g. `resize_tensor(out, {output_sizes, output_ndim})` in matmul/pool kernels
// where the C++ `SizesType[]` overload passes elements through unchanged.
impl TryFromLossy<i32> for i32 {
    fn try_from_lossy(v: i32) -> Self {
        v
    }
}

/// DEPRECATED: Use `resize_tensor()` instead, which can fail non-fatally.
// [spec:et:def:tensor-util.executorch.resize-fn]
// [spec:et:sem:tensor-util.executorch.resize-fn]
#[deprecated]
pub fn resize(t: &Tensor, new_sizes: ArrayRef<SizesType>) {
    let err: Error = resize_tensor_same_type(t, new_sizes);
    et_check_msg!(
        err == Error::Ok,
        "Could not resize Tensor; see logs for details"
    );
}

/// Get dim_order of a Tensor and write it to out_dim_order.
///
/// # Safety
/// `out_dim_order` must point to at least `out_dim_order_size` valid
/// `DimOrderType` elements.
// [spec:et:def:tensor-util.executorch.get-dim-order-fn]
// [spec:et:sem:tensor-util.executorch.get-dim-order-fn]
#[must_use]
pub unsafe fn get_dim_order(
    tensor: &Tensor,
    out_dim_order: *mut DimOrderType,
    out_dim_order_size: usize,
) -> Error {
    unsafe { super::tensor_util_portable::get_dim_order(tensor, out_dim_order, out_dim_order_size) }
}

/// Checks whether a tensor has a valid dim order.
// [spec:et:def:tensor-util.executorch.tensor-has-valid-dim-order-fn]
// [spec:et:sem:tensor-util.executorch.tensor-has-valid-dim-order-fn]
pub fn tensor_has_valid_dim_order(t: &Tensor) -> bool {
    super::tensor_util_portable::tensor_has_valid_dim_order(t)
}

/// Checks whether a tensor has either the default or channels last dim order.
// [spec:et:def:tensor-util.executorch.tensor-is-default-or-channels-last-dim-order-fn]
// [spec:et:sem:tensor-util.executorch.tensor-is-default-or-channels-last-dim-order-fn]
pub fn tensor_is_default_or_channels_last_dim_order(t: &Tensor) -> bool {
    super::tensor_util_portable::tensor_is_default_or_channels_last_dim_order(t)
}

/// Checks whether a tensor has the default dimension order.
// [spec:et:def:tensor-util.executorch.tensor-is-default-dim-order-fn]
// [spec:et:sem:tensor-util.executorch.tensor-is-default-dim-order-fn]
pub fn tensor_is_default_dim_order(t: &Tensor) -> bool {
    super::tensor_util_portable::tensor_is_default_dim_order(t)
}

/// Checks whether a tensor has the channels last dimension order.
// [spec:et:def:tensor-util.executorch.tensor-is-channels-last-dim-order-fn]
// [spec:et:sem:tensor-util.executorch.tensor-is-channels-last-dim-order-fn]
pub fn tensor_is_channels_last_dim_order(t: &Tensor) -> bool {
    super::tensor_util_portable::tensor_is_channels_last_dim_order(t)
}

/// Asserts that a list of tensors have the same dim_order.
pub fn tensors_have_same_dim_order_list(tensor_list: ArrayRef<Tensor>) -> bool {
    super::tensor_util_portable::tensors_have_same_dim_order(tensor_list)
}

/// Asserts that two tensors have the same dim_order.
// PORT-NOTE: C++ packs `Tensor tensor_list[N] = {a, b, ...}` (a copy of the
// pointer-sized handle). The ported `Tensor` is not `Copy`, so the array is
// rebuilt with `Tensor::new(x.unsafe_get_tensor_impl())`, which yields an
// equivalent non-owning handle over the same `TensorImpl`.
pub fn tensors_have_same_dim_order2(a: &Tensor, b: &Tensor) -> bool {
    let tensor_list: [Tensor; 2] = [
        Tensor::new(a.unsafe_get_tensor_impl()),
        Tensor::new(b.unsafe_get_tensor_impl()),
    ];
    tensors_have_same_dim_order_list(ArrayRef::from_raw_parts(tensor_list.as_ptr(), 2))
}

/// Asserts that three tensors have the same dim_order.
pub fn tensors_have_same_dim_order3(a: &Tensor, b: &Tensor, c: &Tensor) -> bool {
    let tensor_list: [Tensor; 3] = [
        Tensor::new(a.unsafe_get_tensor_impl()),
        Tensor::new(b.unsafe_get_tensor_impl()),
        Tensor::new(c.unsafe_get_tensor_impl()),
    ];
    tensors_have_same_dim_order_list(ArrayRef::from_raw_parts(tensor_list.as_ptr(), 3))
}

/// Asserts that four tensors have the same dim_order.
// [spec:et:def:tensor-util.executorch.tensors-have-same-dim-order-fn]
// [spec:et:sem:tensor-util.executorch.tensors-have-same-dim-order-fn]
pub fn tensors_have_same_dim_order4(a: &Tensor, b: &Tensor, c: &Tensor, d: &Tensor) -> bool {
    let tensor_list: [Tensor; 4] = [
        Tensor::new(a.unsafe_get_tensor_impl()),
        Tensor::new(b.unsafe_get_tensor_impl()),
        Tensor::new(c.unsafe_get_tensor_impl()),
        Tensor::new(d.unsafe_get_tensor_impl()),
    ];
    tensors_have_same_dim_order_list(ArrayRef::from_raw_parts(tensor_list.as_ptr(), 4))
}

/// Given an n-dimensional coordinate array and an array of tensor strides,
/// calculates the linear index.
///
/// # Safety
/// `coordinate` and `strides` must each point to at least `ndim` valid elements.
// [spec:et:def:tensor-util.executorch.calculate-linear-index-fn]
// [spec:et:sem:tensor-util.executorch.calculate-linear-index-fn]
pub unsafe fn calculate_linear_index(
    coordinate: *const SizesType,
    strides: *const StridesType,
    ndim: usize,
) -> usize {
    let mut index: usize = 0;
    for i in 0..ndim {
        index += (unsafe { *coordinate.add(i) } * unsafe { *strides.add(i) }) as usize;
    }
    index
}

// Literal port of runtime/core/exec_aten/util/test/tensor_util_test.cpp AND
// runtime/core/exec_aten/util/test/operator_impl_example_test.cpp.
//
// PORT-NOTE: `operator_impl_example_test.cpp` has no source module of its own
// (it is an example demonstrating this package's utilities). Its `add_tensors_op`
// exercises `ET_CHECK_SAME_SHAPE_AND_DTYPE3` (ported as a module-local macro
// above) plus the real-type switch, so it is homed here alongside the
// `tensor_util_test.cpp` cases that share the same check-macro surface.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::core::exec_aten::testing_util::tensor_factory::TensorFactory;
    use crate::runtime::core::tensor_shape_dynamism::TensorShapeDynamism;
    use crate::{assert_tensor_close, assert_tensor_eq};

    // Mirrors `TensorUtilTest::SetUp()` / `DimOrderUtilTest::SetUp()`'s
    // `runtime_init()`; the PAL must be initialized before code paths that call
    // `ET_LOG`.
    fn setup() {
        crate::runtime::platform::runtime::runtime_init();
    }

    //
    // Ports of operator_impl_example_test.cpp
    //

    // Adds the elements of `a` and `b`, overwriting `out`. Mirrors the templated
    // `add_tensors_impl<CTYPE>`.
    fn add_tensors_impl<C>(a: &Tensor, b: &Tensor, out: &Tensor)
    where
        C: Copy
            + core::ops::Add<Output = C>
            + crate::runtime::core::exec_aten::util::scalar_type_util::CppTypeToScalarType,
    {
        debug_assert!(a.numel() == b.numel() && b.numel() == out.numel());
        let n = a.numel() as usize;
        let data_a = a.const_data_ptr::<C>();
        let data_b = b.const_data_ptr::<C>();
        let data_out = out.mutable_data_ptr::<C>();
        for i in 0..n {
            unsafe {
                *data_out.add(i) = *data_a.add(i) + *data_b.add(i);
            }
        }
    }

    // Element-wise sum of `a` and `b`, overwriting `out`. Mirrors
    // `add_tensors_op`. The manual `switch` over `ET_FORALL_REAL_TYPES` with a
    // `default: ET_CHECK_MSG(false, "Unhandled dtype")` is reproduced literally
    // (an unhandled dtype aborts via the local `et_check_msg!`).
    fn add_tensors_op(a: &Tensor, b: &Tensor, out: &Tensor) {
        et_check_same_shape_and_dtype3!(a, b, out);

        match a.scalar_type() {
            ScalarType::Byte => add_tensors_impl::<u8>(a, b, out),
            ScalarType::Char => add_tensors_impl::<i8>(a, b, out),
            ScalarType::Short => add_tensors_impl::<i16>(a, b, out),
            ScalarType::Int => add_tensors_impl::<i32>(a, b, out),
            ScalarType::Long => add_tensors_impl::<i64>(a, b, out),
            ScalarType::Float => add_tensors_impl::<f32>(a, b, out),
            ScalarType::Double => add_tensors_impl::<f64>(a, b, out),
            _ => {
                et_check_msg!(false, "Unhandled dtype {}", a.scalar_type() as i8);
            }
        }
    }

    // [spec:et:sem:scalar-type-util.executorch.runtime.to-string-fn/test]
    #[test]
    fn operator_impl_example_test_add_int_tensors() {
        let tf = TensorFactory::<i32>::new();
        let sizes = vec![2, 2];
        let out = tf.zeros_default(sizes.clone());
        add_tensors_op(
            &tf.make_default(sizes.clone(), vec![1, 2, 4, 8]),
            &tf.ones_default(sizes.clone()),
            &out,
        );
        assert_tensor_eq!(out, tf.make_default(sizes, vec![2, 3, 5, 9]));
    }

    // [spec:et:sem:scalar-type-util.executorch.runtime.to-string-fn/test]
    #[test]
    fn operator_impl_example_test_add_double_tensors() {
        let tf = TensorFactory::<f64>::new();
        let sizes = vec![2, 2];
        let out = tf.zeros_default(sizes.clone());
        add_tensors_op(
            &tf.make_default(sizes.clone(), vec![1.1, 2.2, 4.4, 8.8]),
            &tf.ones_default(sizes.clone()),
            &out,
        );
        assert_tensor_close!(out, tf.make_default(sizes, vec![2.1, 3.2, 5.4, 9.8]));
    }

    // PORT-NOTE: the following are `ET_EXPECT_DEATH` tests. `add_tensors_op`
    // aborts via `et_check_msg!` (`runtime_abort` -> `libc::abort()`), which
    // terminates the process rather than unwinding, so `#[should_panic]` cannot
    // catch it; ported and `#[ignore]`d.
    #[test]
    #[should_panic]
    #[ignore]
    fn operator_impl_example_test_unhandled_dtype_dies() {
        let tf = TensorFactory::<bool>::new();
        let sizes = vec![2, 2];
        let a = tf.make_default(sizes.clone(), vec![false, true, false, true]);
        let b = tf.make_default(sizes.clone(), vec![true, false, true, false]);
        let out = tf.zeros_default(sizes);
        add_tensors_op(&a, &b, &out);
    }

    #[test]
    #[should_panic]
    #[ignore]
    fn op_add_out_kernel_test_mismatched_input_dims_dies() {
        let tf = TensorFactory::<i32>::new();
        let a = tf.ones_default(vec![4]);
        let b = tf.ones_default(vec![2, 2]);
        let out = tf.zeros_default(vec![4]);
        add_tensors_op(&a, &b, &out);
    }

    #[test]
    #[should_panic]
    #[ignore]
    fn op_add_out_kernel_test_mismatched_input_dtypes_dies() {
        let tf_int = TensorFactory::<i32>::new();
        let tf_float = TensorFactory::<f32>::new();
        let sizes = vec![2, 2];
        let a = tf_int.ones_default(sizes.clone());
        let b = tf_float.ones_default(sizes.clone());
        let out = tf_float.zeros_default(sizes);
        add_tensors_op(&a, &b, &out);
    }

    #[test]
    #[should_panic]
    #[ignore]
    fn op_add_out_kernel_test_mixing_unhandled_dtype_dies() {
        let tf_int = TensorFactory::<i32>::new();
        let tf_bool = TensorFactory::<bool>::new();
        let sizes = vec![2, 2];
        let a = tf_int.ones_default(sizes.clone());
        let b = tf_bool.ones_default(sizes.clone());
        let out = tf_int.zeros_default(sizes);
        add_tensors_op(&a, &b, &out);
    }

    //
    // Ports of tensor_util_test.cpp
    //

    // [spec:et:sem:tensor-util.executorch.runtime.tensors-have-same-shape-fn/test]
    // [spec:et:sem:tensor-util.executorch.runtime.tensors-have-same-dtype-fn/test]
    #[test]
    fn tensor_util_test_identity_checks() {
        let tf_byte = TensorFactory::<u8>::new();
        let t = tf_byte.ones_default(vec![2, 2]);

        // A tensor is the same shape as itself.
        et_check_same_shape2!(t, t);
        et_check_same_shape3!(t, t, t);

        // A tensor is the same dtype as itself.
        et_check_same_dtype2!(t, t);
        et_check_same_dtype3!(t, t, t);

        // A tensor is the same shape and dtype as itself.
        et_check_same_shape_and_dtype2!(t, t);
        et_check_same_shape_and_dtype3!(t, t, t);
    }

    // Positive-path halves of the C++ `SameShapesDifferentDtypes`. The
    // `ET_EXPECT_DEATH(ET_CHECK_SAME_DTYPE2(...))` negatives abort in-process and
    // are `#[ignore]`d below.
    // [spec:et:sem:tensor-util.executorch.runtime.tensors-have-same-shape-fn/test]
    #[test]
    fn tensor_util_test_same_shapes_different_dtypes() {
        let tf_byte = TensorFactory::<u8>::new();
        let tf_int = TensorFactory::<i32>::new();
        let tf_float = TensorFactory::<f32>::new();
        let a = tf_byte.ones_default(vec![2, 2]);
        let b = tf_int.ones_default(vec![2, 2]);
        let c = tf_float.ones_default(vec![2, 2]);

        // The tensors have the same shapes.
        et_check_same_shape2!(a, b);
        et_check_same_shape3!(a, b, c);
    }

    #[test]
    #[should_panic]
    #[ignore]
    fn tensor_util_test_same_shapes_different_dtypes_dtype2_dies() {
        let tf_byte = TensorFactory::<u8>::new();
        let tf_int = TensorFactory::<i32>::new();
        let a = tf_byte.ones_default(vec![2, 2]);
        let b = tf_int.ones_default(vec![2, 2]);
        et_check_same_dtype2!(a, b);
    }

    // Positive-path halves of the C++ `DifferentShapesSameDtypes`.
    // [spec:et:sem:tensor-util.executorch.runtime.tensors-have-same-dtype-fn/test]
    #[test]
    fn tensor_util_test_different_shapes_same_dtypes() {
        let tf_int = TensorFactory::<i32>::new();
        let a = tf_int.ones_default(vec![1, 4]);
        let b = tf_int.ones_default(vec![2, 2]);
        let b2 = tf_int.ones_default(vec![2, 2]);

        // They are the same dtypes.
        et_check_same_dtype2!(a, b);
        et_check_same_dtype2!(b, a);
        et_check_same_dtype3!(a, b, b2);
        et_check_same_dtype3!(b, a, b2);
        et_check_same_dtype3!(b, b2, a);
    }

    #[test]
    #[should_panic]
    #[ignore]
    fn tensor_util_test_different_shapes_same_dtypes_shape2_dies() {
        let tf_int = TensorFactory::<i32>::new();
        let a = tf_int.ones_default(vec![1, 4]);
        let b = tf_int.ones_default(vec![2, 2]);
        et_check_same_shape2!(a, b);
    }

    // [spec:et:sem:tensor-util.executorch.runtime.tensors-have-same-shape-fn/test]
    #[test]
    fn tensor_util_test_zero_dimensional_tensor() {
        let tf_int = TensorFactory::<i32>::new();
        let t = tf_int.ones_default(vec![]);

        assert_eq!(t.dim(), 0);

        et_check_same_shape2!(t, t);
        et_check_same_shape3!(t, t, t);
        et_check_same_dtype2!(t, t);
        et_check_same_dtype3!(t, t, t);
        et_check_same_shape_and_dtype2!(t, t);
        et_check_same_shape_and_dtype3!(t, t, t);
    }

    // [spec:et:sem:tensor-util.executorch.runtime.tensors-have-same-shape-fn/test]
    #[test]
    fn tensor_util_test_empty_tensor() {
        let tf_int = TensorFactory::<i32>::new();
        let t = tf_int.ones_default(vec![0]);

        assert_eq!(t.nbytes(), 0);
        assert_eq!(t.numel(), 0);

        et_check_same_shape2!(t, t);
        et_check_same_shape3!(t, t, t);
        et_check_same_dtype2!(t, t);
        et_check_same_dtype3!(t, t, t);
        et_check_same_shape_and_dtype2!(t, t);
        et_check_same_shape_and_dtype3!(t, t, t);
    }

    // [spec:et:sem:tensor-util.executorch.et-runtime-namespace.get-leading-dims-fn/test]
    #[test]
    fn tensor_util_test_get_leading_dims_smoke_test() {
        let tf_int = TensorFactory::<i32>::new();
        let t = tf_int.ones_default(vec![2, 3, 4]);

        assert_eq!(getLeadingDims(&t, 1), 2);
        assert_eq!(getLeadingDims(&t, 2), 6);
        assert_eq!(getLeadingDims(&t, 3), 24);
    }

    // PORT-NOTE: `GetLeadingDimsInputOutOfBoundDies` — `getLeadingDims` aborts
    // via `et_check_msg!`; `#[ignore]`d death test.
    // [spec:et:sem:tensor-util.executorch.et-runtime-namespace.get-leading-dims-fn/test]
    #[test]
    #[should_panic]
    #[ignore]
    fn tensor_util_test_get_leading_dims_input_out_of_bound_dies() {
        setup();
        let tf_int = TensorFactory::<i32>::new();
        let t = tf_int.ones_default(vec![2, 3, 4]);
        let _ = getLeadingDims(&t, -2);
    }

    // [spec:et:sem:tensor-util.executorch.et-runtime-namespace.get-trailing-dims-fn/test]
    #[test]
    fn tensor_util_test_get_trailing_dims_smoke_test() {
        let tf_int = TensorFactory::<i32>::new();
        let t = tf_int.ones_default(vec![2, 3, 4]);

        assert_eq!(getTrailingDims(&t, 1), 4);
        assert_eq!(getTrailingDims(&t, 0), 12);
        assert_eq!(getTrailingDims(&t, -1), 24);
    }

    // PORT-NOTE: `GetTrailingDimsInputOutOfBoundDies` — `#[ignore]`d death test.
    // [spec:et:sem:tensor-util.executorch.et-runtime-namespace.get-trailing-dims-fn/test]
    #[test]
    #[should_panic]
    #[ignore]
    fn tensor_util_test_get_trailing_dims_input_out_of_bound_dies() {
        setup();
        let tf_int = TensorFactory::<i32>::new();
        let t = tf_int.ones_default(vec![2, 3, 4]);
        let _ = getTrailingDims(&t, -2);
    }

    // Positive-path half of `ContiguousCheckSupported`.
    #[test]
    fn tensor_util_test_contiguous_check_supported() {
        let tf_float = TensorFactory::<f32>::new();
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let sizes = vec![1, 2, 3];

        let t_contiguous = tf_float.make_default(sizes.clone(), data);

        // Assert t_contiguous is contiguous.
        et_check_contiguous!(t_contiguous);
    }

    #[test]
    #[should_panic]
    #[ignore]
    fn tensor_util_test_contiguous_check_incontiguous_dies() {
        setup();
        let tf_float = TensorFactory::<f32>::new();
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let sizes = vec![1, 2, 3];
        let t_incontiguous = tf_float.make(sizes, data, vec![2, 1, 2], TensorShapeDynamism::STATIC);
        et_check_contiguous!(t_incontiguous);
    }

    // Positive-path half of `CheckSameContiguousStrideSupported`.
    #[test]
    fn tensor_util_test_check_same_contiguous_stride_supported() {
        let tf_float = TensorFactory::<f32>::new();
        let tf_byte = TensorFactory::<u8>::new();
        let tf_int = TensorFactory::<i32>::new();

        let same_stride_tensor_list = vec![
            tf_float.ones_default(vec![1, 2, 3, 4]),
            tf_byte.ones_default(vec![4, 2, 3, 4]),
            tf_int.ones_default(vec![10, 2, 3, 4]),
            tf_float.make(
                vec![0, 2, 3, 4],
                vec![],
                vec![24, 12, 4, 1],
                TensorShapeDynamism::STATIC,
            ),
        ];

        // Any two tensors in the list have same strides.
        for i in 0..same_stride_tensor_list.len() {
            for j in i..same_stride_tensor_list.len() {
                let ti = &same_stride_tensor_list[i];
                let tj = &same_stride_tensor_list[j];
                et_check_same_strides2!(ti, tj);
            }
        }

        // Any three tensors in the list have same strides.
        for i in 0..same_stride_tensor_list.len() {
            for j in i..same_stride_tensor_list.len() {
                for k in j..same_stride_tensor_list.len() {
                    let ti = &same_stride_tensor_list[i];
                    let tj = &same_stride_tensor_list[j];
                    let tk = &same_stride_tensor_list[k];
                    et_check_same_strides3!(ti, tj, tk);
                }
            }
        }
    }

    // [spec:et:sem:tensor-util.executorch.et-runtime-namespace.extract-scalar-tensor-fn/test]
    #[test]
    fn tensor_util_test_extract_int_scalar_tensor_smoke() {
        let tf_int = TensorFactory::<i32>::new();
        let t = tf_int.ones_default(vec![1]);

        // ET_FORALL_INT_TYPES: Byte, Char, Short, Int, Long.
        let mut out_u8: u8 = 0;
        assert!(extract_scalar_tensor_int(&t, &mut out_u8));
        assert_eq!(out_u8, 1);
        let mut out_i8: i8 = 0;
        assert!(extract_scalar_tensor_int(&t, &mut out_i8));
        assert_eq!(out_i8, 1);
        let mut out_i16: i16 = 0;
        assert!(extract_scalar_tensor_int(&t, &mut out_i16));
        assert_eq!(out_i16, 1);
        let mut out_i32: i32 = 0;
        assert!(extract_scalar_tensor_int(&t, &mut out_i32));
        assert_eq!(out_i32, 1);
        let mut out_i64: i64 = 0;
        assert!(extract_scalar_tensor_int(&t, &mut out_i64));
        assert_eq!(out_i64, 1);
    }

    // PORT-NOTE: the C++ `ET_FORALL_FLOAT_TYPES` extracts into both `float` and
    // `double`; the Rust float overload (`extract_scalar_tensor_float`) is
    // f64-only, so both cases collapse to an `f64` output.
    // [spec:et:sem:tensor-util.executorch.et-runtime-namespace.extract-scalar-tensor-fn/test]
    #[test]
    fn tensor_util_test_extract_float_scalar_tensor_floating_type_smoke() {
        let tf_float = TensorFactory::<f32>::new();
        let t = tf_float.ones_default(vec![1]);
        let mut out: f64 = 0.0;
        assert!(extract_scalar_tensor_float(&t, &mut out));
        assert_eq!(out, 1.0);
    }

    // [spec:et:sem:tensor-util.executorch.et-runtime-namespace.extract-scalar-tensor-fn/test]
    #[test]
    fn tensor_util_test_extract_float_scalar_tensor_integral_type_smoke() {
        let tf_int = TensorFactory::<i32>::new();
        let t = tf_int.ones_default(vec![1]);
        let mut out: f64 = 0.0;
        assert!(extract_scalar_tensor_float(&t, &mut out));
        assert_eq!(out, 1.0);
    }

    // [spec:et:sem:tensor-util.executorch.extract-scalar-tensor-fn/test]
    #[test]
    fn tensor_util_test_extract_bool_scalar_tensor_smoke() {
        let tf_bool = TensorFactory::<bool>::new();
        let t = tf_bool.ones_default(vec![1]);
        let mut out: bool = false;
        assert!(extract_scalar_tensor_bool(&t, &mut out));
        assert_eq!(out, true);
    }

    // [spec:et:sem:tensor-util.executorch.et-runtime-namespace.extract-scalar-tensor-fn/test]
    #[test]
    fn tensor_util_test_float_scalar_tensor_stress_tests() {
        let tf_double = TensorFactory::<f64>::new();
        let mut value: f64 = 0.0;

        // Case: Positive Infinity
        let t_pos_inf = tf_double.make_default(vec![1], vec![f64::INFINITY]);
        assert!(extract_scalar_tensor_float(&t_pos_inf, &mut value));
        assert!(value.is_infinite());

        // Case: Negative Infinity
        let t_neg_inf = tf_double.make_default(vec![1], vec![-f64::INFINITY]);
        assert!(extract_scalar_tensor_float(&t_neg_inf, &mut value));
        assert!(value.is_infinite());

        // Case: Not a Number (NaN)
        let t_nan = tf_double.make_default(vec![1], vec![f64::NAN]);
        assert!(extract_scalar_tensor_float(&t_nan, &mut value));
        assert!(value.is_nan());
    }

    // [spec:et:sem:tensor-util.executorch.et-runtime-namespace.extract-scalar-tensor-fn/test]
    #[test]
    fn tensor_util_test_int_scalar_tensor_not_integral_type_fails() {
        let tf_float = TensorFactory::<f32>::new();
        let t = tf_float.ones_default(vec![1]);
        let mut out: i64 = 0;
        assert!(!extract_scalar_tensor_int(&t, &mut out));
    }

    // [spec:et:sem:tensor-util.executorch.et-runtime-namespace.extract-scalar-tensor-fn/test]
    #[test]
    fn tensor_util_test_float_scalar_tensor_not_floating_type_fails() {
        let tf_bool = TensorFactory::<bool>::new();
        let t = tf_bool.ones_default(vec![1]);
        let mut out: f64 = 0.0;
        assert!(!extract_scalar_tensor_float(&t, &mut out));
    }

    // [spec:et:sem:tensor-util.executorch.et-runtime-namespace.extract-scalar-tensor-fn/test]
    #[test]
    fn tensor_util_test_int_tensor_not_scalar_fails() {
        let tf_int = TensorFactory::<i32>::new();
        let t = tf_int.ones_default(vec![2, 3]);
        let mut out: i64 = 0;
        assert!(!extract_scalar_tensor_int(&t, &mut out));
    }

    // [spec:et:sem:tensor-util.executorch.et-runtime-namespace.extract-scalar-tensor-fn/test]
    #[test]
    fn tensor_util_test_float_tensor_not_scalar_fails() {
        let tf_float = TensorFactory::<f32>::new();
        let t = tf_float.ones_default(vec![2, 3]);
        let mut out: f64 = 0.0;
        assert!(!extract_scalar_tensor_float(&t, &mut out));
    }

    // [spec:et:sem:tensor-util.executorch.et-runtime-namespace.extract-scalar-tensor-fn/test]
    #[test]
    fn tensor_util_test_int_tensor_out_of_bound_fails() {
        let tf_int = TensorFactory::<i32>::new();
        let t = tf_int.make_default(vec![1], vec![256]);
        let mut out: i8 = 0;
        assert!(!extract_scalar_tensor_int(&t, &mut out));
    }

    // PORT-NOTE: the C++ `FloatTensorOutOfBoundFails` extracts a double tensor
    // into a `float`; the Rust float overload is f64-only, so a double value
    // always fits in the f64 output and the "out of bound" case cannot occur.
    // The C++ range check (`val < lowest(float) || val > max(float)`) has no
    // f64->f64 analogue. Recorded rather than ported (would trivially fail its
    // own EXPECT_FALSE if forced through the f64 overload).
    // [spec:et:sem:tensor-util.executorch.et-runtime-namespace.extract-scalar-tensor-fn/test]

    // [spec:et:sem:tensor-util.executorch.extract-scalar-tensor-fn/test]
    #[test]
    fn tensor_util_test_bool_scalar_tensor_not_boolean_type_fails() {
        let tf_byte = TensorFactory::<u8>::new();
        let c = tf_byte.ones_default(vec![1]);
        let mut out: bool = false;
        assert!(!extract_scalar_tensor_bool(&c, &mut out));
    }

    // [spec:et:sem:tensor-util.executorch.extract-scalar-tensor-fn/test]
    #[test]
    fn tensor_util_test_bool_tensor_not_scalar_fails() {
        let tf_bool = TensorFactory::<bool>::new();
        let c = tf_bool.ones_default(vec![2, 3]);
        let mut out: bool = false;
        assert!(!extract_scalar_tensor_bool(&c, &mut out));
    }

    // [spec:et:sem:tensor-util.executorch.et-runtime-namespace.tensor-is-rank-fn/test]
    #[test]
    fn tensor_util_test_tensor_is_rank_test() {
        setup();
        let tf_float = TensorFactory::<f32>::new();
        let a = tf_float.ones_default(vec![2, 3, 5]);

        assert!(tensor_is_rank(&a, 3));
        assert!(!tensor_is_rank(&a, 0));
        assert!(!tensor_is_rank(&a, 5));
    }

    // [spec:et:sem:tensor-util.executorch.et-runtime-namespace.tensor-has-dim-fn/test]
    #[test]
    fn tensor_util_test_tensor_has_dim_test() {
        setup();
        let tf_float = TensorFactory::<f32>::new();
        let a = tf_float.ones_default(vec![2, 3, 5]);

        assert!(tensor_has_dim(&a, 2));
        assert!(tensor_has_dim(&a, 1));
        assert!(tensor_has_dim(&a, 0));
        assert!(tensor_has_dim(&a, -1));
        assert!(tensor_has_dim(&a, -2));
        assert!(tensor_has_dim(&a, -3));

        assert!(!tensor_has_dim(&a, -4));
        assert!(!tensor_has_dim(&a, 3));
    }

    // [spec:et:sem:tensor-util.executorch.et-runtime-namespace.tensors-have-same-dtype-fn/test]
    #[test]
    fn tensor_util_test_tensors_have_same_dtype_test() {
        setup();
        let tf_float = TensorFactory::<f32>::new();
        let tf_int = TensorFactory::<i32>::new();
        let a = tf_float.ones_default(vec![2, 3]);
        let b = tf_float.ones_default(vec![2, 3]);
        let c = tf_float.ones_default(vec![3, 3]);
        let d = tf_int.ones_default(vec![4, 3]);

        assert!(tensors_have_same_dtype2(&a, &b));
        assert!(!tensors_have_same_dtype2(&a, &d));
        assert!(tensors_have_same_dtype(&a, &b, &c));
        assert!(!tensors_have_same_dtype(&a, &b, &d));
    }

    // [spec:et:sem:tensor-util.executorch.et-runtime-namespace.tensors-have-same-size-at-dims-fn/test]
    #[test]
    fn tensor_util_test_tensors_have_same_size_at_dim_test() {
        setup();
        let tf_float = TensorFactory::<f32>::new();
        let a = tf_float.ones_default(vec![2, 3, 4, 5]);
        let b = tf_float.ones_default(vec![5, 4, 3, 2]);

        assert!(tensors_have_same_size_at_dims(&a, 0, &b, 3));
        assert!(tensors_have_same_size_at_dims(&a, 1, &b, 2));
        assert!(!tensors_have_same_size_at_dims(&a, 1, &b, 0));
        assert!(!tensors_have_same_size_at_dims(&a, 4, &b, 0));
        assert!(!tensors_have_same_size_at_dims(&a, 2, &b, 3));
    }

    // [spec:et:sem:tensor-util.executorch.runtime.tensors-have-same-shape-fn/test]
    // [spec:et:sem:tensor-util.executorch.et-runtime-namespace.tensors-have-same-shape-fn/test]
    #[test]
    fn tensor_util_test_tensors_have_same_shape_test() {
        setup();
        let tf_float = TensorFactory::<f32>::new();
        let tf_int = TensorFactory::<i32>::new();
        let tf_byte = TensorFactory::<u8>::new();
        let tf_double = TensorFactory::<f64>::new();
        let tf_bool = TensorFactory::<bool>::new();
        let a = tf_float.ones_default(vec![2, 3]);
        let b = tf_int.ones_default(vec![2, 3]);
        let c = tf_byte.ones_default(vec![2, 3]);
        let d = tf_double.ones_default(vec![3, 2]);
        let e = tf_bool.ones_default(vec![3, 2]);

        assert!(tensors_have_same_shape2(&a, &b));
        assert!(!tensors_have_same_shape2(&a, &d));
        assert!(tensors_have_same_shape2(&d, &e));
        assert!(tensors_have_same_shape(&a, &b, &c));
        assert!(!tensors_have_same_shape(&a, &b, &d));
        assert!(!tensors_have_same_shape(&a, &d, &e));

        let scalar_a = tf_float.ones_default(vec![1, 1]);
        let scalar_b = tf_double.ones_default(vec![1]);
        let scalar_c = tf_int.ones_default(vec![1, 1, 1, 1]);

        assert!(tensors_have_same_shape2(&scalar_a, &scalar_b));
        assert!(tensors_have_same_shape(&scalar_a, &scalar_b, &scalar_c));
    }

    // [spec:et:sem:tensor-util.executorch.runtime.tensors-have-same-shape-and-dtype-fn/test]
    // [spec:et:sem:tensor-util.executorch.et-runtime-namespace.tensors-have-same-shape-and-dtype-fn/test]
    #[test]
    fn tensor_util_test_tensors_have_same_shape_and_dtype_test() {
        setup();
        let tf_float = TensorFactory::<f32>::new();
        let tf_double = TensorFactory::<f64>::new();
        let a = tf_float.ones_default(vec![2, 3]);
        let b = tf_float.ones_default(vec![2, 3]);
        let c = tf_float.ones_default(vec![2, 3]);
        let d = tf_double.ones_default(vec![2, 3]);
        let e = tf_float.ones_default(vec![3, 2]);

        assert!(tensors_have_same_shape_and_dtype2(&a, &b));
        assert!(!tensors_have_same_shape_and_dtype2(&a, &d));
        assert!(tensors_have_same_shape_and_dtype(&a, &b, &c));
        assert!(!tensors_have_same_shape_and_dtype(&a, &b, &d));
        assert!(!tensors_have_same_shape_and_dtype(&a, &d, &e));

        let scalar_a = tf_float.ones_default(vec![1, 1]);
        let scalar_b = tf_float.ones_default(vec![1]);
        let scalar_c = tf_float.ones_default(vec![1, 1, 1, 1]);

        assert!(tensors_have_same_shape_and_dtype2(&scalar_a, &scalar_b));
        assert!(tensors_have_same_shape_and_dtype(
            &scalar_a, &scalar_b, &scalar_c
        ));
    }

    // [spec:et:sem:tensor-util.executorch.runtime.tensors-have-same-strides-fn/test]
    // [spec:et:sem:tensor-util.executorch.et-runtime-namespace.tensors-have-same-strides-fn/test]
    #[test]
    fn tensor_util_test_tensors_have_same_strides_test() {
        setup();
        let tf_float = TensorFactory::<f32>::new();
        let tf_double = TensorFactory::<f64>::new();
        let a = tf_float.full_channels_last(vec![4, 5, 2, 3], 1.0, TensorShapeDynamism::STATIC);
        let b = tf_float.full_channels_last(vec![4, 5, 2, 3], 2.0, TensorShapeDynamism::STATIC);
        let c = tf_float.full_channels_last(vec![4, 5, 2, 3], 3.0, TensorShapeDynamism::STATIC);
        let d = tf_double.ones_default(vec![4, 5, 2, 3]);
        let e = tf_float.ones_default(vec![4, 5, 2, 3]);

        assert!(tensors_have_same_strides2(&a, &b));
        assert!(!tensors_have_same_strides2(&a, &d));
        assert!(tensors_have_same_strides(&a, &b, &c));
        assert!(!tensors_have_same_strides(&a, &b, &d));
        assert!(!tensors_have_same_strides(&a, &d, &e));
    }

    // [spec:et:sem:tensor-util.executorch.et-runtime-namespace.tensor-is-contiguous-fn/test]
    #[test]
    fn tensor_util_test_tensor_is_contiguous() {
        setup();
        let tf_float = TensorFactory::<f32>::new();
        // Note that the strides.size() == 0 case is not tested.
        let a = tf_float.full_channels_last(vec![4, 5, 2, 3], 1.0, TensorShapeDynamism::STATIC);
        let b = tf_float.ones_default(vec![4, 5, 2, 3]);
        let c = tf_float.full_channels_last(vec![1, 1, 1, 1], 1.0, TensorShapeDynamism::STATIC);
        let d = tf_float.ones_default(vec![]);

        assert!(!tensor_is_contiguous(&a));
        assert!(tensor_is_contiguous(&b));
        assert!(tensor_is_contiguous(&c));
        assert!(tensor_is_contiguous(&d));
    }

    // [spec:et:sem:tensor-util.executorch.resize-tensor-fn/test]
    #[test]
    fn tensor_util_test_resize_zero_dim_tensor() {
        setup();
        let tf_float = TensorFactory::<f32>::new();
        let a = tf_float.ones_default(vec![]);

        let new_sizes: [SizesType; 0] = [];
        assert_eq!(
            resize_tensor(&a, ArrayRef::from_raw_parts(new_sizes.as_ptr(), 0)),
            Error::Ok
        );
        assert_eq!(a.dim(), 0);
    }

    // [spec:et:sem:tensor-util.executorch.tensors-have-same-dim-order-fn/test]
    // delegates into (and thus genuinely exercises) the portable implementation:
    // [spec:et:sem:tensor-util-portable.executorch.runtime.tensors-have-same-dim-order-fn/test]
    #[test]
    fn tensor_util_test_same_dim_order_contiguous() {
        let tf_byte = TensorFactory::<u8>::new();
        let tf_int = TensorFactory::<i32>::new();
        let tf_float = TensorFactory::<f32>::new();
        let sizes = vec![3, 5, 2, 1];
        let a = tf_byte.ones_default(sizes.clone());
        let b = tf_int.zeros_default(sizes.clone());
        let c = tf_float.full(sizes, 0.1, TensorShapeDynamism::STATIC);

        assert!(tensors_have_same_dim_order2(&a, &b));
        assert!(tensors_have_same_dim_order2(&b, &a));
        assert!(tensors_have_same_dim_order3(&a, &b, &c));
        assert!(tensors_have_same_dim_order3(&b, &c, &a));
        assert!(tensors_have_same_dim_order3(&c, &a, &b));
    }

    // [spec:et:sem:tensor-util.executorch.tensors-have-same-dim-order-fn/test]
    // [spec:et:sem:tensor-util-portable.executorch.runtime.tensors-have-same-dim-order-fn/test]
    #[test]
    fn tensor_util_test_same_dim_order_channels_last() {
        let tf_byte = TensorFactory::<u8>::new();
        let tf_int = TensorFactory::<i32>::new();
        let tf_float = TensorFactory::<f32>::new();
        let sizes = vec![3, 5, 2, 1];
        let a = tf_byte.full_channels_last(sizes.clone(), 1, TensorShapeDynamism::STATIC);
        let b = tf_int.full_channels_last(sizes.clone(), 0, TensorShapeDynamism::STATIC);
        let c = tf_float.full_channels_last(sizes, 0.1, TensorShapeDynamism::STATIC);

        assert!(tensors_have_same_dim_order2(&a, &b));
        assert!(tensors_have_same_dim_order2(&b, &a));
        assert!(tensors_have_same_dim_order3(&a, &b, &c));
        assert!(tensors_have_same_dim_order3(&b, &c, &a));
        assert!(tensors_have_same_dim_order3(&c, &a, &b));
    }

    // [spec:et:sem:tensor-util.executorch.tensors-have-same-dim-order-fn/test]
    // [spec:et:sem:tensor-util-portable.executorch.runtime.tensors-have-same-dim-order-fn/test]
    #[test]
    fn tensor_util_test_same_shapes_different_dim_order() {
        setup();
        let tf_byte = TensorFactory::<u8>::new();
        let tf_int = TensorFactory::<i32>::new();
        let tf_float = TensorFactory::<f32>::new();
        let sizes = vec![3, 5, 2, 1];
        let a = tf_byte.ones_default(sizes.clone());
        let b = tf_int.full_channels_last(sizes.clone(), 0, TensorShapeDynamism::STATIC);
        let c = tf_float.full_channels_last(sizes, 0.1, TensorShapeDynamism::STATIC);

        assert!(!tensors_have_same_dim_order2(&a, &b));
        assert!(!tensors_have_same_dim_order2(&b, &a));

        assert!(!tensors_have_same_dim_order3(&a, &b, &c));
        assert!(!tensors_have_same_dim_order3(&a, &c, &b));
        assert!(!tensors_have_same_dim_order3(&c, &b, &a));
    }

    //
    // Focused unit tests for pure helpers/predicates not directly exercised by
    // the ported tensor_util_test.cpp suite. Semantics are checked against the
    // C++ tensor_util.h source and its docs/spec/port sem rules.
    //

    // [spec:et:sem:tensor-util.executorch.et-runtime-namespace.dim-is-valid-fn/test]
    #[test]
    fn tensor_util_dim_is_valid() {
        setup();
        // Between -upper_bound and upper_bound - 1 inclusive.
        assert!(dim_is_valid(0, 3));
        assert!(dim_is_valid(2, 3));
        assert!(dim_is_valid(-3, 3));
        assert!(dim_is_valid(-1, 3));
        assert!(!dim_is_valid(3, 3));
        assert!(!dim_is_valid(-4, 3));
    }

    // [spec:et:sem:tensor-util.executorch.et-runtime-namespace.nonzero-dim-fn/test]
    #[test]
    fn tensor_util_nonzero_dim() {
        let tf = TensorFactory::<f32>::new();
        // 0-dim tensor reports 1.
        assert_eq!(nonzero_dim(&tf.ones_default(vec![])), 1);
        // Otherwise reports the actual rank.
        assert_eq!(nonzero_dim(&tf.ones_default(vec![2, 3, 4])), 3);
    }

    // [spec:et:sem:tensor-util.executorch.et-runtime-namespace.nonempty-size-fn/test]
    #[test]
    fn tensor_util_nonempty_size() {
        let tf = TensorFactory::<f32>::new();
        // 0-dim tensor returns 1 regardless of the requested dim.
        assert_eq!(nonempty_size(&tf.ones_default(vec![]), 0), 1);
        // Otherwise returns the size along the dim.
        let t = tf.ones_default(vec![2, 3, 4]);
        assert_eq!(nonempty_size(&t, 0), 2);
        assert_eq!(nonempty_size(&t, 1), 3);
        assert_eq!(nonempty_size(&t, 2), 4);
    }

    // [spec:et:sem:tensor-util.executorch.et-runtime-namespace.tensor-can-cast-to-fn/test]
    #[test]
    fn tensor_util_tensor_can_cast_to() {
        setup();
        let tf_byte = TensorFactory::<u8>::new();
        let t = tf_byte.ones_default(vec![2, 2]);
        // Byte promotes to Int/Float.
        assert!(tensor_can_cast_to(&t, ScalarType::Int));
        assert!(tensor_can_cast_to(&t, ScalarType::Float));
        // Float cannot cast down to Byte.
        let tf_float = TensorFactory::<f32>::new();
        let f = tf_float.ones_default(vec![2, 2]);
        assert!(!tensor_can_cast_to(&f, ScalarType::Byte));
    }

    // [spec:et:sem:tensor-util.executorch.et-runtime-namespace.tensor-is-bool-type-fn/test]
    // [spec:et:sem:tensor-util.executorch.et-runtime-namespace.tensor-is-integral-type-fn/test]
    // [spec:et:sem:tensor-util.executorch.et-runtime-namespace.tensor-is-floating-type-fn/test]
    // [spec:et:sem:tensor-util.executorch.et-runtime-namespace.tensor-is-real-type-fn/test]
    // [spec:et:sem:tensor-util.executorch.et-runtime-namespace.tensor-is-realh-type-fn/test]
    // [spec:et:sem:tensor-util.executorch.et-runtime-namespace.tensor-is-realhb-type-fn/test]
    // [spec:et:sem:tensor-util.executorch.et-runtime-namespace.tensor-is-realhbf16-type-fn/test]
    // [spec:et:sem:tensor-util.executorch.et-runtime-namespace.tensor-is-realhbbf16-type-fn/test]
    // [spec:et:sem:tensor-util.executorch.et-runtime-namespace.tensor-is-complex-type-fn/test]
    // [spec:et:sem:tensor-util.executorch.et-runtime-namespace.tensor-is-bits-type-fn/test]
    // [spec:et:sem:tensor-util.executorch.et-runtime-namespace.tensor-is-type-fn/test]
    #[test]
    fn tensor_util_tensor_type_predicates() {
        setup();
        let tf_bool = TensorFactory::<bool>::new();
        let tf_int = TensorFactory::<i32>::new();
        let tf_float = TensorFactory::<f32>::new();
        let sizes = vec![2, 2];
        let b = tf_bool.ones_default(sizes.clone());
        let i = tf_int.ones_default(sizes.clone());
        let f = tf_float.ones_default(sizes);

        // tensor_is_bool_type
        assert!(tensor_is_bool_type(&b));
        assert!(!tensor_is_bool_type(&i));

        // tensor_is_integral_type: Bool only counts when include_bool.
        assert!(tensor_is_integral_type(&i, false));
        assert!(!tensor_is_integral_type(&b, false));
        assert!(tensor_is_integral_type(&b, true));
        assert!(!tensor_is_integral_type(&f, true));

        // tensor_is_floating_type
        assert!(tensor_is_floating_type(&f));
        assert!(!tensor_is_floating_type(&i));

        // tensor_is_real_type (int + float, not bool)
        assert!(tensor_is_real_type(&i));
        assert!(tensor_is_real_type(&f));
        assert!(!tensor_is_real_type(&b));

        // realh includes real; bool still excluded.
        assert!(tensor_is_realh_type(&f));
        assert!(!tensor_is_realh_type(&b));

        // realhb / realhbf16 / realhbbf16 include bool / bf16 supersets.
        assert!(tensor_is_realhb_type(&b));
        assert!(tensor_is_realhb_type(&i));
        assert!(tensor_is_realhbf16_type(&f));
        assert!(!tensor_is_realhbf16_type(&b));
        assert!(tensor_is_realhbbf16_type(&b));

        // complex / bits: none of these real/bool tensors qualify.
        assert!(!tensor_is_complex_type(&f));
        assert!(!tensor_is_bits_type(&i));

        // tensor_is_type: matches when the tensor's dtype is one of the three.
        assert!(tensor_is_type(
            &i,
            ScalarType::Byte,
            ScalarType::Int,
            ScalarType::Long
        ));
        assert!(!tensor_is_type(
            &i,
            ScalarType::Byte,
            ScalarType::Short,
            ScalarType::Long
        ));
    }

    // [spec:et:sem:tensor-util.executorch.et-runtime-namespace.tensor-is-scalar-fn/test]
    #[test]
    fn tensor_util_tensor_is_scalar() {
        let tf = TensorFactory::<f32>::new();
        // 0-dim, numel 1 -> scalar.
        assert!(tensor_is_scalar(&tf.ones_default(vec![])));
        // 1-dim with one element is NOT a scalar (dim != 0).
        assert!(!tensor_is_scalar(&tf.ones_default(vec![1])));
        // Multi-element tensor is not a scalar.
        assert!(!tensor_is_scalar(&tf.ones_default(vec![2, 2])));
    }

    // [spec:et:sem:tensor-util.executorch.et-runtime-namespace.tensors-have-same-rank-fn/test]
    #[test]
    fn tensor_util_tensors_have_same_rank() {
        setup();
        let tf = TensorFactory::<f32>::new();
        let a = tf.ones_default(vec![2, 3, 4]);
        let b = tf.ones_default(vec![5, 6, 7]);
        let c = tf.ones_default(vec![2, 3]);
        assert!(tensors_have_same_rank(&a, &b));
        assert!(!tensors_have_same_rank(&a, &c));
    }

    // [spec:et:sem:tensor-util.executorch.et-runtime-namespace.tensor-has-rank-greater-or-equal-to-fn/test]
    // [spec:et:sem:tensor-util.executorch.et-runtime-namespace.tensor-has-rank-smaller-or-equal-to-fn/test]
    #[test]
    fn tensor_util_tensor_has_rank_bounds() {
        setup();
        let tf = TensorFactory::<f32>::new();
        let a = tf.ones_default(vec![2, 3, 5]);

        assert!(tensor_has_rank_greater_or_equal_to(&a, 3));
        assert!(tensor_has_rank_greater_or_equal_to(&a, 2));
        assert!(!tensor_has_rank_greater_or_equal_to(&a, 4));

        assert!(tensor_has_rank_smaller_or_equal_to(&a, 3));
        assert!(tensor_has_rank_smaller_or_equal_to(&a, 4));
        assert!(!tensor_has_rank_smaller_or_equal_to(&a, 2));
    }

    // [spec:et:sem:tensor-util.executorch.et-runtime-namespace.tensor-has-expected-size-fn/test]
    #[test]
    fn tensor_util_tensor_has_expected_size() {
        setup();
        let tf = TensorFactory::<f32>::new();
        let a = tf.ones_default(vec![2, 3, 4]);

        let good: [SizesType; 3] = [2, 3, 4];
        assert!(tensor_has_expected_size(
            &a,
            ArrayRef::from_raw_parts(good.as_ptr(), 3)
        ));

        let bad_shape: [SizesType; 3] = [2, 3, 5];
        assert!(!tensor_has_expected_size(
            &a,
            ArrayRef::from_raw_parts(bad_shape.as_ptr(), 3)
        ));

        let bad_rank: [SizesType; 2] = [2, 3];
        assert!(!tensor_has_expected_size(
            &a,
            ArrayRef::from_raw_parts(bad_rank.as_ptr(), 2)
        ));
    }

    // [spec:et:sem:tensor-util.executorch.et-runtime-namespace.tensor-has-non-empty-dim-fn/test]
    #[test]
    fn tensor_util_tensor_has_non_empty_dim() {
        setup();
        let tf = TensorFactory::<f32>::new();
        // No empty dims: every valid dim is non-empty.
        let a = tf.ones_default(vec![2, 3]);
        assert!(tensor_has_non_empty_dim(&a, 0));
        assert!(tensor_has_non_empty_dim(&a, 1));
        assert!(tensor_has_non_empty_dim(&a, -1));

        // A zero-sized dim is empty.
        let e = tf.ones_default(vec![2, 0]);
        assert!(tensor_has_non_empty_dim(&e, 0));
        assert!(!tensor_has_non_empty_dim(&e, 1));
    }

    // [spec:et:sem:tensor-util.executorch.et-runtime-namespace.tensor-dim-has-index-fn/test]
    #[test]
    fn tensor_util_tensor_dim_has_index() {
        setup();
        let tf = TensorFactory::<f32>::new();
        let a = tf.ones_default(vec![2, 3]);
        // dim 1 has size 3: valid indices in [-3, 3).
        assert!(tensor_dim_has_index(&a, 1, 0));
        assert!(tensor_dim_has_index(&a, 1, 2));
        assert!(tensor_dim_has_index(&a, 1, -3));
        assert!(!tensor_dim_has_index(&a, 1, 3));
        assert!(!tensor_dim_has_index(&a, 1, -4));
        // Negative dim normalization: -1 == dim 1.
        assert!(tensor_dim_has_index(&a, -1, 2));
    }

    // [spec:et:sem:tensor-util.executorch.et-runtime-namespace.coordinate-to-index-fn/test]
    // [spec:et:sem:tensor-util.executorch.et-runtime-namespace.memoize-trailing-dims-fn/test]
    // [spec:et:sem:tensor-util.executorch.et-runtime-namespace.coordinate-to-index-with-trailing-dims-memo-fn/test]
    // [spec:et:sem:tensor-util.executorch.et-runtime-namespace.index-to-coordinate-fn/test]
    #[test]
    fn tensor_util_coordinate_index_roundtrip() {
        let tf = TensorFactory::<f32>::new();
        // Contiguous tensor: trailing dims = [12, 4, 1] for sizes [2, 3, 4].
        let t = tf.ones_default(vec![2, 3, 4]);

        // coordinateToIndex: (1, 2, 3) -> 1*12 + 2*4 + 3*1 = 23.
        let coord: [usize; 3] = [1, 2, 3];
        let index = unsafe { coordinateToIndex(&t, coord.as_ptr()) };
        assert_eq!(index, 23);

        // memoizeTrailingDims produces the per-dim trailing products.
        let mut memo: [usize; K_TENSOR_DIMENSION_LIMIT] = [0; K_TENSOR_DIMENSION_LIMIT];
        unsafe { memoizeTrailingDims(&t, memo.as_mut_ptr()) };
        assert_eq!(&memo[0..3], &[12usize, 4, 1]);

        // The memoized path agrees with the direct computation.
        let memo_index =
            unsafe { coordinateToIndexWithTrailingDimsMemo(&t, coord.as_ptr(), memo.as_ptr()) };
        assert_eq!(memo_index, 23);

        // indexToCoordinate inverts coordinateToIndex.
        let mut back: [usize; 3] = [0; 3];
        unsafe { indexToCoordinate(&t, 23, back.as_mut_ptr()) };
        assert_eq!(back, [1usize, 2, 3]);
    }

    // [spec:et:sem:tensor-util.executorch.calculate-linear-index-fn/test]
    #[test]
    fn tensor_util_calculate_linear_index() {
        // coordinate . strides for ndim elements.
        let coord: [SizesType; 3] = [1, 2, 3];
        let strides: [StridesType; 3] = [12, 4, 1];
        let index = unsafe { calculate_linear_index(coord.as_ptr(), strides.as_ptr(), 3) };
        assert_eq!(index, 23);
    }

    // [spec:et:sem:tensor-util.executorch.resize-fn/test]
    #[test]
    #[allow(deprecated)]
    fn tensor_util_resize_deprecated() {
        setup();
        let tf = TensorFactory::<f32>::new();
        // A dynamic-bound tensor can be resized within its rank.
        let a = tf.make(
            vec![2, 3],
            vec![0.0; 6],
            vec![3, 1],
            TensorShapeDynamism::DYNAMIC_BOUND,
        );
        let new_sizes: [SizesType; 2] = [1, 3];
        resize(&a, ArrayRef::from_raw_parts(new_sizes.as_ptr(), 2));
        assert_eq!(a.size(0), 1);
        assert_eq!(a.size(1), 3);
    }

    // Exercises the header-module dim-order forwarders (get_dim_order,
    // tensor_has_valid_dim_order, tensor_is_default_dim_order,
    // tensor_is_channels_last_dim_order), which delegate to tensor_util_portable.
    // [spec:et:sem:tensor-util.executorch.get-dim-order-fn/test]
    // [spec:et:sem:tensor-util.executorch.tensor-has-valid-dim-order-fn/test]
    // [spec:et:sem:tensor-util.executorch.tensor-is-default-dim-order-fn/test]
    // [spec:et:sem:tensor-util.executorch.tensor-is-channels-last-dim-order-fn/test]
    // [spec:et:sem:tensor-util.executorch.tensor-is-default-or-channels-last-dim-order-fn/test]
    #[test]
    fn tensor_util_dim_order_forwarders() {
        setup();
        let tf = TensorFactory::<f32>::new();
        let default = tf.ones_default(vec![2, 3, 4, 5]);
        let channels_last =
            tf.full_channels_last(vec![2, 3, 4, 5], 1.0, TensorShapeDynamism::STATIC);

        // get_dim_order writes the tensor's dim order; default is 0..dim.
        let mut order: [DimOrderType; K_TENSOR_DIMENSION_LIMIT] = [0; K_TENSOR_DIMENSION_LIMIT];
        assert_eq!(
            unsafe { get_dim_order(&default, order.as_mut_ptr(), 4) },
            Error::Ok
        );
        assert_eq!(&order[0..4], &[0, 1, 2, 3]);

        // Validity + classification forwarders.
        assert!(tensor_has_valid_dim_order(&default));
        assert!(tensor_is_default_dim_order(&default));
        assert!(!tensor_is_channels_last_dim_order(&default));
        assert!(tensor_is_default_or_channels_last_dim_order(&default));

        assert!(tensor_has_valid_dim_order(&channels_last));
        assert!(!tensor_is_default_dim_order(&channels_last));
        assert!(tensor_is_channels_last_dim_order(&channels_last));
        assert!(tensor_is_default_or_channels_last_dim_order(&channels_last));
    }

    // Exercises the header-module internal:: forwarders (share/copy/set/reset),
    // which delegate to tensor_util_portable::internal.
    // [spec:et:sem:tensor-util.executorch.internal.share-tensor-data-fn/test]
    // [spec:et:sem:tensor-util.executorch.internal.copy-tensor-data-fn/test]
    // [spec:et:sem:tensor-util.executorch.internal.set-tensor-data-fn/test]
    // [spec:et:sem:tensor-util.executorch.internal.reset-data-ptr-fn/test]
    #[test]
    fn tensor_util_internal_forwarders() {
        setup();
        let tf = TensorFactory::<f32>::new();

        // copy_tensor_data: bytes flow src -> dst.
        let src = tf.make_default(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
        let dst = tf.zeros_default(vec![2, 2]);
        assert_eq!(internal::copy_tensor_data(&dst, &src), Error::Ok);
        let dst_ptr = dst.const_data_ptr::<f32>();
        for i in 0..4 {
            assert_eq!(unsafe { *dst_ptr.add(i) }, unsafe {
                *src.const_data_ptr::<f32>().add(i)
            });
        }

        // share_tensor_data: dst points at src's buffer.
        let shared = tf.zeros_default(vec![2, 2]);
        assert_eq!(internal::share_tensor_data(&shared, &src), Error::Ok);
        assert_eq!(
            shared.const_data_ptr_typed(),
            src.const_data_ptr_typed() as *const core::ffi::c_void
        );

        // set_tensor_data succeeds with a large-enough buffer, fails otherwise.
        let target = tf.zeros_default(vec![2, 2]);
        let mut buf = vec![0u8; target.nbytes()];
        assert_eq!(
            internal::set_tensor_data(
                &target,
                buf.as_mut_ptr() as *mut core::ffi::c_void,
                target.nbytes()
            ),
            Error::Ok
        );
        assert_eq!(target.const_data_ptr_typed(), buf.as_ptr() as *const _);
        assert_eq!(
            internal::set_tensor_data(
                &target,
                buf.as_mut_ptr() as *mut core::ffi::c_void,
                target.nbytes() - 1
            ),
            Error::InvalidArgument
        );

        // reset_data_ptr nulls out the data pointer.
        internal::reset_data_ptr(&target);
        assert!(target.const_data_ptr_typed().is_null());
    }
}
