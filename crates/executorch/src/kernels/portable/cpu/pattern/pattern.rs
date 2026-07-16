//! Literal port of kernels/portable/cpu/pattern/pattern.h.

use crate::runtime::core::portable_type::{BFloat16, Half};

// The C++ pattern.h only *declares* the three `internal::unary_ufunc_*`
// functions (their definitions live in the corresponding .cpp modules) and
// provides the `DEFINE_UNARY_UFUNC_*` macros. In this port:
//   - the function definitions live in
//     `unary_ufunc_realhbf16.rs`, `unary_ufunc_realhbbf16_to_bool.rs`, and
//     `unary_ufunc_realhbbf16_to_floathbf16.rs`, re-exported below;
//   - the `DEFINE_UNARY_UFUNC_*` macros become `macro_rules!` that expand to a
//     full kernel fn `op_name(ctx, in, out)`. The op file that invokes the
//     macro attaches its own annotations above the macro call, so the generated
//     fn carries the op's markers.

// [spec:et:def:pattern.torch.executor.native.internal.unary-ufunc-realhbbf16-to-bool-fn]
// [spec:et:sem:pattern.torch.executor.native.internal.unary-ufunc-realhbbf16-to-bool-fn]
pub use super::unary_ufunc_realhbbf16_to_bool::unary_ufunc_realhbbf16_to_bool;
// [spec:et:def:pattern.torch.executor.native.internal.unary-ufunc-realhbbf16-to-floathbf16-fn]
// [spec:et:sem:pattern.torch.executor.native.internal.unary-ufunc-realhbbf16-to-floathbf16-fn]
pub use super::unary_ufunc_realhbbf16_to_floathbf16::unary_ufunc_realhbbf16_to_floathbf16;
// [spec:et:def:pattern.torch.executor.native.internal.unary-ufunc-realhbf16-fn]
// [spec:et:sem:pattern.torch.executor.native.internal.unary-ufunc-realhbf16-fn]
pub use super::unary_ufunc_realhbf16::unary_ufunc_realhbf16;

// The three `DEFINE_UNARY_UFUNC_*` macros. In C++ each takes a single `fn`
// argument (an overloaded `std::name`, resolved separately for the `float` and
// `double` call sites inside the internal fn). Rust cannot pass one overloaded
// name that resolves to both `fn(f32)->f32` and `fn(f64)->f64`, so the invoking
// op file supplies both the f32 and f64 variants explicitly (e.g. an f32-typed
// and an f64-typed closure over `x.acos()`); this reproduces the two-overload
// resolution the C++ macro triggers.

#[macro_export]
macro_rules! define_unary_ufunc_realhbf16 {
    ($op_name:ident, $fn_float:expr, $fn_double:expr) => {
        pub fn $op_name<'a, 'b>(
            ctx: &mut $crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext,
            in_: &$crate::runtime::core::portable_type::tensor::Tensor,
            out: &'a $crate::runtime::core::portable_type::tensor::Tensor<'b>,
        ) -> &'a $crate::runtime::core::portable_type::tensor::Tensor<'b> {
            $crate::kernels::portable::cpu::pattern::unary_ufunc_realhbf16::unary_ufunc_realhbf16(
                $fn_float, $fn_double, ctx, in_, out,
            )
        }
    };
}

#[macro_export]
macro_rules! define_unary_ufunc_realhbbf16_to_bool {
    ($op_name:ident, $fn_float:expr, $fn_double:expr) => {
        pub fn $op_name<'a, 'b>(
            ctx: &mut $crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext,
            in_: &$crate::runtime::core::portable_type::tensor::Tensor,
            out: &'a $crate::runtime::core::portable_type::tensor::Tensor<'b>,
        ) -> &'a $crate::runtime::core::portable_type::tensor::Tensor<'b> {
            $crate::kernels::portable::cpu::pattern::unary_ufunc_realhbbf16_to_bool::unary_ufunc_realhbbf16_to_bool(
                $fn_float, $fn_double, ctx, in_, out,
            )
        }
    };
}

#[macro_export]
macro_rules! define_unary_ufunc_realhbbf16_to_floathbf16 {
    ($op_name:ident, $fn_float:expr, $fn_double:expr) => {
        pub fn $op_name<'a, 'b>(
            ctx: &mut $crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext,
            in_: &$crate::runtime::core::portable_type::tensor::Tensor,
            out: &'a $crate::runtime::core::portable_type::tensor::Tensor<'b>,
        ) -> &'a $crate::runtime::core::portable_type::tensor::Tensor<'b> {
            $crate::kernels::portable::cpu::pattern::unary_ufunc_realhbbf16_to_floathbf16::unary_ufunc_realhbbf16_to_floathbf16(
                $fn_float, $fn_double, ctx, in_, out,
            )
        }
    };
}

// PORT-NOTE: the C++ `static_cast<float>(val_in)` / `static_cast<double>(val_in)`
// widening conversions and the `static_cast<CTYPE_OUT>(...)` narrowing back are
// applied uniformly over a heterogeneous ctype set (integers, Bool, Half,
// BFloat16, Float, Double). Rust has no single trait covering all these `as`-style
// conversions, so — mirroring math_util.rs's per-type-trait strategy for
// enable_if overload sets — the shared cast traits are defined here and the three
// ufunc modules use them. Each impl reproduces the corresponding `static_cast`.

/// `static_cast<float>(val)` for any ctype accepted by the ufunc patterns.
pub trait AsF32 {
    fn as_f32(self) -> f32;
}
/// `static_cast<double>(val)` for any ctype accepted by the ufunc patterns.
pub trait AsF64 {
    fn as_f64(self) -> f64;
    /// Mirrors `if constexpr (std::is_same_v<CTYPE, double>)`: true only for the
    /// `double` (f64) ctype arm, selecting the double math path.
    const IS_DOUBLE: bool = false;
}
/// `static_cast<Self>(val: f32)` — narrow a float result to the output ctype.
pub trait FromF32 {
    fn from_f32(val: f32) -> Self;
}
/// `static_cast<Self>(val: f64)` — narrow a double result to the output ctype.
pub trait FromF64 {
    fn from_f64(val: f64) -> Self;
}

macro_rules! impl_numeric_casts {
    ($($t:ty),*) => {$(
        impl AsF32 for $t {
            fn as_f32(self) -> f32 { self as f32 }
        }
        impl AsF64 for $t {
            fn as_f64(self) -> f64 { self as f64 }
        }
        impl FromF32 for $t {
            fn from_f32(val: f32) -> Self { val as $t }
        }
        impl FromF64 for $t {
            fn from_f64(val: f64) -> Self { val as $t }
        }
    )*};
}

impl_numeric_casts!(u8, i8, i16, i32, i64, f32);

// f64 gets the same casts but marks IS_DOUBLE so the ufunc closures take the
// `fn_double` path (mirroring `if constexpr (std::is_same_v<CTYPE, double>)`).
impl AsF32 for f64 {
    fn as_f32(self) -> f32 {
        self as f32
    }
}
impl AsF64 for f64 {
    fn as_f64(self) -> f64 {
        self as f64
    }
    const IS_DOUBLE: bool = true;
}
impl FromF32 for f64 {
    fn from_f32(val: f32) -> Self {
        val as f64
    }
}
impl FromF64 for f64 {
    fn from_f64(val: f64) -> Self {
        val as f64
    }
}

// `static_cast<float>(bool)` yields 0.0f/1.0f; `static_cast<bool>(float)` is
// nonzero -> true.
impl AsF32 for bool {
    fn as_f32(self) -> f32 {
        self as i32 as f32
    }
}
impl AsF64 for bool {
    fn as_f64(self) -> f64 {
        self as i32 as f64
    }
}
impl FromF32 for bool {
    fn from_f32(val: f32) -> Self {
        val != 0.0
    }
}
impl FromF64 for bool {
    fn from_f64(val: f64) -> Self {
        val != 0.0
    }
}

impl AsF32 for Half {
    fn as_f32(self) -> f32 {
        self.to_f32()
    }
}
impl AsF64 for Half {
    fn as_f64(self) -> f64 {
        self.to_f64()
    }
}
impl FromF32 for Half {
    fn from_f32(val: f32) -> Self {
        Half::from_f32_const(val)
    }
}
impl FromF64 for Half {
    fn from_f64(val: f64) -> Self {
        Half::from_f64_const(val)
    }
}

impl AsF32 for BFloat16 {
    fn as_f32(self) -> f32 {
        self.to_f32()
    }
}
impl AsF64 for BFloat16 {
    fn as_f64(self) -> f64 {
        self.to_f64()
    }
}
impl FromF32 for BFloat16 {
    fn from_f32(val: f32) -> Self {
        BFloat16::from_f32_const(val)
    }
}
impl FromF64 for BFloat16 {
    fn from_f64(val: f64) -> Self {
        BFloat16::from_f64_const(val)
    }
}
