//! Literal port of kernels/portable/cpu/pattern/bitwise_op.h.

use crate::kernels::portable::cpu::scalar_utils::{promote_type_with_scalar, scalar_to};
use crate::kernels::portable::cpu::util::broadcast_util::resize_to_broadcast_target_size;
use crate::kernels::portable::cpu::util::dtype_util::SupportedTensorDtypes;
use crate::kernels::portable::cpu::util::elementwise_util::{
    apply_bitensor_elementwise_fn, apply_unitensor_elementwise_fn, get_compute_type,
};
use crate::runtime::core::error::Error;
use crate::runtime::core::exec_aten::util::scalar_type_util::{can_cast, promote_types};
use crate::runtime::core::exec_aten::util::tensor_util::{
    resize_tensor_same_type, tensors_have_same_dim_order2, tensors_have_same_dim_order3,
};
use crate::runtime::core::portable_type::scalar::Scalar;
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

// PORT-NOTE: the C++ `DEFINE_BINARY_OPERATOR_TEMPLATE` free-function templates
// (`bitwise_and`/`bitwise_or`/`bitwise_xor`) and the `bit_lshift`/`bit_rshift`
// functors are all generic over `T`. Rust has no template-template params and
// `bool` does not implement `Shl`/`Shr`, so the per-`T` bit operations are
// provided by a `BitwiseScalar` trait (one impl per INT-plus-Bool ctype), and
// the functor/free-fn selection is expressed by a `BitwiseOp` trait whose
// generic `apply<T>` reproduces `BitOp<T>()(val_a, val_b)`.
pub trait BitwiseScalar: Copy {
    fn bitwise_and(self, b: Self) -> Self;
    fn bitwise_or(self, b: Self) -> Self;
    fn bitwise_xor(self, b: Self) -> Self;
    fn bit_lshift(self, b: Self) -> Self;
    fn bit_rshift(self, b: Self) -> Self;
}

macro_rules! impl_bitwise_scalar_int {
    ($($t:ty),*) => {$(
        impl BitwiseScalar for $t {
            fn bitwise_and(self, b: Self) -> Self { self & b }
            fn bitwise_or(self, b: Self) -> Self { self | b }
            fn bitwise_xor(self, b: Self) -> Self { self ^ b }
            // [spec:et:def:bitwise-op.operator-fn]
            // [spec:et:sem:bitwise-op.operator-fn]
            // static_cast<T>(lhs << rhs); raw C++ shift semantics — the `as $t`
            // reproduces the truncating `static_cast<T>` on the (possibly wider)
            // shifted value. Shift counts >= bit width / negative follow Rust's
            // wrapping/overflow behavior mirroring the implementation-defined C++.
            fn bit_lshift(self, b: Self) -> Self {
                ((self as i128) << (b as i128)) as $t
            }
            fn bit_rshift(self, b: Self) -> Self {
                ((self as i128) >> (b as i128)) as $t
            }
        }
    )*};
}

impl_bitwise_scalar_int!(u8, i8, i16, i32, i64);

// bool participates in the INT-plus-Bool compute set. C++ integer-promotes bool
// operands for `<<`/`>>` and yields bool via `static_cast<bool>`; `&`/`|`/`^`
// on bool are the boolean operators.
impl BitwiseScalar for bool {
    fn bitwise_and(self, b: Self) -> Self {
        self & b
    }
    fn bitwise_or(self, b: Self) -> Self {
        self | b
    }
    fn bitwise_xor(self, b: Self) -> Self {
        self ^ b
    }
    // [spec:et:def:bitwise-op.operator-fn]
    // [spec:et:sem:bitwise-op.operator-fn]
    fn bit_lshift(self, b: Self) -> Self {
        ((self as i32) << (b as i32)) != 0
    }
    fn bit_rshift(self, b: Self) -> Self {
        ((self as i32) >> (b as i32)) != 0
    }
}

/// The elementwise operation applied by the bitwise patterns, generic over the
/// compute ctype `T` (mirrors the C++ `BitOp<T>` template-template parameter).
pub trait BitwiseOp {
    fn apply<T: BitwiseScalar>(a: T, b: T) -> T;
}

/// `bitwise_and`: `val_a & val_b` (DEFINE_BINARY_OPERATOR_TEMPLATE).
pub struct BitwiseAnd;
impl BitwiseOp for BitwiseAnd {
    fn apply<T: BitwiseScalar>(a: T, b: T) -> T {
        a.bitwise_and(b)
    }
}

/// `bitwise_or`: `val_a | val_b`.
pub struct BitwiseOr;
impl BitwiseOp for BitwiseOr {
    fn apply<T: BitwiseScalar>(a: T, b: T) -> T {
        a.bitwise_or(b)
    }
}

/// `bitwise_xor`: `val_a ^ val_b`.
pub struct BitwiseXor;
impl BitwiseOp for BitwiseXor {
    fn apply<T: BitwiseScalar>(a: T, b: T) -> T {
        a.bitwise_xor(b)
    }
}

/// `bit_lshift`: `static_cast<T>(lhs << rhs)`.
pub struct BitLshift;
impl BitwiseOp for BitLshift {
    fn apply<T: BitwiseScalar>(a: T, b: T) -> T {
        a.bit_lshift(b)
    }
}

/// `bit_rshift`: `static_cast<T>(lhs >> rhs)`.
pub struct BitRshift;
impl BitwiseOp for BitRshift {
    fn apply<T: BitwiseScalar>(a: T, b: T) -> T {
        a.bit_rshift(b)
    }
}

// PORT-NOTE: `Tensor& out` / returned `Tensor&` become `&'a Tensor` (interior
// mutation through `*mut TensorImpl`). `RuntimeContext` in the C++ signature is
// the same type as `KernelRuntimeContext`.
//
// PORT-NOTE (cross-module): the compile-time `op_name` template parameter is
// dropped (the ported `apply_*_elementwise_fn` take no op-name argument); it is
// still passed to the dtype switch as its diagnostic label.
// `support_noncontiguous_tensors` is fixed `true`, matching the C++ pattern's
// `apply_bitensor_elementwise_fn` call. The compute closure receives the
// operands as a slice and returns `CTYPE_COMPUTE`.

pub fn bitwise_tensor_out<'a, 'b, BitOp: BitwiseOp>(
    ctx: &mut KernelRuntimeContext,
    a: &Tensor,
    b: &Tensor,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    // Common Dtype
    let mut common_type = promote_types(a.scalar_type(), b.scalar_type(), false);

    // Check Common Dtype
    crate::et_kernel_check!(
        ctx,
        can_cast(common_type, out.scalar_type()),
        InvalidArgument,
        out
    );

    // Check Dim Order
    crate::et_kernel_check!(
        ctx,
        tensors_have_same_dim_order3(a, b, out),
        InvalidArgument,
        out
    );

    // Resize
    crate::et_kernel_check!(
        ctx,
        resize_to_broadcast_target_size(a, b, out) == Error::Ok,
        InvalidArgument,
        out
    );

    // Compute Dtype
    let compute_type = get_compute_type(&mut common_type);

    crate::et_switch_int_types_and!(
        Bool,
        compute_type,
        ctx,
        "bitwise_tensor_out",
        CTYPE_COMPUTE,
        {
            apply_bitensor_elementwise_fn::<CTYPE_COMPUTE, _>(
                // TODO: rewrite this to be vectorization-capable.
                |vals: &[CTYPE_COMPUTE]| BitOp::apply::<CTYPE_COMPUTE>(vals[0], vals[1]),
                ctx,
                a,
                SupportedTensorDtypes::INTB,
                b,
                SupportedTensorDtypes::INTB,
                out,
                SupportedTensorDtypes::REALHBBF16,
                true,
            );
        }
    );

    out
}

pub fn bitwise_scalar_out<'a, 'b, BitOp: BitwiseOp>(
    ctx: &mut KernelRuntimeContext,
    a: &Tensor,
    b: &Scalar,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    // Common Dtype
    let mut common_type = promote_type_with_scalar(a.scalar_type(), *b, false);

    // Check Common Dtype
    crate::et_kernel_check!(
        ctx,
        can_cast(common_type, out.scalar_type()),
        InvalidArgument,
        out
    );

    // Check Dim Order
    crate::et_kernel_check!(
        ctx,
        tensors_have_same_dim_order2(a, out),
        InvalidArgument,
        out
    );

    // Resize
    crate::et_kernel_check!(
        ctx,
        resize_tensor_same_type(out, a.sizes()) == Error::Ok,
        InvalidArgument,
        out
    );

    // Compute Dtype
    let compute_type = get_compute_type(&mut common_type);

    crate::et_switch_int_types_and!(
        Bool,
        compute_type,
        ctx,
        "bitwise_scalar_out",
        CTYPE_COMPUTE,
        {
            let val_b: CTYPE_COMPUTE = scalar_to::<CTYPE_COMPUTE>(b);
            apply_unitensor_elementwise_fn::<CTYPE_COMPUTE, _>(
                |vals: &[CTYPE_COMPUTE]| {
                    // TODO: rewrite this to be vectorization-capable.
                    BitOp::apply::<CTYPE_COMPUTE>(vals[0], val_b)
                },
                ctx,
                a,
                SupportedTensorDtypes::INTB,
                out,
                SupportedTensorDtypes::REALHBBF16,
                true,
            );
        }
    );

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    // The `bit_lshift`/`bit_rshift` functor `operator()` is
    // `static_cast<T>(lhs << rhs)`: the shift happens on the integer-promoted
    // value and the cast truncates back to T. Pins the promoted-then-truncated
    // behavior for narrow signed/unsigned types and the bool specialization
    // (promote to int, shift, `!= 0`).
    // [spec:et:sem:bitwise-op.operator-fn/test]
    #[test]
    fn bitwise_op_shift_functors_match_cpp_semantics() {
        // i8: 1 << 7 promotes to int, shifts to 128, static_cast<int8_t> wraps
        // to -128.
        assert_eq!(BitLshift::apply::<i8>(1, 7), -128i8);
        // u8: 0x81 << 1 promotes to 258, truncates back to 0x02.
        assert_eq!(BitLshift::apply::<u8>(0x81, 1), 0x02);
        // Signed right shift is arithmetic: -128 >> 7 == -1.
        assert_eq!(BitRshift::apply::<i8>(-128, 7), -1i8);
        // Unsigned right shift through promotion: 0x80 >> 7 == 1.
        assert_eq!(BitRshift::apply::<u8>(0x80, 7), 1);
        // Wide types shift in-type.
        assert_eq!(BitLshift::apply::<i64>(1, 40), 1i64 << 40);
        assert_eq!(BitRshift::apply::<i32>(-256, 4), -16);
        // bool: promoted (1 << 1) == 2 -> true; (1 >> 1) == 0 -> false.
        assert!(BitLshift::apply::<bool>(true, true));
        assert!(!BitRshift::apply::<bool>(true, true));
        assert!(!BitLshift::apply::<bool>(false, true));
    }
}
