//! Literal port of runtime/core/portable_type/scalar_type.h.
//!
//! Forked from c10/core/ScalarType.h. Everything but the ScalarType definition
//! is in util/ScalarTypeUtil.h. The only critical piece is that the types and
//! indices of the main ScalarType enum line up with ATen, so that serialization
//! is compatible.

/// Placing a bunch of unused dtypes here as our macros don't make it easy to
/// skip scalar types defined in aten that we dont have.
pub mod unused_dtype {
    // [spec:et:def:scalar-type.executorch.runtime.etensor.unused-dtype.float8-e5m2]
    #[repr(C, align(1))]
    #[derive(Clone, Copy)]
    pub struct Float8_e5m2 {
        pub x: u8,
    }

    pub type Float8_e5m2_underlying = u8;

    impl Float8_e5m2 {
        // [spec:et:def:scalar-type.executorch.runtime.etensor.unused-dtype.float8-e5m2.float8-e5m2-fn]
        // [spec:et:sem:scalar-type.executorch.runtime.etensor.unused-dtype.float8-e5m2.float8-e5m2-fn]
        // PORT-NOTE: C++ `= default` leaves `x` uninitialized; placeholder dtype.
        pub fn default_uninit() -> Self {
            Float8_e5m2 {
                x: unsafe { core::mem::MaybeUninit::uninit().assume_init() },
            }
        }

        pub fn new(val: u8) -> Self {
            Float8_e5m2 { x: val }
        }
    }

    // [spec:et:def:scalar-type.executorch.runtime.etensor.unused-dtype.float8-e4m3fn]
    #[repr(C, align(1))]
    #[derive(Clone, Copy)]
    pub struct Float8_e4m3fn {
        pub x: u8,
    }

    pub type Float8_e4m3fn_underlying = u8;

    impl Float8_e4m3fn {
        // [spec:et:def:scalar-type.executorch.runtime.etensor.unused-dtype.float8-e4m3fn.float8-e4m3fn-fn]
        // [spec:et:sem:scalar-type.executorch.runtime.etensor.unused-dtype.float8-e4m3fn.float8-e4m3fn-fn]
        // PORT-NOTE: C++ `= default` leaves `x` uninitialized; placeholder dtype.
        pub fn default_uninit() -> Self {
            Float8_e4m3fn {
                x: unsafe { core::mem::MaybeUninit::uninit().assume_init() },
            }
        }

        pub fn new(val: u8) -> Self {
            Float8_e4m3fn { x: val }
        }
    }

    // [spec:et:def:scalar-type.executorch.runtime.etensor.unused-dtype.float8-e5m2fnuz]
    #[repr(C, align(1))]
    #[derive(Clone, Copy)]
    pub struct Float8_e5m2fnuz {
        pub x: u8,
    }

    pub type Float8_e5m2fnuz_underlying = u8;

    impl Float8_e5m2fnuz {
        // [spec:et:def:scalar-type.executorch.runtime.etensor.unused-dtype.float8-e5m2fnuz.float8-e5m2fnuz-fn]
        // [spec:et:sem:scalar-type.executorch.runtime.etensor.unused-dtype.float8-e5m2fnuz.float8-e5m2fnuz-fn]
        // PORT-NOTE: C++ `= default` leaves `x` uninitialized; placeholder dtype.
        pub fn default_uninit() -> Self {
            Float8_e5m2fnuz {
                x: unsafe { core::mem::MaybeUninit::uninit().assume_init() },
            }
        }

        pub fn new(val: u8) -> Self {
            Float8_e5m2fnuz { x: val }
        }
    }

    // [spec:et:def:scalar-type.executorch.runtime.etensor.unused-dtype.float8-e4m3fnuz]
    #[repr(C, align(1))]
    #[derive(Clone, Copy)]
    pub struct Float8_e4m3fnuz {
        pub x: u8,
    }

    pub type Float8_e4m3fnuz_underlying = u8;

    impl Float8_e4m3fnuz {
        // [spec:et:def:scalar-type.executorch.runtime.etensor.unused-dtype.float8-e4m3fnuz.float8-e4m3fnuz-fn]
        // [spec:et:sem:scalar-type.executorch.runtime.etensor.unused-dtype.float8-e4m3fnuz.float8-e4m3fnuz-fn]
        // PORT-NOTE: C++ `= default` leaves `x` uninitialized; placeholder dtype.
        pub fn default_uninit() -> Self {
            Float8_e4m3fnuz {
                x: unsafe { core::mem::MaybeUninit::uninit().assume_init() },
            }
        }

        pub fn new(val: u8) -> Self {
            Float8_e4m3fnuz { x: val }
        }
    }
}

// The indices and C types must be consistent with
// AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS in c10/core/ScalarType.h. The
// C-type/name pairing that `ET_FORALL_SCALAR_TYPES` expands is preserved by the
// explicit discriminants below (Byte..UInt64 = 0..29).

/// Data types (dtypes) that can be used as element types in ETensors.
// [spec:et:def:scalar-type.executorch.runtime.etensor.scalar-type]
#[repr(i8)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ScalarType {
    Byte = 0,
    Char = 1,
    Short = 2,
    Int = 3,
    Long = 4,
    Half = 5,
    Float = 6,
    Double = 7,
    ComplexHalf = 8,
    ComplexFloat = 9,
    ComplexDouble = 10,
    Bool = 11,
    QInt8 = 12,
    QUInt8 = 13,
    QInt32 = 14,
    BFloat16 = 15,
    QUInt4x2 = 16,
    QUInt2x4 = 17,
    Bits1x8 = 18,
    Bits2x4 = 19,
    Bits4x2 = 20,
    Bits8 = 21,
    Bits16 = 22,
    Float8_e5m2 = 23,
    Float8_e4m3fn = 24,
    Float8_e5m2fnuz = 25,
    Float8_e4m3fnuz = 26,
    UInt16 = 27,
    UInt32 = 28,
    UInt64 = 29,
    /// An explicitly undefined ScalarType. Does not map to any C type.
    Undefined = 30,
    /// The number of ScalarType enumerators.
    NumOptions = 31,
}

// PORT-NOTE: the placeholder float8 dtypes have no dedicated test file; their
// `= default` constructors leave `x` uninitialized, so the observable contract
// is the 1-byte/alignment-1 storage layout and the `new` round-trip. Focused
// unit tests pin those against the sem rules.
#[cfg(test)]
mod tests {
    use super::unused_dtype::*;
    use core::mem::{align_of, size_of};

    // [spec:et:sem:scalar-type.executorch.runtime.etensor.unused-dtype.float8-e5m2.float8-e5m2-fn/test]
    #[test]
    fn scalar_type_test_float8_e5m2_layout() {
        assert_eq!(size_of::<Float8_e5m2>(), 1);
        assert_eq!(align_of::<Float8_e5m2>(), 1);
        assert_eq!(Float8_e5m2::new(0x3C as Float8_e5m2_underlying).x, 0x3C);
        let _ = Float8_e5m2::default_uninit();
    }

    // [spec:et:sem:scalar-type.executorch.runtime.etensor.unused-dtype.float8-e4m3fn.float8-e4m3fn-fn/test]
    #[test]
    fn scalar_type_test_float8_e4m3fn_layout() {
        assert_eq!(size_of::<Float8_e4m3fn>(), 1);
        assert_eq!(align_of::<Float8_e4m3fn>(), 1);
        assert_eq!(Float8_e4m3fn::new(0x48 as Float8_e4m3fn_underlying).x, 0x48);
        let _ = Float8_e4m3fn::default_uninit();
    }

    // [spec:et:sem:scalar-type.executorch.runtime.etensor.unused-dtype.float8-e5m2fnuz.float8-e5m2fnuz-fn/test]
    #[test]
    fn scalar_type_test_float8_e5m2fnuz_layout() {
        assert_eq!(size_of::<Float8_e5m2fnuz>(), 1);
        assert_eq!(align_of::<Float8_e5m2fnuz>(), 1);
        assert_eq!(
            Float8_e5m2fnuz::new(0x7F as Float8_e5m2fnuz_underlying).x,
            0x7F
        );
        let _ = Float8_e5m2fnuz::default_uninit();
    }

    // [spec:et:sem:scalar-type.executorch.runtime.etensor.unused-dtype.float8-e4m3fnuz.float8-e4m3fnuz-fn/test]
    #[test]
    fn scalar_type_test_float8_e4m3fnuz_layout() {
        assert_eq!(size_of::<Float8_e4m3fnuz>(), 1);
        assert_eq!(align_of::<Float8_e4m3fnuz>(), 1);
        assert_eq!(
            Float8_e4m3fnuz::new(0x11 as Float8_e4m3fnuz_underlying).x,
            0x11
        );
        let _ = Float8_e4m3fnuz::default_uninit();
    }
}
