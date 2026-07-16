//! Literal port of runtime/core/portable_type/bits_types.h.

/// bits1x8 is an uninterpreted dtype of a tensor with 1 bit (packed to byte
/// boundary), without any semantics defined.
// [spec:et:def:bits-types.executorch.runtime.etensor.bits1x8]
#[repr(C, align(1))]
#[derive(Clone, Copy)]
pub struct bits1x8 {
    pub val_: u8,
}

pub type bits1x8_underlying = u8;

impl bits1x8 {
    // [spec:et:def:bits-types.executorch.runtime.etensor.bits1x8.bits1x8-fn]
    // [spec:et:sem:bits-types.executorch.runtime.etensor.bits1x8.bits1x8-fn]
    // PORT-NOTE: C++ `bits1x8() = default` leaves `val_` uninitialized. Rust has
    // no uninitialized-by-default; the byte is left unspecified via MaybeUninit.
    pub fn default_uninit() -> Self {
        bits1x8 {
            val_: unsafe { core::mem::MaybeUninit::uninit().assume_init() },
        }
    }

    pub fn new(val: u8) -> Self {
        bits1x8 { val_: val }
    }
}

/// bits2x4 is an uninterpreted dtype of a tensor with 2 bits (packed to byte
/// boundary), without any semantics defined.
// [spec:et:def:bits-types.executorch.runtime.etensor.bits2x4]
#[repr(C, align(1))]
#[derive(Clone, Copy)]
pub struct bits2x4 {
    pub val_: u8,
}

pub type bits2x4_underlying = u8;

impl bits2x4 {
    // [spec:et:def:bits-types.executorch.runtime.etensor.bits2x4.bits2x4-fn]
    // [spec:et:sem:bits-types.executorch.runtime.etensor.bits2x4.bits2x4-fn]
    // PORT-NOTE: C++ `bits2x4() = default` leaves `val_` uninitialized.
    pub fn default_uninit() -> Self {
        bits2x4 {
            val_: unsafe { core::mem::MaybeUninit::uninit().assume_init() },
        }
    }

    pub fn new(val: u8) -> Self {
        bits2x4 { val_: val }
    }
}

/// bits4x2 is an uninterpreted dtype of a tensor with 4 bits (packed to byte
/// boundary), without any semantics defined.
// [spec:et:def:bits-types.executorch.runtime.etensor.bits4x2]
#[repr(C, align(1))]
#[derive(Clone, Copy)]
pub struct bits4x2 {
    pub val_: u8,
}

pub type bits4x2_underlying = u8;

impl bits4x2 {
    // [spec:et:def:bits-types.executorch.runtime.etensor.bits4x2.bits4x2-fn]
    // [spec:et:sem:bits-types.executorch.runtime.etensor.bits4x2.bits4x2-fn]
    // PORT-NOTE: C++ `bits4x2() = default` leaves `val_` uninitialized.
    pub fn default_uninit() -> Self {
        bits4x2 {
            val_: unsafe { core::mem::MaybeUninit::uninit().assume_init() },
        }
    }

    pub fn new(val: u8) -> Self {
        bits4x2 { val_: val }
    }
}

/// bits8 is an uninterpreted dtype of a tensor with 8 bits, without any
/// semantics defined.
// [spec:et:def:bits-types.executorch.runtime.etensor.bits8]
#[repr(C, align(1))]
#[derive(Clone, Copy)]
pub struct bits8 {
    pub val_: u8,
}

impl bits8 {
    // [spec:et:def:bits-types.executorch.runtime.etensor.bits8.bits8-fn]
    // [spec:et:sem:bits-types.executorch.runtime.etensor.bits8.bits8-fn]
    // PORT-NOTE: C++ `bits8() = default` leaves `val_` uninitialized. Unlike the
    // other bits types, `bits8` declares no `using underlying` alias.
    pub fn default_uninit() -> Self {
        bits8 {
            val_: unsafe { core::mem::MaybeUninit::uninit().assume_init() },
        }
    }

    pub fn new(val: u8) -> Self {
        bits8 { val_: val }
    }
}

/// bits16 is an uninterpreted dtype of a tensor with 16 bits, without any
/// semantics defined.
// [spec:et:def:bits-types.executorch.runtime.etensor.bits16]
#[repr(C, align(2))]
#[derive(Clone, Copy)]
pub struct bits16 {
    pub val_: u16,
}

impl bits16 {
    // [spec:et:def:bits-types.executorch.runtime.etensor.bits16.bits16-fn]
    // [spec:et:sem:bits-types.executorch.runtime.etensor.bits16.bits16-fn]
    // PORT-NOTE: C++ `bits16() = default` leaves `val_` uninitialized. Unlike the
    // other bits types, `bits16` declares no `using underlying` alias.
    pub fn default_uninit() -> Self {
        bits16 {
            val_: unsafe { core::mem::MaybeUninit::uninit().assume_init() },
        }
    }

    pub fn new(val: u16) -> Self {
        bits16 { val_: val }
    }
}

// No C++ test file exists for bits_types.h. These focused tests pin the
// size/alignment invariants each sem rule states for the (compiler-defaulted)
// default constructor, and that the value-carrying `new` round-trips the raw
// bits without interpretation. The default ctor leaves `val_` unspecified, so it
// is not asserted for a value; only its type layout is.
#[cfg(test)]
mod tests {
    use super::*;
    use core::mem::{align_of, size_of};

    // sem: bits1x8 is alignment 1, size 1 byte; underlying = u8.
    // [spec:et:sem:bits-types.executorch.runtime.etensor.bits1x8.bits1x8-fn/test]
    #[test]
    fn bits1x8_layout_and_roundtrip() {
        assert_eq!(size_of::<bits1x8>(), 1);
        assert_eq!(align_of::<bits1x8>(), 1);
        assert_eq!(bits1x8::new(0xAB).val_, 0xAB);
        let _ = bits1x8::default_uninit();
        let _: bits1x8_underlying = 0u8;
    }

    // sem: bits2x4 is alignment 1, size 1 byte; underlying = u8.
    // [spec:et:sem:bits-types.executorch.runtime.etensor.bits2x4.bits2x4-fn/test]
    #[test]
    fn bits2x4_layout_and_roundtrip() {
        assert_eq!(size_of::<bits2x4>(), 1);
        assert_eq!(align_of::<bits2x4>(), 1);
        assert_eq!(bits2x4::new(0x3C).val_, 0x3C);
        let _ = bits2x4::default_uninit();
        let _: bits2x4_underlying = 0u8;
    }

    // sem: bits4x2 is alignment 1, size 1 byte; underlying = u8.
    // [spec:et:sem:bits-types.executorch.runtime.etensor.bits4x2.bits4x2-fn/test]
    #[test]
    fn bits4x2_layout_and_roundtrip() {
        assert_eq!(size_of::<bits4x2>(), 1);
        assert_eq!(align_of::<bits4x2>(), 1);
        assert_eq!(bits4x2::new(0x5A).val_, 0x5A);
        let _ = bits4x2::default_uninit();
        let _: bits4x2_underlying = 0u8;
    }

    // sem: bits8 is alignment 1, size 1 byte; no `underlying` alias.
    // [spec:et:sem:bits-types.executorch.runtime.etensor.bits8.bits8-fn/test]
    #[test]
    fn bits8_layout_and_roundtrip() {
        assert_eq!(size_of::<bits8>(), 1);
        assert_eq!(align_of::<bits8>(), 1);
        assert_eq!(bits8::new(0xFF).val_, 0xFF);
        let _ = bits8::default_uninit();
    }

    // sem: bits16 is alignment 2, size 2 bytes; no `underlying` alias.
    // [spec:et:sem:bits-types.executorch.runtime.etensor.bits16.bits16-fn/test]
    #[test]
    fn bits16_layout_and_roundtrip() {
        assert_eq!(size_of::<bits16>(), 2);
        assert_eq!(align_of::<bits16>(), 2);
        assert_eq!(bits16::new(0xBEEF).val_, 0xBEEF);
        let _ = bits16::default_uninit();
    }
}
