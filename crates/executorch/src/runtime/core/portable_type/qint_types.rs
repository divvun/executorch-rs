//! Literal port of runtime/core/portable_type/qint_types.h.

/// qint8 is for signed 8 bit quantized Tensors
// [spec:et:def:qint-types.executorch.runtime.etensor.qint8]
#[repr(C, align(1))]
#[derive(Clone, Copy)]
pub struct qint8 {
    pub val_: i8,
}

pub type qint8_underlying = i8;

impl qint8 {
    // [spec:et:def:qint-types.executorch.runtime.etensor.qint8.qint8-fn]
    // [spec:et:sem:qint-types.executorch.runtime.etensor.qint8.qint8-fn]
    // PORT-NOTE: C++ `qint8() = default` leaves `val_` uninitialized.
    pub fn default_uninit() -> Self {
        qint8 {
            val_: unsafe { core::mem::MaybeUninit::uninit().assume_init() },
        }
    }

    pub fn new(val: i8) -> Self {
        qint8 { val_: val }
    }
}

/// quint8 is for unsigned 8 bit quantized Tensors
// [spec:et:def:qint-types.executorch.runtime.etensor.quint8]
#[repr(C, align(1))]
#[derive(Clone, Copy)]
pub struct quint8 {
    pub val_: u8,
}

pub type quint8_underlying = u8;

impl quint8 {
    // [spec:et:def:qint-types.executorch.runtime.etensor.quint8.quint8-fn]
    // [spec:et:sem:qint-types.executorch.runtime.etensor.quint8.quint8-fn]
    // PORT-NOTE: C++ `quint8() = default` leaves `val_` uninitialized.
    pub fn default_uninit() -> Self {
        quint8 {
            val_: unsafe { core::mem::MaybeUninit::uninit().assume_init() },
        }
    }

    pub fn new(val: u8) -> Self {
        quint8 { val_: val }
    }
}

/// qint32 is for signed 32 bit quantized Tensors
// [spec:et:def:qint-types.executorch.runtime.etensor.qint32]
#[repr(C, align(4))]
#[derive(Clone, Copy)]
pub struct qint32 {
    pub val_: i32,
}

pub type qint32_underlying = i32;

impl qint32 {
    // [spec:et:def:qint-types.executorch.runtime.etensor.qint32.qint32-fn]
    // [spec:et:sem:qint-types.executorch.runtime.etensor.qint32.qint32-fn]
    // PORT-NOTE: C++ `qint32() = default` leaves `val_` uninitialized.
    pub fn default_uninit() -> Self {
        qint32 {
            val_: unsafe { core::mem::MaybeUninit::uninit().assume_init() },
        }
    }

    pub fn new(val: i32) -> Self {
        qint32 { val_: val }
    }
}

/// quint4x2 is for un-signed 4 bit quantized Tensors that are packed to byte
/// boundary.
// [spec:et:def:qint-types.executorch.runtime.etensor.quint4x2]
#[repr(C, align(1))]
#[derive(Clone, Copy)]
pub struct quint4x2 {
    pub val_: u8,
}

pub type quint4x2_underlying = u8;

impl quint4x2 {
    // [spec:et:def:qint-types.executorch.runtime.etensor.quint4x2.quint4x2-fn]
    // [spec:et:sem:qint-types.executorch.runtime.etensor.quint4x2.quint4x2-fn]
    // PORT-NOTE: C++ `quint4x2() = default` leaves `val_` uninitialized.
    pub fn default_uninit() -> Self {
        quint4x2 {
            val_: unsafe { core::mem::MaybeUninit::uninit().assume_init() },
        }
    }

    pub fn new(val: u8) -> Self {
        quint4x2 { val_: val }
    }
}

/// quint2x4 is for un-signed 2 bit quantized Tensors that are packed to byte
/// boundary.
// [spec:et:def:qint-types.executorch.runtime.etensor.quint2x4]
#[repr(C, align(1))]
#[derive(Clone, Copy)]
pub struct quint2x4 {
    pub val_: u8,
}

pub type quint2x4_underlying = u8;

impl quint2x4 {
    // [spec:et:def:qint-types.executorch.runtime.etensor.quint2x4.quint2x4-fn]
    // [spec:et:sem:qint-types.executorch.runtime.etensor.quint2x4.quint2x4-fn]
    // PORT-NOTE: C++ `quint2x4() = default` leaves `val_` uninitialized.
    pub fn default_uninit() -> Self {
        quint2x4 {
            val_: unsafe { core::mem::MaybeUninit::uninit().assume_init() },
        }
    }

    pub fn new(val: u8) -> Self {
        quint2x4 { val_: val }
    }
}

// PORT-NOTE: the C++ qint_types have no dedicated test file; the `= default`
// default constructors leave `val_` uninitialized, so their only observable
// contract is the storage type's structural invariants (size / alignment /
// underlying element type). These focused unit tests pin those against the sem
// rules in docs/spec/port/runtime/core/portable_type/qint_types.md.
#[cfg(test)]
mod tests {
    use super::*;
    use core::mem::{align_of, size_of};

    // [spec:et:sem:qint-types.executorch.runtime.etensor.qint8.qint8-fn/test]
    #[test]
    fn qint_types_test_qint8_layout() {
        // signed 8-bit: size 1, alignment 1, underlying int8_t.
        assert_eq!(size_of::<qint8>(), 1);
        assert_eq!(align_of::<qint8>(), 1);
        let q = qint8::new(-5 as qint8_underlying);
        assert_eq!(q.val_, -5);
        let _ = qint8::default_uninit();
    }

    // [spec:et:sem:qint-types.executorch.runtime.etensor.quint8.quint8-fn/test]
    #[test]
    fn qint_types_test_quint8_layout() {
        // unsigned 8-bit: size 1, alignment 1, underlying uint8_t.
        assert_eq!(size_of::<quint8>(), 1);
        assert_eq!(align_of::<quint8>(), 1);
        let q = quint8::new(200 as quint8_underlying);
        assert_eq!(q.val_, 200);
        let _ = quint8::default_uninit();
    }

    // [spec:et:sem:qint-types.executorch.runtime.etensor.qint32.qint32-fn/test]
    #[test]
    fn qint_types_test_qint32_layout() {
        // signed 32-bit: size 4, alignment 4, underlying int32_t.
        assert_eq!(size_of::<qint32>(), 4);
        assert_eq!(align_of::<qint32>(), 4);
        let q = qint32::new(-100000 as qint32_underlying);
        assert_eq!(q.val_, -100000);
        let _ = qint32::default_uninit();
    }

    // [spec:et:sem:qint-types.executorch.runtime.etensor.quint4x2.quint4x2-fn/test]
    #[test]
    fn qint_types_test_quint4x2_layout() {
        // packed 4-bit x2: byte storage, size 1, alignment 1, underlying uint8_t.
        assert_eq!(size_of::<quint4x2>(), 1);
        assert_eq!(align_of::<quint4x2>(), 1);
        let q = quint4x2::new(0xAB as quint4x2_underlying);
        assert_eq!(q.val_, 0xAB);
        let _ = quint4x2::default_uninit();
    }

    // [spec:et:sem:qint-types.executorch.runtime.etensor.quint2x4.quint2x4-fn/test]
    #[test]
    fn qint_types_test_quint2x4_layout() {
        // packed 2-bit x4: byte storage, size 1, alignment 1, underlying uint8_t.
        assert_eq!(size_of::<quint2x4>(), 1);
        assert_eq!(align_of::<quint2x4>(), 1);
        let q = quint2x4::new(0xCD as quint2x4_underlying);
        assert_eq!(q.val_, 0xCD);
        let _ = quint2x4::default_uninit();
    }
}
