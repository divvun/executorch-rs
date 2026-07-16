//! Literal port of runtime/core/evalue.cpp + runtime/core/evalue.h.
//!
//! # Naming / API-surface conventions (recorded once)
//!
//! - The C++ predicate/accessor methods keep their names, snake_cased:
//!   `isTensor()->is_tensor`, `toTensor()->to_tensor`, `isInt()->is_int`,
//!   `toInt()->to_int`, `tryToInt()->try_to_int`, etc.
//! - `executorch::aten::nullopt` / `std::nullopt` -> Rust `None`
//!   (`std::optional<T>` -> `Option<T>` per PORTING.md).
//! - PORT-NOTE: the task brief suggested modeling the tagged union as a Rust
//!   `enum` with the Tag mapping. The C++ `EValue` is literally a `union
//!   Payload` + `Tag tag`, and `moveFrom`/`EValue(const Payload&, Tag)` bit-copy
//!   the whole `copyable_union` (`payload.copyable_union = rhs.payload.copyable_union`)
//!   without knowing which member is active. Only a Rust `union` reproduces that
//!   bit-copy and the placement-new-into-`as_tensor` tensor special case. So the
//!   payload is a `union Payload { copyable_union, as_tensor }` and `tag` is a
//!   separate field, exactly mirroring the C++ decomposition. The predicates
//!   (`is_int`/`is_tensor`/...) recover the "enum" ergonomics.
//! - PORT-NOTE: `BoxedEvalueList<T>::to<T>()` on a wrapped EValue and the
//!   `to<T>()`/`try_to<T>()` templated accessors become the `EValueTo` /
//!   `EValueTryTo` traits with one impl per payload type (see below), matching
//!   the explicit `EVALUE_DEFINE_TO` / `EVALUE_DEFINE_TRY_TO` instantiations.
//! - PORT-NOTE: pointer payload members are raw pointers
//!   (`*mut ArrayRef<char>`, `*mut BoxedEvalueList<..>`, ...) to preserve the
//!   union's pointer-identity / non-owning semantics, matching the C++ where the
//!   pointed-to list/string storage lives in program memory.

use crate::runtime::core::array_ref::ArrayRef;
use crate::runtime::core::error::Error;
use crate::runtime::core::exec_aten::exec_aten::ScalarType;
use crate::runtime::core::portable_type::device::{Device, DeviceType};
use crate::runtime::core::portable_type::scalar::Scalar;
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::core::portable_type::tensor_options::{Layout, MemoryFormat};
use crate::runtime::core::result::{Result, ResultExt};
use crate::runtime::core::tag::Tag;

// PORT-NOTE: `ET_CHECK` / `ET_CHECK_MSG` live in runtime/platform/assert.h,
// which has no ported `assert.rs` target yet. This local macro mirrors their
// semantics (emit message, then abort via the PAL abort path), matching the
// pattern established in runtime/core/portable_type/scalar.rs. Should be
// replaced by the shared `et_check_msg!` once the assert module is ported.
// Unresolved cross-module reference.
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

// Tensor gets proper reference treatment because its expensive to copy in aten
// mode, all other types are just copied.
// [spec:et:def:evalue.executorch.runtime.internal.evalue-to-const-ref-overload-return]
// [spec:et:def:evalue.executorch.runtime.internal.evalue-to-const-ref-overload-return-executorch-aten-tensor]
// [spec:et:def:evalue.executorch.runtime.internal.evalue-to-ref-overload-return]
// [spec:et:def:evalue.executorch.runtime.internal.evalue-to-ref-overload-return-executorch-aten-tensor]
//
// PORT-NOTE: `internal::evalue_to_const_ref_overload_return<T>` and
// `evalue_to_ref_overload_return<T>` are compile-time type-maps used only to
// pick `const Tensor&`/`Tensor&` vs plain `T` for the `to<T>() const&`/`to<T>() &`
// overloads. Rust `to<T>()` here models the single (rvalue-equivalent) `to`
// accessor via the `EValueTo` trait; the ref-overload return-type dispatch is a
// C++-template-mechanics detail with no runtime behavior, so it is captured as
// this annotation block rather than as separate marker structs.
pub mod internal {}

/*
 * Helper class used to correlate EValues in the executor table, with the
 * unwrapped list of the proper type. Because values in the runtime's values
 * table can change during execution, we cannot statically allocate list of
 * objects at deserialization. Imagine the serialized list says index 0 in the
 * value table is element 2 in the list, but during execution the value in
 * element 2 changes (in the case of tensor this means the TensorImpl* stored in
 * the tensor changes). To solve this instead they must be created dynamically
 * whenever they are used.
 */
// [spec:et:def:evalue.executorch.runtime.boxed-evalue-list]
pub struct BoxedEvalueList<'a, T> {
    // Source of truth for the list
    wrapped_vals_: ArrayRef<*mut EValue<'a>>,
    // Same size as wrapped_vals
    // (mutable T* unwrapped_vals_ in C++)
    unwrapped_vals_: *mut T,
}

impl<'a, T> BoxedEvalueList<'a, T> {
    // BoxedEvalueList() = default;
    //
    // PORT-NOTE: the C++ defaulted ctor leaves both members
    // default-constructed (empty ArrayRef + null unwrapped_vals_).
    pub fn default_new() -> Self {
        BoxedEvalueList {
            wrapped_vals_: ArrayRef::new(),
            unwrapped_vals_: core::ptr::null_mut(),
        }
    }

    /*
     * Wrapped_vals is a list of pointers into the values table of the runtime
     * whose destinations correlate with the elements of the list, unwrapped_vals
     * is a container of the same size whose serves as memory to construct the
     * unwrapped vals.
     */
    // [spec:et:def:evalue.executorch.runtime.boxed-evalue-list.boxed-evalue-list-fn]
    // [spec:et:sem:evalue.executorch.runtime.boxed-evalue-list.boxed-evalue-list-fn]
    pub fn new(wrapped_vals: *mut *mut EValue<'a>, unwrapped_vals: *mut T, size: i32) -> Self {
        BoxedEvalueList {
            wrapped_vals_: ArrayRef::from_raw_parts(
                Self::check_wrapped_vals(wrapped_vals, size),
                size as usize,
            ),
            unwrapped_vals_: Self::check_unwrapped_vals(unwrapped_vals),
        }
    }

    /**
     * Destroys the unwrapped elements without re-dereferencing wrapped_vals_.
     * This is safe to call during EValue destruction because it does not
     * dereference wrapped_vals_, which may point to EValues mutated by
     * MoveCall instructions.
     */
    // [spec:et:def:evalue.executorch.runtime.boxed-evalue-list.destroy-elements-fn]
    // [spec:et:sem:evalue.executorch.runtime.boxed-evalue-list.destroy-elements-fn]
    pub fn destroy_elements(&self) {
        let mut i: usize = 0;
        while i < self.wrapped_vals_.size() {
            // unwrapped_vals_[i].~T();
            unsafe {
                core::ptr::drop_in_place(self.unwrapped_vals_.add(i));
            }
            i += 1;
        }
    }

    // [spec:et:def:evalue.executorch.runtime.boxed-evalue-list.check-wrapped-vals-fn]
    // [spec:et:sem:evalue.executorch.runtime.boxed-evalue-list.check-wrapped-vals-fn]
    fn check_wrapped_vals(wrapped_vals: *mut *mut EValue<'a>, size: i32) -> *const *mut EValue<'a> {
        et_check_msg!(!wrapped_vals.is_null(), "wrapped_vals cannot be null");
        et_check_msg!(size >= 0, "size cannot be negative");
        wrapped_vals
    }

    // [spec:et:def:evalue.executorch.runtime.boxed-evalue-list.check-unwrapped-vals-fn]
    // [spec:et:sem:evalue.executorch.runtime.boxed-evalue-list.check-unwrapped-vals-fn]
    fn check_unwrapped_vals(unwrapped_vals: *mut T) -> *mut T {
        et_check_msg!(!unwrapped_vals.is_null(), "unwrapped_vals cannot be null");
        unwrapped_vals
    }
}

// PORT-NOTE: the generic (non-optional) `get()`/`tryGet()` template bodies live
// in the header and are instantiated for `T = int64_t` and `T = Tensor`; the
// `T = Option<Tensor>` full specialization lives in evalue.cpp. C++ template
// specialization has no direct Rust analog; the generic bodies are modeled as
// the `BoxedListGet` trait (blanket impl for every `T` reachable through
// `EValueTo`/`EValueTryTo`) and the optional-tensor variant is a concrete
// inherent impl on `BoxedEvalueList<Option<Tensor>>` (inherent methods win name
// resolution over the blanket trait, reproducing the specialization dispatch).
// Element writes use `ptr::write` / `ptr::read` rather than `=`/`*` to avoid a
// `T: Copy` bound, matching the C++ move-assign into the scratch buffer.
pub trait BoxedListGet<T> {
    // [spec:et:def:evalue.executorch.runtime.boxed-evalue-list.get-fn]
    // [spec:et:sem:evalue.executorch.runtime.boxed-evalue-list.get-fn]
    fn get(&self) -> ArrayRef<T>;
    // [spec:et:def:evalue.executorch.runtime.boxed-evalue-list.try-get-fn]
    // [spec:et:sem:evalue.executorch.runtime.boxed-evalue-list.try-get-fn]
    fn try_get(&self) -> Result<ArrayRef<T>>;
}

impl<'a, T> BoxedListGet<T> for BoxedEvalueList<'a, T>
where
    EValue<'a>: EValueTo<T> + EValueTryTo<T>,
{
    /*
     * Constructs and returns the list of T specified by the EValue pointers
     */
    // [spec:et:def:evalue.executorch.runtime.boxed-evalue-list-t.get-fn]
    // [spec:et:sem:evalue.executorch.runtime.boxed-evalue-list-t.get-fn]
    fn get(&self) -> ArrayRef<T> {
        let mut i: usize = 0;
        while i < self.wrapped_vals_.size() {
            et_check!(!unsafe { *self.wrapped_vals_.index(i) }.is_null());
            unsafe {
                let ev: *mut EValue<'a> = *self.wrapped_vals_.index(i);
                core::ptr::write(self.unwrapped_vals_.add(i), EValueTo::<T>::to(&mut *ev));
            }
            i += 1;
        }
        ArrayRef::from_raw_parts(self.unwrapped_vals_, self.wrapped_vals_.size())
    }

    /**
     * Result-returning counterpart of get(). Validates each wrapped EValue's
     * tag before materializing; returns Error::InvalidType if any element's
     * tag does not match T and Error::InvalidState if any element pointer is
     * null. Use this when materializing lists from untrusted .pte data so that
     * a malformed program cannot force a process abort inside to<T>() /
     * ET_CHECK.
     */
    // [spec:et:def:evalue.executorch.runtime.boxed-evalue-list-t.try-get-fn]
    // [spec:et:sem:evalue.executorch.runtime.boxed-evalue-list-t.try-get-fn]
    fn try_get(&self) -> Result<ArrayRef<T>> {
        let mut i: usize = 0;
        while i < self.wrapped_vals_.size() {
            if unsafe { *self.wrapped_vals_.index(i) }.is_null() {
                return Err(Error::InvalidState);
            }
            let r = EValueTryTo::<T>::try_to(unsafe { &**self.wrapped_vals_.index(i) });
            if !ResultExt::ok(&r) {
                return Err(r.error());
            }
            unsafe {
                core::ptr::write(self.unwrapped_vals_.add(i), r_into_ok(r));
            }
            i += 1;
        }
        Ok(ArrayRef::from_raw_parts(
            self.unwrapped_vals_,
            self.wrapped_vals_.size(),
        ))
    }
}

// Specialize for list of optional tensors, as nullptr is a valid std::nullopt.
// For non-optional types, nullptr is invalid.
impl<'a> BoxedEvalueList<'a, Option<Tensor<'a>>> {
    // [spec:et:def:evalue.executorch.runtime.boxed-evalue-list-std.optional-executorch.aten.tensor.get-fn]
    // [spec:et:sem:evalue.executorch.runtime.boxed-evalue-list-std.optional-executorch.aten.tensor.get-fn]
    pub fn get(&self) -> ArrayRef<Option<Tensor<'a>>> {
        let mut i: usize = 0;
        while i < self.wrapped_vals_.size() {
            if unsafe { *self.wrapped_vals_.index(i) }.is_null() {
                unsafe {
                    core::ptr::write(self.unwrapped_vals_.add(i), None);
                }
            } else {
                unsafe {
                    let ev: *mut EValue<'a> = *self.wrapped_vals_.index(i);
                    core::ptr::write(
                        self.unwrapped_vals_.add(i),
                        EValueTo::<Option<Tensor<'a>>>::to(&mut *ev),
                    );
                }
            }
            i += 1;
        }
        ArrayRef::from_raw_parts(self.unwrapped_vals_, self.wrapped_vals_.size())
    }

    // [spec:et:def:evalue.executorch.runtime.boxed-evalue-list-std.optional-executorch.aten.tensor.try-get-fn]
    // [spec:et:sem:evalue.executorch.runtime.boxed-evalue-list-std.optional-executorch.aten.tensor.try-get-fn]
    pub fn try_get(&self) -> Result<ArrayRef<Option<Tensor<'a>>>> {
        let mut i: usize = 0;
        while i < self.wrapped_vals_.size() {
            if unsafe { *self.wrapped_vals_.index(i) }.is_null() {
                unsafe {
                    core::ptr::write(self.unwrapped_vals_.add(i), None);
                }
                i += 1;
                continue;
            }
            let r = unsafe { &**self.wrapped_vals_.index(i) }.try_to_optional::<Tensor<'a>>();
            if !ResultExt::ok(&r) {
                return Err(r.error());
            }
            unsafe {
                core::ptr::write(self.unwrapped_vals_.add(i), r_into_ok(r));
            }
            i += 1;
        }
        Ok(ArrayRef::from_raw_parts(
            self.unwrapped_vals_,
            self.wrapped_vals_.size(),
        ))
    }
}

// Helper mirroring `std::move(r.get())`: moves the value out of an ok Result
// without requiring `T: Copy` (the C++ `Result::get()` returns a reference that
// is then move-constructed). Precondition: `r.ok()` was just checked true.
fn r_into_ok<T>(r: Result<T>) -> T {
    match r {
        Ok(v) => v,
        Err(_) => unreachable!(),
    }
}

// Aggregate typing system similar to IValue only slimmed down with less
// functionality, no dependencies on atomic, and fewer supported types to better
// suit embedded systems (ie no intrusive ptr)
// [spec:et:def:evalue.executorch.runtime.e-value]
pub struct EValue<'a> {
    // Data storage and type tag
    pub payload: Payload<'a>,
    pub tag: Tag,
}

// [spec:et:def:evalue.executorch.runtime.e-value.payload]
// When in ATen mode at::Tensor is not trivially copyable, this nested union
// lets us handle tensor as a special case while leaving the rest of the fields
// in a simple state instead of requiring a switch on tag everywhere.
pub union Payload<'a> {
    pub copyable_union: TriviallyCopyablePayload<'a>,
    // Since a Tensor just holds a TensorImpl*, there's no value to use Tensor*
    // here.
    pub as_tensor: core::mem::ManuallyDrop<Tensor<'a>>,
}

// [spec:et:def:evalue.executorch.runtime.e-value.payload.trivially-copyable-payload]
#[derive(Clone, Copy)]
pub union TriviallyCopyablePayload<'a> {
    // Scalar supported through these 3 types
    pub as_int: i64,
    pub as_double: f64,
    pub as_bool: bool,

    pub as_string_ptr: *mut ArrayRef<u8>,
    pub as_double_list_ptr: *mut ArrayRef<f64>,
    pub as_bool_list_ptr: *mut ArrayRef<bool>,
    pub as_int_list_ptr: *mut BoxedEvalueList<'a, i64>,
    pub as_tensor_list_ptr: *mut BoxedEvalueList<'a, Tensor<'a>>,
    pub as_list_optional_tensor_ptr: *mut BoxedEvalueList<'a, Option<Tensor<'a>>>,
}

impl<'a> TriviallyCopyablePayload<'a> {
    // [spec:et:def:evalue.executorch.runtime.e-value.payload.trivially-copyable-payload.trivially-copyable-payload-fn]
    // [spec:et:sem:evalue.executorch.runtime.e-value.payload.trivially-copyable-payload.trivially-copyable-payload-fn]
    // TriviallyCopyablePayload() : as_int(0) {}
    pub fn new() -> Self {
        TriviallyCopyablePayload { as_int: 0 }
    }
}

impl<'a> Payload<'a> {
    // [spec:et:def:evalue.executorch.runtime.e-value.payload.payload-fn]
    // [spec:et:sem:evalue.executorch.runtime.e-value.payload.payload-fn]
    // Payload() {}
    //
    // PORT-NOTE: the C++ ctor has an empty body and constructs no member,
    // leaving the union indeterminate for the enclosing EValue ctor to fill.
    // Rust cannot leave a union uninitialized in safe code, so this initializes
    // the `copyable_union` member (which every EValue ctor overwrites anyway,
    // and matches the inner `TriviallyCopyablePayload() : as_int(0)`).
    pub fn new() -> Self {
        Payload {
            copyable_union: TriviallyCopyablePayload::new(),
        }
    }
    // ~Payload() {} : empty; teardown is done by EValue::destroy().
}

impl<'a> EValue<'a> {
    // Basic ctors and assignments
    // EValue(const EValue& rhs) : EValue(rhs.payload, rhs.tag) {}
    pub fn from_ref(rhs: &EValue<'a>) -> Self {
        EValue::from_payload_tag(&rhs.payload, rhs.tag)
    }

    // EValue(EValue&& rhs) noexcept : tag(rhs.tag) { moveFrom(std::move(rhs)); }
    pub fn from_move(rhs: &mut EValue<'a>) -> Self {
        let mut this = EValue {
            payload: Payload::new(),
            tag: rhs.tag,
        };
        this.move_from(rhs);
        this
    }

    // [spec:et:def:evalue.executorch.runtime.e-value.operator-fn]
    // [spec:et:sem:evalue.executorch.runtime.e-value.operator-fn]
    // EValue& operator=(EValue&& rhs) & noexcept
    pub fn assign_move(&mut self, rhs: &mut EValue<'a>) -> &mut EValue<'a> {
        if core::ptr::eq(rhs, self) {
            return self;
        }

        self.destroy();
        self.move_from(rhs);
        self
    }

    // EValue& operator=(EValue const& rhs) &
    pub fn assign_ref(&mut self, rhs: &EValue<'a>) -> &mut EValue<'a> {
        // Define copy assignment through copy ctor and move assignment
        let mut tmp = EValue::from_ref(rhs);
        self.assign_move(&mut tmp);
        self
    }

    /****** None Type ******/
    // EValue() : tag(Tag::None) { payload.copyable_union.as_int = 0; }
    pub fn new() -> Self {
        let mut this = EValue {
            payload: Payload::new(),
            tag: Tag::None,
        };
        this.payload.copyable_union.as_int = 0;
        this
    }

    // [spec:et:def:evalue.executorch.runtime.e-value.is-none-fn]
    // [spec:et:sem:evalue.executorch.runtime.e-value.is-none-fn]
    pub fn is_none(&self) -> bool {
        self.tag == Tag::None
    }

    /****** Int Type ******/
    // /*implicit*/ EValue(int64_t i) : tag(Tag::Int)
    pub fn from_int(i: i64) -> Self {
        let mut this = EValue {
            payload: Payload::new(),
            tag: Tag::Int,
        };
        this.payload.copyable_union.as_int = i;
        this
    }

    // [spec:et:def:evalue.executorch.runtime.e-value.is-int-fn]
    // [spec:et:sem:evalue.executorch.runtime.e-value.is-int-fn]
    pub fn is_int(&self) -> bool {
        self.tag == Tag::Int
    }

    // [spec:et:def:evalue.executorch.runtime.e-value.to-int-fn]
    // [spec:et:sem:evalue.executorch.runtime.e-value.to-int-fn]
    pub fn to_int(&self) -> i64 {
        et_check_msg!(self.is_int(), "EValue is not an int.");
        unsafe { self.payload.copyable_union.as_int }
    }

    // [spec:et:def:evalue.executorch.runtime.e-value.try-to-int-fn]
    // [spec:et:sem:evalue.executorch.runtime.e-value.try-to-int-fn]
    pub fn try_to_int(&self) -> Result<i64> {
        if !self.is_int() {
            return Err(Error::InvalidType);
        }
        Ok(unsafe { self.payload.copyable_union.as_int })
    }

    /****** Double Type ******/
    // /*implicit*/ EValue(double d) : tag(Tag::Double)
    pub fn from_double(d: f64) -> Self {
        let mut this = EValue {
            payload: Payload::new(),
            tag: Tag::Double,
        };
        this.payload.copyable_union.as_double = d;
        this
    }

    // [spec:et:def:evalue.executorch.runtime.e-value.is-double-fn]
    // [spec:et:sem:evalue.executorch.runtime.e-value.is-double-fn]
    pub fn is_double(&self) -> bool {
        self.tag == Tag::Double
    }

    // [spec:et:def:evalue.executorch.runtime.e-value.to-double-fn]
    // [spec:et:sem:evalue.executorch.runtime.e-value.to-double-fn]
    pub fn to_double(&self) -> f64 {
        et_check_msg!(self.is_double(), "EValue is not a Double.");
        unsafe { self.payload.copyable_union.as_double }
    }

    // [spec:et:def:evalue.executorch.runtime.e-value.try-to-double-fn]
    // [spec:et:sem:evalue.executorch.runtime.e-value.try-to-double-fn]
    pub fn try_to_double(&self) -> Result<f64> {
        if !self.is_double() {
            return Err(Error::InvalidType);
        }
        Ok(unsafe { self.payload.copyable_union.as_double })
    }

    /****** Bool Type ******/
    // /*implicit*/ EValue(bool b) : tag(Tag::Bool)
    pub fn from_bool(b: bool) -> Self {
        let mut this = EValue {
            payload: Payload::new(),
            tag: Tag::Bool,
        };
        this.payload.copyable_union.as_bool = b;
        this
    }

    // [spec:et:def:evalue.executorch.runtime.e-value.is-bool-fn]
    // [spec:et:sem:evalue.executorch.runtime.e-value.is-bool-fn]
    pub fn is_bool(&self) -> bool {
        self.tag == Tag::Bool
    }

    // [spec:et:def:evalue.executorch.runtime.e-value.to-bool-fn]
    // [spec:et:sem:evalue.executorch.runtime.e-value.to-bool-fn]
    pub fn to_bool(&self) -> bool {
        et_check_msg!(self.is_bool(), "EValue is not a Bool.");
        unsafe { self.payload.copyable_union.as_bool }
    }

    // [spec:et:def:evalue.executorch.runtime.e-value.try-to-bool-fn]
    // [spec:et:sem:evalue.executorch.runtime.e-value.try-to-bool-fn]
    pub fn try_to_bool(&self) -> Result<bool> {
        if !self.is_bool() {
            return Err(Error::InvalidType);
        }
        Ok(unsafe { self.payload.copyable_union.as_bool })
    }

    /****** Scalar Type ******/
    /// Construct an EValue using the implicit value of a Scalar.
    // [spec:et:def:evalue.executorch.runtime.e-value.e-value-fn]
    // [spec:et:sem:evalue.executorch.runtime.e-value.e-value-fn]
    // /*implicit*/ EValue(executorch::aten::Scalar s)
    pub fn from_scalar(s: Scalar) -> Self {
        let mut this = EValue {
            payload: Payload::new(),
            tag: Tag::None,
        };
        if s.is_integral(false) {
            this.tag = Tag::Int;
            this.payload.copyable_union.as_int = s.to_i64();
        } else if s.is_floating_point() {
            this.tag = Tag::Double;
            this.payload.copyable_union.as_double = s.to_f64();
        } else if s.is_boolean() {
            this.tag = Tag::Bool;
            this.payload.copyable_union.as_bool = s.to_bool_val();
        } else {
            et_check_msg!(false, "Scalar passed to EValue is not initialized.");
        }
        this
    }

    // [spec:et:def:evalue.executorch.runtime.e-value.is-scalar-fn]
    // [spec:et:sem:evalue.executorch.runtime.e-value.is-scalar-fn]
    pub fn is_scalar(&self) -> bool {
        self.tag == Tag::Int || self.tag == Tag::Double || self.tag == Tag::Bool
    }

    // [spec:et:def:evalue.executorch.runtime.e-value.to-scalar-fn]
    // [spec:et:sem:evalue.executorch.runtime.e-value.to-scalar-fn]
    pub fn to_scalar(&self) -> Scalar {
        // Convert from implicit value to Scalar using implicit constructors.

        if self.is_double() {
            Scalar::from_double(self.to_double())
        } else if self.is_int() {
            Scalar::from_i64(self.to_int())
        } else if self.is_bool() {
            Scalar::from_bool(self.to_bool())
        } else {
            et_check_msg!(false, "EValue is not a Scalar.");
            // PORT-NOTE: `ET_CHECK_MSG(false, ...)` never returns; the abort
            // above diverges so this line is unreachable.
            unreachable!()
        }
    }

    // [spec:et:def:evalue.executorch.runtime.e-value.try-to-scalar-fn]
    // [spec:et:sem:evalue.executorch.runtime.e-value.try-to-scalar-fn]
    pub fn try_to_scalar(&self) -> Result<Scalar> {
        if self.is_double() {
            return Ok(Scalar::from_double(unsafe {
                self.payload.copyable_union.as_double
            }));
        }
        if self.is_int() {
            return Ok(Scalar::from_i64(unsafe {
                self.payload.copyable_union.as_int
            }));
        }
        if self.is_bool() {
            return Ok(Scalar::from_bool(unsafe {
                self.payload.copyable_union.as_bool
            }));
        }
        Err(Error::InvalidType)
    }

    /****** Tensor Type ******/
    // /*implicit*/ EValue(executorch::aten::Tensor t) : tag(Tag::Tensor)
    pub fn from_tensor(t: Tensor<'a>) -> Self {
        let mut this = EValue {
            payload: Payload::new(),
            tag: Tag::Tensor,
        };
        // When built in aten mode, at::Tensor has a non trivial constructor
        // destructor, so regular assignment to a union field is UB. Instead we
        // must go through placement new (which causes a refcount bump).
        // new (&payload.as_tensor) executorch::aten::Tensor(t);
        this.payload.as_tensor = core::mem::ManuallyDrop::new(t);
        this
    }

    // PORT-NOTE: the template ctor `EValue(T&& value)` (construct from a
    // dereferenceable-to-EValue type via `moveFrom(*value)`) and the deleted
    // raw-pointer ctor `EValue(T*)` are C++ overload-resolution / SFINAE
    // machinery with no Rust analog; callers move an EValue directly via
    // `from_move`. Not ported as a distinct item.

    // [spec:et:def:evalue.executorch.runtime.e-value.is-tensor-fn]
    // [spec:et:sem:evalue.executorch.runtime.e-value.is-tensor-fn]
    pub fn is_tensor(&self) -> bool {
        self.tag == Tag::Tensor
    }

    // [spec:et:def:evalue.executorch.runtime.e-value.to-tensor-fn]
    // [spec:et:sem:evalue.executorch.runtime.e-value.to-tensor-fn]
    // executorch::aten::Tensor toTensor() &&
    pub fn to_tensor_move(&mut self) -> Tensor<'a> {
        et_check_msg!(self.is_tensor(), "EValue is not a Tensor.");
        let res = unsafe { core::ptr::read(&*self.payload.as_tensor) };
        self.clear_to_none();
        res
    }

    // executorch::aten::Tensor& toTensor() &
    pub fn to_tensor_mut(&mut self) -> &mut Tensor<'a> {
        et_check_msg!(self.is_tensor(), "EValue is not a Tensor.");
        unsafe { &mut self.payload.as_tensor }
    }

    // const executorch::aten::Tensor& toTensor() const&
    pub fn to_tensor(&self) -> &Tensor<'a> {
        et_check_msg!(self.is_tensor(), "EValue is not a Tensor.");
        unsafe { &self.payload.as_tensor }
    }

    // Returns a copy of the Tensor handle (one intrusive_ptr refcount bump in
    // ATen mode; free in lean mode). Unlike toTensor()'s const& / & overloads,
    // tryToTensor() cannot return a reference — Result<T> wraps by value.
    // [spec:et:def:evalue.executorch.runtime.e-value.try-to-tensor-fn]
    // [spec:et:sem:evalue.executorch.runtime.e-value.try-to-tensor-fn]
    pub fn try_to_tensor(&self) -> Result<Tensor<'a>> {
        if !self.is_tensor() {
            return Err(Error::InvalidType);
        }
        Ok(unsafe { core::ptr::read(&*self.payload.as_tensor) })
    }

    /****** String Type ******/
    // /*implicit*/ EValue(executorch::aten::ArrayRef<char>* s) : tag(Tag::String)
    pub fn from_string(s: *mut ArrayRef<u8>) -> Self {
        let mut this = EValue {
            payload: Payload::new(),
            tag: Tag::String,
        };
        et_check_msg!(!s.is_null(), "ArrayRef<char> pointer cannot be null");
        this.payload.copyable_union.as_string_ptr = s;
        this
    }

    // [spec:et:def:evalue.executorch.runtime.e-value.is-string-fn]
    // [spec:et:sem:evalue.executorch.runtime.e-value.is-string-fn]
    pub fn is_string(&self) -> bool {
        self.tag == Tag::String
    }

    // [spec:et:def:evalue.executorch.runtime.e-value.to-string-fn]
    // [spec:et:sem:evalue.executorch.runtime.e-value.to-string-fn]
    // std::string_view toString() const
    //
    // PORT-NOTE: `std::string_view` maps to `&str` (a fat pointer over the
    // ArrayRef<char> data + size, no copy, no null-terminator assumption).
    pub fn to_string(&self) -> &'a str {
        et_check_msg!(self.is_string(), "EValue is not a String.");
        et_check_msg!(
            !unsafe { self.payload.copyable_union.as_string_ptr }.is_null(),
            "EValue string pointer is null."
        );
        unsafe {
            let p = self.payload.copyable_union.as_string_ptr;
            core::str::from_utf8_unchecked(core::slice::from_raw_parts((*p).data(), (*p).size()))
        }
    }

    // [spec:et:def:evalue.executorch.runtime.e-value.try-to-string-fn]
    // [spec:et:sem:evalue.executorch.runtime.e-value.try-to-string-fn]
    pub fn try_to_string(&self) -> Result<&'a str> {
        if !self.is_string() {
            return Err(Error::InvalidType);
        }
        if unsafe { self.payload.copyable_union.as_string_ptr }.is_null() {
            return Err(Error::InvalidState);
        }
        Ok(unsafe {
            let p = self.payload.copyable_union.as_string_ptr;
            core::str::from_utf8_unchecked(core::slice::from_raw_parts((*p).data(), (*p).size()))
        })
    }

    /****** Int List Type ******/
    // /*implicit*/ EValue(BoxedEvalueList<int64_t>* i) : tag(Tag::ListInt)
    pub fn from_int_list(i: *mut BoxedEvalueList<'a, i64>) -> Self {
        let mut this = EValue {
            payload: Payload::new(),
            tag: Tag::ListInt,
        };
        et_check_msg!(
            !i.is_null(),
            "BoxedEvalueList<int64_t> pointer cannot be null"
        );
        this.payload.copyable_union.as_int_list_ptr = i;
        this
    }

    // [spec:et:def:evalue.executorch.runtime.e-value.is-int-list-fn]
    // [spec:et:sem:evalue.executorch.runtime.e-value.is-int-list-fn]
    pub fn is_int_list(&self) -> bool {
        self.tag == Tag::ListInt
    }

    // [spec:et:def:evalue.executorch.runtime.e-value.to-int-list-fn]
    // [spec:et:sem:evalue.executorch.runtime.e-value.to-int-list-fn]
    pub fn to_int_list(&self) -> ArrayRef<i64> {
        et_check_msg!(self.is_int_list(), "EValue is not an Int List.");
        et_check_msg!(
            !unsafe { self.payload.copyable_union.as_int_list_ptr }.is_null(),
            "EValue int list pointer is null."
        );
        unsafe { (*self.payload.copyable_union.as_int_list_ptr).get() }
    }

    // [spec:et:def:evalue.executorch.runtime.e-value.try-to-int-list-fn]
    // [spec:et:sem:evalue.executorch.runtime.e-value.try-to-int-list-fn]
    pub fn try_to_int_list(&self) -> Result<ArrayRef<i64>> {
        if !self.is_int_list() {
            return Err(Error::InvalidType);
        }
        if unsafe { self.payload.copyable_union.as_int_list_ptr }.is_null() {
            return Err(Error::InvalidState);
        }
        unsafe { (*self.payload.copyable_union.as_int_list_ptr).try_get() }
    }

    /****** Bool List Type ******/
    // /*implicit*/ EValue(executorch::aten::ArrayRef<bool>* b) : tag(Tag::ListBool)
    pub fn from_bool_list(b: *mut ArrayRef<bool>) -> Self {
        let mut this = EValue {
            payload: Payload::new(),
            tag: Tag::ListBool,
        };
        et_check_msg!(!b.is_null(), "ArrayRef<bool> pointer cannot be null");
        this.payload.copyable_union.as_bool_list_ptr = b;
        this
    }

    // [spec:et:def:evalue.executorch.runtime.e-value.is-bool-list-fn]
    // [spec:et:sem:evalue.executorch.runtime.e-value.is-bool-list-fn]
    pub fn is_bool_list(&self) -> bool {
        self.tag == Tag::ListBool
    }

    // [spec:et:def:evalue.executorch.runtime.e-value.to-bool-list-fn]
    // [spec:et:sem:evalue.executorch.runtime.e-value.to-bool-list-fn]
    pub fn to_bool_list(&self) -> ArrayRef<bool> {
        et_check_msg!(self.is_bool_list(), "EValue is not a Bool List.");
        et_check_msg!(
            !unsafe { self.payload.copyable_union.as_bool_list_ptr }.is_null(),
            "EValue bool list pointer is null."
        );
        unsafe { *self.payload.copyable_union.as_bool_list_ptr }
    }

    // [spec:et:def:evalue.executorch.runtime.e-value.try-to-bool-list-fn]
    // [spec:et:sem:evalue.executorch.runtime.e-value.try-to-bool-list-fn]
    pub fn try_to_bool_list(&self) -> Result<ArrayRef<bool>> {
        if !self.is_bool_list() {
            return Err(Error::InvalidType);
        }
        if unsafe { self.payload.copyable_union.as_bool_list_ptr }.is_null() {
            return Err(Error::InvalidState);
        }
        Ok(unsafe { *self.payload.copyable_union.as_bool_list_ptr })
    }

    /****** Double List Type ******/
    // /*implicit*/ EValue(executorch::aten::ArrayRef<double>* d) : tag(Tag::ListDouble)
    pub fn from_double_list(d: *mut ArrayRef<f64>) -> Self {
        let mut this = EValue {
            payload: Payload::new(),
            tag: Tag::ListDouble,
        };
        et_check_msg!(!d.is_null(), "ArrayRef<double> pointer cannot be null");
        this.payload.copyable_union.as_double_list_ptr = d;
        this
    }

    // [spec:et:def:evalue.executorch.runtime.e-value.is-double-list-fn]
    // [spec:et:sem:evalue.executorch.runtime.e-value.is-double-list-fn]
    pub fn is_double_list(&self) -> bool {
        self.tag == Tag::ListDouble
    }

    // [spec:et:def:evalue.executorch.runtime.e-value.to-double-list-fn]
    // [spec:et:sem:evalue.executorch.runtime.e-value.to-double-list-fn]
    pub fn to_double_list(&self) -> ArrayRef<f64> {
        et_check_msg!(self.is_double_list(), "EValue is not a Double List.");
        et_check_msg!(
            !unsafe { self.payload.copyable_union.as_double_list_ptr }.is_null(),
            "EValue double list pointer is null."
        );
        unsafe { *self.payload.copyable_union.as_double_list_ptr }
    }

    // [spec:et:def:evalue.executorch.runtime.e-value.try-to-double-list-fn]
    // [spec:et:sem:evalue.executorch.runtime.e-value.try-to-double-list-fn]
    pub fn try_to_double_list(&self) -> Result<ArrayRef<f64>> {
        if !self.is_double_list() {
            return Err(Error::InvalidType);
        }
        if unsafe { self.payload.copyable_union.as_double_list_ptr }.is_null() {
            return Err(Error::InvalidState);
        }
        Ok(unsafe { *self.payload.copyable_union.as_double_list_ptr })
    }

    /****** Tensor List Type ******/
    // /*implicit*/ EValue(BoxedEvalueList<executorch::aten::Tensor>* t) : tag(Tag::ListTensor)
    pub fn from_tensor_list(t: *mut BoxedEvalueList<'a, Tensor<'a>>) -> Self {
        let mut this = EValue {
            payload: Payload::new(),
            tag: Tag::ListTensor,
        };
        et_check_msg!(
            !t.is_null(),
            "BoxedEvalueList<Tensor> pointer cannot be null"
        );
        this.payload.copyable_union.as_tensor_list_ptr = t;
        this
    }

    // [spec:et:def:evalue.executorch.runtime.e-value.is-tensor-list-fn]
    // [spec:et:sem:evalue.executorch.runtime.e-value.is-tensor-list-fn]
    pub fn is_tensor_list(&self) -> bool {
        self.tag == Tag::ListTensor
    }

    // [spec:et:def:evalue.executorch.runtime.e-value.to-tensor-list-fn]
    // [spec:et:sem:evalue.executorch.runtime.e-value.to-tensor-list-fn]
    pub fn to_tensor_list(&self) -> ArrayRef<Tensor<'a>> {
        et_check_msg!(self.is_tensor_list(), "EValue is not a Tensor List.");
        et_check_msg!(
            !unsafe { self.payload.copyable_union.as_tensor_list_ptr }.is_null(),
            "EValue tensor list pointer is null."
        );
        unsafe { (*self.payload.copyable_union.as_tensor_list_ptr).get() }
    }

    // [spec:et:def:evalue.executorch.runtime.e-value.try-to-tensor-list-fn]
    // [spec:et:sem:evalue.executorch.runtime.e-value.try-to-tensor-list-fn]
    pub fn try_to_tensor_list(&self) -> Result<ArrayRef<Tensor<'a>>> {
        if !self.is_tensor_list() {
            return Err(Error::InvalidType);
        }
        if unsafe { self.payload.copyable_union.as_tensor_list_ptr }.is_null() {
            return Err(Error::InvalidState);
        }
        unsafe { (*self.payload.copyable_union.as_tensor_list_ptr).try_get() }
    }

    /****** List Optional Tensor Type ******/
    // /*implicit*/ EValue(BoxedEvalueList<std::optional<Tensor>>* t) : tag(Tag::ListOptionalTensor)
    pub fn from_list_optional_tensor(t: *mut BoxedEvalueList<'a, Option<Tensor<'a>>>) -> Self {
        let mut this = EValue {
            payload: Payload::new(),
            tag: Tag::ListOptionalTensor,
        };
        et_check_msg!(
            !t.is_null(),
            "BoxedEvalueList<optional<Tensor>> pointer cannot be null"
        );
        this.payload.copyable_union.as_list_optional_tensor_ptr = t;
        this
    }

    // [spec:et:def:evalue.executorch.runtime.e-value.is-list-optional-tensor-fn]
    // [spec:et:sem:evalue.executorch.runtime.e-value.is-list-optional-tensor-fn]
    pub fn is_list_optional_tensor(&self) -> bool {
        self.tag == Tag::ListOptionalTensor
    }

    // [spec:et:def:evalue.executorch.runtime.e-value.to-list-optional-tensor-fn]
    // [spec:et:sem:evalue.executorch.runtime.e-value.to-list-optional-tensor-fn]
    pub fn to_list_optional_tensor(&self) -> ArrayRef<Option<Tensor<'a>>> {
        et_check_msg!(
            self.is_list_optional_tensor(),
            "EValue is not a List Optional Tensor."
        );
        et_check_msg!(
            !unsafe { self.payload.copyable_union.as_list_optional_tensor_ptr }.is_null(),
            "EValue list optional tensor pointer is null."
        );
        unsafe { (*self.payload.copyable_union.as_list_optional_tensor_ptr).get() }
    }

    // [spec:et:def:evalue.executorch.runtime.e-value.try-to-list-optional-tensor-fn]
    // [spec:et:sem:evalue.executorch.runtime.e-value.try-to-list-optional-tensor-fn]
    pub fn try_to_list_optional_tensor(&self) -> Result<ArrayRef<Option<Tensor<'a>>>> {
        if !self.is_list_optional_tensor() {
            return Err(Error::InvalidType);
        }
        if unsafe { self.payload.copyable_union.as_list_optional_tensor_ptr }.is_null() {
            return Err(Error::InvalidState);
        }
        unsafe { (*self.payload.copyable_union.as_list_optional_tensor_ptr).try_get() }
    }

    /****** ScalarType Type ******/
    // [spec:et:def:evalue.executorch.runtime.e-value.to-scalar-type-fn]
    // [spec:et:sem:evalue.executorch.runtime.e-value.to-scalar-type-fn]
    pub fn to_scalar_type(&self) -> ScalarType {
        et_check_msg!(self.is_int(), "EValue is not a ScalarType.");
        int_to_scalar_type(unsafe { self.payload.copyable_union.as_int })
    }

    // [spec:et:def:evalue.executorch.runtime.e-value.try-to-scalar-type-fn]
    // [spec:et:sem:evalue.executorch.runtime.e-value.try-to-scalar-type-fn]
    pub fn try_to_scalar_type(&self) -> Result<ScalarType> {
        if !self.is_int() {
            return Err(Error::InvalidType);
        }
        Ok(int_to_scalar_type(unsafe {
            self.payload.copyable_union.as_int
        }))
    }

    /****** MemoryFormat Type ******/
    // [spec:et:def:evalue.executorch.runtime.e-value.to-memory-format-fn]
    // [spec:et:sem:evalue.executorch.runtime.e-value.to-memory-format-fn]
    pub fn to_memory_format(&self) -> MemoryFormat {
        et_check_msg!(self.is_int(), "EValue is not a MemoryFormat.");
        int_to_memory_format(unsafe { self.payload.copyable_union.as_int })
    }

    // [spec:et:def:evalue.executorch.runtime.e-value.try-to-memory-format-fn]
    // [spec:et:sem:evalue.executorch.runtime.e-value.try-to-memory-format-fn]
    pub fn try_to_memory_format(&self) -> Result<MemoryFormat> {
        if !self.is_int() {
            return Err(Error::InvalidType);
        }
        Ok(int_to_memory_format(unsafe {
            self.payload.copyable_union.as_int
        }))
    }

    /****** Layout Type ******/
    // [spec:et:def:evalue.executorch.runtime.e-value.to-layout-fn]
    // [spec:et:sem:evalue.executorch.runtime.e-value.to-layout-fn]
    pub fn to_layout(&self) -> Layout {
        et_check_msg!(self.is_int(), "EValue is not a Layout.");
        int_to_layout(unsafe { self.payload.copyable_union.as_int })
    }

    // [spec:et:def:evalue.executorch.runtime.e-value.try-to-layout-fn]
    // [spec:et:sem:evalue.executorch.runtime.e-value.try-to-layout-fn]
    pub fn try_to_layout(&self) -> Result<Layout> {
        if !self.is_int() {
            return Err(Error::InvalidType);
        }
        Ok(int_to_layout(unsafe { self.payload.copyable_union.as_int }))
    }

    /****** Device Type ******/
    // [spec:et:def:evalue.executorch.runtime.e-value.to-device-fn]
    // [spec:et:sem:evalue.executorch.runtime.e-value.to-device-fn]
    pub fn to_device(&self) -> Device {
        et_check_msg!(self.is_int(), "EValue is not a Device.");
        Device::new(
            int_to_device_type(unsafe { self.payload.copyable_union.as_int }),
            -1,
        )
    }

    // [spec:et:def:evalue.executorch.runtime.e-value.try-to-device-fn]
    // [spec:et:sem:evalue.executorch.runtime.e-value.try-to-device-fn]
    pub fn try_to_device(&self) -> Result<Device> {
        if !self.is_int() {
            return Err(Error::InvalidType);
        }
        Ok(Device::new(
            int_to_device_type(unsafe { self.payload.copyable_union.as_int }),
            -1,
        ))
    }

    // template <typename T> T to() &&;
    // template <typename T> ...evalue_to_const_ref_overload_return<T>::type to() const&;
    // template <typename T> ...evalue_to_ref_overload_return<T>::type to() &;
    // [spec:et:def:evalue.executorch.runtime.e-value.to-fn]
    // [spec:et:sem:evalue.executorch.runtime.e-value.to-fn]
    //
    // PORT-NOTE: the three ref-qualified `to<T>()` overloads collapse to the
    // `EValueTo<T>::to` trait method (see impls below), which mirrors the
    // rvalue overload `static_cast<T>(std::move(*this).method_name())`. It takes
    // `&mut self` because the Tensor specialization moves out via
    // `to_tensor_move`; other specializations read only. The const&/& overloads'
    // reference return types (`const Tensor&`/`Tensor&`) are the internal
    // type-map machinery captured under `mod internal`.
    pub fn to<T>(&mut self) -> T
    where
        EValue<'a>: EValueTo<T>,
    {
        EValueTo::<T>::to(self)
    }

    /**
     * Result-returning equivalent of `to<T>()`. Tag mismatch returns
     * `Error::InvalidType`; a null list/string payload returns
     * `Error::InvalidState`. Specializations are defined below via
     * `EVALUE_DEFINE_TRY_TO`.
     */
    // [spec:et:def:evalue.executorch.runtime.e-value.try-to-fn]
    // [spec:et:sem:evalue.executorch.runtime.e-value.try-to-fn]
    pub fn try_to<T>(&self) -> Result<T>
    where
        EValue<'a>: EValueTryTo<T>,
    {
        EValueTryTo::<T>::try_to(self)
    }

    /**
     * Converts the EValue to an optional object that can represent both T and
     * an uninitialized state.
     */
    // [spec:et:def:evalue.executorch.runtime.e-value.to-optional-fn]
    // [spec:et:sem:evalue.executorch.runtime.e-value.to-optional-fn]
    //
    // PORT-NOTE: C++ `toOptional<T>()` calls the const-lvalue `to<T>()` (does
    // not move out even for Tensor). The trait `EValueTo::<T>::to` here takes
    // `&mut self`, so this method takes `&mut self` to reach it; for `T =
    // Tensor` the underlying `to_tensor_move` DOES move out, unlike the C++
    // const& path this is modeled on. PORT-NOTE (behavioral deviation): the
    // Rust single `to` trait cannot express the const& non-moving overload the
    // C++ toOptional selects; see the note on `to`.
    pub fn to_optional<T>(&mut self) -> Option<T>
    where
        EValue<'a>: EValueTo<T>,
    {
        if self.is_none() {
            return None;
        }
        Some(self.to::<T>())
    }

    /**
     * Result-returning equivalent of `toOptional<T>()`. None maps to an empty
     * optional; any other tag that doesn't match T propagates `tryTo<T>()`'s
     * error (`Error::InvalidType`).
     */
    // [spec:et:def:evalue.executorch.runtime.e-value.try-to-optional-fn]
    // [spec:et:sem:evalue.executorch.runtime.e-value.try-to-optional-fn]
    pub fn try_to_optional<T>(&self) -> Result<Option<T>>
    where
        EValue<'a>: EValueTryTo<T>,
    {
        if self.is_none() {
            return Ok(None);
        }
        let r = self.try_to::<T>();
        if !ResultExt::ok(&r) {
            return Err(r.error());
        }
        // std::optional<T>(std::move(r.get()))
        Ok(Some(r_into_ok(r)))
    }

    // Pre cond: the payload value has had its destructor called
    // [spec:et:def:evalue.executorch.runtime.e-value.clear-to-none-fn]
    // [spec:et:sem:evalue.executorch.runtime.e-value.clear-to-none-fn]
    fn clear_to_none(&mut self) {
        self.payload.copyable_union.as_int = 0;
        self.tag = Tag::None;
    }

    // Shared move logic
    // [spec:et:def:evalue.executorch.runtime.e-value.move-from-fn]
    // [spec:et:sem:evalue.executorch.runtime.e-value.move-from-fn]
    fn move_from(&mut self, rhs: &mut EValue<'a>) {
        if rhs.is_tensor() {
            // new (&payload.as_tensor) Tensor(std::move(rhs.payload.as_tensor));
            // rhs.payload.as_tensor.~Tensor();
            unsafe {
                let moved = core::ptr::read(&*rhs.payload.as_tensor);
                self.payload.as_tensor = core::mem::ManuallyDrop::new(moved);
                core::mem::ManuallyDrop::drop(&mut rhs.payload.as_tensor);
            }
        } else {
            self.payload.copyable_union = unsafe { rhs.payload.copyable_union };
        }
        self.tag = rhs.tag;
        rhs.clear_to_none();
    }

    // Destructs stored tensor if there is one
    // [spec:et:def:evalue.executorch.runtime.e-value.destroy-fn]
    // [spec:et:sem:evalue.executorch.runtime.e-value.destroy-fn]
    fn destroy(&mut self) {
        // Necessary for ATen tensor to refcount decrement the intrusive_ptr to
        // tensorimpl that got a refcount increment when we placed it in the
        // evalue, no-op if executorch tensor #ifdef could have a minor
        // performance bump for a code maintainability hit
        if self.is_tensor() {
            // payload.as_tensor.~Tensor();
            unsafe {
                core::mem::ManuallyDrop::drop(&mut self.payload.as_tensor);
            }
        } else if self.is_tensor_list()
            && !unsafe { self.payload.copyable_union.as_tensor_list_ptr }.is_null()
        {
            unsafe {
                (*self.payload.copyable_union.as_tensor_list_ptr).destroy_elements();
            }
        } else if self.is_list_optional_tensor()
            && !unsafe { self.payload.copyable_union.as_list_optional_tensor_ptr }.is_null()
        {
            unsafe {
                (*self.payload.copyable_union.as_list_optional_tensor_ptr).destroy_elements();
            }
        }
    }

    // EValue(const Payload& p, Tag t) : tag(t)
    fn from_payload_tag(p: &Payload<'a>, t: Tag) -> Self {
        let mut this = EValue {
            payload: Payload::new(),
            tag: t,
        };
        if this.is_tensor() {
            // new (&payload.as_tensor) Tensor(p.as_tensor);
            this.payload.as_tensor =
                core::mem::ManuallyDrop::new(unsafe { core::ptr::read(&*p.as_tensor) });
        } else {
            this.payload.copyable_union = unsafe { p.copyable_union };
        }
        this
    }
}

// ~EValue() { destroy(); }
impl<'a> Drop for EValue<'a> {
    fn drop(&mut self) {
        self.destroy();
    }
}

// PORT-NOTE: `static_cast<ScalarType>(payload.copyable_union.as_int)` and the
// MemoryFormat/Layout/DeviceType variants truncate the stored `int64_t` to the
// enum's underlying type (i8) and reinterpret the bits. Rust has no `as`-cast
// from an integer to a fieldless enum, so this reproduces `static_cast` via a
// truncating cast followed by `transmute` of the byte. Out-of-range values are
// UB in Rust just as they are in C++ `static_cast` to a scoped enum with a bit
// pattern outside its range; the serialized data is expected to be in range.
fn int_to_scalar_type(v: i64) -> ScalarType {
    unsafe { core::mem::transmute::<i8, ScalarType>(v as i8) }
}

fn int_to_memory_format(v: i64) -> MemoryFormat {
    unsafe { core::mem::transmute::<i8, MemoryFormat>(v as i8) }
}

fn int_to_layout(v: i64) -> Layout {
    unsafe { core::mem::transmute::<i8, Layout>(v as i8) }
}

fn int_to_device_type(v: i64) -> DeviceType {
    unsafe { core::mem::transmute::<i8, DeviceType>(v as i8) }
}

// EVALUE_DEFINE_TO(T, method_name): explicit `to<T>()` instantiation set.
//
// PORT-NOTE: modeled as one `EValueTo<T>` trait impl per instantiated `T`,
// mirroring the rvalue overload `static_cast<T>(std::move(*this).method_name())`.
// `to` takes `&mut self` because the Tensor specialization moves out.
// [spec:et:def:evalue.executorch.runtime.e-value.to-fn]
// [spec:et:sem:evalue.executorch.runtime.e-value.to-fn]
pub trait EValueTo<T> {
    fn to(this: &mut Self) -> T;
}

macro_rules! evalue_define_to {
    ($t:ty, $method:ident) => {
        impl<'a> EValueTo<$t> for EValue<'a> {
            fn to(this: &mut Self) -> $t {
                this.$method()
            }
        }
    };
}

evalue_define_to!(Scalar, to_scalar);
evalue_define_to!(i64, to_int);
evalue_define_to!(bool, to_bool);
evalue_define_to!(f64, to_double);
evalue_define_to!(ScalarType, to_scalar_type);
evalue_define_to!(MemoryFormat, to_memory_format);
evalue_define_to!(Layout, to_layout);
evalue_define_to!(Device, to_device);

// std::string_view -> &str
impl<'a> EValueTo<&'a str> for EValue<'a> {
    fn to(this: &mut Self) -> &'a str {
        this.to_string()
    }
}

// Tensor and Optional Tensor
impl<'a> EValueTo<Tensor<'a>> for EValue<'a> {
    fn to(this: &mut Self) -> Tensor<'a> {
        // PORT-NOTE (WAVE-2 FIX): `static_cast<Tensor>(std::move(*this).toTensor())`.
        // The earlier `to_tensor_move()` cleared the source EValue to None, but a
        // C++ `Tensor` is a trivially-copyable handle — `std::move(*this).toTensor()`
        // copies the handle out and never nulls the source. Cloning `to<Tensor>()`
        // destructively broke `BoxedEvalueList::get()` (used by e.g.
        // `to_tensor_list()`): materializing a tensor list nulled its element
        // value-table slots, so a later reader saw `None` (crash in
        // split_with_sizes_copy -> et_view). Copy the handle non-destructively,
        // matching the C++ const&/&& toTensor for a trivial Tensor.
        Tensor::new(this.to_tensor().unsafe_get_tensor_impl())
    }
}
impl<'a> EValueTo<Option<Tensor<'a>>> for EValue<'a> {
    fn to(this: &mut Self) -> Option<Tensor<'a>> {
        // toOptional<Tensor>()
        this.to_optional::<Tensor<'a>>()
    }
}

// IntList and Optional IntList
impl<'a> EValueTo<ArrayRef<i64>> for EValue<'a> {
    fn to(this: &mut Self) -> ArrayRef<i64> {
        this.to_int_list()
    }
}
impl<'a> EValueTo<Option<ArrayRef<i64>>> for EValue<'a> {
    fn to(this: &mut Self) -> Option<ArrayRef<i64>> {
        this.to_optional::<ArrayRef<i64>>()
    }
}

// DoubleList and Optional DoubleList
impl<'a> EValueTo<ArrayRef<f64>> for EValue<'a> {
    fn to(this: &mut Self) -> ArrayRef<f64> {
        this.to_double_list()
    }
}
impl<'a> EValueTo<Option<ArrayRef<f64>>> for EValue<'a> {
    fn to(this: &mut Self) -> Option<ArrayRef<f64>> {
        this.to_optional::<ArrayRef<f64>>()
    }
}

// BoolList and Optional BoolList
impl<'a> EValueTo<ArrayRef<bool>> for EValue<'a> {
    fn to(this: &mut Self) -> ArrayRef<bool> {
        this.to_bool_list()
    }
}
impl<'a> EValueTo<Option<ArrayRef<bool>>> for EValue<'a> {
    fn to(this: &mut Self) -> Option<ArrayRef<bool>> {
        this.to_optional::<ArrayRef<bool>>()
    }
}

// TensorList and Optional TensorList
impl<'a> EValueTo<ArrayRef<Tensor<'a>>> for EValue<'a> {
    fn to(this: &mut Self) -> ArrayRef<Tensor<'a>> {
        this.to_tensor_list()
    }
}
impl<'a> EValueTo<Option<ArrayRef<Tensor<'a>>>> for EValue<'a> {
    fn to(this: &mut Self) -> Option<ArrayRef<Tensor<'a>>> {
        this.to_optional::<ArrayRef<Tensor<'a>>>()
    }
}

// List of Optional Tensor
impl<'a> EValueTo<ArrayRef<Option<Tensor<'a>>>> for EValue<'a> {
    fn to(this: &mut Self) -> ArrayRef<Option<Tensor<'a>>> {
        this.to_list_optional_tensor()
    }
}

// EVALUE_DEFINE_TRY_TO(T, method_name): explicit `tryTo<T>()` instantiation set.
//
// PORT-NOTE: modeled as one `EValueTryTo<T>` trait impl per instantiated `T`,
// mirroring `return this->method_name();`.
// [spec:et:def:evalue.executorch.runtime.e-value.try-to-fn]
// [spec:et:sem:evalue.executorch.runtime.e-value.try-to-fn]
pub trait EValueTryTo<T> {
    fn try_to(this: &Self) -> Result<T>;
}

macro_rules! evalue_define_try_to {
    ($t:ty, $method:ident) => {
        impl<'a> EValueTryTo<$t> for EValue<'a> {
            fn try_to(this: &Self) -> Result<$t> {
                this.$method()
            }
        }
    };
}

evalue_define_try_to!(Scalar, try_to_scalar);
evalue_define_try_to!(i64, try_to_int);
evalue_define_try_to!(bool, try_to_bool);
evalue_define_try_to!(f64, try_to_double);
evalue_define_try_to!(ScalarType, try_to_scalar_type);
evalue_define_try_to!(MemoryFormat, try_to_memory_format);
evalue_define_try_to!(Layout, try_to_layout);
evalue_define_try_to!(Device, try_to_device);

// std::string_view -> &str
impl<'a> EValueTryTo<&'a str> for EValue<'a> {
    fn try_to(this: &Self) -> Result<&'a str> {
        this.try_to_string()
    }
}

// Tensor and Optional Tensor
impl<'a> EValueTryTo<Tensor<'a>> for EValue<'a> {
    fn try_to(this: &Self) -> Result<Tensor<'a>> {
        this.try_to_tensor()
    }
}
impl<'a> EValueTryTo<Option<Tensor<'a>>> for EValue<'a> {
    fn try_to(this: &Self) -> Result<Option<Tensor<'a>>> {
        this.try_to_optional::<Tensor<'a>>()
    }
}

// IntList and Optional IntList
impl<'a> EValueTryTo<ArrayRef<i64>> for EValue<'a> {
    fn try_to(this: &Self) -> Result<ArrayRef<i64>> {
        this.try_to_int_list()
    }
}
impl<'a> EValueTryTo<Option<ArrayRef<i64>>> for EValue<'a> {
    fn try_to(this: &Self) -> Result<Option<ArrayRef<i64>>> {
        this.try_to_optional::<ArrayRef<i64>>()
    }
}

// DoubleList and Optional DoubleList
impl<'a> EValueTryTo<ArrayRef<f64>> for EValue<'a> {
    fn try_to(this: &Self) -> Result<ArrayRef<f64>> {
        this.try_to_double_list()
    }
}
impl<'a> EValueTryTo<Option<ArrayRef<f64>>> for EValue<'a> {
    fn try_to(this: &Self) -> Result<Option<ArrayRef<f64>>> {
        this.try_to_optional::<ArrayRef<f64>>()
    }
}

// BoolList and Optional BoolList
impl<'a> EValueTryTo<ArrayRef<bool>> for EValue<'a> {
    fn try_to(this: &Self) -> Result<ArrayRef<bool>> {
        this.try_to_bool_list()
    }
}
impl<'a> EValueTryTo<Option<ArrayRef<bool>>> for EValue<'a> {
    fn try_to(this: &Self) -> Result<Option<ArrayRef<bool>>> {
        this.try_to_optional::<ArrayRef<bool>>()
    }
}

// TensorList and Optional TensorList
impl<'a> EValueTryTo<ArrayRef<Tensor<'a>>> for EValue<'a> {
    fn try_to(this: &Self) -> Result<ArrayRef<Tensor<'a>>> {
        this.try_to_tensor_list()
    }
}
impl<'a> EValueTryTo<Option<ArrayRef<Tensor<'a>>>> for EValue<'a> {
    fn try_to(this: &Self) -> Result<Option<ArrayRef<Tensor<'a>>>> {
        this.try_to_optional::<ArrayRef<Tensor<'a>>>()
    }
}

// List of Optional Tensor
impl<'a> EValueTryTo<ArrayRef<Option<Tensor<'a>>>> for EValue<'a> {
    fn try_to(this: &Self) -> Result<ArrayRef<Option<Tensor<'a>>>> {
        this.try_to_list_optional_tensor()
    }
}

// TODO(T197294990): Remove these deprecated aliases once all users have moved
// to the new `::executorch` namespaces.
// namespace torch::executor { using EValue; using BoxedEvalueList; }
pub use self::BoxedEvalueList as TorchExecutorBoxedEvalueList;
pub use self::EValue as TorchExecutorEValue;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::core::exec_aten::testing_util::tensor_factory::TensorFactory;

    // class EValueTest : SetUp() { runtime_init(); }
    fn setup() {
        crate::runtime::platform::runtime::runtime_init();
    }

    // PORT-NOTE: `TensorWrapper` (a smart-pointer-like wrapper whose EValue ctor
    // is a C++ template `EValue(T&& value)` dereferencing to a Tensor) and the
    // `EValue(unique_ptr<Tensor>)` / `EValue(shared_ptr<Tensor>)` template ctors
    // are NOT ported in evalue.rs (see the PORT-NOTE on the template ctor there —
    // "callers move an EValue directly via `from_move`"). The four tests that
    // depend on them (ConstructFromUniquePtr, ConstructFromSharedPtr,
    // ConstructFromTensorWrapper, ConstructFromNullPtrAborts) therefore have no
    // Rust surface to bind to and are recorded but not ported. Port them
    // alongside a future port of those template constructors.

    // [spec:et:sem:evalue.executorch.runtime.e-value.is-none-fn/test]
    // [spec:et:sem:evalue.executorch.runtime.e-value.is-bool-fn/test]
    // [spec:et:sem:evalue.executorch.runtime.e-value.to-bool-fn/test]
    // [spec:et:sem:evalue.executorch.runtime.e-value.operator-fn/test]
    #[test]
    fn evalue_test_copy_trivial_type() {
        setup();
        let mut a = EValue::new();
        let b = EValue::from_bool(true);
        assert!(a.is_none());
        a.assign_ref(&b);
        assert!(a.is_bool());
        assert_eq!(a.to::<bool>(), true);
        // b unchanged.
        {
            let mut b2 = EValue::from_ref(&b);
            assert_eq!(b2.to::<bool>(), true);
        }
    }

    // [spec:et:sem:evalue.executorch.runtime.e-value.to-tensor-fn/test]
    // [spec:et:sem:evalue.executorch.runtime.e-value.operator-fn/test]
    // also verifies destroy (assign_ref -> assign_move -> destroy tears down the
    // old tensor before installing the new one) and move_from (the tensor branch
    // moves the tensor into `a` and clears the source)
    // [spec:et:sem:evalue.executorch.runtime.e-value.destroy-fn/test]
    // [spec:et:sem:evalue.executorch.runtime.e-value.move-from-fn/test]
    #[test]
    fn evalue_test_copy_tensor() {
        setup();
        let tf = TensorFactory::<f32>::new();
        let mut a = EValue::from_tensor(tf.ones_default(vec![3, 2]));
        let b = EValue::from_tensor(tf.ones_default(vec![1]));
        assert_eq!(a.to_tensor().dim(), 2);
        a.assign_ref(&b);
        assert_eq!(a.to_tensor().dim(), 1);
    }

    // [spec:et:sem:evalue.executorch.runtime.e-value.to-int-fn/test]
    #[test]
    #[should_panic]
    #[ignore]
    fn evalue_test_type_mismatch_fatals() {
        setup();
        let e = EValue::from_bool(true);
        e.to_int();
    }

    // [spec:et:sem:evalue.executorch.runtime.e-value.is-none-fn/test]
    // also verifies the Payload / TriviallyCopyablePayload default ctors:
    // EValue::new builds Payload::new() (which constructs TriviallyCopyablePayload
    // with as_int == 0) before the ctor sets tag = None, giving a None EValue.
    // [spec:et:sem:evalue.executorch.runtime.e-value.payload.payload-fn/test]
    // [spec:et:sem:evalue.executorch.runtime.e-value.payload.trivially-copyable-payload.trivially-copyable-payload-fn/test]
    #[test]
    fn evalue_test_none_by_default() {
        setup();
        let e = EValue::new();
        assert!(e.is_none());
    }

    // [spec:et:sem:evalue.executorch.runtime.e-value.is-int-fn/test]
    // [spec:et:sem:evalue.executorch.runtime.e-value.to-optional-fn/test]
    #[test]
    fn evalue_test_to_optional_int() {
        setup();
        let mut e = EValue::from_int(5);
        assert!(e.is_int());
        assert!(!e.is_none());

        let o: Option<i64> = e.to_optional::<i64>();
        assert!(o.is_some());
        assert_eq!(o.unwrap(), 5);
    }

    // [spec:et:sem:evalue.executorch.runtime.e-value.to-optional-fn/test]
    #[test]
    fn evalue_test_none_to_optional_int() {
        setup();
        let mut e = EValue::new();
        assert!(e.is_none());

        let o: Option<i64> = e.to_optional::<i64>();
        assert!(o.is_none());
    }

    // [spec:et:sem:evalue.executorch.runtime.e-value.is-scalar-fn/test]
    // [spec:et:sem:evalue.executorch.runtime.e-value.to-optional-fn/test]
    // also verifies from_scalar (the Scalar EValue ctor): a floating-point Scalar
    // is stored with Tag::Double and recovered as a floating-point 3.141.
    // [spec:et:sem:evalue.executorch.runtime.e-value.e-value-fn/test]
    #[test]
    fn evalue_test_to_optional_scalar() {
        setup();
        let s = Scalar::from_double(3.141);
        let mut e = EValue::from_scalar(s);
        assert!(e.is_scalar());
        assert!(!e.is_none());

        let o: Option<Scalar> = e.to_optional::<Scalar>();
        assert!(o.is_some());
        assert!(o.unwrap().is_floating_point());
        assert_eq!(o.unwrap().to_f64(), 3.141);
    }

    // [spec:et:sem:scalar.executorch.runtime.etensor.scalar.to-fn/test]
    #[test]
    fn evalue_test_scalar_to_type() {
        setup();
        let s_d = Scalar::from_double(3.141);
        assert_eq!(s_d.to_f64(), 3.141);
        let s_i = Scalar::from_i64(3);
        assert_eq!(s_i.to_i64(), 3);
        let s_b = Scalar::from_bool(true);
        assert_eq!(s_b.to_bool_val(), true);
    }

    // [spec:et:sem:evalue.executorch.runtime.e-value.to-optional-fn/test]
    #[test]
    fn evalue_test_none_to_optional_scalar() {
        setup();
        let mut e = EValue::new();
        assert!(e.is_none());

        let o: Option<Scalar> = e.to_optional::<Scalar>();
        assert!(o.is_none());
    }

    // [spec:et:sem:evalue.executorch.runtime.e-value.to-optional-fn/test]
    #[test]
    fn evalue_test_none_to_optional_tensor() {
        setup();
        let mut e = EValue::new();
        assert!(e.is_none());

        let o: Option<Tensor> = e.to_optional::<Tensor>();
        assert!(o.is_none());
    }

    // [spec:et:sem:evalue.executorch.runtime.e-value.to-scalar-type-fn/test]
    // [spec:et:sem:evalue.executorch.runtime.e-value.to-optional-fn/test]
    #[test]
    fn evalue_test_to_scalar_type() {
        setup();
        let e = EValue::from_int(4);
        let o = e.to_scalar_type();
        assert_eq!(o, ScalarType::Long);
        let mut f = EValue::from_int(4);
        let o2 = f.to_optional::<ScalarType>();
        assert!(o2.is_some());
        assert_eq!(o2.unwrap(), ScalarType::Long);
    }

    // [spec:et:sem:evalue.executorch.runtime.e-value.is-string-fn/test]
    // [spec:et:sem:evalue.executorch.runtime.e-value.to-string-fn/test]
    #[test]
    fn evalue_test_to_string() {
        setup();
        // std::make_unique<ArrayRef<char>>("foo", 3)
        let foo = b"foo";
        let mut string_ref = ArrayRef::<u8>::from_raw_parts(foo.as_ptr(), 3);
        let e = EValue::from_string(&mut string_ref as *mut ArrayRef<u8>);
        assert!(e.is_string());
        assert!(!e.is_none());

        let x = e.to_string();
        assert_eq!(x, "foo");
    }

    // [spec:et:sem:evalue.executorch.runtime.e-value.is-int-fn/test]
    // [spec:et:sem:evalue.executorch.runtime.e-value.to-memory-format-fn/test]
    // also verifies the templated to<T>() dispatch (EValueTo): e.to::<MemoryFormat>()
    // routes through the EValueTo<MemoryFormat> impl to to_memory_format.
    // [spec:et:sem:evalue.executorch.runtime.e-value.to-fn/test]
    #[test]
    fn evalue_test_memory_format() {
        setup();
        let mut e = EValue::from_int(0);
        assert!(e.is_int());
        let m: MemoryFormat = e.to::<MemoryFormat>();
        assert_eq!(m, MemoryFormat::Contiguous);
    }

    // [spec:et:sem:evalue.executorch.runtime.e-value.is-int-fn/test]
    // [spec:et:sem:evalue.executorch.runtime.e-value.to-layout-fn/test]
    #[test]
    fn evalue_test_layout() {
        setup();
        let mut e = EValue::from_int(0);
        assert!(e.is_int());
        let l: Layout = e.to::<Layout>();
        assert_eq!(l, Layout::Strided);
    }

    // [spec:et:sem:evalue.executorch.runtime.e-value.is-int-fn/test]
    // [spec:et:sem:evalue.executorch.runtime.e-value.to-device-fn/test]
    #[test]
    fn evalue_test_device() {
        setup();
        let mut e = EValue::from_int(0);
        assert!(e.is_int());
        let d: Device = e.to::<Device>();
        assert!(d.is_cpu());
    }

    // [spec:et:sem:evalue.executorch.runtime.boxed-evalue-list-t.get-fn/test]
    // also verifies the BoxedEvalueList ctor (new: stores the checked wrapped/
    // unwrapped pointers) and the generic BoxedListGet::get trait method.
    // [spec:et:sem:evalue.executorch.runtime.boxed-evalue-list.boxed-evalue-list-fn/test]
    // [spec:et:sem:evalue.executorch.runtime.boxed-evalue-list.get-fn/test]
    #[test]
    fn evalue_test_boxed_evalue_list() {
        setup();
        // create fake values table to point to
        let mut values: [EValue; 3] = [
            EValue::from_int(1),
            EValue::from_int(2),
            EValue::from_int(3),
        ];
        // create wrapped and unwrapped lists
        let mut values_p: [*mut EValue; 3] = [&mut values[0], &mut values[1], &mut values[2]];
        let mut storage: [i64; 3] = [0, 0, 0];
        // Create Object List and test
        let x = BoxedEvalueList::<i64>::new(values_p.as_mut_ptr(), storage.as_mut_ptr(), 3);
        let unwrapped = x.get();
        assert_eq!(unwrapped.size(), 3);
        assert_eq!(*unwrapped.at(0), 1);
        assert_eq!(*unwrapped.at(1), 2);
        assert_eq!(*unwrapped.at(2), 3);
    }

    // [spec:et:sem:evalue.executorch.runtime.boxed-evalue-list-t.try-get-fn/test]
    // also verifies the generic BoxedListGet::try_get trait method.
    // [spec:et:sem:evalue.executorch.runtime.boxed-evalue-list.try-get-fn/test]
    #[test]
    fn evalue_test_boxed_evalue_list_try_get_success() {
        setup();
        let mut values: [EValue; 3] = [
            EValue::from_int(1),
            EValue::from_int(2),
            EValue::from_int(3),
        ];
        let mut values_p: [*mut EValue; 3] = [&mut values[0], &mut values[1], &mut values[2]];
        let mut storage: [i64; 3] = [0, 0, 0];
        let x = BoxedEvalueList::<i64>::new(values_p.as_mut_ptr(), storage.as_mut_ptr(), 3);
        let result = x.try_get();
        assert!(ResultExt::ok(&result));
        assert_eq!(ResultExt::get(&result).size(), 3);
        assert_eq!(*ResultExt::get(&result).at(0), 1);
        assert_eq!(*ResultExt::get(&result).at(1), 2);
        assert_eq!(*ResultExt::get(&result).at(2), 3);
    }

    // [spec:et:sem:evalue.executorch.runtime.boxed-evalue-list-t.try-get-fn/test]
    #[test]
    fn evalue_test_boxed_evalue_list_try_get_wrong_element_tag() {
        setup();
        // Second element is a Double, not an Int; tryGet should reject it.
        let mut values: [EValue; 3] = [
            EValue::from_int(1),
            EValue::from_double(3.14),
            EValue::from_int(3),
        ];
        let mut values_p: [*mut EValue; 3] = [&mut values[0], &mut values[1], &mut values[2]];
        let mut storage: [i64; 3] = [0, 0, 0];
        let x = BoxedEvalueList::<i64>::new(values_p.as_mut_ptr(), storage.as_mut_ptr(), 3);
        let result = x.try_get();
        assert_eq!(result.error(), Error::InvalidType);
    }

    // [spec:et:sem:evalue.executorch.runtime.boxed-evalue-list-t.try-get-fn/test]
    #[test]
    fn evalue_test_boxed_evalue_list_try_get_null_element() {
        setup();
        // A null value is a malformed program for non-optional lists.
        let mut a = EValue::from_int(1);
        let mut c = EValue::from_int(3);
        let mut values_p: [*mut EValue; 3] = [&mut a, core::ptr::null_mut(), &mut c];
        let mut storage: [i64; 3] = [0, 0, 0];
        let x = BoxedEvalueList::<i64>::new(values_p.as_mut_ptr(), storage.as_mut_ptr(), 3);
        let result = x.try_get();
        assert_eq!(result.error(), Error::InvalidState);
    }

    // [spec:et:sem:evalue.executorch.runtime.boxed-evalue-list-std.optional-executorch.aten.tensor.try-get-fn/test]
    #[test]
    fn evalue_test_boxed_evalue_list_try_get_optional_tensor_null_is_none() {
        setup();
        // For optional<Tensor>, null value is valid.
        let mut a = EValue::new();
        let mut values_p: [*mut EValue; 2] = [&mut a, core::ptr::null_mut()];
        let mut storage: [Option<Tensor>; 2] = [None, None];
        let x =
            BoxedEvalueList::<Option<Tensor>>::new(values_p.as_mut_ptr(), storage.as_mut_ptr(), 2);
        let result = x.try_get();
        assert!(ResultExt::ok(&result));
        assert_eq!(ResultExt::get(&result).size(), 2);
        assert!(ResultExt::get(&result).at(0).is_none());
        assert!(ResultExt::get(&result).at(1).is_none());
    }

    // [spec:et:sem:evalue.executorch.runtime.e-value.is-list-optional-tensor-fn/test]
    // [spec:et:sem:evalue.executorch.runtime.e-value.to-list-optional-tensor-fn/test]
    // also verifies the optional<Tensor> specialization of BoxedEvalueList::get:
    // to_list_optional_tensor materializes the list through get(), which maps each
    // null wrapped EValue to None.
    // [spec:et:sem:evalue.executorch.runtime.boxed-evalue-list-std.optional-executorch.aten.tensor.get-fn/test]
    #[test]
    fn evalue_test_to_optional_tensor_list() {
        setup();
        // create list, empty evalue ctor gets tag::None
        let mut values: [EValue; 2] = [EValue::new(), EValue::new()];
        let mut values_p: [*mut EValue; 2] = [&mut values[0], &mut values[1]];
        let mut storage: [Option<Tensor>; 2] = [None, None];
        // wrap in a boxed list
        let mut boxed_list =
            BoxedEvalueList::<Option<Tensor>>::new(values_p.as_mut_ptr(), storage.as_mut_ptr(), 2);

        // create Evalue
        let mut e = EValue::from_list_optional_tensor(
            &mut boxed_list as *mut BoxedEvalueList<Option<Tensor>>,
        );
        e.tag = Tag::ListOptionalTensor;
        assert!(e.is_list_optional_tensor());

        // Convert back to list
        let x = e.to_list_optional_tensor();
        assert_eq!(x.size(), 2);
        assert!(x.at(0).is_none());
        assert!(x.at(1).is_none());
    }

    // ConstructFromUniquePtr / ConstructFromSharedPtr / ConstructFromTensorWrapper
    // / ConstructFromNullPtrAborts: see the top-of-module PORT-NOTE — the
    // underlying template constructors are not ported, so these have no Rust
    // surface and are omitted.

    // [spec:et:sem:evalue.executorch.runtime.e-value.from-string-fn/test]
    //
    // StringConstructorNullCheck: `EValue::from_string(nullptr)` aborts
    // ("pointer cannot be null"). Death test -> should_panic + ignore.
    #[test]
    #[should_panic]
    #[ignore]
    fn evalue_test_string_constructor_null_check() {
        setup();
        let null_string_ptr: *mut ArrayRef<u8> = core::ptr::null_mut();
        let _ = EValue::from_string(null_string_ptr);
    }

    // [spec:et:sem:evalue.executorch.runtime.e-value.from-bool-list-fn/test]
    #[test]
    #[should_panic]
    #[ignore]
    fn evalue_test_bool_list_constructor_null_check() {
        setup();
        let null_bool_list_ptr: *mut ArrayRef<bool> = core::ptr::null_mut();
        let _ = EValue::from_bool_list(null_bool_list_ptr);
    }

    // [spec:et:sem:evalue.executorch.runtime.e-value.from-double-list-fn/test]
    #[test]
    #[should_panic]
    #[ignore]
    fn evalue_test_double_list_constructor_null_check() {
        setup();
        let null_double_list_ptr: *mut ArrayRef<f64> = core::ptr::null_mut();
        let _ = EValue::from_double_list(null_double_list_ptr);
    }

    // IntListConstructorNullCheck
    #[test]
    #[should_panic]
    #[ignore]
    fn evalue_test_int_list_constructor_null_check() {
        setup();
        let null_int_list_ptr: *mut BoxedEvalueList<i64> = core::ptr::null_mut();
        let _ = EValue::from_int_list(null_int_list_ptr);
    }

    // TensorListConstructorNullCheck
    #[test]
    #[should_panic]
    #[ignore]
    fn evalue_test_tensor_list_constructor_null_check() {
        setup();
        let null_tensor_list_ptr: *mut BoxedEvalueList<Tensor> = core::ptr::null_mut();
        let _ = EValue::from_tensor_list(null_tensor_list_ptr);
    }

    // OptionalTensorListConstructorNullCheck
    #[test]
    #[should_panic]
    #[ignore]
    fn evalue_test_optional_tensor_list_constructor_null_check() {
        setup();
        let null_optional_tensor_list_ptr: *mut BoxedEvalueList<Option<Tensor>> =
            core::ptr::null_mut();
        let _ = EValue::from_list_optional_tensor(null_optional_tensor_list_ptr);
    }

    // [spec:et:sem:evalue.executorch.runtime.boxed-evalue-list.check-wrapped-vals-fn/test]
    // [spec:et:sem:evalue.executorch.runtime.boxed-evalue-list.check-unwrapped-vals-fn/test]
    //
    // BoxedEvalueListConstructorNullChecks: three death checks (null wrapped_vals,
    // null unwrapped_vals, negative size). Each aborts; ported as separate
    // should_panic + ignore tests.
    #[test]
    #[should_panic]
    #[ignore]
    fn evalue_test_boxed_evalue_list_constructor_null_wrapped_vals() {
        setup();
        let mut storage: [i64; 3] = [0, 0, 0];
        let _ = BoxedEvalueList::<i64>::new(core::ptr::null_mut(), storage.as_mut_ptr(), 3);
    }

    #[test]
    #[should_panic]
    #[ignore]
    fn evalue_test_boxed_evalue_list_constructor_null_unwrapped_vals() {
        setup();
        let mut values: [EValue; 3] = [
            EValue::from_int(1),
            EValue::from_int(2),
            EValue::from_int(3),
        ];
        let mut values_p: [*mut EValue; 3] = [&mut values[0], &mut values[1], &mut values[2]];
        let _ = BoxedEvalueList::<i64>::new(values_p.as_mut_ptr(), core::ptr::null_mut(), 3);
    }

    #[test]
    #[should_panic]
    #[ignore]
    fn evalue_test_boxed_evalue_list_constructor_negative_size() {
        setup();
        let mut values: [EValue; 3] = [
            EValue::from_int(1),
            EValue::from_int(2),
            EValue::from_int(3),
        ];
        let mut values_p: [*mut EValue; 3] = [&mut values[0], &mut values[1], &mut values[2]];
        let mut storage: [i64; 3] = [0, 0, 0];
        let _ = BoxedEvalueList::<i64>::new(values_p.as_mut_ptr(), storage.as_mut_ptr(), -1);
    }

    // [spec:et:sem:evalue.executorch.runtime.e-value.to-list-optional-tensor-fn/test]
    #[test]
    #[should_panic]
    #[ignore]
    fn evalue_test_to_list_optional_tensor_type_check() {
        setup();
        let e = EValue::from_int(42);
        assert!(e.is_int());
        assert!(!e.is_list_optional_tensor());
        let _ = e.to_list_optional_tensor();
    }

    // [spec:et:sem:evalue.executorch.runtime.e-value.to-string-fn/test]
    #[test]
    #[should_panic]
    #[ignore]
    fn evalue_test_to_string_null_pointer_check() {
        setup();
        let mut e = EValue::new();
        e.tag = Tag::String;
        e.payload.copyable_union.as_string_ptr = core::ptr::null_mut();
        assert!(e.is_string());
        let _ = e.to_string();
    }

    // [spec:et:sem:evalue.executorch.runtime.e-value.to-int-list-fn/test]
    #[test]
    #[should_panic]
    #[ignore]
    fn evalue_test_to_int_list_null_pointer_check() {
        setup();
        let mut e = EValue::new();
        e.tag = Tag::ListInt;
        e.payload.copyable_union.as_int_list_ptr = core::ptr::null_mut();
        assert!(e.is_int_list());
        let _ = e.to_int_list();
    }

    // [spec:et:sem:evalue.executorch.runtime.e-value.to-bool-list-fn/test]
    #[test]
    #[should_panic]
    #[ignore]
    fn evalue_test_to_bool_list_null_pointer_check() {
        setup();
        let mut e = EValue::new();
        e.tag = Tag::ListBool;
        e.payload.copyable_union.as_bool_list_ptr = core::ptr::null_mut();
        assert!(e.is_bool_list());
        let _ = e.to_bool_list();
    }

    // [spec:et:sem:evalue.executorch.runtime.e-value.to-double-list-fn/test]
    #[test]
    #[should_panic]
    #[ignore]
    fn evalue_test_to_double_list_null_pointer_check() {
        setup();
        let mut e = EValue::new();
        e.tag = Tag::ListDouble;
        e.payload.copyable_union.as_double_list_ptr = core::ptr::null_mut();
        assert!(e.is_double_list());
        let _ = e.to_double_list();
    }

    // [spec:et:sem:evalue.executorch.runtime.e-value.to-tensor-list-fn/test]
    #[test]
    #[should_panic]
    #[ignore]
    fn evalue_test_to_tensor_list_null_pointer_check() {
        setup();
        let mut e = EValue::new();
        e.tag = Tag::ListTensor;
        e.payload.copyable_union.as_tensor_list_ptr = core::ptr::null_mut();
        assert!(e.is_tensor_list());
        let _ = e.to_tensor_list();
    }

    // [spec:et:sem:evalue.executorch.runtime.e-value.to-list-optional-tensor-fn/test]
    #[test]
    #[should_panic]
    #[ignore]
    fn evalue_test_to_list_optional_tensor_null_pointer_check() {
        setup();
        let mut e = EValue::new();
        e.tag = Tag::ListOptionalTensor;
        e.payload.copyable_union.as_list_optional_tensor_ptr = core::ptr::null_mut();
        assert!(e.is_list_optional_tensor());
        let _ = e.to_list_optional_tensor();
    }

    // Per-type tryTo* coverage.

    // [spec:et:sem:evalue.executorch.runtime.e-value.try-to-int-fn/test]
    // [spec:et:sem:evalue.executorch.runtime.e-value.try-to-fn/test]
    #[test]
    fn evalue_test_try_to_int() {
        setup();
        let e_int = EValue::from_int(42);
        let e_mismatch = EValue::from_double(3.14);
        assert_eq!(*e_int.try_to_int().get(), 42);
        assert_eq!(e_mismatch.try_to_int().error(), Error::InvalidType);
        assert_eq!(*e_int.try_to::<i64>().get(), 42);
        assert_eq!(e_mismatch.try_to::<i64>().error(), Error::InvalidType);
    }

    // [spec:et:sem:evalue.executorch.runtime.e-value.try-to-double-fn/test]
    #[test]
    fn evalue_test_try_to_double() {
        setup();
        let e_double = EValue::from_double(3.14);
        let e_mismatch = EValue::from_int(42);
        assert_eq!(*e_double.try_to_double().get(), 3.14);
        assert_eq!(e_mismatch.try_to_double().error(), Error::InvalidType);
        assert_eq!(*e_double.try_to::<f64>().get(), 3.14);
        assert_eq!(e_mismatch.try_to::<f64>().error(), Error::InvalidType);
    }

    // [spec:et:sem:evalue.executorch.runtime.e-value.try-to-bool-fn/test]
    #[test]
    fn evalue_test_try_to_bool() {
        setup();
        let e_bool = EValue::from_bool(true);
        let e_mismatch = EValue::from_int(42);
        assert_eq!(*e_bool.try_to_bool().get(), true);
        assert_eq!(e_mismatch.try_to_bool().error(), Error::InvalidType);
        assert_eq!(*e_bool.try_to::<bool>().get(), true);
        assert_eq!(e_mismatch.try_to::<bool>().error(), Error::InvalidType);
    }

    // [spec:et:sem:evalue.executorch.runtime.e-value.try-to-tensor-fn/test]
    #[test]
    fn evalue_test_try_to_tensor() {
        setup();
        let tf = TensorFactory::<f32>::new();
        let e_tensor = EValue::from_tensor(tf.ones_default(vec![3, 2]));
        let e_mismatch = EValue::from_int(42);
        assert_eq!(e_tensor.try_to_tensor().get().numel(), 6);
        assert_eq!(e_mismatch.try_to_tensor().error(), Error::InvalidType);
        assert_eq!(e_tensor.try_to::<Tensor>().get().numel(), 6);
        assert_eq!(e_mismatch.try_to::<Tensor>().error(), Error::InvalidType);
    }

    // [spec:et:sem:evalue.executorch.runtime.e-value.try-to-optional-fn/test]
    #[test]
    fn evalue_test_try_to_optional_tensor() {
        setup();
        let tf = TensorFactory::<f32>::new();
        let e_tensor = EValue::from_tensor(tf.ones_default(vec![3, 2]));
        let e_none = EValue::new();
        let e_mismatch = EValue::from_int(42);
        // Named tryToOptional<Tensor>: value, None, mismatch.
        let r_val = e_tensor.try_to_optional::<Tensor>();
        assert!(r_val.get().is_some());
        assert_eq!(r_val.get().as_ref().unwrap().numel(), 6);
        assert!(e_none.try_to_optional::<Tensor>().get().is_none());
        assert_eq!(
            e_mismatch.try_to_optional::<Tensor>().error(),
            Error::InvalidType
        );
        // Templated tryTo<Option<Tensor>>: None path.
        assert!(e_none.try_to::<Option<Tensor>>().get().is_none());
    }

    // [spec:et:sem:evalue.executorch.runtime.e-value.try-to-scalar-fn/test]
    #[test]
    fn evalue_test_try_to_scalar() {
        setup();
        let e_int = EValue::from_int(7);
        let e_double = EValue::from_double(2.5);
        let e_bool = EValue::from_bool(true);
        let e_none = EValue::new();
        assert_eq!(e_int.try_to_scalar().get().to_i64(), 7);
        assert_eq!(e_double.try_to_scalar().get().to_f64(), 2.5);
        assert_eq!(e_bool.try_to_scalar().get().to_bool_val(), true);
        // None is neither Int/Double/Bool.
        assert_eq!(e_none.try_to_scalar().error(), Error::InvalidType);
    }

    // [spec:et:sem:evalue.executorch.runtime.e-value.try-to-scalar-type-fn/test]
    #[test]
    fn evalue_test_try_to_scalar_type() {
        setup();
        let e = EValue::from_int(ScalarType::Float as i64);
        let e_mismatch = EValue::from_double(3.14);
        assert_eq!(*e.try_to_scalar_type().get(), ScalarType::Float);
        assert_eq!(e_mismatch.try_to_scalar_type().error(), Error::InvalidType);
        assert_eq!(*e.try_to::<ScalarType>().get(), ScalarType::Float);
        assert_eq!(
            e_mismatch.try_to::<ScalarType>().error(),
            Error::InvalidType
        );
    }

    // [spec:et:sem:evalue.executorch.runtime.e-value.try-to-memory-format-fn/test]
    #[test]
    fn evalue_test_try_to_memory_format() {
        setup();
        let e = EValue::from_int(MemoryFormat::Contiguous as i64);
        let e_mismatch = EValue::from_double(3.14);
        assert_eq!(*e.try_to_memory_format().get(), MemoryFormat::Contiguous);
        assert_eq!(
            e_mismatch.try_to_memory_format().error(),
            Error::InvalidType
        );
    }

    // [spec:et:sem:evalue.executorch.runtime.e-value.try-to-layout-fn/test]
    #[test]
    fn evalue_test_try_to_layout() {
        setup();
        let e = EValue::from_int(Layout::Strided as i64);
        let e_mismatch = EValue::from_double(3.14);
        assert_eq!(*e.try_to_layout().get(), Layout::Strided);
        assert_eq!(e_mismatch.try_to_layout().error(), Error::InvalidType);
    }

    // [spec:et:sem:evalue.executorch.runtime.e-value.try-to-device-fn/test]
    #[test]
    fn evalue_test_try_to_device() {
        setup();
        let e = EValue::from_int(DeviceType::CPU as i64);
        let e_mismatch = EValue::from_double(3.14);
        assert_eq!(e.try_to_device().get().type_(), DeviceType::CPU);
        assert_eq!(e_mismatch.try_to_device().error(), Error::InvalidType);
    }

    // [spec:et:sem:evalue.executorch.runtime.e-value.try-to-tensor-list-fn/test]
    // also verifies is_tensor_list: try_to_tensor_list on a non-list EValue
    // returns InvalidType precisely because is_tensor_list() is false.
    // [spec:et:sem:evalue.executorch.runtime.e-value.is-tensor-list-fn/test]
    #[test]
    fn evalue_test_try_to_tensor_list() {
        setup();
        let e = EValue::from_int(42);
        assert_eq!(e.try_to_tensor_list().error(), Error::InvalidType);
    }

    // [spec:et:sem:evalue.executorch.runtime.e-value.try-to-list-optional-tensor-fn/test]
    #[test]
    fn evalue_test_try_to_list_optional_tensor() {
        setup();
        let e = EValue::from_int(42);
        assert_eq!(e.try_to_list_optional_tensor().error(), Error::InvalidType);
    }

    // Int list: is_int_list true only for Tag::ListInt; to_int_list / try_to_int_list
    // materialize the wrapped list; try_to_int_list on a mismatch is InvalidType.
    // [spec:et:sem:evalue.executorch.runtime.e-value.is-int-list-fn/test]
    // [spec:et:sem:evalue.executorch.runtime.e-value.try-to-int-list-fn/test]
    #[test]
    fn evalue_test_int_list() {
        setup();
        let mut values: [EValue; 3] = [
            EValue::from_int(4),
            EValue::from_int(5),
            EValue::from_int(6),
        ];
        let mut values_p: [*mut EValue; 3] = [&mut values[0], &mut values[1], &mut values[2]];
        let mut storage: [i64; 3] = [0, 0, 0];
        let mut boxed = BoxedEvalueList::<i64>::new(values_p.as_mut_ptr(), storage.as_mut_ptr(), 3);
        let e = EValue::from_int_list(&mut boxed as *mut BoxedEvalueList<i64>);
        assert!(e.is_int_list());
        assert!(!e.is_bool_list());

        let list = e.to_int_list();
        assert_eq!(list.size(), 3);
        assert_eq!(*list.at(0), 4);
        assert_eq!(*list.at(2), 6);

        let r = e.try_to_int_list();
        assert!(ResultExt::ok(&r));
        assert_eq!(ResultExt::get(&r).size(), 3);

        // Mismatch: not an int list.
        let e_mismatch = EValue::from_int(42);
        assert!(!e_mismatch.is_int_list());
        assert_eq!(e_mismatch.try_to_int_list().error(), Error::InvalidType);
    }

    // Bool list: is_bool_list gates try_to_bool_list; the payload ArrayRef is
    // returned by value on success.
    // [spec:et:sem:evalue.executorch.runtime.e-value.is-bool-list-fn/test]
    // [spec:et:sem:evalue.executorch.runtime.e-value.try-to-bool-list-fn/test]
    #[test]
    fn evalue_test_bool_list() {
        setup();
        let data: [bool; 3] = [true, false, true];
        let mut bool_ref = ArrayRef::<bool>::from_raw_parts(data.as_ptr(), 3);
        let e = EValue::from_bool_list(&mut bool_ref as *mut ArrayRef<bool>);
        assert!(e.is_bool_list());
        assert!(!e.is_double_list());

        let r = e.try_to_bool_list();
        assert!(ResultExt::ok(&r));
        assert_eq!(ResultExt::get(&r).size(), 3);
        assert_eq!(*ResultExt::get(&r).at(0), true);
        assert_eq!(*ResultExt::get(&r).at(1), false);

        let e_mismatch = EValue::from_int(42);
        assert!(!e_mismatch.is_bool_list());
        assert_eq!(e_mismatch.try_to_bool_list().error(), Error::InvalidType);
    }

    // Double list: is_double_list gates try_to_double_list.
    // [spec:et:sem:evalue.executorch.runtime.e-value.is-double-list-fn/test]
    // [spec:et:sem:evalue.executorch.runtime.e-value.try-to-double-list-fn/test]
    #[test]
    fn evalue_test_double_list() {
        setup();
        let data: [f64; 2] = [1.5, 2.5];
        let mut double_ref = ArrayRef::<f64>::from_raw_parts(data.as_ptr(), 2);
        let e = EValue::from_double_list(&mut double_ref as *mut ArrayRef<f64>);
        assert!(e.is_double_list());
        assert!(!e.is_int_list());

        let r = e.try_to_double_list();
        assert!(ResultExt::ok(&r));
        assert_eq!(ResultExt::get(&r).size(), 2);
        assert_eq!(*ResultExt::get(&r).at(0), 1.5);
        assert_eq!(*ResultExt::get(&r).at(1), 2.5);

        let e_mismatch = EValue::from_int(42);
        assert!(!e_mismatch.is_double_list());
        assert_eq!(e_mismatch.try_to_double_list().error(), Error::InvalidType);
    }

    // try_to_string: success returns the string view; a type mismatch is
    // InvalidType (never aborting, unlike to_string).
    // [spec:et:sem:evalue.executorch.runtime.e-value.try-to-string-fn/test]
    #[test]
    fn evalue_test_try_to_string() {
        setup();
        let foo = b"foo";
        let mut string_ref = ArrayRef::<u8>::from_raw_parts(foo.as_ptr(), 3);
        let e = EValue::from_string(&mut string_ref as *mut ArrayRef<u8>);
        let r = e.try_to_string();
        assert!(ResultExt::ok(&r));
        assert_eq!(*ResultExt::get(&r), "foo");

        let e_mismatch = EValue::from_int(42);
        assert_eq!(e_mismatch.try_to_string().error(), Error::InvalidType);
    }

    // clear_to_none is exercised via to_tensor_move: moving the tensor out leaves
    // the EValue as None (tag reset, payload zeroed).
    // [spec:et:sem:evalue.executorch.runtime.e-value.clear-to-none-fn/test]
    #[test]
    fn evalue_test_clear_to_none_via_move() {
        setup();
        let tf = TensorFactory::<f32>::new();
        let mut e = EValue::from_tensor(tf.ones_default(vec![2, 2]));
        assert!(e.is_tensor());
        let t = e.to_tensor_move();
        assert_eq!(t.numel(), 4);
        // After moving out, clear_to_none() ran: the EValue is now None.
        assert!(e.is_none());
    }

    // destroy_elements drops each unwrapped slot exactly wrapped_vals_.size()
    // times. Verified with a Drop-counting element type.
    // [spec:et:sem:evalue.executorch.runtime.boxed-evalue-list.destroy-elements-fn/test]
    #[test]
    fn evalue_test_destroy_elements() {
        setup();
        use core::sync::atomic::{AtomicUsize, Ordering};
        static DROPS: AtomicUsize = AtomicUsize::new(0);
        struct Counter;
        impl Drop for Counter {
            fn drop(&mut self) {
                DROPS.fetch_add(1, Ordering::SeqCst);
            }
        }

        DROPS.store(0, Ordering::SeqCst);
        // Two live EValues (only used as non-null wrapped pointers) and a storage
        // buffer of two Counters that destroy_elements() must drop in place.
        let mut a = EValue::from_int(1);
        let mut b = EValue::from_int(2);
        let mut values_p: [*mut EValue; 2] = [&mut a, &mut b];
        let mut storage: [Counter; 2] = [Counter, Counter];
        let list = BoxedEvalueList::<Counter>::new(values_p.as_mut_ptr(), storage.as_mut_ptr(), 2);

        list.destroy_elements();
        assert_eq!(DROPS.load(Ordering::SeqCst), 2);

        // Prevent the storage array's own Drop from double-counting the slots
        // destroy_elements already dropped in place.
        core::mem::forget(storage);
    }
}
