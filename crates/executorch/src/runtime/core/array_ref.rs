//! Literal port of runtime/core/array_ref.h.
//!
//! removed llvm-specific functionality
//! removed some implicit const -> non-const conversions that rely on
//! complicated std::enable_if meta-programming
//! removed a bunch of slice variants for simplicity...
//! remove constructors and operators for std::vector
//! removed some prevention of accidental assignments from temporary that
//! required std::enable_if meta-programming
//! removed reverse iterator

/// Represents a constant reference to an array (0 or more elements
/// consecutively in memory), i.e. a start pointer and a length. It allows
/// various APIs to take consecutive elements easily and conveniently.
///
/// This class does not own the underlying data, it is expected to be used in
/// situations where the data resides in some other buffer, whose lifetime
/// extends past that of the ArrayRef. For this reason, it is not in general
/// safe to store an ArrayRef.
///
/// Span and ArrayRef are extrememly similar with the difference being ArrayRef
/// views a list of constant elements and Span views a list of mutable elements.
///
/// This is intended to be trivially copyable, so it should be passed by value.
// [spec:et:def:array-ref.executorch.runtime.array-ref]
pub struct ArrayRef<T> {
    /// The start of the array, in an external buffer.
    data: *const T,
    /// The number of elements.
    length: usize,
}

// Trivially copyable (holds only a pointer + length), independent of `T`;
// mirrors the C++ "intended to be trivially copyable, passed by value".
impl<T> Clone for ArrayRef<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for ArrayRef<T> {}

pub type SizeType = usize;

impl<T> ArrayRef<T> {
    // @name Constructors

    /// Construct an empty ArrayRef.
    pub const fn new() -> Self {
        ArrayRef {
            data: core::ptr::null(),
            length: 0,
        }
    }

    /// Construct a ArrayRef from a single element. Implicitly convert element
    /// type. It is aligned with PyTorch's c10::ArrayRef.
    pub fn from_single(one_elt: &T) -> Self {
        ArrayRef {
            data: one_elt as *const T,
            length: 1,
        }
    }

    /// Construct a ArrayRef from a pointer and length.
    // [spec:et:def:array-ref.executorch.runtime.array-ref.array-ref-fn]
    // [spec:et:sem:array-ref.executorch.runtime.array-ref.array-ref-fn]
    pub fn from_raw_parts(data: *const T, length: usize) -> Self {
        let a = ArrayRef { data, length };
        // PORT-NOTE: C++ enforces `ET_DCHECK(Data != nullptr || Length == 0)`
        // (debug-only, elided in release). `debug_assert!` is the equivalent.
        debug_assert!(!a.data.is_null() || a.length == 0);
        a
    }

    /// Construct a ArrayRef from a range.
    pub fn from_range(begin: *const T, end: *const T) -> Self {
        ArrayRef {
            data: begin,
            length: unsafe { end.offset_from(begin) as usize },
        }
    }

    // @name Simple Operations

    // [spec:et:def:array-ref.executorch.runtime.array-ref.begin-fn]
    // [spec:et:sem:array-ref.executorch.runtime.array-ref.begin-fn]
    pub const fn begin(&self) -> *const T {
        self.data
    }
    // [spec:et:def:array-ref.executorch.runtime.array-ref.end-fn]
    // [spec:et:sem:array-ref.executorch.runtime.array-ref.end-fn]
    pub const fn end(&self) -> *const T {
        unsafe { self.data.add(self.length) }
    }

    // These are actually the same as iterator, since ArrayRef only
    // gives you const iterators.
    // [spec:et:def:array-ref.executorch.runtime.array-ref.cbegin-fn]
    // [spec:et:sem:array-ref.executorch.runtime.array-ref.cbegin-fn]
    pub const fn cbegin(&self) -> *const T {
        self.data
    }
    // [spec:et:def:array-ref.executorch.runtime.array-ref.cend-fn]
    // [spec:et:sem:array-ref.executorch.runtime.array-ref.cend-fn]
    pub const fn cend(&self) -> *const T {
        unsafe { self.data.add(self.length) }
    }

    /// empty - Check if the array is empty.
    // [spec:et:def:array-ref.executorch.runtime.array-ref.empty-fn]
    // [spec:et:sem:array-ref.executorch.runtime.array-ref.empty-fn]
    pub const fn empty(&self) -> bool {
        self.length == 0
    }

    // [spec:et:def:array-ref.executorch.runtime.array-ref.data-fn]
    // [spec:et:sem:array-ref.executorch.runtime.array-ref.data-fn]
    pub const fn data(&self) -> *const T {
        self.data
    }

    /// size - Get the array size.
    // [spec:et:def:array-ref.executorch.runtime.array-ref.size-fn]
    // [spec:et:sem:array-ref.executorch.runtime.array-ref.size-fn]
    pub const fn size(&self) -> usize {
        self.length
    }

    /// front - Get the first element.
    // [spec:et:def:array-ref.executorch.runtime.array-ref.front-fn]
    // [spec:et:sem:array-ref.executorch.runtime.array-ref.front-fn]
    pub fn front(&self) -> &T {
        // ArrayRef: attempted to access front() of empty list
        // PORT-NOTE: C++ `ET_CHECK(!empty())` fatally aborts; `assert!` stands
        // in for the platform assert group's fatal check (still a stub).
        assert!(!self.empty());
        unsafe { &*self.data }
    }

    /// back - Get the last element.
    // [spec:et:def:array-ref.executorch.runtime.array-ref.back-fn]
    // [spec:et:sem:array-ref.executorch.runtime.array-ref.back-fn]
    pub fn back(&self) -> &T {
        // ArrayRef: attempted to access back() of empty list
        assert!(!self.empty());
        unsafe { &*self.data.add(self.length - 1) }
    }

    /// slice(n, m) - Take M elements of the array starting at element N
    // [spec:et:def:array-ref.executorch.runtime.array-ref.slice-fn]
    // [spec:et:sem:array-ref.executorch.runtime.array-ref.slice-fn]
    pub fn slice(&self, n: usize, m: usize) -> ArrayRef<T> {
        // cant slice longer then the array
        let end: usize;
        // PORT-NOTE: C++ uses `c10::add_overflows(N, M, &end)`; `checked_add`
        // is the equivalent checked addition. The `ET_CHECK` fatally aborts.
        match n.checked_add(m) {
            Some(sum) => {
                end = sum;
                assert!(end <= self.size());
            }
            None => {
                assert!(false, "add_overflows");
                end = 0;
            }
        }
        let _ = end;
        ArrayRef::from_raw_parts(unsafe { self.data().add(n) }, m)
    }

    /// slice(n) - Chop off the first N elements of the array.
    pub fn slice_from(&self, n: usize) -> ArrayRef<T> {
        self.slice(n, self.size() - n)
    }

    // @name Operator Overloads
    // [spec:et:def:array-ref.executorch.runtime.array-ref.operator-fn]
    // [spec:et:sem:array-ref.executorch.runtime.array-ref.operator-fn]
    ///
    /// # Safety
    /// Unchecked subscript; behavior is undefined for `index >= length`.
    pub unsafe fn index(&self, index: usize) -> &T {
        unsafe { &*self.data.add(index) }
    }

    /// Vector compatibility
    // [spec:et:def:array-ref.executorch.runtime.array-ref.at-fn]
    // [spec:et:sem:array-ref.executorch.runtime.array-ref.at-fn]
    pub fn at(&self, index: usize) -> &T {
        // invalid index
        assert!(index < self.length);
        unsafe { &*self.data.add(index) }
    }
}

impl<T: PartialEq> ArrayRef<T> {
    /// equals - Check for element-wise equality.
    // [spec:et:def:array-ref.executorch.runtime.array-ref.equals-fn]
    // [spec:et:sem:array-ref.executorch.runtime.array-ref.equals-fn]
    pub fn equals(&self, rhs: ArrayRef<T>) -> bool {
        if self.length != rhs.length {
            return false;
        }
        for i in 0..self.length {
            if unsafe { *self.data.add(i) != *rhs.data.add(i) } {
                return false;
            }
        }
        true
    }
}

impl<T> Default for ArrayRef<T> {
    fn default() -> Self {
        ArrayRef::new()
    }
}

// @name ArrayRef Convenience constructors

/// Construct an ArrayRef from a single element.
// [spec:et:def:array-ref.executorch.runtime.make-array-ref-fn]
// [spec:et:sem:array-ref.executorch.runtime.make-array-ref-fn]
pub fn make_array_ref<T>(one_elt: &T) -> ArrayRef<T> {
    ArrayRef::from_single(one_elt)
}

/// Construct an ArrayRef from a pointer and length.
pub fn make_array_ref_from_raw_parts<T>(data: *const T, length: usize) -> ArrayRef<T> {
    ArrayRef::from_raw_parts(data, length)
}

/// Construct an ArrayRef from a range.
pub fn make_array_ref_from_range<T>(begin: *const T, end: *const T) -> ArrayRef<T> {
    ArrayRef::from_range(begin, end)
}

// WARNING: Template instantiation will NOT be willing to do an implicit
// conversions to get you to an ArrayRef, which is why we need so
// many overloads.

// [spec:et:def:array-ref.executorch.runtime.operator-fn]
// [spec:et:sem:array-ref.executorch.runtime.operator-fn]
impl<T: PartialEq> PartialEq for ArrayRef<T> {
    fn eq(&self, other: &Self) -> bool {
        self.equals(*other)
    }
}

pub type IntArrayRef = ArrayRef<i64>;

#[cfg(test)]
mod tests {
    use super::*;

    // [spec:et:sem:array-ref.executorch.runtime.array-ref.array-ref-fn/test]
    // [spec:et:sem:array-ref.executorch.runtime.make-array-ref-fn/test]
    // [spec:et:sem:array-ref.executorch.runtime.array-ref.size-fn/test]
    #[test]
    fn test_array_ref_implicit_type_conversion() {
        // C++ `ArrayRef<int64_t> oneElement_1 = {1};` builds a single-element
        // ArrayRef via the implicit single-element ctor.
        let e1: i64 = 1;
        let one_element_1: ArrayRef<i64> = ArrayRef::from_single(&e1);
        assert_eq!(one_element_1.size(), 1);

        // C++ `ArrayRef<int64_t> oneElement_2 = 1;` — same single-element ctor.
        let e2: i64 = 1;
        let one_element_2: ArrayRef<i64> = ArrayRef::from_single(&e2);
        assert_eq!(one_element_2.size(), 1);
    }

    // begin() returns Data, end() returns Data + Length, and for an empty
    // ArrayRef begin() == end(); data() returns the stored pointer verbatim.
    // [spec:et:sem:array-ref.executorch.runtime.array-ref.begin-fn/test]
    // [spec:et:sem:array-ref.executorch.runtime.array-ref.end-fn/test]
    // [spec:et:sem:array-ref.executorch.runtime.array-ref.data-fn/test]
    // [spec:et:sem:array-ref.executorch.runtime.array-ref.cbegin-fn/test]
    // [spec:et:sem:array-ref.executorch.runtime.array-ref.cend-fn/test]
    #[test]
    fn test_array_ref_begin_end_data() {
        let buf: [i64; 3] = [10, 20, 30];
        let a = ArrayRef::<i64>::from_raw_parts(buf.as_ptr(), 3);
        assert_eq!(a.data(), buf.as_ptr());
        assert_eq!(a.begin(), buf.as_ptr());
        assert_eq!(a.cbegin(), buf.as_ptr());
        assert_eq!(a.end(), unsafe { buf.as_ptr().add(3) });
        assert_eq!(a.cend(), unsafe { buf.as_ptr().add(3) });
        assert_eq!(unsafe { a.end().offset_from(a.begin()) }, 3);

        // Empty: begin() == end().
        let e = ArrayRef::<i64>::new();
        assert_eq!(e.begin(), e.end());
        assert!(e.data().is_null());
    }

    // empty() is true iff Length == 0, independent of the Data pointer.
    // [spec:et:sem:array-ref.executorch.runtime.array-ref.empty-fn/test]
    #[test]
    fn test_array_ref_empty() {
        let e = ArrayRef::<i64>::new();
        assert!(e.empty());

        let buf: [i64; 2] = [1, 2];
        let a = ArrayRef::<i64>::from_raw_parts(buf.as_ptr(), 2);
        assert!(!a.empty());

        // Non-null data but zero length is still empty.
        let z = ArrayRef::<i64>::from_raw_parts(buf.as_ptr(), 0);
        assert!(z.empty());
    }

    // at(i) returns Data[i] with bounds validation.
    // [spec:et:sem:array-ref.executorch.runtime.array-ref.at-fn/test]
    #[test]
    fn test_array_ref_at() {
        let buf: [i64; 3] = [7, 8, 9];
        let a = ArrayRef::<i64>::from_raw_parts(buf.as_ptr(), 3);
        assert_eq!(*a.at(0), 7);
        assert_eq!(*a.at(1), 8);
        assert_eq!(*a.at(2), 9);
    }

    // at() aborts (assert) when Index >= Length.
    // [spec:et:sem:array-ref.executorch.runtime.array-ref.at-fn/test]
    #[test]
    #[should_panic]
    fn test_array_ref_at_out_of_bounds() {
        let buf: [i64; 2] = [1, 2];
        let a = ArrayRef::<i64>::from_raw_parts(buf.as_ptr(), 2);
        let _ = a.at(2);
    }

    // slice(N, M) returns a sub-view of M elements starting at index N over the
    // same backing buffer; slice_from(N) chops the first N elements.
    // [spec:et:sem:array-ref.executorch.runtime.array-ref.slice-fn/test]
    #[test]
    fn test_array_ref_slice() {
        let buf: [i64; 5] = [0, 1, 2, 3, 4];
        let a = ArrayRef::<i64>::from_raw_parts(buf.as_ptr(), 5);

        let s = a.slice(1, 3);
        assert_eq!(s.size(), 3);
        assert_eq!(s.data(), unsafe { buf.as_ptr().add(1) });
        assert_eq!(*s.at(0), 1);
        assert_eq!(*s.at(2), 3);

        // slice up to the end.
        let tail = a.slice_from(2);
        assert_eq!(tail.size(), 3);
        assert_eq!(*tail.at(0), 2);
        assert_eq!(*tail.at(2), 4);

        // Empty slice at the end is allowed (N == size(), M == 0).
        let empty = a.slice(5, 0);
        assert_eq!(empty.size(), 0);
    }

    // slice() aborts when N + M exceeds size().
    // [spec:et:sem:array-ref.executorch.runtime.array-ref.slice-fn/test]
    #[test]
    #[should_panic]
    fn test_array_ref_slice_out_of_bounds() {
        let buf: [i64; 3] = [0, 1, 2];
        let a = ArrayRef::<i64>::from_raw_parts(buf.as_ptr(), 3);
        let _ = a.slice(2, 2);
    }

    // equals(): length mismatch -> false; same length + same elements -> true;
    // differing element -> false; two empty ArrayRefs compare equal.
    // [spec:et:sem:array-ref.executorch.runtime.array-ref.equals-fn/test]
    // [spec:et:sem:array-ref.executorch.runtime.operator-fn/test]
    #[test]
    fn test_array_ref_equals() {
        let a: [i64; 3] = [1, 2, 3];
        let b: [i64; 3] = [1, 2, 3];
        let c: [i64; 3] = [1, 9, 3];
        let d: [i64; 2] = [1, 2];

        let ra = ArrayRef::<i64>::from_raw_parts(a.as_ptr(), 3);
        let rb = ArrayRef::<i64>::from_raw_parts(b.as_ptr(), 3);
        let rc = ArrayRef::<i64>::from_raw_parts(c.as_ptr(), 3);
        let rd = ArrayRef::<i64>::from_raw_parts(d.as_ptr(), 2);

        // Same length, equal elements (distinct buffers) -> equal by value.
        assert!(ra.equals(rb));
        // PORT-NOTE: `ArrayRef` does not derive `Debug`, so `assert_eq!`/
        // `assert_ne!` (which need `Debug` to format on failure) fail to compile.
        // Use `==`/`!=` directly — same `PartialEq` behavior, no `Debug` bound.
        assert!(ra == rb);
        // Same length, one differing element -> not equal.
        assert!(!ra.equals(rc));
        assert!(ra != rc);
        // Length mismatch -> not equal.
        assert!(!ra.equals(rd));

        // Two empty ArrayRefs (different/null data) compare equal.
        let e1 = ArrayRef::<i64>::new();
        let e2 = ArrayRef::<i64>::from_raw_parts(a.as_ptr(), 0);
        assert!(e1.equals(e2));
    }
}
