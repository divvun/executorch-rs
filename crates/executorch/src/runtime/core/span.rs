//! Literal port of runtime/core/span.h.

/// Represent a reference to an array (0 or more elements consecutively in
/// memory), i.e. a start pointer and a length. It allows various APIs to take
/// consecutive elements easily and conveniently.
///
/// This class does not own the underlying data, it is expected to be used in
/// situations where the data resides in some other buffer, whose lifetime
/// extends past that of the Span.
///
/// Span and ArrayRef are extrememly similar with the difference being ArrayRef
/// views a list of constant elements and Span views a list of mutable elements.
///
/// This is intended to be trivially copyable, so it should be passed by value.
// [spec:et:def:span.executorch.runtime.span]
pub struct Span<T> {
    /// The start of the array, in an external buffer.
    data_: *mut T,
    /// The number of elements.
    length_: usize,
}

// Trivially copyable (holds only a pointer + length), independent of `T`;
// mirrors the C++ "intended to be trivially copyable, passed by value".
impl<T> Clone for Span<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for Span<T> {}

pub type SizeType = usize;

impl<T> Span<T> {
    /// Construct an empty Span.
    pub const fn new() -> Self {
        Span {
            data_: core::ptr::null_mut(),
            length_: 0,
        }
    }

    /// Construct a Span from a pointer and length.
    // [spec:et:def:span.executorch.runtime.span.span-fn]
    // [spec:et:sem:span.executorch.runtime.span.span-fn]
    pub fn from_raw_parts(data: *mut T, length: usize) -> Self {
        let s = Span {
            data_: data,
            length_: length,
        };
        // PORT-NOTE: C++ enforces `ET_DCHECK(data_ != nullptr || length_ == 0)`
        // (debug-only, elided in release). `ET_DCHECK` is owned by the platform
        // assert group (still a stub); `debug_assert!` is the equivalent.
        debug_assert!(!s.data_.is_null() || s.length_ == 0);
        s
    }

    /// Construct a Span from a range.
    pub fn from_range(begin: *mut T, end: *mut T) -> Self {
        Span {
            data_: begin,
            length_: unsafe { end.offset_from(begin) as usize },
        }
    }

    /// @returns a pointer to the start of the underlying element buffer.
    // [spec:et:def:span.executorch.runtime.span.begin-fn]
    // [spec:et:sem:span.executorch.runtime.span.begin-fn]
    pub fn begin(&self) -> *mut T {
        self.data_
    }

    /// @returns a pointer to the end of the underlying element buffer.
    // [spec:et:def:span.executorch.runtime.span.end-fn]
    // [spec:et:sem:span.executorch.runtime.span.end-fn]
    pub fn end(&self) -> *mut T {
        unsafe { self.data_.add(self.length_) }
    }

    /// @retval a boolean indicating if the Span is empty.
    // [spec:et:def:span.executorch.runtime.span.empty-fn]
    // [spec:et:sem:span.executorch.runtime.span.empty-fn]
    pub const fn empty(&self) -> bool {
        self.length_ == 0
    }

    /// @returns a pointer to the start of the underlying element buffer.
    // [spec:et:def:span.executorch.runtime.span.data-fn]
    // [spec:et:sem:span.executorch.runtime.span.data-fn]
    pub const fn data(&self) -> *mut T {
        self.data_
    }

    /// @returns the number of elements in the Span.
    // [spec:et:def:span.executorch.runtime.span.size-fn]
    // [spec:et:sem:span.executorch.runtime.span.size-fn]
    pub const fn size(&self) -> usize {
        self.length_
    }

    /// Unchecked index into the array according to the argument index.
    /// @returns a reference to the element at the specified index.
    // [spec:et:def:span.executorch.runtime.span.operator-fn]
    // [spec:et:sem:span.executorch.runtime.span.operator-fn]
    ///
    /// # Safety
    /// Performs no bounds checking; `index >= length_` (or any index into a
    /// null/empty Span) is undefined behavior.
    pub unsafe fn index(&self, index: usize) -> &mut T {
        unsafe { &mut *self.data_.add(index) }
    }
}

impl<T> Default for Span<T> {
    fn default() -> Self {
        Span::new()
    }
}

// Literal port of runtime/core/test/span_test.cpp.
#[cfg(test)]
mod tests {
    use super::*;

    // [spec:et:sem:span.executorch.runtime.span.span-fn/test]
    // [spec:et:sem:span.executorch.runtime.span.size-fn/test]
    // [spec:et:sem:span.executorch.runtime.span.begin-fn/test]
    // [spec:et:sem:span.executorch.runtime.span.end-fn/test]
    #[test]
    fn span_test_ctors() {
        let mut x: [i64; 2] = [1, 2];
        let x_ptr = x.as_mut_ptr();

        // Span<int64_t> span_range = {x, x + 2};
        let span_range = Span::from_range(x_ptr, unsafe { x_ptr.add(2) });
        // Span<int64_t> span_array = {x}; — array constructor over the full array.
        let span_array = Span::from_raw_parts(x_ptr, 2);

        assert_eq!(span_range.size(), 2);
        assert_eq!(span_array.size(), 2);

        assert_eq!(span_range.begin(), x_ptr);
        assert_eq!(span_range.end(), unsafe { x_ptr.add(2) });

        assert_eq!(span_array.begin(), x_ptr);
        assert_eq!(span_array.end(), unsafe { x_ptr.add(2) });
    }

    // [spec:et:sem:span.executorch.runtime.span.size-fn/test]
    // [spec:et:sem:span.executorch.runtime.span.operator-fn/test]
    #[test]
    fn span_test_mutable_elements() {
        let mut x: [i64; 2] = [1, 2];
        let span = Span::from_raw_parts(x.as_mut_ptr(), 2);
        assert_eq!(span.size(), 2);
        assert_eq!(unsafe { *span.index(0) }, 1);
        unsafe { *span.index(0) = 2 };
        assert_eq!(unsafe { *span.index(0) }, 2);
    }

    // [spec:et:sem:span.executorch.runtime.span.empty-fn/test]
    #[test]
    fn span_test_empty() {
        let mut x: [i64; 2] = [1, 2];
        let x_ptr = x.as_mut_ptr();
        let span_full = Span::from_raw_parts(x_ptr, 2);
        let span_empty = Span::from_raw_parts(x_ptr, 0);

        assert!(!span_full.empty());
        assert!(span_empty.empty());
    }

    // [spec:et:sem:span.executorch.runtime.span.data-fn/test]
    #[test]
    fn span_test_data() {
        let mut x: [i64; 2] = [1, 2];
        let x_ptr = x.as_mut_ptr();
        let span = Span::from_raw_parts(x_ptr, 2);
        assert_eq!(span.data(), x_ptr);
    }

    // [spec:et:sem:span.executorch.runtime.span.data-fn/test]
    // [spec:et:sem:span.executorch.runtime.span.size-fn/test]
    #[test]
    fn span_test_trivially_copyable() {
        let mut x: [i64; 2] = [1, 2];
        let span = Span::from_raw_parts(x.as_mut_ptr(), 2);
        let span_copy = span;
        assert_eq!(span.data(), span_copy.data());
        assert_eq!(span.size(), span_copy.size());
        // std::is_trivially_copyable<Span<int64_t>>::value — mirrored by the
        // `Copy` impl on `Span<T>`; the copy above compiled, proving it.
    }

    // The C++ `SingleElementConstructor` exercises the implicit
    // `Span(T& single_value)` ctor. Rust has no implicit single-element
    // conversion; the equivalent is a length-1 span over the value's address.
    // [spec:et:sem:span.executorch.runtime.span.span-fn/test]
    // [spec:et:sem:span.executorch.runtime.span.size-fn/test]
    // [spec:et:sem:span.executorch.runtime.span.data-fn/test]
    // [spec:et:sem:span.executorch.runtime.span.operator-fn/test]
    // [spec:et:sem:span.executorch.runtime.span.begin-fn/test]
    // [spec:et:sem:span.executorch.runtime.span.end-fn/test]
    #[test]
    fn span_test_single_element_constructor() {
        let mut single_value: i64 = 42;
        let span = Span::from_raw_parts(&mut single_value as *mut i64, 1);

        assert_eq!(span.size(), 1);
        assert_eq!(span.data(), &mut single_value as *mut i64);
        assert_eq!(unsafe { *span.index(0) }, 42);
        assert_eq!(unsafe { *span.begin() }, 42);
        assert_eq!(span.end(), unsafe { span.begin().add(1) });

        // Test that modifying through span affects original value
        unsafe { *span.index(0) = 100 };
        assert_eq!(single_value, 100);
        assert_eq!(unsafe { *span.index(0) }, 100);
    }
}
