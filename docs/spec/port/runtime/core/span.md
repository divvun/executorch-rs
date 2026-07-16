# runtime/core/span.h

> [spec:et:def:span.executorch.runtime.span]
> class Span final {
>   T* data_;
>   size_type length_;
> }

> [spec:et:def:span.executorch.runtime.span.begin-fn]
> iterator begin() const noexcept

> [spec:et:sem:span.executorch.runtime.span.begin-fn]
> Returns `data_`, the pointer to the first element of the viewed buffer, as the
> begin iterator (iterator is `T*`). For an empty Span (`length_ == 0`) this may
> be null. No side effects; noexcept. `begin()` and `end()` together define the
> half-open range `[data_, data_ + length_)`.

> [spec:et:def:span.executorch.runtime.span.data-fn]
> constexpr T* data() const noexcept

> [spec:et:sem:span.executorch.runtime.span.data-fn]
> Returns `data_`, the raw pointer to the start of the viewed element buffer
> (may be null for a default-constructed/empty Span). Identical value to
> `begin()`. constexpr, noexcept, no side effects.

> [spec:et:def:span.executorch.runtime.span.empty-fn]
> constexpr bool empty() const noexcept

> [spec:et:sem:span.executorch.runtime.span.empty-fn]
> Returns `length_ == 0`: true if the Span views zero elements, false otherwise.
> Does not inspect `data_`. constexpr, noexcept, no side effects.

> [spec:et:def:span.executorch.runtime.span.end-fn]
> iterator end() const noexcept

> [spec:et:sem:span.executorch.runtime.span.end-fn]
> Returns `data_ + length_`, the one-past-the-last-element pointer, as the end
> iterator. Pointer arithmetic is on `T*`, so the offset is `length_` elements
> past `data_`. For an empty Span this equals `begin()`. No side effects;
> noexcept.

> [spec:et:def:span.executorch.runtime.span.size-fn]
> constexpr size_t size() const noexcept

> [spec:et:sem:span.executorch.runtime.span.size-fn]
> Returns `length_`, the number of elements in the Span, as a `size_t`.
> constexpr, noexcept, no side effects.

> [spec:et:def:span.executorch.runtime.span.span-fn]
> Span(T* data, size_t length) : data_(data), length_(length)

> [spec:et:sem:span.executorch.runtime.span.span-fn]
> Constructs a Span from a pointer and a length: stores `data` into `data_` and
> `length` into `length_`. The Span does not own the buffer; the caller must
> ensure the pointed-to memory outlives the Span. Enforces the debug invariant
> `ET_DCHECK(data_ != nullptr || length_ == 0)`: in debug builds, if `data` is
> null while `length` is nonzero, the check fires and aborts; in release builds
> the check is elided (no validation). Other constructors exist: a default
> constructor yielding `{nullptr, 0}`; a `(begin, end)` pointer-pair form with
> `length_ = end - begin`; an implicit form from a C array `T(&)[N]` with
> `length_ = N`; and an implicit single-element form from `T&` with
> `length_ = 1`.

> [spec:et:def:span.executorch.runtime.span.operator-fn]
> T& operator[](size_t index) const

> [spec:et:sem:span.executorch.runtime.span.operator-fn]
> Unchecked subscript: returns a reference (`T&`) to the element at `data_[index]`.
> Performs no bounds checking against `length_`; passing `index >= length_` (or
> any index into a null/empty Span) is undefined behavior. No side effects on
> the Span itself; the returned reference aliases the underlying buffer and may
> be used to mutate it (T is non-const for a mutable Span).

