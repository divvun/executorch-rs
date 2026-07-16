# runtime/core/array_ref.h

> [spec:et:def:array-ref.executorch.runtime.array-ref]
> class ArrayRef final {
>   const T* Data;
>   size_type Length;
> }

> [spec:et:def:array-ref.executorch.runtime.array-ref.array-ref-fn]
> ArrayRef(const T* data, size_t length) : Data(data), Length(length)

> [spec:et:sem:array-ref.executorch.runtime.array-ref.array-ref-fn]
> Constructs an `ArrayRef<T>` from a raw pointer `data` and an element
> count `length`. Stores `data` into the `Data` field and `length` into
> the `Length` field verbatim; performs no copy of the pointed-to
> elements (the `ArrayRef` is a non-owning view whose validity is tied to
> the caller's buffer lifetime).
>
> Debug-only invariant: asserts (`ET_DCHECK`) that `Data != nullptr ||
> Length == 0`, i.e. a null data pointer is permitted only when the
> length is zero. In release builds the check is compiled out and no
> validation occurs. The Rust port should encode this as a debug
> assertion / precondition; it is not a checked runtime error.

> [spec:et:def:array-ref.executorch.runtime.array-ref.begin-fn]
> constexpr iterator begin() const

> [spec:et:sem:array-ref.executorch.runtime.array-ref.begin-fn]
> Returns the `Data` pointer as an iterator to the first element. The
> iterator type is `const T*`. For an empty `ArrayRef` this equals
> `end()`. No bounds validation.

> [spec:et:def:array-ref.executorch.runtime.array-ref.cbegin-fn]
> constexpr const_iterator cbegin() const

> [spec:et:sem:array-ref.executorch.runtime.array-ref.cbegin-fn]
> Returns the `Data` pointer as a `const_iterator` (`const T*`) to the
> first element. Identical to `[spec:et:sem:array-ref.executorch.runtime.array-ref.begin-fn]`
> because `ArrayRef` only ever exposes const iterators.

> [spec:et:def:array-ref.executorch.runtime.array-ref.cend-fn]
> constexpr const_iterator cend() const

> [spec:et:sem:array-ref.executorch.runtime.array-ref.cend-fn]
> Returns `Data + Length` as a `const_iterator` (`const T*`), the
> one-past-the-end position. Identical to `[spec:et:sem:array-ref.executorch.runtime.array-ref.end-fn]`.
> Uses pointer arithmetic; not dereferenceable.

> [spec:et:def:array-ref.executorch.runtime.array-ref.data-fn]
> constexpr const T* data() const

> [spec:et:sem:array-ref.executorch.runtime.array-ref.data-fn]
> Returns the stored `Data` pointer (`const T*`) to the start of the
> viewed array. May be null when the `ArrayRef` was default-constructed
> or otherwise empty. No copy, no validation.

> [spec:et:def:array-ref.executorch.runtime.array-ref.empty-fn]
> constexpr bool empty() const

> [spec:et:sem:array-ref.executorch.runtime.array-ref.empty-fn]
> Returns `true` if and only if `Length == 0`, otherwise `false`. Does
> not inspect the `Data` pointer.

> [spec:et:def:array-ref.executorch.runtime.array-ref.end-fn]
> constexpr iterator end() const

> [spec:et:sem:array-ref.executorch.runtime.array-ref.end-fn]
> Returns `Data + Length` (`const T*`), the one-past-the-end iterator.
> Computed by pointer arithmetic on the stored pointer and length; the
> result is not dereferenceable. For an empty `ArrayRef` this equals
> `begin()`.

> [spec:et:def:array-ref.executorch.runtime.array-ref.equals-fn]
> bool equals(ArrayRef RHS) const

> [spec:et:sem:array-ref.executorch.runtime.array-ref.equals-fn]
> Element-wise equality against `RHS` (another `ArrayRef<T>`).
>
> Step 1: If `this->Length != RHS.Length`, return `false` immediately.
>
> Step 2: Otherwise iterate `i` from `0` to `Length - 1` inclusive in
> ascending order. At each index compare `Data[i] != RHS.Data[i]` using
> `T`'s `operator!=`; if any pair differs, return `false`.
>
> Step 3: If all elements compared equal, return `true`. Two empty
> (zero-length) `ArrayRef`s therefore compare equal regardless of their
> `Data` pointers. Comparison is by value of elements, not by pointer
> identity.

> [spec:et:def:array-ref.executorch.runtime.array-ref.size-fn]
> constexpr size_t size() const

> [spec:et:sem:array-ref.executorch.runtime.array-ref.size-fn]
> Returns the stored `Length` (element count, `size_t`). No computation
> or validation.

> [spec:et:def:array-ref.executorch.runtime.array-ref.slice-fn]
> ArrayRef<T> slice(size_t N, size_t M) const

> [spec:et:sem:array-ref.executorch.runtime.array-ref.slice-fn]
> Returns a sub-view of `M` elements starting at index `N`:
> `ArrayRef<T>(data() + N, M)`.
>
> Bounds validation: computes `end = N + M` using a checked addition. If
> the addition overflows `size_t`, or if `end > size()`, the `ET_CHECK`
> fails and the program aborts (fatal, non-recoverable — not an `Error`
> return). Otherwise constructs and returns a new `ArrayRef` viewing the
> same backing buffer offset by `N` with length `M`. The single-argument
> `slice(N)` overload (not separately annotated) delegates to this with
> `M = size() - N`, chopping off the first `N` elements.

> [spec:et:def:array-ref.executorch.runtime.make-array-ref-fn]
> ArrayRef<T> makeArrayRef(const T& OneElt)

> [spec:et:sem:array-ref.executorch.runtime.make-array-ref-fn]
> Free-function convenience constructor. Given a single element reference
> `OneElt`, returns an `ArrayRef<T>` of length 1 viewing that element.
> Implemented by returning `OneElt` directly, relying on `ArrayRef`'s
> implicit single-element constructor which stores `&OneElt` as `Data`
> and `1` as `Length`. The caller must keep `OneElt` alive for the
> lifetime of the returned view. Overloads of `makeArrayRef` for
> (pointer,length), (begin,end), `std::array`, C arrays, and pass-through
> of an existing `ArrayRef` exist but are not separately annotated; each
> forwards to the corresponding `ArrayRef` constructor.

> [spec:et:def:array-ref.executorch.runtime.operator-fn]
> bool operator==(ArrayRef<T> a1, ArrayRef<T> a2)

> [spec:et:sem:array-ref.executorch.runtime.operator-fn]
> Free-function `operator==(a1, a2)` for two `ArrayRef<T>` values (passed
> by value). Returns `a1.equals(a2)`, i.e. element-wise equality per
> `[spec:et:sem:array-ref.executorch.runtime.array-ref.equals-fn]`. The
> companion `operator!=` (not separately annotated) returns
> `!a1.equals(a2)`.

> [spec:et:def:array-ref.executorch.runtime.array-ref.at-fn]
> const T& at(size_t Index) const

> [spec:et:sem:array-ref.executorch.runtime.array-ref.at-fn]
> Bounds-checked element access. Asserts (`ET_CHECK`) that
> `Index < Length`; on failure the check aborts the program (fatal,
> non-recoverable). Otherwise returns a const reference to `Data[Index]`.
> Unlike `operator[]`, this variant always validates the index, even in
> release builds.

> [spec:et:def:array-ref.executorch.runtime.array-ref.back-fn]
> const T& back() const

> [spec:et:sem:array-ref.executorch.runtime.array-ref.back-fn]
> Returns a const reference to the last element, `Data[Length - 1]`.
> First asserts (`ET_CHECK`) that the array is not empty (`!empty()`);
> calling `back()` on an empty `ArrayRef` aborts the program (fatal,
> non-recoverable).

> [spec:et:def:array-ref.executorch.runtime.array-ref.front-fn]
> const T& front() const

> [spec:et:sem:array-ref.executorch.runtime.array-ref.front-fn]
> Returns a const reference to the first element, `Data[0]`. First
> asserts (`ET_CHECK`) that the array is not empty (`!empty()`); calling
> `front()` on an empty `ArrayRef` aborts the program (fatal,
> non-recoverable).

> [spec:et:def:array-ref.executorch.runtime.array-ref.operator-fn]
> constexpr const T& operator[](size_t Index) const

> [spec:et:sem:array-ref.executorch.runtime.array-ref.operator-fn]
> Unchecked subscript. Returns a const reference to `Data[Index]` with no
> bounds validation of any kind (behavior is undefined for
> `Index >= Length`). Contrast with `at()` per
> `[spec:et:sem:array-ref.executorch.runtime.array-ref.at-fn]`, which
> always bounds-checks. In the Rust port this corresponds to an unchecked
> index / `get_unchecked`-style access.

