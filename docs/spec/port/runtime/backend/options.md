# runtime/backend/options.h

> [spec:et:def:options.executorch.runtime.backend-option]
> struct BackendOption {
>   char key[kMaxOptionKeyLength]{};
>   OptionValue value;
> }

> [spec:et:def:options.executorch.runtime.backend-options]
> class BackendOptions {
>   BackendOption options_[MaxCapacity]{};
>   size_t size_;
> }

> [spec:et:def:options.executorch.runtime.backend-options.backend-options-fn]
> BackendOptions(const BackendOptions& other) : size_(other.size_)

> [spec:et:sem:options.executorch.runtime.backend-options.backend-options-fn]
> Copy constructor for `BackendOptions<MaxCapacity>`. Sets this object's
> `size_` to `other.size_`, then copies the first `size_` `BackendOption`
> entries from `other.options_` into this `options_` array, in ascending
> index order (`i` from `0` to `size_ - 1`). Each `BackendOption` is
> copied by value, which copies the fixed 64-byte `key` char array and
> the `OptionValue` variant (bool / int / 256-byte char array). Slots at
> indices `>= size_` are left as default-initialized and are not read.
> Note string option values are stored as owned fixed-size char arrays
> inside the variant, so this is a deep copy of the value contents.

> [spec:et:def:options.executorch.runtime.backend-options.get-option-fn]
> Error get_option(const char (&key)[KeyLen], T& out) const

> [spec:et:sem:options.executorch.runtime.backend-options.get-option-fn]
> Looks up an option by key and writes its value into `out`. Template
> parameters: `T` is the expected value type (`bool`, `int`, or `const
> char*`), `KeyLen` is the key array length. Compile-time: `static_assert`
> that `KeyLen <= kMaxOptionKeyLength` (64).
>
> Iterate `i` from `0` to `size_ - 1` in ascending order. For each stored
> option compare `options_[i].key` against `key` with `std::strcmp`
> (C-string compare, first null-terminated mismatch). On the first key
> match:
>
> - If `T` is `const char*`: attempt `std::get_if<std::array<char,
>   kMaxOptionValueLength>>` on the stored variant. If the variant
>   currently holds the char-array alternative, set `out` to a pointer to
>   that array's data (points into the stored option, valid while this
>   `BackendOptions` is alive and unmodified) and return `Error::Ok`.
> - Otherwise (`T` is `bool` or `int`): attempt `std::get_if<T>` on the
>   variant. If the variant currently holds alternative `T`, set `out` to
>   the held value and return `Error::Ok`.
> - If the matched key's variant does not hold the requested alternative
>   (type mismatch), return `Error::InvalidArgument` (does not continue
>   searching).
>
> If no key matches after scanning all `size_` entries, return
> `Error::NotFound`. `out` is left unmodified on any error path.

> [spec:et:def:options.executorch.runtime.backend-options.set-option-fn]
> Error set_option(const char (&key)[N], const char* value) noexcept

> [spec:et:sem:options.executorch.runtime.backend-options.set-option-fn]
> String-valued overload of `set_option`. `N` is the key array length;
> compile-time `static_assert` that `N <= kMaxOptionKeyLength` (64).
> `noexcept`.
>
> Builds a fixed-size value buffer `std::array<char, kMaxOptionValueLength>`
> (256), zero-initialized. Copies at most `kMaxOptionValueLength - 1`
> (255) bytes of the null-terminated C-string `value` into it via
> `strncpy`, then forcibly sets the last byte (index 255) to `'\0'` to
> guarantee null termination even if `value` was longer than 255 bytes
> (in which case the value is silently truncated). Because a fixed 256-
> byte array is copied by value, the class owns the string contents and
> does not retain the caller's `value` pointer. Delegates the actual
> insert/update to `set_option_impl(key, arr)` per
> `[spec:et:sem:options.executorch.runtime.backend-options.set-option-impl-fn]`
> and returns its `Error`.
>
> The bool and int overloads of `set_option` (not separately annotated)
> apply the same `static_assert` and forward the value directly to
> `set_option_impl`.

> [spec:et:def:options.executorch.runtime.backend-options.set-option-impl-fn]
> Error set_option_impl(const char* key, T value)

> [spec:et:sem:options.executorch.runtime.backend-options.set-option-impl-fn]
> Internal insert-or-update for a key/value pair. `T` is the value type
> (`bool`, `int`, or `std::array<char, kMaxOptionValueLength>`); `key` is
> a null-terminated C-string.
>
> Step 1 (update): iterate `i` from `0` to `size_ - 1` ascending. Compare
> `options_[i].key` to `key` with `strcmp`. On the first match, assign
> `options_[i].value = value` (replacing whatever variant alternative was
> previously held) and return `Error::Ok`.
>
> Step 2 (append): if no existing key matched and `size_ < MaxCapacity`,
> create a new `BackendOption`. Compute `key_len = strlen(key)` and
> `copy_len = min(key_len, kMaxOptionKeyLength - 1)` (max 63). `memcpy`
> `copy_len` bytes of `key` into `new_option.key`, then set
> `new_option.key[copy_len] = '\0'` (keys longer than 63 chars are
> truncated with guaranteed null termination). Set `new_option.value =
> value`. Store the option at `options_[size_]` and increment `size_`
> (post-increment). Return `Error::Ok`.
>
> Step 3 (full): if no key matched and `size_ >= MaxCapacity`, make no
> change and return `Error::InvalidArgument`.

> [spec:et:def:options.executorch.runtime.backend-options.view-fn]
> executorch::runtime::Span<BackendOption> view()

> [spec:et:sem:options.executorch.runtime.backend-options.view-fn]
> Returns a mutable `Span<BackendOption>` over the currently populated
> options: `Span(options_, size_)`. The span aliases this object's
> internal `options_` storage (no copy) and covers exactly the first
> `size_` entries, so the caller can read and mutate stored options in
> place. The span is invalidated by any later mutation that changes
> `size_` or by destruction of this `BackendOptions`.

> [spec:et:def:options.executorch.runtime.backend-options.operator-fn]
> BackendOptions& operator=(const BackendOptions& other)

> [spec:et:sem:options.executorch.runtime.backend-options.operator-fn]
> Copy assignment operator. Self-assignment guard: if `this == &other`,
> do nothing and return `*this`. Otherwise set `size_ = other.size_` and
> copy the first `size_` entries from `other.options_` into `options_` in
> ascending index order (same element-wise value copy as the copy
> constructor per
> `[spec:et:sem:options.executorch.runtime.backend-options.backend-options-fn]`).
> Slots at indices `>= size_` are not overwritten. Returns `*this`.

