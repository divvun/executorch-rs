# extension/tensor/tensor_ptr_maker.cpp, extension/tensor/tensor_ptr_maker.h

> [spec:et:def:tensor-ptr-maker.executorch.extension.empty-fn]
> inline TensorPtr empty( std::vector<executorch::aten::SizesType> sizes, executorch::aten::ScalarType type = executorch::aten::ScalarType::Float, executorch::aten::TensorShapeDynamism dynamism = executorch::aten::TensorShapeDynamism::DYNA...

> [spec:et:sem:tensor-ptr-maker.executorch.extension.empty-fn]
> Convenience wrapper that creates an uninitialized (allocated but not
> value-initialized) owning tensor of the given `sizes` and `type`.
>
> Behavior: forwards to `empty_strided` (see
> `[spec:et:sem:tensor-ptr-maker.executorch.extension.empty-strided-fn]`)
> passing `std::move(sizes)`, an empty strides vector `{}` (which causes
> contiguous strides to be computed from `sizes`), the given `type`, and the
> given `dynamism`. `type` defaults to `Float`; `dynamism` defaults to
> `DYNAMIC_BOUND`.
>
> The returned tensor owns a freshly allocated data buffer of
> `numel * elementSize(type)` bytes whose contents are unspecified.

> [spec:et:def:tensor-ptr-maker.executorch.extension.empty-like-fn]
> inline TensorPtr empty_like( const TensorPtr& other, executorch::aten::ScalarType type = executorch::aten::ScalarType::Undefined, executorch::aten::TensorShapeDynamism dynamism = executorch::aten::TensorShapeDynamism::DYNAMIC_BOUND)

> [spec:et:sem:tensor-ptr-maker.executorch.extension.empty-like-fn]
> Creates an uninitialized owning tensor with the same sizes and strides as
> another tensor `other` (a `TensorPtr`, i.e. a shared_ptr to a Tensor).
>
> Steps:
> 1. If `type == ScalarType::Undefined` (the default), replace `type` with
>    `other->scalar_type()`. Otherwise use the caller-supplied `type`,
>    reinterpreting `other`'s buffer size in the new element type.
> 2. Copy `other`'s sizes into a `SizesType` vector (from `other->sizes()`
>    begin/end) and `other`'s strides into a `StridesType` vector (from
>    `other->strides()` begin/end).
> 3. Forward to `empty_strided` (see
>    `[spec:et:sem:tensor-ptr-maker.executorch.extension.empty-strided-fn]`)
>    with those sizes, strides, the resolved `type`, and `dynamism`
>    (default `DYNAMIC_BOUND`).
>
> The result reproduces `other`'s memory layout (including non-contiguous
> strides) but with a freshly allocated, uninitialized buffer. It does not
> share or copy `other`'s data.

> [spec:et:def:tensor-ptr-maker.executorch.extension.empty-strided-fn]
> TensorPtr empty_strided( std::vector<executorch::aten::SizesType> sizes, std::vector<executorch::aten::StridesType> strides, executorch::aten::ScalarType type, executorch::aten::TensorShapeDynamism dynamism)

> [spec:et:sem:tensor-ptr-maker.executorch.extension.empty-strided-fn]
> Allocates and returns an owning `TensorPtr` of the given `sizes`, `strides`,
> `type`, and `dynamism` with an uninitialized data buffer. This is the core
> allocation primitive; `empty`, `empty_like`, and (via `random_strided`) all
> the random/full factory functions funnel through it.
>
> Steps:
> 1. Compute the element count: call `safe_numel(sizes.data(), sizes.size())`,
>    which multiplies all dimension sizes with overflow checking and returns a
>    `Result<ssize_t>`. If it is not ok, `ET_CHECK_MSG` aborts the program with
>    a message including the numeric error code (this is a fatal check, not a
>    recoverable `Error` return). Let `numel` be the returned count (an empty
>    `sizes` vector yields the scalar count `1`; any zero dimension yields `0`).
> 2. Compute the byte size: `nbytes = numel * elementSize(type)` using
>    `c10::mul_overflows` on `size_t`. If the multiplication overflows,
>    `ET_CHECK_MSG` aborts with an overflow message. `elementSize(type)` is the
>    size in bytes of one element of the scalar type.
> 3. Allocate a `std::vector<uint8_t>` of `nbytes` bytes (value-initialized to
>    zero by std::vector, though callers must treat the contents as
>    unspecified/uninitialized). This vector owns the storage.
> 4. Build the tensor by calling `make_tensor_ptr` (see
>    `[spec:et:sem:tensor-ptr.executorch.extension.make-tensor-ptr-fn]`) with
>    `std::move(sizes)`, `std::move(data)` (the byte vector, transferring
>    ownership so the tensor owns its buffer), an empty `dim_order` vector `{}`
>    (make_tensor_ptr derives a default/contiguous dim order), `std::move(strides)`,
>    `type`, and `dynamism`. If `strides` is empty, make_tensor_ptr computes
>    contiguous strides from `sizes`.
>
> Returns the resulting `TensorPtr`. On any check failure the program aborts;
> there is no in-band error return.

> [spec:et:def:tensor-ptr-maker.executorch.extension.extract-scalar-fn]
> bool extract_scalar(executorch::aten::Scalar scalar, FLOAT_T* out_val)

> [spec:et:sem:tensor-ptr-maker.executorch.extension.extract-scalar-fn]
> This annotation marks the floating-point overload of the internal helper
> `extract_scalar(Scalar, T* out_val)`, selected by SFINAE when `T` is a
> floating-point type or `BFloat16` or `Half`. It attempts to convert an
> `executorch::aten::Scalar` into the concrete C type `FLOAT_T`, returning
> `true` on success (with `*out_val` written) and `false` on failure (with
> `*out_val` untouched). It never aborts; the surrounding `ET_EXTRACT_SCALAR`
> macro turns a `false` return into an `ET_CHECK_MSG` abort at the call site.
>
> Steps:
> 1. If the scalar `isFloatingPoint()`: read `val = scalar.to<double>()`. If
>    `val` is finite AND lies strictly outside `[lowest(FLOAT_T), max(FLOAT_T)]`
>    (i.e. `val < lowest || val > max`), return `false` (out of range).
>    Non-finite values (NaN, +/-inf) skip the range check and are accepted.
> 2. Else if the scalar `isIntegral(includeBool=false)` (an integer, not a
>    bool): set `val = static_cast<double>(scalar.to<int64_t>())` with no range
>    check.
> 3. Else (e.g. a boolean): return `false`.
> 4. Write `*out_val = static_cast<FLOAT_T>(val)` and return `true`.
>
> The sibling overloads (not separately annotated) behave analogously:
> - Integral non-bool `T`: accepts only `isIntegral(includeBool=false)`
>   scalars; converts to `int64_t`, rejects (returns `false`) if the value is
>   outside `[lowest(T), max(T)]`; otherwise casts and returns `true`.
> - `bool` `T`: accepts `isIntegral(includeBool=false)` (writes
>   `static_cast<bool>(int64_t value)`, i.e. nonzero -> true) or `isBoolean()`
>   (writes the bool directly); otherwise returns `false`.

> [spec:et:def:tensor-ptr-maker.executorch.extension.for-blob-fn]
> inline TensorPtrMaker for_blob( void* data, std::vector<executorch::aten::SizesType> sizes, executorch::aten::ScalarType type = executorch::aten::ScalarType::Float)

> [spec:et:sem:tensor-ptr-maker.executorch.extension.for-blob-fn]
> Free function (in namespace `executorch::extension`) that begins the fluent
> builder for a non-owning tensor over an externally owned raw buffer.
>
> Behavior: constructs and returns a `TensorPtrMaker` by value via the maker's
> private constructor (see
> `[spec:et:sem:tensor-ptr-maker.executorch.extension.tensor-ptr-maker.tensor-ptr-maker-fn]`),
> passing `data` (the raw pointer, not copied and not owned), `std::move(sizes)`,
> and `type`. `type` defaults to `Float`.
>
> The returned maker has empty strides/dim_order, no deleter, and
> `dynamism_ = DYNAMIC_BOUND`. The caller chains `.type()/.strides()/`
> `.dim_order()/.dynamism()/.deleter()` and finalizes with `make_tensor_ptr()`
> or the implicit `TensorPtr` conversion. The referenced buffer must outlive
> the resulting tensor.
>
> Note: there is a second, member overload `TensorPtrMaker::for_blob` declared
> as a friend; both share this rule id. See
> `[spec:et:sem:tensor-ptr-maker.executorch.extension.tensor-ptr-maker.for-blob-fn]`.

> [spec:et:def:tensor-ptr-maker.executorch.extension.from-blob-fn]
> inline TensorPtr from_blob( void* data, std::vector<executorch::aten::SizesType> sizes, std::vector<executorch::aten::StridesType> strides, executorch::aten::ScalarType type, std::function<void(void*)>&& deleter, executorch::aten::Tensor...

> [spec:et:sem:tensor-ptr-maker.executorch.extension.from-blob-fn]
> This rule id annotates the most fully-specified `from_blob` overload
> (data + sizes + strides + type + deleter + dynamism); it is representative of
> the whole `from_blob` overload family, which all build a NON-owning tensor
> over the caller's raw `data` buffer in one call by driving the `for_blob`
> builder and immediately finalizing it.
>
> This overload's behavior:
> 1. Call `for_blob(data, std::move(sizes), type)` (see
>    `[spec:et:sem:tensor-ptr-maker.executorch.extension.for-blob-fn]`).
> 2. `.strides(std::move(strides))` — set custom per-dimension strides.
> 3. `.deleter(std::move(deleter))` — install a custom cleanup callback invoked
>    on `data` when the tensor is destroyed (allowing the caller to hand off
>    ownership/cleanup even though the tensor itself does not copy the buffer).
> 4. `.dynamism(dynamism)` — default `DYNAMIC_BOUND`.
> 5. `.make_tensor_ptr()` — finalize (see
>    `[spec:et:sem:tensor-ptr-maker.executorch.extension.tensor-ptr-maker.make-tensor-ptr-fn]`).
>
> Returns the resulting `TensorPtr`. The buffer `data` (unless a deleter takes
> ownership) must outlive the tensor.
>
> The other `from_blob` overloads chain the same builder with a subset of these
> steps:
> - (data, sizes, type=Float, dynamism=DYNAMIC_BOUND): `.dynamism()` only.
> - (data, sizes, strides, type=Float, dynamism): `.strides().dynamism()`.
> - (data, sizes, type, deleter, dynamism): `.deleter().dynamism()`.
> In each case `type` defaults to `Float` and `dynamism` to `DYNAMIC_BOUND`,
> and strides default to contiguous when not supplied.

> [spec:et:def:tensor-ptr-maker.executorch.extension.full-fn]
> inline TensorPtr full( std::vector<executorch::aten::SizesType> sizes, executorch::aten::Scalar fill_value, executorch::aten::ScalarType type = executorch::aten::ScalarType::Float, executorch::aten::TensorShapeDynamism dynamism = executo...

> [spec:et:sem:tensor-ptr-maker.executorch.extension.full-fn]
> Creates an owning tensor of the given `sizes` and `type` with every element
> set to `fill_value`.
>
> Behavior: forwards to `full_strided` (see
> `[spec:et:sem:tensor-ptr-maker.executorch.extension.full-strided-fn]`) with
> `std::move(sizes)`, empty strides `{}` (contiguous layout computed from
> sizes), `fill_value`, `type` (default `Float`), and `dynamism`
> (default `DYNAMIC_BOUND`). The scalar `fill_value` is range-checked and cast
> to the element type per that rule.

> [spec:et:def:tensor-ptr-maker.executorch.extension.full-like-fn]
> inline TensorPtr full_like( const TensorPtr& other, executorch::aten::Scalar fill_value, executorch::aten::ScalarType type = executorch::aten::ScalarType::Undefined, executorch::aten::TensorShapeDynamism dynamism = executorch::aten::Tens...

> [spec:et:sem:tensor-ptr-maker.executorch.extension.full-like-fn]
> Creates an owning tensor with the same sizes and strides as `other`, every
> element set to `fill_value`.
>
> Steps:
> 1. If `type == ScalarType::Undefined` (default), set
>    `type = other->scalar_type()`.
> 2. Copy `other`'s sizes and strides into `SizesType`/`StridesType` vectors
>    (begin/end of `other->sizes()` and `other->strides()`).
> 3. Forward to `full_strided` (see
>    `[spec:et:sem:tensor-ptr-maker.executorch.extension.full-strided-fn]`) with
>    those sizes, strides, `fill_value`, resolved `type`, and `dynamism`
>    (default `DYNAMIC_BOUND`).
>
> Reproduces `other`'s layout with a fresh buffer filled with `fill_value`; does
> not share `other`'s data.

> [spec:et:def:tensor-ptr-maker.executorch.extension.full-strided-fn]
> TensorPtr full_strided( std::vector<executorch::aten::SizesType> sizes, std::vector<executorch::aten::StridesType> strides, executorch::aten::Scalar fill_value, executorch::aten::ScalarType type, executorch::aten::TensorShapeDynamism dyn...

> [spec:et:sem:tensor-ptr-maker.executorch.extension.full-strided-fn]
> Allocates an owning tensor of the given `sizes`, `strides`, `type`, and
> `dynamism` and fills every element with `fill_value`.
>
> Steps:
> 1. Allocate the tensor via `empty_strided(std::move(sizes),`
>    `std::move(strides), type, dynamism)` (see
>    `[spec:et:sem:tensor-ptr-maker.executorch.extension.empty-strided-fn]`).
> 2. Dispatch on `type` with `ET_SWITCH_REALHBBF16_AND_UINT_TYPES`, whose
>    accepted scalar-type set (mapped to concrete C type `CTYPE`) is exactly:
>    Byte (uint8), Char (int8), Short (int16), Int (int32), Long (int64),
>    Float (float), Double (double), Half, Bool, BFloat16, UInt16, UInt32,
>    UInt64. Any other `type` triggers the switch's `fail` handler, which
>    `ET_CHECK_MSG(false, ...)` aborts with "Unsupported data type in
>    full_strided".
> 3. Inside the dispatched body: declare `CTYPE value;`, then
>    `ET_EXTRACT_SCALAR(fill_value, value)` — this calls `extract_scalar`
>    (see `[spec:et:sem:tensor-ptr-maker.executorch.extension.extract-scalar-fn]`)
>    and `ET_CHECK_MSG`-aborts if the scalar cannot be represented in `CTYPE`
>    (wrong category or out of range).
> 4. `std::fill` the entire buffer `[data, data + numel)` (contiguous over the
>    allocation, `numel` = `tensor->numel()`) with `value`. Because it writes
>    the flat allocation rather than iterating logical indices, the whole
>    backing buffer is set even for non-contiguous strides.
>
> Returns the filled `TensorPtr`. Empty tensors (numel 0) allocate but write
> nothing. Failures abort the program.

> [spec:et:def:tensor-ptr-maker.executorch.extension.ones-fn]
> inline TensorPtr ones( std::vector<executorch::aten::SizesType> sizes, executorch::aten::ScalarType type = executorch::aten::ScalarType::Float, executorch::aten::TensorShapeDynamism dynamism = executorch::aten::TensorShapeDynamism::DYNAM...

> [spec:et:sem:tensor-ptr-maker.executorch.extension.ones-fn]
> Creates an owning tensor of the given `sizes` and `type` with every element
> equal to 1.
>
> Behavior: forwards to `full(std::move(sizes), 1, type, dynamism)` (see
> `[spec:et:sem:tensor-ptr-maker.executorch.extension.full-fn]`), passing the
> integer scalar `1` as `fill_value`. `type` defaults to `Float`, `dynamism` to
> `DYNAMIC_BOUND`. The value 1 is extracted/cast into the element type per the
> full/full_strided rules.

> [spec:et:def:tensor-ptr-maker.executorch.extension.ones-like-fn]
> inline TensorPtr ones_like( const TensorPtr& other, executorch::aten::ScalarType type = executorch::aten::ScalarType::Undefined, executorch::aten::TensorShapeDynamism dynamism = executorch::aten::TensorShapeDynamism::DYNAMIC_BOUND)

> [spec:et:sem:tensor-ptr-maker.executorch.extension.ones-like-fn]
> Creates an owning tensor with the same sizes and strides as `other`, every
> element equal to 1.
>
> Behavior: forwards to `full_like(other, 1, type, dynamism)` (see
> `[spec:et:sem:tensor-ptr-maker.executorch.extension.full-like-fn]`), passing
> the integer scalar `1`. If `type == Undefined` (default) the element type is
> taken from `other->scalar_type()`; `dynamism` defaults to `DYNAMIC_BOUND`.

> [spec:et:def:tensor-ptr-maker.executorch.extension.rand-fn]
> inline TensorPtr rand( std::vector<executorch::aten::SizesType> sizes, executorch::aten::ScalarType type = executorch::aten::ScalarType::Float, executorch::aten::TensorShapeDynamism dynamism = executorch::aten::TensorShapeDynamism::DYNAM...

> [spec:et:sem:tensor-ptr-maker.executorch.extension.rand-fn]
> Creates an owning tensor of the given `sizes` and `type` filled with
> independent uniform random values in the half-open range [0, 1).
>
> Behavior: forwards to `rand_strided(std::move(sizes), {}, type, dynamism)`
> (see `[spec:et:sem:tensor-ptr-maker.executorch.extension.rand-strided-fn]`)
> with empty strides `{}` (contiguous layout). `type` defaults to `Float`,
> `dynamism` to `DYNAMIC_BOUND`.

> [spec:et:def:tensor-ptr-maker.executorch.extension.rand-like-fn]
> inline TensorPtr rand_like( const TensorPtr& other, executorch::aten::ScalarType type = executorch::aten::ScalarType::Undefined, executorch::aten::TensorShapeDynamism dynamism = executorch::aten::TensorShapeDynamism::DYNAMIC_BOUND)

> [spec:et:sem:tensor-ptr-maker.executorch.extension.rand-like-fn]
> Creates an owning tensor with the same sizes and strides as `other` filled
> with independent uniform random values in [0, 1).
>
> Steps:
> 1. If `type == Undefined` (default), set `type = other->scalar_type()`.
> 2. Copy `other`'s sizes and strides into `SizesType`/`StridesType` vectors.
> 3. Forward to `rand_strided` (see
>    `[spec:et:sem:tensor-ptr-maker.executorch.extension.rand-strided-fn]`) with
>    those sizes/strides, resolved `type`, and `dynamism` (default
>    `DYNAMIC_BOUND`).

> [spec:et:def:tensor-ptr-maker.executorch.extension.rand-strided-fn]
> TensorPtr rand_strided( std::vector<executorch::aten::SizesType> sizes, std::vector<executorch::aten::StridesType> strides, executorch::aten::ScalarType type, executorch::aten::TensorShapeDynamism dynamism)

> [spec:et:sem:tensor-ptr-maker.executorch.extension.rand-strided-fn]
> Allocates an owning tensor of the given `sizes`, `strides`, `type`, and
> `dynamism` and fills it with independent uniform random values in the
> half-open range [0, upper_bound), using a `std::uniform_real_distribution<float>`.
>
> Steps:
> 1. Set `upper_bound = 1.0f`.
> 2. Precision adjustment to prevent a sample from rounding up to exactly 1.0
>    when converted to a low-precision element type:
>    - if `type == Half`: subtract `float(numeric_limits<Half>::epsilon()) / 2`
>      from `upper_bound`.
>    - else if `type == BFloat16`: subtract
>      `float(numeric_limits<BFloat16>::epsilon()) / 2` from `upper_bound`.
>    - otherwise leave `upper_bound = 1.0f`.
> 3. Forward to `random_strided` (see
>    `[spec:et:sem:tensor-ptr-maker.executorch.extension.random-strided-fn]`)
>    with the sizes/strides/type/dynamism and a
>    `std::uniform_real_distribution<float>(0.0f, upper_bound)`.
>
> The distribution is drawn in `float` and each sample cast to the element type;
> integer element types therefore receive truncated (mostly zero) values.

> [spec:et:def:tensor-ptr-maker.executorch.extension.randint-fn]
> inline TensorPtr randint( int64_t low, int64_t high, std::vector<executorch::aten::SizesType> sizes, executorch::aten::ScalarType type = executorch::aten::ScalarType::Int, executorch::aten::TensorShapeDynamism dynamism = executorch::aten...

> [spec:et:sem:tensor-ptr-maker.executorch.extension.randint-fn]
> Creates an owning tensor of the given `sizes` and `type` filled with random
> integers in the half-open range [low, high).
>
> Behavior: forwards to `randint_strided(low, high, std::move(sizes), {}, type,`
> `dynamism)` (see
> `[spec:et:sem:tensor-ptr-maker.executorch.extension.randint-strided-fn]`) with
> empty strides `{}`. `type` defaults to `Int`, `dynamism` to `DYNAMIC_BOUND`.

> [spec:et:def:tensor-ptr-maker.executorch.extension.randint-like-fn]
> inline TensorPtr randint_like( const TensorPtr& other, int64_t low, int64_t high, executorch::aten::ScalarType type = executorch::aten::ScalarType::Undefined, executorch::aten::TensorShapeDynamism dynamism = executorch::aten::TensorShape...

> [spec:et:sem:tensor-ptr-maker.executorch.extension.randint-like-fn]
> Creates an owning tensor with the same sizes and strides as `other` filled
> with random integers in [low, high).
>
> Steps:
> 1. If `type == Undefined` (default), set `type = other->scalar_type()`.
> 2. Copy `other`'s sizes and strides into `SizesType`/`StridesType` vectors.
> 3. Forward to `randint_strided(low, high, sizes, strides, type, dynamism)`
>    (see `[spec:et:sem:tensor-ptr-maker.executorch.extension.randint-strided-fn]`);
>    `dynamism` defaults to `DYNAMIC_BOUND`.

> [spec:et:def:tensor-ptr-maker.executorch.extension.randint-strided-fn]
> TensorPtr randint_strided( int64_t low, int64_t high, std::vector<executorch::aten::SizesType> sizes, std::vector<executorch::aten::StridesType> strides, executorch::aten::ScalarType type, executorch::aten::TensorShapeDynamism dynamism)

> [spec:et:sem:tensor-ptr-maker.executorch.extension.randint-strided-fn]
> Allocates an owning tensor of the given `sizes`, `strides`, `type`, and
> `dynamism` and fills it with random integers drawn uniformly from the
> half-open range [low, high).
>
> Behavior: forwards to `random_strided` (see
> `[spec:et:sem:tensor-ptr-maker.executorch.extension.random-strided-fn]`) with
> the sizes/strides/type/dynamism and a
> `std::uniform_int_distribution<int64_t>(low, high - 1)`. Note the distribution
> is inclusive on both ends, and `high - 1` makes the upper bound exclusive as
> documented; if `high <= low` behavior follows `std::uniform_int_distribution`
> with a degenerate/invalid range (caller must ensure `low < high`). Each
> `int64_t` sample is cast to the element `CTYPE` inside `random_strided`;
> `type` defaults to `Int`, `dynamism` to `DYNAMIC_BOUND`.

> [spec:et:def:tensor-ptr-maker.executorch.extension.randn-fn]
> inline TensorPtr randn( std::vector<executorch::aten::SizesType> sizes, executorch::aten::ScalarType type = executorch::aten::ScalarType::Float, executorch::aten::TensorShapeDynamism dynamism = executorch::aten::TensorShapeDynamism::DYNA...

> [spec:et:sem:tensor-ptr-maker.executorch.extension.randn-fn]
> Creates an owning tensor of the given `sizes` and `type` filled with
> independent samples from the standard normal distribution N(0, 1).
>
> Behavior: forwards to `randn_strided(std::move(sizes), {}, type, dynamism)`
> (see `[spec:et:sem:tensor-ptr-maker.executorch.extension.randn-strided-fn]`)
> with empty strides `{}`. `type` defaults to `Float`, `dynamism` to
> `DYNAMIC_BOUND`.

> [spec:et:def:tensor-ptr-maker.executorch.extension.randn-like-fn]
> inline TensorPtr randn_like( const TensorPtr& other, executorch::aten::ScalarType type = executorch::aten::ScalarType::Undefined, executorch::aten::TensorShapeDynamism dynamism = executorch::aten::TensorShapeDynamism::DYNAMIC_BOUND)

> [spec:et:sem:tensor-ptr-maker.executorch.extension.randn-like-fn]
> Creates an owning tensor with the same sizes and strides as `other` filled
> with independent standard-normal N(0, 1) samples.
>
> Steps:
> 1. If `type == Undefined` (default), set `type = other->scalar_type()`.
> 2. Copy `other`'s sizes and strides into `SizesType`/`StridesType` vectors.
> 3. Forward to `randn_strided` (see
>    `[spec:et:sem:tensor-ptr-maker.executorch.extension.randn-strided-fn]`) with
>    those sizes/strides, resolved `type`, and `dynamism` (default
>    `DYNAMIC_BOUND`).

> [spec:et:def:tensor-ptr-maker.executorch.extension.randn-strided-fn]
> TensorPtr randn_strided( std::vector<executorch::aten::SizesType> sizes, std::vector<executorch::aten::StridesType> strides, executorch::aten::ScalarType type, executorch::aten::TensorShapeDynamism dynamism)

> [spec:et:sem:tensor-ptr-maker.executorch.extension.randn-strided-fn]
> Allocates an owning tensor of the given `sizes`, `strides`, `type`, and
> `dynamism` and fills it with independent samples from the standard normal
> distribution.
>
> Behavior: forwards to `random_strided` (see
> `[spec:et:sem:tensor-ptr-maker.executorch.extension.random-strided-fn]`) with
> the sizes/strides/type/dynamism and a
> `std::normal_distribution<float>(0.0f, 1.0f)` (mean 0, stddev 1). Each `float`
> sample is cast to the element `CTYPE`. `type` defaults to `Float`, `dynamism`
> to `DYNAMIC_BOUND`.

> [spec:et:def:tensor-ptr-maker.executorch.extension.random-strided-fn]
> TensorPtr random_strided( std::vector<executorch::aten::SizesType> sizes, std::vector<executorch::aten::StridesType> strides, executorch::aten::ScalarType type, executorch::aten::TensorShapeDynamism dynamism, Distribution&& distribution)

> [spec:et:sem:tensor-ptr-maker.executorch.extension.random-strided-fn]
> Internal helper (template over a `Distribution` type) that allocates a tensor
> and fills it by drawing one sample per element from a caller-supplied
> distribution. It is the common backbone of `rand_strided`, `randn_strided`,
> and `randint_strided`.
>
> Steps:
> 1. Allocate via `empty_strided(std::move(sizes), std::move(strides), type,`
>    `dynamism)` (see
>    `[spec:et:sem:tensor-ptr-maker.executorch.extension.empty-strided-fn]`).
> 2. Construct a PRNG engine: `std::default_random_engine gen{`
>    `std::random_device{}()}` — seeded from a fresh `std::random_device` on
>    every call (non-deterministic; no user-controllable seed).
> 3. Dispatch on `type` with `ET_SWITCH_REALHBBF16_AND_UINT_TYPES`, whose
>    accepted scalar types (bound to concrete `CTYPE`) are exactly: Byte (uint8),
>    Char (int8), Short (int16), Int (int32), Long (int64), Float, Double, Half,
>    Bool, BFloat16, UInt16, UInt32, UInt64. Any other `type` invokes the local
>    context's `fail`, which `ET_CHECK_MSG(false, ...)` aborts with "Unsupported
>    dtype in random_strided".
> 4. Inside the dispatched body, run `std::generate_n(mutable_data_ptr<CTYPE>(),`
>    `numel, gen_fn)` where `gen_fn` returns `static_cast<CTYPE>(distribution(gen))`.
>    This writes `numel = tensor->numel()` samples sequentially across the flat
>    allocation starting at the data pointer (contiguous fill of the backing
>    buffer, independent of logical strides). Each element gets one fresh draw;
>    samples are cast to `CTYPE` (float/int truncation and clamping-by-cast
>    apply as usual).
>
> Returns the filled `TensorPtr`. numel-0 tensors write nothing. Failures abort.

> [spec:et:def:tensor-ptr-maker.executorch.extension.scalar-tensor-fn]
> inline TensorPtr scalar_tensor( executorch::aten::Scalar value, executorch::aten::ScalarType type = executorch::aten::ScalarType::Float)

> [spec:et:sem:tensor-ptr-maker.executorch.extension.scalar-tensor-fn]
> Creates an owning 0-dimensional (scalar) tensor of the given `type` holding
> `value`.
>
> Behavior: forwards to `full({}, value, type)` (see
> `[spec:et:sem:tensor-ptr-maker.executorch.extension.full-fn]`) with an empty
> `sizes` vector, which produces a rank-0 tensor with `numel == 1`. `type`
> defaults to `Float`; `dynamism` is left at `full`'s default (`DYNAMIC_BOUND`).
> The single element is set to `value`, extracted/cast into the element type per
> the full/full_strided rules.

> [spec:et:def:tensor-ptr-maker.executorch.extension.tensor-ptr-maker]
> class TensorPtrMaker final {
>   std::vector<executorch::aten::SizesType> sizes_;
>   std::vector<executorch::aten::StridesType> strides_;
>   std::vector<executorch::aten::DimOrderType> dim_order_;
>   std::function<void(void*)> deleter_ = nullptr;
>   void* data_ = nullptr;
>   executorch::aten::ScalarType type_ = executorch::aten::ScalarType::Float;
>   executorch::aten::TensorShapeDynamism dynamism_ = executorch::aten::TensorShapeDynamism::DYNAMIC_BOUND;
> }

> [spec:et:def:tensor-ptr-maker.executorch.extension.tensor-ptr-maker.for-blob-fn]
> TensorPtrMaker for_blob( void* data, std::vector<executorch::aten::SizesType> sizes, executorch::aten::ScalarType type)

> [spec:et:sem:tensor-ptr-maker.executorch.extension.tensor-ptr-maker.for-blob-fn]
> This rule id annotates the `friend TensorPtrMaker for_blob(void* data,`
> `vector<SizesType> sizes, ScalarType type)` declaration inside the
> `TensorPtrMaker` class. It grants the free `for_blob` function access to the
> maker's private constructor. The friend declaration has no behavior of its
> own; the actual construction logic is the free `for_blob` function (see
> `[spec:et:sem:tensor-ptr-maker.executorch.extension.for-blob-fn]`), which
> returns a `TensorPtrMaker(data, std::move(sizes), type)` built via the private
> constructor (see
> `[spec:et:sem:tensor-ptr-maker.executorch.extension.tensor-ptr-maker.tensor-ptr-maker-fn]`).
> In Rust this corresponds simply to `for_blob` being able to invoke the maker's
> private constructor; no separate member exists.

> [spec:et:def:tensor-ptr-maker.executorch.extension.tensor-ptr-maker.make-tensor-ptr-fn]
> TensorPtr make_tensor_ptr() &&

> [spec:et:sem:tensor-ptr-maker.executorch.extension.tensor-ptr-maker.make-tensor-ptr-fn]
> Rvalue-qualified (`&&`) terminal method of the fluent builder: consumes the
> `TensorPtrMaker` and produces the final `TensorPtr`.
>
> Behavior: calls the free `::executorch::extension::make_tensor_ptr` (see
> `[spec:et:sem:tensor-ptr.executorch.extension.make-tensor-ptr-fn]`), moving out
> all accumulated builder fields in this argument order: `std::move(sizes_)`,
> `data_` (the raw non-owning pointer), `std::move(dim_order_)`,
> `std::move(strides_)`, `type_`, `dynamism_`, `std::move(deleter_)`. Empty
> `dim_order_`/`strides_` cause make_tensor_ptr to derive contiguous defaults;
> a null `deleter_` means the tensor does not free `data_` on destruction.
>
> Returns the constructed `TensorPtr`. Because it is `&&`-qualified it must be
> invoked on an rvalue (temporary or `std::move`d maker) and leaves the maker in
> a moved-from state. The implicit `operator TensorPtr() &&` is defined to call
> `std::move(*this).make_tensor_ptr()`, giving the same result.

> [spec:et:def:tensor-ptr-maker.executorch.extension.tensor-ptr-maker.tensor-ptr-maker-fn]
> TensorPtrMaker( void* data, std::vector<executorch::aten::SizesType> sizes, executorch::aten::ScalarType type) : sizes_(std::move(sizes)), data_(data), type_(type)

> [spec:et:sem:tensor-ptr-maker.executorch.extension.tensor-ptr-maker.tensor-ptr-maker-fn]
> Private constructor of `TensorPtrMaker`, callable only by the friend
> `for_blob`. Initializes the builder's essential fields:
> `sizes_ = std::move(sizes)`, `data_ = data`, `type_ = type`. All remaining
> fields keep their in-class defaults: `strides_` empty, `dim_order_` empty,
> `deleter_` null, `dynamism_ = DYNAMIC_BOUND` (and `type_`'s class default of
> `Float` is overridden by the passed `type`).
>
> It stores the raw `data` pointer without copying or taking ownership. It
> performs no validation of `sizes`, `data`, or `type`.

> [spec:et:def:tensor-ptr-maker.executorch.extension.zeros-fn]
> inline TensorPtr zeros( std::vector<executorch::aten::SizesType> sizes, executorch::aten::ScalarType type = executorch::aten::ScalarType::Float, executorch::aten::TensorShapeDynamism dynamism = executorch::aten::TensorShapeDynamism::DYNA...

> [spec:et:sem:tensor-ptr-maker.executorch.extension.zeros-fn]
> Creates an owning tensor of the given `sizes` and `type` with every element
> equal to 0.
>
> Behavior: forwards to `full(std::move(sizes), 0, type, dynamism)` (see
> `[spec:et:sem:tensor-ptr-maker.executorch.extension.full-fn]`), passing the
> integer scalar `0`. `type` defaults to `Float`, `dynamism` to `DYNAMIC_BOUND`.

> [spec:et:def:tensor-ptr-maker.executorch.extension.zeros-like-fn]
> inline TensorPtr zeros_like( const TensorPtr& other, executorch::aten::ScalarType type = executorch::aten::ScalarType::Undefined, executorch::aten::TensorShapeDynamism dynamism = executorch::aten::TensorShapeDynamism::DYNAMIC_BOUND)

> [spec:et:sem:tensor-ptr-maker.executorch.extension.zeros-like-fn]
> Creates an owning tensor with the same sizes and strides as `other`, every
> element equal to 0.
>
> Behavior: forwards to `full_like(other, 0, type, dynamism)` (see
> `[spec:et:sem:tensor-ptr-maker.executorch.extension.full-like-fn]`), passing
> the integer scalar `0`. If `type == Undefined` (default) the element type is
> taken from `other->scalar_type()`; `dynamism` defaults to `DYNAMIC_BOUND`.

> [spec:et:def:tensor-ptr-maker.executorch.extension.tensor-ptr-maker.deleter-fn]
> TensorPtrMaker&& deleter(std::function<void(void*)>&& deleter)

> [spec:et:sem:tensor-ptr-maker.executorch.extension.tensor-ptr-maker.deleter-fn]
> Fluent setter (rvalue-qualified `&&`): stores a custom cleanup callback for
> the data buffer. Sets `deleter_ = std::move(deleter)` (an rvalue
> `std::function<void(void*)>`, consumed by move to avoid copying captured
> state) and returns `std::move(*this)` so the call chain continues on an
> rvalue. The stored deleter is later moved into `make_tensor_ptr`; if set, it
> is invoked with the data pointer when the tensor is destroyed. No validation.

> [spec:et:def:tensor-ptr-maker.executorch.extension.tensor-ptr-maker.dim-order-fn]
> TensorPtrMaker&& dim_order( std::vector<executorch::aten::DimOrderType> dim_order)

> [spec:et:sem:tensor-ptr-maker.executorch.extension.tensor-ptr-maker.dim-order-fn]
> Fluent setter (rvalue-qualified `&&`): sets `dim_order_ = std::move(dim_order)`
> (a `std::vector<DimOrderType>` specifying the physical order of dimensions in
> memory) and returns `std::move(*this)` for chaining. No validation is done
> here; consistency between `dim_order_`, `strides_`, and `sizes_` is enforced
> later by `make_tensor_ptr`.

> [spec:et:def:tensor-ptr-maker.executorch.extension.tensor-ptr-maker.dynamism-fn]
> TensorPtrMaker&& dynamism(executorch::aten::TensorShapeDynamism dynamism)

> [spec:et:sem:tensor-ptr-maker.executorch.extension.tensor-ptr-maker.dynamism-fn]
> Fluent setter (rvalue-qualified `&&`): sets `dynamism_ = dynamism` (the
> `TensorShapeDynamism` value controlling whether the tensor's shape is treated
> as static, dynamic-bound, or fully dynamic) and returns `std::move(*this)` for
> chaining. No validation.

> [spec:et:def:tensor-ptr-maker.executorch.extension.tensor-ptr-maker.operator-fn]
> TensorPtrMaker& operator=(const TensorPtrMaker&) = delete

> [spec:et:sem:tensor-ptr-maker.executorch.extension.tensor-ptr-maker.operator-fn]
> The copy-assignment operator `operator=(const TensorPtrMaker&)` is explicitly
> `= delete`d: `TensorPtrMaker` is non-copy-assignable (and its copy constructor
> is likewise deleted). Only move construction and move assignment are provided
> (both `= default`). In Rust this maps to a non-`Clone`, move-only builder type;
> there is no runtime behavior to implement — copying a `TensorPtrMaker` is a
> compile error.

> [spec:et:def:tensor-ptr-maker.executorch.extension.tensor-ptr-maker.strides-fn]
> TensorPtrMaker&& strides(std::vector<executorch::aten::StridesType> strides)

> [spec:et:sem:tensor-ptr-maker.executorch.extension.tensor-ptr-maker.strides-fn]
> Fluent setter (rvalue-qualified `&&`): sets `strides_ = std::move(strides)`
> (a `std::vector<StridesType>`, one stride per dimension) and returns
> `std::move(*this)` for chaining. No validation here; if left empty,
> `make_tensor_ptr` computes contiguous strides from `sizes_`.

> [spec:et:def:tensor-ptr-maker.executorch.extension.tensor-ptr-maker.type-fn]
> TensorPtrMaker&& type(executorch::aten::ScalarType type)

> [spec:et:sem:tensor-ptr-maker.executorch.extension.tensor-ptr-maker.type-fn]
> Fluent setter (rvalue-qualified `&&`): sets `type_ = type` (the
> `ScalarType` of the tensor elements) and returns `std::move(*this)` for
> chaining. Overrides the class default of `Float`. No validation.

