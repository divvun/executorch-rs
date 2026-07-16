# kernels/portable/cpu/util/broadcast_indexes_range.h

> [spec:et:def:broadcast-indexes-range.broadcast-indexes-range]
> class BroadcastIndexesRange {
>   std::array<const Tensor*, kNumInputs + 1> tensors_;
> }

> [spec:et:def:broadcast-indexes-range.broadcast-indexes-range.begin-fn]
> iterator begin() const

> [spec:et:sem:broadcast-indexes-range.broadcast-indexes-range.begin-fn]
> Returns the start iterator of the range. It constructs a
> `BroadcastIndexesIterator<kNumInputs, support_noncontiguous_input_tensors>`
> from the stored tensor pointers `tensors_` (element 0 is the output, elements
> 1..kNumInputs are the inputs): `std::apply` unpacks the array and calls
> `iterator(output, input0, input1, ...)` (dereferencing each stored pointer),
> i.e. the non-end constructor per
> `[spec:et:sem:broadcast-indexes-range.torch.executor.broadcast-indexes-iterator-1-false.broadcast-indexes-iterator-fn]`
> (or the general iterator's non-end constructor). The resulting iterator sits
> at output linear index 0 with all input indices at 0.

> [spec:et:def:broadcast-indexes-range.broadcast-indexes-range.broadcast-indexes-range-fn]
> BroadcastIndexesRange(const Tensor& output, const Args&... args)

> [spec:et:sem:broadcast-indexes-range.broadcast-indexes-range.broadcast-indexes-range-fn]
> Constructor of the range object. Takes the `output` tensor and exactly
> `kNumInputs` input tensors (`args...`) by const reference and stores their
> addresses into the member array `tensors_` of size `kNumInputs + 1`, with
> `tensors_[0] = &output` and `tensors_[1..] = &args_k`. It performs no
> iteration or validation itself; it only records the tensors so that `begin()`
> and `end()` can build iterators. The tensors must outlive the range (only
> pointers are held).

> [spec:et:def:broadcast-indexes-range.broadcast-indexes-range.end-fn]
> iterator end() const

> [spec:et:sem:broadcast-indexes-range.broadcast-indexes-range.end-fn]
> Returns the past-the-end iterator. It `std::apply`s over `tensors_` and calls
> the iterator's end constructor `iterator(make_end_t{}, output, input0, ...)`
> (dereferencing each stored pointer). That constructor sets the iterator's
> output linear index to `output.numel()` (the total element count), so that
> comparing against `begin()` iterates exactly `output.numel()` steps. When
> `output.numel() == 0` (any output dim is 0) begin equals end and the loop body
> never runs.

> [spec:et:def:broadcast-indexes-range.torch.executor.broadcast-indexes-iterator-1-false]
> class BroadcastIndexesIterator<1, false> {
>   struct make_end_t { explicit constexpr make_end_t() = default; };
>   value_type current_indexes_ = {{0, 0}};
> }

> [spec:et:def:broadcast-indexes-range.torch.executor.broadcast-indexes-iterator-1-false.add-to-current-index-fn]
> void add_to_current_index(ssize_t n)

> [spec:et:sem:broadcast-indexes-range.torch.executor.broadcast-indexes-iterator-1-false.add-to-current-index-fn]
> Advances the single-input, contiguous specialization `BroadcastIndexesIterator<1, false>`
> by `n` steps. `current_indexes_` is a 2-element array `{output_index, input_index}`.
> The operation adds `n` to `current_indexes_[0]` (the output linear index) and
> then sets `current_indexes_[1] = current_indexes_[0]`. Because with a single
> non-broadcast input the input's flat index equals the output's flat index,
> both entries stay identical. `n` may be any ssize_t (positive advance).

> [spec:et:def:broadcast-indexes-range.torch.executor.broadcast-indexes-iterator-1-false.broadcast-indexes-iterator-fn]
> BroadcastIndexesIterator( make_end_t, const Tensor& output, [[maybe_unused]] const Tensor& input) : current_indexes_({output.numel(), output.numel()})

> [spec:et:sem:broadcast-indexes-range.torch.executor.broadcast-indexes-iterator-1-false.broadcast-indexes-iterator-fn]
> The end constructor of the single-input specialization
> `BroadcastIndexesIterator<1, false>`. Signature
> `(make_end_t, const Tensor& output, const Tensor& input)`; the `input` is
> unused. It initializes `current_indexes_ = {output.numel(), output.numel()}`,
> placing both the output and input linear indices at `output.numel()` so this
> iterator compares equal to any iterator that has advanced through all output
> elements. (The specialization also has a default constructor giving
> `{0, 0}` and a non-end constructor `(output, input)` that ignores both
> arguments and leaves `current_indexes_` at `{0, 0}` — the start position.)

> [spec:et:def:broadcast-indexes-range.torch.executor.broadcast-indexes-iterator-1-false.current-index-fn]
> ssize_t current_index() const

> [spec:et:sem:broadcast-indexes-range.torch.executor.broadcast-indexes-iterator-1-false.current-index-fn]
> Accessor (private) for the single-input specialization: returns
> `current_indexes_[0]`, i.e. the current output linear index. Used by
> `operator==`/`operator!=`/`operator-` to compare iterator positions.

> [spec:et:def:broadcast-indexes-range.torch.executor.broadcast-indexes-iterator-1-false.make-end-t]
> struct make_end_t

> [spec:et:def:broadcast-indexes-range.torch.executor.broadcast-indexes-iterator-1-false.make-end-t.make-end-t-fn]
> explicit constexpr make_end_t() = default

> [spec:et:sem:broadcast-indexes-range.torch.executor.broadcast-indexes-iterator-1-false.make-end-t.make-end-t-fn]
> Defaulted `constexpr` constructor of the empty tag type `make_end_t`. It is a
> zero-field marker struct whose only purpose is to disambiguate the end
> constructor of the iterator from the normal (begin) constructor via
> tag-dispatch. Constructing it has no runtime effect and carries no state. In a
> Rust port this corresponds to a unit marker (or simply calling a distinct
> `end()` constructor).

> [spec:et:def:broadcast-indexes-range.torch.executor.broadcast-indexes-iterator-1-false.operator-fn]
> BroadcastIndexesIterator operator++(int)

> [spec:et:sem:broadcast-indexes-range.torch.executor.broadcast-indexes-iterator-1-false.operator-fn]
> Post-increment `operator++(int)` for the single-input specialization: copies
> the current iterator into a temporary `it`, applies pre-increment
> `operator++()` (which calls `add_to_current_index(1)`, advancing both the
> output and input linear indices by 1 per
> `[spec:et:sem:broadcast-indexes-range.torch.executor.broadcast-indexes-iterator-1-false.add-to-current-index-fn]`),
> and returns the pre-increment copy `it` by value. Standard postfix-increment
> semantics.

> [spec:et:def:broadcast-indexes-range.torch.executor.internal.arrayref-begin-ignoring-leading-1s-fn]
> inline const Tensor::SizesType* arrayref_begin_ignoring_leading_1s( ArrayRef<Tensor::SizesType> arr)

> [spec:et:sem:broadcast-indexes-range.torch.executor.internal.arrayref-begin-ignoring-leading-1s-fn]
> Returns a pointer to the first element of the sizes array `arr` that is not
> equal to 1, i.e. it skips any leading run of size-1 dimensions. Implemented as
> `std::find_if(arr.begin(), arr.end(), x != 1)`. If every element is 1 (or the
> array is empty), it returns `arr.end()`. Used to compare shapes while ignoring
> broadcastable leading unit dimensions.

> [spec:et:def:broadcast-indexes-range.torch.executor.internal.broadcast-indexes-iterator-operator-fn]
> class BroadcastIndexesIterator

> [spec:et:sem:broadcast-indexes-range.torch.executor.internal.broadcast-indexes-iterator-operator-fn]
> The general `BroadcastIndexesIterator<kNumInputs, support_noncontiguous_input_tensors>`
> forward iterator. Its `value_type` is `std::array<ssize_t, kNumInputs + 1>`:
> element 0 is the output linear index, elements 1..kNumInputs are the
> corresponding (possibly broadcast) input linear indices. State:
> `current_indexes_` (the value_type), `delinearized_output_index_` (per-dim
> output coordinate), `output_dim_or_zero_if_no_broadcasting_`, `output_shape_`
> (ArrayRef of output sizes), and `effective_input_broadcast_strides_` (one
> stride array per input, see
> `[spec:et:sem:broadcast-indexes-range.torch.executor.internal.effective-input-broadcast-stride-fn]`).
>
> Non-end constructor `(output, args...)` (requires exactly `kNumInputs` Tensor
> args):
> - Sets `output_dim_or_zero_if_no_broadcasting_`: if
>   `!support_noncontiguous_input_tensors` AND every input's sizes match the
>   output's sizes ignoring leading 1s (per
>   `[spec:et:sem:broadcast-indexes-range.torch.executor.internal.sizes-match-ignoring-leading-1s-fn]`),
>   it is 0 (fast path: no broadcasting needed, all indices track the output
>   index); otherwise it is `output.dim()`.
> - Sets `output_shape_ = output.sizes()`.
> - If noncontiguous support is on OR the field is nonzero, computes
>   `effective_input_broadcast_strides_` = one effective-stride array per input.
> - All of `current_indexes_` and `delinearized_output_index_` start at 0.
>
> End constructor `(make_end_t, t, args...)`: sets `current_indexes_ = {t.numel(), 0}`
> (output index = number of output elements), marking past-the-end.
>
> Comparison: `operator==` compares `output_index()` (element 0); `operator!=`
> is its negation. `operator-` returns the difference of output indices.
> Dereference `operator*` returns a const ref to `current_indexes_`.
>
> Pre-increment `operator++()`:
> 1. Increment `output_index()` (element 0).
> 2. If `output_dim_or_zero_if_no_broadcasting_ == 0` (fast path): fill every
>    input index (elements 1..kNumInputs) with the new output index and return.
> 3. Otherwise perform an odometer step over the output dims from last
>    (`ii = output_dim - 1`) down to 0: if
>    `delinearized_output_index_[ii] == output_shape_[ii] - 1`, this dim rolls
>    over — reset it to 0 and, for each input `jj`, subtract
>    `old_value * effective_input_broadcast_strides_[jj-1][ii]` from that input's
>    index (undoing the accumulated contribution of this dim), then continue to
>    the next-higher dim. Otherwise increment `delinearized_output_index_[ii]` by
>    1 and, for each input `jj`, add
>    `effective_input_broadcast_strides_[jj-1][ii]` to that input's index, then
>    break. This maintains each input's flat index incrementally without
>    division/modulo. (If some output dim is 0, numel is 0 so begin==end and no
>    iteration occurs.)
>
> Post-increment `operator++(int)`: copy, pre-increment, return the copy.
>
> `operator+=(n)`:
> - If `n <= 3`: fall back to `std::advance` (repeated pre-increment).
> - Else add `n` to the output index; if fast path (field == 0), refill all input
>   indices with the new output index and return. Otherwise fully recompute:
>   delinearize the new output index into `delinearized_output_index_` (this uses
>   `delinearize_index`), then for each input `ii` set
>   `current_indexes_[ii] = sum over output dims jj of
>   delinearized_output_index_[jj] * effective_input_broadcast_strides_[ii-1][jj]`.
> `operator+(n)` returns a copy advanced by `n` (see
> `[spec:et:sem:broadcast-indexes-range.torch.executor.internal.operator-fn]`).

> [spec:et:def:broadcast-indexes-range.torch.executor.internal.effective-input-broadcast-stride-fn]
> ShapeType effective_input_broadcast_stride( const Tensor& output, const Tensor& t) const

> [spec:et:sem:broadcast-indexes-range.torch.executor.internal.effective-input-broadcast-stride-fn]
> Builds the per-output-dimension stride array used to advance an input's flat
> index as the output odometer steps, with broadcasting baked in. Returns a
> `ShapeType` (`std::array<size_t, kTensorDimensionLimit>`) `result` aligned to
> the OUTPUT dims. Steps:
> 1. ET_CHECK_MSG(`t.dim() <= output.dim()`): the input rank must not exceed the
>    output rank; aborts otherwise.
> 2. `num_leading_ones = output.dim() - t.dim()` — the count of implicit leading
>    size-1 dims the input is padded with to align with the output.
> 3. For output dims `idx` in `[0, num_leading_ones)`: `result[idx] = 0` (these
>    padded dims never advance the input pointer).
> 4. For output dims `idx` in `[num_leading_ones, num_leading_ones + t.dim())`:
>    let `k = idx - num_leading_ones` be the input's own dim. If
>    `t.sizes()[k] == 1`, `result[idx] = 0` (this dim is broadcast, so stepping
>    the output along it must not move the input); otherwise
>    `result[idx] = t.strides()[k]` (the input's real stride for that dim).
> 5. Entries beyond `num_leading_ones + t.dim()` remain 0 (default-init).
> The resulting array lets the iterator compute an input's flat index as the sum
> over output dims of `delinearized_output_index_[i] * result[i]`.

> [spec:et:def:broadcast-indexes-range.torch.executor.internal.operator-fn]
> BroadcastIndexesIterator operator+(difference_type n)

> [spec:et:sem:broadcast-indexes-range.torch.executor.internal.operator-fn]
> `operator+(difference_type n)` for the general iterator: copies the current
> iterator into `it`, applies `it += n` (per the `operator+=` behavior described
> in `[spec:et:sem:broadcast-indexes-range.torch.executor.internal.broadcast-indexes-iterator-operator-fn]`
> — `std::advance` for `n <= 3`, otherwise recompute all input indices from the
> delinearized output index), and returns the advanced copy `it` by value. The
> receiver is left unchanged.

> [spec:et:def:broadcast-indexes-range.torch.executor.internal.output-index-fn]
> ssize_t output_index() const

> [spec:et:sem:broadcast-indexes-range.torch.executor.internal.output-index-fn]
> Private accessor of the general iterator: the const overload returns
> `current_indexes_[0]`, the current output linear index. (A non-const overload
> returns a mutable reference `ssize_t&` to the same slot, used by `operator++`
> and `operator+=` to advance it.) This value drives iterator equality and
> distance comparisons.

> [spec:et:def:broadcast-indexes-range.torch.executor.internal.sizes-match-ignoring-leading-1s-fn]
> inline bool sizes_match_ignoring_leading_1s( ArrayRef<Tensor::SizesType> lhs, ArrayRef<Tensor::SizesType> rhs)

> [spec:et:sem:broadcast-indexes-range.torch.executor.internal.sizes-match-ignoring-leading-1s-fn]
> Returns true iff two size arrays `lhs` and `rhs` describe the same shape once
> leading size-1 dimensions are stripped from each. Steps:
> 1. `lhs_begin = arrayref_begin_ignoring_leading_1s(lhs)` (first non-1 element,
>    per `[spec:et:sem:broadcast-indexes-range.torch.executor.internal.arrayref-begin-ignoring-leading-1s-fn]`),
>    `lhs_end = lhs.end()`; likewise for `rhs`.
> 2. Require the two trimmed lengths to be equal
>    (`lhs_end - lhs_begin == rhs_end - rhs_begin`) AND
>    `std::equal(lhs_begin, lhs_end, rhs_begin)` — element-wise equality over the
>    trimmed ranges.
> Return the conjunction. This is the test the iterator uses to detect that an
> input needs no broadcasting relative to the output (only leading 1s differ),
> enabling the fast path where the input index tracks the output index directly.

