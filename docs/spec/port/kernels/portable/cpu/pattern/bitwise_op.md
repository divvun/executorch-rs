# kernels/portable/cpu/pattern/bitwise_op.h

> [spec:et:def:bitwise-op.operator-fn]
> constexpr T operator()(const T& lhs, const T& rhs) const

> [spec:et:sem:bitwise-op.operator-fn]
> Functor call operator for a left-shift bit operation on two operands of the
> same integral type `T`. Given `lhs` and `rhs`, computes the C++ left-shift
> `lhs << rhs` and casts the result back to `T`: `static_cast<T>(lhs << rhs)`.
> No bounds/validity checking is performed on `rhs` — the shift is a raw C++
> `<<`, so shift counts that are negative or >= the bit width of `T`, and shifts
> that overflow the value range of `T`, follow C++'s (implementation-defined /
> undefined) shift semantics; the caller is responsible for supplying integral
> types (this functor is instantiated only for the INT-plus-Bool compute set).
> The sibling `bit_rshift` functor is identical but uses `>>` (arithmetic /
> logical right shift per C++ rules for the signedness of `T`). The three
> free-function templates `bitwise_and`, `bitwise_or`, `bitwise_xor` defined via
> `DEFINE_BINARY_OPERATOR_TEMPLATE` compute `val_a & val_b`, `val_a | val_b`,
> and `val_a ^ val_b` respectively (no cast wrapper), also on same-typed
> operands. This operator is the elementwise kernel applied by the
> `bitwise_tensor_out` / `bitwise_scalar_out` patterns: both promote input dtype
> with the other operand (`promoteTypes` for tensor-tensor,
> `promote_type_with_scalar` per
> `[spec:et:sem:scalar-utils.torch.executor.native.utils.promote-type-with-scalar-fn]`
> for tensor-scalar), ET_KERNEL_CHECK that `common_type` can cast to
> `out.scalar_type()`, ET_KERNEL_CHECK same dim order, resize `out` (broadcast
> target size for tensor-tensor, `a.sizes()` for scalar), pick a compute dtype
> over the INT-plus-Bool set (`ET_SWITCH_INT_TYPES_AND(Bool, ...)`), read inputs
> as INTB, apply this functor elementwise, and store into `out` as REALHBBF16;
> failed checks set Error::InvalidArgument and return `out` unchanged.

