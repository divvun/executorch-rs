# runtime/core/exec_aten/exec_aten.h

> [spec:et:def:exec-aten.executorch.aten.compute-numel-fn]
> inline ssize_t compute_numel(const SizesType* sizes, ssize_t dim)

> [spec:et:sem:exec-aten.executorch.aten.compute-numel-fn]
> Free function computing the total number of elements (numel) of a tensor
> from its sizes. Arguments: `sizes` (pointer to a `SizesType` array of
> length `dim`) and `dim` (`ssize_t`, the number of dimensions).
>
> Behavior: form the array view `sizes[0..dim)` and return the product of all
> its entries as an `ssize_t`. For `dim == 0` (scalar tensor) the product of
> zero factors is `1`; `sizes` may be null in that case since it is not
> dereferenced. If any size is `0`, the product is `0`.
>
> This is the unchecked variant: it does NOT validate that sizes are
> non-negative and does NOT detect integer overflow of the running product
> (multiplication wraps per two's-complement `ssize_t` arithmetic). Callers
> that need those guarantees use
> `[spec:et:sem:exec-aten.executorch.aten.safe-numel-fn]` instead. In the
> executor (non-aten) build this name aliases the equivalent
> `torch::executor::compute_numel`; the port needs one implementation with
> this behavior.

> [spec:et:def:exec-aten.executorch.aten.safe-numel-fn]
> inline ::executorch::runtime::Result<ssize_t> safe_numel( const SizesType* sizes, ssize_t dim)

> [spec:et:sem:exec-aten.executorch.aten.safe-numel-fn]
> Free function computing numel with input validation and overflow checking,
> returning a `Result<ssize_t>`. Arguments: `sizes` (pointer to a `SizesType`
> array of length `dim`) and `dim` (`ssize_t`).
>
> Steps (each check uses `ET_CHECK_OR_RETURN_ERROR`: on failure it emits a
> log message and immediately returns an error `Result` carrying
> `Error::InvalidArgument`; on success execution continues):
> - Check that `dim == 0` OR `sizes != nullptr`. If it fails (non-scalar
>   tensor with a null sizes pointer), return `Error::InvalidArgument`
>   ("Sizes must be provided for non-scalar tensors").
> - Initialize accumulator `numel = 1` (`ssize_t`).
> - Iterate `i` from `0` to `dim - 1` inclusive, in ascending order:
>   - Check `sizes[i] >= 0`. If a size is negative, return
>     `Error::InvalidArgument` ("Size must be non-negative, got %zd at
>     dimension %zd").
>   - Compute `next_numel = numel * sizes[i]` using a checked multiply
>     (`c10::mul_overflows` on `ssize_t`). If the multiplication overflows
>     the `ssize_t` range, return `Error::InvalidArgument` ("Overflow
>     computing numel at dimension %zd").
>   - Set `numel = next_numel`.
> - After the loop, return the `ssize_t` `numel` as a successful `Result`.
>
> For `dim == 0` the loop body does not run and the result is `1`. If any
> size is `0`, the accumulated numel becomes `0` and stays `0` (never
> overflows). Unlike
> `[spec:et:sem:exec-aten.executorch.aten.compute-numel-fn]`, this validates
> non-negativity and overflow. In the executor (non-aten) build this name
> aliases `torch::executor::safe_numel`; the port needs one implementation
> with this behavior.

