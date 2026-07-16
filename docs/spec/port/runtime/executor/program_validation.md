# runtime/executor/program_validation.cpp

> [spec:et:def:program-validation.executorch.runtime.validate-program-fn]
> ET_NODISCARD Error

> [spec:et:sem:program-validation.executorch.runtime.validate-program-fn]
> Performs semantic validation of a parsed flatbuffer `Program` beyond what the
> flatbuffer `Verifier` checks: verifies tensor size/numel/nbytes arithmetic and
> that TensorList entries reference valid Tensor evalues. Returns `Error::Ok` if
> the whole program is valid, else the first violation's error (always
> `Error::InvalidProgram` here). Called from
> `[spec:et:sem:program.executorch.et-runtime-namespace.program.load-fn]` under
> InternalConsistency verification.
>
> Steps:
> 1. If `program == nullptr`, log and return `Error::InvalidProgram`.
> 2. `execution_plans = program->execution_plan()`. If null, log and return
>    `Error::InvalidProgram`.
> 3. For each `plan_idx` from 0 to `execution_plans->size() - 1` in order:
>    a. `plan = execution_plans->Get(plan_idx)`. If null, log and return
>       `Error::InvalidProgram`.
>    b. `values = plan->values()`. If null, log and return
>       `Error::InvalidProgram`.
>    c. Capture `inputs = plan->inputs()` for the dynamic-input predicate
>       (below). Note `inputs` may be null; the predicate then reports false.
>    d. For each `value_idx` from 0 to `values->size() - 1` in order:
>       - `value = values->Get(value_idx)`. If null, `continue` (skip this
>         value).
>       - If `value->val_type() == KernelTypes::Tensor`: reinterpret
>         `value->val()` as a `Tensor*` and call `validate_tensor`
>         (`[spec:et:sem:program-validation.executorch.runtime.validate-tensor-fn]`).
>         If it returns non-Ok, apply the dynamic-input exception: if this
>         `value_idx` is a dynamic input (see predicate), log Info and continue
>         (tolerate the failure); otherwise log Error and return that error.
>       - If `value->val_type() == KernelTypes::TensorList` (checked
>         independently, not `else`): reinterpret `value->val()` as a
>         `TensorList*`. If that pointer is null, log and return
>         `Error::InvalidProgram`. `items = tensor_list->items()`; if null, log
>         and return `Error::InvalidProgram`. For each `item_idx` from 0 to
>         `items->size() - 1`: read `evalue_index = items->Get(item_idx)` (an
>         `int32_t`). If `evalue_index < 0 || (uint)evalue_index >=
>         values->size()`, log and return `Error::InvalidProgram`. Fetch
>         `referenced_value = values->Get(evalue_index)`; if null, log and return
>         `Error::InvalidProgram`; if its `val_type() != KernelTypes::Tensor`,
>         log and return `Error::InvalidProgram`.
> 4. After all plans/values pass, return `Error::Ok`.
>
> Dynamic-input predicate `is_dynamic_input(idx)`: returns true iff `inputs` is
> non-null AND some entry `inputs->Get(i)` equals `idx`, AND `values->Get(idx)`
> is non-null, AND its `val()` reinterpreted as `Tensor*` is non-null with
> `shape_dynamism() != TensorShapeDynamism::STATIC`. Rationale: dynamic input
> tensors may carry 64-bit upper-bound sizes that overflow numel on 32-bit
> targets; their real sizes arrive at `set_input` time, so their validation
> failures are deferred rather than fatal.

> [spec:et:def:program-validation.executorch.runtime.validate-tensor-fn]
> ET_NODISCARD Error

> [spec:et:sem:program-validation.executorch.runtime.validate-tensor-fn]
> Validates a single flatbuffer `Tensor`: non-negative dimensions, valid scalar
> type, and that both `numel` and `numel * elementSize` fit without overflow.
> Returns `Error::Ok` if valid, else `Error::InvalidProgram`.
>
> Steps:
> 1. If `tensor == nullptr`, log and return `Error::InvalidProgram`.
> 2. `sizes = tensor->sizes()` (flatbuffer vector of `int32_t`). If null, log and
>    return `Error::InvalidProgram`.
> 3. Initialize `numel` = 1 (as `ssize_t`) and a `numel_overflowed` flag = false.
>    Iterate `i` from 0 to `sizes->size() - 1` in order:
>    - `size = sizes->Get(i)` (`int32_t`). If `size < 0`, log and return
>      `Error::InvalidProgram`.
>    - While `numel_overflowed` is still false, multiply: `numel_overflowed =
>      mul_overflows(numel, (ssize_t)size, &numel)` — i.e. accumulate
>      `numel *= size` in `ssize_t`, latching the flag on the first multiply that
>      overflows. Once latched, stop updating `numel` (but keep scanning
>      remaining dims for the `size < 0` check). An empty `sizes` leaves `numel`
>      = 1 (scalar).
> 4. `scalar_type = (aten::ScalarType)tensor->scalar_type()`. If
>    `isValid(scalar_type)` is false, log and return `Error::InvalidProgram`.
> 5. If `numel_overflowed` is true, return `Error::InvalidProgram` (note: this is
>    checked only after the scalar-type validity check, so an invalid scalar type
>    is reported first).
> 6. Compute `nbytes = (size_t)numel * elementSize(scalar_type)` via
>    `mul_overflows`; if that multiply overflows, return `Error::InvalidProgram`.
> 7. Otherwise return `Error::Ok`.
>
> `numel` is accumulated in signed `ssize_t`; the final byte-size product is
> computed in unsigned `size_t`. `elementSize(scalar_type)` is the per-element
> byte width for that dtype. This function does not read or dereference tensor
> data — it validates metadata only.

