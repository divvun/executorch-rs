# kernels/portable/cpu/op_allclose.cpp

> [spec:et:def:op-allclose.torch.executor.native.allclose-tensor-fn]
> Tensor allclose_tensor( ET_UNUSED const Tensor& self, ET_UNUSED const Tensor& other, ET_UNUSED double rtol, ET_UNUSED double atol, ET_UNUSED bool equal_nan, ET_UNUSED bool dummy_param)

> [spec:et:sem:op-allclose.torch.executor.native.allclose-tensor-fn]
> `allclose_tensor(self, other, rtol, atol, equal_nan, dummy_param)`: the
> functional (non-out) variant that returns a freshly allocated bool tensor. This
> variant only exists for ATen-mode registration on the compiler side.
>
> - When compiled with `USE_ATEN_LIB` defined: allocates a scalar Bool tensor
>   `out` initialized to `false`, calls `allclose_out(self, other, rtol, atol,
>   equal_nan, dummy_param, out)` per
>   `[spec:et:sem:op-allclose.torch.executor.native.allclose-out-fn]`, and
>   returns `out`.
> - Otherwise (ExecuTorch runtime build): `ET_ASSERT_UNREACHABLE()` — this path
>   is never reachable and aborts if invoked. (A `ctx`-taking overload likewise
>   just calls `ET_ASSERT_UNREACHABLE()`.)
>
> A clean-room Rust port targeting the runtime need not implement the functional
> variant; only `allclose.out` is exposed at runtime.

> [spec:et:def:op-allclose.torch.executor.native.data-is-close-fn]
> bool data_is_close( const T* a, const T* b, size_t numel, double rtol, double atol)

> [spec:et:sem:op-allclose.torch.executor.native.data-is-close-fn]
> Template `data_is_close<T>(a, b, numel, rtol, atol)`: returns true iff every
> pair `a[i], b[i]` (flat index `i` in `[0, numel)`) is close. T is a floating
> point type (Float, Double, Half, BFloat16).
>
> Per element:
> - If `rtol == 0 && atol == 0`: exact comparison — if `a[i] != b[i]`, return
>   false immediately.
> - Otherwise: `allowed_error = atol + std::fabs(rtol * b[i])`;
>   `actual_error = std::fabs(a[i] - b[i])`; if `!std::isfinite(actual_error)`
>   (NaN or inf error, e.g. from NaN operands or inf difference) OR
>   `actual_error > allowed_error`, return false immediately.
>
> If all elements pass, return true. Note the tolerance band is asymmetric: it
> scales with `b` (the second argument), not `a`. `equal_nan` is not honored here
> — any NaN produces a non-finite `actual_error` and returns false.

> [spec:et:def:op-allclose.torch.executor.native.tensors-are-close-fn]
> bool tensors_are_close( const Tensor& a, const Tensor& b, double rtol, double atol)

> [spec:et:sem:op-allclose.torch.executor.native.tensors-are-close-fn]
> `tensors_are_close(a, b, rtol, atol)`: returns true iff all elements of `a` and
> `b` are close. Assumes `a` and `b` have the same shape and dtype and are
> contiguous (strides are not consulted).
>
> Dispatch on `a.scalar_type()`:
> - Float: `data_is_close<float>(a_ptr, b_ptr, a.numel(), rtol, atol)`.
> - Double: `data_is_close<double>(...)`.
> - Half: `data_is_close<Half>(...)`.
> - BFloat16: `data_is_close<BFloat16>(...)`.
> - Any other (non-floating-point) dtype: bitwise `memcmp(a.mutable_data_ptr(),
>   b.mutable_data_ptr(), a.nbytes()) == 0` — exact byte equality, ignoring
>   `rtol`/`atol`.
>
> Floating cases follow
> `[spec:et:sem:op-allclose.torch.executor.native.data-is-close-fn]`. Returns the
> resulting bool.

> [spec:et:def:op-allclose.torch.executor.native.allclose-out-fn]
> Tensor& allclose_out( const Tensor& self, const Tensor& other, double rtol, double atol, ET_UNUSED bool equal_nan, ET_UNUSED bool dummy_param, Tensor& out)

> [spec:et:sem:op-allclose.torch.executor.native.allclose-out-fn]
> Implements `allclose.out(self, other, rtol, atol, equal_nan, dummy_param,
> out)`: writes a single-element bool result into `out` indicating whether
> `self` and `other` are element-wise close. `equal_nan` and `dummy_param` are
> unused. (Two overloads exist: one without `ctx` containing the real logic, and
> a `ctx`-taking wrapper that discards `ctx` and delegates to it.)
>
> Preconditions (each `ET_CHECK*`, which aborts/fails hard rather than setting a
> context Error):
> 1. `ET_CHECK_SAME_SHAPE_AND_DTYPE2(self, other)` — same shape and dtype.
> 2. `out.scalar_type() == ScalarType::Bool` (else fails with a type message).
> 3. `tensors_have_same_dim_order(self, other, out)`.
> 4. `out.numel() == 1` (single element).
>
> Then `out_data[0] = tensors_are_close(self, other, rtol, atol)` per
> `[spec:et:sem:op-allclose.torch.executor.native.tensors-are-close-fn]`. Returns
> `out`.

