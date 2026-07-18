# Rust kernel registration

> [spec:et:req:kernel-registration.quiet-idempotent-probe]
> The explicit Rust CPU-kernel registration entry point MUST determine whether
> a fallback kernel is already registered without invoking a lookup that emits
> an error for an expected miss. Initial registration MUST NOT report missing
> kernels merely because they have not yet been registered, repeated calls MUST
> remain idempotent, and optimized kernels MUST retain precedence over portable
> fallbacks.
