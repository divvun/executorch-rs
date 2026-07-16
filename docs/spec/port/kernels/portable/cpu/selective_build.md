# kernels/portable/cpu/selective_build.h

> [spec:et:def:selective-build.should-include-kernel-dtype-fn]
> inline constexpr bool should_include_kernel_dtype( const char* /*operator_name*/, executorch::aten::ScalarType /*scalar_type*/ )

> [spec:et:sem:selective-build.should-include-kernel-dtype-fn]
> Compile-time predicate deciding whether a kernel should be built/run for a
> given `(operator_name, scalar_type)` pair, used by the ET_SWITCH dtype-dispatch
> machinery (via `ET_INTERNAL_CHECK_SELECTIVE_BUILD`) to abort at runtime on a
> deselected dtype. This is the DEFAULT (non-selective) implementation, compiled
> when `EXECUTORCH_SELECTIVE_BUILD_DTYPE` is NOT defined: it ignores both
> arguments and unconditionally returns `true` (every operator/dtype combination
> is included). When `EXECUTORCH_SELECTIVE_BUILD_DTYPE` IS defined, this
> definition is replaced by a generated `selected_op_variants.h` header whose
> version returns true only for the operator/dtype pairs selected at build time;
> a Rust port that does not implement dtype selective build should return `true`
> unconditionally to match this default.

