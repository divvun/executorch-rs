//! Literal port of kernels/portable/cpu/selective_build.h.

use crate::runtime::core::portable_type::scalar_type::ScalarType;

// PORT-NOTE: The C++ header has two branches keyed on the
// `EXECUTORCH_SELECTIVE_BUILD_DTYPE` preprocessor macro. When defined it
// includes a codegen-generated `selected_op_variants.h`; that generated header
// is out of scope (kernel registration / functions.yaml codegen is not ported).
// Only the default (non-selective) branch is translated here, which
// unconditionally returns `true`.

// dummy implementation
// [spec:et:def:selective-build.should-include-kernel-dtype-fn]
// [spec:et:sem:selective-build.should-include-kernel-dtype-fn]
pub const fn should_include_kernel_dtype(_operator_name: &str, _scalar_type: ScalarType) -> bool {
    true
}

// PORT-NOTE: `ET_INTERNAL_CHECK_SELECTIVE_BUILD` is a C++ macro consumed by the
// ET_SWITCH dtype-dispatch machinery. In this port dtype dispatch lives in
// scalar_type_util.rs and does not currently wire in a selective-build gate, so
// this macro has no direct call site and is omitted. Unresolved cross-module
// reference: if selective build is later implemented, the `et_switch_*_types!`
// macros must consult `should_include_kernel_dtype`.

#[cfg(test)]
mod tests {
    use super::*;

    // Only the default (non-selective) branch is ported: unconditionally true for
    // any operator name / dtype.
    // [spec:et:sem:selective-build.should-include-kernel-dtype-fn/test]
    #[test]
    fn should_include_kernel_dtype_default_true() {
        assert!(should_include_kernel_dtype("add.out", ScalarType::Float));
        assert!(should_include_kernel_dtype("", ScalarType::Bool));
        assert!(should_include_kernel_dtype("mul.out", ScalarType::Long));
    }
}
