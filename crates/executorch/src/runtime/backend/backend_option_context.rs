//! Literal port of runtime/backend/backend_option_context.h.

/// BackendOptionContext will be used to inject runtime info for to initialize
/// delegate.
// [spec:et:def:backend-option-context.executorch.et-runtime-namespace.backend-option-context]
pub struct BackendOptionContext {}

impl BackendOptionContext {
    // [spec:et:def:backend-option-context.executorch.et-runtime-namespace.backend-option-context.backend-option-context-fn]
    // [spec:et:sem:backend-option-context.executorch.et-runtime-namespace.backend-option-context.backend-option-context-fn]
    pub fn new() -> Self {
        BackendOptionContext {}
    }
}

impl Default for BackendOptionContext {
    fn default() -> Self {
        BackendOptionContext::new()
    }
}
