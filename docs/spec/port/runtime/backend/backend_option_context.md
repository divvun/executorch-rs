# runtime/backend/backend_option_context.h

> [spec:et:def:backend-option-context.executorch.et-runtime-namespace.backend-option-context]
> class BackendOptionContext final

> [spec:et:def:backend-option-context.executorch.et-runtime-namespace.backend-option-context.backend-option-context-fn]
> explicit BackendOptionContext()

> [spec:et:sem:backend-option-context.executorch.et-runtime-namespace.backend-option-context.backend-option-context-fn]
> Explicit default constructor with an empty body (`explicit BackendOptionContext() {}`). Takes no arguments and does nothing — the class has no data members. Constructs an empty context object used as a placeholder handle when getting/setting backend options. In a Rust port this is a unit/empty struct with a trivial constructor.

