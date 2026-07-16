//! Literal port of runtime/core/tensor_shape_dynamism.h.

/// The resizing capabilities of a Tensor.
///
/// The rank of an ExecuTorch Tensors can never change, but shape sometimes can.
// [spec:et:def:tensor-shape-dynamism.executorch.runtime.tensor-shape-dynamism]
#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[allow(non_camel_case_types)] // literal C++ enumerator names preserved
pub enum TensorShapeDynamism {
    /// Cannot change shape.
    STATIC = 0,
    /// Shape cannot exceed initial capacity.
    DYNAMIC_BOUND = 1,
    /// No restriction on shape and capacity.
    DYNAMIC_UNBOUND = 2,
}

// Literal port of runtime/core/test/tensor_shape_dynamism_test_aten.cpp.
//
// PORT-NOTE: The C++ file is an ATen-mode-only build check
// (`#ifndef USE_ATEN_LIB #error`) whose sole runtime assertion is that two
// enumerators differ. The Rust port has a single `TensorShapeDynamism` enum
// (no separate ATen surface), so only the enumerator-inequality assertion maps;
// the aten-mode `#error` build gate is out of scope.
#[cfg(test)]
mod tests {
    use super::*;

    // [spec:et:sem:tensor-shape-dynamism.executorch.runtime.tensor-shape-dynamism/test]
    #[test]
    fn tensor_shape_dynamism_test_can_build_in_aten_mode() {
        assert_ne!(
            TensorShapeDynamism::STATIC,
            TensorShapeDynamism::DYNAMIC_BOUND
        );
    }
}
