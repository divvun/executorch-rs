//! Literal port of runtime/core/portable_type/tensor_options.h.

/// Tensor data memory formats supported by ExecuTorch. This concept only exists
/// for compatibility with ATen; use dim_order to describe non-contiguous
/// layouts.
// [spec:et:def:tensor-options.executorch.runtime.etensor.memory-format]
#[repr(i8)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MemoryFormat {
    /// Row-major contiguous data.
    Contiguous = 0,
    /// Output tensor format should remain the same as the input tensor format.
    /// E.g. if the input tensor is in channels_last format, operator output
    /// should be in channels_last format.
    Preserve = 1,
}

/// Tensor data memory layout. This concept only exists for compatibility
/// with ATen.
// [spec:et:def:tensor-options.executorch.runtime.etensor.layout]
#[repr(i8)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Layout {
    /// The tensor occupies memory densely and indexing is managed through
    /// strides. Contrasted with a sparse tensor layout where the memory
    /// structure of the data blob will be more complicated and indexing
    /// requires larger structures.
    ///
    /// This is the only layout supported by ExecuTorch.
    Strided = 0,
}
