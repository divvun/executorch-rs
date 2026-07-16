pub mod optimized;
pub mod portable;
pub mod quantized;
// NOT part of the literal port: the operator-registration table assembled from
// `#[et_kernel]`-annotated kernels (stand-in for the C++ RegisterKernels codegen).
pub mod registry;
