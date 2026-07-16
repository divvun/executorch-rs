//! Literal port of runtime/executor/tensor_parser_aten.cpp.
//!
//! ATen-mode `parseTensor` (and the `deleteNothing` no-op storage deleter).
//! This file is only compiled in ATen mode, mirroring the C++ file which is
//! only built when `USE_ATEN_LIB` is set.
//!
//! PORT-NOTE: the entire module is gated behind `#[cfg(feature = "aten")]`.
//! The ATen path depends on libtorch (`at::Tensor`, `at::from_blob`,
//! `at::CPU`, `at::getCPUAllocator`, `at::DataPtr`, `at::StorageImpl`), none of
//! which are available in this pure-Rust port — there is no `at::Tensor` type
//! and no bindings to the C++ ATen library. The bodies are therefore left as a
//! documented, unbuilt translation reference: the control flow, symbol names,
//! and spec annotations from the C++ source are preserved in comments below so
//! that a future FFI-backed ATen binding can be filled in literally. Until such
//! a binding exists (or `at::Tensor` is ported), this module compiles to
//! nothing. Unresolved cross-module reference: an ATen `Tensor` type + libtorch
//! FFI, plus the still-stubbed `MemoryManager`/`Program`.

#![cfg(feature = "aten")]

// [spec:et:def:tensor-parser-aten.executorch.runtime.aten.deserialization.delete-nothing-fn]
// [spec:et:sem:tensor-parser-aten.executorch.runtime.aten.deserialization.delete-nothing-fn]
//
// A no-op deleter with signature `void(void*)`. Passed to `at::from_blob` as the
// storage deleter so ATen never frees the executorch-owned data buffer. In Rust
// this is a deleter closure/fn that does nothing on the pointer.
//
// PORT-NOTE: cannot be exercised without libtorch's `at::from_blob` deleter
// slot; retained as the literal reference.
extern "C" fn delete_nothing(_: *mut core::ffi::c_void) {}

// [spec:et:def:tensor-parser-aten.executorch.runtime.aten.deserialization.parse-tensor-fn]
// [spec:et:sem:tensor-parser-aten.executorch.runtime.aten.deserialization.parse-tensor-fn]
//
// ATen-mode implementation of `parseTensor`. Literal reference of the C++
// control flow (no compilable body — needs the libtorch FFI described in the
// module PORT-NOTE):
//
//   1. EXECUTORCH_SCOPE_PROF("TensorParser::parseTensor").
//   2. ET_CHECK_OR_RETURN_ERROR(storage_offset() == 0, NotSupported, ...).
//   3. type = static_cast<at::ScalarType>(scalar_type());
//      ET_CHECK_OR_RETURN_ERROR(isValid(type), InvalidProgram, ...);
//      options = at::CPU(type).options();
//   4. ET_CHECK_OR_RETURN_ERROR(sizes() != nullptr, InvalidProgram, ...);
//      ndim = sizes()->size();
//   5. ET_CHECK_OR_RETURN_ERROR(ndim <= kTensorDimensionLimit, InvalidProgram, ...);
//   6. ET_CHECK_OR_RETURN_ERROR(dim_order() != nullptr, InvalidProgram, ...);
//      ET_CHECK_OR_RETURN_ERROR(dim_order()->size() == ndim, InvalidProgram, ...);
//   7. sizes: Vec<i64> widened from serialized int32 sizes;
//      strides: Vec<i64>(ndim) filled by
//      dim_order_to_stride(sizes()->data(), dim_order()->data(), ndim, strides);
//      ET_CHECK_OR_RETURN_ERROR(status == Error::Ok, Internal, ...);
//   8. tensor = at::from_blob(nullptr, sizes, strides, 0, delete_nothing, options);
//   9. if shape_dynamism() == DYNAMIC_UNBOUND:
//        storage.set_allocator(at::getCPUAllocator());
//        storage.set_resizable(true); storage.set_nbytes(0);
//        impl.set_sizes_contiguous(0);   // leave data null
//      else:
//        data_ptr = getTensorDataPtr(s_tensor, program, tensor.nbytes(),
//                     memory_manager->planned_memory(), named_data_map,
//                     external_constants);
//        if !data_ptr.ok(): ET_LOG(Error, "getTensorDataPtr() failed ..."); return err;
//        tensor.storage().set_data_ptr(at::DataPtr(data_ptr, CPU));
//   10. return tensor.

// Silence the unused-fn warning in the (currently always-empty at-runtime)
// compiled surface; `delete_nothing` documents the deleter slot.
#[allow(dead_code)]
const _: extern "C" fn(*mut core::ffi::c_void) = delete_nothing;
