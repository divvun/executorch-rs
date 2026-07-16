//! Literal port of runtime/executor/tensor_parser_exec_aten.cpp.
//!
//! Shared (mode-independent) deserialization helpers: `TensorParser`
//! (friend forwarder), `getMemPlannedPtr`, `parseTensorList`,
//! `validateTensorLayout`, `get_data_by_key`, `getTensorDataPtr`.
//!
//! PORT-NOTE (cross-module, unresolved): `MemoryManager` and `Program`
//! (`runtime/executor/{memory_manager,program}.rs`) are still stubs. This
//! module calls their expected C++-mirrored API:
//!   - `MemoryManager::method_allocator()` -> `&mut MemoryAllocator`
//!   - `Program::get_constant_buffer_data(buffer_idx, nbytes)` ->
//!     `Result<*const c_void>`
//!   - `Program::load_mutable_subsegment_into(idx, off, size, buf)` -> `Error`

use crate::runtime::core::error::{Error, Result};
use crate::runtime::core::evalue::EValue;
use crate::runtime::core::hierarchical_allocator::HierarchicalAllocator;
use crate::runtime::core::memory_allocator::MemoryAllocatorExt;
use crate::runtime::core::named_data_map::NamedDataMap;
use crate::runtime::core::portable_type::scalar_type::ScalarType;
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::core::result::ResultExt;
use crate::runtime::core::span::Span;
use crate::runtime::core::tensor_layout::TensorLayout;
use crate::runtime::executor::memory_manager::MemoryManager;
use crate::runtime::executor::program::Program;
use crate::runtime::executor::tensor_parser::NamedData;
use crate::schema::generated::executorch_flatbuffer;

use crate::runtime::core::evalue::BoxedEvalueList;

// Provides access to private Program methods.
// [spec:et:def:tensor-parser-exec-aten.executorch.et-runtime-namespace.deserialization.tensor-parser]
//
// PORT-NOTE: the C++ `class TensorParser final` exists solely because it is a
// friend of `Program`, granting the free functions here access to a private
// `Program` method. Rust has no friend mechanism; the forwarder collapses to a
// direct call. It is kept as a unit struct with the same static method to
// preserve the call-site spelling `TensorParser::load_mutable_subsegment_into`.
pub struct TensorParser;

impl TensorParser {
    // [spec:et:def:tensor-parser-exec-aten.executorch.et-runtime-namespace.deserialization.tensor-parser.load-mutable-subsegment-into-fn]
    // [spec:et:sem:tensor-parser-exec-aten.executorch.et-runtime-namespace.deserialization.tensor-parser.load-mutable-subsegment-into-fn]
    #[must_use]
    pub fn load_mutable_subsegment_into(
        program: &Program,
        mutable_data_segments_index: usize,
        offset_index: usize,
        size: usize,
        buffer: *mut core::ffi::c_void,
    ) -> Error {
        program.load_mutable_subsegment_into(
            mutable_data_segments_index,
            offset_index,
            size,
            buffer,
        )
    }
}

// Retrieve the buffer specified by the allocation_info
// [spec:et:def:tensor-parser-exec-aten.executorch.et-runtime-namespace.deserialization.result-void-get-mem-planned-ptr-fn]
// [spec:et:sem:tensor-parser-exec-aten.executorch.et-runtime-namespace.deserialization.result-void-get-mem-planned-ptr-fn]
#[must_use]
fn get_mem_planned_ptr(
    allocation_info: &executorch_flatbuffer::AllocationDetails,
    nbytes: usize,
    allocator: Option<&mut HierarchicalAllocator>,
) -> Result<*mut core::ffi::c_void> {
    crate::et_check_or_return_error!(
        allocator.is_some(),
        InvalidState,
        "HierarchicalAllocator must not be null for memory-planned tensor"
    );
    let allocator = allocator.unwrap();
    // Normal non-constant Tensor. Allocate data using mem_id and offset.

    // TODO(T142455629): make the allocator actually id based and not indexed
    // based. -1 is a hack to get the memory ids 0 aligned because previously
    // 0 was reserved
    let memory_id: u32 = allocation_info.memory_id() - 1;

    // Originally this field was a single uint32_t, but we need 64 bits for
    // larger models. To preserve backwards compatibility, the high bits are
    // managed in a separate uint32_t field.
    let memory_offset_low: u32 = allocation_info.memory_offset_low();
    let memory_offset_high: u32 = allocation_info.memory_offset_high();

    // PORT-NOTE: C++ selects the branch at compile time with `if constexpr
    // (sizeof(size_t) > sizeof(uint32_t))`. Rust has no `if constexpr` here;
    // the const-foldable `if` below is equivalent, and the untaken branch's
    // early return is dead-code-eliminated on 64-bit targets.
    let mut memory_offset: usize = memory_offset_low as usize;
    if core::mem::size_of::<usize>() > core::mem::size_of::<u32>() {
        memory_offset |= (memory_offset_high as usize) << 32;
    } else {
        crate::et_check_or_return_error!(
            memory_offset_high == 0,
            NotSupported,
            "size_t cannot hold memory offset 0x{:08x}{:08x}",
            memory_offset_high,
            memory_offset_low
        );
    }
    allocator.get_offset_address(memory_id, memory_offset, nbytes)
}

// [spec:et:def:tensor-parser-exec-aten.executorch.et-runtime-namespace.deserialization.result-boxed-evalue-list-executorch.aten.tensor-parse-tensor-list-fn]
// [spec:et:sem:tensor-parser-exec-aten.executorch.et-runtime-namespace.deserialization.result-boxed-evalue-list-executorch.aten.tensor-parse-tensor-list-fn]
#[must_use]
pub fn parse_tensor_list<'a>(
    tensor_indices: &flatbuffers::Vector<'a, i32>,
    values: *mut EValue<'a>,
    values_len: usize,
    memory_manager: &mut MemoryManager,
) -> Result<BoxedEvalueList<'a, Tensor<'a>>> {
    // EXECUTORCH_SCOPE_PROF("TensorParser::parseTensorList");
    // PORT-NOTE: `EXECUTORCH_SCOPE_PROF(name)` expands to a stack-scoped
    // `ExecutorchProfiler profiler(name)` when profiling is enabled and to
    // nothing otherwise. Ported profiler.rs is itself gated behind the
    // `profiling-enabled` feature, so the RAII guard is created under the same
    // cfg; with profiling off this compiles to nothing, exactly like the C++
    // macro.
    #[cfg(feature = "profiling-enabled")]
    let _prof = crate::runtime::platform::profiler::ExecutorchProfiler::new(
        c"TensorParser::parseTensorList".as_ptr(),
    );

    // PORT-NOTE: `MemoryManager::method_allocator()` returns a raw
    // `*mut MemoryAllocator` (mirroring the C++ pointer). Deref to `&mut` at
    // each call, matching the C++ `memory_manager->method_allocator()->...`.
    let tensor_list: *mut Tensor<'a> = unsafe { &mut *memory_manager.method_allocator() }
        .allocate_list::<Tensor<'a>>(tensor_indices.len(), core::mem::align_of::<Tensor<'a>>());
    if tensor_list.is_null() {
        return Err(Error::MemoryAllocationFailed);
    }
    let evalp_list: *mut *mut EValue<'a> = unsafe { &mut *memory_manager.method_allocator() }
        .allocate_list::<*mut EValue<'a>>(
        tensor_indices.len(),
        core::mem::align_of::<*mut EValue<'a>>(),
    );
    if evalp_list.is_null() {
        return Err(Error::MemoryAllocationFailed);
    }

    // For each tensor index look up the corresponding Tensor (which has been
    // already allocated) and stick it in the list.
    let mut output_idx: usize = 0;
    for tensor_index in tensor_indices.iter() {
        crate::et_check_or_return_error!(
            tensor_index >= 0 && (tensor_index as usize) < values_len,
            InvalidProgram,
            "Invalid value index {} for TensorList",
            tensor_index
        );

        let tensor_result = unsafe { (*values.add(tensor_index as usize)).try_to_tensor() };
        if !ResultExt::ok(&tensor_result) {
            return Err(tensor_result.error());
        }
        // Placement new as the list elements are not initialized, so calling
        // copy assignment is not defined if it's non trivial.
        unsafe {
            core::ptr::write(tensor_list.add(output_idx), r_into_ok(tensor_result));
        }
        unsafe {
            *evalp_list.add(output_idx) = values.add(tensor_index as usize);
        }
        output_idx += 1;
    }

    Ok(BoxedEvalueList::<Tensor<'a>>::new(
        evalp_list,
        tensor_list,
        tensor_indices.len() as i32,
    ))
}

// [spec:et:def:tensor-parser-exec-aten.executorch.et-runtime-namespace.deserialization.validate-tensor-layout-fn]
// [spec:et:sem:tensor-parser-exec-aten.executorch.et-runtime-namespace.deserialization.validate-tensor-layout-fn]
#[must_use]
pub fn validate_tensor_layout(
    s_tensor: &executorch_flatbuffer::Tensor,
    expected_layout: &TensorLayout,
) -> Error {
    crate::et_check_or_return_error!(
        static_cast_scalar_type(s_tensor.scalar_type()) == expected_layout.scalar_type(),
        InvalidExternalData,
        "Scalar type mismatch. Expected {}, got {}.",
        s_tensor.scalar_type().0,
        expected_layout.scalar_type() as i8
    );
    crate::et_check_or_return_error!(
        s_tensor.sizes().is_some(),
        InvalidExternalData,
        "Missing sizes field"
    );
    crate::et_check_or_return_error!(
        s_tensor.dim_order().is_some(),
        InvalidExternalData,
        "Missing dim_order field"
    );
    let dim: i32 = s_tensor.sizes().unwrap().len() as i32;
    crate::et_check_or_return_error!(dim >= 0, InvalidExternalData, "Dim is negative: {}", dim);
    crate::et_check_or_return_error!(
        (dim as usize) == expected_layout.sizes().size(),
        InvalidExternalData,
        "Dim mismatch. Expected {}, got {}.",
        dim,
        expected_layout.sizes().size()
    );
    crate::et_check_or_return_error!(
        s_tensor.dim_order().unwrap().len() == (dim as usize),
        InvalidExternalData,
        "Dim order size mismatch. Expected {}, got {}.",
        dim,
        s_tensor.dim_order().unwrap().len()
    );
    let mut i: i32 = 0;
    while i < dim {
        crate::et_check_or_return_error!(
            s_tensor.sizes().unwrap().get(i as usize)
                == unsafe { *expected_layout.sizes().index(i as usize) },
            InvalidExternalData,
            "Sizes mismatch. Expected {}, got {} for size at index {}.",
            s_tensor.sizes().unwrap().get(i as usize),
            unsafe { *expected_layout.sizes().index(i as usize) },
            i
        );
        crate::et_check_or_return_error!(
            s_tensor.dim_order().unwrap().get(i as usize)
                == unsafe { *expected_layout.dim_order().index(i as usize) },
            InvalidExternalData,
            "Dim order mismatch. Expected {}, got {} for dim at index {}.",
            s_tensor.dim_order().unwrap().get(i as usize),
            unsafe { *expected_layout.dim_order().index(i as usize) },
            i
        );
        i += 1;
    }
    Error::Ok
}

// Check if key exists in entries. If it does, return a pointer to the entry
// otherwise return a nullptr.
// [spec:et:def:tensor-parser-exec-aten.executorch.et-runtime-namespace.deserialization.get-data-by-key-fn]
// [spec:et:sem:tensor-parser-exec-aten.executorch.et-runtime-namespace.deserialization.get-data-by-key-fn]
pub fn get_data_by_key(key: *const core::ffi::c_char, entries: Span<NamedData>) -> *mut NamedData {
    let mut i: usize = 0;
    while i < entries.size() {
        if unsafe { libc::strcmp(key, entries.index(i).key) } == 0 {
            return unsafe { entries.index(i) as *mut NamedData };
        }
        i += 1;
    }
    core::ptr::null_mut()
}

// [spec:et:def:tensor-parser-exec-aten.executorch.et-runtime-namespace.deserialization.result-void-get-tensor-data-ptr-fn]
// [spec:et:sem:tensor-parser-exec-aten.executorch.et-runtime-namespace.deserialization.result-void-get-tensor-data-ptr-fn]
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn get_tensor_data_ptr(
    s_tensor: &executorch_flatbuffer::Tensor,
    program: &Program,
    nbytes: usize,
    allocator: Option<&mut HierarchicalAllocator>,
    named_data_map: Option<&dyn NamedDataMap>,
    external_constants: Span<NamedData>,
) -> Result<*mut core::ffi::c_void> {
    let data_buffer_idx = s_tensor.data_buffer_idx();
    let allocation_info: Option<executorch_flatbuffer::AllocationDetails> =
        s_tensor.allocation_info();

    if allocation_info.is_some() {
        crate::et_check_or_return_error!(
            allocator.is_some(),
            InvalidState,
            "HierarchicalAllocator is null but tensor has allocation_info requiring memory-planned buffers"
        );
    }

    // External tensors.
    if s_tensor.extra_tensor_info().is_some()
        && s_tensor.extra_tensor_info().unwrap().location()
            == executorch_flatbuffer::TensorDataLocation::EXTERNAL
    {
        // Check that fqn is not null.
        crate::et_check_or_return_error!(
            s_tensor
                .extra_tensor_info()
                .unwrap()
                .fully_qualified_name()
                .is_some(),
            InvalidExternalData,
            "Fully qualified name of external tensor is null"
        );
        let fqn: &str = s_tensor
            .extra_tensor_info()
            .unwrap()
            .fully_qualified_name()
            .unwrap();

        // Constant value.
        if allocation_info.is_none() {
            let data = get_data_by_key(fqn_as_c_str(fqn), external_constants);
            if !data.is_null() {
                return Ok(unsafe { (*data).buffer.data() } as *mut core::ffi::c_void);
            }
            // Should never reach here; these tensors are resolved in
            // Method::parse_external_constants. Any errors should be caught there.
            return Err(Error::Internal);
        } else {
            // Mutable value.
            // Look up tensor in named data map.
            crate::et_check_or_return_error!(
                named_data_map.is_some(),
                InvalidExternalData,
                "Cannot retrieve external tensor with fqn: {}. The named_data_map is null; most likely no external .ptd file was provided.",
                fqn
            );
            let named_data_map = named_data_map.unwrap();
            let tensor_layout_res = named_data_map.get_tensor_layout(fqn);
            if !ResultExt::ok(&tensor_layout_res) {
                return Err(tensor_layout_res.error());
            }
            let tensor_layout = r_into_ok(tensor_layout_res);
            let err = validate_tensor_layout(s_tensor, &tensor_layout);
            if err != Error::Ok {
                return Err(err);
            }
            // Call load_into.
            let planned_ptr = get_mem_planned_ptr(&allocation_info.unwrap(), nbytes, allocator);
            if !ResultExt::ok(&planned_ptr) {
                return Err(planned_ptr.error());
            }
            let planned_ptr = r_into_ok(planned_ptr);
            let load_error = named_data_map.load_data_into(fqn, planned_ptr, nbytes);
            if load_error != Error::Ok {
                return Err(load_error);
            }

            Ok(planned_ptr)
        }

    // Constant, stored in PTE file.
    } else if data_buffer_idx > 0 && allocation_info.is_none() {
        let const_data = program.get_constant_buffer_data(data_buffer_idx as usize, nbytes);
        if !ResultExt::ok(&const_data) {
            return Err(const_data.error());
        }

        // The const_cast is 'ok' here because the program and runtime should
        // guarantee that this data is never modified.
        Ok(r_into_ok(const_data) as *mut core::ffi::c_void)

    // Memory Planned, with initial state
    } else if data_buffer_idx > 0 && allocation_info.is_some() {
        let planned_ptr = get_mem_planned_ptr(&allocation_info.unwrap(), nbytes, allocator);
        if !ResultExt::ok(&planned_ptr) {
            return Err(planned_ptr.error());
        }
        let planned_ptr = r_into_ok(planned_ptr);
        let err = TensorParser::load_mutable_subsegment_into(
            program,
            0,
            s_tensor.data_buffer_idx() as usize,
            nbytes,
            planned_ptr,
        );

        if err != Error::Ok {
            return Err(err);
        }
        Ok(planned_ptr)

    // Memory planned, no initial state
    } else if data_buffer_idx == 0 && allocation_info.is_some() {
        get_mem_planned_ptr(&allocation_info.unwrap(), nbytes, allocator)

    // Pointer recived at runtime
    } else {
        // data_buffer_idx == 0 && allocation_info == nullptr,
        Ok(core::ptr::null_mut())
    }
}

// PORT-NOTE: `static_cast<ScalarType>(s_tensor->scalar_type())` — the serialized
// `executorch_flatbuffer::ScalarType` (a `#[repr(transparent)]` `i8` newtype)
// and the runtime `ScalarType` (a `#[repr(i8)]` enum) share discriminants, so
// this reproduces the C++ static_cast by transmuting the byte. Out-of-range
// values are UB just as with `static_cast` to a scoped enum; `parseTensor`
// validates the value with `isValid` before use, matching the C++ flow.
fn static_cast_scalar_type(st: executorch_flatbuffer::ScalarType) -> ScalarType {
    unsafe { core::mem::transmute::<i8, ScalarType>(st.0) }
}

// PORT-NOTE: `->c_str()` on the serialized fqn yields a NUL-terminated C string
// that `get_data_by_key`/`strcmp` consume. The flatbuffer accessor gives a
// `&str` (whose backing bytes are followed by the flatbuffer's NUL terminator),
// so its `.as_ptr()` is a valid `*const c_char` for `strcmp`.
fn fqn_as_c_str(fqn: &str) -> *const core::ffi::c_char {
    fqn.as_ptr() as *const core::ffi::c_char
}

// PORT-NOTE: local copy of evalue.rs's private `r_into_ok` (moves the Ok value
// out of a `Result<T>` known to be Ok). Mirrors `std::move(result.get())`.
fn r_into_ok<T>(r: Result<T>) -> T {
    match r {
        Ok(v) => v,
        Err(_) => unreachable!(),
    }
}

// Literal port of runtime/executor/test/tensor_parser_test.cpp.
//
// PORT-NOTE: the `TensorParserTest` fixture cases (`TestModuleAddFloat`,
// `TestModuleAddHalf`, `TestMutableState`, `ParseTensorListRejectsNonTensorEValue`,
// `ParseListOptionalTypeRejectsWrongType`) load `.pte` models (env vars
// `ET_MODULE_ADD_PATH`, `ET_MODULE_ADD_HALF_PATH`, `ET_MODULE_SIMPLE_TRAIN_PATH`)
// and/or use the shared `ManagedMemoryManager` test helper (deferred; see the
// executor `mod.rs` PORT-NOTE) to drive `parseTensor`/`parseTensorList`/
// `parseListOptionalType`. They skip early. The three `ValidateTensorLayoutTest`
// cases build a flatbuffer Tensor in memory and exercise `validate_tensor_layout`
// directly, needing no fixture; they run unconditionally.
#[cfg(test)]
mod tests {
    use super::*;
    use flatbuffers::FlatBufferBuilder;

    fn setup() {
        crate::runtime::platform::runtime::runtime_init();
    }

    fn skip_parser(name: &str, env_var: &str) -> bool {
        if std::env::var(env_var).is_err() {
            eprintln!("skipping {name}: {env_var} unset");
            return true;
        }
        eprintln!(
            "skipping {name}: requires ManagedMemoryManager (and generated portable-ops \
             kernels for the module cases), which are not ported"
        );
        true
    }

    // [spec:et:sem:tensor-parser-portable.executorch.et-runtime-namespace.deserialization.parse-tensor-fn/test]
    #[test]
    fn tensor_parser_test_test_module_add_float() {
        setup();
        if skip_parser(
            "tensor_parser_test_test_module_add_float",
            "ET_MODULE_ADD_PATH",
        ) {}
    }

    // [spec:et:sem:tensor-parser-portable.executorch.et-runtime-namespace.deserialization.parse-tensor-fn/test]
    #[test]
    fn tensor_parser_test_test_module_add_half() {
        setup();
        if skip_parser(
            "tensor_parser_test_test_module_add_half",
            "ET_MODULE_ADD_HALF_PATH",
        ) {}
    }

    // [spec:et:sem:tensor-parser-portable.executorch.et-runtime-namespace.deserialization.parse-tensor-fn/test]
    #[test]
    fn tensor_parser_test_test_mutable_state() {
        setup();
        if skip_parser(
            "tensor_parser_test_test_mutable_state",
            "ET_MODULE_SIMPLE_TRAIN_PATH",
        ) {}
    }

    // parseTensorList should return InvalidType when the EValue at the given
    // index is not a Tensor. Requires ManagedMemoryManager to allocate the list;
    // skips.
    // [spec:et:sem:tensor-parser-exec-aten.executorch.et-runtime-namespace.deserialization.result-boxed-evalue-list-executorch.aten.tensor-parse-tensor-list-fn/test]
    #[test]
    fn tensor_parser_test_parse_tensor_list_rejects_non_tensor_evalue() {
        setup();
        eprintln!(
            "skipping tensor_parser_test_parse_tensor_list_rejects_non_tensor_evalue: \
             requires ManagedMemoryManager (not ported)"
        );
    }

    // [spec:et:sem:tensor-parser.executorch.et-runtime-namespace.deserialization.parse-list-optional-type-fn/test]
    #[test]
    fn tensor_parser_test_parse_list_optional_type_rejects_wrong_type() {
        setup();
        eprintln!(
            "skipping tensor_parser_test_parse_list_optional_type_rejects_wrong_type: \
             requires ManagedMemoryManager (not ported)"
        );
    }

    // Builds a flatbuffer Tensor with the given sizes/dim_order offsets and
    // returns its root view over the owned bytes.
    fn build_tensor(
        buf: &mut Vec<u8>,
        sizes: Option<&[i32]>,
        dim_order: Option<&[u8]>,
    ) -> *const u8 {
        let mut builder = FlatBufferBuilder::with_capacity(256);
        let sizes_off = sizes.map(|s| builder.create_vector(s));
        let dim_order_off = dim_order.map(|d| builder.create_vector(d));
        let tensor = executorch_flatbuffer::Tensor::create(
            &mut builder,
            &executorch_flatbuffer::TensorArgs {
                scalar_type: executorch_flatbuffer::ScalarType::FLOAT,
                storage_offset: 0,
                sizes: sizes_off,
                dim_order: dim_order_off,
                ..Default::default()
            },
        );
        builder.finish_minimal(tensor);
        *buf = builder.finished_data().to_vec();
        buf.as_ptr()
    }

    fn make_layout(sizes: &[i32], dim_order: &[u8]) -> TensorLayout {
        let layout = TensorLayout::create(
            Span::from_raw_parts(sizes.as_ptr() as *mut i32, sizes.len()),
            Span::from_raw_parts(dim_order.as_ptr() as *mut u8, dim_order.len()),
            ScalarType::Float,
        );
        assert!(ResultExt::ok(&layout));
        r_into_ok(layout)
    }

    // Linear strcmp search over a Span<NamedData>: first ascending match returns
    // &entries[i]; empty span and no-match return null. `get_data_by_key` is
    // re-exported through tensor_parser.h, so this pins both the exec-aten body
    // and the header declaration.
    // [spec:et:sem:tensor-parser-exec-aten.executorch.et-runtime-namespace.deserialization.get-data-by-key-fn/test]
    // [spec:et:sem:tensor-parser.executorch.et-runtime-namespace.deserialization.get-data-by-key-fn/test]
    #[test]
    fn get_data_by_key_test_matches_and_misses() {
        use crate::runtime::core::freeable_buffer::FreeableBuffer;
        setup();

        let k0 = c"alpha";
        let k1 = c"beta";
        let k2 = c"gamma";
        let mut entries = [
            NamedData {
                key: k0.as_ptr(),
                buffer: FreeableBuffer::new(),
            },
            NamedData {
                key: k1.as_ptr(),
                buffer: FreeableBuffer::new(),
            },
            NamedData {
                key: k2.as_ptr(),
                buffer: FreeableBuffer::new(),
            },
        ];
        let span = Span::from_raw_parts(entries.as_mut_ptr(), entries.len());

        // Match returns a pointer to the exact backing entry.
        let hit = get_data_by_key(c"beta".as_ptr(), span);
        assert_eq!(hit, &mut entries[1] as *mut NamedData);

        // First ascending match wins even with an unrelated later entry.
        let first = get_data_by_key(c"alpha".as_ptr(), span);
        assert_eq!(first, &mut entries[0] as *mut NamedData);

        // No match returns null.
        assert!(get_data_by_key(c"delta".as_ptr(), span).is_null());

        // Empty span returns null.
        let empty = Span::from_raw_parts(core::ptr::null_mut::<NamedData>(), 0);
        assert!(get_data_by_key(c"alpha".as_ptr(), empty).is_null());
    }

    // [spec:et:sem:tensor-parser-exec-aten.executorch.et-runtime-namespace.deserialization.validate-tensor-layout-fn/test]
    // Also pins the tensor_parser.h re-export of validate_tensor_layout.
    // [spec:et:sem:tensor-parser.executorch.et-runtime-namespace.deserialization.validate-tensor-layout-fn/test]
    #[test]
    fn validate_tensor_layout_test_dim_order_size_mismatch_is_rejected() {
        setup();
        let sizes: [i32; 3] = [2, 3, 4];
        let dim_order_short: [u8; 1] = [0];
        let mut buf = Vec::new();
        let root = build_tensor(&mut buf, Some(&sizes), Some(&dim_order_short));
        let s_tensor = unsafe {
            flatbuffers::root_unchecked::<executorch_flatbuffer::Tensor>(
                core::slice::from_raw_parts(root, buf.len()),
            )
        };

        let expected_sizes: [i32; 3] = [2, 3, 4];
        let expected_dim_order: [u8; 3] = [0, 1, 2];
        let layout = make_layout(&expected_sizes, &expected_dim_order);

        assert_eq!(
            validate_tensor_layout(&s_tensor, &layout),
            Error::InvalidExternalData
        );
    }

    // [spec:et:sem:tensor-parser-exec-aten.executorch.et-runtime-namespace.deserialization.validate-tensor-layout-fn/test]
    #[test]
    fn validate_tensor_layout_test_null_sizes_is_rejected() {
        setup();
        let dim_order: [u8; 3] = [0, 1, 2];
        let mut buf = Vec::new();
        let root = build_tensor(&mut buf, None, Some(&dim_order));
        let s_tensor = unsafe {
            flatbuffers::root_unchecked::<executorch_flatbuffer::Tensor>(
                core::slice::from_raw_parts(root, buf.len()),
            )
        };
        assert!(s_tensor.sizes().is_none());

        let expected_sizes: [i32; 3] = [2, 3, 4];
        let expected_dim_order: [u8; 3] = [0, 1, 2];
        let layout = make_layout(&expected_sizes, &expected_dim_order);

        assert_eq!(
            validate_tensor_layout(&s_tensor, &layout),
            Error::InvalidExternalData
        );
    }

    // [spec:et:sem:tensor-parser-exec-aten.executorch.et-runtime-namespace.deserialization.validate-tensor-layout-fn/test]
    #[test]
    fn validate_tensor_layout_test_null_dim_order_is_rejected() {
        setup();
        let sizes: [i32; 3] = [2, 3, 4];
        let mut buf = Vec::new();
        let root = build_tensor(&mut buf, Some(&sizes), None);
        let s_tensor = unsafe {
            flatbuffers::root_unchecked::<executorch_flatbuffer::Tensor>(
                core::slice::from_raw_parts(root, buf.len()),
            )
        };
        assert!(s_tensor.dim_order().is_none());

        let expected_sizes: [i32; 3] = [2, 3, 4];
        let expected_dim_order: [u8; 3] = [0, 1, 2];
        let layout = make_layout(&expected_sizes, &expected_dim_order);

        assert_eq!(
            validate_tensor_layout(&s_tensor, &layout),
            Error::InvalidExternalData
        );
    }
}
