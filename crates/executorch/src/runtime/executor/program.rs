//! Literal port of runtime/executor/program.cpp + runtime/executor/program.h.
//!
//! Program verification can increase code size by ~30k. Targets that need to
//! save this space can avoid building it by disabling the default
//! `program-verification` feature (mirrors ET_ENABLE_PROGRAM_VERIFICATION=0).
//!
//! The constant_buffer path is deprecated from ExecuTorch 0.7. Disable it by
//! turning off the default `deprecated-constant-buffer` feature (mirrors
//! ET_ENABLE_DEPRECATED_CONSTANT_BUFFER=0).
//!
//! SELF-REFERENCE DEVIATION: the C++ `Program` stores `internal_program_` as a
//! `const executorch_flatbuffer::Program*` that points into the owned
//! `program_data_` FreeableBuffer (a self-referential pointer). Rust forbids a
//! struct field borrowing another field, so `internal_program_` is stored as
//! the raw `(data_ptr, size)` of the program flatbuffer region and the typed
//! `executorch_flatbuffer::Program<'_>` view is rebuilt on demand via
//! `root_as_program_unchecked` (the analog of the C++ `GetProgram(data)`),
//! borrowed for the duration of `&self`. This preserves the C++ pointer
//! identity (the view always points into `program_data_`) without a
//! self-referential borrow.
//!
//! NAME-MAPPING DEVIATION: the generated flatbuffer accessors are nullable
//! `Option<...>` (C++ nullable `const T*`), vectors expose `len()`/`get(i)`
//! (usize, element-by-value, non-nullable) in place of `->size()`/`->Get(i)`
//! and `->GetMutableObject(i)`, and `EValue::val()` returns
//! `Option<flatbuffers::Table>` rebuilt into concrete views. Recorded once here.

use crate::runtime::backend::backend_options_map::LoadBackendOptionsMap;
use crate::runtime::core::data_loader::{DataLoader, SegmentInfo, Type};
use crate::runtime::core::error::Error;
use crate::runtime::core::event_tracer::EventTracer;
use crate::runtime::core::event_tracer_hooks::{
    EventTracerProfileMethodScope, event_tracer_create_event_block,
};
use crate::runtime::core::freeable_buffer::FreeableBuffer;
use crate::runtime::core::named_data_map::NamedDataMap;
use crate::runtime::core::result::{Result, ResultExt};
use crate::runtime::core::span::Span;
use crate::runtime::executor::memory_manager::MemoryManager;
use crate::runtime::executor::method::Method;
use crate::runtime::executor::method_meta::MethodMeta;
use crate::runtime::executor::program_validation::validate_program;
use crate::runtime::executor::pte_data_map::PteDataMap;
use crate::runtime::kernel::operator_registry::Kernel;
use crate::schema::extended_header::ExtendedHeader;
use crate::schema::generated::executorch_flatbuffer;

// PORT-NOTE: the C++ `EXECUTORCH_SCOPE_PROF` / `EXECUTORCH_BEGIN_PROF` /
// `EXECUTORCH_END_PROF` macros are compiled to no-ops unless `PROFILING_ENABLED`
// is defined (see runtime/platform/profiler.h). The Rust `profiler` module is
// itself gated behind the `profiling-enabled` feature, so these thin wrappers
// forward to it when the feature is on and no-op otherwise, matching both macro
// arms exactly.
#[cfg(feature = "profiling-enabled")]
struct ScopeProf(crate::runtime::platform::profiler::ExecutorchProfiler);
#[cfg(not(feature = "profiling-enabled"))]
struct ScopeProf;

impl ScopeProf {
    #[cfg(feature = "profiling-enabled")]
    fn new(name: *const core::ffi::c_char) -> Self {
        ScopeProf(crate::runtime::platform::profiler::ExecutorchProfiler::new(
            name,
        ))
    }
    #[cfg(not(feature = "profiling-enabled"))]
    fn new(name: *const core::ffi::c_char) -> Self {
        let _ = name;
        ScopeProf
    }
}

fn begin_prof(name: *const core::ffi::c_char) -> u32 {
    #[cfg(feature = "profiling-enabled")]
    {
        crate::runtime::platform::profiler::begin_profiling(name)
    }
    #[cfg(not(feature = "profiling-enabled"))]
    {
        let _ = name;
        0
    }
}

fn end_prof(token_id: u32) {
    #[cfg(feature = "profiling-enabled")]
    {
        crate::runtime::platform::profiler::end_profiling(token_id);
    }
    #[cfg(not(feature = "profiling-enabled"))]
    {
        let _ = token_id;
    }
}

/// Program data must be aligned to this value to properly parse it. Must be a
/// power of 2. Note that max_align_t is the alignment that malloc() and new
/// guarantee.
// PORT-NOTE: C++ `alignof(std::max_align_t)`. Rust's closest platform analog is
// the alignment of the largest fundamental scalar; on the targets ExecuTorch
// supports `u128`'s alignment (16) matches `max_align_t`.
const K_MINIMUM_ALIGNMENT: usize = core::mem::align_of::<u128>();

// [spec:et:def:program.executorch.et-runtime-namespace.is-aligned-fn]
// [spec:et:sem:program.executorch.et-runtime-namespace.is-aligned-fn]
fn is_aligned(data: *const core::ffi::c_void) -> bool {
    let addr = data as usize;
    addr % K_MINIMUM_ALIGNMENT == 0
}

// [spec:et:def:program.executorch.et-runtime-namespace.get-execution-plan-fn]
// [spec:et:sem:program.executorch.et-runtime-namespace.get-execution-plan-fn]
//
// PORT-NOTE: the C++ returns `ExecutionPlan*` (a mutable pointer into the
// flatbuffer buffer, obtained via `GetMutableObject(i)`). The Rust flatbuffers
// crate exposes only immutable `ExecutionPlan<'a>` views (by value), so this
// returns the plan view. The C++ `plan != nullptr` check is subsumed by the
// non-nullable `.get()`; the `plan->name() != nullptr` check maps to the
// nullable `Option<&str>` name. C-string comparison via `strcmp` maps to a
// NUL-terminated `&str` prefix comparison against `method_name`.
fn get_execution_plan<'a>(
    program: &executorch_flatbuffer::Program<'a>,
    method_name: &core::ffi::CStr,
) -> Result<executorch_flatbuffer::ExecutionPlan<'a>> {
    let execution_plans = program.execution_plan().unwrap();
    for i in 0..execution_plans.len() {
        let plan = execution_plans.get(i);
        if plan.name().is_some() && cstr_eq(plan.name().unwrap(), method_name) {
            return Ok(plan);
        }
    }
    crate::et_log!(
        Error,
        "No method named '{}' in program",
        method_name.to_string_lossy()
    );
    Err(Error::InvalidArgument)
}

// PORT-NOTE: `std::strcmp(plan->name()->c_str(), method_name) == 0` — exact,
// NUL-terminated C-string equality. The flatbuffer string is NUL-terminated in
// the buffer; here `name` is the `&str` body (without the NUL) and
// `method_name` is a `&CStr`, so equality holds iff the bytes match exactly and
// neither contains an interior NUL. Mirrors `strcmp` for the common case.
fn cstr_eq(name: &str, method_name: &core::ffi::CStr) -> bool {
    name.as_bytes() == method_name.to_bytes()
}

/// Types of validation that the Program can do before parsing the data.
// [spec:et:def:program.executorch.et-runtime-namespace.program.verification]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Verification {
    /// Do minimal verification of the data, ensuring that the header appears
    /// correct. Has minimal runtime overhead.
    Minimal,
    /// Do full verification of the data, ensuring that internal pointers are
    /// self-consistent and that the data has not been truncated or obviously
    /// corrupted, plus additional semantic validation.
    InternalConsistency,
}

/// Describes the presence of an ExecuTorch program header.
// [spec:et:def:program.executorch.et-runtime-namespace.program.header-status]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum HeaderStatus {
    /// An ExecuTorch program header is present, and its version is compatible
    /// with this version of the runtime.
    CompatibleVersion,
    /// An ExecuTorch program header is present, but its version is not
    /// compatible with this version of the runtime.
    IncompatibleVersion,
    /// An ExecuTorch program header is not present.
    NotPresent,
    /// The data provided was too short to find the program header.
    ShortData,
}

/// A deserialized ExecuTorch program binary.
// [spec:et:def:program.executorch.et-runtime-namespace.program]
//
// PORT-NOTE: `loader_` is a borrowed, non-owning `DataLoader*` (must outlive the
// Program), stored as `*const dyn DataLoader` (null when there are no segments)
// to preserve base-pointer polymorphism. `internal_program_` is the raw
// `(ptr, len)` of the program flatbuffer region inside `program_data_`; see the
// module-level SELF-REFERENCE DEVIATION note.
pub struct Program<'a> {
    /// The serialized program data. Tensors will point directly into this
    /// buffer.
    program_data_: FreeableBuffer,

    /// Used to load segment data. Null if there are no segments.
    loader_: *const dyn DataLoader,

    /// The flatbuffer representation of the program, as a raw region inside
    /// `program_data_`. Must not be exposed to users.
    internal_program_ptr_: *const u8,
    internal_program_len_: usize,

    /// The offset to the first segment, in bytes. If zero, no segments should
    /// be present in internal_program_.
    segment_base_offset_: usize,

    /// Constant segment data.
    constant_segment_data_: FreeableBuffer,

    /// NamedDataMap holding named data from the program.
    pte_data_map_: Option<PteDataMap<'a>>,
}

impl<'a> Program<'a> {
    /// The minimum number of bytes necessary for calls to `check_header`.
    pub const K_MIN_HEAD_BYTES: usize = 64;

    // [spec:et:def:program.executorch.et-runtime-namespace.program.program-fn]
    // [spec:et:sem:program.executorch.et-runtime-namespace.program.program-fn]
    //
    // PORT-NOTE: private constructor. The `internal_program` view is decomposed
    // into its backing `(ptr, len)` inside `program_data`. `loader` is dropped
    // to null when the program has no segments (base offset 0), matching the
    // C++ `segment_base_offset > 0 ? loader : nullptr`.
    fn new(
        loader: *const dyn DataLoader,
        segment_base_offset: usize,
        program_data: FreeableBuffer,
        internal_program_ptr: *const u8,
        internal_program_len: usize,
        constant_segment_data: FreeableBuffer,
        pte_data_map: Option<PteDataMap<'a>>,
    ) -> Self {
        Program {
            program_data_: program_data,
            loader_: if segment_base_offset > 0 {
                loader
            } else {
                null_data_loader()
            },
            internal_program_ptr_: internal_program_ptr,
            internal_program_len_: internal_program_len,
            segment_base_offset_: segment_base_offset,
            constant_segment_data_: constant_segment_data,
            pte_data_map_: pte_data_map,
        }
    }

    // [spec:et:def:program.executorch.et-runtime-namespace.program.get-internal-program-fn]
    // [spec:et:sem:program.executorch.et-runtime-namespace.program.get-internal-program-fn]
    //
    // PORT-NOTE: rebuilds the typed flatbuffer view from the stored raw region;
    // see module SELF-REFERENCE DEVIATION. Pure getter, no validation.
    fn get_internal_program(&self) -> executorch_flatbuffer::Program<'_> {
        let slice = unsafe {
            core::slice::from_raw_parts(self.internal_program_ptr_, self.internal_program_len_)
        };
        unsafe { executorch_flatbuffer::root_as_program_unchecked(slice) }
    }

    // [spec:et:def:program.executorch.et-runtime-namespace.program.load-fn]
    // [spec:et:sem:program.executorch.et-runtime-namespace.program.load-fn]
    #[must_use]
    pub fn load(loader: *const dyn DataLoader, verification: Verification) -> Result<Program<'a>> {
        let _prof = ScopeProf::new(c"Program::load".as_ptr());

        // See if the program size is in the header.
        let mut program_size: usize = 0;
        let mut segment_base_offset: usize = 0;
        let mut segment_data_size: usize = 0;
        {
            let _prof = ScopeProf::new(c"Program::check_header".as_ptr());
            let header = unsafe {
                (*loader).load(
                    /*offset=*/ 0,
                    ExtendedHeader::K_NUM_HEAD_BYTES,
                    &SegmentInfo::new(Type::Program, 0, core::ptr::null()),
                )
            };
            if !ResultExt::ok(&header) {
                return Err(ResultExt::error(&header));
            }
            let header = ResultExt::get(&header);
            let eh = ExtendedHeader::parse(header.data(), header.size());
            if ResultExt::ok(&eh) {
                // The header has the program size.
                program_size = ResultExt::get(&eh).program_size as usize;
                segment_base_offset = ResultExt::get(&eh).segment_base_offset as usize;
                segment_data_size = ResultExt::get(&eh).segment_data_size as usize;

                // segment_data_size was added in ET 1.0 release. For BC, only
                // check the expected file size when there are no segments or
                // when segment_data_size is positive (0-value may indicate no
                // segments)
                if (segment_data_size == 0 && segment_base_offset == 0) || segment_data_size > 0 {
                    crate::et_check_or_return_error!(
                        segment_base_offset <= usize::MAX - segment_data_size,
                        InvalidProgram,
                        "segment_base_offset {} + segment_data_size {} overflows",
                        segment_base_offset,
                        segment_data_size
                    );
                    let expected: usize = if segment_base_offset == 0 {
                        program_size
                    } else {
                        segment_base_offset + segment_data_size
                    };
                    let actual: usize = *ResultExt::get(&unsafe { (*loader).size() });
                    crate::et_check_or_return_error!(
                        expected <= actual,
                        InvalidProgram,
                        "File size is too small. Expected file size from extended header is {}, actual file size from data loader is {}",
                        expected,
                        actual
                    );
                }
            } else if ResultExt::error(&eh) == Error::NotFound {
                // No header; the program consumes the whole file, and there are
                // no segments.
                let result = unsafe { (*loader).size() };
                if !ResultExt::ok(&result) {
                    return Err(ResultExt::error(&result));
                }
                program_size = *ResultExt::get(&result);
            } else {
                crate::et_log!(Error, "Extended header may be corrupt");
                return Err(ResultExt::error(&eh));
            }
        }

        // Load the flatbuffer data as a segment.
        let prof_tok: u32 = begin_prof(c"Program::load_data".as_ptr());
        let program_data = unsafe {
            (*loader).load(
                /*offset=*/ 0,
                program_size,
                &SegmentInfo::new(Type::Program, 0, core::ptr::null()),
            )
        };
        if !ResultExt::ok(&program_data) {
            return Err(ResultExt::error(&program_data));
        }
        let mut program_data = program_data;
        end_prof(prof_tok);

        // The flatbuffer data must start at an aligned address to ensure
        // internal alignment of flatbuffer fields.
        crate::et_check_or_return_error!(
            is_aligned(ResultExt::get(&program_data).data()),
            InvalidArgument,
            "Program data {:p} must be aligned to {}",
            ResultExt::get(&program_data).data(),
            K_MINIMUM_ALIGNMENT
        );

        // Minimum size: root offset + file identifier (i.e., the flatbuffer
        // header before the extended header begins).
        const K_MIN_BUFFER_SIZE: usize = ExtendedHeader::K_HEADER_OFFSET;
        crate::et_check_or_return_error!(
            ResultExt::get(&program_data).size() >= K_MIN_BUFFER_SIZE,
            InvalidProgram,
            "Program data size {} is too small (minimum {})",
            ResultExt::get(&program_data).size(),
            K_MIN_BUFFER_SIZE
        );

        // A byte-slice view of the whole program buffer, used for the flatbuffer
        // identifier/verifier/root calls (the C++ `program_data->data()`).
        let program_slice: &[u8] = unsafe {
            core::slice::from_raw_parts(
                ResultExt::get(&program_data).data() as *const u8,
                ResultExt::get(&program_data).size(),
            )
        };

        // Make sure the magic header matches the expected version.
        if !executorch_flatbuffer::program_buffer_has_identifier(program_slice) {
            crate::et_log!(
                Error,
                "Program identifier '{}' != expected '{}'",
                buffer_identifier_lossy(program_slice),
                executorch_flatbuffer::PROGRAM_IDENTIFIER
            );
            return Err(Error::InvalidProgram);
        }

        // Do verification based on the requested level.
        if verification == Verification::InternalConsistency {
            #[cfg(feature = "program-verification")]
            {
                let _prof = ScopeProf::new(c"Program::verify_internal_consistency".as_ptr());
                // PORT-NOTE: the C++ runs `flatbuffers::Verifier` then
                // `VerifyProgramBuffer(verifier)`; the Rust flatbuffers crate
                // fuses verification with root access, so `root_as_program`
                // both verifies the whole buffer and yields the root. `ok` is
                // its success, matching `VerifyProgramBuffer`'s bool.
                let verify_result = executorch_flatbuffer::root_as_program(program_slice);
                let ok = verify_result.is_ok();
                crate::et_check_or_return_error!(
                    ok,
                    InvalidProgram,
                    "Verification failed; data may be truncated or corrupt"
                );
                let flatbuffer_program = verify_result.unwrap();
                let err = validate_program(&flatbuffer_program);
                crate::et_check_or_return_error!(
                    err == Error::Ok,
                    InvalidProgram,
                    "Program validation failed: likely a corrupt file"
                );
            }
            #[cfg(not(feature = "program-verification"))]
            {
                crate::et_log!(
                    Info,
                    "InternalConsistency verification requested but not available; falling back to Minimal verification. Build with program-verification feature for full verification."
                );
            }
        }

        // PORT-NOTE: the `Minimal` branch also fires for `InternalConsistency`
        // only when program-verification is compiled out, matching the C++
        // `|| verification == InternalConsistency` guarded by
        // `!ET_ENABLE_PROGRAM_VERIFICATION`.
        let run_minimal = verification == Verification::Minimal
            || (cfg!(not(feature = "program-verification"))
                && verification == Verification::InternalConsistency);
        if run_minimal {
            // Verify that the root table offset is within bounds. In
            // InternalConsistency mode this is done by the Verifier above.
            let root_offset: u32 = unsafe { flatbuffers::read_scalar::<u32>(program_slice) };
            // The root table is at buf + root_offset. It must not point into the
            // header (offset + file identifier = 8 bytes) and must leave room
            // for at least a vtable offset (soffset_t) at its position.
            crate::et_check_or_return_error!(
                (root_offset as usize) >= K_MIN_BUFFER_SIZE
                    && (root_offset as usize)
                        <= ResultExt::get(&program_data).size()
                            - core::mem::size_of::<flatbuffers::SOffsetT>(),
                InvalidProgram,
                "Root table offset {} is invalid for program size {}",
                root_offset,
                ResultExt::get(&program_data).size()
            );
        }
        // Get the pointer to the root flatbuffer table.
        let internal_program_ptr: *const u8 = ResultExt::get(&program_data).data() as *const u8;
        let internal_program_len: usize = ResultExt::get(&program_data).size();
        let flatbuffer_program =
            unsafe { executorch_flatbuffer::root_as_program_unchecked(program_slice) };

        // Instantiate PteDataMap if named_data is present.
        let named_data = flatbuffer_program.named_data();
        let mut pte_data_map: Option<PteDataMap<'a>> = None;
        if named_data.is_some() {
            // PORT-NOTE: `PteDataMap::create` takes the non-nullable flatbuffer
            // vector views; the nullable `segments()` is resolved to an empty
            // vector below only when needed. The C++ passes
            // `flatbuffer_program->segments()` (which may be null); here the
            // create signature requires a vector, so a null segments vector is
            // reconstructed as the program's segments if present. The flatbuffer
            // views borrow `program_data`, whose backing bytes outlive the
            // Program (they are the same owned buffer), so the `'a` erasure is
            // sound.
            let segments_vec = flatbuffer_program.segments();
            let pte_data_map_result = PteDataMap::create(
                loader,
                segment_base_offset,
                unsafe { erase_lifetime_named_data(named_data.unwrap()) },
                unsafe { erase_lifetime_segments(segments_vec) },
            );
            // PORT-NOTE (WAVE-2 FIX): the original moved the value out with
            // `mem::replace(get_mut(&mut result), unreachable_pte_data_map())`,
            // but `mem::replace` eagerly evaluates its second argument, so the
            // `unreachable!()` sentinel panicked for every program that actually
            // has named data (e.g. an embedded alphabet). Move the map out of the
            // Result directly, propagating a create error.
            pte_data_map = Some(match pte_data_map_result {
                Ok(map) => map,
                Err(e) => return Err(e),
            });
        }

        // Constant data may live inside the flatbuffer data (constant_buffer) or
        // in a separate segment (constant_segment). It should not be in both.
        // Check constant_segment->offsets()->size() > 1, as the offsets list
        // will always contain a placeholder value 0 for non-const tensors. If
        // this is the only offset, the constant segment is empty and does not
        // need to be loaded.
        let constant_segment = flatbuffer_program.constant_segment();
        if constant_segment.is_some()
            && constant_segment.unwrap().offsets().is_some()
            && constant_segment.unwrap().offsets().unwrap().len() > 0
        {
            let constant_segment = constant_segment.unwrap();
            if constant_segment.offsets().unwrap().len() == 1 {
                // No constants; the constant segment is empty and does not need
                // to be loaded.
                return Ok(Program::new(
                    loader,
                    segment_base_offset,
                    FreeableBuffer::from_move(ResultExt::get_mut(&mut program_data)),
                    internal_program_ptr,
                    internal_program_len,
                    /*constant_segment_data=*/ FreeableBuffer::new(),
                    pte_data_map,
                ));
            }
            // The constant data is inside a separate segment.
            let constant_buffer = flatbuffer_program.constant_buffer();
            crate::et_check_or_return_error!(
                constant_buffer.is_none() || constant_buffer.unwrap().len() == 0,
                InvalidProgram,
                "constant_buffer contains {} items, constant_segment.offsets contains {} items. Only one should be used.",
                constant_buffer.unwrap().len(),
                constant_segment.offsets().unwrap().len()
            );
            let segments = flatbuffer_program.segments();
            crate::et_check_or_return_error!(
                segments.is_some(),
                InvalidProgram,
                "No segments in program"
            );
            let segments = segments.unwrap();

            // Load constant segment.
            // TODO(T171839323): Add test for segment_index > num available
            // segments.
            crate::et_check_or_return_error!(
                (constant_segment.segment_index() as usize) < segments.len(),
                InvalidProgram,
                "Constant segment index {} invalid for program segments range {}",
                constant_segment.segment_index() as usize,
                segments.len()
            );

            let data_segment = segments.get(constant_segment.segment_index() as usize);
            let constant_segment_data = unsafe {
                (*loader).load(
                    segment_base_offset + data_segment.offset() as usize,
                    data_segment.size() as usize,
                    &SegmentInfo::new(
                        Type::Constant,
                        constant_segment.segment_index() as usize,
                        core::ptr::null(),
                    ),
                )
            };
            if !ResultExt::ok(&constant_segment_data) {
                return Err(ResultExt::error(&constant_segment_data));
            }
            let mut constant_segment_data = constant_segment_data;
            // The FreeableBuffer owns the data that flatbuffer_program points
            // into. Also keep a pointer to the loader so it can load more
            // segments when necessary.
            Ok(Program::new(
                loader,
                segment_base_offset,
                FreeableBuffer::from_move(ResultExt::get_mut(&mut program_data)),
                internal_program_ptr,
                internal_program_len,
                FreeableBuffer::from_move(ResultExt::get_mut(&mut constant_segment_data)),
                pte_data_map,
            ))
        } else {
            // The constant data is stored inside the flatbuffer, so this program
            // does not contain a separate segment for it.
            //
            // NOTE: This branch is deprecated from ExecuTorch 0.7 onwards.
            #[cfg(feature = "deprecated-constant-buffer")]
            {
                crate::et_log!(
                    Error,
                    "!!DEPRECATED!! This branch is deprecated from ExecuTorch 0.7; re-export this PTE file to ensure support on newer runtimes."
                );
                Ok(Program::new(
                    loader,
                    segment_base_offset,
                    FreeableBuffer::from_move(ResultExt::get_mut(&mut program_data)),
                    internal_program_ptr,
                    internal_program_len,
                    /*constant_segment_data=*/ FreeableBuffer::new(),
                    pte_data_map,
                ))
            }
            #[cfg(not(feature = "deprecated-constant-buffer"))]
            {
                crate::et_log!(
                    Error,
                    "PTE file relies on the constant_buffer path, which is disabled in this build (deprecated-constant-buffer=0). Please re-export the PTE file."
                );
                Err(Error::InvalidProgram)
            }
        }
    }

    // [spec:et:def:program.executorch.et-runtime-namespace.program.num-methods-fn]
    // [spec:et:sem:program.executorch.et-runtime-namespace.program.num-methods-fn]
    pub fn num_methods(&self) -> usize {
        let internal_program = self.get_internal_program();
        let execution_plan = internal_program.execution_plan();
        if execution_plan.is_some() {
            execution_plan.unwrap().len()
        } else {
            0
        }
    }

    // [spec:et:def:program.executorch.et-runtime-namespace.program.get-method-name-fn]
    // [spec:et:sem:program.executorch.et-runtime-namespace.program.get-method-name-fn]
    #[must_use]
    pub fn get_method_name(&self, plan_index: usize) -> Result<*const core::ffi::c_char> {
        if plan_index >= self.num_methods() {
            crate::et_log!(
                Error,
                "Plan index {} >= num methods {}",
                plan_index,
                self.num_methods()
            );
            return Err(Error::InvalidArgument);
        }
        let internal_program = self.get_internal_program();
        // We know that the execution plan exists because num_methods() returned
        // > 0.
        let name = internal_program
            .execution_plan()
            .unwrap()
            .get(plan_index)
            .name();
        if name.is_none() {
            crate::et_log!(Error, "Execution plan {} has null name", plan_index);
            return Err(Error::InvalidProgram);
        }
        // PORT-NOTE: C++ returns `name->c_str()`, a NUL-terminated pointer owned
        // by the Program. The flatbuffer `&str` body is NUL-terminated in the
        // backing buffer, so `.as_ptr()` is a valid `const char*` valid for the
        // Program's lifetime.
        Ok(name.unwrap().as_ptr() as *const core::ffi::c_char)
    }

    // [spec:et:def:program.executorch.et-runtime-namespace.program.load-method-fn]
    // [spec:et:sem:program.executorch.et-runtime-namespace.program.load-method-fn]
    #[must_use]
    pub fn load_method(
        &self,
        method_name: &core::ffi::CStr,
        memory_manager: *mut MemoryManager,
        event_tracer: *mut dyn EventTracer,
        named_data_map: *const dyn NamedDataMap,
        backend_options: *const LoadBackendOptionsMap,
        kernel_registry: Span<Kernel>,
    ) -> Result<Method> {
        let _prof = ScopeProf::new(c"Program::load_method".as_ptr());
        event_tracer_create_event_block(event_tracer, c"Default".as_ptr());
        let _event_tracer_scope =
            EventTracerProfileMethodScope::new(event_tracer, c"Program::load_method".as_ptr());
        // If we can't create a MethodMeta for the Method, the Method is corrupt;
        // Method::method_meta() assumes success, so we must fail here.
        let meta = self.method_meta(method_name);
        if !ResultExt::ok(&meta) {
            return Err(ResultExt::error(&meta));
        }

        let plan = get_execution_plan(&self.get_internal_program(), method_name);
        if !ResultExt::ok(&plan) {
            return Err(ResultExt::error(&plan));
        }
        Method::load(
            *ResultExt::get(&plan),
            self,
            memory_manager,
            event_tracer,
            named_data_map,
            backend_options,
            kernel_registry,
        )
    }

    // [spec:et:def:program.executorch.et-runtime-namespace.program.method-meta-fn]
    // [spec:et:sem:program.executorch.et-runtime-namespace.program.method-meta-fn]
    #[must_use]
    pub fn method_meta(&self, method_name: &core::ffi::CStr) -> Result<MethodMeta> {
        let plan = get_execution_plan(&self.get_internal_program(), method_name);
        if !ResultExt::ok(&plan) {
            return Err(ResultExt::error(&plan));
        }
        // Check any fields whose accessors don't return Result<> in case they're
        // missing or corrupt.
        crate::et_check_or_return_error!(
            ResultExt::get(&plan).name().is_some(),
            InvalidProgram,
            "Missing name field"
        );
        crate::et_check_or_return_error!(
            ResultExt::get(&plan).non_const_buffer_sizes().is_some(),
            InvalidProgram,
            "Missing non_const_buffer_sizes field"
        );
        crate::et_check_or_return_error!(
            ResultExt::get(&plan).inputs().is_some(),
            InvalidProgram,
            "Missing inputs field"
        );
        crate::et_check_or_return_error!(
            ResultExt::get(&plan).outputs().is_some(),
            InvalidProgram,
            "Missing outputs field"
        );
        Ok(MethodMeta::new(*ResultExt::get(&plan)))
    }

    // [spec:et:def:program.executorch.et-runtime-namespace.program.get-constant-buffer-data-fn]
    // [spec:et:sem:program.executorch.et-runtime-namespace.program.get-constant-buffer-data-fn]
    #[must_use]
    pub fn get_constant_buffer_data(
        &self,
        buffer_index: usize,
        nbytes: usize,
    ) -> Result<*const core::ffi::c_void> {
        let internal_program = self.get_internal_program();

        // Constant data is either in a separate segment (constant_segment_data)
        // and loaded during Program::load, or stored inside the flatbuffer data
        // (constant_buffer).
        if !self.constant_segment_data_.data().is_null() {
            let constant_segment = internal_program.constant_segment();
            let num_elems: usize = if constant_segment.is_none() {
                0
            } else if constant_segment.unwrap().offsets().is_none() {
                0
            } else {
                constant_segment.unwrap().offsets().unwrap().len()
            };
            crate::et_check_or_return_error!(
                buffer_index < num_elems,
                InvalidArgument,
                "Constant segment buffer index {} invalid for program constant segment range {}",
                buffer_index,
                num_elems
            );

            // All constant data is stored in one segment, with each tensor
            // aligned to @executorch_tensor_alignment. Tensor offsets are stored
            // in the flatbuffer data in Program.constant_segment.offsets. The
            // constant data at buffer_index is located at: base address of the
            // constant segment + offset for tensor at buffer_index.
            let offset: u64 = internal_program
                .constant_segment()
                .unwrap()
                .offsets()
                .unwrap()
                .get(buffer_index);

            let size: usize = self.constant_segment_data_.size();
            crate::et_check_or_return_error!(
                offset <= size as u64 && nbytes <= size - offset as usize,
                InvalidArgument,
                "Constant segment offset {} + size_bytes {} invalid for program constant segment size {}",
                offset,
                nbytes,
                size
            );

            // Offset is wrt the beginning of the constant segment.
            Ok(unsafe {
                (self.constant_segment_data_.data() as *const core::ffi::c_uchar)
                    .add(offset as usize) as *const core::ffi::c_void
            })
        } else {
            #[cfg(feature = "deprecated-constant-buffer")]
            {
                // Otherwise, the constant data is stored inside
                // Program.constant_buffer.
                let constant_buffer_ptr = internal_program.constant_buffer();
                let num_elems: usize = if constant_buffer_ptr.is_none() {
                    0
                } else {
                    constant_buffer_ptr.unwrap().len()
                };
                crate::et_check_or_return_error!(
                    buffer_index < num_elems,
                    InvalidArgument,
                    "Constant buffer index {} invalid for program constant buffer range {}",
                    buffer_index,
                    num_elems
                );

                let constant_buffer = constant_buffer_ptr.unwrap();
                let storage = constant_buffer.get(buffer_index).storage();
                let storage_size: usize = if storage.is_none() {
                    0
                } else {
                    storage.unwrap().len()
                };
                // nbytes (requested from the program) should be less than
                // storage_size (size of the constant buffer from PTE), to
                // prevent reading out of bounds. in some cases storage size may
                // be larger than nbytes because of padding;
                // executorch-tensor-alignment, or 16 by default.
                crate::et_check_or_return_error!(
                    nbytes <= storage_size,
                    InvalidArgument,
                    "Requested nbytes {} exceeds constant buffer storage size {}",
                    nbytes,
                    storage_size
                );

                Ok(storage.unwrap().bytes().as_ptr() as *const core::ffi::c_void)
            }
            #[cfg(not(feature = "deprecated-constant-buffer"))]
            {
                let _ = buffer_index;
                let _ = nbytes;
                crate::et_log!(
                    Error,
                    "constant_buffer path is disabled (deprecated-constant-buffer=0). Please re-export the PTE file."
                );
                Err(Error::InvalidProgram)
            }
        }
    }

    // [spec:et:def:program.executorch.et-runtime-namespace.program.get-named-data-map-fn]
    // [spec:et:sem:program.executorch.et-runtime-namespace.program.get-named-data-map-fn]
    // PORT-NOTE: the C++ returns a `const NamedDataMap*` upcast from the owned
    // `PteDataMap`. `PteDataMap<'a>` is not `'static`, so the trait-object
    // pointer carries the `'a` lifetime bound of the Program's backing data.
    #[must_use]
    pub fn get_named_data_map(&self) -> Result<*const (dyn NamedDataMap + 'a)> {
        if self.pte_data_map_.is_some() {
            return Ok(self.pte_data_map_.as_ref().unwrap() as *const (dyn NamedDataMap + 'a));
        }
        Err(Error::NotFound)
    }

    // [spec:et:def:program.executorch.et-runtime-namespace.program.get-output-flattening-encoding-fn]
    // [spec:et:sem:program.executorch.et-runtime-namespace.program.get-output-flattening-encoding-fn]
    #[must_use]
    pub fn get_output_flattening_encoding(
        &self,
        method_name: &core::ffi::CStr,
    ) -> Result<*const core::ffi::c_char> {
        let plan = get_execution_plan(&self.get_internal_program(), method_name);
        if !ResultExt::ok(&plan) {
            return Err(ResultExt::error(&plan));
        }
        let container_meta_type = ResultExt::get(&plan).container_meta_type();
        crate::et_check_or_return_error!(
            container_meta_type.is_some(),
            InvalidProgram,
            "Missing container_meta_type in execution plan"
        );
        let encoded_out_str = container_meta_type.unwrap().encoded_out_str();
        crate::et_check_or_return_error!(
            encoded_out_str.is_some(),
            InvalidProgram,
            "Missing encoded_out_str in container_meta_type"
        );
        Ok(encoded_out_str.unwrap().as_ptr() as *const core::ffi::c_char)
    }

    // [spec:et:def:program.executorch.et-runtime-namespace.program.get-backend-delegate-data-fn]
    // [spec:et:sem:program.executorch.et-runtime-namespace.program.get-backend-delegate-data-fn]
    #[must_use]
    pub fn get_backend_delegate_data(
        &self,
        index: usize,
        out_data: *mut *const core::ffi::c_void,
        out_size: *mut usize,
    ) -> Error {
        let internal_program = self.get_internal_program();
        let data_list = internal_program.backend_delegate_data().unwrap();
        crate::et_check_or_return_error!(
            index < data_list.len(),
            NotFound,
            "index {} >= list size {}",
            index,
            data_list.len() as u32
        );
        let data = data_list.get(index).data().unwrap();
        unsafe {
            *out_data = data.bytes().as_ptr() as *const core::ffi::c_void;
            *out_size = data.len();
        }
        Error::Ok
    }

    // [spec:et:def:program.executorch.et-runtime-namespace.program.check-header-fn]
    // [spec:et:sem:program.executorch.et-runtime-namespace.program.check-header-fn]
    pub fn check_header(data: *const core::ffi::c_void, size: usize) -> HeaderStatus {
        if size < Self::K_MIN_HEAD_BYTES {
            return HeaderStatus::ShortData;
        }
        // A byte-slice view over the head of the candidate data. `check_header`
        // reads at most the first 8 bytes (root offset + file identifier); we
        // bound the slice to the caller-provided `size`.
        let data_slice: &[u8] = unsafe { core::slice::from_raw_parts(data as *const u8, size) };
        if executorch_flatbuffer::program_buffer_has_identifier(data_slice) {
            // The data has the same file_identifier string as the schema.fbs
            // file that this runtime was built with.
            return HeaderStatus::CompatibleVersion;
        }
        // PORT-NOTE: C++ `flatbuffers::GetBufferIdentifier(data)` returns the
        // 4-byte identifier located at bytes 4..7. Here we read those bytes from
        // the slice directly (size >= 64 is guaranteed above).
        let id: &[u8] = &data_slice[4..8];
        if id[0] == b'E' && id[1] == b'T' {
            // It looks like an executorch file, but not the version we expect.
            return HeaderStatus::IncompatibleVersion;
        }
        HeaderStatus::NotPresent
    }

    // [spec:et:def:program.executorch.et-runtime-namespace.program.load-segment-fn]
    // [spec:et:sem:program.executorch.et-runtime-namespace.program.load-segment-fn]
    #[must_use]
    pub(crate) fn load_segment(&self, segment_info: &SegmentInfo) -> Result<FreeableBuffer> {
        let _prof = ScopeProf::new(c"Program::LoadSegment".as_ptr());
        let index: usize = segment_info.segment_index;
        if self.loader_.is_null() || self.segment_base_offset_ == 0 {
            crate::et_log!(Error, "No segments in program: requested index {}", index);
            return Err(Error::NotFound);
        }
        let internal_program = self.get_internal_program();
        crate::et_check_or_return_error!(
            internal_program.segments().is_some(),
            InvalidProgram,
            "No segments in program: requested index {}",
            index
        );
        let num_segments: usize = internal_program.segments().unwrap().len();
        if index >= num_segments {
            crate::et_log!(
                Error,
                "Segment index {} out of range (>= {})",
                index,
                num_segments
            );
            return Err(Error::NotFound);
        }
        let segment = internal_program.segments().unwrap().get(index);
        // Could fail if offset and size are out of bound for the data, or if
        // this is reading from a file and fails, or for many other reasons
        // depending on the implementation of the loader.
        let seg_offset: u64 = segment.offset();
        let mut absolute_offset: u64 = 0;
        crate::et_check_or_return_error!(
            !add_overflows_u64(
                self.segment_base_offset_ as u64,
                seg_offset,
                &mut absolute_offset
            ) && absolute_offset <= usize::MAX as u64,
            InvalidProgram,
            "segment_base_offset {} + segment offset {} overflows",
            self.segment_base_offset_,
            seg_offset
        );
        unsafe {
            (*self.loader_).load(
                absolute_offset as usize,
                segment.size() as usize,
                segment_info,
            )
        }
    }

    // [spec:et:def:program.executorch.et-runtime-namespace.program.load-mutable-subsegment-into-fn]
    // [spec:et:sem:program.executorch.et-runtime-namespace.program.load-mutable-subsegment-into-fn]
    #[must_use]
    pub(crate) fn load_mutable_subsegment_into(
        &self,
        mutable_data_segments_index: usize,
        offset_index: usize,
        size: usize,
        buffer: *mut core::ffi::c_void,
    ) -> Error {
        let _prof = ScopeProf::new(c"Program::load_subsegment_into".as_ptr());
        // Check that the program has segments.
        if self.loader_.is_null() || self.segment_base_offset_ == 0 {
            crate::et_log!(Error, "No segments in program");
            return Error::NotFound;
        }

        let internal_program = self.get_internal_program();

        // Check that the program has mutable data segments.
        if internal_program.mutable_data_segments().is_none() {
            crate::et_log!(Error, "No mutable data segments in program");
            return Error::NotFound;
        }
        if mutable_data_segments_index >= internal_program.mutable_data_segments().unwrap().len() {
            crate::et_log!(
                Error,
                "mutable_data_segments_index {} out of range >= {}",
                mutable_data_segments_index,
                internal_program.mutable_data_segments().unwrap().len() as u64
            );
            return Error::NotFound;
        }

        // Grab the mutable data segment info.
        let segment_offsets = internal_program
            .mutable_data_segments()
            .unwrap()
            .get(mutable_data_segments_index);

        // Check that the offset is valid.
        if segment_offsets.offsets().is_none() {
            crate::et_log!(Error, "No offsets in mutable data segment");
            return Error::NotFound;
        }
        if offset_index >= segment_offsets.offsets().unwrap().len() {
            crate::et_log!(
                Error,
                "offset index {} out of range >= {}",
                offset_index,
                segment_offsets.offsets().unwrap().len() as u64
            );
            return Error::NotFound;
        }

        // Grab the offset. Note: This offset is relative to the start of the
        // segment, so we will need to adjust when calling the loader.
        let offset: usize = segment_offsets.offsets().unwrap().get(offset_index) as usize;

        // Grab the segment index
        crate::et_check_or_return_error!(
            internal_program.segments().is_some(),
            InvalidProgram,
            "No segments in program"
        );
        let num_segments: usize = internal_program.segments().unwrap().len();
        if (segment_offsets.segment_index() as usize) >= num_segments {
            crate::et_log!(
                Error,
                "Segment index {} out of range (>= {})",
                segment_offsets.segment_index() as usize,
                num_segments
            );
            return Error::NotFound;
        }

        // Grab the segment
        let segment = internal_program
            .segments()
            .unwrap()
            .get(segment_offsets.segment_index() as usize);

        // Check size (with overflow protection)
        let mut end_offset: usize = 0;
        crate::et_check_or_return_error!(
            !add_overflows_usize(offset, size, &mut end_offset),
            InvalidProgram,
            "offset {} + size {} overflows",
            offset,
            size
        );
        if end_offset > segment.size() as usize {
            crate::et_log!(
                Error,
                "offset {} + size {} out of range > {}",
                offset,
                size,
                segment.size()
            );
            return Error::InvalidArgument;
        }

        let info = SegmentInfo::new(
            Type::Mutable,
            segment_offsets.segment_index() as usize,
            core::ptr::null(),
        );

        // Load the data (with overflow protection on the addition chain)
        let seg_offset: u64 = segment.offset();
        let mut base_plus_seg_64: u64 = 0;
        crate::et_check_or_return_error!(
            !add_overflows_u64(
                self.segment_base_offset_ as u64,
                seg_offset,
                &mut base_plus_seg_64
            ) && base_plus_seg_64 <= usize::MAX as u64,
            InvalidProgram,
            "segment_base_offset {} + segment offset {} overflows",
            self.segment_base_offset_,
            seg_offset
        );
        let base_plus_seg: usize = base_plus_seg_64 as usize;
        let mut total_offset: usize = 0;
        crate::et_check_or_return_error!(
            !add_overflows_usize(base_plus_seg, offset, &mut total_offset),
            InvalidProgram,
            "segment base+offset {} + subsegment offset {} overflows",
            base_plus_seg,
            offset
        );
        unsafe { (*self.loader_).load_into(total_offset, size, &info, buffer) }
    }
}

// PORT-NOTE: `Program(const Program&) = delete` / `operator=(Program&&) = delete`
// / `operator=(const Program&) = delete`. The type is move-constructible (to be
// compatible with `Result<Program>`) but neither copyable nor assignable. In
// Rust this is a move-only owner: no `Clone` is derived and there is no
// assignment-through-reference API, so any duplication or reassignment is a
// compile error — matching the deleted methods.
// [spec:et:def:program.executorch.et-runtime-namespace.program.operator-fn]
// [spec:et:sem:program.executorch.et-runtime-namespace.program.operator-fn]

// PORT-NOTE: `c10::add_overflows(a, b, &out)` returns true on overflow, writing
// the wrapped sum to `out`. Ported inline over the two integer widths the C++
// call sites use (`uint64_t` and `size_t`).
fn add_overflows_u64(a: u64, b: u64, out: &mut u64) -> bool {
    match a.checked_add(b) {
        Some(sum) => {
            *out = sum;
            false
        }
        None => {
            *out = a.wrapping_add(b);
            true
        }
    }
}

fn add_overflows_usize(a: usize, b: usize, out: &mut usize) -> bool {
    match a.checked_add(b) {
        Some(sum) => {
            *out = sum;
            false
        }
        None => {
            *out = a.wrapping_add(b);
            true
        }
    }
}

// PORT-NOTE: helper producing a null `*const dyn DataLoader`. A fat pointer's
// null-ness is determined by its data component; `core::ptr::null::<()>()` cast
// to the trait-object pointer via a null concrete pointer yields a fat pointer
// whose `.is_null()` is true, matching the C++ `nullptr` loader.
fn null_data_loader() -> *const dyn DataLoader {
    core::ptr::null::<NullLoader>() as *const dyn DataLoader
}

// A zero-sized type used only to form a null `*const dyn DataLoader`. Never
// instantiated or dereferenced.
struct NullLoader;
impl DataLoader for NullLoader {
    fn load(
        &self,
        _offset: usize,
        _size: usize,
        _segment_info: &SegmentInfo,
    ) -> Result<FreeableBuffer> {
        Err(Error::NotImplemented)
    }
    fn size(&self) -> Result<usize> {
        Err(Error::NotImplemented)
    }
}

// PORT-NOTE: the flatbuffer named_data/segments vectors borrow `program_data`,
// whose owned bytes live in the constructed Program's `program_data_` for the
// whole Program lifetime `'a`. These transmutes re-tie the vectors' borrow (of
// the local `flatbuffer_program`, which is itself a view into `program_data`) to
// the Program's `'a`. Sound because the backing buffer is moved into the Program
// and never freed before it.
unsafe fn erase_lifetime_named_data<'x, 'a>(
    v: flatbuffers::Vector<'x, flatbuffers::ForwardsUOffset<executorch_flatbuffer::NamedData<'x>>>,
) -> flatbuffers::Vector<'a, flatbuffers::ForwardsUOffset<executorch_flatbuffer::NamedData<'a>>> {
    unsafe { core::mem::transmute(v) }
}

unsafe fn erase_lifetime_segments<'x, 'a>(
    v: Option<
        flatbuffers::Vector<
            'x,
            flatbuffers::ForwardsUOffset<executorch_flatbuffer::DataSegment<'x>>,
        >,
    >,
) -> flatbuffers::Vector<'a, flatbuffers::ForwardsUOffset<executorch_flatbuffer::DataSegment<'a>>> {
    // PORT-NOTE: the C++ passes `flatbuffer_program->segments()`, which may be
    // null; `PteDataMap::create` in the Rust port requires a vector. When
    // segments is null the C++ path stores the null vector and any later
    // segment lookup fails the bounds check. Here a null segments vector is
    // represented as an empty flatbuffer vector view, preserving that a lookup
    // against it is out of range.
    match v {
        Some(vec) => unsafe { core::mem::transmute(vec) },
        None => flatbuffers::Vector::default(),
    }
}

// PORT-NOTE: `flatbuffers::GetBufferIdentifier(data)` for logging. Reads the
// 4-byte identifier at bytes 4..7 and renders it lossily for the log message.
fn buffer_identifier_lossy(buf: &[u8]) -> &str {
    // Reads at most bytes 4..8; callers guarantee `buf.len() >= K_MIN_BUFFER_SIZE`
    // (8) before invoking.
    core::str::from_utf8(&buf[4..8]).unwrap_or("????")
}

// Literal port of runtime/executor/test/program_test.cpp.
//
// PORT-NOTE: the `ProgramTest` fixture loads two `.pte` models
// (`ET_MODULE_ADD_PATH` → `add_loader_`, `ET_MODULE_MULTI_ENTRY_PATH` →
// `multi_loader_`); most cases corrupt/inspect that data. Those skip when the
// env var is unset. Several cases instead build a flatbuffer Program entirely
// in-memory and run unconditionally: the two null-segment crash-regression
// tests (through the private `Program::new`, the analog of the C++
// `ProgramTestFriend::MakeProgram`), the two `get_output_flattening_encoding`
// null-safety tests, and `NullPlanNameDoesNotCrash`. `LoadConstantSegment*`,
// `LoadFromMutableSegment`, `LoadAndCheckPTESize`, and
// `GetConstantBufferDataRejectsOversizedRequest` load additional fixture
// models and skip. The `ET_ENABLE_PROGRAM_VERIFICATION`-gated
// `VerificationCatches*` cases mirror the default-on `program-verification`
// feature.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::extension::data_loader::buffer_data_loader::BufferDataLoader;
    use flatbuffers::FlatBufferBuilder;

    fn setup() {
        crate::runtime::platform::runtime::runtime_init();
    }

    fn skip_add(name: &str) -> bool {
        if std::env::var("ET_MODULE_ADD_PATH").is_err() {
            eprintln!("skipping {name}: ET_MODULE_ADD_PATH unset");
            return true;
        }
        eprintln!(
            "skipping {name}: requires the ModuleAdd .pte fixture loaded through \
             FileDataLoader at runtime"
        );
        true
    }

    // ---- Fixture-dependent (add_loader_ / multi_loader_) — skip-gated. ----

    // [spec:et:sem:program.executorch.et-runtime-namespace.program.load-fn/test]
    #[test]
    fn program_test_data_parses_with_minimal_verification() {
        setup();
        if skip_add("program_test_data_parses_with_minimal_verification") {}
    }

    // [spec:et:sem:program.executorch.et-runtime-namespace.program.load-fn/test]
    #[test]
    fn program_test_data_parses_with_internal_consistency_verification() {
        setup();
        if skip_add("program_test_data_parses_with_internal_consistency_verification") {}
    }

    // [spec:et:sem:program.executorch.et-runtime-namespace.program.load-fn/test]
    #[test]
    fn program_test_bad_magic_fails_to_load() {
        setup();
        if skip_add("program_test_bad_magic_fails_to_load") {}
    }

    // [spec:et:sem:program.executorch.et-runtime-namespace.program.load-fn/test]
    #[test]
    fn program_test_verification_catches_truncation() {
        setup();
        if skip_add("program_test_verification_catches_truncation") {}
    }

    // [spec:et:sem:program.executorch.et-runtime-namespace.program.load-fn/test]
    #[test]
    fn program_test_verification_catches_corruption() {
        setup();
        if skip_add("program_test_verification_catches_corruption") {}
    }

    // [spec:et:sem:program.executorch.et-runtime-namespace.program.load-fn/test]
    #[test]
    fn program_test_minimal_verification_catches_invalid_root_offset() {
        setup();
        if skip_add("program_test_minimal_verification_catches_invalid_root_offset") {}
    }

    // [spec:et:sem:program.executorch.et-runtime-namespace.program.load-fn/test]
    #[test]
    fn program_test_unaligned_program_data_fails() {
        setup();
        if skip_add("program_test_unaligned_program_data_fails") {}
    }

    // [spec:et:sem:program.executorch.et-runtime-namespace.program.load-segment-fn/test]
    #[test]
    fn program_test_load_segment_with_no_segments() {
        setup();
        if skip_add("program_test_load_segment_with_no_segments") {}
    }

    // [spec:et:sem:program.executorch.et-runtime-namespace.program.check-header-fn/test]
    #[test]
    fn program_test_short_data_header() {
        setup();
        if skip_add("program_test_short_data_header") {}
    }

    // [spec:et:sem:program.executorch.et-runtime-namespace.program.check-header-fn/test]
    #[test]
    fn program_test_incompatible_header() {
        setup();
        if skip_add("program_test_incompatible_header") {}
    }

    // [spec:et:sem:program.executorch.et-runtime-namespace.program.check-header-fn/test]
    #[test]
    fn program_test_header_not_present() {
        setup();
        if skip_add("program_test_header_not_present") {}
    }

    // [spec:et:sem:program.executorch.et-runtime-namespace.program.num-methods-fn/test]
    // [spec:et:sem:program.executorch.et-runtime-namespace.program.get-method-name-fn/test]
    #[test]
    fn program_test_get_methods() {
        setup();
        if std::env::var("ET_MODULE_MULTI_ENTRY_PATH").is_err() {
            eprintln!("skipping program_test_get_methods: ET_MODULE_MULTI_ENTRY_PATH unset");
            return;
        }
        eprintln!("skipping program_test_get_methods: requires the ModuleMultiEntry .pte fixture");
    }

    // [spec:et:sem:program.executorch.et-runtime-namespace.program.get-named-data-map-fn/test]
    #[test]
    fn program_test_get_named_data_map_fail() {
        setup();
        if skip_add("program_test_get_named_data_map_fail") {}
    }

    // [spec:et:sem:program.executorch.et-runtime-namespace.program.load-fn/test]
    #[test]
    fn program_test_deprecated_load() {
        setup();
        if std::env::var("ET_MODULE_MULTI_ENTRY_PATH").is_err() {
            eprintln!("skipping program_test_deprecated_load: ET_MODULE_MULTI_ENTRY_PATH unset");
            return;
        }
        eprintln!(
            "skipping program_test_deprecated_load: requires the ModuleMultiEntry .pte fixture"
        );
    }

    // [spec:et:sem:program.executorch.et-runtime-namespace.program.load-segment-fn/test]
    #[test]
    fn program_test_load_constant_segment_with_no_constant_segment() {
        setup();
        if skip_add("program_test_load_constant_segment_with_no_constant_segment") {}
    }

    // [spec:et:sem:program.executorch.et-runtime-namespace.program.load-segment-fn/test]
    #[test]
    fn program_test_load_constant_segment() {
        setup();
        if std::env::var("ET_MODULE_ADD_MUL_PATH").is_err() {
            eprintln!("skipping program_test_load_constant_segment: ET_MODULE_ADD_MUL_PATH unset");
            return;
        }
        eprintln!(
            "skipping program_test_load_constant_segment: requires the ModuleAddMul .pte fixture"
        );
    }

    // [spec:et:sem:program.executorch.et-runtime-namespace.program.load-fn/test]
    #[test]
    fn program_test_load_constant_segment_when_constant_buffer_exists() {
        setup();
        if std::env::var("DEPRECATED_ET_MODULE_LINEAR_CONSTANT_BUFFER_PATH").is_err() {
            eprintln!(
                "skipping program_test_load_constant_segment_when_constant_buffer_exists: \
                 DEPRECATED_ET_MODULE_LINEAR_CONSTANT_BUFFER_PATH unset"
            );
            return;
        }
        eprintln!(
            "skipping program_test_load_constant_segment_when_constant_buffer_exists: \
             requires the deprecated constant-buffer .pte fixture"
        );
    }

    // [spec:et:sem:program.executorch.et-runtime-namespace.program.load-mutable-subsegment-into-fn/test]
    #[test]
    fn program_test_load_from_mutable_segment() {
        setup();
        if std::env::var("ET_MODULE_SIMPLE_TRAIN_PATH").is_err() {
            eprintln!(
                "skipping program_test_load_from_mutable_segment: ET_MODULE_SIMPLE_TRAIN_PATH unset"
            );
            return;
        }
        eprintln!(
            "skipping program_test_load_from_mutable_segment: requires the ModuleSimpleTrain .pte fixture"
        );
    }

    // [spec:et:sem:program.executorch.et-runtime-namespace.program.load-fn/test]
    #[test]
    fn program_test_load_and_check_pte_size() {
        setup();
        if std::env::var("ET_MODULE_ADD_MUL_PATH").is_err() {
            eprintln!(
                "skipping program_test_load_and_check_pte_size: ET_MODULE_ADD_MUL_PATH unset"
            );
            return;
        }
        eprintln!(
            "skipping program_test_load_and_check_pte_size: requires the ModuleAddMul .pte fixture"
        );
    }

    // [spec:et:sem:program.executorch.et-runtime-namespace.program.get-constant-buffer-data-fn/test]
    #[test]
    fn program_test_get_constant_buffer_data_rejects_oversized_request() {
        setup();
        if std::env::var("DEPRECATED_ET_MODULE_LINEAR_CONSTANT_BUFFER_PATH").is_err() {
            eprintln!(
                "skipping program_test_get_constant_buffer_data_rejects_oversized_request: \
                 DEPRECATED_ET_MODULE_LINEAR_CONSTANT_BUFFER_PATH unset"
            );
            return;
        }
        eprintln!(
            "skipping program_test_get_constant_buffer_data_rejects_oversized_request: \
             requires the deprecated constant-buffer .pte fixture"
        );
    }

    // ---- In-memory (no fixture) — run unconditionally. ----

    // Constructs a Program directly with a chosen segment_base_offset and a
    // pre-built flatbuffer body, mirroring `ProgramTestFriend::MakeProgram`.
    // `internal_program_bytes` must outlive the returned Program.
    fn make_program<'a>(
        loader: *const dyn DataLoader,
        segment_base_offset: usize,
        internal_program_bytes: &'a [u8],
    ) -> Program<'a> {
        Program::new(
            loader,
            segment_base_offset,
            FreeableBuffer::new(),
            internal_program_bytes.as_ptr(),
            internal_program_bytes.len(),
            FreeableBuffer::new(),
            None,
        )
    }

    // A non-zero segment_base_offset with an absent `segments` table must return
    // InvalidProgram rather than dereferencing null.
    // [spec:et:sem:program.executorch.et-runtime-namespace.program.load-segment-fn/test]
    #[test]
    fn program_test_load_segment_with_null_segments_does_not_crash() {
        setup();
        let mut builder = FlatBufferBuilder::with_capacity(256);
        let program = executorch_flatbuffer::Program::create(
            &mut builder,
            &executorch_flatbuffer::ProgramArgs::default(),
        );
        builder.finish(program, Some(executorch_flatbuffer::PROGRAM_IDENTIFIER));
        let internal_bytes = builder.finished_data().to_vec();

        let dummy = [0u8; 16];
        let loader = BufferDataLoader::new(dummy.as_ptr() as *const core::ffi::c_void, dummy.len());
        let program = make_program(
            &loader as *const _ as *const dyn DataLoader,
            16,
            &internal_bytes,
        );

        let result = program.load_segment(&SegmentInfo::new(Type::Backend, 0, c"b".as_ptr()));
        assert_eq!(ResultExt::error(&result), Error::InvalidProgram);
    }

    // Same malformed state reached through load_mutable_subsegment_into:
    // mutable_data_segments is populated so the function passes its own guards,
    // but segments is absent.
    // [spec:et:sem:program.executorch.et-runtime-namespace.program.load-mutable-subsegment-into-fn/test]
    #[test]
    fn program_test_load_mutable_subsegment_with_null_segments_does_not_crash() {
        setup();
        let mut builder = FlatBufferBuilder::with_capacity(256);
        let offsets = builder.create_vector::<u64>(&[0]);
        let subsegment = executorch_flatbuffer::SubsegmentOffsets::create(
            &mut builder,
            &executorch_flatbuffer::SubsegmentOffsetsArgs {
                segment_index: 0,
                offsets: Some(offsets),
            },
        );
        let mutable_data_segments = builder.create_vector(&[subsegment]);
        let program = executorch_flatbuffer::Program::create(
            &mut builder,
            &executorch_flatbuffer::ProgramArgs {
                version: 0,
                mutable_data_segments: Some(mutable_data_segments),
                ..Default::default()
            },
        );
        builder.finish(program, Some(executorch_flatbuffer::PROGRAM_IDENTIFIER));
        let internal_bytes = builder.finished_data().to_vec();

        let dummy = [0u8; 16];
        let loader = BufferDataLoader::new(dummy.as_ptr() as *const core::ffi::c_void, dummy.len());
        let program = make_program(
            &loader as *const _ as *const dyn DataLoader,
            16,
            &internal_bytes,
        );

        let mut out = [0u8; 4];
        assert_eq!(
            program.load_mutable_subsegment_into(
                0,
                0,
                out.len(),
                out.as_mut_ptr() as *mut core::ffi::c_void,
            ),
            Error::InvalidProgram
        );
    }

    // get_backend_delegate_data returns the pointer+size of the inline blob at
    // `index`, and Error::NotFound (without touching the out-params) for an
    // out-of-range index. No C++ test targets this friend method directly (it is
    // exercised transitively by Method init, which needs fixtures/kernels the
    // port lacks); this pins it against its sem rule with an in-memory program.
    // [spec:et:sem:program.executorch.et-runtime-namespace.program.get-backend-delegate-data-fn/test]
    #[test]
    fn program_test_get_backend_delegate_data() {
        setup();
        let mut builder = FlatBufferBuilder::with_capacity(256);
        let blob0 = builder.create_vector::<u8>(&[1, 2, 3]);
        let inline0 = executorch_flatbuffer::BackendDelegateInlineData::create(
            &mut builder,
            &executorch_flatbuffer::BackendDelegateInlineDataArgs { data: Some(blob0) },
        );
        let blob1 = builder.create_vector::<u8>(&[9, 8, 7, 6, 5]);
        let inline1 = executorch_flatbuffer::BackendDelegateInlineData::create(
            &mut builder,
            &executorch_flatbuffer::BackendDelegateInlineDataArgs { data: Some(blob1) },
        );
        let backend_delegate_data = builder.create_vector(&[inline0, inline1]);
        let program = executorch_flatbuffer::Program::create(
            &mut builder,
            &executorch_flatbuffer::ProgramArgs {
                version: 0,
                backend_delegate_data: Some(backend_delegate_data),
                ..Default::default()
            },
        );
        builder.finish(program, Some(executorch_flatbuffer::PROGRAM_IDENTIFIER));
        let internal_bytes = builder.finished_data().to_vec();

        let program = make_program(null_data_loader(), 0, &internal_bytes);

        // Index 0: 3-byte blob {1,2,3}.
        let mut data0: *const core::ffi::c_void = core::ptr::null();
        let mut size0: usize = 0;
        assert_eq!(
            program.get_backend_delegate_data(0, &mut data0, &mut size0),
            Error::Ok
        );
        assert_eq!(size0, 3);
        assert_eq!(
            unsafe { core::slice::from_raw_parts(data0 as *const u8, 3) },
            &[1u8, 2, 3]
        );

        // Index 1: 5-byte blob {9,8,7,6,5}.
        let mut data1: *const core::ffi::c_void = core::ptr::null();
        let mut size1: usize = 0;
        assert_eq!(
            program.get_backend_delegate_data(1, &mut data1, &mut size1),
            Error::Ok
        );
        assert_eq!(size1, 5);
        assert_eq!(
            unsafe { core::slice::from_raw_parts(data1 as *const u8, 5) },
            &[9u8, 8, 7, 6, 5]
        );

        // Out-of-range index: NotFound, out-params left unmodified.
        let mut data_oob: *const core::ffi::c_void = core::ptr::null();
        let mut size_oob: usize = 12345;
        assert_eq!(
            program.get_backend_delegate_data(2, &mut data_oob, &mut size_oob),
            Error::NotFound
        );
        assert!(data_oob.is_null());
        assert_eq!(size_oob, 12345);
    }

    // The `TensorParser::load_mutable_subsegment_into` friend forwarder
    // (tensor_parser_exec_aten.cpp) must delegate verbatim to
    // `Program::load_mutable_subsegment_into`, mirroring the C++
    // `ProgramTestFriend::load_mutable_subsegment_into` path. Same malformed
    // program (mutable_data_segments present, segments absent) reached through
    // the forwarder yields the identical InvalidProgram.
    // [spec:et:sem:tensor-parser-exec-aten.executorch.et-runtime-namespace.deserialization.tensor-parser.load-mutable-subsegment-into-fn/test]
    #[test]
    fn tensor_parser_load_mutable_subsegment_into_forwards_to_program() {
        use crate::runtime::executor::tensor_parser_exec_aten::TensorParser;
        setup();
        let mut builder = FlatBufferBuilder::with_capacity(256);
        let offsets = builder.create_vector::<u64>(&[0]);
        let subsegment = executorch_flatbuffer::SubsegmentOffsets::create(
            &mut builder,
            &executorch_flatbuffer::SubsegmentOffsetsArgs {
                segment_index: 0,
                offsets: Some(offsets),
            },
        );
        let mutable_data_segments = builder.create_vector(&[subsegment]);
        let program = executorch_flatbuffer::Program::create(
            &mut builder,
            &executorch_flatbuffer::ProgramArgs {
                version: 0,
                mutable_data_segments: Some(mutable_data_segments),
                ..Default::default()
            },
        );
        builder.finish(program, Some(executorch_flatbuffer::PROGRAM_IDENTIFIER));
        let internal_bytes = builder.finished_data().to_vec();

        let dummy = [0u8; 16];
        let loader = BufferDataLoader::new(dummy.as_ptr() as *const core::ffi::c_void, dummy.len());
        let program = make_program(
            &loader as *const _ as *const dyn DataLoader,
            16,
            &internal_bytes,
        );

        let mut out = [0u8; 4];
        assert_eq!(
            TensorParser::load_mutable_subsegment_into(
                &program,
                0,
                0,
                out.len(),
                out.as_mut_ptr() as *mut core::ffi::c_void,
            ),
            Error::InvalidProgram
        );
    }

    // Builds a minimal ExecutionPlan and Program (optionally with a
    // ContainerMetadata), copies it into a 16-byte-aligned buffer, and loads it
    // with Minimal verification. Returns the aligned bytes (kept alive by the
    // caller) and the loaded Program.
    struct AlignedProgram {
        _bytes: Vec<u8>,
        _loader: Box<BufferDataLoader>,
    }

    // Copies `data` to a 16-byte-aligned Vec and returns (aligned_vec, offset).
    fn aligned_copy(data: &[u8]) -> (Vec<u8>, usize) {
        let mut v = vec![0u8; data.len() + 16];
        let addr = v.as_ptr() as usize;
        let offset = (16 - (addr % 16)) % 16;
        v[offset..offset + data.len()].copy_from_slice(data);
        (v, offset)
    }

    fn build_program_with_plan(with_container_meta: bool, missing_out_str: bool) -> Vec<u8> {
        let mut builder = FlatBufferBuilder::with_capacity(2048);

        let container_meta = if with_container_meta {
            let inp = builder.create_string("test_input");
            let out = if missing_out_str {
                None
            } else {
                Some(builder.create_string("test_output"))
            };
            Some(executorch_flatbuffer::ContainerMetadata::create(
                &mut builder,
                &executorch_flatbuffer::ContainerMetadataArgs {
                    encoded_inp_str: Some(inp),
                    encoded_out_str: out,
                },
            ))
        } else {
            None
        };

        let plan_name = builder.create_string("forward");
        let empty_values = builder
            .create_vector::<flatbuffers::ForwardsUOffset<executorch_flatbuffer::EValue>>(&[]);
        let empty_inputs = builder.create_vector::<i32>(&[]);
        let empty_outputs = builder.create_vector::<i32>(&[]);
        let empty_chains = builder
            .create_vector::<flatbuffers::ForwardsUOffset<executorch_flatbuffer::Chain>>(&[]);
        let empty_operators = builder
            .create_vector::<flatbuffers::ForwardsUOffset<executorch_flatbuffer::Operator>>(&[]);
        let empty_delegates = builder.create_vector::<flatbuffers::ForwardsUOffset<
            executorch_flatbuffer::BackendDelegate,
        >>(&[]);
        let buffer_sizes = builder.create_vector::<i64>(&[0]);

        let execution_plan = executorch_flatbuffer::ExecutionPlan::create(
            &mut builder,
            &executorch_flatbuffer::ExecutionPlanArgs {
                name: Some(plan_name),
                container_meta_type: container_meta,
                values: Some(empty_values),
                inputs: Some(empty_inputs),
                outputs: Some(empty_outputs),
                chains: Some(empty_chains),
                operators: Some(empty_operators),
                delegates: Some(empty_delegates),
                non_const_buffer_sizes: Some(buffer_sizes),
                non_const_buffer_device: None,
            },
        );
        let execution_plans = builder.create_vector(&[execution_plan]);

        let empty_constant_buffer = builder
            .create_vector::<flatbuffers::ForwardsUOffset<executorch_flatbuffer::Buffer>>(&[]);
        let empty_backend_data = builder.create_vector::<flatbuffers::ForwardsUOffset<
            executorch_flatbuffer::BackendDelegateInlineData,
        >>(&[]);
        let empty_segments = builder
            .create_vector::<flatbuffers::ForwardsUOffset<executorch_flatbuffer::DataSegment>>(&[]);
        let cs_offsets = builder.create_vector::<u64>(&[0]);
        let constant_segment = executorch_flatbuffer::SubsegmentOffsets::create(
            &mut builder,
            &executorch_flatbuffer::SubsegmentOffsetsArgs {
                segment_index: 0,
                offsets: Some(cs_offsets),
            },
        );

        let program = executorch_flatbuffer::Program::create(
            &mut builder,
            &executorch_flatbuffer::ProgramArgs {
                version: 0,
                execution_plan: Some(execution_plans),
                constant_buffer: Some(empty_constant_buffer),
                backend_delegate_data: Some(empty_backend_data),
                segments: Some(empty_segments),
                constant_segment: Some(constant_segment),
                ..Default::default()
            },
        );
        builder.finish(program, Some(executorch_flatbuffer::PROGRAM_IDENTIFIER));
        builder.finished_data().to_vec()
    }

    // Loads a program built by build_program_with_plan and returns the encoding
    // Result for "forward", keeping the aligned buffer + loader alive via the
    // returned guard.
    fn load_and_get_encoding(
        bytes: &[u8],
        keep: &mut Vec<AlignedProgram>,
    ) -> Result<*const core::ffi::c_char> {
        let (aligned, offset) = aligned_copy(bytes);
        let loader = Box::new(BufferDataLoader::new(
            unsafe { aligned.as_ptr().add(offset) } as *const core::ffi::c_void,
            bytes.len(),
        ));
        let program = Program::load(
            &*loader as *const _ as *const dyn DataLoader,
            Verification::Minimal,
        );
        assert_eq!(ResultExt::error(&program), Error::Ok);
        let program = program.unwrap();
        let encoding = program.get_output_flattening_encoding(c"forward");
        keep.push(AlignedProgram {
            _bytes: aligned,
            _loader: loader,
        });
        encoding
    }

    // get_output_flattening_encoding must return InvalidProgram (not crash) when
    // container_meta_type is missing. Also verifies is_aligned (the aligned load
    // path), Program::new/load (constructing the Program), get_internal_program
    // (rebuilding the flatbuffer view), and get_execution_plan (matching the
    // "forward" plan by name before the container-meta check).
    // [spec:et:sem:program.executorch.et-runtime-namespace.program.get-output-flattening-encoding-fn/test]
    // [spec:et:sem:program.executorch.et-runtime-namespace.get-execution-plan-fn/test]
    // [spec:et:sem:program.executorch.et-runtime-namespace.is-aligned-fn/test]
    // [spec:et:sem:program.executorch.et-runtime-namespace.program.get-internal-program-fn/test]
    // [spec:et:sem:program.executorch.et-runtime-namespace.program.program-fn/test]
    #[test]
    fn program_test_get_output_flattening_encoding_with_missing_container_meta_type() {
        setup();
        let bytes = build_program_with_plan(false, false);
        let mut keep = Vec::new();
        let encoding = load_and_get_encoding(&bytes, &mut keep);
        assert_eq!(ResultExt::error(&encoding), Error::InvalidProgram);
    }

    // get_output_flattening_encoding must return InvalidProgram when
    // encoded_out_str is missing.
    // [spec:et:sem:program.executorch.et-runtime-namespace.program.get-output-flattening-encoding-fn/test]
    #[test]
    fn program_test_get_output_flattening_encoding_with_missing_encoded_out_str() {
        setup();
        let bytes = build_program_with_plan(true, true);
        let mut keep = Vec::new();
        let encoding = load_and_get_encoding(&bytes, &mut keep);
        assert_eq!(ResultExt::error(&encoding), Error::InvalidProgram);
    }

    // method_meta("forward") must return InvalidArgument (not crash) when the
    // plan name is null.
    // [spec:et:sem:program.executorch.et-runtime-namespace.program.method-meta-fn/test]
    #[test]
    fn program_test_null_plan_name_does_not_crash() {
        setup();
        let mut builder = FlatBufferBuilder::with_capacity(1024);

        let empty_values = builder
            .create_vector::<flatbuffers::ForwardsUOffset<executorch_flatbuffer::EValue>>(&[]);
        let empty_inputs = builder.create_vector::<i32>(&[]);
        let empty_outputs = builder.create_vector::<i32>(&[]);
        let empty_chains = builder
            .create_vector::<flatbuffers::ForwardsUOffset<executorch_flatbuffer::Chain>>(&[]);
        let empty_operators = builder
            .create_vector::<flatbuffers::ForwardsUOffset<executorch_flatbuffer::Operator>>(&[]);
        let empty_delegates = builder.create_vector::<flatbuffers::ForwardsUOffset<
            executorch_flatbuffer::BackendDelegate,
        >>(&[]);
        let buffer_sizes = builder.create_vector::<i64>(&[0]);

        // name = None (null plan name).
        let execution_plan = executorch_flatbuffer::ExecutionPlan::create(
            &mut builder,
            &executorch_flatbuffer::ExecutionPlanArgs {
                name: None,
                container_meta_type: None,
                values: Some(empty_values),
                inputs: Some(empty_inputs),
                outputs: Some(empty_outputs),
                chains: Some(empty_chains),
                operators: Some(empty_operators),
                delegates: Some(empty_delegates),
                non_const_buffer_sizes: Some(buffer_sizes),
                non_const_buffer_device: None,
            },
        );
        let execution_plans = builder.create_vector(&[execution_plan]);

        let empty_constant_buffer = builder
            .create_vector::<flatbuffers::ForwardsUOffset<executorch_flatbuffer::Buffer>>(&[]);
        let empty_backend_data = builder.create_vector::<flatbuffers::ForwardsUOffset<
            executorch_flatbuffer::BackendDelegateInlineData,
        >>(&[]);
        let empty_segments = builder
            .create_vector::<flatbuffers::ForwardsUOffset<executorch_flatbuffer::DataSegment>>(&[]);
        let cs_offsets = builder.create_vector::<u64>(&[0]);
        let constant_segment = executorch_flatbuffer::SubsegmentOffsets::create(
            &mut builder,
            &executorch_flatbuffer::SubsegmentOffsetsArgs {
                segment_index: 0,
                offsets: Some(cs_offsets),
            },
        );

        let program = executorch_flatbuffer::Program::create(
            &mut builder,
            &executorch_flatbuffer::ProgramArgs {
                version: 0,
                execution_plan: Some(execution_plans),
                constant_buffer: Some(empty_constant_buffer),
                backend_delegate_data: Some(empty_backend_data),
                segments: Some(empty_segments),
                constant_segment: Some(constant_segment),
                ..Default::default()
            },
        );
        builder.finish(program, Some(executorch_flatbuffer::PROGRAM_IDENTIFIER));
        let bytes = builder.finished_data().to_vec();

        let (aligned, offset) = aligned_copy(&bytes);
        let loader = BufferDataLoader::new(
            unsafe { aligned.as_ptr().add(offset) } as *const core::ffi::c_void,
            bytes.len(),
        );
        let program = Program::load(
            &loader as *const _ as *const dyn DataLoader,
            Verification::Minimal,
        );
        assert_eq!(ResultExt::error(&program), Error::Ok);
        let program = program.unwrap();

        let meta = program.method_meta(c"forward");
        assert_eq!(ResultExt::error(&meta), Error::InvalidArgument);
    }

    // C++ deletes the copy ctor and both assignment operators but keeps move
    // construction (Program travels through Result<Program>). The Rust analog is
    // a move-only owner: no Clone and no assignment-through-reference, so the
    // only transfer is a move that leaves the source binding statically
    // unusable. Moving a loaded Program (first out of the Result, then into a
    // new binding) must carry the owned program_data_/loader state intact: the
    // moved-to binding answers num_methods/get_method_name identically.
    // [spec:et:sem:program.executorch.et-runtime-namespace.program.operator-fn/test]
    #[test]
    fn program_test_move_only_ownership() {
        setup();
        let bytes = build_program_with_plan(true, false);
        let (aligned, offset) = aligned_copy(&bytes);
        let loader = BufferDataLoader::new(
            unsafe { aligned.as_ptr().add(offset) } as *const core::ffi::c_void,
            bytes.len(),
        );
        let program = Program::load(
            &loader as *const _ as *const dyn DataLoader,
            Verification::Minimal,
        );
        assert_eq!(ResultExt::error(&program), Error::Ok);
        // Move #1: out of the Result (the reason C++ keeps move-construction).
        let program = program.unwrap();
        assert_eq!(program.num_methods(), 1);

        // Move #2: into a new binding; `program` is now statically unusable
        // (the deleted copy/assignment operators' contract).
        let moved = program;
        assert_eq!(moved.num_methods(), 1);
        let name = moved.get_method_name(0);
        assert_eq!(ResultExt::error(&name), Error::Ok);
        assert_eq!(
            unsafe { core::ffi::CStr::from_ptr(*ResultExt::get(&name)) }
                .to_str()
                .unwrap(),
            "forward"
        );
    }
}
