//! Literal port of runtime/executor/method.cpp + runtime/executor/method.h.
//!
//! WAVE-2 SCOPE (complete): PASS 1 covered the `Method` struct,
//! `Method::load`/`init`, `parse_values` (all EValue tag arms),
//! `parse_external_constants`, `get_num_external_constants`, the delegate init
//! loop (`BackendDelegate` Init/GetProcessedData/PopulateCompileSpecs and the
//! destructor glue), `gen_instruction_arguments`, `resolve_operator`,
//! `populate_operator_name`, and the `InitializationState`/`StepState`
//! machinery. PASS 2 filled the input/output plumbing (`set_input`/`set_inputs`,
//! `set_output_data_ptr`, `get_inputs`/`get_outputs`, `get_attribute`,
//! `method_meta`, the value/index accessors). PASS 3 filled the execution engine
//! (`execute`, `step`/`experimental_step`, `reset_execution`/
//! `experimental_reset_execution`, `in_progress`, `execute_instruction` with the
//! KernelCall/DelegateCall/JumpFalseCall/MoveCall/FreeCall arms, `log_outputs`,
//! and the event-tracer/profiling hooks along the execute path). `parse_cond_value`
//! was ported alongside its JumpFalseCall consumer.
//!
//! OWNERSHIP DEVIATION: the C++ `Method` owns raw pointers (`values_`,
//! `delegates_`, `chains_`, `external_constants_`, `merged_data_map_`) into
//! `method_allocator`-owned bump memory and frees the non-trivially-destructible
//! ones by hand in `~Method`. Rust mirrors this literally: the fields are raw
//! pointers, construction placement-writes into allocator memory via
//! `core::ptr::write`, and `Drop` reproduces `~Method`'s manual teardown. `'a`
//! is the lifetime of the backing Program flatbuffer / allocator arena that all
//! these pointers view; the callers (`Program::load_method`) keep it alive.

use crate::runtime::backend::backend_init_context::BackendInitContext;
use crate::runtime::backend::backend_options_map::LoadBackendOptionsMap;
use crate::runtime::backend::interface::{
    BackendInterface, CompileSpec, DelegateHandle, SizedBuffer, get_backend_class,
};
use crate::runtime::core::array_ref::ArrayRef;
use crate::runtime::core::data_loader::{SegmentInfo, Type};
use crate::runtime::core::error::Error;
use crate::runtime::core::evalue::{BoxedEvalueList, EValue};
use crate::runtime::core::event_tracer::EventTracer;
use crate::runtime::core::event_tracer_hooks::{
    EventTracerProfileInstructionScope, EventTracerProfileMethodScope, EventTracerProfileOpScope,
};
use crate::runtime::core::exec_aten::util::scalar_type_util::{element_size, to_string};
use crate::runtime::core::exec_aten::util::tensor_util::{
    get_dim_order, internal, resize_tensor_same_type,
};
use crate::runtime::core::freeable_buffer::FreeableBuffer;
use crate::runtime::core::memory_allocator::{MemoryAllocatorBase, MemoryAllocatorExt};
use crate::runtime::core::named_data_map::NamedDataMap;
use crate::runtime::core::portable_type::scalar_type::ScalarType;
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::core::portable_type::tensor_impl::{DimOrderType, ssize_t};
use crate::runtime::core::result::{Result, ResultExt};
use crate::runtime::core::span::Span;
use crate::runtime::core::tag::tag_to_string_static;
use crate::runtime::executor::memory_manager::MemoryManager;
use crate::runtime::executor::merged_data_map::MergedDataMap;
use crate::runtime::executor::platform_memory_allocator::PlatformMemoryAllocator;
use crate::runtime::executor::program::Program;
use crate::runtime::executor::tensor_parser::{
    NamedData, get_data_by_key, parse_list_optional_type, parse_tensor, parse_tensor_list,
    validate_tensor_layout,
};
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;
use crate::runtime::kernel::operator_registry::{
    Kernel, OpFunction, TensorMeta, get_op_function_from_registry, get_op_function_from_registry_2,
};
use crate::schema::generated::executorch_flatbuffer;

// PORT-NOTE: local `ET_CHECK` / `ET_CHECK_MSG` fatal-assertion macro; logs at
// Fatal level and aborts (never returns an Error). Mirrors the same local
// definition used in memory_manager.rs, pending a shared assert module.
// Unresolved cross-module reference.
macro_rules! et_check_msg {
    ($cond:expr) => {
        if !($cond) {
            $crate::runtime::platform::abort::runtime_abort();
        }
    };
    ($cond:expr, $($arg:tt)*) => {
        if !($cond) {
            $crate::et_log!(Fatal, $($arg)*);
            $crate::runtime::platform::abort::runtime_abort();
        }
    };
}

// PORT-NOTE: the C++ `EXECUTORCH_SCOPE_PROF` macro is compiled to a no-op unless
// `PROFILING_ENABLED` is defined (see runtime/platform/profiler.h). The Rust
// `profiler` module is gated behind the `profiling-enabled` feature, so this
// thin RAII wrapper forwards to it when the feature is on and no-ops otherwise,
// matching both macro arms exactly (same pattern as program.rs).
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

// Maximum number of instructions that Method::execute() will run before
// returning an error. Prevents infinite loops caused by malformed programs
// (e.g., JumpFalseCall instructions whose destination_instruction points to
// themselves). Override at compile time via -DET_MAX_INSTRUCTIONS=<value>.
//
// PORT-NOTE: C++ `#ifndef ET_MAX_INSTRUCTIONS ... 10000000`. Rust has no
// preprocessor; the default is a plain const. The C++ `static_assert(> 0)`
// guard is unnecessary because the value is a fixed positive literal here.
const K_MAX_INSTRUCTIONS: usize = 10000000;

/// Runtime state for a backend delegate.
// [spec:et:def:method.executorch.et-runtime-namespace.backend-delegate]
//
// PORT-NOTE: field order preserved from C++ (`segment_`, `backend_`, `handle_`).
// The C++ copy/move ctors and assignment are all `= delete`d (the delegate owns
// a backend handle + FreeableBuffer, pinned in the delegates array and only
// constructed in place by `Init`); Rust reproduces this with a non-Clone,
// non-Copy struct that Method places directly into allocator memory. `backend_`
// is a borrowed `*mut dyn BackendInterface` (the C++ `const BackendInterface*`).
// The deleted copy-assignment (`operator=`) carries no runtime behavior; its
// markers collapse onto this non-`Clone`/non-`Copy` struct.
// [spec:et:def:method.executorch.et-runtime-namespace.backend-delegate.operator-fn]
// [spec:et:sem:method.executorch.et-runtime-namespace.backend-delegate.operator-fn]
struct BackendDelegate {
    segment_: FreeableBuffer,
    backend_: *mut dyn BackendInterface,
    handle_: *mut DelegateHandle,
}

impl BackendDelegate {
    /// Initializes an already-allocated BackendDelegate from its serialized
    /// representation.
    // [spec:et:def:method.executorch.et-runtime-namespace.backend-delegate.init-fn]
    // [spec:et:sem:method.executorch.et-runtime-namespace.backend-delegate.init-fn]
    fn init(
        delegate: &executorch_flatbuffer::BackendDelegate,
        program: *const Program,
        backend_init_context: &mut BackendInitContext,
        out: *mut BackendDelegate,
    ) -> Error {
        // Look up the backend.
        crate::et_check_or_return_error!(
            delegate.id().is_some(),
            InvalidProgram,
            "Missing backend id"
        );
        let backend_id: *const core::ffi::c_char =
            delegate.id().unwrap().as_ptr() as *const core::ffi::c_char;
        let backend: *mut dyn BackendInterface = get_backend_class(backend_id);
        crate::et_check_or_return_error!(
            !backend.is_null(),
            NotFound,
            "Backend {:?} is not registered.",
            backend_id
        );
        crate::et_check_or_return_error!(
            unsafe { (*backend).is_available() },
            NotFound,
            "Backend {:?} is not available.",
            backend_id
        );

        // Get the delegate data.
        let mut processed_data = Self::get_processed_data(delegate, program);
        if !ResultExt::ok(&processed_data) {
            crate::et_log!(Error, "Failed to load data for backend {:?}", backend_id);
            return ResultExt::error(&processed_data);
        }

        // Parse compilation specs from program
        let mut compile_specs: *mut CompileSpec = core::ptr::null_mut();
        let mut num_compile_specs: usize = 0;
        if delegate.compile_specs().is_some() {
            let err = Self::populate_compile_specs(
                &delegate.compile_specs().unwrap(),
                backend_init_context,
                &mut compile_specs,
            );
            if err != Error::Ok {
                crate::et_log!(
                    Error,
                    "Failed to get compile specs for backend {:?}",
                    backend_id
                );
                return err;
            }
            num_compile_specs = delegate.compile_specs().unwrap().len();
        }

        unsafe {
            (*out).backend_ = backend;
            (*out).handle_ = core::ptr::null_mut();
            // Pass a pointer to this buffer to the backend. It's safe for the
            // backend to point its handle to this object, since it will outlive
            // the backend.
            core::ptr::write(
                &mut (*out).segment_,
                FreeableBuffer::from_move(ResultExt::get_mut(&mut processed_data)),
            );
        }

        // Initialize the delegate.
        let handle = unsafe {
            (*backend).init(
                backend_init_context,
                &mut (*out).segment_,
                ArrayRef::from_raw_parts(compile_specs, num_compile_specs),
            )
        };
        if !ResultExt::ok(&handle) {
            crate::et_log!(
                Error,
                "Init failed for backend {:?}: 0x{:x}",
                backend_id,
                ResultExt::error(&handle) as u32
            );
            unsafe {
                (*out).segment_.free();
            }
            return ResultExt::error(&handle);
        }
        unsafe {
            (*out).handle_ = *ResultExt::get(&handle);
        }
        Error::Ok
    }

    // [spec:et:def:method.executorch.et-runtime-namespace.backend-delegate.backend-delegate-fn]
    // [spec:et:sem:method.executorch.et-runtime-namespace.backend-delegate.backend-delegate-fn]
    //
    // PORT-NOTE: `~BackendDelegate()` — if `backend_` is non-null, call
    // `backend_->destroy(handle_)`; then `segment_` frees itself. Modeled as
    // `Drop`. The `backend_` null pointer test mirrors the C++ zero/allocated-
    // but-uninitialized slot check.
    fn destroy(&mut self) {
        if !self.backend_.is_null() {
            unsafe {
                (*self.backend_).destroy(self.handle_);
            }
        }
    }

    // [spec:et:def:method.executorch.et-runtime-namespace.backend-delegate.execute-fn]
    // [spec:et:sem:method.executorch.et-runtime-namespace.backend-delegate.execute-fn]
    fn execute(
        &self,
        backend_execution_context: &mut crate::runtime::backend::backend_execution_context::BackendExecutionContext,
        args: Span<*mut EValue>,
    ) -> Error {
        let _scope_prof = ScopeProf::new(c"delegate_execute".as_ptr());
        unsafe { (*self.backend_).execute(backend_execution_context, self.handle_, args) }
    }

    // [spec:et:def:method.executorch.et-runtime-namespace.backend-delegate.populate-compile-specs-fn]
    // [spec:et:sem:method.executorch.et-runtime-namespace.backend-delegate.populate-compile-specs-fn]
    fn populate_compile_specs(
        compile_specs_in_program: &flatbuffers::Vector<
            flatbuffers::ForwardsUOffset<executorch_flatbuffer::CompileSpec>,
        >,
        backend_init_context: &mut BackendInitContext,
        out_spec: *mut *mut CompileSpec,
    ) -> Error {
        let number_of_compile_specs = compile_specs_in_program.len();

        let compile_specs_list: *mut CompileSpec =
            unsafe { &mut *backend_init_context.get_runtime_allocator() }
                .allocate_list::<CompileSpec>(
                    number_of_compile_specs,
                    core::mem::align_of::<CompileSpec>(),
                );
        if compile_specs_list.is_null() {
            return Error::MemoryAllocationFailed;
        }

        // Initialize the spec list for each method spec
        for j in 0..number_of_compile_specs {
            let compile_spec_in_program = compile_specs_in_program.get(j);

            unsafe {
                (*compile_specs_list.add(j)).key =
                    compile_spec_in_program.key().unwrap().as_ptr() as *const core::ffi::c_char;
                (*compile_specs_list.add(j)).value = SizedBuffer {
                    buffer: compile_spec_in_program.value().unwrap().bytes().as_ptr()
                        as *mut core::ffi::c_void,
                    nbytes: compile_spec_in_program.value().unwrap().len(),
                };
            }
        }

        unsafe {
            *out_spec = compile_specs_list;
        }
        Error::Ok
    }

    // [spec:et:def:method.executorch.et-runtime-namespace.backend-delegate.get-processed-data-fn]
    // [spec:et:sem:method.executorch.et-runtime-namespace.backend-delegate.get-processed-data-fn]
    fn get_processed_data(
        delegate: &executorch_flatbuffer::BackendDelegate,
        program: *const Program,
    ) -> Result<FreeableBuffer> {
        let processed = delegate.processed().unwrap();
        match processed.location() {
            executorch_flatbuffer::DataLocation::INLINE => {
                let mut data: *const core::ffi::c_void = core::ptr::null();
                let mut size: usize = 0;
                let err = unsafe {
                    (*program).get_backend_delegate_data(
                        processed.index() as usize,
                        &mut data,
                        &mut size,
                    )
                };
                if err != Error::Ok {
                    return Err(err);
                }
                Ok(FreeableBuffer::from_pointer(
                    data,
                    size,
                    /*free_fn=*/ None,
                    core::ptr::null_mut(),
                ))
            }
            executorch_flatbuffer::DataLocation::SEGMENT => {
                let backend_id: *const core::ffi::c_char =
                    delegate.id().unwrap().as_ptr() as *const core::ffi::c_char;
                unsafe {
                    (*program).load_segment(&SegmentInfo::new(
                        Type::Backend,
                        processed.index() as usize,
                        backend_id,
                    ))
                }
            }
            other => {
                crate::et_log!(Error, "Unknown data location {}", other.0 as u32);
                Err(Error::Internal)
            }
        }
    }
}

// PORT-NOTE: `~BackendDelegate()` mapped to `Drop`; see `destroy`.
impl Drop for BackendDelegate {
    fn drop(&mut self) {
        self.destroy();
    }
}

/// Runtime state for a chain of instructions.
// [spec:et:def:method.executorch.et-runtime-namespace.chain]
//
// PORT-NOTE: `s_chain_` is the borrowed `const executorch_flatbuffer::Chain*`,
// carried as the flatbuffer view over the Program buffer (`'a`).
// `argument_lists_` is `Span<InstructionArgs>` and `kernels_` is the
// `OpFunction*` array head, both into `method_allocator` memory.
struct Chain<'a> {
    s_chain_: executorch_flatbuffer::Chain<'a>,
    argument_lists_: Span<InstructionArgs<'a>>,
    kernels_: *mut Option<OpFunction>,
}

/// A list of pointers into the master values table that together compose the
/// argument list for a single instruction.
//
// PORT-NOTE: C++ `using InstructionArgs = Span<EValue*>`.
type InstructionArgs<'a> = Span<*mut EValue<'a>>;

// [spec:et:def:method.executorch.et-runtime-namespace.gen-instruction-arguments-fn]
// [spec:et:sem:method.executorch.et-runtime-namespace.gen-instruction-arguments-fn]
fn gen_instruction_arguments<'a>(
    method_allocator: *mut dyn MemoryAllocatorBase,
    num_values: usize,
    values: *mut EValue<'a>,
    num_args: usize,
    arg_idxs: &flatbuffers::Vector<'a, i32>,
) -> Result<InstructionArgs<'a>> {
    let arg_list: *mut *mut EValue<'a> = unsafe { &mut *method_allocator }
        .allocate_list::<*mut EValue<'a>>(num_args, core::mem::align_of::<*mut EValue<'a>>());
    if arg_list.is_null() {
        return Err(Error::MemoryAllocationFailed);
    }
    for i in 0..num_args {
        let arg_idx: i32 = arg_idxs.get(i);
        crate::et_check_or_return_error!(
            (arg_idx as usize) < num_values,
            InvalidProgram,
            "Arg index {} >= {}",
            arg_idx as isize,
            num_values
        );
        unsafe {
            *arg_list.add(i) = values.add(arg_idx as usize);
        }
    }
    Ok(Span::from_raw_parts(arg_list, num_args))
}

// [spec:et:def:method.executorch.et-runtime-namespace.parse-cond-value-fn]
// [spec:et:sem:method.executorch.et-runtime-namespace.parse-cond-value-fn]
fn parse_cond_value(cond_value: &EValue) -> Result<bool> {
    // The cond value attached to the JF instruction at the beginning of an
    // if/else branch is a Tensor which we parse and decide whether to continue
    // to execute the if branch or jump to the else branch.
    // The cond value attached to the JF instruction at the end of the if branch
    // is a Bool Scalar which resolves to false and points us to the instruction
    // to jump to which will take us to a point that is after the else branch.
    if cond_value.is_tensor() {
        let cond_val = cond_value.to_tensor();

        // All the tensors and scalar cond values should be of bool type
        // currently. If that's not the case then something is wrong in the model
        // and we should exit.
        crate::et_check_or_return_error!(
            ScalarType::Bool == cond_val.scalar_type(),
            InvalidProgram,
            "Expected dtype of {} got {}",
            ScalarType::Bool as i8,
            cond_val.scalar_type() as i8
        );

        let cond_data: *const bool = cond_val.const_data_ptr::<bool>();
        crate::et_check_or_return_error!(!cond_data.is_null(), InvalidState, "Tensor data is null");
        for i in 0..(cond_val.numel() as usize) {
            if !unsafe { *cond_data.add(i) } {
                return Ok(false);
            }
        }
    } else if cond_value.is_bool() {
        if !cond_value.to_bool() {
            return Ok(false);
        }
    } else {
        crate::et_log!(
            Error,
            "Unsupported JF EValue type {}",
            cond_value.tag as u32
        );
        return Err(Error::InvalidProgram);
    }

    Ok(true)
}

/// Tracks what step in program execution we are on.
// [spec:et:def:method.executorch.et-runtime-namespace.method.step-state]
#[derive(Clone, Copy)]
struct StepState {
    chain_idx: usize,
    instr_idx: usize,
}

// [spec:et:def:method.executorch.et-runtime-namespace.method.initialization-state]
#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
enum InitializationState {
    Uninitialized,
    Initialized,
    InitializationFailed,
}

/// An executable method of an executorch program. Maps to a python method like
/// `forward()` on the original nn.Module.
// [spec:et:def:method.executorch.et-runtime-namespace.method]
//
// The deleted copy-assignment (`Method& operator=(const Method&)`) carries no
// runtime behavior; `Method` is modeled as a non-`Clone`, move-only owner, so
// its markers collapse onto this struct.
// [spec:et:def:method.executorch.et-runtime-namespace.method.operator-fn]
// [spec:et:sem:method.executorch.et-runtime-namespace.method.operator-fn]
pub struct Method<'a> {
    step_state_: StepState,
    program_: *const Program<'a>,
    memory_manager_: *mut MemoryManager,
    temp_allocator_: *mut dyn MemoryAllocatorBase,
    serialization_plan_: Option<executorch_flatbuffer::ExecutionPlan<'a>>,
    event_tracer_: *mut dyn EventTracer,

    n_value_: usize,
    values_: *mut EValue<'a>,
    input_set_: *mut bool,

    n_delegate_: usize,
    delegates_: *mut BackendDelegate,

    n_chains_: usize,
    chains_: *mut Chain<'a>,

    merged_data_map_: *mut MergedDataMap,
    external_constants_: *mut NamedData,
    n_external_constants_: usize,

    kernel_registry_: Span<Kernel>,

    init_state_: InitializationState,
}

impl<'a> Method<'a> {
    // PORT-NOTE: private ctor `Method(program, memory_manager, event_tracer,
    // temp_allocator, kernel_registry)`. Zero-initializes all state and sets
    // `init_state_ = Uninitialized`, mirroring the C++ member-init list.
    fn new(
        program: *const Program<'a>,
        memory_manager: *mut MemoryManager,
        event_tracer: *mut dyn EventTracer,
        temp_allocator: *mut dyn MemoryAllocatorBase,
        kernel_registry: Span<Kernel>,
    ) -> Self {
        Method {
            step_state_: StepState {
                chain_idx: 0,
                instr_idx: 0,
            },
            program_: program,
            memory_manager_: memory_manager,
            temp_allocator_: temp_allocator,
            serialization_plan_: None,
            event_tracer_: event_tracer,
            n_value_: 0,
            values_: core::ptr::null_mut(),
            input_set_: core::ptr::null_mut(),
            n_delegate_: 0,
            delegates_: core::ptr::null_mut(),
            n_chains_: 0,
            chains_: core::ptr::null_mut(),
            merged_data_map_: core::ptr::null_mut(),
            external_constants_: core::ptr::null_mut(),
            n_external_constants_: 0,
            kernel_registry_: kernel_registry,
            init_state_: InitializationState::Uninitialized,
        }
    }

    /// Returns true if the Method was successfully initialized.
    // [spec:et:def:method.executorch.et-runtime-namespace.method.initialized-fn]
    // [spec:et:sem:method.executorch.et-runtime-namespace.method.initialized-fn]
    fn initialized(&self) -> bool {
        self.init_state_ == InitializationState::Initialized
    }

    // [spec:et:def:method.executorch.et-runtime-namespace.method.get-num-external-constants-fn]
    // [spec:et:sem:method.executorch.et-runtime-namespace.method.get-num-external-constants-fn]
    fn get_num_external_constants(&mut self) -> Result<usize> {
        let flatbuffer_values = self.serialization_plan_.unwrap().values().unwrap();
        let n_value = flatbuffer_values.len();

        let mut n_external_constants: usize = 0;
        for i in 0..n_value {
            let serialization_value = flatbuffer_values.get(i);
            // Ensure values are non-null.
            // Note that as a side-effect of this check, we're guaranteed that
            // all values are non-null, so later loops can skip that check.
            //
            // PORT-NOTE: the flatbuffer accessor returns a value (non-nullable
            // table); the `serialization_value != nullptr` half is always true
            // here, so the check reduces to `val_type() == Null || val() !=
            // nullptr`.
            crate::et_check_or_return_error!(
                serialization_value.val_type() == executorch_flatbuffer::KernelTypes::Null
                    || serialization_value.val().is_some(),
                InvalidProgram,
                "Null value at index {}",
                i
            );
            // Ignore non-tensor types.
            if serialization_value.val_type() != executorch_flatbuffer::KernelTypes::Tensor {
                continue;
            }
            let s_tensor = serialization_value.val_as_tensor().unwrap();

            // An external constant is tagged with EXTERNAL and has no
            // allocation_info.
            if s_tensor.extra_tensor_info().is_some()
                && s_tensor.extra_tensor_info().unwrap().location()
                    == executorch_flatbuffer::TensorDataLocation::EXTERNAL
                && s_tensor.allocation_info().is_none()
            {
                n_external_constants += 1;
            }
        }
        Ok(n_external_constants)
    }

    // [spec:et:def:method.executorch.et-runtime-namespace.method.parse-external-constants-fn]
    // [spec:et:sem:method.executorch.et-runtime-namespace.method.parse-external-constants-fn]
    fn parse_external_constants(&mut self, external_data_map: *const dyn NamedDataMap) -> Error {
        crate::et_check_or_return_error!(
            !is_null_named_data_map(external_data_map),
            InvalidState,
            "external_data_map is null"
        );
        let flatbuffer_values = self.serialization_plan_.unwrap().values().unwrap();
        let n_value = flatbuffer_values.len();

        // n_external_constants_ counts the number of successfully-initialized
        // external constants for ~Method() to clean up, and is incremented at
        // the bottom of the loop. This makes it safe for errors to return
        // without updating any state.
        self.n_external_constants_ = 0;
        for i in 0..n_value {
            let serialization_value = flatbuffer_values.get(i);
            // Ignore non-tensor types.
            if serialization_value.val_type() != executorch_flatbuffer::KernelTypes::Tensor {
                continue;
            }
            let s_tensor = serialization_value.val_as_tensor().unwrap();
            // Constant tensors are resolved here; tensors with allocation_info
            // are mutable and are resolved in parse_values.
            if s_tensor.extra_tensor_info().is_none()
                || s_tensor.extra_tensor_info().unwrap().location()
                    != executorch_flatbuffer::TensorDataLocation::EXTERNAL
                || s_tensor.allocation_info().is_some()
            {
                continue;
            }
            crate::et_check_or_return_error!(
                s_tensor
                    .extra_tensor_info()
                    .unwrap()
                    .fully_qualified_name()
                    .is_some(),
                InvalidExternalData,
                "Fully qualified name of external tensor is null at index {}",
                i
            );

            let key: *const core::ffi::c_char = s_tensor
                .extra_tensor_info()
                .unwrap()
                .fully_qualified_name()
                .unwrap()
                .as_ptr()
                as *const core::ffi::c_char;

            // Check if this tensor has already been resolved.
            if !get_data_by_key(
                key,
                Span::from_raw_parts(self.external_constants_, self.n_external_constants_),
            )
            .is_null()
            {
                continue;
            }
            let key_str = unsafe { core::ffi::CStr::from_ptr(key).to_str().unwrap_or("") };
            let tensor_layout = unsafe { (*external_data_map).get_tensor_layout(key_str) };
            if !ResultExt::ok(&tensor_layout) {
                crate::et_log!(Info, "Failed to get metadata for key {:?}", key);
                return ResultExt::error(&tensor_layout);
            }
            // Check external tensor compatibility.
            let err = validate_tensor_layout(&s_tensor, ResultExt::get(&tensor_layout));
            if err != Error::Ok {
                return err;
            }
            // Save the key.
            unsafe {
                (*self.external_constants_.add(self.n_external_constants_)).key = key;
            }

            // Save the buffer.
            let mut buffer = unsafe { (*external_data_map).get_data(key_str) };
            crate::et_check_or_return_error!(
                ResultExt::ok(&buffer),
                InvalidExternalData,
                "Buffer retrieved from get_data is not valid"
            );
            unsafe {
                core::ptr::write(
                    &mut (*self.external_constants_.add(self.n_external_constants_)).buffer,
                    FreeableBuffer::from_move(ResultExt::get_mut(&mut buffer)),
                );
            }

            self.n_external_constants_ += 1;
        }
        Error::Ok
    }

    // [spec:et:def:method.executorch.et-runtime-namespace.method.parse-values-fn]
    // [spec:et:sem:method.executorch.et-runtime-namespace.method.parse-values-fn]
    fn parse_values(&mut self, external_data_map: *const dyn NamedDataMap) -> Error {
        let flatbuffer_values_opt = self.serialization_plan_.unwrap().values();
        crate::et_check_or_return_error!(
            flatbuffer_values_opt.is_some(),
            InvalidProgram,
            "Missing values"
        );
        let flatbuffer_values = flatbuffer_values_opt.unwrap();
        let n_value = flatbuffer_values.len();
        self.values_ = unsafe { &mut *(*self.memory_manager_).method_allocator() }
            .allocate_list::<EValue<'a>>(n_value, core::mem::align_of::<EValue<'a>>());
        if self.values_.is_null() {
            crate::et_log!(Error, "Failed to allocate values array of size {}", n_value);
            return Error::MemoryAllocationFailed;
        }
        let n_input = self.inputs_size();
        if n_input > 0 {
            self.input_set_ = unsafe { &mut *(*self.memory_manager_).method_allocator() }
                .allocate_list::<bool>(n_input, core::mem::align_of::<bool>());
            if self.input_set_.is_null() {
                crate::et_log!(
                    Error,
                    "Failed to allocate input_set array of size {}",
                    n_input
                );
                return Error::MemoryAllocationFailed;
            }
            for i in 0..n_input {
                unsafe {
                    *self.input_set_.add(i) = false;
                }
            }
        }

        // Count the number of tensors marked as EXTERNAL for this method. The
        // actual number of external constants may be smaller, eg. if multiple
        // tensors point to the same underlying data buffer. This function also
        // ensures that all flatbuffer_values entries are non-null, so
        // `val_as_X()` calls below are guaranteed to return non-null pointers.
        let max_external_constants = self.get_num_external_constants();
        if !ResultExt::ok(&max_external_constants) {
            return ResultExt::error(&max_external_constants);
        }
        if *ResultExt::get(&max_external_constants) > 0 {
            // Allocate space for external tensors.
            self.external_constants_ = unsafe { &mut *(*self.memory_manager_).method_allocator() }
                .allocate_list::<NamedData>(
                *ResultExt::get(&max_external_constants),
                core::mem::align_of::<NamedData>(),
            );
            if self.external_constants_.is_null() {
                crate::et_log!(
                    Error,
                    "Failed to allocate external_constants array of size {}",
                    *ResultExt::get(&max_external_constants)
                );
                return Error::MemoryAllocationFailed;
            }
            let err = self.parse_external_constants(external_data_map);
            if err != Error::Ok {
                return err;
            }
        }

        // n_value_ counts the number of successfully-initialized values for
        // ~Method() to clean up, and is incremented at the bottom of the loop.
        // This makes it safe for errors to return without updating any state.
        self.n_value_ = 0;

        for i in 0..n_value {
            let serialization_value = flatbuffer_values.get(i);

            match serialization_value.val_type() {
                executorch_flatbuffer::KernelTypes::Null => {
                    // Placement new as the list elements are not initialized, so
                    // calling copy assignment is not defined if its non trivial
                    // (Imagine the garbage in values_[i] thinks its an
                    // at::Tensor).
                    unsafe {
                        core::ptr::write(self.values_.add(i), EValue::new());
                    }
                }
                executorch_flatbuffer::KernelTypes::Int => {
                    let int_val = serialization_value.val_as_int().unwrap().int_val();
                    unsafe {
                        core::ptr::write(self.values_.add(i), EValue::from_int(int_val));
                    }
                }
                executorch_flatbuffer::KernelTypes::Double => {
                    let double_val = serialization_value.val_as_double().unwrap().double_val();
                    unsafe {
                        core::ptr::write(self.values_.add(i), EValue::from_double(double_val));
                    }
                }
                executorch_flatbuffer::KernelTypes::Bool => {
                    let bool_val = serialization_value.val_as_bool().unwrap().bool_val();
                    unsafe {
                        core::ptr::write(self.values_.add(i), EValue::from_bool(bool_val));
                    }
                }
                executorch_flatbuffer::KernelTypes::IntList => {
                    let items_opt = serialization_value.val_as_int_list().unwrap().items();
                    crate::et_check_or_return_error!(
                        items_opt.is_some(),
                        InvalidProgram,
                        "Missing list at index {}",
                        i
                    );
                    let items = items_opt.unwrap();
                    // Allocate space for boxed and unboxed list representations
                    // using values_ as source of truth
                    let evalp_list: *mut *mut EValue<'a> =
                        unsafe { &mut *(*self.memory_manager_).method_allocator() }
                            .allocate_list::<*mut EValue<'a>>(
                                items.len(),
                                core::mem::align_of::<*mut EValue<'a>>(),
                            );
                    let int_list: *mut i64 =
                        unsafe { &mut *(*self.memory_manager_).method_allocator() }
                            .allocate_list::<i64>(items.len(), core::mem::align_of::<i64>());

                    // initialize boxed list
                    for j in 0..items.len() {
                        let value_index = items.get(j);
                        crate::et_check_or_return_error!(
                            value_index >= 0 && (value_index as usize) < n_value,
                            InvalidProgram,
                            "Invalid value index {} for IntList {} index {}",
                            value_index,
                            i,
                            j
                        );
                        unsafe {
                            *evalp_list.add(j) = self.values_.add(value_index as usize);
                        }
                    }
                    let boxed_list_mem: *mut BoxedEvalueList<'a, i64> =
                        unsafe { &mut *(*self.memory_manager_).method_allocator() }
                            .allocate_instance::<BoxedEvalueList<'a, i64>>(core::mem::align_of::<
                                BoxedEvalueList<'a, i64>,
                            >(
                            ));
                    unsafe {
                        core::ptr::write(
                            boxed_list_mem,
                            BoxedEvalueList::<i64>::new(evalp_list, int_list, items.len() as i32),
                        );
                        core::ptr::write(
                            self.values_.add(i),
                            EValue::from_int_list(boxed_list_mem),
                        );
                    }
                }
                executorch_flatbuffer::KernelTypes::BoolList => {
                    let items_opt = serialization_value.val_as_bool_list().unwrap().items();
                    crate::et_check_or_return_error!(
                        items_opt.is_some(),
                        InvalidProgram,
                        "Missing list at index {}",
                        i
                    );
                    let items = items_opt.unwrap();
                    // NOTE: This is technically not portable. A platform could
                    // technically define boolean as something longer than a
                    // byte. This would be an exceptionally rare case, and this
                    // type is currently unused in any operators in ATen that we
                    // would need to support. To be properly portable here we
                    // need to allocate a new array of bool and copy cast the
                    // flatbuffer data into it, but because of how exceptionally
                    // rare this case is its low prio TODO: jakeszwe
                    let bool_list_mem: *mut ArrayRef<bool> =
                        unsafe { &mut *(*self.memory_manager_).method_allocator() }
                            .allocate_instance::<ArrayRef<bool>>(core::mem::align_of::<
                                ArrayRef<bool>,
                            >());
                    unsafe {
                        core::ptr::write(
                            bool_list_mem,
                            ArrayRef::<bool>::from_raw_parts(
                                items.bytes().as_ptr() as *const bool,
                                items.len(),
                            ),
                        );
                        core::ptr::write(
                            self.values_.add(i),
                            EValue::from_bool_list(bool_list_mem),
                        );
                    }
                }
                executorch_flatbuffer::KernelTypes::DoubleList => {
                    let items_opt = serialization_value.val_as_double_list().unwrap().items();
                    crate::et_check_or_return_error!(
                        items_opt.is_some(),
                        InvalidProgram,
                        "Missing list at index {}",
                        i
                    );
                    let items = items_opt.unwrap();
                    let double_list_mem: *mut ArrayRef<f64> =
                        unsafe { &mut *(*self.memory_manager_).method_allocator() }
                            .allocate_instance::<ArrayRef<f64>>(
                                core::mem::align_of::<ArrayRef<f64>>(),
                            );
                    unsafe {
                        core::ptr::write(
                            double_list_mem,
                            ArrayRef::<f64>::from_raw_parts(
                                items.bytes().as_ptr() as *const f64,
                                items.len(),
                            ),
                        );
                        core::ptr::write(
                            self.values_.add(i),
                            EValue::from_double_list(double_list_mem),
                        );
                    }
                }
                executorch_flatbuffer::KernelTypes::String => {
                    let fb_str_opt = serialization_value.val_as_string().unwrap().string_val();
                    crate::et_check_or_return_error!(
                        fb_str_opt.is_some(),
                        InvalidProgram,
                        "Missing string at index {}",
                        i
                    );
                    let fb_str = fb_str_opt.unwrap();
                    let char_list_mem: *mut ArrayRef<u8> =
                        unsafe { &mut *(*self.memory_manager_).method_allocator() }
                            .allocate_instance::<ArrayRef<u8>>(
                                core::mem::align_of::<ArrayRef<u8>>(),
                            );
                    unsafe {
                        core::ptr::write(
                            char_list_mem,
                            ArrayRef::<u8>::from_raw_parts(fb_str.as_ptr(), fb_str.len()),
                        );
                        core::ptr::write(self.values_.add(i), EValue::from_string(char_list_mem));
                    }
                }
                executorch_flatbuffer::KernelTypes::Tensor => {
                    let s_tensor = serialization_value.val_as_tensor().unwrap();
                    let t = parse_tensor(
                        unsafe { &*self.program_ },
                        unsafe { &mut *self.memory_manager_ },
                        &s_tensor,
                        named_data_map_ref(external_data_map),
                        Span::from_raw_parts(self.external_constants_, self.n_external_constants_),
                    );
                    if !ResultExt::ok(&t) {
                        crate::et_log!(
                            Error,
                            "Failed parsing tensor at index {}: 0x{:x}",
                            i,
                            ResultExt::error(&t) as u32
                        );
                        return ResultExt::error(&t);
                    }
                    // PORT-NOTE (WAVE-2 FIX): `mem::replace(get_mut(&mut t),
                    // unreachable_tensor())` panicked because `mem::replace`
                    // eagerly evaluates the sentinel; `t` is already Ok-checked
                    // above, so move the tensor out via `unwrap`.
                    unsafe {
                        core::ptr::write(self.values_.add(i), EValue::from_tensor(t.unwrap()));
                    }
                }
                executorch_flatbuffer::KernelTypes::TensorList => {
                    let items_opt = serialization_value.val_as_tensor_list().unwrap().items();
                    crate::et_check_or_return_error!(
                        items_opt.is_some(),
                        InvalidProgram,
                        "Missing list at index {}",
                        i
                    );
                    let items = items_opt.unwrap();
                    // get list of serialization tensors and allocate storage for
                    // executor tensors
                    let tensors = parse_tensor_list(
                        &items,
                        self.values_,
                        n_value, // The size of the full array.
                        unsafe { &mut *self.memory_manager_ },
                    );
                    if !ResultExt::ok(&tensors) {
                        crate::et_log!(
                            Error,
                            "Failed parsing tensor list at index {}: 0x{:x}",
                            i,
                            ResultExt::error(&tensors) as u32
                        );
                        return ResultExt::error(&tensors);
                    }
                    let mut tensors = tensors;
                    let boxed_tensor_list_mem: *mut BoxedEvalueList<'a, Tensor<'a>> =
                        unsafe { &mut *(*self.memory_manager_).method_allocator() }
                            .allocate_instance::<BoxedEvalueList<'a, Tensor<'a>>>(
                                core::mem::align_of::<BoxedEvalueList<'a, Tensor<'a>>>(),
                            );
                    unsafe {
                        core::ptr::write(
                            boxed_tensor_list_mem,
                            core::mem::replace(
                                ResultExt::get_mut(&mut tensors),
                                BoxedEvalueList::<Tensor<'a>>::default_new(),
                            ),
                        );
                        core::ptr::write(
                            self.values_.add(i),
                            EValue::from_tensor_list(boxed_tensor_list_mem),
                        );
                    }
                }
                executorch_flatbuffer::KernelTypes::OptionalTensorList => {
                    let items_opt = serialization_value
                        .val_as_optional_tensor_list()
                        .unwrap()
                        .items();
                    crate::et_check_or_return_error!(
                        items_opt.is_some(),
                        InvalidProgram,
                        "Missing list at index {}",
                        i
                    );
                    let items = items_opt.unwrap();
                    // Same as TensorList but optional<Tensor> instead of Tensor
                    let tensors = parse_list_optional_type::<Tensor<'a>>(
                        &items,
                        self.values_,
                        n_value, // The size of the full array.
                        unsafe { &mut *self.memory_manager_ },
                    );
                    if !ResultExt::ok(&tensors) {
                        crate::et_log!(
                            Error,
                            "Failed parsing optional tensor list at index {}: 0x{:x}",
                            i,
                            ResultExt::error(&tensors) as u32
                        );
                        return ResultExt::error(&tensors);
                    }
                    let mut tensors = tensors;
                    let boxed_optional_tensor_list_mem: *mut BoxedEvalueList<
                        'a,
                        Option<Tensor<'a>>,
                    > = unsafe { &mut *(*self.memory_manager_).method_allocator() }
                        .allocate_instance::<BoxedEvalueList<'a, Option<Tensor<'a>>>>(
                            core::mem::align_of::<BoxedEvalueList<'a, Option<Tensor<'a>>>>(),
                        );
                    unsafe {
                        core::ptr::write(
                            boxed_optional_tensor_list_mem,
                            core::mem::replace(
                                ResultExt::get_mut(&mut tensors),
                                BoxedEvalueList::<Option<Tensor<'a>>>::default_new(),
                            ),
                        );
                        core::ptr::write(
                            self.values_.add(i),
                            EValue::from_list_optional_tensor(boxed_optional_tensor_list_mem),
                        );
                    }
                }
                other => {
                    // flatbuffer enums start at 0, but they generate a hidden
                    // NONE enum and give it that value. schema.fbs doesnt show
                    // this type, so I subtract one to keep the output in 0 based
                    // indexing for a disgruntled debugger seeing this error
                    // message and checking schema.fbs
                    crate::et_log!(
                        Error,
                        "Unknown KernelTypes value {} at index {}",
                        (other.0 as u32).wrapping_sub(1),
                        i
                    );
                    return Error::InvalidProgram;
                }
            }

            // ~Method() will try to clean up n_value_ entries in the values_
            // array. Only increment this once we know the entry is valid, so
            // that we don't try to clean up an uninitialized entry.
            self.n_value_ = i + 1;
        }
        Error::Ok
    }

    // [spec:et:def:method.executorch.et-runtime-namespace.method.resolve-operator-fn]
    // [spec:et:sem:method.executorch.et-runtime-namespace.method.resolve-operator-fn]
    fn resolve_operator(
        &mut self,
        op_index: i32,
        kernels: *mut Option<OpFunction>,
        kernel_index: usize,
        args: InstructionArgs<'a>,
        n_args: usize,
    ) -> Error {
        // TODO(T153506819) Investigate optimizing this function for both
        // space and time.

        // resolve name
        const K_TEMP_BUFFER_SIZE_FOR_NAME: usize = 100;
        let mut operator_name = [0 as core::ffi::c_char; K_TEMP_BUFFER_SIZE_FOR_NAME];
        let ops_opt = self.serialization_plan_.unwrap().operators();
        crate::et_check_or_return_error!(
            ops_opt.is_some() && (op_index as u32) < ops_opt.unwrap().len() as u32,
            InvalidProgram,
            "Op index {} out of range",
            op_index as u32
        );
        let op = ops_opt.unwrap().get(op_index as usize);

        let err =
            populate_operator_name(&op, K_TEMP_BUFFER_SIZE_FOR_NAME, operator_name.as_mut_ptr());
        if err != Error::Ok {
            return err;
        }

        // resolve tensor meta
        // Since temp allocator can be freed, we optimistically try to use that
        // allocator first.
        let mut allocator = unsafe { (*self.memory_manager_).temp_allocator() };
        // However, it does not have to be provided, so if it is not provided (or
        // an empty one is provided), we fall back to the method allocator.
        if allocator.is_null() || unsafe { (*allocator).size() } == 0 {
            allocator = unsafe { (*self.memory_manager_).method_allocator() };
        }
        let meta: *mut TensorMeta = unsafe { &mut *allocator }
            .allocate_list::<TensorMeta>(n_args, core::mem::align_of::<TensorMeta>());
        if meta.is_null() {
            if core::ptr::addr_eq(allocator, unsafe {
                (*self.memory_manager_).temp_allocator()
            }) {
                unsafe {
                    (*(*self.memory_manager_).temp_allocator()).reset();
                }
            }
            return Error::MemoryAllocationFailed;
        }

        let mut count: usize = 0;
        for i in 0..n_args {
            let eval: *mut EValue<'a> = unsafe { *args.data().add(i) };
            // handle tensor list as well
            if unsafe { (*eval).is_tensor() } {
                let tensor = unsafe { (*eval).to_tensor() };
                unsafe {
                    (*meta.add(count)).dtype_ = tensor.scalar_type();
                }
                let dim_order_ptr: *mut DimOrderType = unsafe { &mut *allocator }
                    .allocate_list::<DimOrderType>(
                        tensor.dim() as usize,
                        core::mem::align_of::<DimOrderType>(),
                    );
                if dim_order_ptr.is_null() {
                    if core::ptr::addr_eq(allocator, unsafe {
                        (*self.memory_manager_).temp_allocator()
                    }) {
                        unsafe {
                            (*(*self.memory_manager_).temp_allocator()).reset();
                        }
                    }
                    return Error::MemoryAllocationFailed;
                }
                let size = tensor.dim() as usize;
                let err = unsafe { get_dim_order(tensor, dim_order_ptr, size) };
                crate::et_check_or_return_error!(
                    err == Error::Ok,
                    InvalidArgument,
                    "Error setting dim_order {}: 0x{:x}",
                    i,
                    err as u32
                );
                unsafe {
                    (*meta.add(count)).dim_order_ = Span::from_raw_parts(dim_order_ptr, size);
                }
                count += 1;
            }
        }

        // Find a kernel with the matching name and tensor meta. Try
        // method-scoped registry first (if provided), then fall back to global.
        let op_function: Result<OpFunction> = {
            let mut resolved: Option<Result<OpFunction>> = None;
            if !self.kernel_registry_.empty() {
                let method_scoped_op_function = get_op_function_from_registry(
                    operator_name.as_ptr(),
                    Span::from_raw_parts(meta, count),
                    self.kernel_registry_,
                );
                if ResultExt::ok(&method_scoped_op_function) {
                    resolved = Some(method_scoped_op_function);
                }
            }
            match resolved {
                Some(r) => r,
                None => get_op_function_from_registry_2(
                    operator_name.as_ptr(),
                    Span::from_raw_parts(meta, count),
                ),
            }
        };
        if !ResultExt::ok(&op_function) {
            crate::et_log!(
                Error,
                "Missing operator: [{}] {}",
                op_index as isize,
                cstr_lossy(operator_name.as_ptr())
            );
            if core::ptr::addr_eq(allocator, unsafe {
                (*self.memory_manager_).temp_allocator()
            }) {
                unsafe {
                    (*(*self.memory_manager_).temp_allocator()).reset();
                }
            }
            return ResultExt::error(&op_function);
        }
        unsafe {
            *kernels.add(kernel_index) = Some(*ResultExt::get(&op_function));
        }

        // If we used the temp allocator here, reset it.
        if core::ptr::addr_eq(allocator, unsafe {
            (*self.memory_manager_).temp_allocator()
        }) {
            unsafe {
                (*(*self.memory_manager_).temp_allocator()).reset();
            }
        }

        Error::Ok
    }

    /// Static factory used by Program.
    // [spec:et:def:method.executorch.et-runtime-namespace.method.load-fn]
    // [spec:et:sem:method.executorch.et-runtime-namespace.method.load-fn]
    #[must_use]
    pub fn load(
        s_plan: executorch_flatbuffer::ExecutionPlan<'a>,
        program: &Program<'a>,
        memory_manager: *mut MemoryManager,
        event_tracer: *mut dyn EventTracer,
        external_data_map: *const dyn NamedDataMap,
        backend_options: *const LoadBackendOptionsMap,
        kernel_registry: Span<Kernel>,
    ) -> Result<Method<'a>> {
        let mut temp_allocator: *mut dyn MemoryAllocatorBase =
            unsafe { (*memory_manager).temp_allocator() };
        if temp_allocator.is_null() {
            let platform_allocator: *mut PlatformMemoryAllocator =
                unsafe { &mut *(*memory_manager).method_allocator() }
                    .allocate_instance::<PlatformMemoryAllocator>(core::mem::align_of::<
                        PlatformMemoryAllocator,
                    >());
            if platform_allocator.is_null() {
                crate::et_log!(Error, "Failed to allocate PlatformMemoryAllocator");
                return Err(Error::MemoryAllocationFailed);
            }
            unsafe {
                core::ptr::write(platform_allocator, PlatformMemoryAllocator::new());
            }
            temp_allocator = platform_allocator;
        }
        let mut method = Method::new(
            program as *const Program<'a>,
            memory_manager,
            event_tracer,
            temp_allocator,
            kernel_registry,
        );
        crate::et_log!(
            Debug,
            "Loading method: {}.",
            cstr_lossy(s_plan.name().unwrap().as_ptr() as *const core::ffi::c_char)
        );
        let err = method.init(s_plan, external_data_map, backend_options);
        if err != Error::Ok {
            Err(err)
        } else {
            et_check_msg!(method.initialized());
            Ok(method)
        }
    }

    // [spec:et:def:method.executorch.et-runtime-namespace.method.init-fn]
    // [spec:et:sem:method.executorch.et-runtime-namespace.method.init-fn]
    fn init(
        &mut self,
        s_plan: executorch_flatbuffer::ExecutionPlan<'a>,
        external_data_map: *const dyn NamedDataMap,
        backend_options: *const LoadBackendOptionsMap,
    ) -> Error {
        let _event_tracer_scope =
            EventTracerProfileMethodScope::new(self.event_tracer_, c"Method::init".as_ptr());
        crate::et_check_or_return_error!(
            // Don't use !initialized() here because we also want to fail on the
            // InitializationFailed state.
            self.init_state_ == InitializationState::Uninitialized,
            InvalidState,
            "Method already initialized, or previously failed to initialize."
        );
        self.init_state_ = InitializationState::InitializationFailed; // Until proven otherwise
        self.serialization_plan_ = Some(s_plan);
        let method_allocator: *mut dyn MemoryAllocatorBase =
            unsafe { (*self.memory_manager_).method_allocator() };

        {
            // Parse the elements of the values_ array.
            let err = self.parse_values(external_data_map);
            if err != Error::Ok {
                return err;
            }
        }

        {
            // Resolve delegates
            let delegates_opt = self.serialization_plan_.unwrap().delegates();
            crate::et_check_or_return_error!(
                delegates_opt.is_some(),
                InvalidProgram,
                "Missing delegates field"
            );
            let delegates = delegates_opt.unwrap();
            let n_delegate = delegates.len();
            self.delegates_ = unsafe { &mut *method_allocator }.allocate_list::<BackendDelegate>(
                n_delegate,
                core::mem::align_of::<BackendDelegate>(),
            );
            if self.delegates_.is_null() {
                crate::et_log!(
                    Error,
                    "Failed to allocate delegates array of size {}",
                    n_delegate
                );
                return Error::MemoryAllocationFailed;
            }

            // Get PTE data map, if it exists.
            //
            // PORT-NOTE: `get_named_data_map()` returns `*const (dyn NamedDataMap
            // + 'a)` (the PteDataMap borrows the Program buffer). The
            // `MergedDataMap::load`/`BackendInitContext::new` APIs and the local
            // `named_data_map` are typed `*const dyn NamedDataMap` (= `+ 'static`),
            // matching the C++ raw `const NamedDataMap*` with no lifetime. Erase
            // the `'a` bound here; sound because the Program (and its buffer)
            // outlives this Method, which is what the C++ relies on.
            let pte_data_map = unsafe { (*self.program_).get_named_data_map() };
            crate::et_check_or_return_error!(
                ResultExt::ok(&pte_data_map) || ResultExt::error(&pte_data_map) == Error::NotFound,
                InvalidProgram,
                "Failed to get named data map from program: 0x{:x}",
                ResultExt::error(&pte_data_map) as u32
            );
            let pte_data_map_ptr: *const dyn NamedDataMap = if ResultExt::ok(&pte_data_map) {
                unsafe { erase_named_data_map_lifetime(*ResultExt::get(&pte_data_map)) }
            } else {
                null_named_data_map()
            };

            let mut named_data_map: *const dyn NamedDataMap = null_named_data_map();
            if !is_null_named_data_map(external_data_map) && ResultExt::ok(&pte_data_map) {
                // Merge external_data_map and pte_data_map if both are present.
                let merged = MergedDataMap::load(external_data_map, pte_data_map_ptr);
                if !ResultExt::ok(&merged) {
                    return ResultExt::error(&merged);
                }
                // Allocate memory for the merged data map.
                self.merged_data_map_ = unsafe { &mut *method_allocator }
                    .allocate_instance::<MergedDataMap>(core::mem::align_of::<MergedDataMap>());
                if self.merged_data_map_.is_null() {
                    crate::et_log!(Error, "Failed to allocate MergedDataMap");
                    return Error::MemoryAllocationFailed;
                }
                // PORT-NOTE (WAVE-2 FIX): move the (already Ok-checked) merged map
                // out via `unwrap`; the old `mem::replace(..., unreachable_*())`
                // panicked because the sentinel is eagerly evaluated.
                unsafe {
                    core::ptr::write(self.merged_data_map_, merged.unwrap());
                }
                named_data_map = self.merged_data_map_ as *const dyn NamedDataMap;
            } else if !is_null_named_data_map(external_data_map) {
                named_data_map = external_data_map;
            } else if ResultExt::ok(&pte_data_map) {
                named_data_map = pte_data_map_ptr;
            }

            // n_delegate_ counts the number of successfully-initialized
            // delegates for ~Method() to clean up, and is incremented at the
            // bottom of the loop. This makes it safe for errors to return
            // without updating any state.
            self.n_delegate_ = 0;

            for i in 0..n_delegate {
                let delegate = delegates.get(i);

                // Get per-delegate runtime specs from the LoadBackendOptionsMap
                // if provided
                let mut delegate_runtime_specs: Span<
                    crate::runtime::backend::options::BackendOption,
                > = Span::new();
                if !backend_options.is_null() && delegate.id().is_some() {
                    delegate_runtime_specs = unsafe {
                        (*backend_options).get_options(
                            delegate.id().unwrap().as_ptr() as *const core::ffi::c_char
                        )
                    };
                }

                let mut backend_init_context = BackendInitContext::new(
                    method_allocator,
                    self.event_tracer_,
                    self.serialization_plan_.unwrap().name().unwrap().as_ptr()
                        as *const core::ffi::c_char,
                    named_data_map,
                    delegate_runtime_specs,
                );
                let err = BackendDelegate::init(
                    &delegate,
                    self.program_,
                    &mut backend_init_context,
                    unsafe { self.delegates_.add(i) },
                );
                if err != Error::Ok {
                    return err;
                }
                // ~Method() will try to clean up n_delegate_ entries in the
                // delegates_ array. Only increment this once we know the entry
                // is valid, so that we don't try to clean up an uninitialized
                // entry.
                self.n_delegate_ = i + 1;
            }
        }

        {
            // Load chains
            let chains_opt = self.serialization_plan_.unwrap().chains();
            crate::et_check_or_return_error!(
                chains_opt.is_some() && chains_opt.unwrap().len() > 0,
                InvalidProgram,
                "No chains"
            );
            let chains = chains_opt.unwrap();
            self.n_chains_ = chains.len();
            self.chains_ = unsafe { &mut *method_allocator }
                .allocate_list::<Chain<'a>>(self.n_chains_, core::mem::align_of::<Chain<'a>>());
            if self.chains_.is_null() {
                crate::et_log!(
                    Error,
                    "Failed to allocate chains array of size {}",
                    self.n_chains_
                );
                return Error::MemoryAllocationFailed;
            }

            // Try resolving all operators before failing, to make it easier to
            // debug multiple problems at once.
            let mut delayed_error = Error::Ok;
            let mut num_instructions_missing_op: i32 = 0;
            for i in 0..self.n_chains_ {
                let s_chain = chains.get(i);
                let s_instructions_opt = s_chain.instructions();
                crate::et_check_or_return_error!(
                    s_instructions_opt.is_some(),
                    InvalidProgram,
                    "Missing instructions in chain {}",
                    i
                );
                let s_instructions = s_instructions_opt.unwrap();
                let num_instructions = s_instructions.len();
                let chain_instruction_kernels: *mut Option<OpFunction> =
                    unsafe { &mut *method_allocator }.allocate_list::<Option<OpFunction>>(
                        num_instructions,
                        core::mem::align_of::<Option<OpFunction>>(),
                    );
                if chain_instruction_kernels.is_null() {
                    crate::et_log!(
                        Error,
                        "Failed to allocate instruction kernels for chain {}",
                        i
                    );
                    return Error::MemoryAllocationFailed;
                }
                let chain_instruction_arg_lists: *mut InstructionArgs<'a> =
                    unsafe { &mut *method_allocator }.allocate_list::<InstructionArgs<'a>>(
                        num_instructions,
                        core::mem::align_of::<InstructionArgs<'a>>(),
                    );
                if chain_instruction_arg_lists.is_null() {
                    crate::et_log!(
                        Error,
                        "Failed to allocate instruction arg lists for chain {}",
                        i
                    );
                    return Error::MemoryAllocationFailed;
                }

                // Set up the argument lists ahead of time and store pointers to
                // them to use when the instructions are called
                for instr_idx in 0..s_instructions.len() {
                    let instruction = s_instructions.get(instr_idx);
                    // Ensure that the `instr_args_as_X()` calls will return
                    // non-null.
                    crate::et_check_or_return_error!(
                        instruction.instr_args().is_some(),
                        InvalidProgram,
                        "Null instruction at index {}",
                        instr_idx
                    );

                    match instruction.instr_args_type() {
                        executorch_flatbuffer::InstructionArguments::KernelCall => {
                            let instr_args_as_kernel_call =
                                instruction.instr_args_as_kernel_call().unwrap();
                            let arg_idxs_opt = instr_args_as_kernel_call.args();
                            crate::et_check_or_return_error!(
                                arg_idxs_opt.is_some(),
                                InvalidProgram,
                                "KernelCall args missing"
                            );
                            let arg_idxs = arg_idxs_opt.unwrap();
                            let res = gen_instruction_arguments(
                                method_allocator,
                                self.n_value_,
                                self.values_,
                                arg_idxs.len(),
                                &arg_idxs,
                            );
                            if !ResultExt::ok(&res) {
                                return ResultExt::error(&res);
                            }
                            unsafe {
                                *chain_instruction_arg_lists.add(instr_idx) = *ResultExt::get(&res);
                            }
                            let err = self.resolve_operator(
                                instr_args_as_kernel_call.op_index(),
                                chain_instruction_kernels,
                                instr_idx,
                                *ResultExt::get(&res),
                                arg_idxs.len(),
                            );
                            if err == Error::OperatorMissing {
                                num_instructions_missing_op += 1;
                            } else if err == Error::MemoryAllocationFailed {
                                return err;
                            } else {
                                delayed_error = err;
                            }
                        }
                        executorch_flatbuffer::InstructionArguments::DelegateCall => {
                            let arg_idxs_opt =
                                instruction.instr_args_as_delegate_call().unwrap().args();
                            crate::et_check_or_return_error!(
                                arg_idxs_opt.is_some(),
                                InvalidProgram,
                                "DelegateCall args missing"
                            );
                            let arg_idxs = arg_idxs_opt.unwrap();
                            let res = gen_instruction_arguments(
                                method_allocator,
                                self.n_value_,
                                self.values_,
                                arg_idxs.len(),
                                &arg_idxs,
                            );
                            if !ResultExt::ok(&res) {
                                return ResultExt::error(&res);
                            }
                            unsafe {
                                *chain_instruction_arg_lists.add(instr_idx) = *ResultExt::get(&res);
                            }
                        }
                        executorch_flatbuffer::InstructionArguments::JumpFalseCall => {
                            let index = instruction
                                .instr_args_as_jump_false_call()
                                .unwrap()
                                .cond_value_index();
                            et_check_valid_value_index!(index, self.n_value_);
                            unsafe {
                                *chain_instruction_arg_lists.add(instr_idx) = Span::new();
                            }
                        }
                        executorch_flatbuffer::InstructionArguments::MoveCall => {
                            let move_call = instruction.instr_args_as_move_call().unwrap();
                            et_check_valid_value_index!(move_call.move_from(), self.n_value_);
                            et_check_valid_value_index!(move_call.move_to(), self.n_value_);
                            unsafe {
                                *chain_instruction_arg_lists.add(instr_idx) = Span::new();
                            }
                        }
                        executorch_flatbuffer::InstructionArguments::FreeCall => {
                            let index =
                                instruction.instr_args_as_free_call().unwrap().value_index();
                            et_check_valid_value_index!(index, self.n_value_);
                            unsafe {
                                *chain_instruction_arg_lists.add(instr_idx) = Span::new();
                            }
                        }
                        other => {
                            crate::et_log!(Error, "Invalid instruction type {}", other.0);
                            return Error::InvalidProgram;
                        }
                    }
                }
                unsafe {
                    core::ptr::write(
                        self.chains_.add(i),
                        Chain {
                            s_chain_: s_chain,
                            argument_lists_: Span::from_raw_parts(
                                chain_instruction_arg_lists,
                                num_instructions,
                            ),
                            kernels_: chain_instruction_kernels,
                        },
                    );
                }
            }
            crate::et_check_or_return_error!(
                num_instructions_missing_op == 0,
                OperatorMissing,
                "There are {} instructions don't have corresponding operator registered. See logs for details",
                num_instructions_missing_op as usize
            );
            if delayed_error != Error::Ok {
                return delayed_error;
            }
        }

        self.step_state_ = StepState {
            chain_idx: 0,
            instr_idx: 0,
        };

        self.init_state_ = InitializationState::Initialized;
        Error::Ok
    }

    // [spec:et:def:method.executorch.et-runtime-namespace.method.inputs-size-fn]
    // [spec:et:sem:method.executorch.et-runtime-namespace.method.inputs-size-fn]
    fn inputs_size(&self) -> usize {
        let inputs = self.serialization_plan_.unwrap().inputs();
        match inputs {
            None => 0,
            Some(v) => v.len(),
        }
    }

    // ==== PASS 2 (I/O plumbing). ====

    // [spec:et:def:method.executorch.et-runtime-namespace.method.set-input-fn]
    // [spec:et:sem:method.executorch.et-runtime-namespace.method.set-input-fn]
    #[must_use]
    pub fn set_input(&mut self, input_evalue: &EValue<'a>, input_idx: usize) -> Error {
        crate::et_check_or_return_error!(
            self.initialized(),
            InvalidState,
            "Input can not be set until method has been initialized."
        );

        crate::et_check_or_return_error!(
            self.step_state_.instr_idx == 0 && self.step_state_.chain_idx == 0,
            InvalidState,
            "Inputs can not be set mid execution."
        );

        crate::et_check_or_return_error!(
            input_idx < self.inputs_size(),
            InvalidArgument,
            "Input index ({}) must be less than the number of inputs in method ({}).",
            input_idx,
            self.inputs_size()
        );

        let e_idx = self.get_input_index(input_idx);
        let e = self.get_value(e_idx);

        if !(e.is_none() || e.is_tensor() || e.is_scalar() || e.is_string()) {
            crate::et_log!(
                Error,
                "Input {} was expected to be a Tensor or primitive but was {}.",
                input_idx,
                tag_to_string_static(e.tag)
            );
            return Error::InvalidArgument;
        }

        if e.tag != input_evalue.tag {
            crate::et_log!(
                Error,
                "Input {} was expected to have type {} but was {}.",
                input_idx,
                tag_to_string_static(e.tag),
                tag_to_string_static(input_evalue.tag)
            );
            return Error::InvalidArgument;
        }

        if e.is_none() {
            // no-op
        } else if e.is_tensor() {
            let t_dst = e.to_tensor();
            let t_src = input_evalue.to_tensor();

            crate::et_check_or_return_error!(
                t_dst.scalar_type() == t_src.scalar_type(),
                InvalidArgument,
                "Input {} has unexpected scalar type: expected {} but was {}.",
                input_idx,
                to_string(t_dst.scalar_type()),
                to_string(t_src.scalar_type())
            );

            let mut numel: ssize_t = 1;
            for i in 0..t_src.dim() {
                let overflow = mul_overflows_ssize(numel, t_src.size(i), &mut numel);
                crate::et_check_or_return_error!(
                    !overflow,
                    InvalidArgument,
                    "Input {}: numel overflowed at dimension {} with size {}",
                    input_idx,
                    i as usize,
                    t_src.size(i) as usize
                );
            }
            let mut nbytes: usize = 0;
            let nbytes_overflow = mul_overflows_usize(
                numel as usize,
                element_size(t_src.scalar_type()),
                &mut nbytes,
            );
            crate::et_check_or_return_error!(
                !nbytes_overflow,
                InvalidArgument,
                "Input {}: nbytes overflowed: numel {} with element size {}",
                input_idx,
                numel,
                element_size(t_src.scalar_type())
            );
            // Reset the shape for the Method's input as the size of forwarded
            // input tensor for shape dynamism. Also is a safety check if need
            // memcpy.
            let resize_err = resize_tensor_same_type(t_dst, t_src.sizes());
            crate::et_check_ok_or_return_error!(
                resize_err,
                "Error resizing tensor at input {}",
                input_idx
            );
            let tensor_meta = self.method_meta().input_tensor_meta(input_idx);
            if ResultExt::get(&tensor_meta).is_memory_planned() {
                crate::et_check_ok_or_return_error!(
                    internal::copy_tensor_data(t_dst, t_src),
                    "Error copying tensor data at input {}",
                    input_idx
                );
            } else {
                crate::et_check_ok_or_return_error!(
                    internal::share_tensor_data(t_dst, t_src),
                    "Error sharing tensor data at input {}",
                    input_idx
                );
            }
            // Prims have to be the same as what was traced
        } else if e.is_int() {
            crate::et_check_or_return_error!(
                e.to_int() == input_evalue.to_int(),
                InvalidArgument,
                "The {}-th input of method should have the same value as the input_evalue, but got {} and {}",
                input_idx,
                e.to_int(),
                input_evalue.to_int()
            );
        } else if e.is_bool() {
            crate::et_check_or_return_error!(
                e.to_bool() == input_evalue.to_bool(),
                InvalidArgument,
                "The {}-th input of method should have the same value as the input_evalue, but got {} and {}",
                input_idx,
                e.to_bool() as i64,
                input_evalue.to_bool() as i64
            );
        } else if e.is_double() {
            let lhs = input_evalue.to_double();
            let rhs = e.to_double();
            let atol = 1e-4;
            let rtol = 1e-5;
            let mut is_equal = true;
            if lhs.is_nan() && rhs.is_nan() {
                // NaN == NaN
            } else if !lhs.is_finite() && !rhs.is_finite() && ((lhs > 0.0) == (rhs > 0.0)) {
                // -Inf == -Inf
                // +Inf == +Inf
            } else {
                let allowed_error = atol + (rtol * rhs).abs();
                let actual_error = (lhs - rhs).abs();
                if !actual_error.is_finite() || actual_error > allowed_error {
                    is_equal = false;
                }
            }
            crate::et_check_or_return_error!(
                is_equal,
                InvalidArgument,
                "The {}-th input of method should have the same value as the input_evalue, but get {} and {}",
                input_idx,
                lhs,
                rhs
            );
        } else if e.is_string() {
            crate::et_check_or_return_error!(
                e.to_string() == input_evalue.to_string(),
                InvalidArgument,
                "The {}-th input of method should have the same value as the input_evalue, but get {} and {}",
                input_idx,
                e.to_string(),
                input_evalue.to_string()
            );
        } else {
            crate::et_log!(
                Error,
                "Unsupported input type: {}",
                tag_to_string_static(e.tag)
            );
            return Error::InvalidArgument;
        }
        unsafe {
            *self.input_set_.add(input_idx) = true;
        }

        Error::Ok
    }

    // [spec:et:def:method.executorch.method.set-inputs-fn]
    // [spec:et:sem:method.executorch.method.set-inputs-fn]
    // [spec:et:def:method.executorch.et-runtime-namespace.method.set-inputs-fn]
    // [spec:et:sem:method.executorch.et-runtime-namespace.method.set-inputs-fn]
    // [spec:et:sem:method.executorch.method.set-inputs-fn]
    // [spec:et:def:method.method.set-inputs-fn]
    // [spec:et:sem:method.method.set-inputs-fn]
    #[must_use]
    pub fn set_inputs(&mut self, input_evalues: ArrayRef<EValue<'a>>) -> Error {
        let n_input = self.inputs_size();
        crate::et_check_or_return_error!(
            input_evalues.size() == n_input,
            InvalidArgument,
            "Invalid number of inputs provided. Expected {}, but got {}",
            n_input,
            input_evalues.size()
        );
        for i in 0..n_input {
            crate::et_check_ok_or_return_error!(
                self.set_input(unsafe { input_evalues.index(i) }, i)
            );
        }
        Error::Ok
    }

    // [spec:et:def:method.executorch.method.set-output-data-ptr-fn]
    // [spec:et:sem:method.executorch.method.set-output-data-ptr-fn]
    // [spec:et:def:method.executorch.et-runtime-namespace.method.set-output-data-ptr-fn]
    // [spec:et:sem:method.executorch.et-runtime-namespace.method.set-output-data-ptr-fn]
    // [spec:et:sem:method.executorch.method.set-output-data-ptr-fn]
    // [spec:et:def:method.method.set-output-data-ptr-fn]
    // [spec:et:sem:method.method.set-output-data-ptr-fn]
    #[must_use]
    pub fn set_output_data_ptr(
        &mut self,
        buffer: *mut core::ffi::c_void,
        size: usize,
        output_idx: usize,
    ) -> Error {
        // Check method state
        crate::et_check_or_return_error!(
            self.initialized(),
            InvalidState,
            "Outputs can not be retrieved until method has been initialized."
        );

        // Check the args
        crate::et_check_or_return_error!(
            output_idx < self.outputs_size(),
            InvalidArgument,
            "output_idx: {} > num_outputs: {}",
            output_idx,
            self.outputs_size()
        );

        let out_idx = self.get_output_index(output_idx);
        let output = self.mutable_value(out_idx);
        if !output.is_tensor() {
            crate::et_log!(
                Error,
                "Output type: {} is not a tensor.",
                tag_to_string_static(output.tag)
            );
            return Error::InvalidArgument;
        }

        let tensor_meta = self.method_meta().output_tensor_meta(output_idx);
        if ResultExt::get(&tensor_meta).is_memory_planned() {
            crate::et_log!(
                Error,
                "Output {} is memory planned, or is a constant. Cannot override the existing data pointer.",
                output_idx
            );
            return Error::InvalidState;
        }

        let out_idx = self.get_output_index(output_idx);
        let output = self.mutable_value(out_idx);
        let t = output.to_tensor();
        if !output.is_tensor() {
            crate::et_log!(
                Error,
                "output type: {} is not a tensor.",
                tag_to_string_static(output.tag)
            );
            return Error::InvalidArgument;
        }

        crate::et_check_or_return_error!(
            t.nbytes() <= size,
            InvalidArgument,
            "buffer size: {} is smaller then expected tensor size: {}",
            size,
            t.nbytes()
        );

        // Set data
        internal::set_tensor_data(t, buffer, size)
    }

    // The public header declaration and the out-of-line definition collapse onto
    // this one Rust fn.
    // [spec:et:def:method.executorch.et-runtime-namespace.method.get-outputs-fn]
    // [spec:et:sem:method.executorch.et-runtime-namespace.method.get-outputs-fn]
    // [spec:et:def:method.method.get-outputs-fn]
    // [spec:et:sem:method.method.get-outputs-fn]
    #[must_use]
    pub fn get_outputs(&self, output_evalues: *mut EValue<'a>, length: usize) -> Error {
        crate::et_check_or_return_error!(
            self.initialized(),
            InvalidState,
            "Outputs can not be retrieved until method has been initialized."
        );
        let n_output = self.outputs_size();
        crate::et_check_or_return_error!(
            length >= n_output,
            InvalidArgument,
            "The given array is not large enough to hold all outputs."
        );
        for i in 0..n_output {
            let mut out = self.get_output(i);
            unsafe {
                (*output_evalues.add(i)).assign_move(ResultExt::get_mut(&mut out));
            }
        }
        for i in n_output..length {
            let mut none = EValue::new();
            unsafe {
                (*output_evalues.add(i)).assign_move(&mut none);
            }
        }
        Error::Ok
    }

    // The public header declaration and the out-of-line definition collapse onto
    // this one Rust fn.
    // [spec:et:def:method.executorch.et-runtime-namespace.method.get-inputs-fn]
    // [spec:et:sem:method.executorch.et-runtime-namespace.method.get-inputs-fn]
    // [spec:et:def:method.method.get-inputs-fn]
    // [spec:et:sem:method.method.get-inputs-fn]
    #[must_use]
    pub fn get_inputs(&mut self, input_evalues: *mut EValue<'a>, length: usize) -> Error {
        crate::et_check_or_return_error!(
            self.initialized(),
            InvalidState,
            "Inputs can not be retrieved until method has been initialized."
        );
        let n_input = self.inputs_size();
        crate::et_check_or_return_error!(
            length >= n_input,
            InvalidArgument,
            "The given array is not large enough to hold all inputs."
        );

        for i in 0..n_input {
            let idx = self.get_input_index(i);
            let mut src = EValue::from_ref(self.get_value(idx));
            unsafe {
                (*input_evalues.add(i)).assign_move(&mut src);
            }
            // Accessing inputs this way is deprecated.
            // We assume the users to be responsible to set the inputs they get.
            unsafe {
                *self.input_set_.add(i) = true;
            }
        }
        for i in n_input..length {
            let mut none = EValue::new();
            unsafe {
                (*input_evalues.add(i)).assign_move(&mut none);
            }
        }
        Error::Ok
    }

    // [spec:et:def:method.executorch.et-runtime-namespace.method.get-attribute-fn]
    // [spec:et:sem:method.executorch.et-runtime-namespace.method.get-attribute-fn]
    #[must_use]
    pub fn get_attribute(&mut self, name: &str) -> Result<Tensor<'a>> {
        let flatbuffer_values = self.serialization_plan_.unwrap().values().unwrap();
        let mut counter: usize = 0;

        for i in 0..flatbuffer_values.len() {
            let serialization_value = flatbuffer_values.get(i);
            if serialization_value.val_type() == executorch_flatbuffer::KernelTypes::Tensor {
                let s_tensor = serialization_value.val_as_tensor().unwrap();
                if s_tensor.extra_tensor_info().is_some()
                    && s_tensor
                        .extra_tensor_info()
                        .unwrap()
                        .fully_qualified_name()
                        .is_some()
                    && s_tensor
                        .extra_tensor_info()
                        .unwrap()
                        .fully_qualified_name()
                        .unwrap()
                        == name
                {
                    if !unsafe { (*self.values_.add(counter)).is_tensor() } {
                        crate::et_log!(
                            Error,
                            "Attribute tensor not at the expected location. The .pte is likely malformed. Please file a bug report on https://github.com/pytorch/executorch/issues"
                        );
                        return Err(Error::Internal);
                    }
                    // PORT-NOTE: C++ returns `values_[counter].toTensor()` by
                    // value — a shallow copy of the non-owning Tensor (copies
                    // the impl pointer). Reproduce with `Tensor::new` over the
                    // same impl.
                    return Ok(Tensor::new(unsafe {
                        (*self.values_.add(counter))
                            .to_tensor()
                            .unsafe_get_tensor_impl()
                    }));
                }
            }
            counter += 1;
        }

        Err(Error::NotFound)
    }

    // [spec:et:def:method.executorch.et-runtime-namespace.method.execute-fn]
    // [spec:et:sem:method.executorch.et-runtime-namespace.method.execute-fn]
    // [spec:et:def:method.method.execute-fn]
    // [spec:et:sem:method.method.execute-fn]
    #[must_use]
    pub fn execute(&mut self) -> Error {
        crate::runtime::core::event_tracer_hooks::event_tracer_create_event_block(
            self.event_tracer_,
            c"Execute".as_ptr(),
        );
        let event_tracer_entry =
            crate::runtime::core::event_tracer_hooks::event_tracer_begin_profiling_event(
                self.event_tracer_,
                c"Method::execute".as_ptr(),
            );
        let _scope_prof = ScopeProf::new(c"Method::execute".as_ptr());
        crate::et_check_or_return_error!(
            self.initialized(),
            NotSupported,
            "Cannot execute until method has been initialized."
        );
        crate::et_check_or_return_error!(
            !self.in_progress(),
            InvalidState,
            "Method execution is in progress"
        );
        let n_input = self.inputs_size();
        for i in 0..n_input {
            crate::et_check_or_return_error!(
                unsafe { *self.input_set_.add(i) },
                InvalidArgument,
                "Input {} has not been set.",
                i
            );
        }
        crate::et_log!(Debug, "Executing method: {}.", self.method_meta().name());
        if !self.temp_allocator_.is_null() {
            unsafe {
                (*self.temp_allocator_).reset();
            }
        }

        // Chains are executed sequentially today, but future async designs may
        // branch and run many in parallel or out of order.
        let mut instruction_count: usize = 0;
        self.step_state_.chain_idx = 0;
        while self.step_state_.chain_idx < self.n_chains_ {
            let chain: *mut Chain<'a> = unsafe { self.chains_.add(self.step_state_.chain_idx) };
            let instructions = unsafe { (*chain).s_chain_.instructions() };
            crate::et_check_or_return_error!(
                instructions.is_some(),
                Internal,
                "chain {} has no instructions field",
                self.step_state_.chain_idx
            );

            // Loop over instructions
            self.step_state_.instr_idx = 0;
            while self.step_state_.instr_idx
                < unsafe { (*chain).s_chain_.instructions().unwrap().len() }
            {
                if instruction_count >= K_MAX_INSTRUCTIONS {
                    crate::et_log!(
                        Error,
                        "Instruction execution limit ({}) exceeded at chain {}, instruction {}. Possible infinite loop detected. If this is a legitimate large model, raise the limit by rebuilding with -DET_MAX_INSTRUCTIONS=<value>.",
                        K_MAX_INSTRUCTIONS,
                        self.step_state_.chain_idx,
                        self.step_state_.instr_idx
                    );
                    self.step_state_ = StepState {
                        chain_idx: 0,
                        instr_idx: 0,
                    };
                    return Error::InvalidProgram;
                }
                instruction_count += 1;
                let _event_tracer_instr_scope = EventTracerProfileInstructionScope::new(
                    self.event_tracer_,
                    self.step_state_.chain_idx as i32,
                    self.step_state_.instr_idx as u32,
                );
                let status = self.execute_instruction();
                if status != Error::Ok {
                    self.step_state_ = StepState {
                        chain_idx: 0,
                        instr_idx: 0,
                    };
                    return status;
                }
            }
            self.step_state_.chain_idx += 1;
        }
        crate::runtime::core::event_tracer_hooks::event_tracer_end_profiling_event(
            self.event_tracer_,
            event_tracer_entry,
        );
        self.log_outputs();

        // TODO(jakeszwe, dbort): Decide on calling execute back to back without
        // going through the reset api first.
        self.reset_execution()
    }

    // [spec:et:def:method.executorch.et-runtime-namespace.method.step-fn]
    // [spec:et:sem:method.executorch.et-runtime-namespace.method.step-fn]
    // [spec:et:def:method.method.step-fn]
    // [spec:et:sem:method.method.step-fn]
    #[must_use]
    pub fn step(&mut self) -> Error {
        let _event_tracer_instr_scope = EventTracerProfileInstructionScope::new(
            self.event_tracer_,
            self.step_state_.chain_idx as i32,
            self.step_state_.instr_idx as u32,
        );
        let _scope_prof = ScopeProf::new(c"Method::step".as_ptr());
        let event_tracer_entry =
            crate::runtime::core::event_tracer_hooks::event_tracer_begin_profiling_event(
                self.event_tracer_,
                c"Method::step".as_ptr(),
            );
        crate::et_check_or_return_error!(
            self.initialized(),
            InvalidState,
            "Cannot execute until method has been initialized."
        );

        // If chain_step_ is on n_chains_, then we have no instructions run.
        if self.step_state_.chain_idx == self.n_chains_ {
            return Error::EndOfMethod;
        }

        let num_instructions = unsafe {
            (*self.chains_.add(self.step_state_.chain_idx))
                .s_chain_
                .instructions()
                .unwrap()
                .len()
        };

        // Special case chains with no instructions. These appear for example in a
        // model that just returns the input/a constant.
        if num_instructions == 0 {
            self.step_state_.chain_idx += 1;
            return Error::Ok;
        }

        let status = self.execute_instruction();
        if status != Error::Ok {
            return status;
        }

        crate::runtime::core::event_tracer_hooks::event_tracer_end_profiling_event(
            self.event_tracer_,
            event_tracer_entry,
        );
        // end of the current chain, advance to the next chain
        if self.step_state_.instr_idx == num_instructions {
            self.step_state_.instr_idx = 0;
            self.step_state_.chain_idx += 1;
            self.log_outputs();
        }
        Error::Ok
    }

    // [spec:et:def:method.executorch.et-runtime-namespace.method.experimental-step-fn]
    // [spec:et:sem:method.executorch.et-runtime-namespace.method.experimental-step-fn]
    // [spec:et:def:method.method.experimental-step-fn]
    // [spec:et:sem:method.method.experimental-step-fn]
    #[must_use]
    pub fn experimental_step(&mut self) -> Error {
        self.step()
    }

    // [spec:et:def:method.executorch.et-runtime-namespace.method.in-progress-fn]
    // [spec:et:sem:method.executorch.et-runtime-namespace.method.in-progress-fn]
    // [spec:et:def:method.method.in-progress-fn]
    // [spec:et:sem:method.method.in-progress-fn]
    #[must_use]
    pub fn in_progress(&self) -> bool {
        (self.step_state_.chain_idx != 0 || self.step_state_.instr_idx != 0)
            && self.step_state_.chain_idx < self.n_chains_
    }

    // [spec:et:def:method.executorch.et-runtime-namespace.method.reset-execution-fn]
    // [spec:et:sem:method.executorch.et-runtime-namespace.method.reset-execution-fn]
    // [spec:et:def:method.method.reset-execution-fn]
    // [spec:et:sem:method.method.reset-execution-fn]
    #[must_use]
    pub fn reset_execution(&mut self) -> Error {
        crate::et_check_or_return_error!(
            self.step_state_.chain_idx == self.n_chains_,
            InvalidState,
            "Cannot reset until EndOfMethod has been reached."
        );
        self.step_state_ = StepState {
            chain_idx: 0,
            instr_idx: 0,
        };
        Error::Ok
    }

    // [spec:et:def:method.executorch.et-runtime-namespace.method.experimental-reset-execution-fn]
    // [spec:et:sem:method.executorch.et-runtime-namespace.method.experimental-reset-execution-fn]
    // [spec:et:def:method.method.experimental-reset-execution-fn]
    // [spec:et:sem:method.method.experimental-reset-execution-fn]
    #[must_use]
    pub fn experimental_reset_execution(&mut self) -> Error {
        self.reset_execution()
    }

    // [spec:et:def:method.executorch.et-runtime-namespace.method.method-meta-fn]
    // [spec:et:sem:method.executorch.et-runtime-namespace.method.method-meta-fn]
    // [spec:et:def:method.method.method-meta-fn]
    // [spec:et:sem:method.method.method-meta-fn]
    pub fn method_meta(&self) -> crate::runtime::executor::method_meta::MethodMeta<'a> {
        let name =
            self.serialization_plan_.unwrap().name().unwrap().as_ptr() as *const core::ffi::c_char;
        let method_meta = unsafe { (*self.program_).method_meta(core::ffi::CStr::from_ptr(name)) };
        et_check_msg!(
            ResultExt::ok(&method_meta),
            "Internal error: method_meta({}) returned 0x{:x}",
            cstr_lossy(name),
            ResultExt::error(&method_meta) as u32
        );
        // PORT-NOTE (WAVE-2 FIX): move the (et_check_msg-guarded) MethodMeta out
        // via `unwrap`; the old `mem::replace(..., unreachable_*())` panicked
        // because the sentinel is eagerly evaluated.
        method_meta.unwrap()
    }

    // [spec:et:def:method.executorch.et-runtime-namespace.method.outputs-size-fn]
    // [spec:et:sem:method.executorch.et-runtime-namespace.method.outputs-size-fn]
    // [spec:et:def:method.method.outputs-size-fn]
    // [spec:et:sem:method.method.outputs-size-fn]
    #[must_use]
    pub fn outputs_size(&self) -> usize {
        let outputs = self.serialization_plan_.unwrap().outputs();
        match outputs {
            None => 0,
            Some(v) => v.len(),
        }
    }

    // [spec:et:def:method.executorch.et-runtime-namespace.method.get-output-fn]
    // [spec:et:sem:method.executorch.et-runtime-namespace.method.get-output-fn]
    //
    // PORT-NOTE: the C++ returns `const EValue&`; the Rust surface returns a
    // shallow-copied `EValue` (via `EValue::from_ref`, the copy ctor) so callers
    // do not have to thread the Method's borrow. Semantically identical to
    // reading `get_value(get_output_index(i))` — the copy aliases the same
    // tensor impl, matching the C++ shallow-copy warning.
    #[must_use]
    pub fn get_output(&self, i: usize) -> Result<EValue<'a>> {
        Ok(EValue::from_ref(self.get_value(self.get_output_index(i))))
    }

    // [spec:et:def:method.executorch.et-runtime-namespace.method.get-event-tracer-fn]
    // [spec:et:sem:method.executorch.et-runtime-namespace.method.get-event-tracer-fn]
    pub fn get_event_tracer(&mut self) -> *mut dyn EventTracer {
        self.event_tracer_
    }

    // [spec:et:def:method.executorch.et-runtime-namespace.method.get-input-fn]
    // [spec:et:sem:method.executorch.et-runtime-namespace.method.get-input-fn]
    pub fn get_input(&self, i: usize) -> &EValue<'a> {
        // Accessing inputs this way is deprecated.
        // We assume the users to be responsible to set the inputs they get.
        //
        // PORT-NOTE: the C++ accessor is `const` but mutates `input_set_[i]`
        // through the raw `bool*`. Reproduced with a raw-pointer write behind
        // `&self`, matching the C++ aliasing.
        unsafe {
            *self.input_set_.add(i) = true;
        }
        self.get_value(self.get_input_index(i))
    }

    // [spec:et:def:method.executorch.et-runtime-namespace.method.mutable-input-fn]
    // [spec:et:sem:method.executorch.et-runtime-namespace.method.mutable-input-fn]
    pub fn mutable_input(&mut self, i: usize) -> &mut EValue<'a> {
        // Accessing inputs this way is deprecated.
        // We assume the users to be responsible to set the inputs they get.
        unsafe {
            *self.input_set_.add(i) = true;
        }
        let idx = self.get_input_index(i);
        self.mutable_value(idx)
    }

    // [spec:et:def:method.executorch.et-runtime-namespace.method.mutable-output-fn]
    // [spec:et:sem:method.executorch.et-runtime-namespace.method.mutable-output-fn]
    pub fn mutable_output(&mut self, i: usize) -> &mut EValue<'a> {
        let idx = self.get_output_index(i);
        self.mutable_value(idx)
    }

    // [spec:et:def:method.executorch.et-runtime-namespace.method.get-value-fn]
    // [spec:et:sem:method.executorch.et-runtime-namespace.method.get-value-fn]
    fn get_value(&self, i: usize) -> &EValue<'a> {
        // [spec:et:def:method.et-check-msg-fn]
        // [spec:et:sem:method.et-check-msg-fn]
        et_check_msg!(i < self.n_value_, "{} >= {}", i, self.n_value_);
        unsafe { &*self.values_.add(i) }
    }

    // [spec:et:def:method.executorch.et-runtime-namespace.method.mutable-value-fn]
    // [spec:et:sem:method.executorch.et-runtime-namespace.method.mutable-value-fn]
    fn mutable_value(&mut self, i: usize) -> &mut EValue<'a> {
        et_check_msg!(i < self.n_value_, "{} >= {}", i, self.n_value_);
        unsafe { &mut *self.values_.add(i) }
    }

    // [spec:et:def:method.executorch.et-runtime-namespace.method.get-input-index-fn]
    // [spec:et:sem:method.executorch.et-runtime-namespace.method.get-input-index-fn]
    fn get_input_index(&self, i: usize) -> usize {
        et_check_msg!(i < self.inputs_size(), "{} >= {}", i, self.inputs_size());
        self.serialization_plan_.unwrap().inputs().unwrap().get(i) as usize
    }

    // [spec:et:def:method.executorch.et-runtime-namespace.method.get-output-index-fn]
    // [spec:et:sem:method.executorch.et-runtime-namespace.method.get-output-index-fn]
    fn get_output_index(&self, i: usize) -> usize {
        et_check_msg!(i < self.outputs_size(), "{} >= {}", i, self.outputs_size());
        self.serialization_plan_.unwrap().outputs().unwrap().get(i) as usize
    }

    // [spec:et:def:method.executorch.et-runtime-namespace.method.execute-instruction-fn]
    // [spec:et:sem:method.executorch.et-runtime-namespace.method.execute-instruction-fn]
    // [spec:et:def:method.method.execute-instruction-fn]
    // [spec:et:sem:method.method.execute-instruction-fn]
    fn execute_instruction(&mut self) -> Error {
        // PORT-NOTE: `auto& chain = chains_[step_state_.chain_idx]`. `chains_` is a
        // raw `*mut Chain`; we operate through the pointer to avoid holding a
        // borrow of `self` across the `mutable_value`/`get_value` calls below.
        let chain: *mut Chain<'a> = unsafe { self.chains_.add(self.step_state_.chain_idx) };
        let instructions = unsafe { (*chain).s_chain_.instructions().unwrap() };

        crate::et_check_or_return_error!(
            self.step_state_.instr_idx < instructions.len(),
            Internal,
            "Instr index {} >= chain[{}] instr count {}",
            self.step_state_.instr_idx,
            self.step_state_.chain_idx,
            instructions.len()
        );

        let instruction = instructions.get(self.step_state_.instr_idx);
        let mut next_instr_idx = self.step_state_.instr_idx + 1;
        let mut err = Error::Ok;

        match instruction.instr_args_type() {
            executorch_flatbuffer::InstructionArguments::KernelCall => {
                let _scope_prof = ScopeProf::new(c"OPERATOR_CALL".as_ptr());
                let _event_tracer_op_scope =
                    EventTracerProfileOpScope::new(self.event_tracer_, c"OPERATOR_CALL".as_ptr());
                // TODO(T147221312): Also expose tensor resizer via the context.
                let mut context =
                    KernelRuntimeContext::new(self.event_tracer_, self.temp_allocator_);
                let args = unsafe { *(*chain).argument_lists_.index(self.step_state_.instr_idx) };
                let kernel =
                    unsafe { (*(*chain).kernels_.add(self.step_state_.instr_idx)).unwrap() };
                kernel(&mut context, args);
                // We reset the temp_allocator after the switch statement
                err = context.failure_state();
                if err != Error::Ok {
                    // We know that instr_args_as_KernelCall is non-null because it
                    // was checked at init time.
                    let op_index = instruction.instr_args_as_kernel_call().unwrap().op_index();
                    let op = self
                        .serialization_plan_
                        .unwrap()
                        .operators()
                        .unwrap()
                        .get(op_index as usize);
                    crate::et_log!(
                        Error,
                        "KernelCall failed at instruction {}:{} in operator {}.{}: 0x{:x}",
                        self.step_state_.chain_idx,
                        self.step_state_.instr_idx,
                        op.name().unwrap(),
                        op.overload().unwrap(),
                        err as u32
                    );
                    for i in 0..args.size() {
                        crate::et_log!(Error, "arg {} with type id {}", i as u32, unsafe {
                            (**args.index(i)).tag
                        }
                            as u32);
                    }
                    // TODO(T153804650): Consider logging the EValues to help with
                    // debugging. This is a failure path, and it doesn't matter if
                    // it's a little slow. Do the same for DelegateCall errors.
                }
            }
            executorch_flatbuffer::InstructionArguments::DelegateCall => {
                let _scope_prof = ScopeProf::new(c"DELEGATE_CALL".as_ptr());
                let _event_tracer_op_scope =
                    EventTracerProfileOpScope::new(self.event_tracer_, c"DELEGATE_CALL".as_ptr());
                // We know that instr_args_as_DelegateCall is non-null because it
                // was checked at init time.
                let delegate_idx = instruction
                    .instr_args_as_delegate_call()
                    .unwrap()
                    .delegate_index();
                crate::et_check_or_return_error!(
                    (delegate_idx as usize) < self.n_delegate_,
                    Internal,
                    "DELEGATE_CALL index {} >= num delegates {} at instruction {}",
                    delegate_idx as u32,
                    self.n_delegate_,
                    self.step_state_.instr_idx
                );
                let mut backend_execution_context =
                    crate::runtime::backend::backend_execution_context::BackendExecutionContext::new(
                        /*event_tracer=*/ self.event_tracer_,
                        /*temp_allocator=*/ self.temp_allocator_,
                        /*method_name=*/
                        self.serialization_plan_.unwrap().name().unwrap().as_ptr()
                            as *const core::ffi::c_char,
                    );
                err = unsafe {
                    (*self.delegates_.add(delegate_idx as usize)).execute(
                        &mut backend_execution_context,
                        *(*chain).argument_lists_.index(self.step_state_.instr_idx),
                    )
                };
                if err != Error::Ok {
                    crate::et_log!(
                        Error,
                        "CALL_DELEGATE execute failed at instruction {}: 0x{:x}",
                        self.step_state_.instr_idx,
                        err as u32
                    );
                }

                // Log all the arguments of the delegate call. Ideally we'd only
                // like to log the outputs of the delegate, but currently we cannot
                // know from the arguments which are the inputs and which are the
                // outputs, so we just log everything. This will be changed in the
                // future when the inputs and ouputs are separate lists.
                #[cfg(feature = "event-tracer")]
                {
                    let arg_list =
                        unsafe { *(*chain).argument_lists_.index(self.step_state_.instr_idx) };
                    for i in 0..arg_list.size() {
                        let arg: *mut EValue<'a> = unsafe { *arg_list.data().add(i) };
                        crate::runtime::core::event_tracer_hooks::event_tracer_log_evalue(
                            self.event_tracer_,
                            unsafe { &mut *arg },
                        );
                    }
                }
            }
            executorch_flatbuffer::InstructionArguments::JumpFalseCall => {
                let _scope_prof = ScopeProf::new(c"JF_CALL".as_ptr());
                let _event_tracer_op_scope =
                    EventTracerProfileOpScope::new(self.event_tracer_, c"JF_CALL".as_ptr());
                // We know that instr_args_as_JumpFalseCall is non-null because it
                // was checked at init time.
                let jf_call = instruction.instr_args_as_jump_false_call().unwrap();
                // We know that index is a valid values_ index because it was
                // checked at init time.
                let index = jf_call.cond_value_index();
                let jf_result = parse_cond_value(unsafe { &*self.values_.add(index as usize) });
                if ResultExt::ok(&jf_result) {
                    if !*ResultExt::get(&jf_result) {
                        next_instr_idx = jf_call.destination_instruction() as usize;
                    }
                } else {
                    err = ResultExt::error(&jf_result);
                }
            }
            executorch_flatbuffer::InstructionArguments::MoveCall => {
                let _scope_prof = ScopeProf::new(c"MOVE_CALL".as_ptr());
                let _event_tracer_op_scope =
                    EventTracerProfileOpScope::new(self.event_tracer_, c"MOVE_CALL".as_ptr());
                // We know that instr_args_as_MoveCall is non-null because it was
                // checked at init time.
                let move_call = instruction.instr_args_as_move_call().unwrap();
                // PORT-NOTE: `mutable_value(to) = get_value(from)` is EValue copy-
                // assignment. Both operands index into `values_`; done via raw
                // pointers to reproduce the C++ aliasing without a borrow clash.
                let to = self.get_value_index_checked(move_call.move_to());
                let from = self.get_value_index_checked(move_call.move_from());
                unsafe {
                    (*self.values_.add(to)).assign_ref(&*self.values_.add(from));
                }
            }
            executorch_flatbuffer::InstructionArguments::FreeCall => {
                let _scope_prof = ScopeProf::new(c"FREE_CALL".as_ptr());
                let _event_tracer_op_scope =
                    EventTracerProfileOpScope::new(self.event_tracer_, c"FREE_CALL".as_ptr());
                // We know that instr_args_as_FreeCall is non-null because it was
                // checked at init time.
                let free_call = instruction.instr_args_as_free_call().unwrap();
                let value_index = free_call.value_index();
                let t = self.mutable_value(value_index as usize).try_to_tensor();
                if !ResultExt::ok(&t) {
                    crate::et_log!(
                        Error,
                        "FreeCall target at index {} is not a Tensor",
                        value_index as u32
                    );
                    err = ResultExt::error(&t);
                    // PORT-NOTE: C++ `break;` out of the switch; falls through to
                    // the post-switch temp-allocator reset and commit logic. We
                    // mirror that by taking no further action in this arm.
                } else {
                    internal::reset_data_ptr(ResultExt::get(&t));
                }
            }
            other => {
                crate::et_log!(Error, "Unknown instruction: {}", other.0);
                err = Error::InvalidProgram;
            }
        }
        // Reset the temp allocator for every instruction.
        if !self.temp_allocator_.is_null() {
            unsafe {
                (*self.temp_allocator_).reset();
            }
        }
        if err == Error::Ok {
            self.step_state_.instr_idx = next_instr_idx;
        }
        err
    }

    // PORT-NOTE: helper reproducing `get_value(i)`/`mutable_value(i)`'s
    // `ET_CHECK_MSG(i < n_value_)` fatal bounds check without returning a borrow,
    // so the MoveCall arm can resolve both operand indices before aliasing
    // `values_` through raw pointers.
    fn get_value_index_checked(&self, i: i32) -> usize {
        et_check_msg!(
            (i as usize) < self.n_value_,
            "{} >= {}",
            i as usize,
            self.n_value_
        );
        i as usize
    }

    // [spec:et:def:method.executorch.et-runtime-namespace.method.log-outputs-fn]
    // [spec:et:sem:method.executorch.et-runtime-namespace.method.log-outputs-fn]
    // [spec:et:def:method.method.log-outputs-fn]
    // [spec:et:sem:method.method.log-outputs-fn]
    //
    // PORT-NOTE: C++ body is entirely inside `#ifdef ET_EVENT_TRACER_ENABLED`; a
    // no-op otherwise. Gated on the `event-tracer` feature to match both arms.
    fn log_outputs(&mut self) {
        #[cfg(feature = "event-tracer")]
        {
            if !self.event_tracer_.is_null() {
                if unsafe { (*self.event_tracer_).event_tracer_debug_level() }
                    >= crate::runtime::core::event_tracer::EventTracerDebugLogLevel::KProgramOutputs
                {
                    for i in 0..self.outputs_size() {
                        let out = self.get_output(i);
                        crate::runtime::core::event_tracer_hooks::event_tracer_log_evalue_output(
                            self.event_tracer_,
                            ResultExt::get(&out),
                        );
                    }
                }
            }
        }
    }
}

// PORT-NOTE: `Method(const Method&) = delete` / `operator=(const Method&) =
// delete` / `operator=(Method&&) = delete`; only the move ctor is allowed. A
// Rust `Method` is a non-Clone, move-only owner, matching those deletions. The
// C++ move ctor
// (`[spec:et:sem:method.executorch.et-runtime-namespace.method.method-fn]`) is
// the natural Rust move of the owned struct; the moved-from value is not reused.
// [spec:et:def:method.executorch.et-runtime-namespace.method.method-fn]
// [spec:et:sem:method.executorch.et-runtime-namespace.method.method-fn]

// PORT-NOTE: `~Method()` — hand-destroys the non-trivially-destructible owned
// state (values_ EValues, delegates_ BackendDelegates, external_constants_
// FreeableBuffers, merged_data_map_), then the trivially-destructible fields
// need nothing. Reproduced literally in `Drop`; the underlying arena memory is
// owned by the MemoryManager, not freed here.
impl<'a> Drop for Method<'a> {
    fn drop(&mut self) {
        // Destroy the values. It's necessary in ATen mode, where the refcount of
        // Tensors needs to be decremented properly.
        if !self.values_.is_null() {
            for i in 0..self.n_value_ {
                unsafe {
                    core::ptr::drop_in_place(self.values_.add(i));
                }
            }
        }
        // Free any resources associated with delegate backends.
        if !self.delegates_.is_null() {
            for i in 0..self.n_delegate_ {
                unsafe {
                    core::ptr::drop_in_place(self.delegates_.add(i));
                }
            }
        }
        // Free resources associated with external constants.
        for i in 0..self.n_external_constants_ {
            unsafe {
                core::ptr::drop_in_place(&mut (*self.external_constants_.add(i)).buffer);
            }
        }
        // Free the MergedDataMap.
        if !self.merged_data_map_.is_null() {
            unsafe {
                core::ptr::drop_in_place(self.merged_data_map_);
            }
        }
        // All other fields are trivially destructible.
    }
}

// Private/helper method for populating operator_name from the Operator.
// operator_name is a char pointer that is already allocated. The size of this
// buffer is of size operator_name_size.
// [spec:et:def:method.executorch.et-runtime-namespace.populate-operator-name-fn]
// [spec:et:sem:method.executorch.et-runtime-namespace.populate-operator-name-fn]
fn populate_operator_name(
    op: &executorch_flatbuffer::Operator,
    operator_name_size: usize,
    operator_name: *mut core::ffi::c_char,
) -> Error {
    let has_overload = op.overload().is_some() && !op.overload().unwrap().is_empty();

    crate::et_check_or_return_error!(op.name().is_some(), InvalidProgram, "Missing operator name");
    // PORT-NOTE: C++ `snprintf(buf, size, "%s%s%s", name, has_overload ? "." :
    // "", has_overload ? overload : "")`. Reproduced with a bounded byte writer
    // that returns the number of bytes the full string *would* occupy (like
    // snprintf's return), then applies the same `< size` fit check.
    let name = op.name().unwrap();
    let overload = if has_overload {
        op.overload().unwrap()
    } else {
        ""
    };
    let dot = if has_overload { "." } else { "" };
    let cx = snprintf3(operator_name, operator_name_size, name, dot, overload);
    crate::et_check_or_return_error!(cx >= 0, Internal, "snprintf failed: {}", cx);
    crate::et_check_or_return_error!(
        (cx as usize) < operator_name_size,
        Internal,
        "Operator name {}{}{} with length {} truncated to {} due to internal buffer limit.",
        name,
        dot,
        overload,
        cx,
        operator_name_size
    );

    Error::Ok
}

// PORT-NOTE: minimal `snprintf("%s%s%s")` for the operator-name buffer. Writes
// the concatenation of `a`/`b`/`c` into `buf` (NUL-terminated, truncated to
// `size`) and returns the total length that would be written excluding the NUL
// (matching snprintf's return value), or -1 on a size of 0.
fn snprintf3(buf: *mut core::ffi::c_char, size: usize, a: &str, b: &str, c: &str) -> i32 {
    let total = a.len() + b.len() + c.len();
    if size == 0 {
        return total as i32;
    }
    let cap = size - 1; // leave room for NUL
    let mut written: usize = 0;
    for part in [a, b, c] {
        for &byte in part.as_bytes() {
            if written >= cap {
                break;
            }
            unsafe {
                *buf.add(written) = byte as core::ffi::c_char;
            }
            written += 1;
        }
    }
    unsafe {
        *buf.add(written) = 0;
    }
    total as i32
}

/// Validate that a value index from a FlatBuffer instruction is in bounds.
// PORT-NOTE: C++ macro `ET_CHECK_VALID_VALUE_INDEX(index, n_value)` — requires
// `index >= 0 && (size_t)index < n_value` else returns Error::InvalidProgram.
macro_rules! et_check_valid_value_index {
    ($index:expr, $n_value:expr) => {
        crate::et_check_or_return_error!(
            $index >= 0 && ($index as usize) < $n_value,
            InvalidProgram,
            "Index {} negative or >= {}",
            $index as isize,
            $n_value
        )
    };
}
use et_check_valid_value_index;

// PORT-NOTE: helper producing a null `*const dyn NamedDataMap`. A fat pointer's
// null-ness is its data component; a null concrete pointer coerced to the trait
// object is `.is_null()` true, matching a C++ nullptr NamedDataMap*.
fn null_named_data_map() -> *const dyn NamedDataMap {
    core::ptr::null::<NullNamedDataMap>() as *const dyn NamedDataMap
}

fn is_null_named_data_map(p: *const dyn NamedDataMap) -> bool {
    (p as *const ()).is_null()
}

// PORT-NOTE: re-ties a `*const (dyn NamedDataMap + 'a)` (the PteDataMap borrow of
// the Program buffer) to `*const dyn NamedDataMap` (`+ 'static`), matching the
// C++ lifetime-free `const NamedDataMap*`. Sound because the Program outlives the
// Method that stores this pointer.
unsafe fn erase_named_data_map_lifetime<'a>(
    p: *const (dyn NamedDataMap + 'a),
) -> *const dyn NamedDataMap {
    unsafe { core::mem::transmute(p) }
}

// PORT-NOTE: `parse_tensor` takes `Option<&dyn NamedDataMap>` where the C++
// passes a possibly-null `const NamedDataMap*`. Convert a raw trait pointer to
// that option, mirroring the null check.
fn named_data_map_ref<'x>(p: *const dyn NamedDataMap) -> Option<&'x dyn NamedDataMap> {
    if is_null_named_data_map(p) {
        None
    } else {
        Some(unsafe { &*p })
    }
}

struct NullNamedDataMap;
impl NamedDataMap for NullNamedDataMap {
    fn get_tensor_layout(
        &self,
        _key: &str,
    ) -> Result<crate::runtime::core::tensor_layout::TensorLayout> {
        unreachable!()
    }
    fn get_data(&self, _key: &str) -> Result<FreeableBuffer> {
        unreachable!()
    }
    fn load_data_into(&self, _key: &str, _buffer: *mut core::ffi::c_void, _size: usize) -> Error {
        unreachable!()
    }
    fn get_num_keys(&self) -> Result<u32> {
        unreachable!()
    }
    fn get_key(&self, _index: u32) -> Result<*const core::ffi::c_char> {
        unreachable!()
    }
}

// PORT-NOTE: `c10::mul_overflows(a, b, &out)` on `ssize_t`/`size_t` — returns
// true on overflow and writes the (possibly wrapped) product to `out`. Inline
// port matching the other tensor helpers in the tree.
fn mul_overflows_ssize(a: ssize_t, b: ssize_t, out: &mut ssize_t) -> bool {
    let (res, overflow) = a.overflowing_mul(b);
    *out = res;
    overflow
}

fn mul_overflows_usize(a: usize, b: usize, out: &mut usize) -> bool {
    let (res, overflow) = a.overflowing_mul(b);
    *out = res;
    overflow
}

// PORT-NOTE: renders a NUL-terminated `const char*` operator name lossily for
// log messages (analog of passing the C string to `ET_LOG`'s `%s`).
fn cstr_lossy<'x>(p: *const core::ffi::c_char) -> &'x str {
    unsafe { core::ffi::CStr::from_ptr(p).to_str().unwrap_or("????") }
}

// Literal port of runtime/executor/test/method_test.cpp.
//
// PORT-NOTE: every `MethodTest` case loads a `.pte` model (env vars:
// `ET_MODULE_ADD_PATH`, `ET_MODULE_INDEX_PATH`,
// `ET_MODULE_DYNAMIC_CAT_UNALLOCATED_IO_PATH`, `ET_MODULE_ADD_MUL_PATH`,
// `ET_MODULE_STATEFUL_PATH`, `DEPRECATED_ET_MODULE_LINEAR_CONSTANT_BUFFER_PATH`,
// `ET_MODULE_ADD_MUL_PROGRAM_PATH`, `ET_MODULE_ADD_MUL_DATA_PATH`), then loads a
// `Method` and executes it. Running end-to-end additionally requires:
//   - the shared `ManagedMemoryManager` test helper (deferred; see the executor
//     `mod.rs` PORT-NOTE),
//   - `extension::runner_util::prepare_input_tensors` (not ported),
//   - `extension::flat_tensor::FlatTensorDataMap::load` (unimplemented stub),
//   - the generated portable-ops kernels (`aten::add.out` etc., not registered
//     in the Rust global registry).
// So every case skips early: it checks its fixture env var, then notes the
// unported execution dependencies and returns. The `ET_EXPECT_DEATH` sub-checks
// in the Get/Mutable Input/Output tests abort via the PAL abort path (they
// cannot unwind in-process) and are dropped with this note. The disabled
// `OptionalTensorListDeserialization` case (a C++ block comment, T161163608)
// is recorded here but not emitted.
#[cfg(test)]
mod tests {
    fn setup() {
        crate::runtime::platform::runtime::runtime_init();
    }

    fn skip(name: &str, env_var: &str) -> bool {
        if std::env::var(env_var).is_err() {
            eprintln!("skipping {name}: {env_var} unset");
            return true;
        }
        eprintln!(
            "skipping {name}: requires ManagedMemoryManager, prepare_input_tensors, \
             and the generated portable-ops kernels, none of which are ported"
        );
        true
    }

    // [spec:et:sem:program.executorch.et-runtime-namespace.program.load-method-fn/test]
    // [spec:et:sem:method.executorch.et-runtime-namespace.method.load-fn/test]
    // [spec:et:sem:method.executorch.et-runtime-namespace.method.execute-fn/test]
    #[test]
    fn method_test_move_test() {
        setup();
        if skip("method_test_move_test", "ET_MODULE_ADD_PATH") {}
    }

    // [spec:et:sem:method.executorch.et-runtime-namespace.method.get-input-fn/test]
    // [spec:et:sem:method.executorch.et-runtime-namespace.method.inputs-size-fn/test]
    #[test]
    fn method_test_get_input_tests() {
        setup();
        if skip("method_test_get_input_tests", "ET_MODULE_ADD_PATH") {}
    }

    // [spec:et:sem:method.executorch.et-runtime-namespace.method.mutable-input-fn/test]
    #[test]
    fn method_test_mutable_input_tests() {
        setup();
        if skip("method_test_mutable_input_tests", "ET_MODULE_ADD_PATH") {}
    }

    // [spec:et:sem:method.executorch.et-runtime-namespace.method.get-output-fn/test]
    // [spec:et:sem:method.executorch.et-runtime-namespace.method.outputs-size-fn/test]
    #[test]
    fn method_test_get_output_tests() {
        setup();
        if skip("method_test_get_output_tests", "ET_MODULE_ADD_PATH") {}
    }

    // [spec:et:sem:method.executorch.et-runtime-namespace.method.mutable-output-fn/test]
    #[test]
    fn method_test_mutable_output_tests() {
        setup();
        if skip("method_test_mutable_output_tests", "ET_MODULE_ADD_PATH") {}
    }

    // [spec:et:sem:method.executorch.et-runtime-namespace.method.set-input-fn/test]
    // [spec:et:sem:method.executorch.et-runtime-namespace.method.execute-fn/test]
    #[test]
    fn method_test_set_prim_input_test() {
        setup();
        if skip("method_test_set_prim_input_test", "ET_MODULE_ADD_PATH") {}
    }

    // [spec:et:sem:method.executorch.et-runtime-namespace.method.method-meta-fn/test]
    // [spec:et:sem:method.method.method-meta-fn/test]
    #[test]
    fn method_test_method_meta_test() {
        setup();
        if skip("method_test_method_meta_test", "ET_MODULE_ADD_PATH") {}
    }

    // [spec:et:sem:method.executorch.et-runtime-namespace.method.set-input-fn/test]
    // [spec:et:sem:method.executorch.et-runtime-namespace.method.set-output-data-ptr-fn/test]
    // [spec:et:sem:method.executorch.et-runtime-namespace.method.execute-fn/test]
    #[test]
    fn method_test_aliased_io_test() {
        setup();
        if skip(
            "method_test_aliased_io_test",
            "ET_MODULE_DYNAMIC_CAT_UNALLOCATED_IO_PATH",
        ) {}
    }

    // [spec:et:sem:method.executorch.et-runtime-namespace.method.set-input-fn/test]
    #[test]
    fn method_test_set_input_rejects_overflowing_sizes() {
        setup();
        if skip(
            "method_test_set_input_rejects_overflowing_sizes",
            "ET_MODULE_DYNAMIC_CAT_UNALLOCATED_IO_PATH",
        ) {}
    }

    // [spec:et:sem:method.executorch.et-runtime-namespace.method.execute-fn/test]
    #[test]
    fn method_test_constant_segment_test() {
        setup();
        if skip(
            "method_test_constant_segment_test",
            "ET_MODULE_ADD_MUL_PATH",
        ) {}
    }

    // PORT-NOTE: gated in C++ behind ET_ENABLE_DEPRECATED_CONSTANT_BUFFER (on by
    // default; the Rust `deprecated-constant-buffer` feature is a default too).
    // [spec:et:sem:method.executorch.et-runtime-namespace.method.execute-fn/test]
    #[test]
    fn method_test_constant_buffer_test() {
        setup();
        if skip(
            "method_test_constant_buffer_test",
            "DEPRECATED_ET_MODULE_LINEAR_CONSTANT_BUFFER_PATH",
        ) {}
    }

    // [spec:et:sem:program.executorch.et-runtime-namespace.program.load-method-fn/test]
    // [spec:et:sem:method.executorch.et-runtime-namespace.method.execute-fn/test]
    #[test]
    fn method_test_program_data_separation_test() {
        setup();
        if skip(
            "method_test_program_data_separation_test",
            "ET_MODULE_ADD_MUL_PROGRAM_PATH",
        ) {}
    }

    // [spec:et:sem:method.executorch.et-runtime-namespace.method.get-attribute-fn/test]
    // [spec:et:sem:method.executorch.et-runtime-namespace.method.execute-fn/test]
    #[test]
    fn method_test_method_get_attribute_test() {
        setup();
        if skip(
            "method_test_method_get_attribute_test",
            "ET_MODULE_STATEFUL_PATH",
        ) {}
    }

    // [spec:et:sem:method.executorch.et-runtime-namespace.method.in-progress-fn/test]
    #[test]
    fn method_test_in_progress_initial_state() {
        setup();
        if skip(
            "method_test_in_progress_initial_state",
            "ET_MODULE_ADD_PATH",
        ) {}
    }

    // [spec:et:sem:method.executorch.et-runtime-namespace.method.step-fn/test]
    // [spec:et:sem:method.executorch.et-runtime-namespace.method.in-progress-fn/test]
    #[test]
    fn method_test_in_progress_during_step_execution() {
        setup();
        if skip(
            "method_test_in_progress_during_step_execution",
            "ET_MODULE_ADD_MUL_PATH",
        ) {}
    }

    // [spec:et:sem:method.executorch.et-runtime-namespace.method.execute-fn/test]
    // [spec:et:sem:method.executorch.et-runtime-namespace.method.step-fn/test]
    #[test]
    fn method_test_execute_fails_when_in_progress() {
        setup();
        if skip(
            "method_test_execute_fails_when_in_progress",
            "ET_MODULE_ADD_MUL_PATH",
        ) {}
    }

    // [spec:et:sem:method.executorch.et-runtime-namespace.method.reset-execution-fn/test]
    // [spec:et:sem:method.executorch.et-runtime-namespace.method.execute-fn/test]
    #[test]
    fn method_test_execute_succeeds_after_reset() {
        setup();
        if skip(
            "method_test_execute_succeeds_after_reset",
            "ET_MODULE_ADD_PATH",
        ) {}
    }

    // [spec:et:sem:method.executorch.et-runtime-namespace.method.execute-fn/test]
    #[test]
    fn method_test_execute_resets_on_error() {
        setup();
        if skip("method_test_execute_resets_on_error", "ET_MODULE_ADD_PATH") {}
    }

    // ==== Focused unit tests for the fixture-free helpers ====
    //
    // The C++ method_test.cpp suite is all end-to-end (loads a .pte, runs
    // Method::execute) and cannot run until ManagedMemoryManager,
    // prepare_input_tensors and the portable-ops kernels are ported (see the
    // module PORT-NOTE and the `skip` stubs above). The three functions below
    // are pure/deterministic and are exercised directly against their sem rules
    // in docs/spec/port/runtime/executor/method.md.
    use crate::runtime::core::error::Error;
    use crate::runtime::core::evalue::EValue;
    use crate::runtime::core::exec_aten::testing_util::tensor_factory::TensorFactory;
    use crate::runtime::core::memory_allocator::MemoryAllocator;
    use crate::runtime::core::memory_allocator::MemoryAllocatorBase;
    use crate::schema::generated::executorch_flatbuffer;
    use flatbuffers::FlatBufferBuilder;

    // parse_cond_value: bool-scalar and bool-tensor shapes plus the unsupported
    // type error branch. Per the sem rule, a tensor cond is true iff every
    // element is true (empty tensor => true); a bool scalar returns its value;
    // any other tag => InvalidProgram.
    // [spec:et:sem:method.executorch.et-runtime-namespace.parse-cond-value-fn/test]
    #[test]
    fn method_parse_cond_value_test() {
        setup();

        // Bool scalar arms.
        assert_eq!(super::parse_cond_value(&EValue::from_bool(true)), Ok(true));
        assert_eq!(
            super::parse_cond_value(&EValue::from_bool(false)),
            Ok(false)
        );

        let tf = TensorFactory::<bool>::new();

        // All-true tensor => true.
        let all_true = EValue::from_tensor(tf.make_default(vec![3], vec![true, true, true]));
        assert_eq!(super::parse_cond_value(&all_true), Ok(true));

        // Any false element => false (short-circuits).
        let has_false = EValue::from_tensor(tf.make_default(vec![3], vec![true, false, true]));
        assert_eq!(super::parse_cond_value(&has_false), Ok(false));

        // Empty tensor => true (vacuously all-true).
        let empty = EValue::from_tensor(tf.make_default(vec![0], vec![]));
        assert_eq!(super::parse_cond_value(&empty), Ok(true));

        // Non-bool, non-tensor EValue => InvalidProgram.
        assert_eq!(
            super::parse_cond_value(&EValue::from_int(1)),
            Err(Error::InvalidProgram)
        );
    }

    // populate_operator_name: "%s%s%s" of name / "." / overload, with the
    // snprintf-style truncation check. No overload => bare name; with overload
    // => "name.overload"; a buffer too small to hold name+NUL => Internal.
    // [spec:et:sem:method.executorch.et-runtime-namespace.populate-operator-name-fn/test]
    #[test]
    fn method_populate_operator_name_test() {
        setup();

        fn build_op(name: Option<&str>, overload: Option<&str>) -> Vec<u8> {
            let mut b = FlatBufferBuilder::with_capacity(256);
            let name_off = name.map(|s| b.create_string(s));
            let overload_off = overload.map(|s| b.create_string(s));
            let op = executorch_flatbuffer::Operator::create(
                &mut b,
                &executorch_flatbuffer::OperatorArgs {
                    name: name_off,
                    overload: overload_off,
                },
            );
            b.finish_minimal(op);
            b.finished_data().to_vec()
        }

        fn name_of(buf: &[u8], size: usize) -> (Error, String) {
            let op = flatbuffers::root::<executorch_flatbuffer::Operator>(buf).unwrap();
            let mut out = vec![0 as core::ffi::c_char; size];
            let err = super::populate_operator_name(&op, size, out.as_mut_ptr());
            let s = if err == Error::Ok {
                unsafe {
                    core::ffi::CStr::from_ptr(out.as_ptr())
                        .to_str()
                        .unwrap()
                        .to_string()
                }
            } else {
                String::new()
            };
            (err, s)
        }

        // No overload => bare name.
        let buf = build_op(Some("aten::add"), None);
        assert_eq!(name_of(&buf, 100), (Error::Ok, "aten::add".to_string()));

        // Empty overload string is treated as no overload.
        let buf = build_op(Some("aten::add"), Some(""));
        assert_eq!(name_of(&buf, 100), (Error::Ok, "aten::add".to_string()));

        // With overload => "name.overload".
        let buf = build_op(Some("aten::add"), Some("out"));
        assert_eq!(name_of(&buf, 100), (Error::Ok, "aten::add.out".to_string()));

        // Missing name => InvalidProgram.
        let buf = build_op(None, None);
        assert_eq!(name_of(&buf, 100).0, Error::InvalidProgram);

        // "aten::add" is 9 bytes; a 9-byte buffer cannot hold name + NUL, so the
        // fit check (cx < size) fails => Internal.
        let buf = build_op(Some("aten::add"), None);
        assert_eq!(name_of(&buf, 9).0, Error::Internal);
        // 10 bytes exactly fits name + NUL.
        assert_eq!(name_of(&buf, 10), (Error::Ok, "aten::add".to_string()));
    }

    // gen_instruction_arguments: builds an array of EValue* by indexing the
    // master values table with the flatbuffer arg-index vector; an out-of-range
    // index => InvalidProgram.
    // [spec:et:sem:method.executorch.et-runtime-namespace.gen-instruction-arguments-fn/test]
    #[test]
    fn method_gen_instruction_arguments_test() {
        setup();

        fn build_i32_vector(items: &[i32]) -> Vec<u8> {
            let mut b = FlatBufferBuilder::with_capacity(64);
            let v = b.create_vector::<i32>(items);
            b.finish_minimal(v);
            b.finished_data().to_vec()
        }

        // Buffers declared first so their flatbuffer views outlive `values`
        // (drop order is reverse of declaration; the returned InstructionArgs
        // ties its lifetime to the arg-index vector view).
        let idx_buf = build_i32_vector(&[2, 0, 3]);
        let bad_buf = build_i32_vector(&[4]);

        // Master values table of 4 EValues.
        let mut values: [EValue; 4] = [
            EValue::from_int(0),
            EValue::from_int(1),
            EValue::from_int(2),
            EValue::from_int(3),
        ];
        let num_values = values.len();
        let values_ptr = values.as_mut_ptr();

        let mut arena = vec![0u8; 4096];
        let mut allocator = MemoryAllocator::new(arena.len() as u32, arena.as_mut_ptr());
        let allocator_ref: *mut dyn MemoryAllocatorBase = &mut allocator;

        // Build a flatbuffer Vector<i32> of arg indices {2, 0, 3}.
        let arg_idxs = flatbuffers::root::<flatbuffers::Vector<i32>>(&idx_buf).unwrap();

        let args =
            super::gen_instruction_arguments(allocator_ref, num_values, values_ptr, 3, &arg_idxs)
                .expect("gen_instruction_arguments");
        assert_eq!(args.size(), 3);
        unsafe {
            assert!(core::ptr::eq(*args.data().add(0), values_ptr.add(2)));
            assert!(core::ptr::eq(*args.data().add(1), values_ptr.add(0)));
            assert!(core::ptr::eq(*args.data().add(2), values_ptr.add(3)));
        }

        // An index >= num_values => InvalidProgram.
        let bad_idxs = flatbuffers::root::<flatbuffers::Vector<i32>>(&bad_buf).unwrap();
        assert_eq!(
            super::gen_instruction_arguments(allocator_ref, num_values, values_ptr, 1, &bad_idxs)
                .err(),
            Some(Error::InvalidProgram)
        );
    }

    // ==== Focused unit tests for the Method state machine and helpers ====
    //
    // These construct `Method` values directly (the tests module can reach the
    // private ctor and fields) plus hand-built flatbuffers, so they exercise
    // the real guards, step-state machine, and value plumbing without a .pte
    // fixture. Paths that need a loaded Program (Method::load end-to-end,
    // method_meta, GetProcessedData's INLINE/SEGMENT arms, kernel/delegate
    // execution) remain covered by the fixture-gated MethodTest stubs above.
    use crate::runtime::backend::backend_execution_context::BackendExecutionContext;
    use crate::runtime::backend::interface::{Backend, register_backend};
    use crate::runtime::core::array_ref::ArrayRef;
    use crate::runtime::core::event_tracer::EventTracer;
    use crate::runtime::core::freeable_buffer::FreeableBuffer;
    use crate::runtime::core::span::Span;
    use crate::runtime::executor::memory_manager::MemoryManager;
    use crate::runtime::kernel::operator_registry::OpFunction;
    use core::cell::Cell;

    fn null_tracer() -> *mut dyn EventTracer {
        crate::extension::module::module::null_event_tracer()
    }

    fn null_temp_allocator() -> *mut dyn MemoryAllocatorBase {
        core::ptr::null_mut::<MemoryAllocator>() as *mut dyn MemoryAllocatorBase
    }

    // A Method as the private ctor builds it: uninitialized, all state zeroed,
    // null program/memory-manager/tracer, empty method-scoped kernel registry.
    fn test_method<'a>() -> super::Method<'a> {
        super::Method::new(
            core::ptr::null::<super::Program>(),
            core::ptr::null_mut(),
            null_tracer(),
            null_temp_allocator(),
            Span::from_raw_parts(core::ptr::null_mut(), 0),
        )
    }

    fn build_plan(inputs: Option<&[i32]>, outputs: Option<&[i32]>) -> Vec<u8> {
        let mut b = FlatBufferBuilder::with_capacity(256);
        let name = b.create_string("forward");
        let inputs_off = inputs.map(|v| b.create_vector::<i32>(v));
        let outputs_off = outputs.map(|v| b.create_vector::<i32>(v));
        let plan = executorch_flatbuffer::ExecutionPlan::create(
            &mut b,
            &executorch_flatbuffer::ExecutionPlanArgs {
                name: Some(name),
                inputs: inputs_off,
                outputs: outputs_off,
                ..Default::default()
            },
        );
        b.finish_minimal(plan);
        b.finished_data().to_vec()
    }

    fn plan_view(buf: &[u8]) -> executorch_flatbuffer::ExecutionPlan<'_> {
        flatbuffers::root::<executorch_flatbuffer::ExecutionPlan>(buf).unwrap()
    }

    // initialized() is true iff init_state_ == Initialized (a failed init does
    // not count); get_event_tracer() returns the stored event_tracer_ pointer
    // verbatim (here, the null tracer passed to the ctor).
    // [spec:et:sem:method.executorch.et-runtime-namespace.method.initialized-fn/test]
    // [spec:et:sem:method.executorch.et-runtime-namespace.method.get-event-tracer-fn/test]
    #[test]
    fn method_initialized_and_get_event_tracer_test() {
        setup();
        let mut method = test_method();
        assert!(!method.initialized());
        assert!(method.get_event_tracer().is_null());
        method.init_state_ = super::InitializationState::InitializationFailed;
        assert!(!method.initialized());
        method.init_state_ = super::InitializationState::Initialized;
        assert!(method.initialized());
    }

    // execute(): NotSupported before init; InvalidState while step execution is
    // in progress (both guards fire before any chain/plan access).
    // [spec:et:sem:method.method.execute-fn/test]
    #[test]
    fn method_execute_state_guards_test() {
        setup();
        let mut method = test_method();
        assert_eq!(method.execute(), Error::NotSupported);

        method.init_state_ = super::InitializationState::Initialized;
        method.n_chains_ = 2;
        method.step_state_ = super::StepState {
            chain_idx: 1,
            instr_idx: 0,
        };
        assert_eq!(method.execute(), Error::InvalidState);
    }

    // step()/experimental_step() over a single zero-instruction chain:
    // InvalidState before init; an empty chain is skipped with Ok; EndOfMethod
    // once chain_idx == n_chains_. reset_execution() (and its deprecated alias)
    // rejects a reset before EndOfMethod and rewinds step_state_ to {0,0} after
    // it. in_progress() is true only between the initial state and the end.
    // [spec:et:sem:method.method.step-fn/test]
    // [spec:et:sem:method.method.experimental-step-fn/test]
    // [spec:et:sem:method.executorch.et-runtime-namespace.method.experimental-step-fn/test]
    // [spec:et:sem:method.method.reset-execution-fn/test]
    // [spec:et:sem:method.method.experimental-reset-execution-fn/test]
    // [spec:et:sem:method.executorch.et-runtime-namespace.method.experimental-reset-execution-fn/test]
    // [spec:et:sem:method.method.in-progress-fn/test]
    #[test]
    fn method_step_reset_in_progress_test() {
        setup();

        // Uninitialized: step (via the deprecated alias) is rejected.
        let mut uninit = test_method();
        assert_eq!(uninit.experimental_step(), Error::InvalidState);

        // One chain with zero instructions.
        let chain_buf = {
            let mut b = FlatBufferBuilder::with_capacity(64);
            let no_instructions: Vec<flatbuffers::WIPOffset<executorch_flatbuffer::Instruction>> =
                vec![];
            let instructions = b.create_vector(&no_instructions);
            let chain = executorch_flatbuffer::Chain::create(
                &mut b,
                &executorch_flatbuffer::ChainArgs {
                    instructions: Some(instructions),
                    ..Default::default()
                },
            );
            b.finish_minimal(chain);
            b.finished_data().to_vec()
        };
        let s_chain = flatbuffers::root::<executorch_flatbuffer::Chain>(&chain_buf).unwrap();
        let mut chain = super::Chain {
            s_chain_: s_chain,
            argument_lists_: Span::from_raw_parts(core::ptr::null_mut(), 0),
            kernels_: core::ptr::null_mut(),
        };

        let mut method = test_method();
        method.init_state_ = super::InitializationState::Initialized;
        method.n_chains_ = 1;
        method.chains_ = &mut chain;

        // Initial state: not in progress; reset before EndOfMethod is invalid.
        assert!(!method.in_progress());
        assert_eq!(method.reset_execution(), Error::InvalidState);

        // A mid-chain step state counts as in progress.
        method.step_state_ = super::StepState {
            chain_idx: 0,
            instr_idx: 1,
        };
        assert!(method.in_progress());
        method.step_state_ = super::StepState {
            chain_idx: 0,
            instr_idx: 0,
        };

        // Empty chain: step skips it and advances the chain index.
        assert_eq!(method.step(), Error::Ok);
        assert_eq!(method.step_state_.chain_idx, 1);
        assert!(!method.in_progress()); // completed, no longer "in progress"

        // Past the last chain: EndOfMethod (via the deprecated alias).
        assert_eq!(method.experimental_step(), Error::EndOfMethod);

        // reset_execution rewinds to {0, 0} (via the deprecated alias).
        assert_eq!(method.experimental_reset_execution(), Error::Ok);
        assert_eq!(method.step_state_.chain_idx, 0);
        assert_eq!(method.step_state_.instr_idx, 0);
    }

    // The C++ move ctor transfers the whole runtime state and leaves the source
    // unusable; the Rust Method is a non-Clone move-only owner, so a plain move
    // must preserve that state.
    // [spec:et:sem:method.executorch.et-runtime-namespace.method.method-fn/test]
    #[test]
    fn method_move_preserves_state_test() {
        setup();
        let mut method = test_method();
        method.init_state_ = super::InitializationState::Initialized;
        method.n_chains_ = 3;
        method.step_state_ = super::StepState {
            chain_idx: 2,
            instr_idx: 1,
        };

        let moved = method;
        assert!(moved.initialized());
        assert!(moved.in_progress());
        assert_eq!(moved.step_state_.chain_idx, 2);
        assert_eq!(moved.step_state_.instr_idx, 1);
    }

    // The input/output plumbing over a hand-built plan (input ordinal 0 ->
    // value 1, output ordinal 0 -> value 0): outputs_size counts the serialized
    // outputs; get/mutable_value index the master table through the serialized
    // input/output index mappings; get_outputs/get_inputs require an
    // initialized method and a large-enough array, shallow-copy the mapped
    // values, None-fill the tail, and get_inputs marks the inputs as set.
    // [spec:et:sem:method.method.outputs-size-fn/test]
    // [spec:et:sem:method.method.get-outputs-fn/test]
    // [spec:et:sem:method.executorch.et-runtime-namespace.method.get-outputs-fn/test]
    // [spec:et:sem:method.method.get-inputs-fn/test]
    // [spec:et:sem:method.executorch.et-runtime-namespace.method.get-inputs-fn/test]
    // [spec:et:sem:method.executorch.et-runtime-namespace.method.get-value-fn/test]
    // [spec:et:sem:method.executorch.et-runtime-namespace.method.mutable-value-fn/test]
    // [spec:et:sem:method.executorch.et-runtime-namespace.method.get-input-index-fn/test]
    // [spec:et:sem:method.executorch.et-runtime-namespace.method.get-output-index-fn/test]
    #[test]
    fn method_io_accessors_test() {
        setup();

        // Uninitialized: both accessors are rejected.
        {
            let mut uninit = test_method();
            let mut sink = [EValue::new()];
            assert_eq!(
                uninit.get_outputs(sink.as_mut_ptr(), 1),
                Error::InvalidState
            );
            assert_eq!(uninit.get_inputs(sink.as_mut_ptr(), 1), Error::InvalidState);
        }

        let plan_buf = build_plan(Some(&[1]), Some(&[0]));
        let plan = plan_view(&plan_buf);
        let mut values = [EValue::from_int(42), EValue::from_int(7)];
        let mut input_set = [false];

        let mut method = test_method();
        method.serialization_plan_ = Some(plan);
        method.init_state_ = super::InitializationState::Initialized;
        method.n_value_ = values.len();
        method.values_ = values.as_mut_ptr();
        method.input_set_ = input_set.as_mut_ptr();

        // A plan with no outputs vector reports zero outputs.
        let no_outputs_buf = build_plan(None, None);
        let mut no_outputs = test_method();
        no_outputs.serialization_plan_ = Some(plan_view(&no_outputs_buf));
        assert_eq!(no_outputs.outputs_size(), 0);

        assert_eq!(method.outputs_size(), 1);
        assert_eq!(method.inputs_size(), 1);

        // The value/index accessors, with the serialized index mapping.
        assert_eq!(method.get_input_index(0), 1);
        assert_eq!(method.get_output_index(0), 0);
        assert_eq!(method.get_value(1).to_int(), 7);
        method.mutable_value(1).assign_ref(&EValue::from_int(9));
        assert_eq!(method.get_value(1).to_int(), 9);

        // get_outputs: the length must cover all outputs; copies then
        // None-fills the tail.
        let mut outs = [EValue::new(), EValue::new(), EValue::new()];
        assert_eq!(
            method.get_outputs(outs.as_mut_ptr(), 0),
            Error::InvalidArgument
        );
        assert_eq!(method.get_outputs(outs.as_mut_ptr(), 3), Error::Ok);
        assert_eq!(outs[0].to_int(), 42);
        assert!(outs[1].is_none());
        assert!(outs[2].is_none());

        // get_inputs: copies via the input index mapping, marks the inputs as
        // set, and None-fills the tail.
        let mut ins = [EValue::new(), EValue::new()];
        assert_eq!(
            method.get_inputs(ins.as_mut_ptr(), 0),
            Error::InvalidArgument
        );
        assert_eq!(method.get_inputs(ins.as_mut_ptr(), 2), Error::Ok);
        assert_eq!(ins[0].to_int(), 9);
        assert!(ins[1].is_none());
        assert!(input_set[0]);

        // Detach the borrowed test arrays before Drop runs.
        method.values_ = core::ptr::null_mut();
        method.n_value_ = 0;
    }

    // set_inputs(): requires exactly inputs_size() EValues, then delegates each
    // element to set_input (which validates prim inputs against the traced
    // value and marks them set).
    // [spec:et:sem:method.executorch.method.set-inputs-fn/test]
    // [spec:et:sem:method.executorch.et-runtime-namespace.method.set-inputs-fn/test]
    #[test]
    fn method_set_inputs_test() {
        setup();
        let plan_buf = build_plan(Some(&[0]), None);
        let plan = plan_view(&plan_buf);
        let mut values = [EValue::from_int(5)];
        let mut input_set = [false];

        let mut method = test_method();
        method.serialization_plan_ = Some(plan);
        method.init_state_ = super::InitializationState::Initialized;
        method.n_value_ = 1;
        method.values_ = values.as_mut_ptr();
        method.input_set_ = input_set.as_mut_ptr();

        // Wrong arity.
        let empty: [EValue; 0] = [];
        assert_eq!(
            method.set_inputs(ArrayRef::from_raw_parts(empty.as_ptr(), 0)),
            Error::InvalidArgument
        );

        // A prim input must match the traced value.
        let wrong = [EValue::from_int(6)];
        assert_eq!(
            method.set_inputs(ArrayRef::from_raw_parts(wrong.as_ptr(), 1)),
            Error::InvalidArgument
        );
        assert!(!input_set[0]);

        let right = [EValue::from_int(5)];
        assert_eq!(
            method.set_inputs(ArrayRef::from_raw_parts(right.as_ptr(), 1)),
            Error::Ok
        );
        assert!(input_set[0]);

        method.values_ = core::ptr::null_mut();
        method.n_value_ = 0;
    }

    // set_output_data_ptr(): InvalidState before init; InvalidArgument for an
    // out-of-range output ordinal and for a non-tensor output value.
    // [spec:et:sem:method.executorch.method.set-output-data-ptr-fn/test]
    #[test]
    fn method_set_output_data_ptr_test() {
        setup();
        let mut buffer = [0u8; 16];
        let buffer_ptr = buffer.as_mut_ptr() as *mut core::ffi::c_void;

        let mut uninit = test_method();
        assert_eq!(
            uninit.set_output_data_ptr(buffer_ptr, 16, 0),
            Error::InvalidState
        );

        let plan_buf = build_plan(None, Some(&[0]));
        let plan = plan_view(&plan_buf);
        let mut values = [EValue::from_int(3)];
        let mut method = test_method();
        method.serialization_plan_ = Some(plan);
        method.init_state_ = super::InitializationState::Initialized;
        method.n_value_ = 1;
        method.values_ = values.as_mut_ptr();

        // Out-of-range output ordinal.
        assert_eq!(
            method.set_output_data_ptr(buffer_ptr, 16, 1),
            Error::InvalidArgument
        );
        // The output value is not a tensor.
        assert_eq!(
            method.set_output_data_ptr(buffer_ptr, 16, 0),
            Error::InvalidArgument
        );

        method.values_ = core::ptr::null_mut();
        method.n_value_ = 0;
    }

    // init(): a Method that is already Initialized (or whose previous init
    // failed) is rejected with InvalidState before any state is overwritten.
    // [spec:et:sem:method.executorch.et-runtime-namespace.method.init-fn/test]
    #[test]
    fn method_init_reinit_guard_test() {
        setup();
        let plan_buf = build_plan(None, None);
        let plan = plan_view(&plan_buf);

        let mut method = test_method();
        method.init_state_ = super::InitializationState::Initialized;
        assert_eq!(
            method.init(plan, super::null_named_data_map(), core::ptr::null()),
            Error::InvalidState
        );

        method.init_state_ = super::InitializationState::InitializationFailed;
        assert_eq!(
            method.init(plan, super::null_named_data_map(), core::ptr::null()),
            Error::InvalidState
        );
    }

    // parse_values(): a plan without a `values` vector is an invalid program.
    // parse_external_constants(): requires a non-null NamedDataMap.
    // [spec:et:sem:method.executorch.et-runtime-namespace.method.parse-values-fn/test]
    // [spec:et:sem:method.executorch.et-runtime-namespace.method.parse-external-constants-fn/test]
    #[test]
    fn method_parse_values_guards_test() {
        setup();
        let plan_buf = build_plan(None, None); // no `values` vector
        let plan = plan_view(&plan_buf);

        let mut method = test_method();
        method.serialization_plan_ = Some(plan);
        assert_eq!(
            method.parse_values(super::null_named_data_map()),
            Error::InvalidProgram
        );
        assert_eq!(
            method.parse_external_constants(super::null_named_data_map()),
            Error::InvalidState
        );
    }

    // get_num_external_constants(): counts only tensors tagged EXTERNAL that
    // have no allocation_info; a non-Null value with a missing union payload is
    // an invalid program.
    // [spec:et:sem:method.executorch.et-runtime-namespace.method.get-num-external-constants-fn/test]
    #[test]
    fn method_get_num_external_constants_test() {
        setup();

        fn tensor_evalue<'b>(
            b: &mut FlatBufferBuilder<'b>,
            location: Option<executorch_flatbuffer::TensorDataLocation>,
            with_allocation_info: bool,
        ) -> flatbuffers::WIPOffset<executorch_flatbuffer::EValue<'b>> {
            let extra = location.map(|loc| {
                executorch_flatbuffer::ExtraTensorInfo::create(
                    b,
                    &executorch_flatbuffer::ExtraTensorInfoArgs {
                        location: loc,
                        ..Default::default()
                    },
                )
            });
            let allocation_info = if with_allocation_info {
                Some(executorch_flatbuffer::AllocationDetails::create(
                    b,
                    &executorch_flatbuffer::AllocationDetailsArgs::default(),
                ))
            } else {
                None
            };
            let tensor = executorch_flatbuffer::Tensor::create(
                b,
                &executorch_flatbuffer::TensorArgs {
                    extra_tensor_info: extra,
                    allocation_info,
                    ..Default::default()
                },
            );
            executorch_flatbuffer::EValue::create(
                b,
                &executorch_flatbuffer::EValueArgs {
                    val_type: executorch_flatbuffer::KernelTypes::Tensor,
                    val: Some(tensor.as_union_value()),
                },
            )
        }

        let buf = {
            let mut b = FlatBufferBuilder::with_capacity(512);
            let external = tensor_evalue(
                &mut b,
                Some(executorch_flatbuffer::TensorDataLocation::EXTERNAL),
                false,
            );
            let external_mutable = tensor_evalue(
                &mut b,
                Some(executorch_flatbuffer::TensorDataLocation::EXTERNAL),
                true,
            );
            let plain = tensor_evalue(&mut b, None, false);
            let int_table = executorch_flatbuffer::Int::create(
                &mut b,
                &executorch_flatbuffer::IntArgs { int_val: 4 },
            );
            let int_value = executorch_flatbuffer::EValue::create(
                &mut b,
                &executorch_flatbuffer::EValueArgs {
                    val_type: executorch_flatbuffer::KernelTypes::Int,
                    val: Some(int_table.as_union_value()),
                },
            );
            let null_table =
                executorch_flatbuffer::Null::create(&mut b, &executorch_flatbuffer::NullArgs {});
            let null_value = executorch_flatbuffer::EValue::create(
                &mut b,
                &executorch_flatbuffer::EValueArgs {
                    val_type: executorch_flatbuffer::KernelTypes::Null,
                    val: Some(null_table.as_union_value()),
                },
            );
            let values =
                b.create_vector(&[external, external_mutable, plain, int_value, null_value]);
            let plan = executorch_flatbuffer::ExecutionPlan::create(
                &mut b,
                &executorch_flatbuffer::ExecutionPlanArgs {
                    values: Some(values),
                    ..Default::default()
                },
            );
            b.finish_minimal(plan);
            b.finished_data().to_vec()
        };
        let mut method = test_method();
        method.serialization_plan_ = Some(plan_view(&buf));
        // Only the EXTERNAL tensor without allocation_info counts.
        assert_eq!(method.get_num_external_constants(), Ok(1));

        // Malformed: a non-Null tag with no union payload.
        let bad_buf = {
            let mut b = FlatBufferBuilder::with_capacity(128);
            let bad = executorch_flatbuffer::EValue::create(
                &mut b,
                &executorch_flatbuffer::EValueArgs {
                    val_type: executorch_flatbuffer::KernelTypes::Int,
                    val: None,
                },
            );
            let values = b.create_vector(&[bad]);
            let plan = executorch_flatbuffer::ExecutionPlan::create(
                &mut b,
                &executorch_flatbuffer::ExecutionPlanArgs {
                    values: Some(values),
                    ..Default::default()
                },
            );
            b.finish_minimal(plan);
            b.finished_data().to_vec()
        };
        let mut bad_method = test_method();
        bad_method.serialization_plan_ = Some(unsafe {
            flatbuffers::root_unchecked::<executorch_flatbuffer::ExecutionPlan>(&bad_buf)
        });
        assert_eq!(
            bad_method.get_num_external_constants(),
            Err(Error::InvalidProgram)
        );
    }

    // resolve_operator(): the op_index bounds check, then the registry lookup
    // failure for an operator name that is not registered anywhere.
    // [spec:et:sem:method.executorch.et-runtime-namespace.method.resolve-operator-fn/test]
    #[test]
    fn method_resolve_operator_test() {
        setup();
        let plan_buf = {
            let mut b = FlatBufferBuilder::with_capacity(128);
            let name = b.create_string("method_rs_test::not_registered");
            let overload = b.create_string("out");
            let op = executorch_flatbuffer::Operator::create(
                &mut b,
                &executorch_flatbuffer::OperatorArgs {
                    name: Some(name),
                    overload: Some(overload),
                },
            );
            let operators = b.create_vector(&[op]);
            let plan = executorch_flatbuffer::ExecutionPlan::create(
                &mut b,
                &executorch_flatbuffer::ExecutionPlanArgs {
                    operators: Some(operators),
                    ..Default::default()
                },
            );
            b.finish_minimal(plan);
            b.finished_data().to_vec()
        };

        let mut arena = vec![0u8; 4096];
        let mut method_allocator = MemoryAllocator::new(arena.len() as u32, arena.as_mut_ptr());
        let method_allocator_ref: *mut dyn MemoryAllocatorBase = &mut method_allocator;
        let mut memory_manager = MemoryManager::new(
            method_allocator_ref,
            core::ptr::null_mut(),
            null_temp_allocator(),
        );

        let mut method = test_method();
        method.serialization_plan_ = Some(plan_view(&plan_buf));
        method.memory_manager_ = &mut memory_manager;

        let mut kernels: [Option<OpFunction>; 1] = [None];

        // Out-of-range operator index.
        assert_eq!(
            method.resolve_operator(
                3,
                kernels.as_mut_ptr(),
                0,
                Span::from_raw_parts(core::ptr::null_mut(), 0),
                0
            ),
            Error::InvalidProgram
        );
        // Unregistered operator name.
        assert_eq!(
            method.resolve_operator(
                0,
                kernels.as_mut_ptr(),
                0,
                Span::from_raw_parts(core::ptr::null_mut(), 0),
                0
            ),
            Error::OperatorMissing
        );
        assert!(kernels[0].is_none());
    }

    // execute_instruction(): the instruction-index bounds check, the unknown-
    // instruction arm, and the JumpFalseCall/MoveCall arms, which drive
    // step_state_.instr_idx and the values_ table without a kernel or delegate.
    // [spec:et:sem:method.method.execute-instruction-fn/test]
    // [spec:et:sem:method.executorch.et-runtime-namespace.method.execute-instruction-fn/test]
    #[test]
    fn method_execute_instruction_test() {
        setup();

        // Instruction index past the end of the chain (zero instructions).
        {
            let buf = {
                let mut b = FlatBufferBuilder::with_capacity(64);
                let no_instructions: Vec<
                    flatbuffers::WIPOffset<executorch_flatbuffer::Instruction>,
                > = vec![];
                let instructions = b.create_vector(&no_instructions);
                let chain = executorch_flatbuffer::Chain::create(
                    &mut b,
                    &executorch_flatbuffer::ChainArgs {
                        instructions: Some(instructions),
                        ..Default::default()
                    },
                );
                b.finish_minimal(chain);
                b.finished_data().to_vec()
            };
            let s_chain = flatbuffers::root::<executorch_flatbuffer::Chain>(&buf).unwrap();
            let mut chain = super::Chain {
                s_chain_: s_chain,
                argument_lists_: Span::from_raw_parts(core::ptr::null_mut(), 0),
                kernels_: core::ptr::null_mut(),
            };
            let mut method = test_method();
            method.n_chains_ = 1;
            method.chains_ = &mut chain;
            assert_eq!(method.execute_instruction(), Error::Internal);
        }

        // Unknown instruction: InvalidProgram and instr_idx does not advance.
        {
            let buf = {
                let mut b = FlatBufferBuilder::with_capacity(64);
                let instr = executorch_flatbuffer::Instruction::create(
                    &mut b,
                    &executorch_flatbuffer::InstructionArgs {
                        instr_args_type: executorch_flatbuffer::InstructionArguments::NONE,
                        instr_args: None,
                    },
                );
                let instructions = b.create_vector(&[instr]);
                let chain = executorch_flatbuffer::Chain::create(
                    &mut b,
                    &executorch_flatbuffer::ChainArgs {
                        instructions: Some(instructions),
                        ..Default::default()
                    },
                );
                b.finish_minimal(chain);
                b.finished_data().to_vec()
            };
            let s_chain = flatbuffers::root::<executorch_flatbuffer::Chain>(&buf).unwrap();
            let mut chain = super::Chain {
                s_chain_: s_chain,
                argument_lists_: Span::from_raw_parts(core::ptr::null_mut(), 0),
                kernels_: core::ptr::null_mut(),
            };
            let mut method = test_method();
            method.n_chains_ = 1;
            method.chains_ = &mut chain;
            assert_eq!(method.execute_instruction(), Error::InvalidProgram);
            assert_eq!(method.step_state_.instr_idx, 0);
        }

        // JumpFalseCall: a true cond falls through (instr_idx + 1); a false
        // cond jumps to destination_instruction.
        {
            let buf = {
                let mut b = FlatBufferBuilder::with_capacity(64);
                let jf = executorch_flatbuffer::JumpFalseCall::create(
                    &mut b,
                    &executorch_flatbuffer::JumpFalseCallArgs {
                        cond_value_index: 0,
                        destination_instruction: 3,
                    },
                );
                let instr = executorch_flatbuffer::Instruction::create(
                    &mut b,
                    &executorch_flatbuffer::InstructionArgs {
                        instr_args_type: executorch_flatbuffer::InstructionArguments::JumpFalseCall,
                        instr_args: Some(jf.as_union_value()),
                    },
                );
                let instructions = b.create_vector(&[instr]);
                let chain = executorch_flatbuffer::Chain::create(
                    &mut b,
                    &executorch_flatbuffer::ChainArgs {
                        instructions: Some(instructions),
                        ..Default::default()
                    },
                );
                b.finish_minimal(chain);
                b.finished_data().to_vec()
            };
            let s_chain = flatbuffers::root::<executorch_flatbuffer::Chain>(&buf).unwrap();
            let mut chain = super::Chain {
                s_chain_: s_chain,
                argument_lists_: Span::from_raw_parts(core::ptr::null_mut(), 0),
                kernels_: core::ptr::null_mut(),
            };
            let mut values = [EValue::from_bool(true)];
            let mut method = test_method();
            method.n_chains_ = 1;
            method.chains_ = &mut chain;
            method.n_value_ = 1;
            method.values_ = values.as_mut_ptr();

            assert_eq!(method.execute_instruction(), Error::Ok);
            assert_eq!(method.step_state_.instr_idx, 1);

            method.step_state_.instr_idx = 0;
            values[0] = EValue::from_bool(false);
            assert_eq!(method.execute_instruction(), Error::Ok);
            assert_eq!(method.step_state_.instr_idx, 3);

            method.values_ = core::ptr::null_mut();
            method.n_value_ = 0;
        }

        // MoveCall: copy-assigns values_[move_from] into values_[move_to].
        {
            let buf = {
                let mut b = FlatBufferBuilder::with_capacity(64);
                let mv = executorch_flatbuffer::MoveCall::create(
                    &mut b,
                    &executorch_flatbuffer::MoveCallArgs {
                        move_from: 0,
                        move_to: 1,
                    },
                );
                let instr = executorch_flatbuffer::Instruction::create(
                    &mut b,
                    &executorch_flatbuffer::InstructionArgs {
                        instr_args_type: executorch_flatbuffer::InstructionArguments::MoveCall,
                        instr_args: Some(mv.as_union_value()),
                    },
                );
                let instructions = b.create_vector(&[instr]);
                let chain = executorch_flatbuffer::Chain::create(
                    &mut b,
                    &executorch_flatbuffer::ChainArgs {
                        instructions: Some(instructions),
                        ..Default::default()
                    },
                );
                b.finish_minimal(chain);
                b.finished_data().to_vec()
            };
            let s_chain = flatbuffers::root::<executorch_flatbuffer::Chain>(&buf).unwrap();
            let mut chain = super::Chain {
                s_chain_: s_chain,
                argument_lists_: Span::from_raw_parts(core::ptr::null_mut(), 0),
                kernels_: core::ptr::null_mut(),
            };
            let mut values = [EValue::from_int(7), EValue::from_int(0)];
            let mut method = test_method();
            method.n_chains_ = 1;
            method.chains_ = &mut chain;
            method.n_value_ = 2;
            method.values_ = values.as_mut_ptr();

            assert_eq!(method.execute_instruction(), Error::Ok);
            assert_eq!(method.step_state_.instr_idx, 1);
            assert_eq!(values[1].to_int(), 7);

            method.values_ = core::ptr::null_mut();
            method.n_value_ = 0;
        }
    }

    // log_outputs(): with no event tracer attached this is a no-op (the C++
    // body is a null-tracer check under ET_EVENT_TRACER_ENABLED, and empty
    // otherwise); no state is disturbed.
    // [spec:et:sem:method.method.log-outputs-fn/test]
    // [spec:et:sem:method.executorch.et-runtime-namespace.method.log-outputs-fn/test]
    #[test]
    fn method_log_outputs_null_tracer_test() {
        setup();
        let plan_buf = build_plan(None, Some(&[0]));
        let plan = plan_view(&plan_buf);
        let mut values = [EValue::from_int(1)];
        let mut method = test_method();
        method.serialization_plan_ = Some(plan);
        method.init_state_ = super::InitializationState::Initialized;
        method.n_value_ = 1;
        method.values_ = values.as_mut_ptr();

        method.log_outputs();

        assert_eq!(method.step_state_.chain_idx, 0);
        assert_eq!(method.get_value(0).to_int(), 1);

        method.values_ = core::ptr::null_mut();
        method.n_value_ = 0;
    }

    // ==== BackendDelegate ====

    fn build_delegate(id: Option<&str>, location: executorch_flatbuffer::DataLocation) -> Vec<u8> {
        let mut b = FlatBufferBuilder::with_capacity(128);
        let id_off = id.map(|s| b.create_string(s));
        let processed = executorch_flatbuffer::BackendDelegateDataReference::create(
            &mut b,
            &executorch_flatbuffer::BackendDelegateDataReferenceArgs { location, index: 0 },
        );
        let delegate = executorch_flatbuffer::BackendDelegate::create(
            &mut b,
            &executorch_flatbuffer::BackendDelegateArgs {
                id: id_off,
                processed: Some(processed),
                compile_specs: None,
            },
        );
        b.finish_minimal(delegate);
        b.finished_data().to_vec()
    }

    // A registered backend that reports itself unavailable.
    struct UnavailableBackend;
    impl super::BackendInterface for UnavailableBackend {
        fn is_available(&self) -> bool {
            false
        }
        fn init(
            &self,
            _context: &mut super::BackendInitContext,
            _processed: *mut FreeableBuffer,
            _compile_specs: ArrayRef<super::CompileSpec>,
        ) -> super::Result<*mut super::DelegateHandle> {
            unreachable!()
        }
        fn execute(
            &self,
            _context: &mut BackendExecutionContext,
            _handle: *mut super::DelegateHandle,
            _args: Span<*mut EValue>,
        ) -> Error {
            unreachable!()
        }
    }

    // BackendDelegate::Init: a delegate without a backend id is an invalid
    // program; an id that is not registered, or that is registered but reports
    // itself unavailable, is NotFound. All three fire before the delegate data
    // or program is touched.
    // [spec:et:sem:method.executorch.et-runtime-namespace.backend-delegate.init-fn/test]
    #[test]
    fn backend_delegate_init_errors_test() {
        setup();
        let mut arena = vec![0u8; 1024];
        let mut allocator = MemoryAllocator::new(arena.len() as u32, arena.as_mut_ptr());
        let allocator_ref: *mut dyn MemoryAllocatorBase = &mut allocator;
        let mut ctx = super::BackendInitContext::new(
            allocator_ref,
            null_tracer(),
            core::ptr::null(),
            super::null_named_data_map(),
            Span::from_raw_parts(core::ptr::null_mut(), 0),
        );
        let mut out = core::mem::MaybeUninit::<super::BackendDelegate>::uninit();

        // Missing backend id.
        let no_id = build_delegate(None, executorch_flatbuffer::DataLocation::INLINE);
        let view = flatbuffers::root::<executorch_flatbuffer::BackendDelegate>(&no_id).unwrap();
        assert_eq!(
            super::BackendDelegate::init(
                &view,
                core::ptr::null::<super::Program>(),
                &mut ctx,
                out.as_mut_ptr()
            ),
            Error::InvalidProgram
        );

        // Unregistered backend id.
        let absent = build_delegate(
            Some("MethodTestAbsentBackend"),
            executorch_flatbuffer::DataLocation::INLINE,
        );
        let view = flatbuffers::root::<executorch_flatbuffer::BackendDelegate>(&absent).unwrap();
        assert_eq!(
            super::BackendDelegate::init(
                &view,
                core::ptr::null::<super::Program>(),
                &mut ctx,
                out.as_mut_ptr()
            ),
            Error::NotFound
        );

        // Registered but unavailable backend.
        let backend_impl: &'static mut UnavailableBackend = Box::leak(Box::new(UnavailableBackend));
        let backend_ptr: *mut dyn super::BackendInterface = backend_impl;
        let registration = Backend {
            name: c"MethodTestUnavailableBackend".as_ptr(),
            backend: backend_ptr,
        };
        let reg = register_backend(&registration);
        assert!(reg == Error::Ok || reg == Error::InvalidArgument);
        let unavailable = build_delegate(
            Some("MethodTestUnavailableBackend"),
            executorch_flatbuffer::DataLocation::INLINE,
        );
        let view =
            flatbuffers::root::<executorch_flatbuffer::BackendDelegate>(&unavailable).unwrap();
        assert_eq!(
            super::BackendDelegate::init(
                &view,
                core::ptr::null::<super::Program>(),
                &mut ctx,
                out.as_mut_ptr()
            ),
            Error::NotFound
        );
    }

    struct RecordingBackend {
        execute_count: Cell<usize>,
        destroy_count: Cell<usize>,
        last_handle: Cell<usize>,
    }
    impl super::BackendInterface for RecordingBackend {
        fn is_available(&self) -> bool {
            true
        }
        fn init(
            &self,
            _context: &mut super::BackendInitContext,
            _processed: *mut FreeableBuffer,
            _compile_specs: ArrayRef<super::CompileSpec>,
        ) -> super::Result<*mut super::DelegateHandle> {
            unreachable!()
        }
        fn execute(
            &self,
            _context: &mut BackendExecutionContext,
            handle: *mut super::DelegateHandle,
            args: Span<*mut EValue>,
        ) -> Error {
            self.execute_count.set(self.execute_count.get() + 1);
            self.last_handle.set(handle as usize);
            assert_eq!(args.size(), 0);
            Error::Ok
        }
        fn destroy(&self, handle: *mut super::DelegateHandle) {
            self.destroy_count.set(self.destroy_count.get() + 1);
            self.last_handle.set(handle as usize);
        }
    }

    // BackendDelegate::Execute forwards (context, handle_, args) to the backend
    // and returns its Error verbatim; dropping the delegate (~BackendDelegate)
    // calls backend->destroy(handle_) exactly once.
    // [spec:et:sem:method.executorch.et-runtime-namespace.backend-delegate.execute-fn/test]
    // [spec:et:sem:method.executorch.et-runtime-namespace.backend-delegate.backend-delegate-fn/test]
    #[test]
    fn backend_delegate_execute_and_destroy_test() {
        setup();
        let mut backend = RecordingBackend {
            execute_count: Cell::new(0),
            destroy_count: Cell::new(0),
            last_handle: Cell::new(0),
        };
        let backend_ptr: *mut dyn super::BackendInterface = &mut backend;
        let handle = 0x1234usize as *mut super::DelegateHandle;

        {
            let delegate = super::BackendDelegate {
                segment_: FreeableBuffer::new(),
                backend_: backend_ptr,
                handle_: handle,
            };
            let mut context = BackendExecutionContext::new(
                null_tracer(),
                null_temp_allocator(),
                core::ptr::null(),
            );
            assert_eq!(
                delegate.execute(&mut context, Span::from_raw_parts(core::ptr::null_mut(), 0)),
                Error::Ok
            );
            assert_eq!(backend.execute_count.get(), 1);
            assert_eq!(backend.last_handle.get(), 0x1234);
        } // ~BackendDelegate: destroy(handle_).
        assert_eq!(backend.destroy_count.get(), 1);
        assert_eq!(backend.last_handle.get(), 0x1234);
    }

    // PopulateCompileSpecs: copies every (key, value) pair out of the
    // flatbuffer into an allocator-owned CompileSpec list; a too-small
    // allocator surfaces as MemoryAllocationFailed.
    // [spec:et:sem:method.executorch.et-runtime-namespace.backend-delegate.populate-compile-specs-fn/test]
    #[test]
    fn backend_delegate_populate_compile_specs_test() {
        setup();
        let buf = {
            let mut b = FlatBufferBuilder::with_capacity(256);
            let key1 = b.create_string("num_threads");
            let val1 = b.create_vector(&[4u8, 0, 0, 0]);
            let spec1 = executorch_flatbuffer::CompileSpec::create(
                &mut b,
                &executorch_flatbuffer::CompileSpecArgs {
                    key: Some(key1),
                    value: Some(val1),
                },
            );
            let key2 = b.create_string("mode");
            let val2 = b.create_vector(&[1u8]);
            let spec2 = executorch_flatbuffer::CompileSpec::create(
                &mut b,
                &executorch_flatbuffer::CompileSpecArgs {
                    key: Some(key2),
                    value: Some(val2),
                },
            );
            let vec = b.create_vector(&[spec1, spec2]);
            b.finish_minimal(vec);
            b.finished_data().to_vec()
        };
        let specs = flatbuffers::root::<
            flatbuffers::Vector<flatbuffers::ForwardsUOffset<executorch_flatbuffer::CompileSpec>>,
        >(&buf)
        .unwrap();

        let mut arena = vec![0u8; 1024];
        let mut allocator = MemoryAllocator::new(arena.len() as u32, arena.as_mut_ptr());
        let allocator_ref: *mut dyn MemoryAllocatorBase = &mut allocator;
        let mut ctx = super::BackendInitContext::new(
            allocator_ref,
            null_tracer(),
            core::ptr::null(),
            super::null_named_data_map(),
            Span::from_raw_parts(core::ptr::null_mut(), 0),
        );

        let mut out: *mut super::CompileSpec = core::ptr::null_mut();
        assert_eq!(
            super::BackendDelegate::populate_compile_specs(&specs, &mut ctx, &mut out),
            Error::Ok
        );
        assert!(!out.is_null());
        unsafe {
            let s0 = &*out;
            assert_eq!(
                core::ffi::CStr::from_ptr(s0.key).to_str().unwrap(),
                "num_threads"
            );
            assert_eq!(s0.value.nbytes, 4);
            assert_eq!(
                core::slice::from_raw_parts(s0.value.buffer as *const u8, 4),
                &[4, 0, 0, 0]
            );
            let s1 = &*out.add(1);
            assert_eq!(core::ffi::CStr::from_ptr(s1.key).to_str().unwrap(), "mode");
            assert_eq!(s1.value.nbytes, 1);
        }

        // Allocation failure surfaces as MemoryAllocationFailed.
        let mut tiny = [0u8; 1];
        let mut tiny_allocator = MemoryAllocator::new(1, tiny.as_mut_ptr());
        let tiny_ref: *mut dyn MemoryAllocatorBase = &mut tiny_allocator;
        let mut tiny_ctx = super::BackendInitContext::new(
            tiny_ref,
            null_tracer(),
            core::ptr::null(),
            super::null_named_data_map(),
            Span::from_raw_parts(core::ptr::null_mut(), 0),
        );
        let mut out2: *mut super::CompileSpec = core::ptr::null_mut();
        assert_eq!(
            super::BackendDelegate::populate_compile_specs(&specs, &mut tiny_ctx, &mut out2),
            Error::MemoryAllocationFailed
        );
    }

    // GetProcessedData: an unknown DataLocation is rejected with
    // Error::Internal before the program is touched (the INLINE/SEGMENT arms
    // need a loaded Program and stay with the fixture-gated tests above).
    // [spec:et:sem:method.executorch.et-runtime-namespace.backend-delegate.get-processed-data-fn/test]
    #[test]
    fn backend_delegate_get_processed_data_unknown_location_test() {
        setup();
        let buf = build_delegate(Some("AnyBackend"), executorch_flatbuffer::DataLocation(2));
        let view =
            unsafe { flatbuffers::root_unchecked::<executorch_flatbuffer::BackendDelegate>(&buf) };
        let result =
            super::BackendDelegate::get_processed_data(&view, core::ptr::null::<super::Program>());
        assert_eq!(result.err(), Some(Error::Internal));
    }
}
