//! Literal port of backends/xnnpack/runtime/XNNExecutor.cpp +
//! backends/xnnpack/runtime/XNNExecutor.h.
//!
//! PORT-NOTE: The `XNNExecutor` owns a compiled XNNPACK runtime and drives the
//! XNNPACK C API (`xnn_setup_runtime_v2`, `xnn_reshape_*`, `xnn_invoke_runtime`,
//! `xnn_delete_runtime`), so the whole module is gated behind the `xnnpack`
//! feature. `XNNCompiler` is a friend that populates the executor via
//! `initialize`; that access is modeled as the `initialize` method plus the
//! public field-setting entry points the compiler needs.
#![cfg(feature = "xnnpack")]
#![allow(non_snake_case)]

use super::XNNStatus::xnn_status_to_string;
use super::XNNWeightsCache::XNNWeightsCache;
use super::XNNWorkspace::XNNWorkspace;
use super::profiling::XNNProfiler::XNNProfiler;
use super::sys::{
    XNN_MAX_TENSOR_DIMS, xnn_delete_runtime, xnn_external_value, xnn_get_external_value_shape,
    xnn_invoke_runtime, xnn_reshape_external_value, xnn_reshape_runtime, xnn_runtime_t,
    xnn_setup_runtime_v2, xnn_status, xnn_status_success,
};

use crate::runtime::backend::backend_execution_context::BackendExecutionContext;
use crate::runtime::core::array_ref::ArrayRef;
use crate::runtime::core::error::Error;
use crate::runtime::core::evalue::EValue;
use crate::runtime::core::exec_aten::exec_aten::{DimOrderType, SizesType};
use crate::runtime::core::exec_aten::util::tensor_util::{
    K_TENSOR_DIMENSION_LIMIT, get_dim_order, resize_tensor,
};
use crate::runtime::core::freeable_buffer::FreeableBuffer;
use crate::runtime::core::portable_type::scalar_type::ScalarType;
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::core::span::Span;

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
#[cfg(any(
    feature = "xnnpack-profiling",
    feature = "profiling-enabled",
    feature = "event-tracer"
))]
use std::sync::{Mutex, OnceLock};

#[cfg(any(
    feature = "xnnpack-profiling",
    feature = "profiling-enabled",
    feature = "event-tracer"
))]
struct ProfilerRegistry {
    by_runtime: std::collections::HashMap<usize, XNNProfiler>,
}

// Each runtime is already serialized by its executor/workspace contract. The
// registry moves profiler state out of the method's bounded runtime allocator,
// whose size must not depend on optional profiling fields.
#[cfg(any(
    feature = "xnnpack-profiling",
    feature = "profiling-enabled",
    feature = "event-tracer"
))]
unsafe impl Send for ProfilerRegistry {}

#[cfg(any(
    feature = "xnnpack-profiling",
    feature = "profiling-enabled",
    feature = "event-tracer"
))]
fn profiler_registry() -> &'static Mutex<ProfilerRegistry> {
    static REGISTRY: OnceLock<Mutex<ProfilerRegistry>> = OnceLock::new();
    REGISTRY.get_or_init(|| {
        Mutex::new(ProfilerRegistry {
            by_runtime: std::collections::HashMap::new(),
        })
    })
}

#[cfg(any(
    feature = "xnnpack-profiling",
    feature = "profiling-enabled",
    feature = "event-tracer"
))]
fn runtime_key(runtime: xnn_runtime_t) -> usize {
    runtime.0 as usize
}

// PORT-NOTE: `ET_DCHECK_MSG` — debug-only assert. Not exported crate-wide;
// mirrored locally over `debug_assert!` (elided in release, exactly like the
// C++ `ET_DCHECK_MSG`).
macro_rules! et_dcheck_msg {
    ($cond:expr, $($arg:tt)*) => {
        debug_assert!($cond, $($arg)*);
    };
}

// [spec:et:def:xnn-executor.executorch.backends.xnnpack.delegate.in-use-guard]
//
// PORT-NOTE: RAII scope guard over a borrowed `&AtomicBool`. The C++ holds a
// reference member `std::atomic<bool>& flag_`; ported as a `&'a AtomicBool`. The
// automatic release is a `Drop` impl that release-stores `false` unless
// dismissed. The deleted copy ctor / copy-assign is the Rust default (no
// `Copy`/`Clone`), so its markers collapse onto this move-only guard.
// [spec:et:def:xnn-executor.executorch.backends.xnnpack.delegate.in-use-guard.operator-fn]
// [spec:et:sem:xnn-executor.executorch.backends.xnnpack.delegate.in-use-guard.operator-fn]
struct InUseGuard<'a> {
    flag_: &'a AtomicBool,
    dismissed_: bool,
}

impl<'a> InUseGuard<'a> {
    fn new(flag: &'a AtomicBool) -> Self {
        InUseGuard {
            flag_: flag,
            dismissed_: false,
        }
    }

    // [spec:et:def:xnn-executor.executorch.backends.xnnpack.delegate.in-use-guard.dismiss-fn]
    // [spec:et:sem:xnn-executor.executorch.backends.xnnpack.delegate.in-use-guard.dismiss-fn]
    fn dismiss(&mut self) {
        self.dismissed_ = true;
    }
}

// [spec:et:def:xnn-executor.executorch.backends.xnnpack.delegate.in-use-guard.in-use-guard-fn]
// [spec:et:sem:xnn-executor.executorch.backends.xnnpack.delegate.in-use-guard.in-use-guard-fn]
impl<'a> Drop for InUseGuard<'a> {
    fn drop(&mut self) {
        if !self.dismissed_ {
            self.flag_.store(false, Ordering::Release);
        }
    }
}

// PORT-NOTE: `std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)>` —
// modeled as a newtype owning the raw handle whose `Drop` calls
// `xnn_delete_runtime`, mirroring the unique_ptr deleter. A null handle is not
// deleted (matches unique_ptr with a null pointer).
struct RuntimePtr(xnn_runtime_t);

impl RuntimePtr {
    fn null() -> Self {
        RuntimePtr(xnn_runtime_t(core::ptr::null_mut()))
    }
    fn get(&self) -> xnn_runtime_t {
        self.0
    }
    fn is_null(&self) -> bool {
        self.0.0.is_null()
    }
}

impl Drop for RuntimePtr {
    fn drop(&mut self) {
        if !self.0.0.is_null() {
            unsafe {
                xnn_delete_runtime(self.0);
            }
        }
    }
}

// The raw XNNPACK runtime handle is used behind serialization (the workspace
// lock); mark the owner Send/Sync to reflect the C++ sharing contract.
unsafe impl Send for RuntimePtr {}
unsafe impl Sync for RuntimePtr {}

// [spec:et:def:xnn-executor.executorch.backends.xnnpack.delegate.xnn-executor]
pub struct XNNExecutor {
    runtime_: RuntimePtr,

    #[cfg(not(any(
        feature = "xnnpack-profiling",
        feature = "profiling-enabled",
        feature = "event-tracer"
    )))]
    profiler_: XNNProfiler,
    input_ids_: Vec<u32>,
    output_ids_: Vec<u32>,
    externals_: Vec<xnn_external_value>,
    packed_data_names_: Vec<String>,
    // PORT-NOTE (deviation from C++): named-data constant buffers referenced by
    // the compiled graph. xnnpack.h requires static tensor data to outlive the
    // Subgraph AND any Runtime created from it, but the C++ compileModel frees
    // its unpacked_buffers right after xnn_create_runtime — safe upstream only
    // because upstream AoT routes exclusively *packed* consumers (conv/FC
    // weights, copied during runtime creation) through named data. Constants
    // feeding binary/elementwise ops are read by POINTER at invoke time, so
    // they must live as long as the runtime: the port retains them here and
    // drops them after `runtime_` (Rust field-order drop: `runtime_` above
    // drops first).
    unpacked_buffers_: Vec<FreeableBuffer>,
    workspace_: Arc<XNNWorkspace>,
    // Owned so the cache outlives delete_packed_data in destroy(), even when
    // every other executor sharing it is gone. `None` when no file-backed cache
    // is in use.
    weights_cache_: Option<Arc<XNNWeightsCache>>,
    in_use_: AtomicBool,
    destroyed_: AtomicBool,
}

impl XNNExecutor {
    // PORT-NOTE: `XNNExecutor(std::shared_ptr<XNNWorkspace> workspace)` — the
    // sole public ctor, called by placement-new in `XNNPACKBackend::init`. Sets
    // `workspace_` and default-initializes the rest (`runtime_` null, empty id /
    // externals / packed-name vectors, `in_use_`/`destroyed_` false).
    pub fn new(workspace: Arc<XNNWorkspace>) -> Self {
        XNNExecutor {
            runtime_: RuntimePtr::null(),
            #[cfg(not(any(
                feature = "xnnpack-profiling",
                feature = "profiling-enabled",
                feature = "event-tracer"
            )))]
            profiler_: XNNProfiler::new(),
            input_ids_: Vec::new(),
            output_ids_: Vec::new(),
            externals_: Vec::new(),
            packed_data_names_: Vec::new(),
            unpacked_buffers_: Vec::new(),
            workspace_: workspace,
            weights_cache_: None,
            in_use_: AtomicBool::new(false),
            destroyed_: AtomicBool::new(false),
        }
    }

    // PORT-NOTE: `XNNPACKBackend::destroy` calls `executor->print_avg_op_timings()`
    // under `#ifdef ENABLE_XNNPACK_PROFILING`, but NO definition of this method
    // exists anywhere in the current C++ tree (neither XNNExecutor.h/.cpp nor
    // the profiler). The default build never defines that macro, so the call is
    // dead and the missing definition is a latent source bug. To keep the port
    // compilable under the equivalent `profiling-enabled` feature, this
    // no-op method is provided under the same cfg. It should be reconciled with
    // the C++ once the upstream definition (or the dead call) is resolved.
    #[cfg(any(feature = "profiling-enabled", feature = "xnnpack-profiling"))]
    pub fn print_avg_op_timings(&mut self) {}

    // [spec:et:def:xnn-executor.executorch.backends.xnnpack.delegate.xnn-executor.get-num-inputs-fn]
    // [spec:et:sem:xnn-executor.executorch.backends.xnnpack.delegate.xnn-executor.get-num-inputs-fn]
    #[inline]
    pub fn getNumInputs(&self) -> usize {
        self.input_ids_.len()
    }

    // [spec:et:def:xnn-executor.executorch.backends.xnnpack.delegate.xnn-executor.get-num-outputs-fn]
    // [spec:et:sem:xnn-executor.executorch.backends.xnnpack.delegate.xnn-executor.get-num-outputs-fn]
    #[inline]
    pub fn getNumOutputs(&self) -> usize {
        self.output_ids_.len()
    }

    // [spec:et:def:xnn-executor.executorch.backends.xnnpack.delegate.xnn-executor.get-packed-data-names-fn]
    // [spec:et:sem:xnn-executor.executorch.backends.xnnpack.delegate.xnn-executor.get-packed-data-names-fn]
    #[inline]
    pub fn get_packed_data_names(&self) -> Vec<String> {
        self.packed_data_names_.clone()
    }

    // [spec:et:def:xnn-executor.executorch.backends.xnnpack.delegate.xnn-executor.uses-weight-cache-fn]
    // [spec:et:sem:xnn-executor.executorch.backends.xnnpack.delegate.xnn-executor.uses-weight-cache-fn]
    #[inline]
    pub fn uses_weight_cache(&self) -> bool {
        !self.packed_data_names_.is_empty()
    }

    // [spec:et:def:xnn-executor.executorch.backends.xnnpack.delegate.xnn-executor.get-workspace-fn]
    // [spec:et:sem:xnn-executor.executorch.backends.xnnpack.delegate.xnn-executor.get-workspace-fn]
    #[inline]
    pub fn get_workspace(&self) -> Arc<XNNWorkspace> {
        self.workspace_.clone()
    }

    /// Test accessor: the `destroyed_` flag (relaxed load). The destructor
    /// release-stores `true` here, which isn't observable after drop, so tests
    /// assert it is `false` while the executor is alive.
    pub fn dbg_destroyed(&self) -> bool {
        self.destroyed_.load(Ordering::Relaxed)
    }

    // [spec:et:def:xnn-executor.executorch.backends.xnnpack.delegate.xnn-executor.set-weights-cache-fn]
    // [spec:et:sem:xnn-executor.executorch.backends.xnnpack.delegate.xnn-executor.set-weights-cache-fn]
    //
    // PORT-NOTE: The C++ signature is `set_weights_cache(std::shared_ptr<
    // XNNWeightsCache> cache)` (moves in). Passing an empty shared_ptr is valid;
    // here `None` models the empty shared_ptr, so the parameter is
    // `Option<Arc<...>>`.
    #[inline]
    pub fn set_weights_cache(&mut self, cache: Option<Arc<XNNWeightsCache>>) {
        self.weights_cache_ = cache;
    }

    // [spec:et:def:xnn-executor.executorch.backends.xnnpack.delegate.xnn-executor.get-weights-cache-fn]
    // [spec:et:sem:xnn-executor.executorch.backends.xnnpack.delegate.xnn-executor.get-weights-cache-fn]
    #[inline]
    pub fn get_weights_cache(&self) -> Option<Arc<XNNWeightsCache>> {
        self.weights_cache_.clone()
    }

    // [spec:et:def:xnn-executor.executorch.backends.xnnpack.delegate.xnn-executor.initialize-fn]
    // [spec:et:sem:xnn-executor.executorch.backends.xnnpack.delegate.xnn-executor.initialize-fn]
    //
    // PORT-NOTE: The C++ takes `runtime` (raw handle) and rvalue-ref id / name
    // vectors (`std::vector&&`). In Rust the moved-in vectors are passed by
    // value.
    #[must_use]
    pub fn initialize(
        &mut self,
        runtime: xnn_runtime_t,
        input_ids: Vec<u32>,
        output_ids: Vec<u32>,
        packed_data_names: Vec<String>,
        unpacked_buffers: Vec<FreeableBuffer>,
        enable_profiling: bool,
    ) -> Error {
        self.runtime_ = RuntimePtr(runtime);
        self.unpacked_buffers_ = unpacked_buffers;

        #[cfg(any(
            feature = "xnnpack-profiling",
            feature = "profiling-enabled",
            feature = "event-tracer"
        ))]
        if enable_profiling {
            let mut profiler = XNNProfiler::new();
            let error = profiler.initialize(runtime);
            if error != Error::Ok {
                crate::et_log!(Error, "Failed to initialize profiling: {}.", error as u32);
            } else {
                profiler_registry()
                    .lock()
                    .unwrap()
                    .by_runtime
                    .insert(runtime_key(runtime), profiler);
            }
        }
        #[cfg(not(any(
            feature = "xnnpack-profiling",
            feature = "profiling-enabled",
            feature = "event-tracer"
        )))]
        let _ = enable_profiling;

        // Initialize the external values for inputs and outputs mapping the
        // executorch arg idx to external IDs.
        self.input_ids_ = input_ids;
        self.input_ids_.sort();

        self.output_ids_ = output_ids;
        self.output_ids_.sort();

        self.externals_.resize(
            self.input_ids_.len() + self.output_ids_.len(),
            xnn_external_value {
                id: 0,
                data: core::ptr::null_mut(),
            },
        );
        self.packed_data_names_ = packed_data_names;

        Error::Ok
    }

    // [spec:et:def:xnn-executor.executorch.backends.xnnpack.delegate.xnn-executor.prepare-args-fn]
    // [spec:et:sem:xnn-executor.executorch.backends.xnnpack.delegate.xnn-executor.prepare-args-fn]
    #[must_use]
    pub fn prepare_args(&mut self, args: Span<*mut EValue>) -> Error {
        et_dcheck_msg!(
            !self.destroyed_.load(Ordering::Acquire),
            "XNNExecutor::prepare_args called after destroy"
        );

        let was_in_use = self.in_use_.swap(true, Ordering::Acquire);
        if was_in_use {
            crate::et_log!(Error, "XNNExecutor::prepare_args called concurrently");
        }
        et_dcheck_msg!(!was_in_use, "XNNExecutor::prepare_args called concurrently");

        let mut in_use_guard = InUseGuard::new(&self.in_use_);
        if was_in_use {
            in_use_guard.dismiss();
        }

        crate::et_check_or_return_error!(
            !self.runtime_.is_null(),
            Internal,
            "XNNPACK Delegate did not compile correctly"
        );

        // Create xnn_externals_value from evalue args.
        let mut status: xnn_status;
        for i in 0..self.externals_.len() as u32 {
            if (i as usize) < self.input_ids_.len() {
                self.externals_[i as usize].id = self.input_ids_[i as usize];
            } else {
                self.externals_[i as usize].id =
                    self.output_ids_[i as usize - self.input_ids_.len()];
            }
            let ext_id: u32 = self.externals_[i as usize].id;

            let arg: &mut EValue = unsafe { &mut **args.index(ext_id as usize) };
            crate::et_check_or_return_error!(
                arg.is_tensor(),
                InvalidArgument,
                "Expected argument to delegate at index {} to be a Tensor, but got {}",
                i,
                arg.tag as u32
            );

            let tensor: &Tensor = arg.to_tensor();
            self.externals_[i as usize].data =
                tensor.mutable_data_ptr::<f32>() as *mut core::ffi::c_void;

            let mut dim_order: [DimOrderType; K_TENSOR_DIMENSION_LIMIT] =
                [0; K_TENSOR_DIMENSION_LIMIT];

            // Reshape runtime inputs.
            if (i as usize) < self.input_ids_.len() {
                let num_dims: usize = tensor.dim() as usize;
                let err = unsafe { get_dim_order(tensor, dim_order.as_mut_ptr(), num_dims) };
                crate::et_check_or_return_error!(
                    err == Error::Ok,
                    Internal,
                    "Failed to retrieve dim order from tensor!"
                );
                let mut dims: [usize; XNN_MAX_TENSOR_DIMS] = [0; XNN_MAX_TENSOR_DIMS];
                crate::et_check_or_return_error!(
                    num_dims <= XNN_MAX_TENSOR_DIMS,
                    InvalidArgument,
                    "XNNPACK backend accepts tensors with at most {} dims, but got {}",
                    XNN_MAX_TENSOR_DIMS,
                    num_dims
                );

                for j in 0..num_dims {
                    dims[j] = tensor.size(dim_order[j] as isize) as usize;
                }
                status = unsafe {
                    xnn_reshape_external_value(self.runtime_.get(), ext_id, num_dims, dims.as_ptr())
                };
                crate::et_check_or_return_error!(
                    status == xnn_status_success,
                    Internal,
                    "Internal Error: Reshape Input Tensor Failed with code: {:?}",
                    xnn_status_to_string(status)
                );
            }
        }
        // Propagate Input Shape and Memory Plan for increased allocation.
        status = unsafe { xnn_reshape_runtime(self.runtime_.get()) };

        crate::et_check_or_return_error!(
            status == xnn_status_success,
            Internal,
            "Internal Error: Propagating input shapes failed with code: {:?}",
            xnn_status_to_string(status)
        );

        // Resize output tensors.
        let err = self.resize_outputs(args);
        if err != Error::Ok {
            return err;
        }

        in_use_guard.dismiss();
        Error::Ok
    }

    // [spec:et:def:xnn-executor.executorch.backends.xnnpack.delegate.xnn-executor.forward-fn]
    // [spec:et:sem:xnn-executor.executorch.backends.xnnpack.delegate.xnn-executor.forward-fn]
    #[must_use]
    pub fn forward(&mut self, context: &mut BackendExecutionContext) -> Error {
        let _in_use_guard = InUseGuard::new(&self.in_use_);

        crate::et_check_or_return_error!(
            !self.runtime_.is_null(),
            Internal,
            "XNNPACK Delegate did not compile correctly"
        );

        let mut status: xnn_status = unsafe {
            xnn_setup_runtime_v2(
                self.runtime_.get(),
                self.externals_.len(),
                self.externals_.as_ptr(),
            )
        };

        if status != xnn_status_success {
            crate::et_log!(
                Error,
                "Internal Error: Setting up the runtime failed with code: {:?}",
                xnn_status_to_string(status)
            );
            return Error::Internal;
        }

        #[cfg(any(
            feature = "xnnpack-profiling",
            feature = "profiling-enabled",
            feature = "event-tracer"
        ))]
        let profiling_started = profiler_registry()
            .lock()
            .unwrap()
            .by_runtime
            .get_mut(&runtime_key(self.runtime_.get()))
            .is_some_and(|profiler| {
                let error = profiler.start(context.event_tracer());
                if error != Error::Ok {
                    crate::et_log!(Error, "Failed to start profiling: {}.", error as u32);
                }
                error == Error::Ok
            });

        status = unsafe { xnn_invoke_runtime(self.runtime_.get()) };

        #[cfg(any(
            feature = "xnnpack-profiling",
            feature = "profiling-enabled",
            feature = "event-tracer"
        ))]
        if profiling_started {
            if let Some(profiler) = profiler_registry()
                .lock()
                .unwrap()
                .by_runtime
                .get_mut(&runtime_key(self.runtime_.get()))
            {
                let end_error = profiler.end();
                if end_error != Error::Ok {
                    crate::et_log!(Error, "Failed to end profiling: {}.", end_error as u32);
                }
            }
        }

        crate::et_check_or_return_error!(
            status == xnn_status_success,
            Internal,
            "XNN Runtime invoke failed with code: {:?}",
            xnn_status_to_string(status)
        );

        Error::Ok
    }

    // [spec:et:def:xnn-executor.executorch.backends.xnnpack.delegate.xnn-executor.resize-outputs-fn]
    // [spec:et:sem:xnn-executor.executorch.backends.xnnpack.delegate.xnn-executor.resize-outputs-fn]
    #[must_use]
    pub fn resize_outputs(&self, args: Span<*mut EValue>) -> Error {
        let output_idx_start: usize = self.input_ids_.len();
        for i in output_idx_start..self.externals_.len() {
            let ext_id: u32 = self.externals_[i].id;
            let out_tensor: &Tensor = unsafe { (**args.index(ext_id as usize)).to_tensor() };

            let mut num_dim: usize = 0;
            let mut dims: [usize; XNN_MAX_TENSOR_DIMS] = [0; XNN_MAX_TENSOR_DIMS];

            // Fetch the updated output shapes from xnnpack runtime.
            let status = unsafe {
                xnn_get_external_value_shape(
                    self.runtime_.get(),
                    ext_id,
                    &mut num_dim,
                    dims.as_mut_ptr(),
                )
            };

            crate::et_check_or_return_error!(
                status == xnn_status_success,
                Internal,
                "Internal Error: Failed to retrieve graph output shapes"
            );

            // Convert new output shape into SizesType.
            let mut expected_output_size: [SizesType; K_TENSOR_DIMENSION_LIMIT] =
                [0; K_TENSOR_DIMENSION_LIMIT];
            let mut dim_order: [DimOrderType; K_TENSOR_DIMENSION_LIMIT] =
                [0; K_TENSOR_DIMENSION_LIMIT];
            let errr = unsafe { get_dim_order(out_tensor, dim_order.as_mut_ptr(), num_dim) };
            crate::et_check_or_return_error!(
                errr == Error::Ok,
                Internal,
                "Failed to retrieve dim order from tensor!"
            );

            for j in 0..num_dim {
                expected_output_size[dim_order[j] as usize] = dims[j] as SizesType;
            }

            let output_size: ArrayRef<SizesType> =
                ArrayRef::from_raw_parts(expected_output_size.as_ptr(), num_dim);

            crate::et_log!(Debug, "Resizing output tensor to a new shape");
            let err = resize_tensor(out_tensor, output_size);
            if err != Error::Ok {
                crate::et_log!(Error, "Failed to resize output tensor for XNNExecutor");
                return err;
            }
        }

        Error::Ok
    }

    // [spec:et:def:xnn-executor.executorch.backends.xnnpack.delegate.xnn-executor.convert-outputs-fn]
    // [spec:et:sem:xnn-executor.executorch.backends.xnnpack.delegate.xnn-executor.convert-outputs-fn]
    #[must_use]
    pub fn convert_outputs(&self, args: Span<*mut EValue>) -> Error {
        let output_idx_start: usize = self.input_ids_.len();
        for i in output_idx_start..self.externals_.len() {
            let ext_id: u32 = self.externals_[i].id;
            let out_tensor: &Tensor = unsafe { (**args.index(ext_id as usize)).to_tensor() };

            // Output datatype is int64. However, XNNPACK doesn't support int64.
            // This means that the data was put into this tensor by XNNPACK as
            // int32 and needs to be copied to int64 form.
            if out_tensor.scalar_type() == ScalarType::Long {
                let data_64: *mut i64 = out_tensor.mutable_data_ptr::<i64>();
                let data_32: *const i32 = out_tensor.const_data_ptr::<i32>();
                // PORT-NOTE: iterate DESCENDING from numel()-1 to 0. Essential:
                // the int64 element at index `j` overlaps the int32 elements at
                // higher indices, so writing from the high end backward avoids
                // clobbering not-yet-read int32 values (an in-place expanding
                // copy). `data_32[j]` is sign-extended to int64.
                let mut j: isize = out_tensor.numel() - 1;
                while j >= 0 {
                    unsafe {
                        *data_64.offset(j) = *data_32.offset(j) as i64;
                    }
                    j -= 1;
                }
            }
        }

        Error::Ok
    }
}

// [spec:et:def:xnn-executor.executorch.backends.xnnpack.delegate.xnn-executor.xnn-executor-fn]
// [spec:et:sem:xnn-executor.executorch.backends.xnnpack.delegate.xnn-executor.xnn-executor-fn]
//
// PORT-NOTE: The C++ destructor debug-asserts `!in_use_` then release-stores
// `true` into `destroyed_`. Member destruction then runs in reverse declaration
// order; in particular the `runtime_` unique_ptr deleter calls
// `xnn_delete_runtime`. In Rust the `RuntimePtr` field's own `Drop` handles the
// runtime deletion after this `Drop` body runs, so ordering is preserved
// (fields drop after the struct's `Drop::drop`).
impl Drop for XNNExecutor {
    fn drop(&mut self) {
        et_dcheck_msg!(
            !self.in_use_.load(Ordering::Acquire),
            "XNNExecutor destroyed while in use"
        );
        #[cfg(any(
            feature = "xnnpack-profiling",
            feature = "profiling-enabled",
            feature = "event-tracer"
        ))]
        profiler_registry()
            .lock()
            .unwrap()
            .by_runtime
            .remove(&runtime_key(self.runtime_.get()));
        self.destroyed_.store(true, Ordering::Release);
    }
}

// Literal port of backends/xnnpack/test/runtime/test_xnnexecutor.cpp.
//
// PORT-NOTE: LINK GAP + non-delegate XNNPACK surface. These tests build XNNPACK
// subgraphs and runtimes directly through the C API (`xnn_initialize`,
// `xnn_create_subgraph`, `xnn_define_*`, `xnn_create_runtime`) — the graph
// *construction* surface that the C++ compiler produces, which is outside the
// ported delegate runtime (the delegate consumes a pre-built runtime handle).
// The two extra construction entry points used only here (`xnn_define_clamp`,
// `xnn_create_runtime`) are declared in `sys.rs`. Nothing links the XNNPACK C
// library yet, so these compile under `--features xnnpack` but cannot link/run;
// they are `#[ignore]`d and gated by the feature. The default `cargo test`
// build does not enable the feature.
//
// PORT-NOTE: The C++ constructs `XNNExecutor executor({})` with a null
// `shared_ptr<XNNWorkspace>`. Rust `XNNExecutor::new` takes a non-nullable
// `Arc<XNNWorkspace>`; a real workspace is created here instead. Neither
// `initialize`/`prepare_args`/`forward`/`convert_outputs` dereference the
// workspace in these tests, so the behavior under test is unchanged.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::core::evalue::EValue;
    use crate::runtime::core::event_tracer::EventTracer;
    use crate::runtime::core::exec_aten::testing_util::tensor_factory::TensorFactory;
    use crate::runtime::core::memory_allocator::{MemoryAllocator, MemoryAllocatorBase};
    use crate::runtime::core::tensor_shape_dynamism::TensorShapeDynamism;
    use crate::runtime::platform::platform::et_pal_init;
    use crate::runtime::platform::types::et_timestamp_t;

    use super::super::sys::{
        XNN_INVALID_VALUE_ID, XNN_VALUE_FLAG_EXTERNAL_INPUT, XNN_VALUE_FLAG_EXTERNAL_OUTPUT,
        xnn_create_runtime, xnn_create_subgraph, xnn_datatype_fp32, xnn_datatype_int32,
        xnn_datatype_qint8, xnn_define_argmax_pooling_2d, xnn_define_clamp,
        xnn_define_quantized_tensor_value, xnn_define_tensor_value, xnn_delete_subgraph,
        xnn_initialize, xnn_runtime_t, xnn_status_success, xnn_subgraph_t,
    };

    use super::super::XNNWorkspace::WorkspacePtr;
    use super::super::sys::xnn_workspace_t;

    fn null_allocator() -> *mut dyn MemoryAllocatorBase {
        core::ptr::null_mut::<MemoryAllocator>() as *mut dyn MemoryAllocatorBase
    }
    fn null_event_tracer() -> *mut dyn EventTracer {
        core::ptr::null_mut::<NullEventTracer>() as *mut dyn EventTracer
    }

    // Never-instantiated concrete implementor used only to spell the null
    // `*mut dyn EventTracer` slot for the default `BackendExecutionContext`
    // (mirroring the C++ default-constructed context whose `event_tracer_` is
    // nullptr).
    struct NullEventTracer;
    impl EventTracer for NullEventTracer {
        fn state(&self) -> &crate::runtime::core::event_tracer::EventTracerState {
            unreachable!()
        }
        fn state_mut(&mut self) -> &mut crate::runtime::core::event_tracer::EventTracerState {
            unreachable!()
        }
        fn create_event_block(&mut self, _name: *const core::ffi::c_char) {
            unreachable!()
        }
        fn start_profiling(
            &mut self,
            _name: *const core::ffi::c_char,
            _chain_id: crate::runtime::core::event_tracer::ChainID,
            _debug_handle: crate::runtime::core::event_tracer::DebugHandle,
        ) -> crate::runtime::core::event_tracer::EventTracerEntry {
            unreachable!()
        }
        fn start_profiling_delegate(
            &mut self,
            _name: *const core::ffi::c_char,
            _delegate_debug_index: crate::runtime::core::event_tracer::DelegateDebugIntId,
        ) -> crate::runtime::core::event_tracer::EventTracerEntry {
            unreachable!()
        }
        fn end_profiling_delegate(
            &mut self,
            _event_tracer_entry: crate::runtime::core::event_tracer::EventTracerEntry,
            _metadata: *const core::ffi::c_void,
            _metadata_len: usize,
        ) {
            unreachable!()
        }
        fn log_profiling_delegate(
            &mut self,
            _name: *const core::ffi::c_char,
            _delegate_debug_index: crate::runtime::core::event_tracer::DelegateDebugIntId,
            _start_time: et_timestamp_t,
            _end_time: et_timestamp_t,
            _metadata: *const core::ffi::c_void,
            _metadata_len: usize,
        ) {
            unreachable!()
        }
        fn end_profiling(
            &mut self,
            _prof_entry: crate::runtime::core::event_tracer::EventTracerEntry,
        ) {
            unreachable!()
        }
        fn track_allocation(
            &mut self,
            _id: crate::runtime::core::event_tracer::AllocatorID,
            _size: usize,
        ) {
            unreachable!()
        }
        fn track_allocator(
            &mut self,
            _name: *const core::ffi::c_char,
        ) -> crate::runtime::core::event_tracer::AllocatorID {
            unreachable!()
        }
        fn log_evalue(
            &mut self,
            _evalue: &crate::runtime::core::evalue::EValue,
            _evalue_type: crate::runtime::core::event_tracer::LoggedEValueType,
        ) -> crate::runtime::core::result::Result<bool> {
            unreachable!()
        }
        fn log_intermediate_output_delegate_tensor(
            &mut self,
            _name: *const core::ffi::c_char,
            _delegate_debug_index: crate::runtime::core::event_tracer::DelegateDebugIntId,
            _output: &crate::runtime::core::portable_type::tensor::Tensor,
        ) -> crate::runtime::core::result::Result<bool> {
            unreachable!()
        }
        fn log_intermediate_output_delegate_tensor_array(
            &mut self,
            _name: *const core::ffi::c_char,
            _delegate_debug_index: crate::runtime::core::event_tracer::DelegateDebugIntId,
            _output: crate::runtime::core::array_ref::ArrayRef<
                crate::runtime::core::portable_type::tensor::Tensor,
            >,
        ) -> crate::runtime::core::result::Result<bool> {
            unreachable!()
        }
        fn log_intermediate_output_delegate_int(
            &mut self,
            _name: *const core::ffi::c_char,
            _delegate_debug_index: crate::runtime::core::event_tracer::DelegateDebugIntId,
            _output: &i32,
        ) -> crate::runtime::core::result::Result<bool> {
            unreachable!()
        }
        fn log_intermediate_output_delegate_bool(
            &mut self,
            _name: *const core::ffi::c_char,
            _delegate_debug_index: crate::runtime::core::event_tracer::DelegateDebugIntId,
            _output: &bool,
        ) -> crate::runtime::core::result::Result<bool> {
            unreachable!()
        }
        fn log_intermediate_output_delegate_double(
            &mut self,
            _name: *const core::ffi::c_char,
            _delegate_debug_index: crate::runtime::core::event_tracer::DelegateDebugIntId,
            _output: &f64,
        ) -> crate::runtime::core::result::Result<bool> {
            unreachable!()
        }
        fn set_delegation_intermediate_output_filter(
            &mut self,
            _event_tracer_filter: *mut dyn crate::runtime::core::event_tracer::EventTracerFilterBase,
        ) {
            unreachable!()
        }
    }

    // A real workspace, standing in for the C++ null `shared_ptr<XNNWorkspace>`
    // passed to `XNNExecutor({})` (see module-level PORT-NOTE).
    fn make_executor() -> XNNExecutor {
        XNNExecutor::new(XNNWorkspace::create().unwrap())
    }

    // A workspace wrapping a NULL `xnn_workspace_t`, built WITHOUT any XNNPACK C
    // call (`XNNWorkspace::create` would call `xnn_create_workspace`, which needs
    // `xnn_initialize` + the linked lib). This is the closest faithful analog of
    // the C++ `XNNExecutor({})` (null `shared_ptr<XNNWorkspace>`): the state-only
    // executor surface below never dereferences the workspace, so these tests run
    // without the graph-construction/link dependency the `#[ignore]`d tests have.
    fn make_executor_null() -> XNNExecutor {
        let ws = Arc::new(XNNWorkspace::new(WorkspacePtr::new(xnn_workspace_t(
            core::ptr::null_mut(),
        ))));
        XNNExecutor::new(ws)
    }

    // [spec:et:sem:xnn-executor.executorch.backends.xnnpack.delegate.xnn-executor.initialize-fn/test]
    // [spec:et:sem:xnn-executor.executorch.backends.xnnpack.delegate.xnn-executor.prepare-args-fn/test]
    #[test]
    #[ignore]
    fn argument_with_too_many_dimensions() {
        let mut executor = make_executor();
        let mut subgraph: xnn_subgraph_t = xnn_subgraph_t(core::ptr::null_mut());
        let mut rt: xnn_runtime_t = xnn_runtime_t(core::ptr::null_mut());
        unsafe {
            et_pal_init();
            assert_eq!(xnn_initialize(core::ptr::null()), xnn_status_success);
            assert_eq!(xnn_create_subgraph(2, 0, &mut subgraph), xnn_status_success);

            let mut input_id: u32 = XNN_INVALID_VALUE_ID;
            let dims: Vec<usize> = vec![1];
            assert_eq!(
                xnn_define_quantized_tensor_value(
                    subgraph,
                    xnn_datatype_qint8,
                    0,
                    1.0,
                    dims.len(),
                    dims.as_ptr(),
                    core::ptr::null(),
                    /*external_id=*/ 0,
                    /*flags=*/ XNN_VALUE_FLAG_EXTERNAL_INPUT,
                    &mut input_id,
                ),
                xnn_status_success
            );
            assert_ne!(input_id, XNN_INVALID_VALUE_ID);

            let mut output_id: u32 = XNN_INVALID_VALUE_ID;
            assert_eq!(
                xnn_define_quantized_tensor_value(
                    subgraph,
                    xnn_datatype_qint8,
                    0,
                    1.0,
                    dims.len(),
                    dims.as_ptr(),
                    core::ptr::null(),
                    /*external_id=*/ 1,
                    /*flags=*/ XNN_VALUE_FLAG_EXTERNAL_OUTPUT,
                    &mut output_id,
                ),
                xnn_status_success
            );
            assert_ne!(output_id, XNN_INVALID_VALUE_ID);

            assert_eq!(
                xnn_define_clamp(subgraph, 1.0, 2.0, input_id, output_id, 0),
                xnn_status_success
            );

            assert_eq!(xnn_create_runtime(subgraph, &mut rt), xnn_status_success);
        }

        assert_eq!(
            executor.initialize(rt, vec![0], vec![1], vec![], vec![], false),
            Error::Ok
        );

        let tf = TensorFactory::<i32>::new();
        let input_tensor = tf.make(
            vec![1, 1, 1, 1, 1, 1, 1, 1, 1],
            vec![42],
            Vec::new(),
            TensorShapeDynamism::STATIC,
        );
        assert_eq!(input_tensor.dim(), 9);
        let output_tensor = tf.make(vec![1], vec![1], Vec::new(), TensorShapeDynamism::STATIC);

        let mut input_ev = EValue::from_tensor(input_tensor);
        let mut output_ev = EValue::from_tensor(output_tensor);
        let mut args: [*mut EValue; 2] = [&mut input_ev, &mut output_ev];
        let stack_args: Span<*mut EValue> = Span::from_raw_parts(args.as_mut_ptr(), 2);

        // Check for invalid number of dimensions should fail without stack
        // overflow.
        assert_eq!(executor.prepare_args(stack_args), Error::InvalidArgument);

        unsafe {
            xnn_delete_subgraph(subgraph);
        }
    }

    // [spec:et:sem:xnn-executor.executorch.backends.xnnpack.delegate.xnn-executor.initialize-fn/test]
    // [spec:et:sem:xnn-executor.executorch.backends.xnnpack.delegate.xnn-executor.prepare-args-fn/test]
    // [spec:et:sem:xnn-executor.executorch.backends.xnnpack.delegate.xnn-executor.resize-outputs-fn/test]
    // [spec:et:sem:xnn-executor.executorch.backends.xnnpack.delegate.xnn-executor.forward-fn/test]
    // [spec:et:sem:xnn-executor.executorch.backends.xnnpack.delegate.xnn-executor.convert-outputs-fn/test]
    #[test]
    #[ignore]
    fn resize_outputs_with_long_tensor_converts_int32_to_int64() {
        let mut executor = make_executor();
        let mut rt: xnn_runtime_t = xnn_runtime_t(core::ptr::null_mut());
        let mut subgraph: xnn_subgraph_t = xnn_subgraph_t(core::ptr::null_mut());

        unsafe {
            et_pal_init();
            assert_eq!(xnn_initialize(core::ptr::null()), xnn_status_success);
            assert_eq!(xnn_create_subgraph(3, 0, &mut subgraph), xnn_status_success);

            let in_dims: Vec<usize> = vec![1, 4, 4, 4];
            let out_dims: Vec<usize> = vec![1, 2, 2, 4];
            let mut input_id: u32 = XNN_INVALID_VALUE_ID;
            let mut value_id: u32 = XNN_INVALID_VALUE_ID;
            let mut index_id: u32 = XNN_INVALID_VALUE_ID;

            assert_eq!(
                xnn_define_tensor_value(
                    subgraph,
                    xnn_datatype_fp32,
                    in_dims.len(),
                    in_dims.as_ptr(),
                    core::ptr::null(),
                    0,
                    XNN_VALUE_FLAG_EXTERNAL_INPUT,
                    &mut input_id,
                ),
                xnn_status_success
            );
            assert_eq!(
                xnn_define_tensor_value(
                    subgraph,
                    xnn_datatype_fp32,
                    out_dims.len(),
                    out_dims.as_ptr(),
                    core::ptr::null(),
                    1,
                    XNN_VALUE_FLAG_EXTERNAL_OUTPUT,
                    &mut value_id,
                ),
                xnn_status_success
            );
            assert_eq!(
                xnn_define_tensor_value(
                    subgraph,
                    xnn_datatype_int32,
                    out_dims.len(),
                    out_dims.as_ptr(),
                    core::ptr::null(),
                    2,
                    XNN_VALUE_FLAG_EXTERNAL_OUTPUT,
                    &mut index_id,
                ),
                xnn_status_success
            );
            assert_eq!(
                xnn_define_argmax_pooling_2d(
                    subgraph, 0, 0, 0, 0, 2, 2, input_id, value_id, index_id, 0
                ),
                xnn_status_success
            );

            assert_eq!(xnn_create_runtime(subgraph, &mut rt), xnn_status_success);
        }

        assert_eq!(
            executor.initialize(rt, vec![0], vec![1, 2], vec![], vec![], false),
            Error::Ok
        );

        let tf_float = TensorFactory::<f32>::new();
        let tf_long = TensorFactory::<i64>::new();

        // 4 channels to provide XNN_EXTRA_BYTES padding for SIMD reads.
        let mut input_data: Vec<f32> = vec![0.0; 4 * 4 * 4];
        for i in 0..16 {
            input_data[i * 4] = (i + 1) as f32; // channel 0 has values 1-16
        }
        let input = tf_float.make(
            vec![1, 4, 4, 4],
            input_data,
            Vec::new(),
            TensorShapeDynamism::STATIC,
        );
        let out_value = tf_float.make(
            vec![1, 2, 2, 4],
            vec![0.0; 16],
            Vec::new(),
            TensorShapeDynamism::STATIC,
        );
        let out_index = tf_long.make(
            vec![1, 2, 2, 4],
            vec![0i64; 16],
            Vec::new(),
            TensorShapeDynamism::STATIC,
        );

        let mut ev_in = EValue::from_tensor(input);
        let mut ev_val = EValue::from_tensor(out_value);
        let mut ev_idx = EValue::from_tensor(out_index);
        let mut args: [*mut EValue; 3] = [&mut ev_in, &mut ev_val, &mut ev_idx];
        let span: Span<*mut EValue> = Span::from_raw_parts(args.as_mut_ptr(), 3);

        assert_eq!(executor.prepare_args(span), Error::Ok);
        let mut context =
            BackendExecutionContext::new(null_event_tracer(), null_allocator(), core::ptr::null());
        assert_eq!(executor.forward(&mut context), Error::Ok);
        assert_eq!(executor.convert_outputs(span), Error::Ok);

        let result: &Tensor = unsafe { (*args[2]).to_tensor() };
        assert_eq!(result.scalar_type(), ScalarType::Long);

        // Verify all 4 spatial positions for channel 0 (channels 1-3 all zeros).
        // Output is NHWC {1,2,2,4}, so channel 0 is at indices 0, 4, 8, 12; each
        // 2x2 quadrant → index of max (3 = bottom-right).
        for i in 0..4 {
            assert_eq!(unsafe { *result.const_data_ptr::<i64>().add(i * 4) }, 3);
        }
    }

    // ------------------------------------------------------------------------
    // State-only executor + InUseGuard unit tests.
    //
    // PORT-NOTE: These have NO gtest counterpart (the C++ suite only exercises
    // the two runtime tests above). They cover the accessor / lifecycle surface
    // that touches only Rust state — no XNNPACK C call — so they run without the
    // link/graph-construction dependency the `#[ignore]`d tests carry. In the
    // default (non-profiling) `xnnpack` build, `XNNProfiler` is a ZST stub, so a
    // null-handle executor constructs and drops without any XNNPACK C call.
    // ------------------------------------------------------------------------

    // A freshly constructed executor exposes empty input/output/packed-name
    // views and does not report using a weight cache, matching the C++ ctor
    // (`getNumInputs`/`getNumOutputs` over empty `input_ids_`/`output_ids_`,
    // `get_packed_data_names` over an empty `packed_data_names_`,
    // `uses_weight_cache` = `!packed_data_names_.empty()`).
    // [spec:et:sem:xnn-executor.executorch.backends.xnnpack.delegate.xnn-executor.get-num-inputs-fn/test]
    // [spec:et:sem:xnn-executor.executorch.backends.xnnpack.delegate.xnn-executor.get-num-outputs-fn/test]
    // [spec:et:sem:xnn-executor.executorch.backends.xnnpack.delegate.xnn-executor.get-packed-data-names-fn/test]
    // [spec:et:sem:xnn-executor.executorch.backends.xnnpack.delegate.xnn-executor.uses-weight-cache-fn/test]
    #[test]
    fn fresh_executor_has_empty_state() {
        let executor = make_executor_null();
        assert_eq!(executor.getNumInputs(), 0);
        assert_eq!(executor.getNumOutputs(), 0);
        assert!(executor.get_packed_data_names().is_empty());
        assert!(!executor.uses_weight_cache());
    }

    // `get_workspace` returns the same shared workspace the executor was
    // constructed with (C++ returns `workspace_` by shared_ptr copy).
    // [spec:et:sem:xnn-executor.executorch.backends.xnnpack.delegate.xnn-executor.get-workspace-fn/test]
    #[test]
    fn get_workspace_returns_construction_workspace() {
        let ws = Arc::new(XNNWorkspace::new(WorkspacePtr::new(xnn_workspace_t(
            core::ptr::null_mut(),
        ))));
        let executor = XNNExecutor::new(ws.clone());
        let got = executor.get_workspace();
        assert!(Arc::ptr_eq(&ws, &got));
    }

    // `set_weights_cache(Some)` then `get_weights_cache` round-trips the same
    // cache; `set_weights_cache(None)` (the empty-shared_ptr case) clears it.
    // [spec:et:sem:xnn-executor.executorch.backends.xnnpack.delegate.xnn-executor.set-weights-cache-fn/test]
    // [spec:et:sem:xnn-executor.executorch.backends.xnnpack.delegate.xnn-executor.get-weights-cache-fn/test]
    #[test]
    fn weights_cache_round_trips() {
        let mut executor = make_executor_null();
        assert!(executor.get_weights_cache().is_none());

        let cache = Arc::new(XNNWeightsCache::new());
        executor.set_weights_cache(Some(cache.clone()));
        let got = executor
            .get_weights_cache()
            .expect("cache should be present");
        assert!(Arc::ptr_eq(&cache, &got));

        executor.set_weights_cache(None);
        assert!(executor.get_weights_cache().is_none());
    }

    // The XNNExecutor destructor release-stores `true` into `destroyed_`. Not
    // observable after drop, so instead assert the flag is unset while alive
    // (`dbg_destroyed`) and that the normal (not-in-use) drop runs its dcheck
    // without aborting.
    // [spec:et:sem:xnn-executor.executorch.backends.xnnpack.delegate.xnn-executor.xnn-executor-fn/test]
    #[test]
    fn destructor_normal_path_when_not_in_use() {
        let executor = make_executor_null();
        assert!(!executor.dbg_destroyed());
        drop(executor);
    }

    // The destructor debug-asserts `!in_use_`; dropping while in-use must abort
    // (C++ `ET_DCHECK_MSG`). `et_dcheck_msg!` is a `debug_assert!`, so this only
    // fires in debug builds.
    // [spec:et:sem:xnn-executor.executorch.backends.xnnpack.delegate.xnn-executor.xnn-executor-fn/test]
    #[cfg(debug_assertions)]
    #[test]
    #[should_panic]
    fn destructor_aborts_when_in_use() {
        let executor = make_executor_null();
        executor.in_use_.store(true, Ordering::Release);
        drop(executor);
    }

    // Dropping a non-dismissed InUseGuard release-stores `false` into the flag it
    // borrows (C++ `~InUseGuard`).
    // [spec:et:sem:xnn-executor.executorch.backends.xnnpack.delegate.in-use-guard.in-use-guard-fn/test]
    #[test]
    fn in_use_guard_releases_flag_on_drop() {
        let flag = AtomicBool::new(true);
        {
            let _guard = InUseGuard::new(&flag);
        }
        assert!(!flag.load(Ordering::Acquire));
    }

    // A dismissed guard does NOT touch the flag on drop (C++ `dismiss` sets the
    // `dismissed_` bit consulted by the destructor).
    // [spec:et:sem:xnn-executor.executorch.backends.xnnpack.delegate.in-use-guard.dismiss-fn/test]
    #[test]
    fn dismissed_in_use_guard_leaves_flag_untouched() {
        let flag = AtomicBool::new(true);
        {
            let mut guard = InUseGuard::new(&flag);
            guard.dismiss();
        }
        assert!(flag.load(Ordering::Acquire));
    }

    // The deleted copy-assign (`operator=`) collapses onto the move-only guard
    // in Rust (no `Copy`/`Clone`). Moving the guard preserves its single-release
    // contract: the flag is released exactly once, by the final owner's drop, and
    // not by the moved-from binding.
    // [spec:et:sem:xnn-executor.executorch.backends.xnnpack.delegate.in-use-guard.operator-fn/test]
    #[test]
    fn in_use_guard_is_move_only_single_release() {
        let flag = AtomicBool::new(true);
        {
            let guard = InUseGuard::new(&flag);
            let moved = guard;
            // Flag not released by the move itself; still set while `moved` lives.
            assert!(flag.load(Ordering::Acquire));
            drop(moved);
            assert!(!flag.load(Ordering::Acquire));
        }
    }
}
