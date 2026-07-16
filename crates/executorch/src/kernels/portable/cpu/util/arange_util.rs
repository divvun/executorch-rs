//! Literal port of kernels/portable/cpu/util/arange_util.cpp + kernels/portable/cpu/util/arange_util.h.

use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::core::portable_type::tensor_impl::SizesType;
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

// PORT-NOTE: `ET_SWITCH_REALHBF16_TYPES` is provided by the crate-exported
// `et_switch_realhbf16_types!` in scalar_type_util.rs (REALHBF16 = real +
// Half + BFloat16, no Bool, no complex); imported here for unqualified use.
use crate::et_switch_realhbf16_types;

// PORT-NOTE: `ET_CHECK_MSG` mirrors the C++ fatal check. There is no shared
// runtime macro; each module defines a local one (see tensor_util.rs). Message
// formatting is dropped since a fatal abort follows.
macro_rules! et_check_msg {
    ($cond:expr, $($arg:tt)*) => {
        if !($cond) {
            crate::runtime::platform::abort::runtime_abort();
        }
    };
}

// PORT-NOTE: the C++ `ET_ARANGE_IMPL(...)` macro is expanded inline at each of
// its two use sites (the four-arg and two-arg forms). Rust `macro_rules!` cannot
// re-invoke a crate-exported dtype-switch macro through another opaque local
// macro cleanly, so the switch body is written out at each call site verbatim,
// matching the C++ macro expansion.

// [spec:et:def:arange-util.torch.executor.native.compute-arange-out-size-fn]
// [spec:et:sem:arange-util.torch.executor.native.compute-arange-out-size-fn]
pub fn compute_arange_out_size(start: f64, end: f64, step: f64) -> SizesType {
    let numel: SizesType = ((end - start) / step).ceil() as SizesType;

    et_check_msg!(
        numel >= 0,
        "numel should be non-negative, but got ({}). start ({}), end ({}), step ({})",
        numel as i64,
        start,
        end,
        step
    );
    numel
}

// [spec:et:def:arange-util.torch.executor.native.compute-arange-out-size-fn]
// [spec:et:sem:arange-util.torch.executor.native.compute-arange-out-size-fn]
//
// Inline convenience overload `compute_arange_out_size(end)`.
pub fn compute_arange_out_size_end(end: f64) -> SizesType {
    compute_arange_out_size(0.0, end, 1.0)
}

// [spec:et:def:arange-util.torch.executor.native.arange-out-impl-fn]
// [spec:et:sem:arange-util.torch.executor.native.arange-out-impl-fn]
pub fn arange_out_impl(
    ctx: &mut KernelRuntimeContext,
    start: f64,
    end: f64,
    step: f64,
    out: &Tensor,
) {
    let _ = &ctx;
    let numel: SizesType = compute_arange_out_size(start, end, step);
    et_switch_realhbf16_types!(out.scalar_type(), ctx, "arange.start_out", CTYPE, {
        let out_data = out.mutable_data_ptr::<CTYPE>();
        let mut i: SizesType = 0;
        while i < numel {
            unsafe {
                *out_data.add(i as usize) =
                    <f64 as crate::extension::tensor::tensor_ptr::NumericCast<CTYPE>>::numeric_cast(
                        start + i as f64 * step,
                    );
            }
            i += 1;
        }
    });
}

// [spec:et:def:arange-util.torch.executor.native.arange-out-impl-fn]
// [spec:et:sem:arange-util.torch.executor.native.arange-out-impl-fn]
//
// Two-argument form `arange_out_impl(ctx, end, out)`.
pub fn arange_out_impl_end(ctx: &mut KernelRuntimeContext, end: f64, out: &Tensor) {
    let _ = &ctx;
    let numel: SizesType = compute_arange_out_size(0.0, end, 1.0);
    et_switch_realhbf16_types!(out.scalar_type(), ctx, "arange.out", CTYPE, {
        let out_data = out.mutable_data_ptr::<CTYPE>();
        let mut i: SizesType = 0;
        while i < numel {
            unsafe {
                *out_data.add(i as usize) =
                    <f64 as crate::extension::tensor::tensor_ptr::NumericCast<CTYPE>>::numeric_cast(
                        0.0 + i as f64 * 1.0,
                    );
            }
            i += 1;
        }
    });
}

// PORT-NOTE: header inline convenience overloads that omit `ctx` construct a
// fresh default-constructed `KernelRuntimeContext ctx;`. The ported
// `KernelRuntimeContext::new` requires two raw `dyn` pointers (no default
// arguments in Rust). Null `dyn` fat pointers are built here — the event-tracer
// slot reuses the existing crate-internal `module::null_event_tracer()` helper
// and the allocator slot casts a null concrete `MemoryAllocator` pointer,
// mirroring the established `core::ptr::null_mut::<Concrete>() as *mut dyn Trait`
// pattern (runtime/backend/backend_init_context.rs). A fat pointer's null-ness
// is determined by its data component; this context is never dereferenced.
mod null_ctx {
    use crate::runtime::core::memory_allocator::{MemoryAllocator, MemoryAllocatorBase};

    pub fn default_context()
    -> crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext<'static> {
        crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext::new(
            crate::extension::module::module::null_event_tracer(),
            core::ptr::null_mut::<MemoryAllocator>() as *mut dyn MemoryAllocatorBase,
        )
    }
}

// [spec:et:def:arange-util.torch.executor.native.arange-out-impl-fn]
// [spec:et:sem:arange-util.torch.executor.native.arange-out-impl-fn]
//
// Inline convenience overload `arange_out_impl(start, end, step, out)`.
pub fn arange_out_impl_no_ctx(start: f64, end: f64, step: f64, out: &Tensor) {
    let mut ctx = null_ctx::default_context();
    arange_out_impl(&mut ctx, start, end, step, out);
}

// [spec:et:def:arange-util.torch.executor.native.arange-out-impl-fn]
// [spec:et:sem:arange-util.torch.executor.native.arange-out-impl-fn]
//
// Inline convenience overload `arange_out_impl(end, out)`.
pub fn arange_out_impl_end_no_ctx(end: f64, out: &Tensor) {
    let mut ctx = null_ctx::default_context();
    arange_out_impl(&mut ctx, 0.0, end, 1.0, out);
}
