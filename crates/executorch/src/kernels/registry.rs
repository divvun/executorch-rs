//! Kernel registration table assembled from `#[et_kernel]`-annotated kernels.
//!
//! Each `#[executorch_macros::et_kernel("aten::op.overload")]` on a kernel fn
//! emits an unboxing wrapper and a [`KernelReg`] entry into the [`ET_KERNELS`]
//! distributed slice (via `linkme`). [`register_all`] walks the slice and
//! registers every entry with the global operator registry — the runtime
//! equivalent of the C++ `RegisterKernels` codegen output.
//!
//! Prim ops (`aten::sym_size.int`, `executorch_prim::et_view.default`) don't fit
//! the signature-derived unboxing (they act on `EValue`s directly), so they are
//! hand-written here and registered into the same slice manually.

use linkme::distributed_slice;

use crate::runtime::core::array_ref::ArrayRef;
use crate::runtime::core::error::Error;
use crate::runtime::core::evalue::EValue;
use crate::runtime::core::exec_aten::util::tensor_util::{
    K_TENSOR_DIMENSION_LIMIT, internal::set_tensor_data, resize_tensor,
};
use crate::runtime::core::portable_type::scalar::Scalar;
use crate::runtime::core::portable_type::scalar_type::ScalarType;
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::core::portable_type::tensor_impl::SizesType;
use crate::runtime::core::portable_type::tensor_options::MemoryFormat;
use crate::runtime::core::span::Span;
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;
use crate::runtime::kernel::operator_registry::{Kernel, OpFunction, register_kernels};

/// One registered kernel: operator name + its unboxing `OpFunction`. `Sync`
/// (both fields are), so it can live in a `linkme` distributed-slice static.
pub struct KernelReg {
    pub name: &'static core::ffi::CStr,
    pub op: OpFunction,
}

/// Every `#[et_kernel]`-annotated kernel plus the manual prim ops below land
/// here, gathered by the linker across the whole crate.
#[distributed_slice]
pub static ET_KERNELS: [KernelReg];

/// Register every kernel in [`ET_KERNELS`] with the global operator registry.
/// Call once (re-registering the same name aborts).
#[must_use]
pub fn register_all() -> Error {
    let mut kernels: Vec<Kernel> = ET_KERNELS
        .iter()
        .map(|r| Kernel::new_fallback(r.name.as_ptr(), r.op))
        .collect();
    if kernels.is_empty() {
        return Error::Ok;
    }
    register_kernels(Span::from_raw_parts(kernels.as_mut_ptr(), kernels.len()))
}

// ---- accessors used by generated unboxing wrappers ----

/// `&EValue` at stack position `i`.
#[inline]
pub fn arg<'a>(stack: Span<*mut EValue<'a>>, i: usize) -> &'a EValue<'a> {
    unsafe { &*(*stack.index(i)) }
}

/// `&mut EValue` at stack position `i` (for ops that write a scalar output).
#[inline]
pub fn arg_mut<'a>(stack: Span<*mut EValue<'a>>, i: usize) -> &'a mut EValue<'a> {
    unsafe { &mut *(*stack.index(i)) }
}

pub fn memory_format_from_int(v: i64) -> MemoryFormat {
    match v {
        1 => MemoryFormat::Preserve,
        _ => MemoryFormat::Contiguous,
    }
}

pub fn scalar_type_from_int(v: i64) -> ScalarType {
    // ScalarType is #[repr(i8)] with contiguous discriminants 0..=30.
    if (0..=30).contains(&v) {
        unsafe { core::mem::transmute::<i8, ScalarType>(v as i8) }
    } else {
        ScalarType::Undefined
    }
}

pub fn dtype(e: &EValue) -> ScalarType {
    scalar_type_from_int(e.to_int())
}
pub fn opt_scalar(e: &EValue) -> Option<Scalar> {
    if e.is_none() {
        None
    } else {
        Some(e.to_scalar())
    }
}
pub fn opt_int(e: &EValue) -> Option<i64> {
    if e.is_none() { None } else { Some(e.to_int()) }
}
pub fn opt_memory_format(e: &EValue) -> Option<MemoryFormat> {
    if e.is_none() {
        None
    } else {
        Some(memory_format_from_int(e.to_int()))
    }
}
pub fn opt_dtype(e: &EValue) -> Option<ScalarType> {
    if e.is_none() {
        None
    } else {
        Some(scalar_type_from_int(e.to_int()))
    }
}
pub fn opt_int_list(e: &EValue) -> Option<ArrayRef<i64>> {
    if e.is_none() {
        None
    } else {
        Some(e.to_int_list())
    }
}
pub fn opt_tensor<'a>(e: &'a EValue<'a>) -> Option<&'a Tensor<'a>> {
    if e.is_none() {
        None
    } else {
        Some(e.to_tensor())
    }
}

/// Like [`opt_tensor`] but yields an owned `Tensor` handle, so callers can bind
/// it to a local and pass `&Option<Tensor>` (the shape most `*_out` kernels want
/// for optional-tensor args like a convolution bias). `Tensor` is a copyable
/// handle over the same `TensorImpl`, so this aliases rather than copies data.
pub fn opt_tensor_owned<'a>(e: &'a EValue<'a>) -> Option<Tensor<'a>> {
    if e.is_none() {
        None
    } else {
        Some(Tensor::new(e.to_tensor().unsafe_get_tensor_impl()))
    }
}

// ---- prim ops (act on EValues directly; hand-written, registered manually) ----

/// `aten::sym_size.int(Tensor self, int dim) -> SymInt`.
fn sym_size_int_shim(ctx: &mut KernelRuntimeContext, stack: Span<*mut EValue>) {
    if stack.size() != 3 {
        ctx.fail(Error::InvalidArgument);
        return;
    }
    let self_ = arg(stack, 0).to_tensor();
    let dim = arg(stack, 1).to_int();
    let size = self_.size(dim as isize) as i64;
    *arg_mut(stack, 2) = EValue::from_int(size);
}

/// Resolve a view `size` (with at most one `-1`) into concrete dims. Port of
/// `get_view_target_size` in the C++ `et_view.cpp`.
fn view_target_size(
    self_numel: i64,
    size: ArrayRef<i64>,
    dim: usize,
    out: &mut [SizesType],
) -> bool {
    if size.size() != dim {
        return false;
    }
    let mut minus1_dim: i64 = -1;
    let mut numel_without_minus1: i64 = 1;
    for i in 0..dim {
        let s = *size.at(i);
        if s == -1 {
            if minus1_dim != -1 {
                return false;
            }
            minus1_dim = i as i64;
        } else if s < -1 {
            return false;
        } else {
            numel_without_minus1 *= s;
            out[i] = s as SizesType;
        }
    }
    if minus1_dim >= 0 {
        if numel_without_minus1 == 0 {
            return false;
        }
        out[minus1_dim as usize] = (self_numel / numel_without_minus1) as SizesType;
    }
    true
}

/// `executorch_prim::et_view.default(Tensor self, int[] size) -> Tensor`: make
/// `out` a view of `self`'s storage with the resolved shape. Port of `et_view.cpp`.
fn et_view_shim(ctx: &mut KernelRuntimeContext, stack: Span<*mut EValue>) {
    if stack.size() != 3 {
        ctx.fail(Error::InvalidArgument);
        return;
    }
    let self_ = arg(stack, 0).to_tensor();
    let size = arg(stack, 1).to_int_list();
    let out = arg(stack, 2).to_tensor();

    if self_.scalar_type() != out.scalar_type() {
        ctx.fail(Error::InvalidArgument);
        return;
    }
    let dim = out.dim() as usize;
    if dim > K_TENSOR_DIMENSION_LIMIT {
        ctx.fail(Error::InvalidArgument);
        return;
    }
    let mut expected = [0 as SizesType; K_TENSOR_DIMENSION_LIMIT];
    if !view_target_size(self_.numel() as i64, size, dim, &mut expected) {
        ctx.fail(Error::InvalidArgument);
        return;
    }
    if resize_tensor(out, ArrayRef::from_raw_parts(expected.as_ptr(), dim)) != Error::Ok {
        ctx.fail(Error::Internal);
        return;
    }
    if self_.numel() != out.numel() {
        ctx.fail(Error::InvalidArgument);
        return;
    }
    if set_tensor_data(out, self_.mutable_data_ptr_typed(), out.nbytes()) != Error::Ok {
        ctx.fail(Error::Internal);
    }
}

/// `sin_out` is generated by a ufunc macro (no `pub fn` for `#[et_kernel]` to
/// attach to), so its unboxing wrapper is hand-written here.
fn sin_out_shim(ctx: &mut KernelRuntimeContext, stack: Span<*mut EValue>) {
    let _ = crate::kernels::portable::cpu::op_sin::sin_out(
        ctx,
        arg(stack, 0).to_tensor(),
        arg(stack, 1).to_tensor(),
    );
}

#[distributed_slice(ET_KERNELS)]
static __ET_REG_SIN: KernelReg = KernelReg {
    name: c"aten::sin.out",
    op: sin_out_shim,
};

#[distributed_slice(ET_KERNELS)]
static __ET_REG_SYM_SIZE: KernelReg = KernelReg {
    name: c"aten::sym_size.int",
    op: sym_size_int_shim,
};

#[distributed_slice(ET_KERNELS)]
static __ET_REG_ET_VIEW: KernelReg = KernelReg {
    name: c"executorch_prim::et_view.default",
    op: et_view_shim,
};
