//! Registration glue for the optimized CPU kernels.
//!
//! The optimized ops override the portable set: registering them makes
//! `Method::execute` dispatch to the fast implementations here instead of the
//! portable ones. This is the Rust analogue of the C++ `functions.yaml` codegen
//! + `optimized_native_cpu_ops_lib`.
//!
//! PORT-NOTE (codegen deviation): the C++ generates the unboxing wrappers and
//! the operator-name -> function table from `optimized/optimized.yaml`. The port
//! hand-writes that layer, mirroring `kernels::portable::register` — one
//! `OpFunction` shim per optimized op plus [`register_all`].

use crate::runtime::core::array_ref::ArrayRef;
use crate::runtime::core::error::Error;
use crate::runtime::core::evalue::EValue;
use crate::runtime::core::portable_type::scalar_type::ScalarType;
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::core::span::Span;
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;
use crate::runtime::kernel::operator_registry::{Kernel, OpFunction, register_kernels};

use crate::kernels::optimized::cpu::op_add::{opt_add_out, opt_add_scalar_out};
use crate::kernels::optimized::cpu::op_bmm::opt_bmm_out;
use crate::kernels::optimized::cpu::op_div::{opt_div_out, opt_div_scalar_out};
use crate::kernels::optimized::cpu::op_elu::opt_elu_out;
use crate::kernels::optimized::cpu::op_exp::opt_exp_out;
use crate::kernels::optimized::cpu::op_fft_c2r::opt_fft_c2r_out;
use crate::kernels::optimized::cpu::op_fft_r2c::opt_fft_r2c_out;
use crate::kernels::optimized::cpu::op_gelu::opt_gelu_out;
use crate::kernels::optimized::cpu::op_grid_sampler_2d::opt_grid_sampler_2d_out;
use crate::kernels::optimized::cpu::op_le::{opt_le_scalar_out, opt_le_tensor_out};
use crate::kernels::optimized::cpu::op_linear::opt_linear_out;
use crate::kernels::optimized::cpu::op_log_softmax::opt_log_softmax_out;
use crate::kernels::optimized::cpu::op_mm::opt_mm_out;
use crate::kernels::optimized::cpu::op_mul::{opt_mul_out, opt_mul_scalar_out};
use crate::kernels::optimized::cpu::op_native_layer_norm::opt_native_layer_norm_out;
use crate::kernels::optimized::cpu::op_sub::{opt_sub_out, opt_sub_scalar_out};
use crate::kernels::optimized::cpu::op_sum::opt_sum_dim_out;
use crate::kernels::optimized::cpu::op_where::opt_where_out;

/// `&EValue` at stack position `$i`.
macro_rules! ev {
    ($stack:expr, $i:expr) => {
        unsafe { &*(*$stack.index($i)) }
    };
}

fn scalar_type_from_int(v: i64) -> ScalarType {
    // ScalarType is #[repr(i8)] with contiguous discriminants 0..=30.
    if (0..=30).contains(&v) {
        unsafe { core::mem::transmute::<i8, ScalarType>(v as i8) }
    } else {
        ScalarType::Undefined
    }
}

fn opt_dtype(e: &EValue) -> Option<ScalarType> {
    if e.is_none() {
        None
    } else {
        Some(scalar_type_from_int(e.to_int()))
    }
}
fn opt_int_list(e: &EValue) -> Option<ArrayRef<i64>> {
    if e.is_none() {
        None
    } else {
        Some(e.to_int_list())
    }
}
fn opt_tensor<'a>(e: &'a EValue<'a>) -> Option<&'a Tensor<'a>> {
    if e.is_none() {
        None
    } else {
        Some(e.to_tensor())
    }
}

// ---- unboxing shims (stack order = kernel arg order, out tensor(s) last) ----

fn opt_sum_dim_shim(ctx: &mut KernelRuntimeContext, stack: Span<*mut EValue>) {
    opt_sum_dim_out(
        ctx,
        ev!(stack, 0).to_tensor(),
        opt_int_list(ev!(stack, 1)),
        ev!(stack, 2).to_bool(),
        opt_dtype(ev!(stack, 3)),
        ev!(stack, 4).to_tensor(),
    );
}

// _log_softmax.out(Tensor self, int dim, bool half_to_float, *, Tensor(a!) out)
fn opt_log_softmax_shim(ctx: &mut KernelRuntimeContext, stack: Span<*mut EValue>) {
    opt_log_softmax_out(
        ctx,
        ev!(stack, 0).to_tensor(),
        ev!(stack, 1).to_int(),
        ev!(stack, 2).to_bool(),
        ev!(stack, 3).to_tensor(),
    );
}

// elu.out(Tensor self, Scalar alpha, Scalar scale, Scalar input_scale, *, Tensor(a!) out)
fn opt_elu_shim(ctx: &mut KernelRuntimeContext, stack: Span<*mut EValue>) {
    opt_elu_out(
        ctx,
        ev!(stack, 0).to_tensor(),
        &ev!(stack, 1).to_scalar(),
        &ev!(stack, 2).to_scalar(),
        &ev!(stack, 3).to_scalar(),
        ev!(stack, 4).to_tensor(),
    );
}

// exp.out(Tensor self, *, Tensor(a!) out)
fn opt_exp_shim(ctx: &mut KernelRuntimeContext, stack: Span<*mut EValue>) {
    opt_exp_out(ctx, ev!(stack, 0).to_tensor(), ev!(stack, 1).to_tensor());
}

// gelu.out(Tensor self, *, str approximate, Tensor(a!) out)
fn opt_gelu_shim(ctx: &mut KernelRuntimeContext, stack: Span<*mut EValue>) {
    opt_gelu_out(
        ctx,
        ev!(stack, 0).to_tensor(),
        ev!(stack, 1).to_string(),
        ev!(stack, 2).to_tensor(),
    );
}

fn opt_native_layer_norm_shim(ctx: &mut KernelRuntimeContext, stack: Span<*mut EValue>) {
    opt_native_layer_norm_out(
        ctx,
        ev!(stack, 0).to_tensor(),
        ev!(stack, 1).to_int_list(),
        opt_tensor(ev!(stack, 2)),
        opt_tensor(ev!(stack, 3)),
        ev!(stack, 4).to_double(),
        ev!(stack, 5).to_tensor(),
        ev!(stack, 6).to_tensor(),
        ev!(stack, 7).to_tensor(),
    );
}

// add.out(Tensor self, Tensor other, *, Scalar alpha=1, Tensor(a!) out)
fn opt_add_out_shim(ctx: &mut KernelRuntimeContext, stack: Span<*mut EValue>) {
    let alpha = ev!(stack, 2).to_scalar();
    opt_add_out(
        ctx,
        ev!(stack, 0).to_tensor(),
        ev!(stack, 1).to_tensor(),
        &alpha,
        ev!(stack, 3).to_tensor(),
    );
}

// add.Scalar_out(Tensor self, Scalar other, Scalar alpha=1, Tensor(a!) out)
fn opt_add_scalar_out_shim(ctx: &mut KernelRuntimeContext, stack: Span<*mut EValue>) {
    let b = ev!(stack, 1).to_scalar();
    let alpha = ev!(stack, 2).to_scalar();
    opt_add_scalar_out(
        ctx,
        ev!(stack, 0).to_tensor(),
        &b,
        &alpha,
        ev!(stack, 3).to_tensor(),
    );
}

// sub.out(Tensor self, Tensor other, *, Scalar alpha=1, Tensor(a!) out)
fn opt_sub_out_shim(ctx: &mut KernelRuntimeContext, stack: Span<*mut EValue>) {
    let alpha = ev!(stack, 2).to_scalar();
    opt_sub_out(
        ctx,
        ev!(stack, 0).to_tensor(),
        ev!(stack, 1).to_tensor(),
        &alpha,
        ev!(stack, 3).to_tensor(),
    );
}

// sub.Scalar_out(Tensor self, Scalar other, Scalar alpha=1, Tensor(a!) out)
fn opt_sub_scalar_out_shim(ctx: &mut KernelRuntimeContext, stack: Span<*mut EValue>) {
    let b = ev!(stack, 1).to_scalar();
    let alpha = ev!(stack, 2).to_scalar();
    opt_sub_scalar_out(
        ctx,
        ev!(stack, 0).to_tensor(),
        &b,
        &alpha,
        ev!(stack, 3).to_tensor(),
    );
}

// mul.out(Tensor self, Tensor other, Tensor(a!) out)
fn opt_mul_out_shim(ctx: &mut KernelRuntimeContext, stack: Span<*mut EValue>) {
    opt_mul_out(
        ctx,
        ev!(stack, 0).to_tensor(),
        ev!(stack, 1).to_tensor(),
        ev!(stack, 2).to_tensor(),
    );
}

// mul.Scalar_out(Tensor self, Scalar other, Tensor(a!) out)
fn opt_mul_scalar_out_shim(ctx: &mut KernelRuntimeContext, stack: Span<*mut EValue>) {
    let b = ev!(stack, 1).to_scalar();
    opt_mul_scalar_out(
        ctx,
        ev!(stack, 0).to_tensor(),
        &b,
        ev!(stack, 2).to_tensor(),
    );
}

// div.out(Tensor self, Tensor other, Tensor(a!) out)
fn opt_div_out_shim(ctx: &mut KernelRuntimeContext, stack: Span<*mut EValue>) {
    opt_div_out(
        ctx,
        ev!(stack, 0).to_tensor(),
        ev!(stack, 1).to_tensor(),
        ev!(stack, 2).to_tensor(),
    );
}

// div.Scalar_out(Tensor self, Scalar other, Tensor(a!) out)
fn opt_div_scalar_out_shim(ctx: &mut KernelRuntimeContext, stack: Span<*mut EValue>) {
    let b = ev!(stack, 1).to_scalar();
    opt_div_scalar_out(
        ctx,
        ev!(stack, 0).to_tensor(),
        &b,
        ev!(stack, 2).to_tensor(),
    );
}

// le.Tensor_out(Tensor self, Tensor other, Tensor(a!) out)
fn opt_le_tensor_out_shim(ctx: &mut KernelRuntimeContext, stack: Span<*mut EValue>) {
    opt_le_tensor_out(
        ctx,
        ev!(stack, 0).to_tensor(),
        ev!(stack, 1).to_tensor(),
        ev!(stack, 2).to_tensor(),
    );
}

// le.Scalar_out(Tensor self, Scalar other, Tensor(a!) out)
fn opt_le_scalar_out_shim(ctx: &mut KernelRuntimeContext, stack: Span<*mut EValue>) {
    let b = ev!(stack, 1).to_scalar();
    opt_le_scalar_out(
        ctx,
        ev!(stack, 0).to_tensor(),
        &b,
        ev!(stack, 2).to_tensor(),
    );
}

// grid_sampler_2d.out(Tensor input, Tensor grid, int interpolation_mode,
//   int padding_mode, bool align_corners, *, Tensor(a!) out)
fn opt_grid_sampler_2d_out_shim(ctx: &mut KernelRuntimeContext, stack: Span<*mut EValue>) {
    opt_grid_sampler_2d_out(
        ctx,
        ev!(stack, 0).to_tensor(),
        ev!(stack, 1).to_tensor(),
        ev!(stack, 2).to_int(),
        ev!(stack, 3).to_int(),
        ev!(stack, 4).to_bool(),
        ev!(stack, 5).to_tensor(),
    );
}

// where.self_out(Tensor condition, Tensor self, Tensor other, Tensor(a!) out)
fn opt_where_self_out_shim(ctx: &mut KernelRuntimeContext, stack: Span<*mut EValue>) {
    opt_where_out(
        ctx,
        ev!(stack, 0).to_tensor(),
        ev!(stack, 1).to_tensor(),
        ev!(stack, 2).to_tensor(),
        ev!(stack, 3).to_tensor(),
    );
}

// mm.out(Tensor self, Tensor mat2, *, Tensor(a!) out)
fn opt_mm_out_shim(ctx: &mut KernelRuntimeContext, stack: Span<*mut EValue>) {
    opt_mm_out(
        ctx,
        ev!(stack, 0).to_tensor(),
        ev!(stack, 1).to_tensor(),
        ev!(stack, 2).to_tensor(),
    );
}

// bmm.out(Tensor self, Tensor mat2, *, Tensor(a!) out)
fn opt_bmm_out_shim(ctx: &mut KernelRuntimeContext, stack: Span<*mut EValue>) {
    opt_bmm_out(
        ctx,
        ev!(stack, 0).to_tensor(),
        ev!(stack, 1).to_tensor(),
        ev!(stack, 2).to_tensor(),
    );
}

// linear.out(Tensor input, Tensor weight, Tensor? bias=None, *, Tensor(a!) out)
//
// PORT-NOTE: `opt_linear_out` takes `bias: &Option<Tensor>` (unlike
// `native_layer_norm`'s `Option<&Tensor>`), so the optional bias EValue is
// materialized into an owned `Option<Tensor>`. `Tensor` is a non-owning handle
// over a `TensorImpl*`, so re-wrapping the borrowed impl pointer is a pure alias
// (no copy), matching the C++ where the optional wraps the same `TensorImpl`.
fn opt_linear_out_shim(ctx: &mut KernelRuntimeContext, stack: Span<*mut EValue>) {
    let bias_ev = ev!(stack, 2);
    let bias = if bias_ev.is_none() {
        None
    } else {
        Some(Tensor::new(bias_ev.to_tensor().unsafe_get_tensor_impl()))
    };
    opt_linear_out(
        ctx,
        ev!(stack, 0).to_tensor(),
        ev!(stack, 1).to_tensor(),
        &bias,
        ev!(stack, 3).to_tensor(),
    );
}

// _fft_r2c.out(Tensor self, int[] dim, int normalization, bool onesided, *, Tensor(a!) out)
fn opt_fft_r2c_out_shim(ctx: &mut KernelRuntimeContext, stack: Span<*mut EValue>) {
    opt_fft_r2c_out(
        ctx,
        ev!(stack, 0).to_tensor(),
        ev!(stack, 1).to_int_list(),
        ev!(stack, 2).to_int(),
        ev!(stack, 3).to_bool(),
        ev!(stack, 4).to_tensor(),
    );
}

// _fft_c2r.out(Tensor self, int[] dim, int normalization, int last_dim_size, *, Tensor(a!) out)
fn opt_fft_c2r_out_shim(ctx: &mut KernelRuntimeContext, stack: Span<*mut EValue>) {
    opt_fft_c2r_out(
        ctx,
        ev!(stack, 0).to_tensor(),
        ev!(stack, 1).to_int_list(),
        ev!(stack, 2).to_int(),
        ev!(stack, 3).to_int(),
        ev!(stack, 4).to_tensor(),
    );
}

/// The optimized-CPU operator shims, one [`Kernel`] per op name in
/// `optimized.yaml`, each keyed by a fallback (dtype/dim-order-agnostic) key.
/// Shared by [`register_all`] and the merged [`super::register`] seam.
pub(crate) fn optimized_kernels() -> [Kernel; 23] {
    [
        Kernel::new_fallback(
            c"aten::sum.IntList_out".as_ptr(),
            opt_sum_dim_shim as OpFunction,
        ),
        Kernel::new_fallback(
            c"aten::native_layer_norm.out".as_ptr(),
            opt_native_layer_norm_shim,
        ),
        Kernel::new_fallback(c"aten::_log_softmax.out".as_ptr(), opt_log_softmax_shim),
        Kernel::new_fallback(c"aten::elu.out".as_ptr(), opt_elu_shim),
        Kernel::new_fallback(c"aten::exp.out".as_ptr(), opt_exp_shim),
        Kernel::new_fallback(c"aten::gelu.out".as_ptr(), opt_gelu_shim),
        Kernel::new_fallback(c"aten::add.out".as_ptr(), opt_add_out_shim),
        Kernel::new_fallback(c"aten::add.Scalar_out".as_ptr(), opt_add_scalar_out_shim),
        Kernel::new_fallback(c"aten::sub.out".as_ptr(), opt_sub_out_shim),
        Kernel::new_fallback(c"aten::sub.Scalar_out".as_ptr(), opt_sub_scalar_out_shim),
        Kernel::new_fallback(c"aten::mul.out".as_ptr(), opt_mul_out_shim),
        Kernel::new_fallback(c"aten::mul.Scalar_out".as_ptr(), opt_mul_scalar_out_shim),
        Kernel::new_fallback(c"aten::div.out".as_ptr(), opt_div_out_shim),
        Kernel::new_fallback(c"aten::div.Scalar_out".as_ptr(), opt_div_scalar_out_shim),
        Kernel::new_fallback(c"aten::mm.out".as_ptr(), opt_mm_out_shim),
        Kernel::new_fallback(c"aten::bmm.out".as_ptr(), opt_bmm_out_shim),
        Kernel::new_fallback(c"aten::linear.out".as_ptr(), opt_linear_out_shim),
        Kernel::new_fallback(c"aten::le.Tensor_out".as_ptr(), opt_le_tensor_out_shim),
        Kernel::new_fallback(c"aten::le.Scalar_out".as_ptr(), opt_le_scalar_out_shim),
        Kernel::new_fallback(c"aten::where.self_out".as_ptr(), opt_where_self_out_shim),
        Kernel::new_fallback(
            c"aten::grid_sampler_2d.out".as_ptr(),
            opt_grid_sampler_2d_out_shim,
        ),
        Kernel::new_fallback(c"aten::_fft_r2c.out".as_ptr(), opt_fft_r2c_out_shim),
        Kernel::new_fallback(c"aten::_fft_c2r.out".as_ptr(), opt_fft_c2r_out_shim),
    ]
}

/// Register all optimized-CPU operator shims. These override the portable
/// registrations for the same op names, so call after (or instead of)
/// `kernels::portable::register::register_all` when the optimized path is wanted.
/// Idempotent only in the sense that re-registering the same name aborts.
#[must_use]
pub fn register_all() -> Error {
    let mut kernels = optimized_kernels();
    register_kernels(Span::from_raw_parts(kernels.as_mut_ptr(), kernels.len()))
}
