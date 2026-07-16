//! Literal port of kernels/optimized/cpu/op_exp.cpp.

use crate::runtime::core::error::Error;
use crate::runtime::core::exec_aten::util::tensor_util::{resize_tensor, tensor_is_floating_type};
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::core::portable_type::{BFloat16, Half};
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

// PORT-NOTE: `std::exp` over CTYPE_OUT (the FLOATHBF16 set: Float, Double, Half,
// BFloat16). Rust has no generic `std::exp`; a per-CTYPE_OUT `ExpOut` trait
// supplies it. For the reduced float types the C++ `std::exp(CTYPE_OUT)`
// promotes the argument to float, calls `expf`, and rounds back — mirrored here.
trait ExpOut: Copy {
    fn exp_out(self) -> Self;
}
impl ExpOut for f32 {
    fn exp_out(self) -> Self {
        f32::exp(self)
    }
}
impl ExpOut for f64 {
    fn exp_out(self) -> Self {
        f64::exp(self)
    }
}
impl ExpOut for Half {
    fn exp_out(self) -> Self {
        Half::from_f32(f32::exp(self.to_f32()))
    }
}
impl ExpOut for BFloat16 {
    fn exp_out(self) -> Self {
        BFloat16::from_f32(f32::exp(self.to_f32()))
    }
}

// PORT-NOTE: `CastFromIn` models the C++ `static_cast<CTYPE_OUT>(in_data[i])`
// for every (CTYPE_IN, CTYPE_OUT) pair the REALHBBF16 x FLOATHBF16 dispatch can
// produce. Named after the target CTYPE_OUT; each impl narrows/promotes the
// source ctype the way the C++ static_cast does.
trait CastFromIn<IN>: Sized {
    fn cast_from(v: IN) -> Self;
}

macro_rules! cast_from_impls {
    ($out:ty, $to:expr) => {
        impl CastFromIn<u8> for $out {
            fn cast_from(v: u8) -> Self {
                let f = v as f64;
                $to(f)
            }
        }
        impl CastFromIn<i8> for $out {
            fn cast_from(v: i8) -> Self {
                let f = v as f64;
                $to(f)
            }
        }
        impl CastFromIn<i16> for $out {
            fn cast_from(v: i16) -> Self {
                let f = v as f64;
                $to(f)
            }
        }
        impl CastFromIn<i32> for $out {
            fn cast_from(v: i32) -> Self {
                let f = v as f64;
                $to(f)
            }
        }
        impl CastFromIn<i64> for $out {
            fn cast_from(v: i64) -> Self {
                let f = v as f64;
                $to(f)
            }
        }
        impl CastFromIn<bool> for $out {
            fn cast_from(v: bool) -> Self {
                let f = if v { 1.0f64 } else { 0.0f64 };
                $to(f)
            }
        }
        impl CastFromIn<f32> for $out {
            fn cast_from(v: f32) -> Self {
                $to(v as f64)
            }
        }
        impl CastFromIn<f64> for $out {
            fn cast_from(v: f64) -> Self {
                $to(v)
            }
        }
        impl CastFromIn<Half> for $out {
            fn cast_from(v: Half) -> Self {
                $to(v.to_f64())
            }
        }
        impl CastFromIn<BFloat16> for $out {
            fn cast_from(v: BFloat16) -> Self {
                $to(v.to_f64())
            }
        }
    };
}

cast_from_impls!(f32, |f: f64| f as f32);
cast_from_impls!(f64, |f: f64| f);
cast_from_impls!(Half, |f: f64| Half::from_f64(f));
cast_from_impls!(BFloat16, |f: f64| BFloat16::from_f64(f));

// DEVIATION: the C++ has two `exp_data` overloads — a fast path using
// `at::vec::Vectorized<CTYPE_IN>::exp()` (only when CTYPE_IN == CTYPE_OUT and
// neither is a reduced float) and a scalar slow path. Per PORTING.md the sleef
// vectorized transcendentals collapse to a scalar `std::exp` loop; the fast
// path only fired when the cast was a no-op, so both overloads compute the same
// value. They are unified into this single scalar form (cast to CTYPE_OUT, then
// `std::exp`), which Rust autovectorizes.
// [spec:et:def:op-exp.torch.executor.native.exp-data-fn]
// [spec:et:sem:op-exp.torch.executor.native.exp-data-fn]
#[allow(non_camel_case_types)] // literal C++ template param names
fn exp_data<CTYPE_IN, CTYPE_OUT>(in_data: *const CTYPE_IN, numel: usize, out_data: *mut CTYPE_OUT)
where
    CTYPE_IN: Copy,
    CTYPE_OUT: Copy + ExpOut + CastFromIn<CTYPE_IN>,
{
    for i in 0..numel {
        let xi: CTYPE_OUT = CTYPE_OUT::cast_from(unsafe { *in_data.add(i) });
        unsafe {
            *out_data.add(i) = xi.exp_out();
        }
    }
}

// PORT-NOTE: `(void)ctx;` dropped. `Tensor& out` / returned `Tensor&` become
// `&'a Tensor` (interior mutation through `*mut TensorImpl`).
// [spec:et:def:op-exp.torch.executor.native.opt-exp-out-fn]
// [spec:et:sem:op-exp.torch.executor.native.opt-exp-out-fn]
pub fn opt_exp_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    in_: &Tensor,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    // (void)ctx;

    // Resize for dynamic shape
    let error = resize_tensor(out, in_.sizes());
    crate::et_kernel_check_msg!(
        ctx,
        error == Error::Ok,
        InvalidArgument,
        out,
        "Failed to resize output tensor."
    );

    crate::et_kernel_check!(ctx, tensor_is_floating_type(out), InvalidArgument, out);

    crate::et_switch_realhbbf16_types!(in_.scalar_type(), ctx, "exp.out", CtypeIn, {
        crate::et_switch_floathbf16_types!(out.scalar_type(), ctx, "exp.out", CtypeOut, {
            exp_data::<CtypeIn, CtypeOut>(
                in_.const_data_ptr::<CtypeIn>(),
                in_.numel() as usize,
                out.mutable_data_ptr::<CtypeOut>(),
            );
        });
    });

    out
}

// Port of kernels/test/op_exp_test.cpp via the shared
// UnaryUfuncRealHBBF16ToFloatHBF16Test harness (the C++ suite is linked against
// whichever kernel is registered; here it drives opt_exp_out directly). Every
// case runs exp_data underneath, so both symbols are exercised.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::portable::cpu::pattern::unary_ufunc_realhbbf16_to_floathbf16::test_harness as h;

    fn op_reference(x: f64) -> f64 {
        x.exp()
    }

    // [spec:et:sem:op-exp.torch.executor.native.opt-exp-out-fn/test]
    // [spec:et:sem:op-exp.torch.executor.native.exp-data-fn/test]
    #[test]
    fn op_exp_out_test_handle_bool_input() {
        h::test_bool_input(opt_exp_out, op_reference);
    }

    // [spec:et:sem:op-exp.torch.executor.native.opt-exp-out-fn/test]
    // [spec:et:sem:op-exp.torch.executor.native.exp-data-fn/test]
    #[test]
    fn op_exp_out_test_all_real_input_half_output_static_dynamism_support() {
        h::test_all_real_input_half_output_static_dynamism_support(opt_exp_out, op_reference);
    }

    // [spec:et:sem:op-exp.torch.executor.native.opt-exp-out-fn/test]
    // [spec:et:sem:op-exp.torch.executor.native.exp-data-fn/test]
    #[test]
    fn op_exp_out_test_all_real_input_bfloat16_output_static_dynamism_support() {
        h::test_all_real_input_bfloat16_output_static_dynamism_support(opt_exp_out, op_reference);
    }

    // [spec:et:sem:op-exp.torch.executor.native.opt-exp-out-fn/test]
    // [spec:et:sem:op-exp.torch.executor.native.exp-data-fn/test]
    #[test]
    fn op_exp_out_test_all_real_input_float_output_static_dynamism_support() {
        h::test_all_real_input_float_output_static_dynamism_support(opt_exp_out, op_reference);
    }

    // [spec:et:sem:op-exp.torch.executor.native.opt-exp-out-fn/test]
    // [spec:et:sem:op-exp.torch.executor.native.exp-data-fn/test]
    #[test]
    fn op_exp_out_test_all_real_input_double_output_static_dynamism_support() {
        h::test_all_real_input_double_output_static_dynamism_support(opt_exp_out, op_reference);
    }

    // [spec:et:sem:op-exp.torch.executor.native.opt-exp-out-fn/test]
    // [spec:et:sem:op-exp.torch.executor.native.exp-data-fn/test]
    #[test]
    fn op_exp_out_test_all_real_input_bfloat16_output_bound_dynamism_support() {
        h::test_all_real_input_bfloat16_output_bound_dynamism_support(opt_exp_out, op_reference);
    }

    // [spec:et:sem:op-exp.torch.executor.native.opt-exp-out-fn/test]
    // [spec:et:sem:op-exp.torch.executor.native.exp-data-fn/test]
    #[test]
    fn op_exp_out_test_all_real_input_float_output_bound_dynamism_support() {
        h::test_all_real_input_float_output_bound_dynamism_support(opt_exp_out, op_reference);
    }

    // [spec:et:sem:op-exp.torch.executor.native.opt-exp-out-fn/test]
    // [spec:et:sem:op-exp.torch.executor.native.exp-data-fn/test]
    #[test]
    fn op_exp_out_test_all_real_input_double_output_bound_dynamism_support() {
        h::test_all_real_input_double_output_bound_dynamism_support(opt_exp_out, op_reference);
    }

    // [spec:et:sem:op-exp.torch.executor.native.opt-exp-out-fn/test]
    // [spec:et:sem:op-exp.torch.executor.native.exp-data-fn/test]
    #[test]
    fn op_exp_out_test_all_real_input_float_output_unbound_dynamism_support() {
        h::test_all_real_input_float_output_unbound_dynamism_support(opt_exp_out, op_reference);
    }

    // [spec:et:sem:op-exp.torch.executor.native.opt-exp-out-fn/test]
    // [spec:et:sem:op-exp.torch.executor.native.exp-data-fn/test]
    #[test]
    fn op_exp_out_test_all_real_input_double_output_unbound_dynamism_support() {
        h::test_all_real_input_double_output_unbound_dynamism_support(opt_exp_out, op_reference);
    }

    // [spec:et:sem:op-exp.torch.executor.native.opt-exp-out-fn/test]
    #[test]
    fn op_exp_out_test_all_non_float_output_d_type_dies() {
        h::test_non_float_output_dtype_dies(opt_exp_out);
    }

    // [spec:et:sem:op-exp.torch.executor.native.opt-exp-out-fn/test]
    #[test]
    fn op_exp_out_test_mismatched_input_shapes_dies() {
        h::test_mismatched_input_shapes_dies(opt_exp_out);
    }
}
