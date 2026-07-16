//! Literal port of kernels/optimized/cpu/op_sum.cpp.
//!
//! DEVIATION: the C++ fast helpers use `at::vec::Vectorized<float>` (loadu /
//! vector add / `vec_reduce_all`) with a scalar tail. Per PORTING.md's
//! optimized-kernels rule, the SIMD lane type collapses to scalar: each helper
//! is a plain `f32`-accumulating loop (Rust autovectorizes). The op control flow
//! (fast-path eligibility, contiguous single-dim specialization, portable
//! fallback) is preserved bug-for-bug.

use crate::kernels::portable::cpu::op_sum::sum_dim_out;
#[cfg(not(feature = "aten"))]
use crate::kernels::portable::cpu::util::reduce_util::check_reduction_args;
use crate::kernels::portable::cpu::util::reduce_util::resize_reduction_out;
use crate::runtime::core::array_ref::ArrayRef;
use crate::runtime::core::error::Error;
use crate::runtime::core::exec_aten::util::scalar_type_util::is_complex_type;
use crate::runtime::core::exec_aten::util::tensor_util::{
    tensor_is_contiguous, tensor_is_default_dim_order, tensors_have_same_dim_order2,
};
use crate::runtime::core::portable_type::scalar_type::ScalarType;
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::core::portable_type::{BFloat16, Half};
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

// PORT-NOTE: `optional<ArrayRef<int64_t>> dim_list` maps to
// `Option<ArrayRef<i64>>`; `optional<ScalarType> dtype` to `Option<ScalarType>`.
// The forward-declared portable fallback `sum_dim_out` is the ported
// `crate::kernels::portable::cpu::op_sum::sum_dim_out`.

/// Element type interop for the fast helpers, which accumulate in `f32`
/// regardless of CTYPE (mirroring the C++ `Vectorized<float>` accumulator) and
/// `static_cast<CTYPE>` on store. Covers the REALHBBF16 set.
trait SumF32: Copy {
    fn to_f32(self) -> f32;
    fn from_f32(v: f32) -> Self;
}

macro_rules! impl_sum_f32_num {
    ($($t:ty),*) => {$(
        impl SumF32 for $t {
            fn to_f32(self) -> f32 {
                self as f32
            }
            fn from_f32(v: f32) -> Self {
                v as $t
            }
        }
    )*};
}
impl_sum_f32_num!(u8, i8, i16, i32, i64, f32, f64);

impl SumF32 for Half {
    fn to_f32(self) -> f32 {
        Half::to_f32(self)
    }
    fn from_f32(v: f32) -> Self {
        Half::from_f32(v)
    }
}
impl SumF32 for BFloat16 {
    fn to_f32(self) -> f32 {
        BFloat16::to_f32(self)
    }
    fn from_f32(v: f32) -> Self {
        BFloat16::from_f32(v)
    }
}

// REALHBBF16 includes Bool: `static_cast<float>(bool)` -> 0.0/1.0,
// `static_cast<bool>(float)` -> nonzero-test on store.
impl SumF32 for bool {
    fn to_f32(self) -> f32 {
        if self { 1.0 } else { 0.0 }
    }
    fn from_f32(v: f32) -> Self {
        v != 0.0
    }
}

// [spec:et:def:op-sum.torch.executor.native.sum-innermost-fn]
// [spec:et:sem:op-sum.torch.executor.native.sum-innermost-fn]
// DEVIATION: `Vectorized<float>` loop + `vec_reduce_all` horizontal add + scalar
// tail collapses to a single scalar accumulation loop over `reduce_size`.
///
/// # Safety
/// `in_`/`out` must be valid contiguous buffers of `outer_size*reduce_size` /
/// `outer_size` elements respectively.
unsafe fn sum_innermost<CTYPE: SumF32>(
    in_: *const CTYPE,
    out: *mut CTYPE,
    outer_size: i64,
    reduce_size: i64,
) {
    for i in 0..outer_size {
        let row: *const CTYPE = unsafe { in_.offset((i * reduce_size) as isize) };
        let mut sum: f32 = 0.0f32;
        for j in 0..reduce_size {
            sum += unsafe { (*row.offset(j as isize)).to_f32() };
        }
        unsafe {
            *out.offset(i as isize) = CTYPE::from_f32(sum);
        }
    }
}

// [spec:et:def:op-sum.torch.executor.native.sum-strided-fn]
// [spec:et:sem:op-sum.torch.executor.native.sum-strided-fn]
// DEVIATION: the C++ vectorizes across the contiguous inner axis (kVecSize
// output positions per step) with a scalar tail; this collapses to a plain
// scalar loop nest.
///
/// # Safety
/// `in_`/`out` must be valid contiguous buffers of
/// `outer_size*reduce_size*inner_size` / `outer_size*inner_size` elements.
unsafe fn sum_strided<CTYPE: SumF32>(
    in_: *const CTYPE,
    out: *mut CTYPE,
    outer_size: i64,
    reduce_size: i64,
    inner_size: i64,
) {
    let outer_stride: i64 = reduce_size * inner_size;
    for o in 0..outer_size {
        let in_o: *const CTYPE = unsafe { in_.offset((o * outer_stride) as isize) };
        let out_o: *mut CTYPE = unsafe { out.offset((o * inner_size) as isize) };
        for j in 0..inner_size {
            let mut sum: f32 = 0.0f32;
            for k in 0..reduce_size {
                sum += unsafe { (*in_o.offset((k * inner_size + j) as isize)).to_f32() };
            }
            unsafe {
                *out_o.offset(j as isize) = CTYPE::from_f32(sum);
            }
        }
    }
}

// [spec:et:def:op-sum.torch.executor.native.opt-sum-dim-out-fn]
// [spec:et:sem:op-sum.torch.executor.native.opt-sum-dim-out-fn]
pub fn opt_sum_dim_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    in_: &Tensor,
    dim_list: Option<ArrayRef<i64>>,
    keepdim: bool,
    dtype: Option<ScalarType>,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    // PORT-NOTE: C++ calls `check_reduction_args` unconditionally, but the ported
    // reduce_util gates the portable arg-checkers behind `#[cfg(not(aten))]`
    // (absent in the ATen build); gated here to match the ported util, as in the
    // portable op_sum port.
    #[cfg(not(feature = "aten"))]
    crate::et_kernel_check!(
        ctx,
        check_reduction_args(in_, &dim_list, keepdim, dtype, out),
        InvalidArgument,
        out
    );

    crate::et_kernel_check!(
        ctx,
        resize_reduction_out(in_, &dim_list, keepdim, out) == Error::Ok,
        InvalidArgument,
        out
    );

    crate::et_kernel_check!(
        ctx,
        tensors_have_same_dim_order2(in_, out),
        InvalidArgument,
        out
    );

    crate::et_kernel_check!(ctx, tensor_is_default_dim_order(in_), InvalidArgument, out);

    if in_.numel() == 0 {
        if out.numel() > 0 {
            unsafe {
                core::ptr::write_bytes(out.mutable_data_ptr_typed() as *mut u8, 0, out.nbytes());
            }
        }
        return out;
    }

    // Fast path: single reduction dim, matching dtype, non-complex, contiguous.
    // Anything else falls through to the portable kernel.
    let fast_eligible: bool = dim_list.is_some()
        && dim_list.as_ref().unwrap().size() == 1
        && in_.scalar_type() == out.scalar_type()
        && !is_complex_type(in_.scalar_type())
        && tensor_is_contiguous(in_);

    if fast_eligible {
        let d0: i64 = *dim_list.as_ref().unwrap().at(0);
        let d: i64 = if d0 < 0 { d0 + in_.dim() as i64 } else { d0 };
        let mut outer_size: i64 = 1;
        let reduce_size: i64 = in_.size(d as _) as i64;
        let mut inner_size: i64 = 1;
        for i in 0..d {
            outer_size *= in_.size(i as _) as i64;
        }
        for i in (d + 1)..(in_.dim() as i64) {
            inner_size *= in_.size(i as _) as i64;
        }

        let op_name = "sum.IntList_out";
        let mut handled = false;
        crate::et_switch_realhbbf16_types!(in_.scalar_type(), ctx, op_name, CTYPE, {
            let ip: *const CTYPE = in_.const_data_ptr::<CTYPE>();
            let op: *mut CTYPE = out.mutable_data_ptr::<CTYPE>();
            if inner_size == 1 {
                unsafe {
                    sum_innermost::<CTYPE>(ip, op, outer_size, reduce_size);
                }
                handled = true;
            } else {
                unsafe {
                    sum_strided::<CTYPE>(ip, op, outer_size, reduce_size, inner_size);
                }
                handled = true;
            }
        });
        if handled {
            return out;
        }
    }

    // Fallback.
    sum_dim_out(ctx, in_, dim_list, keepdim, dtype, out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::core::exec_aten::testing_util::tensor_factory::TensorFactory;
    use crate::runtime::core::memory_allocator::{MemoryAllocator, MemoryAllocatorBase};
    use crate::{assert_tensor_close, assert_tensor_eq};

    fn context() -> KernelRuntimeContext<'static> {
        crate::runtime::platform::platform::pal_init();
        KernelRuntimeContext::new(
            crate::extension::module::module::null_event_tracer(),
            core::ptr::null_mut::<MemoryAllocator>() as *mut dyn MemoryAllocatorBase,
        )
    }

    fn dim_ref(v: &[i64]) -> ArrayRef<i64> {
        ArrayRef::from_raw_parts(v.as_ptr(), v.len())
    }

    // Direct unit test of the contiguous innermost reduction helper.
    // [spec:et:sem:op-sum.torch.executor.native.sum-innermost-fn/test]
    #[test]
    fn sum_innermost_direct() {
        let input: [f32; 6] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut out = [0.0f32; 2];
        unsafe {
            sum_innermost::<f32>(input.as_ptr(), out.as_mut_ptr(), 2, 3);
        }
        assert_eq!(out, [6.0, 15.0]);

        // Half input still accumulates in f32.
        let input_h: Vec<Half> = input.iter().map(|&v| Half::from_f32(v * 0.5)).collect();
        let mut out_h = [Half::from_f32(0.0); 2];
        unsafe {
            sum_innermost::<Half>(input_h.as_ptr(), out_h.as_mut_ptr(), 2, 3);
        }
        assert_eq!(out_h[0].to_f32(), 3.0);
        assert_eq!(out_h[1].to_f32(), 7.5);
    }

    // Direct unit test of the strided reduction helper: [2,2,3] reduced over
    // the middle dim.
    // [spec:et:sem:op-sum.torch.executor.native.sum-strided-fn/test]
    #[test]
    fn sum_strided_direct() {
        let input: [f32; 12] = [
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ];
        let mut out = [0.0f32; 6];
        unsafe {
            sum_strided::<f32>(input.as_ptr(), out.as_mut_ptr(), 2, 2, 3);
        }
        assert_eq!(out, [5.0, 7.0, 9.0, 17.0, 19.0, 21.0]);
    }

    // Innermost fast path via the op: [2,3,4] summed over dim 2, keepdim.
    // [spec:et:sem:op-sum.torch.executor.native.opt-sum-dim-out-fn/test]
    // [spec:et:sem:op-sum.torch.executor.native.sum-innermost-fn/test]
    #[test]
    fn opt_sum_dim_out_innermost_fast_path() {
        let tf = TensorFactory::<f32>::new();
        let data: Vec<f32> = (0..24).map(|v| v as f32).collect();
        let in_ = tf.make_default(vec![2, 3, 4], data);
        let out = tf.zeros_default(vec![2, 3, 1]);
        let dims = [2i64];

        let mut ctx = context();
        opt_sum_dim_out(&mut ctx, &in_, Some(dim_ref(&dims)), true, None, &out);
        assert_eq!(ctx.failure_state(), Error::Ok);
        let expected = tf.make_default(vec![2, 3, 1], vec![6.0, 22.0, 38.0, 54.0, 70.0, 86.0]);
        assert_tensor_close!(out, expected);
    }

    // Strided fast path via the op: same input summed over dim 1, keepdim.
    // [spec:et:sem:op-sum.torch.executor.native.opt-sum-dim-out-fn/test]
    // [spec:et:sem:op-sum.torch.executor.native.sum-strided-fn/test]
    #[test]
    fn opt_sum_dim_out_strided_fast_path() {
        let tf = TensorFactory::<f32>::new();
        let data: Vec<f32> = (0..24).map(|v| v as f32).collect();
        let in_ = tf.make_default(vec![2, 3, 4], data);
        let out = tf.zeros_default(vec![2, 1, 4]);
        let dims = [1i64];

        let mut ctx = context();
        opt_sum_dim_out(&mut ctx, &in_, Some(dim_ref(&dims)), true, None, &out);
        let expected = tf.make_default(
            vec![2, 1, 4],
            vec![12.0, 15.0, 18.0, 21.0, 48.0, 51.0, 54.0, 57.0],
        );
        assert_tensor_close!(out, expected);
    }

    // Negative dim is normalized (dim -1 == dim 2), keepdim=false.
    // [spec:et:sem:op-sum.torch.executor.native.opt-sum-dim-out-fn/test]
    #[test]
    fn opt_sum_dim_out_negative_dim() {
        let tf = TensorFactory::<f32>::new();
        let data: Vec<f32> = (0..24).map(|v| v as f32).collect();
        let in_ = tf.make_default(vec![2, 3, 4], data);
        let out = tf.zeros_default(vec![2, 3]);
        let dims = [-1i64];

        let mut ctx = context();
        opt_sum_dim_out(&mut ctx, &in_, Some(dim_ref(&dims)), false, None, &out);
        let expected = tf.make_default(vec![2, 3], vec![6.0, 22.0, 38.0, 54.0, 70.0, 86.0]);
        assert_tensor_close!(out, expected);
    }

    // Integer dtype takes the fast path too (REALHBBF16 switch).
    // [spec:et:sem:op-sum.torch.executor.native.opt-sum-dim-out-fn/test]
    #[test]
    fn opt_sum_dim_out_int_fast_path() {
        let tf = TensorFactory::<i32>::new();
        let in_ = tf.make_default(vec![2, 3], vec![1, 2, 3, 4, 5, 6]);
        let out = tf.zeros_default(vec![2, 1]);
        let dims = [1i64];

        let mut ctx = context();
        opt_sum_dim_out(&mut ctx, &in_, Some(dim_ref(&dims)), true, None, &out);
        assert_tensor_eq!(out, tf.make_default(vec![2, 1], vec![6, 15]));
    }

    // Mismatched in/out dtype is not fast-eligible: falls back to the portable
    // sum_dim_out with dtype conversion.
    // [spec:et:sem:op-sum.torch.executor.native.opt-sum-dim-out-fn/test]
    #[test]
    fn opt_sum_dim_out_dtype_conversion_fallback() {
        let tf = TensorFactory::<f32>::new();
        let tf_out = TensorFactory::<f64>::new();
        let data: Vec<f32> = (0..24).map(|v| v as f32).collect();
        let in_ = tf.make_default(vec![2, 3, 4], data);
        let out = tf_out.zeros_default(vec![2, 3, 1]);
        let dims = [2i64];

        let mut ctx = context();
        opt_sum_dim_out(
            &mut ctx,
            &in_,
            Some(dim_ref(&dims)),
            true,
            Some(ScalarType::Double),
            &out,
        );
        assert_eq!(ctx.failure_state(), Error::Ok);
        let expected = tf_out.make_default(vec![2, 3, 1], vec![6.0, 22.0, 38.0, 54.0, 70.0, 86.0]);
        assert_tensor_close!(out, expected);
    }

    // Multi-dim dim list is not fast-eligible: portable fallback.
    // [spec:et:sem:op-sum.torch.executor.native.opt-sum-dim-out-fn/test]
    #[test]
    fn opt_sum_dim_out_multi_dim_fallback() {
        let tf = TensorFactory::<f32>::new();
        let data: Vec<f32> = (0..24).map(|v| v as f32).collect();
        let in_ = tf.make_default(vec![2, 3, 4], data);
        let out = tf.zeros_default(vec![1, 3, 1]);
        let dims = [0i64, 2];

        let mut ctx = context();
        opt_sum_dim_out(&mut ctx, &in_, Some(dim_ref(&dims)), true, None, &out);
        // Per middle index j: sum of both batches' row j.
        let expected = tf.make_default(vec![1, 3, 1], vec![60.0, 92.0, 124.0]);
        assert_tensor_close!(out, expected);
    }

    // Empty input zero-fills the (non-empty) output and returns.
    // [spec:et:sem:op-sum.torch.executor.native.opt-sum-dim-out-fn/test]
    #[test]
    fn opt_sum_dim_out_empty_input() {
        let tf = TensorFactory::<f32>::new();
        let in_ = tf.make_default(vec![2, 0, 4], vec![]);
        let out = tf.ones_default(vec![2, 1, 4]);
        let dims = [1i64];

        let mut ctx = context();
        opt_sum_dim_out(&mut ctx, &in_, Some(dim_ref(&dims)), true, None, &out);
        assert_eq!(ctx.failure_state(), Error::Ok);
        assert_tensor_eq!(out, tf.zeros_default(vec![2, 1, 4]));
    }

    // Half sums accumulate in f32 before narrowing on store.
    // [spec:et:sem:op-sum.torch.executor.native.opt-sum-dim-out-fn/test]
    // [spec:et:sem:op-sum.torch.executor.native.sum-innermost-fn/test]
    #[test]
    fn opt_sum_dim_out_half_accumulates_in_float() {
        let tf = TensorFactory::<Half>::new();
        let data: Vec<Half> = (0..8).map(|v| Half::from_f32(v as f32 + 0.5)).collect();
        let in_ = tf.make_default(vec![2, 4], data);
        let out = tf.zeros_default(vec![2, 1]);
        let dims = [1i64];

        let mut ctx = context();
        opt_sum_dim_out(&mut ctx, &in_, Some(dim_ref(&dims)), true, None, &out);
        // Rows: 0.5+1.5+2.5+3.5 = 8, 4.5+5.5+6.5+7.5 = 24.
        let expected = tf.make_default(vec![2, 1], vec![Half::from_f32(8.0), Half::from_f32(24.0)]);
        assert_tensor_eq!(out, expected);
    }
}
