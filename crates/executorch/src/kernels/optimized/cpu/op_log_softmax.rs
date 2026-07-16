//! Literal port of kernels/optimized/cpu/op_log_softmax.cpp.

use crate::kernels::portable::cpu::util::activation_ops_util::check_log_softmax_args;
use crate::runtime::core::error::Error;
use crate::runtime::core::exec_aten::util::tensor_util::{nonzero_dim, resize_tensor};
use crate::runtime::core::portable_type::scalar_type::ScalarType;
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::core::portable_type::{BFloat16, Half};
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;
use crate::runtime::kernel::thread_parallel_interface::{internal::GRAIN_SIZE, parallel_for};

// PORT-NOTE: the C++ dispatches over (IN_T, OUT_T) and leans on ATen's
// `serial_vec_log_softmax_lastdim_range` / `serial_vec_logsoftmax_range`, which
// accumulate the exp-sum in a float `acc_type` for the reduced floating types.
// Rust has no `std::conditional_t`; the per-CTYPE accumulation type and the
// max/exp/log/cast operations are modeled by the `LogSoftmaxCtype` /
// `LogSoftmaxAcc` traits, mirroring kernels/portable/cpu/op_log_softmax.rs.
trait LogSoftmaxAcc: Copy {
    fn zero() -> Self;
    fn exp(self) -> Self;
    fn ln(self) -> Self;
    fn add(self, other: Self) -> Self;
    fn sub(self, other: Self) -> Self;
}

impl LogSoftmaxAcc for f32 {
    fn zero() -> Self {
        0.0
    }
    fn exp(self) -> Self {
        f32::exp(self)
    }
    fn ln(self) -> Self {
        f32::ln(self)
    }
    fn add(self, other: Self) -> Self {
        self + other
    }
    fn sub(self, other: Self) -> Self {
        self - other
    }
}

trait LogSoftmaxCtype: Copy {
    type Acc: LogSoftmaxAcc;
    fn neg_infinity() -> Self;
    fn max(a: Self, b: Self) -> Self;
    fn to_acc(self) -> Self::Acc;
    fn from_acc(val: Self::Acc) -> Self;
}

impl LogSoftmaxCtype for f32 {
    type Acc = f32;
    fn neg_infinity() -> Self {
        f32::NEG_INFINITY
    }
    fn max(a: Self, b: Self) -> Self {
        f32::max(a, b)
    }
    fn to_acc(self) -> Self::Acc {
        self
    }
    fn from_acc(val: Self::Acc) -> Self {
        val
    }
}

impl LogSoftmaxCtype for Half {
    type Acc = f32;
    fn neg_infinity() -> Self {
        Half::NEG_INFINITY
    }
    fn max(a: Self, b: Self) -> Self {
        if a > b { a } else { b }
    }
    fn to_acc(self) -> Self::Acc {
        self.to_f32()
    }
    fn from_acc(val: Self::Acc) -> Self {
        Half::from_f32(val)
    }
}

impl LogSoftmaxCtype for BFloat16 {
    type Acc = f32;
    fn neg_infinity() -> Self {
        BFloat16::NEG_INFINITY
    }
    fn max(a: Self, b: Self) -> Self {
        if a > b { a } else { b }
    }
    fn to_acc(self) -> Self::Acc {
        self.to_f32()
    }
    fn from_acc(val: Self::Acc) -> Self {
        BFloat16::from_f32(val)
    }
}

// DEVIATION: ATen's `serial_vec_log_softmax_lastdim_range` /
// `serial_vec_logsoftmax_range` and their chunk-size helpers implement the
// blocked/vectorized reductions. Per PORTING.md the `Vectorized<T>` lanes
// collapse to scalar loops; the chunk-size bookkeeping (BLOCK_SIZE = 64*1024,
// halved for mobile caches) is a cache-tiling optimization with no observable
// effect, so the general case reduces over the `dim_size` elements at
// `inner_size` stride directly. Both cases keep the numerically stable
// max-subtract then subtract-logsumexp formulation identical to
// kernels/portable/cpu/op_log_softmax.rs.

// PORT-NOTE: the per-row reduction reproduces
// `at::native::_vec_log_softmax_lastdim`: max over the row, sum of exp(x - max)
// accumulated in ACC, log of that sum, then out = (x - max) - log_sum.
#[allow(non_camel_case_types)] // literal C++ template param names
fn log_softmax_reduce_row<IN_T, OUT_T>(
    in_row: *const IN_T,
    out_row: *mut OUT_T,
    dim_size: i64,
    stride: i64,
) where
    IN_T: LogSoftmaxCtype,
    OUT_T: LogSoftmaxCtype<Acc = <IN_T as LogSoftmaxCtype>::Acc>,
{
    // max in the log_softmax dim.
    let mut max_in: IN_T = IN_T::neg_infinity();
    {
        let mut d: i64 = 0;
        while d < dim_size {
            let v = unsafe { *in_row.add((d * stride) as usize) };
            max_in = IN_T::max(max_in, v);
            d += 1;
        }
    }

    let max_acc = max_in.to_acc();
    let mut exp_sum: <IN_T as LogSoftmaxCtype>::Acc =
        <<IN_T as LogSoftmaxCtype>::Acc as LogSoftmaxAcc>::zero();
    {
        let mut d: i64 = 0;
        while d < dim_size {
            let v = unsafe { *in_row.add((d * stride) as usize) };
            let e = LogSoftmaxAcc::exp(v.to_acc().sub(max_acc));
            exp_sum = exp_sum.add(e);
            d += 1;
        }
    }
    let log_sum = LogSoftmaxAcc::ln(exp_sum);

    {
        let mut d: i64 = 0;
        while d < dim_size {
            let v = unsafe { *in_row.add((d * stride) as usize) };
            let r: <IN_T as LogSoftmaxCtype>::Acc = v.to_acc().sub(max_acc).sub(log_sum);
            unsafe {
                *out_row.add((d * stride) as usize) = OUT_T::from_acc(r);
            }
            d += 1;
        }
    }
}

// [spec:et:def:op-log-softmax.torch.executor.native.log-softmax-kernel-fn]
// [spec:et:sem:op-log-softmax.torch.executor.native.log-softmax-kernel-fn]
#[allow(non_camel_case_types)] // literal C++ template param names
fn log_softmax_kernel<IN_T, OUT_T>(input: &Tensor, dim: i64, out: &Tensor)
where
    IN_T: LogSoftmaxCtype,
    OUT_T: LogSoftmaxCtype<Acc = <IN_T as LogSoftmaxCtype>::Acc>,
{
    let input_data_base: *const IN_T = input.const_data_ptr::<IN_T>();
    let output_data_base: *mut OUT_T = out.mutable_data_ptr::<OUT_T>();

    if input.dim() == 0 {
        unsafe {
            *output_data_base =
                OUT_T::from_acc(<<OUT_T as LogSoftmaxCtype>::Acc as LogSoftmaxAcc>::zero()); // output_data_base[0] = 0
        }
        return;
    }

    let dim_size: i64 = input.size(dim as isize) as i64;

    let mut outer_size: i64 = 1;
    let mut inner_size: i64 = 1;
    {
        let mut i: i64 = 0;
        while i < dim {
            outer_size *= input.size(i as isize) as i64;
            i += 1;
        }
    }
    {
        let mut i: i64 = dim + 1;
        while i < input.dim() as i64 {
            inner_size *= input.size(i as isize) as i64;
            i += 1;
        }
    }

    if dim == input.dim() as i64 - 1 {
        // Last-dim case: each outer row is `dim_size` contiguous elements.
        parallel_for(0, outer_size, GRAIN_SIZE, &|begin: i64, end: i64| {
            let mut outer: i64 = begin;
            while outer < end {
                let base = outer * dim_size;
                log_softmax_reduce_row::<IN_T, OUT_T>(
                    unsafe { input_data_base.add(base as usize) },
                    unsafe { output_data_base.add(base as usize) },
                    dim_size,
                    1,
                );
                outer += 1;
            }
        });
    } else {
        // General case: reduce over `dim_size` elements at `inner_size` stride
        // for every (outer, inner) position.
        parallel_for(0, outer_size, GRAIN_SIZE, &|begin: i64, end: i64| {
            let mut outer: i64 = begin;
            while outer < end {
                let outer_base = outer * dim_size * inner_size;
                let mut inner: i64 = 0;
                while inner < inner_size {
                    let base = outer_base + inner;
                    log_softmax_reduce_row::<IN_T, OUT_T>(
                        unsafe { input_data_base.add(base as usize) },
                        unsafe { output_data_base.add(base as usize) },
                        dim_size,
                        inner_size,
                    );
                    inner += 1;
                }
                outer += 1;
            }
        });
    }
}

// PORT-NOTE: the C++ `log_softmax_wrapper<OUT_T>` uses `if constexpr` to select
// the (IN_T, OUT_T) instantiation. Rust models the per-OUT_T dispatch with the
// `LogSoftmaxWrapper` trait: reduced types run `<OUT_T, OUT_T>`; Float runs a
// runtime switch on the input dtype (only Float supported; Double not yet).
trait LogSoftmaxWrapper: LogSoftmaxCtype {
    fn wrapper(x: &Tensor, dim: i64, out: &Tensor) -> bool;
}

// OUT_T is the corresponding C++ type for out.scalar_type().
// [spec:et:def:op-log-softmax.torch.executor.native.log-softmax-wrapper-fn]
// [spec:et:sem:op-log-softmax.torch.executor.native.log-softmax-wrapper-fn]
impl LogSoftmaxWrapper for f32 {
    fn wrapper(x: &Tensor, dim: i64, out: &Tensor) -> bool {
        let input_scalar_type = x.scalar_type();
        match input_scalar_type {
            // TODO: support Double as well
            ScalarType::Float => {
                log_softmax_kernel::<f32, f32>(x, dim, out);
                true
            }
            _ => false, // Unsupported input dtype
        }
    }
}

impl LogSoftmaxWrapper for BFloat16 {
    fn wrapper(x: &Tensor, dim: i64, out: &Tensor) -> bool {
        // Input dtype equals output dtype (enforced by check_log_softmax_args).
        log_softmax_kernel::<BFloat16, BFloat16>(x, dim, out);
        true
    }
}

impl LogSoftmaxWrapper for Half {
    fn wrapper(x: &Tensor, dim: i64, out: &Tensor) -> bool {
        log_softmax_kernel::<Half, Half>(x, dim, out);
        true
    }
}

// _log_softmax.out(Tensor self, int dim, bool half_to_float, *, Tensor(a!) out)
// -> Tensor(a!)
// PORT-NOTE: `(void)context;` dropped. `Tensor& out` / returned `Tensor&`
// become `&'a Tensor`.
// [spec:et:def:op-log-softmax.torch.executor.native.opt-log-softmax-out-fn]
// [spec:et:sem:op-log-softmax.torch.executor.native.opt-log-softmax-out-fn]
pub fn opt_log_softmax_out<'a, 'b>(
    context: &mut KernelRuntimeContext,
    self_: &Tensor,
    dim: i64,
    half_to_float: bool,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    // (void)context;

    crate::et_kernel_check!(
        context,
        check_log_softmax_args(self_, dim, half_to_float, out),
        InvalidArgument,
        out
    );

    crate::et_kernel_check!(
        context,
        resize_tensor(out, self_.sizes()) == Error::Ok,
        InvalidArgument,
        out
    );

    let dim: i64 = if dim < 0 {
        dim + nonzero_dim(self_) as i64
    } else {
        dim
    };

    let out_scalar_type = out.scalar_type();
    match out_scalar_type {
        // TODO: support Double as well
        ScalarType::Float => {
            let success = <f32 as LogSoftmaxWrapper>::wrapper(self_, dim, out);
            crate::et_kernel_check!(context, success, InvalidArgument, out);
        }
        ScalarType::BFloat16 => {
            let success = <BFloat16 as LogSoftmaxWrapper>::wrapper(self_, dim, out);
            crate::et_kernel_check!(context, success, InvalidArgument, out);
        }
        ScalarType::Half => {
            let success = <Half as LogSoftmaxWrapper>::wrapper(self_, dim, out);
            crate::et_kernel_check!(context, success, InvalidArgument, out);
        }
        _ => {
            crate::et_kernel_check!(context, false, InvalidArgument, out);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::core::exec_aten::testing_util::tensor_factory::TensorFactory;
    use crate::runtime::core::exec_aten::testing_util::tensor_util::internal::K_DEFAULT_ATOL;
    use crate::runtime::core::memory_allocator::{MemoryAllocator, MemoryAllocatorBase};
    use crate::{assert_tensor_close, assert_tensor_close_with_tol};

    fn context() -> KernelRuntimeContext<'static> {
        crate::runtime::platform::platform::pal_init();
        KernelRuntimeContext::new(
            crate::extension::module::module::null_event_tracer(),
            core::ptr::null_mut::<MemoryAllocator>() as *mut dyn MemoryAllocatorBase,
        )
    }

    macro_rules! et_expect_kernel_failure {
        ($ctx:expr, $stmt:expr) => {{
            let _ = $stmt;
            assert_ne!(
                $ctx.failure_state(),
                Error::Ok,
                "Expected kernel failure but found success."
            );
        }};
    }

    // OpLogSoftmaxOutTest.Smoke.
    // [spec:et:sem:op-log-softmax.torch.executor.native.opt-log-softmax-out-fn/test]
    // [spec:et:sem:op-log-softmax.torch.executor.native.log-softmax-wrapper-fn/test]
    // [spec:et:sem:op-log-softmax.torch.executor.native.log-softmax-kernel-fn/test]
    #[test]
    fn opt_log_softmax_out_smoke() {
        let tf = TensorFactory::<f32>::new();
        let x = tf.make_default(vec![1, 3], vec![0.0, 1.0, 2.0]);
        let out = tf.zeros_default(vec![1, 3]);

        let mut ctx = context();
        opt_log_softmax_out(&mut ctx, &x, 1, false, &out);
        assert_eq!(ctx.failure_state(), Error::Ok);
        let expected = tf.make_default(vec![1, 3], vec![-2.40761, -1.40761, -0.407606]);
        assert_tensor_close!(out, expected);
    }

    // OpLogSoftmaxOutTest test_dtype: (2, 3) rows 0..5, dim=1 (last-dim path).
    // [spec:et:sem:op-log-softmax.torch.executor.native.opt-log-softmax-out-fn/test]
    // [spec:et:sem:op-log-softmax.torch.executor.native.log-softmax-wrapper-fn/test]
    // [spec:et:sem:op-log-softmax.torch.executor.native.log-softmax-kernel-fn/test]
    #[test]
    fn opt_log_softmax_out_supported_float_dtypes() {
        let expected_f64 = [-2.40760596, -1.40760596, -0.40760596f64];

        // Float
        {
            let tf = TensorFactory::<f32>::new();
            let x = tf.make_default(vec![2, 3], vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
            let out = tf.zeros_default(vec![2, 3]);
            let mut ctx = context();
            opt_log_softmax_out(&mut ctx, &x, 1, false, &out);
            let expected_data: Vec<f32> = expected_f64
                .iter()
                .cycle()
                .take(6)
                .map(|&v| v as f32)
                .collect();
            assert_tensor_close!(out, tf.make_default(vec![2, 3], expected_data));
        }
        // Half
        {
            let tf = TensorFactory::<Half>::new();
            let d = |v: f64| Half::from_f32(v as f32);
            let x = tf.make_default(
                vec![2, 3],
                vec![d(0.0), d(1.0), d(2.0), d(3.0), d(4.0), d(5.0)],
            );
            let out = tf.zeros_default(vec![2, 3]);
            let mut ctx = context();
            opt_log_softmax_out(&mut ctx, &x, 1, false, &out);
            let expected_data: Vec<Half> =
                expected_f64.iter().cycle().take(6).map(|&v| d(v)).collect();
            assert_tensor_close!(out, tf.make_default(vec![2, 3], expected_data));
        }
        // BFloat16 (C++ uses rtol 1e-2 for this dtype)
        {
            let tf = TensorFactory::<BFloat16>::new();
            let d = |v: f64| BFloat16::from_f32(v as f32);
            let x = tf.make_default(
                vec![2, 3],
                vec![d(0.0), d(1.0), d(2.0), d(3.0), d(4.0), d(5.0)],
            );
            let out = tf.zeros_default(vec![2, 3]);
            let mut ctx = context();
            opt_log_softmax_out(&mut ctx, &x, 1, false, &out);
            let expected_data: Vec<BFloat16> =
                expected_f64.iter().cycle().take(6).map(|&v| d(v)).collect();
            assert_tensor_close_with_tol!(
                out,
                tf.make_default(vec![2, 3], expected_data),
                1e-2,
                K_DEFAULT_ATOL
            );
        }
    }

    // OpLogSoftmaxOutTest.NonContiguous: dim=0 exercises the strided (non
    // last-dim) reduction path with inner_size > 1.
    // [spec:et:sem:op-log-softmax.torch.executor.native.opt-log-softmax-out-fn/test]
    // [spec:et:sem:op-log-softmax.torch.executor.native.log-softmax-kernel-fn/test]
    #[test]
    fn opt_log_softmax_out_non_contiguous_dim() {
        let tf = TensorFactory::<f32>::new();
        #[rustfmt::skip]
        let x = tf.make_default(
            vec![9, 3],
            vec![
                0.0, 9.0, 18.0,
                1.0, 10.0, 19.0,
                2.0, 11.0, 20.0,
                3.0, 12.0, 21.0,
                4.0, 13.0, 22.0,
                5.0, 14.0, 23.0,
                6.0, 15.0, 24.0,
                7.0, 16.0, 25.0,
                8.0, 17.0, 26.0,
            ],
        );
        let out = tf.zeros_default(vec![9, 3]);

        let mut ctx = context();
        opt_log_softmax_out(&mut ctx, &x, 0, false, &out);
        let col = [
            -8.45855f32,
            -7.45855,
            -6.45855,
            -5.45855,
            -4.45855,
            -3.45855,
            -2.45855,
            -1.45855,
            -0.458552,
        ];
        let expected_data: Vec<f32> = col.iter().flat_map(|&v| [v, v, v]).collect();
        assert_tensor_close!(out, tf.make_default(vec![9, 3], expected_data));
    }

    // OpLogSoftmaxOutTest.NegativeDim: dim=-1 equals dim=1.
    // [spec:et:sem:op-log-softmax.torch.executor.native.opt-log-softmax-out-fn/test]
    #[test]
    fn opt_log_softmax_out_negative_dim() {
        let tf = TensorFactory::<f32>::new();
        let x = tf.make_default(vec![2, 3], vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        let out = tf.zeros_default(vec![2, 3]);
        let out_negative_dim = tf.zeros_default(vec![2, 3]);

        let mut ctx = context();
        opt_log_softmax_out(&mut ctx, &x, 1, false, &out);
        opt_log_softmax_out(&mut ctx, &x, -1, false, &out_negative_dim);
        assert_eq!(ctx.failure_state(), Error::Ok);
        assert_tensor_close!(out, out_negative_dim);
    }

    // The optimized kernel supports Float/Half/BFloat16 only: Double out hits
    // the default switch arm and records InvalidArgument.
    // [spec:et:sem:op-log-softmax.torch.executor.native.opt-log-softmax-out-fn/test]
    #[test]
    fn opt_log_softmax_out_double_unsupported_dies() {
        let tf = TensorFactory::<f64>::new();
        let x = tf.make_default(vec![1, 3], vec![0.0, 1.0, 2.0]);
        let out = tf.zeros_default(vec![1, 3]);

        let mut ctx = context();
        et_expect_kernel_failure!(ctx, opt_log_softmax_out(&mut ctx, &x, 1, false, &out));
    }

    // OpLogSoftmaxOutTest.MismatchedDimensionsDies: dim out of bounds.
    // [spec:et:sem:op-log-softmax.torch.executor.native.opt-log-softmax-out-fn/test]
    #[test]
    fn opt_log_softmax_out_dim_out_of_bounds_dies() {
        let tf = TensorFactory::<f32>::new();
        let x = tf.make_default(vec![1, 3], vec![0.0, 1.0, 2.0]);
        let out = tf.zeros_default(vec![1, 3]);

        let mut ctx = context();
        et_expect_kernel_failure!(ctx, opt_log_softmax_out(&mut ctx, &x, 3, false, &out));
    }

    // The Float wrapper's runtime input-dtype switch: a non-Float input returns
    // false (unsupported combination).
    // [spec:et:sem:op-log-softmax.torch.executor.native.log-softmax-wrapper-fn/test]
    #[test]
    fn log_softmax_wrapper_float_rejects_non_float_input() {
        let tf = TensorFactory::<f64>::new();
        let x = tf.make_default(vec![1, 3], vec![0.0, 1.0, 2.0]);
        let tf_out = TensorFactory::<f32>::new();
        let out = tf_out.zeros_default(vec![1, 3]);

        assert!(!<f32 as LogSoftmaxWrapper>::wrapper(&x, 1, &out));
        // And the supported combination returns true.
        let xf = tf_out.make_default(vec![1, 3], vec![0.0, 1.0, 2.0]);
        assert!(<f32 as LogSoftmaxWrapper>::wrapper(&xf, 1, &out));
    }
}
