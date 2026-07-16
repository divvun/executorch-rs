//! Literal port of kernels/optimized/cpu/op_native_layer_norm.cpp.
//!
//! DEVIATION: the C++ numeric path uses `at::vec::Vectorized<T>` /
//! `at::vec::map3` and the vectorized `RowwiseMoments`. Per PORTING.md's
//! optimized-kernels rule, the SIMD lane type collapses to the scalar element
//! type: `RowwiseMoments` is the scalar-lane port in `moments_utils.rs`, and the
//! `map3` fused affine map becomes a plain scalar loop. The op's two-path
//! structure (portable scalar fallback for small N, Welford path for large N) is
//! preserved bug-for-bug.

use crate::kernels::optimized::cpu::moments_utils::{AccFloat, MomentScalar, rowwise_moments};
use crate::kernels::portable::cpu::util::normalization_ops_util::{
    LayerNormCtype, check_layer_norm_args, get_layer_norm_out_target_size, layer_norm_scalar,
};
use crate::runtime::core::array_ref::{ArrayRef, IntArrayRef};
use crate::runtime::core::error::Error;
use crate::runtime::core::exec_aten::util::tensor_util::{
    K_TENSOR_DIMENSION_LIMIT, getLeadingDims, getTrailingDims, resize_tensor,
};
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::core::portable_type::tensor_impl::SizesType;
use crate::runtime::core::portable_type::{BFloat16, Half};
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

// PORT-NOTE: `Tensor& out/mean/rstd` and the returned `std::tuple<Tensor&,
// Tensor&, Tensor&>` become `&'a Tensor` handles and a Rust 3-tuple.
// `const optional<Tensor>&` maps to `Option<&Tensor>`. `IntArrayRef` maps to the
// `IntArrayRef = ArrayRef<i64>` alias. The templated anonymous-namespace worker
// `layer_norm<CTYPE>` becomes a generic `fn layer_norm<CTYPE: OptLayerNormCtype>`.

const K_SMALL_N_THRESHOLD: usize = 256;

/// The element type of the optimized layer_norm worker: real float / 16-bit
/// float. Bundles the `acc_t<CTYPE>` interop the C++ performs — the affine map
/// `(x*scale + offset)*gamma + beta` evaluates in `acc_t` (float) then narrows
/// to CTYPE on store — plus the CTYPE(0)/CTYPE(NAN) fills and the
/// `1 / sqrt(var + eps)` reciprocal-std in acc_t.
pub trait OptLayerNormCtype: MomentScalar {
    fn zero() -> Self;
    fn nan() -> Self;
    fn from_acc(acc: <Self as MomentScalar>::Acc) -> Self;
    // eps arrives as CTYPE; promoted to acc_t for `var + eps`.
    fn eps_to_acc(eps: Self) -> <Self as MomentScalar>::Acc;
    // C++: `gamma_null ? CTYPE(1) : gamma_data[j]` assigned to acc_t.
    fn one_acc() -> <Self as MomentScalar>::Acc;
    fn zero_acc() -> <Self as MomentScalar>::Acc;
}

macro_rules! impl_opt_layer_norm_ctype_float {
    ($t:ty) => {
        impl OptLayerNormCtype for $t {
            fn zero() -> Self {
                0.0
            }
            fn nan() -> Self {
                <$t>::NAN
            }
            fn from_acc(acc: <Self as MomentScalar>::Acc) -> Self {
                acc
            }
            fn eps_to_acc(eps: Self) -> <Self as MomentScalar>::Acc {
                eps
            }
            fn one_acc() -> <Self as MomentScalar>::Acc {
                1.0
            }
            fn zero_acc() -> <Self as MomentScalar>::Acc {
                0.0
            }
        }
    };
}
impl_opt_layer_norm_ctype_float!(f32);
impl_opt_layer_norm_ctype_float!(f64);

impl OptLayerNormCtype for Half {
    fn zero() -> Self {
        Half::from_f32(0.0)
    }
    fn nan() -> Self {
        Half::from_f32(f32::NAN)
    }
    fn from_acc(acc: f32) -> Self {
        Half::from_f32(acc)
    }
    fn eps_to_acc(eps: Self) -> f32 {
        eps.to_f32()
    }
    fn one_acc() -> f32 {
        1.0
    }
    fn zero_acc() -> f32 {
        0.0
    }
}

impl OptLayerNormCtype for BFloat16 {
    fn zero() -> Self {
        BFloat16::from_f32(0.0)
    }
    fn nan() -> Self {
        BFloat16::from_f32(f32::NAN)
    }
    fn from_acc(acc: f32) -> Self {
        BFloat16::from_f32(acc)
    }
    fn eps_to_acc(eps: Self) -> f32 {
        eps.to_f32()
    }
    fn one_acc() -> f32 {
        1.0
    }
    fn zero_acc() -> f32 {
        0.0
    }
}

// [spec:et:def:op-native-layer-norm.torch.executor.native.layer-norm-fn]
// [spec:et:sem:op-native-layer-norm.torch.executor.native.layer-norm-fn]
fn layer_norm<CTYPE: OptLayerNormCtype + LayerNormCtype>(
    input: &Tensor,
    normalized_shape: IntArrayRef,
    weight: Option<&Tensor>,
    bias: Option<&Tensor>,
    eps: CTYPE,
    out: &Tensor,
    mean: &Tensor,
    rstd: &Tensor,
) {
    let dim: usize = (input.dim() as usize) - normalized_shape.size();
    let dim_size: usize = input.size(dim as _) as usize;

    let m: usize = getLeadingDims(input, dim as i64);
    let n: usize = getTrailingDims(input, dim as i64) * dim_size;

    if m == 0 {
        return;
    }

    let out_data: *mut CTYPE = out.mutable_data_ptr::<CTYPE>();
    let mean_data: *mut CTYPE = mean.mutable_data_ptr::<CTYPE>();
    let rstd_data: *mut CTYPE = rstd.mutable_data_ptr::<CTYPE>();

    if n == 0 {
        for i in 0..m {
            unsafe {
                *mean_data.add(i) = <CTYPE as OptLayerNormCtype>::zero();
                *rstd_data.add(i) = <CTYPE as OptLayerNormCtype>::nan();
            }
        }
        return;
    }

    let input_data: *const CTYPE = input.const_data_ptr::<CTYPE>();
    let gamma_data: *const CTYPE = match weight {
        Some(w) => w.const_data_ptr::<CTYPE>(),
        None => core::ptr::null(),
    };
    let beta_data: *const CTYPE = match bias {
        Some(b) => b.const_data_ptr::<CTYPE>(),
        None => core::ptr::null(),
    };

    let gamma_null: bool = gamma_data.is_null();
    let beta_null: bool = beta_data.is_null();

    // For small normalized dimensions, fall back to the portable scalar
    // implementation since SIMD vectorization setup/tail-handling overhead
    // exceeds the benefit for small N.
    if n < K_SMALL_N_THRESHOLD {
        // PORT-NOTE (cross-module): the ported `layer_norm_scalar` takes
        // `eps: f32`; the C++ passes `CTYPE eps`. Mirrors op_native_layer_norm
        // (portable). `LayerNormCtype::to_f32` narrows CTYPE eps -> f32.
        unsafe {
            layer_norm_scalar::<CTYPE>(
                input_data,
                gamma_data,
                beta_data,
                out_data,
                mean_data,
                rstd_data,
                m,
                n,
                <CTYPE as LayerNormCtype>::to_f32(eps),
            );
        }
        return;
    }

    for i in 0..m {
        let src_ptr: *const CTYPE = unsafe { input_data.add(i * n) };
        let dst_ptr: *mut CTYPE = unsafe { out_data.add(i * n) };

        let (mean_val, mut rstd_val): (<CTYPE as MomentScalar>::Acc, <CTYPE as MomentScalar>::Acc) =
            unsafe { rowwise_moments::<CTYPE>(src_ptr, n as i64, 0) };
        rstd_val = <CTYPE as MomentScalar>::Acc::from_i64(1)
            / (rstd_val + <CTYPE as OptLayerNormCtype>::eps_to_acc(eps)).sqrt();

        let scale: <CTYPE as MomentScalar>::Acc = rstd_val;
        let offset: <CTYPE as MomentScalar>::Acc =
            (<CTYPE as MomentScalar>::Acc::zero() - rstd_val) * mean_val;

        // DEVIATION: the C++ `at::vec::map3` fused affine map collapses to this
        // scalar loop. The gamma_null/beta_null and both-present branches are
        // arithmetically identical here, so the single loop covers both C++
        // branches (evaluated in acc_t, narrowed to CTYPE on store).
        for j in 0..n {
            let gamma_v: <CTYPE as MomentScalar>::Acc = if gamma_null {
                <CTYPE as OptLayerNormCtype>::one_acc()
            } else {
                unsafe { (*gamma_data.add(j)).to_acc() }
            };
            let beta_v: <CTYPE as MomentScalar>::Acc = if beta_null {
                <CTYPE as OptLayerNormCtype>::zero_acc()
            } else {
                unsafe { (*beta_data.add(j)).to_acc() }
            };
            let x: <CTYPE as MomentScalar>::Acc = unsafe { (*src_ptr.add(j)).to_acc() };
            unsafe {
                *dst_ptr.add(j) =
                    <CTYPE as OptLayerNormCtype>::from_acc((x * scale + offset) * gamma_v + beta_v);
            }
        }

        unsafe {
            *mean_data.add(i) = <CTYPE as OptLayerNormCtype>::from_acc(mean_val);
            *rstd_data.add(i) = <CTYPE as OptLayerNormCtype>::from_acc(rstd_val);
        }
    }
}

// [spec:et:def:op-native-layer-norm.torch.executor.native.opt-native-layer-norm-out-fn]
// [spec:et:sem:op-native-layer-norm.torch.executor.native.opt-native-layer-norm-out-fn]
#[allow(clippy::too_many_arguments)]
pub fn opt_native_layer_norm_out<'a, 'b, 'c, 'd>(
    ctx: &mut KernelRuntimeContext,
    input: &Tensor,
    normalized_shape: IntArrayRef,
    weight: Option<&Tensor>,
    bias: Option<&Tensor>,
    eps: f64,
    out: &'a Tensor<'b>,
    mean_out: &'a Tensor<'c>,
    rstd_out: &'a Tensor<'d>,
) -> (&'a Tensor<'b>, &'a Tensor<'c>, &'a Tensor<'d>) {
    let _ = &ctx;

    let ret_val = (out, mean_out, rstd_out);

    crate::et_kernel_check!(
        ctx,
        check_layer_norm_args(
            input,
            normalized_shape,
            weight,
            bias,
            out,
            mean_out,
            rstd_out
        ),
        InvalidArgument,
        ret_val
    );

    let mut mean_rstd_sizes: [SizesType; K_TENSOR_DIMENSION_LIMIT] = [0; K_TENSOR_DIMENSION_LIMIT];
    let mut mean_rstd_ndim: usize = 0;
    unsafe {
        get_layer_norm_out_target_size(
            input,
            normalized_shape,
            mean_rstd_sizes.as_mut_ptr(),
            &mut mean_rstd_ndim,
        );
    }

    crate::et_kernel_check!(
        ctx,
        resize_tensor(out, input.sizes()) == Error::Ok,
        InvalidArgument,
        ret_val
    );

    crate::et_kernel_check!(
        ctx,
        resize_tensor(
            mean_out,
            ArrayRef::from_raw_parts(mean_rstd_sizes.as_ptr(), mean_rstd_ndim)
        ) == Error::Ok,
        InvalidArgument,
        ret_val
    );

    crate::et_kernel_check!(
        ctx,
        resize_tensor(
            rstd_out,
            ArrayRef::from_raw_parts(mean_rstd_sizes.as_ptr(), mean_rstd_ndim)
        ) == Error::Ok,
        InvalidArgument,
        ret_val
    );

    crate::et_switch_floathbf16_types!(input.scalar_type(), ctx, "native_layer_norm.out", CTYPE, {
        layer_norm::<CTYPE>(
            input,
            normalized_shape,
            weight,
            bias,
            <CTYPE as LayerNormCtype>::from_f64(eps),
            out,
            mean_out,
            rstd_out,
        );
    });

    ret_val
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::core::error::Error;
    use crate::runtime::core::exec_aten::testing_util::tensor_factory::TensorFactory;
    use crate::runtime::core::memory_allocator::{MemoryAllocator, MemoryAllocatorBase};
    use crate::runtime::core::tensor_shape_dynamism::TensorShapeDynamism;
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

    fn dim_ref(v: &[i64]) -> IntArrayRef {
        ArrayRef::from_raw_parts(v.as_ptr(), v.len())
    }

    // OpNativeLayerNormTest "Simple negative/positive layer norm": N=3 is below
    // kSmallNThreshold, so this runs the portable-scalar fallback branch, and
    // also pins the mean/rstd outputs.
    // [spec:et:sem:op-native-layer-norm.torch.executor.native.opt-native-layer-norm-out-fn/test]
    #[test]
    fn opt_native_layer_norm_out_small_n() {
        let tf = TensorFactory::<f32>::new();
        let input = tf.make_default(vec![2, 3], vec![1.0, 0.0, -1.0, -1.0, 4.0, 0.0]);
        let weight = tf.make_default(vec![3], vec![1.0, 1.0, 1.0]);
        let bias = tf.make_default(vec![3], vec![0.0, 0.0, 0.0]);
        let out = tf.zeros_default(vec![2, 3]);
        let mean = tf.zeros(vec![2, 3], TensorShapeDynamism::DYNAMIC_BOUND);
        let rstd = tf.zeros(vec![2, 3], TensorShapeDynamism::DYNAMIC_BOUND);
        let shape = [3i64];

        let mut ctx = context();
        opt_native_layer_norm_out(
            &mut ctx,
            &input,
            dim_ref(&shape),
            Some(&weight),
            Some(&bias),
            1.0e-5,
            &out,
            &mean,
            &rstd,
        );
        assert_eq!(ctx.failure_state(), Error::Ok);

        let expected = tf.make_default(
            vec![2, 3],
            vec![1.22474, 0.0, -1.22474, -0.925819, 1.38873, -0.46291],
        );
        assert_tensor_close!(out, expected);
        assert_tensor_close!(mean, tf.make_default(vec![2, 1], vec![0.0, 1.0]));
        assert_tensor_close!(rstd, tf.make_default(vec![2, 1], vec![1.22474, 0.46291]));
    }

    // N=512 >= kSmallNThreshold exercises the Welford (rowwise_moments) branch
    // and the fused affine map; reference computed naively in f64.
    // [spec:et:sem:op-native-layer-norm.torch.executor.native.opt-native-layer-norm-out-fn/test]
    #[test]
    fn opt_native_layer_norm_out_large_n_welford() {
        const N: usize = 512;
        const M: usize = 2;
        let eps = 1.0e-5f64;

        let input_data: Vec<f32> = (0..M * N)
            .map(|i| ((i * 31 + 7) % 97) as f32 * 0.25 - 10.0)
            .collect();
        let weight_data: Vec<f32> = (0..N).map(|j| 1.0 + (j % 3) as f32 * 0.25).collect();
        let bias_data: Vec<f32> = (0..N).map(|j| (j % 5) as f32 * 0.1 - 0.2).collect();

        // f64 reference.
        let mut expected = vec![0.0f32; M * N];
        let mut expected_mean = vec![0.0f32; M];
        let mut expected_rstd = vec![0.0f32; M];
        for i in 0..M {
            let row = &input_data[i * N..(i + 1) * N];
            let mean: f64 = row.iter().map(|&v| v as f64).sum::<f64>() / N as f64;
            let var: f64 = row.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / N as f64;
            let rstd = 1.0 / (var + eps).sqrt();
            expected_mean[i] = mean as f32;
            expected_rstd[i] = rstd as f32;
            for j in 0..N {
                expected[i * N + j] = (((row[j] as f64 - mean) * rstd) * weight_data[j] as f64
                    + bias_data[j] as f64) as f32;
            }
        }

        let tf = TensorFactory::<f32>::new();
        let input = tf.make_default(vec![M as i32, N as i32], input_data);
        let weight = tf.make_default(vec![N as i32], weight_data);
        let bias = tf.make_default(vec![N as i32], bias_data);
        let out = tf.zeros_default(vec![M as i32, N as i32]);
        let mean = tf.zeros(vec![M as i32, N as i32], TensorShapeDynamism::DYNAMIC_BOUND);
        let rstd = tf.zeros(vec![M as i32, N as i32], TensorShapeDynamism::DYNAMIC_BOUND);
        let shape = [N as i64];

        let mut ctx = context();
        opt_native_layer_norm_out(
            &mut ctx,
            &input,
            dim_ref(&shape),
            Some(&weight),
            Some(&bias),
            eps,
            &out,
            &mean,
            &rstd,
        );
        assert_eq!(ctx.failure_state(), Error::Ok);

        assert_tensor_close_with_tol!(
            out,
            tf.make_default(vec![M as i32, N as i32], expected),
            1e-3,
            1e-3
        );
        assert_tensor_close_with_tol!(
            mean,
            tf.make_default(vec![M as i32, 1], expected_mean),
            1e-3,
            1e-3
        );
        assert_tensor_close_with_tol!(
            rstd,
            tf.make_default(vec![M as i32, 1], expected_rstd),
            1e-3,
            1e-3
        );
    }

    // Welford branch with weight=None/bias=None (gamma_null/beta_null).
    // [spec:et:sem:op-native-layer-norm.torch.executor.native.opt-native-layer-norm-out-fn/test]
    #[test]
    fn opt_native_layer_norm_out_large_n_no_affine() {
        const N: usize = 300;
        let eps = 1.0e-5f64;

        let input_data: Vec<f32> = (0..N).map(|i| ((i * 13 + 3) % 41) as f32 * 0.5).collect();

        let mut expected = vec![0.0f32; N];
        let mean: f64 = input_data.iter().map(|&v| v as f64).sum::<f64>() / N as f64;
        let var: f64 = input_data
            .iter()
            .map(|&v| (v as f64 - mean).powi(2))
            .sum::<f64>()
            / N as f64;
        let rstd_ref = 1.0 / (var + eps).sqrt();
        for j in 0..N {
            expected[j] = ((input_data[j] as f64 - mean) * rstd_ref) as f32;
        }

        let tf = TensorFactory::<f32>::new();
        let input = tf.make_default(vec![1, N as i32], input_data);
        let out = tf.zeros_default(vec![1, N as i32]);
        let mean_t = tf.zeros(vec![1, N as i32], TensorShapeDynamism::DYNAMIC_BOUND);
        let rstd_t = tf.zeros(vec![1, N as i32], TensorShapeDynamism::DYNAMIC_BOUND);
        let shape = [N as i64];

        let mut ctx = context();
        opt_native_layer_norm_out(
            &mut ctx,
            &input,
            dim_ref(&shape),
            None,
            None,
            eps,
            &out,
            &mean_t,
            &rstd_t,
        );
        assert_eq!(ctx.failure_state(), Error::Ok);
        assert_tensor_close_with_tol!(
            out,
            tf.make_default(vec![1, N as i32], expected),
            1e-3,
            1e-3
        );
    }

    // OpNativeLayerNormTest.IntTensorsDies: non-floating input dtype fails.
    // [spec:et:sem:op-native-layer-norm.torch.executor.native.opt-native-layer-norm-out-fn/test]
    #[test]
    fn opt_native_layer_norm_out_int_dies() {
        let tf = TensorFactory::<i32>::new();
        let input = tf.make_default(vec![2, 3], vec![1, 0, -1, -1, 4, 0]);
        let weight = tf.make_default(vec![3], vec![1, 1, 1]);
        let bias = tf.make_default(vec![3], vec![0, 0, 0]);
        let out = tf.zeros_default(vec![2, 3]);
        let mean = tf.zeros(vec![2, 3], TensorShapeDynamism::DYNAMIC_BOUND);
        let rstd = tf.zeros(vec![2, 3], TensorShapeDynamism::DYNAMIC_BOUND);
        let shape = [3i64];

        let mut ctx = context();
        et_expect_kernel_failure!(
            ctx,
            opt_native_layer_norm_out(
                &mut ctx,
                &input,
                dim_ref(&shape),
                Some(&weight),
                Some(&bias),
                1.0e-5,
                &out,
                &mean,
                &rstd,
            )
        );
    }

    // OpNativeLayerNormTest.WrongNomalizedShape: normalized_shape must match
    // the trailing input dim.
    // [spec:et:sem:op-native-layer-norm.torch.executor.native.opt-native-layer-norm-out-fn/test]
    #[test]
    fn opt_native_layer_norm_out_wrong_normalized_shape_dies() {
        let tf = TensorFactory::<f32>::new();
        let input = tf.make_default(vec![2, 3], vec![1.0, 0.0, -1.0, -1.0, 4.0, 0.0]);
        let weight = tf.make_default(vec![4], vec![1.0, 1.0, 1.0, 1.0]);
        let bias = tf.make_default(vec![4], vec![0.0, 0.0, 0.0, 0.0]);
        let out = tf.zeros_default(vec![2, 3]);
        let mean = tf.zeros(vec![2, 3], TensorShapeDynamism::DYNAMIC_BOUND);
        let rstd = tf.zeros(vec![2, 3], TensorShapeDynamism::DYNAMIC_BOUND);
        let shape = [4i64];

        let mut ctx = context();
        et_expect_kernel_failure!(
            ctx,
            opt_native_layer_norm_out(
                &mut ctx,
                &input,
                dim_ref(&shape),
                Some(&weight),
                Some(&bias),
                1.0e-5,
                &out,
                &mean,
                &rstd,
            )
        );
    }
}
