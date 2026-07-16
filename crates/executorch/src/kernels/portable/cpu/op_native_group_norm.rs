//! Literal port of kernels/portable/cpu/op_native_group_norm.cpp.

use crate::kernels::portable::cpu::util::normalization_ops_util::check_group_norm_args;
use crate::kernels::portable::cpu::vec_ops::{ReduceToF32, reduce_add, vec_powerf};
use crate::runtime::core::array_ref::ArrayRef;
use crate::runtime::core::error::Error;
use crate::runtime::core::exec_aten::util::tensor_util::{
    K_TENSOR_DIMENSION_LIMIT, resize_tensor, tensor_is_default_dim_order,
    tensors_have_same_dim_order2, tensors_have_same_dim_order4,
};
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::core::portable_type::tensor_impl::SizesType;
use crate::runtime::core::portable_type::{BFloat16, Half};
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

// PORT-NOTE: the templated anonymous-namespace `group_norm<CTYPE>` becomes a
// generic `fn group_norm<CTYPE: GnElem>`. The worker computes statistics in
// `float` (via `reduce_add`/`vec_powerf`) then finalizes/normalizes in `double`,
// narrowing back to CTYPE for the stores. Rust cannot express these
// `static_cast<double>(CTYPE)` / `static_cast<CTYPE>(double)` conversions and the
// `std::sqrt`/`NAN` generically, so — mirroring op_native_batch_norm.rs's
// per-type-trait strategy — `GnElem` provides the conversions, one impl per
// FLOATHBF16 ctype.
trait GnElem: Copy {
    fn to_f64(self) -> f64;
    fn from_f64(v: f64) -> Self;
    fn from_i32(v: i32) -> Self;
    fn from_f32(v: f32) -> Self;
}
macro_rules! impl_gn_elem_native {
    ($t:ty) => {
        impl GnElem for $t {
            fn to_f64(self) -> f64 {
                self as f64
            }
            fn from_f64(v: f64) -> Self {
                v as $t
            }
            fn from_i32(v: i32) -> Self {
                v as $t
            }
            fn from_f32(v: f32) -> Self {
                v as $t
            }
        }
    };
}
impl_gn_elem_native!(f32);
impl_gn_elem_native!(f64);
impl GnElem for Half {
    fn to_f64(self) -> f64 {
        self.to_f64()
    }
    fn from_f64(v: f64) -> Self {
        Half::from_f64(v)
    }
    fn from_i32(v: i32) -> Self {
        Half::from_f32(v as f32)
    }
    fn from_f32(v: f32) -> Self {
        Half::from_f32(v)
    }
}
impl GnElem for BFloat16 {
    fn to_f64(self) -> f64 {
        self.to_f64()
    }
    fn from_f64(v: f64) -> Self {
        BFloat16::from_f64(v)
    }
    fn from_i32(v: i32) -> Self {
        BFloat16::from_f32(v as f32)
    }
    fn from_f32(v: f32) -> Self {
        BFloat16::from_f32(v)
    }
}

// [spec:et:def:op-native-group-norm.torch.executor.native.group-norm-fn]
// [spec:et:sem:op-native-group-norm.torch.executor.native.group-norm-fn]
#[allow(clippy::too_many_arguments)]
fn group_norm<CTYPE: GnElem + ReduceToF32>(
    input: &Tensor,
    weight: Option<&Tensor>,
    bias: Option<&Tensor>,
    s_n: i64,
    s_c: i64,
    s_hxw: i64,
    group: i64,
    eps: f64,
    out: &Tensor,
    mean: &Tensor,
    rstd: &Tensor,
) {
    let _n: usize = s_n as usize;
    let c: usize = s_c as usize;
    let hxw: usize = s_hxw as usize;
    let g: usize = group as usize;

    let leading: usize = _n * g;
    let d: usize = c / g;
    let inner_size: usize = d * hxw;

    if leading == 0 {
        return;
    }

    let out_data: *mut CTYPE = out.mutable_data_ptr::<CTYPE>();
    let mean_data: *mut CTYPE = mean.mutable_data_ptr::<CTYPE>();
    let rstd_data: *mut CTYPE = rstd.mutable_data_ptr::<CTYPE>();

    if inner_size == 0 {
        for i in 0..leading {
            unsafe {
                *mean_data.add(i) = CTYPE::from_i32(0);
                *rstd_data.add(i) = CTYPE::from_f32(f32::NAN);
            }
        }
        return;
    }

    let input_data: *const CTYPE = input.const_data_ptr::<CTYPE>();
    let weight_data: *const CTYPE = match weight {
        Some(w) => w.const_data_ptr::<CTYPE>(),
        None => core::ptr::null(),
    };
    let bias_data: *const CTYPE = match bias {
        Some(b) => b.const_data_ptr::<CTYPE>(),
        None => core::ptr::null(),
    };

    for i in 0..leading {
        let mut x: *const CTYPE = unsafe { input_data.add(i * inner_size) };

        // compute E[X] and Var[x] = E[x^2] - E[x]^2
        let sum: f32 = unsafe { reduce_add(x, inner_size) };
        let sq_sum: f32 = unsafe { vec_powerf(x, inner_size) };
        let mean_value: f64 = (sum as f64) / (inner_size as f64);
        let variance: f64 = (sq_sum as f64) / (inner_size as f64) - mean_value * mean_value;
        let std: f64 = (variance + eps).sqrt();
        let rstd_value: f64 = 1.0 / std;

        // Calculate the elements of output
        if weight_data.is_null() && bias_data.is_null() {
            let y: *mut CTYPE = unsafe { out_data.add(i * inner_size) };
            for j in 0..inner_size {
                unsafe {
                    *y.add(j) = CTYPE::from_f64(((*x.add(j)).to_f64() - mean_value) * rstd_value);
                }
            }
        } else {
            let g_idx: usize = i % g;
            for j in 0..d {
                let ch: usize = g_idx * d + j;
                let scale: f64 = rstd_value
                    * (if weight_data.is_null() {
                        1.0f64
                    } else {
                        unsafe { *weight_data.add(ch) }.to_f64()
                    });
                let beta: f64 = -scale * mean_value
                    + (if bias_data.is_null() {
                        0.0f64
                    } else {
                        unsafe { *bias_data.add(ch) }.to_f64()
                    });
                x = unsafe { input_data.add((i * d + j) * hxw) };
                let y: *mut CTYPE = unsafe { out_data.add((i * d + j) * hxw) };
                for k in 0..hxw {
                    unsafe {
                        *y.add(k) = CTYPE::from_f64(scale * (*x.add(k)).to_f64() + beta);
                    }
                }
            }
        }

        unsafe {
            *mean_data.add(i) = CTYPE::from_f64(mean_value);
            *rstd_data.add(i) = CTYPE::from_f64(rstd_value);
        }
    }
}

// PORT-NOTE: `Tensor& out/mean_out/rstd_out` and the returned
// `std::tuple<Tensor&, Tensor&, Tensor&>` become `&'a Tensor` handles and a Rust
// 3-tuple. `const std::optional<Tensor>&` maps to `Option<&Tensor>`. The C++
// `resize_tensor(mean_out, {mean_rstd_sizes, 2})` uses a fixed-ndim (2) shape
// built from `{N, group}`.

// [spec:et:def:op-native-group-norm.torch.executor.native.native-group-norm-out-fn]
// [spec:et:sem:op-native-group-norm.torch.executor.native.native-group-norm-out-fn]
#[allow(clippy::too_many_arguments)]
pub fn native_group_norm_out<'a, 'b, 'c, 'd>(
    ctx: &mut KernelRuntimeContext,
    input: &Tensor,
    weight: Option<&Tensor>,
    bias: Option<&Tensor>,
    n: i64,
    c: i64,
    hxw: i64,
    group: i64,
    eps: f64,
    out: &'a Tensor<'b>,
    mean_out: &'a Tensor<'c>,
    rstd_out: &'a Tensor<'d>,
) -> (&'a Tensor<'b>, &'a Tensor<'c>, &'a Tensor<'d>) {
    // (void)ctx;

    let ret_val = (out, mean_out, rstd_out);

    crate::et_kernel_check!(
        ctx,
        check_group_norm_args(
            input, weight, bias, n, c, hxw, group, out, mean_out, rstd_out
        ),
        InvalidArgument,
        ret_val
    );

    let mut mean_rstd_sizes: [SizesType; K_TENSOR_DIMENSION_LIMIT] = [0; K_TENSOR_DIMENSION_LIMIT];
    mean_rstd_sizes[0] = n as SizesType;
    mean_rstd_sizes[1] = group as SizesType;

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
            ArrayRef::from_raw_parts(mean_rstd_sizes.as_ptr(), 2)
        ) == Error::Ok,
        InvalidArgument,
        ret_val
    );

    crate::et_kernel_check!(
        ctx,
        resize_tensor(
            rstd_out,
            ArrayRef::from_raw_parts(mean_rstd_sizes.as_ptr(), 2)
        ) == Error::Ok,
        InvalidArgument,
        ret_val
    );

    crate::et_kernel_check!(
        ctx,
        tensor_is_default_dim_order(input),
        InvalidArgument,
        ret_val
    );

    crate::et_kernel_check!(
        ctx,
        tensors_have_same_dim_order4(input, out, mean_out, rstd_out),
        InvalidArgument,
        ret_val
    );

    if let Some(weight) = weight {
        crate::et_kernel_check!(
            ctx,
            tensors_have_same_dim_order2(input, weight),
            InvalidArgument,
            ret_val
        );
    }

    if let Some(bias) = bias {
        crate::et_kernel_check!(
            ctx,
            tensors_have_same_dim_order2(input, bias),
            InvalidArgument,
            ret_val
        );
    }

    let name = "native_group_norm.out";

    crate::et_switch_floathbf16_types!(input.scalar_type(), ctx, name, CTYPE, {
        group_norm::<CTYPE>(
            input, weight, bias, n, c, hxw, group, eps, out, mean_out, rstd_out,
        );
    });

    ret_val
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::core::exec_aten::testing_util::tensor_factory::{
        FactoryValue, TensorFactory,
    };
    use crate::runtime::core::exec_aten::testing_util::tensor_util::internal::{
        K_DEFAULT_BFLOAT16_ATOL, K_DEFAULT_HALF_ATOL,
    };
    use crate::runtime::core::exec_aten::util::scalar_type_util::CppTypeToScalarType;
    use crate::runtime::core::memory_allocator::{MemoryAllocator, MemoryAllocatorBase};
    use crate::runtime::core::portable_type::scalar_type::ScalarType;
    use crate::runtime::core::portable_type::{BFloat16, Half};
    use crate::runtime::core::tensor_shape_dynamism::TensorShapeDynamism;
    use crate::{assert_tensor_close, assert_tensor_close_with_tol};

    fn context() -> KernelRuntimeContext<'static> {
        crate::runtime::platform::platform::pal_init();
        KernelRuntimeContext::new(
            crate::extension::module::module::null_event_tracer(),
            core::ptr::null_mut::<MemoryAllocator>() as *mut dyn MemoryAllocatorBase,
        )
    }

    trait FromF64: Copy {
        fn from_f64(v: f64) -> Self;
    }
    impl FromF64 for f32 {
        fn from_f64(v: f64) -> Self {
            v as f32
        }
    }
    impl FromF64 for f64 {
        fn from_f64(v: f64) -> Self {
            v
        }
    }
    impl FromF64 for i32 {
        fn from_f64(v: f64) -> Self {
            v as i32
        }
    }
    impl FromF64 for Half {
        fn from_f64(v: f64) -> Self {
            Half::from_f32(v as f32)
        }
    }
    impl FromF64 for BFloat16 {
        fn from_f64(v: f64) -> Self {
            BFloat16::from_f32(v as f32)
        }
    }

    struct TestCase {
        sizes: Vec<i32>,
        input_data: Vec<f64>,
        weight_data: Vec<f64>,
        bias_data: Vec<f64>,
        n: i64,
        c: i64,
        hxw: i64,
        group: i64,
        eps: f64,
        expected_data: Vec<f64>,
        expected_mean_data: Vec<f64>,
        expected_rstd_data: Vec<f64>,
    }

    fn run_test_cases<T>(test_cases: Vec<TestCase>)
    where
        T: CppTypeToScalarType + FactoryValue + FromF64,
    {
        let tf = TensorFactory::<T>::new();
        let vv = |vals: &[f64]| -> Vec<T> { vals.iter().map(|&x| T::from_f64(x)).collect() };
        for tc in &test_cases {
            let in_ = tf.make_default(tc.sizes.clone(), vv(&tc.input_data));
            let weight = tf.make_default(vec![tc.c as i32], vv(&tc.weight_data));
            let bias = tf.make_default(vec![tc.c as i32], vv(&tc.bias_data));
            let out0 = tf.zeros_default(tc.sizes.clone());
            let out1 = tf.zeros(
                vec![tc.n as i32, tc.group as i32],
                TensorShapeDynamism::DYNAMIC_BOUND,
            );
            let out2 = tf.zeros(
                vec![tc.n as i32, tc.group as i32],
                TensorShapeDynamism::DYNAMIC_BOUND,
            );

            let mut ctx = context();
            let result = native_group_norm_out(
                &mut ctx,
                &in_,
                Some(&weight),
                Some(&bias),
                tc.n,
                tc.c,
                tc.hxw,
                tc.group,
                tc.eps,
                &out0,
                &out1,
                &out2,
            );
            assert_tensor_close!(out0, *result.0);

            let expected = tf.make_default(tc.sizes.clone(), vv(&tc.expected_data));
            let expected_mean = tf.make_default(
                vec![tc.n as i32, tc.group as i32],
                vv(&tc.expected_mean_data),
            );
            let expected_rstd = tf.make_default(
                vec![tc.n as i32, tc.group as i32],
                vv(&tc.expected_rstd_data),
            );

            if T::VALUE == ScalarType::Half {
                assert_tensor_close_with_tol!(out0, expected, 1e-2, K_DEFAULT_HALF_ATOL);
                assert_tensor_close_with_tol!(out1, expected_mean, 1e-2, K_DEFAULT_HALF_ATOL);
                assert_tensor_close_with_tol!(out2, expected_rstd, 1e-2, K_DEFAULT_HALF_ATOL);
            } else if T::VALUE == ScalarType::BFloat16 {
                assert_tensor_close_with_tol!(out0, expected, 1e-2, K_DEFAULT_BFLOAT16_ATOL);
                assert_tensor_close_with_tol!(out1, expected_mean, 1e-2, K_DEFAULT_BFLOAT16_ATOL);
                assert_tensor_close_with_tol!(out2, expected_rstd, 1e-2, K_DEFAULT_BFLOAT16_ATOL);
            } else {
                assert_tensor_close!(out0, expected);
                assert_tensor_close!(out1, expected_mean);
                assert_tensor_close!(out2, expected_rstd);
            }
        }
    }

    fn run_floating_point_test_cases<T>()
    where
        T: CppTypeToScalarType + FactoryValue + FromF64,
    {
        let test_cases = vec![
            TestCase {
                sizes: vec![5, 6, 2, 2],
                input_data: vec![
                    -0.8125, 0.0625, -2.7500, -3.0625, -1.1250, -2.1250, -1.3125, -4.0625, 2.8125,
                    -2.0625, 4.2500, 3.5000, -0.3750, 1.6250, 4.3125, -1.0625, -2.8750, 3.3750,
                    4.9375, 4.0625, -3.0625, -1.8750, -2.7500, -2.5625, -0.1875, -3.0000, -2.7500,
                    0.6875, -3.2500, -3.1875, 1.0000, -4.6250, -0.1875, -1.7500, 4.5000, -1.8750,
                    -2.6875, 4.8125, -3.8125, -2.9375, -1.1875, 2.8750, 0.7500, 2.8750, 1.1250,
                    -0.6250, -2.2500, -3.7500, 3.2500, -0.3750, -2.0625, -4.7500, 2.0625, 3.0000,
                    -3.1875, -4.1250, -3.7500, 1.2500, -2.3125, 1.5625, 3.1250, 0.3125, 3.2500,
                    -2.7500, -3.8125, -4.2500, -4.3125, -0.5625, -0.4375, 2.9375, -1.3750, -0.6250,
                    -2.5625, -4.5625, 0.1250, -3.5000, -5.0000, -1.0000, -4.6875, -0.6875, 1.1250,
                    1.8750, -4.5000, 4.3125, 4.5625, 0.2500, -3.6250, 4.5625, -3.5000, -2.1250,
                    -3.6250, -2.9375, 3.6875, 3.9375, 4.3750, 3.0625, 2.4375, 2.0625, -2.4375,
                    -3.9375, 3.6875, 2.7500, -0.8750, -0.9375, 2.7500, -2.4375, -2.3750, -0.9375,
                    -4.8750, 0.1875, 3.5000, -2.0000, -0.2500, -2.7500, 0.3125, 1.2500, -0.5625,
                    0.0000, 1.8125, 1.0625,
                ],
                weight_data: vec![4.5625, -2.8750, -0.6875, 0.5625, -2.0625, -2.7500],
                bias_data: vec![-0.5000, -2.7500, 1.1875, 3.6875, 3.8125, 4.6875],
                n: 5,
                c: 6,
                hxw: 4,
                group: 3,
                eps: 1e-5,
                expected_data: vec![
                    3.419882, 6.578348, -3.573864, -4.701888, -4.509254, -2.234663, -4.082768,
                    2.172355, 0.838826, 2.270225, 0.416747, 0.636962, 3.207030, 3.687500, 4.333131,
                    3.041869, 5.547079, 1.649148, 0.674665, 1.220376, 7.156189, 6.168714, 6.896327,
                    6.740410, 3.509863, -3.022041, -2.441427, 5.542011, -0.794903, -0.886369,
                    -7.014627, 1.217361, 1.120617, 1.463606, 0.091652, 1.491045, 3.293219,
                    4.640229, 3.091168, 3.248319, 4.895990, 1.114683, 3.092597, 1.114683, 3.262238,
                    5.434066, 7.450763, 9.312329, 5.570122, 0.101119, -2.444796, -6.499403,
                    -5.446074, -6.337338, -0.454995, 0.436269, 2.228491, 0.871598, 1.838385,
                    0.786793, 4.362284, 3.737805, 4.390039, 3.057817, 5.814659, 6.202621, 6.258044,
                    2.932658, 3.366583, -0.623879, 4.475045, 3.588276, -0.082914, -4.936279,
                    6.438795, -2.357929, 0.714463, -5.402106, 0.236606, -5.879963, 1.176247,
                    1.021916, 2.333727, 0.520341, 4.275447, 3.549392, 2.896994, 4.275447, 6.120910,
                    5.298480, 6.195676, 5.784461, 2.033296, 1.833920, 1.485010, 2.531738, 3.193988,
                    2.532378, -5.406940, -8.053379, -6.467402, -5.425139, -1.395059, -1.325575,
                    0.266062, 1.622680, 1.606336, 1.230405, 2.809896, 3.893110, 4.601880, 3.425055,
                    4.374411, 8.283354, 3.494898, 2.029045, 6.088204, 4.915522, 1.136877, 2.700454,
                ],
                expected_mean_data: vec![
                    -1.89843750,
                    1.62500000,
                    -0.09375000,
                    -1.91406250,
                    -0.49218744,
                    -0.02343750,
                    -0.77343756,
                    0.08593753,
                    -1.55468738,
                    -2.73437500,
                    1.07031238,
                    0.35937503,
                    0.34374997,
                    -0.77343750,
                    0.10937499,
                ],
                expected_rstd_data: vec![
                    0.79116172, 0.42708409, 0.30238494, 0.50903118, 0.31929117, 0.45128885,
                    0.33067191, 0.39473253, 0.42994878, 0.53187561, 0.29930803, 0.29000264,
                    0.38669431, 0.38038814, 0.75809801,
                ],
            },
            TestCase {
                sizes: vec![1, 4, 3],
                input_data: vec![
                    0.0, 1000.0, 2000.0, 3000.0, 4000.0, 5000.0, 6000.0, 7000.0, 8000.0, 9000.0,
                    10000.0, 11000.0,
                ],
                weight_data: vec![1.0, 1.0, 1.0, 1.0],
                bias_data: vec![0.0, 0.0, 0.0, 0.0],
                n: 1,
                c: 4,
                hxw: 3,
                group: 2,
                eps: 1e-5,
                expected_data: vec![
                    -1.46385, -0.87831, -0.29277, 0.29277, 0.87831, 1.46385, -1.46385, -0.87831,
                    -0.29277, 0.29277, 0.87831, 1.46385,
                ],
                expected_mean_data: vec![2500.0, 8500.0],
                expected_rstd_data: vec![0.00058554, 0.00058554],
            },
        ];
        run_test_cases::<T>(test_cases);
    }

    // run_death_test_cases: builds weight/bias only when data is non-empty; mean/rstd
    // outputs sized {N, group} with default (STATIC) dynamism.
    fn run_death_test_cases<T>(test_cases: Vec<TestCase>)
    where
        T: CppTypeToScalarType + FactoryValue + FromF64,
    {
        let tf = TensorFactory::<T>::new();
        let vv = |vals: &[f64]| -> Vec<T> { vals.iter().map(|&x| T::from_f64(x)).collect() };
        for tc in &test_cases {
            let in_ = tf.make_default(tc.sizes.clone(), vv(&tc.input_data));
            let weight = if !tc.weight_data.is_empty() {
                Some(tf.make_default(vec![tc.c as i32], vv(&tc.weight_data)))
            } else {
                None
            };
            let bias = if !tc.bias_data.is_empty() {
                Some(tf.make_default(vec![tc.c as i32], vv(&tc.bias_data)))
            } else {
                None
            };
            let out0 = tf.zeros_default(tc.sizes.clone());
            let out1 = tf.zeros_default(vec![tc.n as i32, tc.group as i32]);
            let out2 = tf.zeros_default(vec![tc.n as i32, tc.group as i32]);

            let mut ctx = context();
            native_group_norm_out(
                &mut ctx,
                &in_,
                weight.as_ref(),
                bias.as_ref(),
                tc.n,
                tc.c,
                tc.hxw,
                tc.group,
                tc.eps,
                &out0,
                &out1,
                &out2,
            );
            assert_ne!(ctx.failure_state(), Error::Ok);
        }
    }

    fn run_int_test_cases<T>()
    where
        T: CppTypeToScalarType + FactoryValue + FromF64,
    {
        let test_cases = vec![TestCase {
            // Cannot be represented by a type other than float.
            sizes: vec![2, 4, 2, 2],
            input_data: vec![
                1.0, 0.0, -1.0, -1.0, 4.0, 0.0, 2.0, -2.0, 1.0, 0.0, -1.0, -1.0, 4.0, 0.0, 2.0,
                -2.0, 1.0, 0.0, -1.0, -1.0, 4.0, 0.0, 2.0, -2.0, 1.0, 0.0, -1.0, -1.0, 4.0, 0.0,
                2.0, -2.0,
            ],
            weight_data: vec![1.0, 1.0, 1.0, 1.0],
            bias_data: vec![0.0, 0.0, 0.0, 0.0],
            n: 2,
            c: 4,
            hxw: 4,
            group: 2,
            eps: 1.0,
            expected_data: vec![
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
            expected_mean_data: vec![0.0, 0.0, 0.0, 0.0],
            expected_rstd_data: vec![1.0, 1.0, 1.0, 1.0],
        }];
        run_death_test_cases::<T>(test_cases);
    }

    // [spec:et:sem:op-native-group-norm.torch.executor.native.native-group-norm-out-fn/test]
    // [spec:et:sem:op-native-group-norm.torch.executor.native.group-norm-fn/test]
    #[test]
    fn op_native_group_norm_test_double_tensors() {
        run_floating_point_test_cases::<f64>();
    }

    // [spec:et:sem:op-native-group-norm.torch.executor.native.native-group-norm-out-fn/test]
    // [spec:et:sem:op-native-group-norm.torch.executor.native.group-norm-fn/test]
    #[test]
    fn op_native_group_norm_test_float_tensors() {
        run_floating_point_test_cases::<f32>();
    }

    // [spec:et:sem:op-native-group-norm.torch.executor.native.native-group-norm-out-fn/test]
    // [spec:et:sem:op-native-group-norm.torch.executor.native.group-norm-fn/test]
    #[test]
    fn op_native_group_norm_test_half_tensors() {
        run_floating_point_test_cases::<Half>();
    }

    // [spec:et:sem:op-native-group-norm.torch.executor.native.native-group-norm-out-fn/test]
    // [spec:et:sem:op-native-group-norm.torch.executor.native.group-norm-fn/test]
    #[test]
    fn op_native_group_norm_test_b_float16_tensors() {
        run_floating_point_test_cases::<BFloat16>();
    }

    // [spec:et:sem:op-native-group-norm.torch.executor.native.native-group-norm-out-fn/test]
    #[test]
    fn op_native_group_norm_test_int_tensors_dies() {
        // Cannot be represented by a type other than float.
        run_int_test_cases::<i32>();
    }
}
