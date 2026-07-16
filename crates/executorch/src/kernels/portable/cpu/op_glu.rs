//! Literal port of kernels/portable/cpu/op_glu.cpp.

use crate::kernels::portable::cpu::util::activation_ops_util::{check_glu_args, resize_glu_out};
use crate::kernels::portable::cpu::util::dtype_util::SupportedTensorDtypes;
use crate::kernels::portable::cpu::util::elementwise_util::apply_bitensor_elementwise_fn;
use crate::runtime::core::error::Error;
use crate::runtime::core::exec_aten::util::scalar_type_util::is_floating_type;
use crate::runtime::core::exec_aten::util::tensor_util::{
    K_TENSOR_DIMENSION_LIMIT, tensors_have_same_dim_order2,
};
use crate::runtime::core::portable_type::device::DeviceType;
use crate::runtime::core::portable_type::scalar_type::ScalarType;
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::core::portable_type::tensor_impl::{SizesType, TensorImpl};
use crate::runtime::core::portable_type::{BFloat16, Half};
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

// PORT-NOTE: the C++ closure `val_a * (one / (one + std::exp(-val_b)))` runs over
// the FLOATHBF16 compute set. `std::exp` is not generic in Rust and does not
// accept Half/BFloat16, and those types have no `Float`-trait impl; mirroring
// op_gelu.rs / op_elu.rs, the element op is a `GluSigmoid` trait with one impl
// per CTYPE reproducing the exact C++ arithmetic (native for Float/Double; the
// `std::exp(half)`→float promotion for the reduced types).
trait GluSigmoid: Copy {
    fn glu_sigmoid(self, val_b: Self) -> Self;
}
macro_rules! impl_glu_sigmoid_native {
    ($t:ty) => {
        impl GluSigmoid for $t {
            fn glu_sigmoid(self, val_b: $t) -> $t {
                let val_a = self;
                let one: $t = 1.0 as $t;
                val_a * (one / (one + (-val_b).exp()))
            }
        }
    };
}
impl_glu_sigmoid_native!(f32);
impl_glu_sigmoid_native!(f64);
macro_rules! impl_glu_sigmoid_reduced {
    ($t:ty) => {
        impl GluSigmoid for $t {
            fn glu_sigmoid(self, val_b: $t) -> $t {
                // std::exp(-val_b): -val_b (Half) promotes to float, expf; the
                // remaining arithmetic then proceeds in float.
                let val_a = self.to_f32();
                let one: f32 = 1.0;
                <$t>::from_f32(val_a * (one / (one + (-val_b.to_f32()).exp())))
            }
        }
    };
}
impl_glu_sigmoid_reduced!(Half);
impl_glu_sigmoid_reduced!(BFloat16);

// PORT-NOTE: the C++ anonymous-namespace `SplitGLUInputTensor` owns two aliasing
// `TensorImpl`s (views into `self`'s buffer) plus the two `Tensor`s wrapping
// them. In Rust the `Tensor`s hold `*mut TensorImpl`, so the wrapping `Tensor`s
// are built by the caller (`glu_out_tensor`) from raw pointers into this struct's
// impls after it is pinned, keeping the impls alive for the duration of use.
// [spec:et:def:op-glu.torch.executor.native.split-glu-input-tensor]
struct SplitGLUInputTensor {
    #[allow(dead_code)]
    half_sizes: [SizesType; K_TENSOR_DIMENSION_LIMIT],
    first_half_impl: TensorImpl,
    second_half_impl: TensorImpl,
}

impl SplitGLUInputTensor {
    // [spec:et:def:op-glu.torch.executor.native.split-glu-input-tensor.get-half-sizes-fn]
    // [spec:et:sem:op-glu.torch.executor.native.split-glu-input-tensor.get-half-sizes-fn]
    fn get_half_sizes(self_: &Tensor, dim: i64) -> [SizesType; K_TENSOR_DIMENSION_LIMIT] {
        let mut half_sizes: [SizesType; K_TENSOR_DIMENSION_LIMIT] = [0; K_TENSOR_DIMENSION_LIMIT];
        let sizes = self_.sizes();
        // std::copy(self.sizes().begin(), self.sizes().end(), half_sizes.begin())
        for i in 0..sizes.size() {
            half_sizes[i] = *sizes.at(i);
        }
        half_sizes[dim as usize] /= 2;
        half_sizes
    }

    // [spec:et:def:op-glu.torch.executor.native.split-glu-input-tensor.split-glu-input-tensor-fn]
    // [spec:et:sem:op-glu.torch.executor.native.split-glu-input-tensor.split-glu-input-tensor-fn]
    fn new(self_: &Tensor, dim: i64) -> alloc::boxed::Box<SplitGLUInputTensor> {
        let half_sizes = Self::get_half_sizes(self_, dim);

        // The two impls alias `self`'s buffer and point at THIS struct's
        // `half_sizes` (which must have a stable address). Allocate the box
        // uninitialized to obtain that address before constructing the impls,
        // matching the C++ member-init order (`half_sizes` first, then the impls
        // reference `half_sizes.data()`).
        let mut boxed: alloc::boxed::Box<core::mem::MaybeUninit<SplitGLUInputTensor>> =
            alloc::boxed::Box::new_uninit();
        let struct_ptr = boxed.as_mut_ptr();
        let sizes_ptr =
            unsafe { core::ptr::addr_of_mut!((*struct_ptr).half_sizes) as *mut SizesType };
        unsafe {
            core::ptr::addr_of_mut!((*struct_ptr).half_sizes).write(half_sizes);
        }

        // first_half_impl: aliases self's data at the start of the buffer.
        let first = Self::make_impl(self_, sizes_ptr, self_.mutable_data_ptr_typed());

        // second_half_impl: same, but data advanced to the second half:
        // base + strides[dim] * size(dim) / 2 * element_size() bytes.
        let strides = self_.strides();
        let offset_bytes: isize =
            (*strides.at(dim as usize) as isize) * (self_.size(dim as isize) as isize) / 2
                * (self_.element_size() as isize);
        let second_data = unsafe {
            (self_.mutable_data_ptr_typed() as *mut u8).offset(offset_bytes)
                as *mut core::ffi::c_void
        };
        let second = Self::make_impl(self_, sizes_ptr, second_data);

        unsafe {
            core::ptr::addr_of_mut!((*struct_ptr).first_half_impl).write(first);
            core::ptr::addr_of_mut!((*struct_ptr).second_half_impl).write(second);
            boxed.assume_init()
        }
    }

    fn make_impl(
        self_: &Tensor,
        sizes: *mut SizesType,
        data: *mut core::ffi::c_void,
    ) -> TensorImpl {
        TensorImpl::new(
            self_.scalar_type(),
            self_.dim(),
            sizes,
            data,
            self_.dim_order().data() as *mut _,
            self_.strides().data() as *mut _,
            self_.shape_dynamism(),
            DeviceType::CPU,
            0,
        )
    }
}

// PORT-NOTE: `glu_out_tensor` is templated on `CTYPE_IN`/`CTYPE_OUT` in C++ but
// its body never names them (they only drive the FLOATHBF16 double-dispatch in
// `glu_out`); the ported function drops the unused type parameters. `(void)ctx;`
// is implicit.
// [spec:et:def:op-glu.torch.executor.native.glu-out-tensor-fn]
// [spec:et:sem:op-glu.torch.executor.native.glu-out-tensor-fn]
fn glu_out_tensor<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    self_: &Tensor,
    dim: i64,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    crate::et_kernel_check!(
        ctx,
        self_.dim() <= K_TENSOR_DIMENSION_LIMIT as isize,
        InvalidArgument,
        out
    );
    let mut split_input = SplitGLUInputTensor::new(self_, dim);
    let first_half = Tensor::new(&mut split_input.first_half_impl as *mut TensorImpl);
    let second_half = Tensor::new(&mut split_input.second_half_impl as *mut TensorImpl);

    let mut compute_type: ScalarType = if is_floating_type(self_.scalar_type()) {
        self_.scalar_type()
    } else {
        ScalarType::Float
    };
    let _ = &mut compute_type;

    let op_name = "glu.out";
    crate::et_switch_floathbf16_types!(compute_type, ctx, op_name, CTYPE_COMPUTE, {
        apply_bitensor_elementwise_fn::<CTYPE_COMPUTE, _>(
            |vals: &[CTYPE_COMPUTE]| -> CTYPE_COMPUTE {
                let val_a = vals[0];
                let val_b = vals[1];
                // TODO: rewrite this to be vectorization-capable? the tensors
                // might not be contiguous.
                <CTYPE_COMPUTE as GluSigmoid>::glu_sigmoid(val_a, val_b)
            },
            ctx,
            &first_half,
            SupportedTensorDtypes::FLOATHBF16,
            &second_half,
            SupportedTensorDtypes::FLOATHBF16,
            out,
            SupportedTensorDtypes::FLOATHBF16,
            // utils::internal::SupportNoncontiguousInputTensors()
            true,
        );
    });
    out
}

// PORT-NOTE: `(void)ctx;` dropped. `Tensor& out` / returned `Tensor&` become
// `&'a Tensor`. The C++ double-switch selects CTYPE_IN/CTYPE_OUT only to
// instantiate `glu_out_tensor`; since the ported body ignores both, the inner
// switch collapses to a single call after the FLOATHBF16 validity checks.
// [spec:et:def:op-glu.torch.executor.native.glu-out-fn]
// [spec:et:sem:op-glu.torch.executor.native.glu-out-fn]
pub fn glu_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    self_: &Tensor,
    dim: i64,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    // (void)ctx;

    crate::et_kernel_check!(
        ctx,
        resize_glu_out(self_, dim, out) == Error::Ok,
        InvalidArgument,
        out
    );

    crate::et_kernel_check!(
        ctx,
        tensors_have_same_dim_order2(self_, out),
        InvalidArgument,
        out
    );

    crate::et_kernel_check!(ctx, check_glu_args(self_, dim, out), InvalidArgument, out);

    let non_negative_dim: usize = if dim < 0 {
        (dim + self_.dim() as i64) as usize
    } else {
        dim as usize
    };
    let in_dtype = self_.scalar_type();

    crate::et_switch_floathbf16_types!(in_dtype, ctx, "glu", CTYPE_IN, {
        crate::et_switch_floathbf16_types!(out.scalar_type(), ctx, "glu", CTYPE_OUT, {
            glu_out_tensor(ctx, self_, non_negative_dim as i64, out);
        });
    });

    out
}

extern crate alloc;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::core::exec_aten::testing_util::tensor_factory::{
        FactoryValue, TensorFactory,
    };
    use crate::runtime::core::exec_aten::testing_util::tensor_util::{internal, tensors_are_close};
    use crate::runtime::core::exec_aten::util::scalar_type_util::CppTypeToScalarType;
    use crate::runtime::core::memory_allocator::{MemoryAllocator, MemoryAllocatorBase};
    use crate::runtime::core::portable_type::tensor::Tensor as PTensor;
    use crate::runtime::core::tensor_shape_dynamism::TensorShapeDynamism;

    fn context() -> KernelRuntimeContext<'static> {
        crate::runtime::platform::runtime::runtime_init();
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

    trait FromF64: Copy {
        const SCALAR: ScalarType;
        fn from_f64(v: f64) -> Self;
    }
    impl FromF64 for f32 {
        const SCALAR: ScalarType = ScalarType::Float;
        fn from_f64(v: f64) -> Self {
            v as f32
        }
    }
    impl FromF64 for f64 {
        const SCALAR: ScalarType = ScalarType::Double;
        fn from_f64(v: f64) -> Self {
            v
        }
    }
    impl FromF64 for Half {
        const SCALAR: ScalarType = ScalarType::Half;
        fn from_f64(v: f64) -> Self {
            Half::from_f64(v)
        }
    }
    impl FromF64 for BFloat16 {
        const SCALAR: ScalarType = ScalarType::BFloat16;
        fn from_f64(v: f64) -> Self {
            BFloat16::from_f64(v)
        }
    }

    fn d<T: FromF64>(v: &[f64]) -> Vec<T> {
        v.iter().map(|&x| T::from_f64(x)).collect()
    }

    // Mirrors OpGluOutTest::expect_tensor_close<DTYPE, OUT_DTYPE>.
    fn expect_tensor_close<DTYPE: FromF64, OUT_DTYPE: FromF64>(
        actual: &PTensor,
        expected: &PTensor,
    ) {
        if DTYPE::SCALAR == ScalarType::Half
            || DTYPE::SCALAR == ScalarType::BFloat16
            || OUT_DTYPE::SCALAR == ScalarType::Half
            || OUT_DTYPE::SCALAR == ScalarType::BFloat16
        {
            assert!(
                tensors_are_close(actual, expected, 1e-2, Some(internal::K_DEFAULT_ATOL)),
                "tensors are not close within tolerance"
            );
        } else {
            assert!(
                tensors_are_close(actual, expected, internal::K_DEFAULT_RTOL, None),
                "tensors are not close"
            );
        }
    }

    fn test_glu_out<DTYPE, OUT_DTYPE>()
    where
        DTYPE: CppTypeToScalarType + FactoryValue + FromF64,
        OUT_DTYPE: CppTypeToScalarType + FactoryValue + FromF64,
    {
        let tf = TensorFactory::<DTYPE>::new();
        let tf_out = TensorFactory::<OUT_DTYPE>::new();

        let sizes = vec![4, 2];
        let out_sizes_1 = vec![2, 2];

        let in_ = tf.make_default(sizes.clone(), d::<DTYPE>(&[0., 1., 2., 3., 4., 5., 6., 7.]));
        let out = tf_out.zeros_default(out_sizes_1.clone());
        let mut ctx = context();
        glu_out(&mut ctx, &in_, 0, &out);
        expect_tensor_close::<DTYPE, OUT_DTYPE>(
            &out,
            &tf_out.make_default(
                out_sizes_1,
                d::<OUT_DTYPE>(&[0., 0.99330717, 1.99505484, 2.99726701]),
            ),
        );

        let out_sizes_2 = vec![4, 1];
        let out = tf_out.zeros_default(out_sizes_2.clone());
        glu_out(&mut ctx, &in_, 1, &out);
        expect_tensor_close::<DTYPE, OUT_DTYPE>(
            &out,
            &tf_out.make_default(
                out_sizes_2,
                d::<OUT_DTYPE>(&[0., 1.90514827, 3.97322869, 5.99453402]),
            ),
        );
    }

    fn test_glu_out_mismatched_shape<INPUT_DTYPE>()
    where
        INPUT_DTYPE: CppTypeToScalarType + FactoryValue + FromF64,
    {
        let tf_in = TensorFactory::<INPUT_DTYPE>::new();

        let in_ = tf_in.zeros_default(vec![4, 4, 4]);
        let out = tf_in.zeros_default(vec![2, 4, 2]);

        let mut ctx = context();
        et_expect_kernel_failure!(ctx, glu_out(&mut ctx, &in_, 0, &out));

        let out = tf_in.zeros_default(vec![4, 4, 4]);
        et_expect_kernel_failure!(ctx, glu_out(&mut ctx, &in_, 0, &out));
    }

    fn test_glu_out_invalid_dim<INPUT_DTYPE>()
    where
        INPUT_DTYPE: CppTypeToScalarType + FactoryValue + FromF64,
    {
        let tf_in = TensorFactory::<INPUT_DTYPE>::new();
        let in_ = tf_in.zeros_default(vec![2, 2]);
        let out_sizes = vec![1, 2];
        let out = tf_in.zeros_default(out_sizes);

        let mut ctx = context();
        // Dim is not valid
        et_expect_kernel_failure!(ctx, glu_out(&mut ctx, &in_, 3, &out));

        // Dim size is not even
        let in_ = tf_in.zeros_default(vec![3, 2]);
        et_expect_kernel_failure!(ctx, glu_out(&mut ctx, &in_, 0, &out));
    }

    fn test_div_invalid_input_dtype_dies<INPUT_DTYPE>()
    where
        INPUT_DTYPE: CppTypeToScalarType + FactoryValue,
    {
        let tf_in = TensorFactory::<INPUT_DTYPE>::new();
        let tf_float = TensorFactory::<f32>::new();

        let sizes = vec![2, 2];
        let out_sizes = vec![1, 2];
        let in_ = tf_in.ones_default(sizes);
        let out = tf_float.zeros_default(out_sizes);

        let mut ctx = context();
        et_expect_kernel_failure!(ctx, glu_out(&mut ctx, &in_, 0, &out));
    }

    fn test_div_invalid_output_dtype_dies<OUTPUT_DTYPE>()
    where
        OUTPUT_DTYPE: CppTypeToScalarType + FactoryValue,
    {
        let tf_float = TensorFactory::<f32>::new();
        let tf_out = TensorFactory::<OUTPUT_DTYPE>::new();

        let sizes = vec![2, 2];
        let out_sizes = vec![1, 2];
        let in_ = tf_float.ones_default(sizes);
        let out = tf_out.zeros_default(out_sizes);

        let mut ctx = context();
        et_expect_kernel_failure!(ctx, glu_out(&mut ctx, &in_, 0, &out));
    }

    fn forall_floathbf16_out<OUT>()
    where
        OUT: CppTypeToScalarType + FactoryValue + FromF64,
    {
        test_glu_out::<Half, OUT>();
        test_glu_out::<f32, OUT>();
        test_glu_out::<f64, OUT>();
        test_glu_out::<BFloat16, OUT>();
    }

    // [spec:et:sem:op-glu.torch.executor.native.glu-out-fn/test]
    // also verifies resize_glu_out: the output is dynamically resized to half the
    // input size along the halving dimension, and the computed values pin that shape.
    // Splitting [4,2] along dim 0 and dim 1 and computing a*sigmoid(b) pins
    // glu_out_tensor plus SplitGLUInputTensor::{get_half_sizes, new}: a wrong half
    // size or split offset would misalign the a/b operands and change the result.
    // [spec:et:sem:activation-ops-util.torch.executor.resize-glu-out-fn/test]
    // [spec:et:sem:op-glu.torch.executor.native.glu-out-tensor-fn/test]
    // [spec:et:sem:op-glu.torch.executor.native.split-glu-input-tensor.get-half-sizes-fn/test]
    // [spec:et:sem:op-glu.torch.executor.native.split-glu-input-tensor.split-glu-input-tensor-fn/test]
    #[test]
    fn op_glu_out_test_all_input_float_output_support() {
        forall_floathbf16_out::<f32>();
    }
    // [spec:et:sem:op-glu.torch.executor.native.glu-out-fn/test]
    #[test]
    fn op_glu_out_test_all_input_double_output_support() {
        forall_floathbf16_out::<f64>();
    }
    // [spec:et:sem:op-glu.torch.executor.native.glu-out-fn/test]
    #[test]
    fn op_glu_out_test_all_input_half_output_support() {
        forall_floathbf16_out::<Half>();
    }
    // [spec:et:sem:op-glu.torch.executor.native.glu-out-fn/test]
    #[test]
    fn op_glu_out_test_all_input_bfloat16_output_support() {
        forall_floathbf16_out::<BFloat16>();
    }

    // [spec:et:sem:op-glu.torch.executor.native.glu-out-fn/test]
    #[test]
    fn op_glu_out_test_infinity_and_nan_test() {
        let tf = TensorFactory::<f32>::new();
        let sizes = vec![4, 2];
        let out_sizes = vec![4, 1];
        let in_ = tf.make_default(
            sizes,
            vec![
                f32::INFINITY,
                1.,
                f32::NEG_INFINITY,
                1.,
                f32::INFINITY,
                f32::NEG_INFINITY,
                f32::NAN,
                1.,
            ],
        );
        let out = tf.zeros_default(out_sizes.clone());
        let mut ctx = context();
        glu_out(&mut ctx, &in_, 1, &out);
        assert!(tensors_are_close(
            &out,
            &tf.make_default(
                out_sizes,
                vec![f32::INFINITY, f32::NEG_INFINITY, f32::NAN, f32::NAN]
            ),
            internal::K_DEFAULT_RTOL,
            None
        ));
    }

    // PORT-NOTE: guarded in C++ by `ET_SKIP_IF(is_aten, ...)`; non-ATen build runs.
    // [spec:et:sem:op-glu.torch.executor.native.glu-out-fn/test]
    #[test]
    fn op_glu_out_test_mismatched_shapes_dies() {
        // ET_FORALL_FLOAT_TYPES
        test_glu_out_mismatched_shape::<f32>();
        test_glu_out_mismatched_shape::<f64>();
    }

    // [spec:et:sem:op-glu.torch.executor.native.glu-out-fn/test]
    // also verifies check_glu_args: exercises both the invalid-dim and the
    // non-even halving-dimension-size failure branches.
    // [spec:et:sem:activation-ops-util.torch.executor.check-glu-args-fn/test]
    #[test]
    fn op_glu_out_test_invalid_dim_dies() {
        // ET_FORALL_FLOAT_TYPES
        test_glu_out_invalid_dim::<f32>();
        test_glu_out_invalid_dim::<f64>();
    }

    // [spec:et:sem:op-glu.torch.executor.native.glu-out-fn/test]
    #[test]
    fn op_glu_out_test_all_non_float_input_dtype_dies() {
        // ET_FORALL_INT_TYPES_AND(Bool)
        test_div_invalid_input_dtype_dies::<u8>();
        test_div_invalid_input_dtype_dies::<i8>();
        test_div_invalid_input_dtype_dies::<i16>();
        test_div_invalid_input_dtype_dies::<i32>();
        test_div_invalid_input_dtype_dies::<i64>();
        test_div_invalid_input_dtype_dies::<bool>();
    }

    // [spec:et:sem:op-glu.torch.executor.native.glu-out-fn/test]
    #[test]
    fn op_glu_out_test_all_non_float_output_dtype_dies() {
        // ET_FORALL_INT_TYPES_AND(Bool)
        test_div_invalid_output_dtype_dies::<u8>();
        test_div_invalid_output_dtype_dies::<i8>();
        test_div_invalid_output_dtype_dies::<i16>();
        test_div_invalid_output_dtype_dies::<i32>();
        test_div_invalid_output_dtype_dies::<i64>();
        test_div_invalid_output_dtype_dies::<bool>();
    }

    // DISABLED: Dynamic shape not supported
    // [spec:et:sem:op-glu.torch.executor.native.glu-out-fn/test]
    #[test]
    #[ignore]
    fn op_glu_out_test_disabled_dynamic_shape_upper_bound_same_as_expected() {
        let tf = TensorFactory::<f32>::new();

        let x = tf.make_default(
            vec![4, 2],
            vec![
                0.057747602462768555,
                0.8781633377075195,
                0.4503108263015747,
                0.40363800525665283,
                0.3379024863243103,
                0.13906866312026978,
                0.6991606950759888,
                0.4374786615371704,
            ],
        );
        let expected_result = tf.make_default(
            vec![2, 2],
            vec![
                0.0337061733007431,
                0.4695638120174408,
                0.3008083701133728,
                0.2452739030122757,
            ],
        );

        let out = tf.zeros(vec![4, 1], TensorShapeDynamism::DYNAMIC_BOUND);
        let mut ctx = context();
        glu_out(&mut ctx, &x, 0, &out);
        assert!(tensors_are_close(
            &out,
            &expected_result,
            internal::K_DEFAULT_RTOL,
            None
        ));
    }

    // [spec:et:sem:op-glu.torch.executor.native.glu-out-fn/test]
    #[test]
    fn op_glu_out_test_dynamic_shape_upper_bound_larger_than_expected() {
        let tf = TensorFactory::<f32>::new();

        let x = tf.make_default(
            vec![4, 2],
            vec![
                0.057747602462768555,
                0.8781633377075195,
                0.4503108263015747,
                0.40363800525665283,
                0.3379024863243103,
                0.13906866312026978,
                0.6991606950759888,
                0.4374786615371704,
            ],
        );
        let expected_result = tf.make_default(
            vec![2, 2],
            vec![
                0.0337061733007431,
                0.4695638120174408,
                0.3008083701133728,
                0.2452739030122757,
            ],
        );

        let out = tf.zeros(vec![10, 10], TensorShapeDynamism::DYNAMIC_BOUND);
        let mut ctx = context();
        glu_out(&mut ctx, &x, 0, &out);
        assert!(tensors_are_close(
            &out,
            &expected_result,
            internal::K_DEFAULT_RTOL,
            None
        ));
    }

    // DISABLED: Dynamic shape unbound not supported
    // [spec:et:sem:op-glu.torch.executor.native.glu-out-fn/test]
    #[test]
    #[ignore]
    fn op_glu_out_test_disabled_dynamic_shape_unbound() {
        let tf = TensorFactory::<f32>::new();

        let x = tf.make_default(
            vec![4, 2],
            vec![
                0.057747602462768555,
                0.8781633377075195,
                0.4503108263015747,
                0.40363800525665283,
                0.3379024863243103,
                0.13906866312026978,
                0.6991606950759888,
                0.4374786615371704,
            ],
        );
        let expected_result = tf.make_default(
            vec![2, 2],
            vec![
                0.0337061733007431,
                0.4695638120174408,
                0.3008083701133728,
                0.2452739030122757,
            ],
        );

        let out = tf.zeros(vec![1, 1], TensorShapeDynamism::DYNAMIC_UNBOUND);
        let mut ctx = context();
        glu_out(&mut ctx, &x, 0, &out);
        assert!(tensors_are_close(
            &out,
            &expected_result,
            internal::K_DEFAULT_RTOL,
            None
        ));
    }
}
