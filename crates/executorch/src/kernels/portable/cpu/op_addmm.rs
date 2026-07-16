//! Literal port of kernels/portable/cpu/op_addmm.cpp.

use crate::kernels::portable::cpu::scalar_utils::scalar_to;
use crate::kernels::portable::cpu::util::broadcast_util::tensor_is_broadcastable_to_tensors;
use crate::kernels::portable::cpu::util::dtype_util::SupportedTensorDtypes;
use crate::kernels::portable::cpu::util::elementwise_util::apply_bitensor_elementwise_fn;
use crate::kernels::portable::cpu::util::matmul_ops_util::{
    check_addmm_args, get_mm_out_target_size,
};
use crate::kernels::portable::cpu::vec_ops::{vec_addmm, vec_matmul};
use crate::runtime::core::array_ref::ArrayRef;
use crate::runtime::core::error::Error;
use crate::runtime::core::exec_aten::exec_aten::SizesType;
use crate::runtime::core::exec_aten::util::tensor_util::{
    K_TENSOR_DIMENSION_LIMIT, resize_tensor, tensor_is_default_dim_order,
    tensors_have_same_dim_order4,
};
use crate::runtime::core::portable_type::scalar::Scalar;
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

// PORT-NOTE: C++ computes `val_a * alpha_val + val_b * beta_val` in CTYPE over
// the REALHBF16 set. `c10::Half`/`BFloat16` perform arithmetic by promoting to
// float; the primitive ctypes use their native operators. This local trait
// reproduces that per-type behavior for the elementwise addition path.
trait Rhf16Arith: Copy {
    fn a_add(self, other: Self) -> Self;
    fn a_mul(self, other: Self) -> Self;
}
macro_rules! impl_rhf16_arith_prim {
    ($($t:ty),*) => {$(
        impl Rhf16Arith for $t {
            fn a_add(self, other: Self) -> Self { self + other }
            fn a_mul(self, other: Self) -> Self { self * other }
        }
    )*};
}
impl_rhf16_arith_prim!(u8, i8, i16, i32, i64, f32, f64);
impl Rhf16Arith for crate::runtime::core::portable_type::Half {
    fn a_add(self, other: Self) -> Self {
        crate::runtime::core::portable_type::Half::from_f32(self.to_f32() + other.to_f32())
    }
    fn a_mul(self, other: Self) -> Self {
        crate::runtime::core::portable_type::Half::from_f32(self.to_f32() * other.to_f32())
    }
}
impl Rhf16Arith for crate::runtime::core::portable_type::BFloat16 {
    fn a_add(self, other: Self) -> Self {
        crate::runtime::core::portable_type::BFloat16::from_f32(self.to_f32() + other.to_f32())
    }
    fn a_mul(self, other: Self) -> Self {
        crate::runtime::core::portable_type::BFloat16::from_f32(self.to_f32() * other.to_f32())
    }
}

// [spec:et:def:op-addmm.torch.executor.native.addmm-out-fn]
// [spec:et:sem:op-addmm.torch.executor.native.addmm-out-fn]
#[executorch_macros::et_kernel("aten::addmm.out")]
pub fn addmm_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    in_: &Tensor,
    mat1: &Tensor,
    mat2: &Tensor,
    beta: &Scalar,
    alpha: &Scalar,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    crate::et_kernel_check!(
        ctx,
        check_addmm_args(in_, mat1, mat2, beta, alpha, out),
        InvalidArgument,
        out
    );

    let mut output_ndim: usize = 0;
    let mut output_sizes: [SizesType; K_TENSOR_DIMENSION_LIMIT] = [0; K_TENSOR_DIMENSION_LIMIT];
    unsafe {
        get_mm_out_target_size(mat1, mat2, output_sizes.as_mut_ptr(), &mut output_ndim);
    }
    crate::et_kernel_check!(
        ctx,
        resize_tensor(
            out,
            ArrayRef::from_raw_parts(output_sizes.as_ptr(), output_ndim)
        ) == Error::Ok,
        InvalidArgument,
        out
    );

    crate::et_kernel_check!(
        ctx,
        tensor_is_broadcastable_to_tensors(in_, out),
        InvalidArgument,
        out
    );

    crate::et_kernel_check!(
        ctx,
        tensors_have_same_dim_order4(in_, mat1, mat2, out),
        InvalidArgument,
        out
    );

    crate::et_kernel_check!(ctx, tensor_is_default_dim_order(in_), InvalidArgument, out);

    let op_name = "addmm.out";

    crate::et_switch_realhbf16_types!(in_.scalar_type(), ctx, op_name, CTYPE, {
        let alpha_val: CTYPE = scalar_to::<CTYPE>(alpha);
        let beta_val: CTYPE = scalar_to::<CTYPE>(beta);
        let m = mat1.size(0) as i64;
        let n = mat1.size(1) as i64;
        let p = mat2.size(1) as i64;

        if out.sizes().equals(in_.sizes()) {
            // vec_addmm assumes that no broadcasting is required.
            unsafe {
                vec_addmm::<CTYPE>(
                    out.mutable_data_ptr::<CTYPE>(),
                    in_.const_data_ptr::<CTYPE>(),
                    mat1.const_data_ptr::<CTYPE>(),
                    mat2.const_data_ptr::<CTYPE>(),
                    m,
                    n,
                    p,
                    beta_val,
                    alpha_val,
                );
            }
        } else {
            // If broadcasting is required, them compute the matmul
            // and addition separately, using
            // apply_binary_elementwise_fn to perform the addition
            // while applying broadcasting
            unsafe {
                vec_matmul::<CTYPE>(
                    out.mutable_data_ptr::<CTYPE>(),
                    mat1.const_data_ptr::<CTYPE>(),
                    mat2.const_data_ptr::<CTYPE>(),
                    m,
                    n,
                    p,
                );
            }

            apply_bitensor_elementwise_fn::<CTYPE, _>(
                move |vals: &[CTYPE]| vals[0].a_mul(alpha_val).a_add(vals[1].a_mul(beta_val)),
                ctx,
                out,
                SupportedTensorDtypes::REALHBF16,
                in_,
                SupportedTensorDtypes::REALHBF16,
                out,
                SupportedTensorDtypes::REALHBF16,
                false,
            );
        }
    });

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assert_tensor_close;
    use crate::assert_tensor_eq;
    use crate::runtime::core::exec_aten::testing_util::tensor_factory::{
        FactoryValue, TensorFactory,
    };
    use crate::runtime::core::exec_aten::util::scalar_type_util::CppTypeToScalarType;
    use crate::runtime::core::memory_allocator::{MemoryAllocator, MemoryAllocatorBase};
    use crate::runtime::core::portable_type::Half;
    use crate::runtime::core::tensor_shape_dynamism::TensorShapeDynamism;

    fn context() -> KernelRuntimeContext<'static> {
        // Mirrors `OperatorTest::SetUp()`'s `runtime_init()` PAL initialization.
        crate::runtime::platform::platform::pal_init();
        KernelRuntimeContext::new(
            crate::extension::module::module::null_event_tracer(),
            core::ptr::null_mut::<MemoryAllocator>() as *mut dyn MemoryAllocatorBase,
        )
    }

    trait FromI64: Copy {
        fn from_i64(v: i64) -> Self;
    }
    macro_rules! impl_from_i64_num {
        ($($t:ty),*) => {$(impl FromI64 for $t { fn from_i64(v: i64) -> Self { v as $t } })*};
    }
    impl_from_i64_num!(u8, i8, i16, i32, i64, f32, f64);
    impl FromI64 for Half {
        fn from_i64(v: i64) -> Self {
            Half::from_f32(v as f32)
        }
    }

    fn full<T: CppTypeToScalarType + FactoryValue>(
        tf: &TensorFactory<T>,
        sizes: Vec<i32>,
        value: T,
    ) -> crate::runtime::core::portable_type::tensor::Tensor<'_> {
        tf.full(sizes, value, TensorShapeDynamism::STATIC)
    }

    // PORT-NOTE: `test_dtype<CTYPE, DTYPE>` dispatched via `ET_FORALL_REAL_TYPES_AND(
    // Half, ...)` in `AllDtypesSupported`. In the portable build the ATen-only
    // `ET_SKIP_IF(DTYPE == Half, ...)` guard is inert, so Half runs. Each dtype is a
    // separate helper call.
    fn test_dtype<T>()
    where
        T: CppTypeToScalarType + FactoryValue + FromI64,
    {
        let tf = TensorFactory::<T>::new();

        // matmul gives 4*2*3=24, α*24=48, 48 + β*self = 51
        let self_ = full(&tf, vec![3, 5], T::from_i64(1));
        let x = full(&tf, vec![3, 4], T::from_i64(2));
        let y = full(&tf, vec![4, 5], T::from_i64(3));

        let out = tf.zeros_default(vec![3, 5]);

        let alpha = Scalar::from_double(2.0);
        let beta = Scalar::from_double(3.0);

        let mut ctx = context();
        addmm_out(&mut ctx, &self_, &x, &y, &beta, &alpha, &out);

        let expected = full(&tf, vec![3, 5], T::from_i64(51));

        assert_tensor_eq!(out, expected);
    }

    // [spec:et:sem:op-addmm.torch.executor.native.addmm-out-fn/test]
    #[test]
    fn op_addmm_out_test_output_dim() {
        let tf = TensorFactory::<i32>::new();

        let self_ = tf.ones_default(vec![3, 5]);
        let x = tf.ones_default(vec![3, 4]);
        let y = tf.ones_default(vec![4, 5]);

        let out = tf.zeros_default(vec![3, 5]);

        let alpha = Scalar::from_i64(1);
        let beta = Scalar::from_i64(1);

        let mut ctx = context();
        let ret = addmm_out(&mut ctx, &self_, &x, &y, &beta, &alpha, &out);

        assert_tensor_eq!(*ret, out);

        let expected = full(&tf, vec![3, 5], 5);
        assert_tensor_eq!(out, expected);
    }

    // [spec:et:sem:op-addmm.torch.executor.native.addmm-out-fn/test]
    #[test]
    fn op_addmm_out_test_all_dtypes_supported() {
        test_dtype::<u8>();
        test_dtype::<i8>();
        test_dtype::<i16>();
        test_dtype::<i32>();
        test_dtype::<i64>();
        test_dtype::<f32>();
        test_dtype::<f64>();
        test_dtype::<Half>();
    }

    // [spec:et:sem:op-addmm.torch.executor.native.addmm-out-fn/test]
    #[test]
    fn op_addmm_out_test_empty_input_with_empty_out_tensor_passes() {
        let tf = TensorFactory::<f32>::new();

        let self_ = tf.make_default(vec![0, 0], vec![]);
        let x = tf.make_default(vec![0, 3], vec![]);
        let y = tf.make_default(vec![3, 0], vec![]);

        let out = tf.make_default(vec![0, 0], vec![]);
        let expected = tf.make_default(vec![0, 0], vec![]);

        let mut ctx = context();
        let ret = addmm_out(
            &mut ctx,
            &self_,
            &x,
            &y,
            &Scalar::from_i64(2),
            &Scalar::from_i64(3),
            &out,
        );
        assert_tensor_eq!(*ret, expected);
    }

    // [spec:et:sem:op-addmm.torch.executor.native.addmm-out-fn/test]
    #[test]
    fn op_addmm_out_test_float_tensor_dtype_and_int_scalar_type_passes() {
        let tff = TensorFactory::<f32>::new();
        // matmul gives 24, α*24=72, 72 + β*self = 74
        let self_ = full(&tff, vec![3, 5], 1.0);
        let x = full(&tff, vec![3, 4], 2.0);
        let y = full(&tff, vec![4, 5], 3.0);

        let out = tff.zeros_default(vec![3, 5]);
        let expected = full(&tff, vec![3, 5], 74.0);

        let mut ctx = context();
        let ret = addmm_out(
            &mut ctx,
            &self_,
            &x,
            &y,
            &Scalar::from_i64(2),
            &Scalar::from_i64(3),
            &out,
        );
        assert_tensor_eq!(*ret, expected);
    }

    // [spec:et:sem:op-addmm.torch.executor.native.addmm-out-fn/test]
    #[test]
    fn op_addmm_out_test_int_tensor_dtype_and_float_scalar_type_passes() {
        let tfi = TensorFactory::<i32>::new();
        let self_ = full(&tfi, vec![3, 5], 1);
        let x = full(&tfi, vec![3, 4], 2);
        let y = full(&tfi, vec![4, 5], 3);

        let out = tfi.zeros_default(vec![3, 5]);
        let expected = full(&tfi, vec![3, 5], 74);

        let mut ctx = context();
        let ret = addmm_out(
            &mut ctx,
            &self_,
            &x,
            &y,
            &Scalar::from_double(2.0),
            &Scalar::from_double(3.0),
            &out,
        );
        assert_tensor_eq!(*ret, expected);
    }

    // [spec:et:sem:op-addmm.torch.executor.native.addmm-out-fn/test]
    #[test]
    fn op_addmm_out_test_infinity_tensor_and_float_scalar_type_passes() {
        let tff = TensorFactory::<f32>::new();

        let self_ = full(&tff, vec![3, 5], f32::INFINITY);
        let x = full(&tff, vec![3, 4], 2.0);
        let y = full(&tff, vec![4, 5], 3.0);

        let out = tff.zeros_default(vec![3, 5]);
        let expected = full(&tff, vec![3, 5], f32::INFINITY);

        let mut ctx = context();
        let ret = addmm_out(
            &mut ctx,
            &self_,
            &x,
            &y,
            &Scalar::from_i64(2),
            &Scalar::from_i64(3),
            &out,
        );
        assert_tensor_eq!(*ret, expected);
    }

    // [spec:et:sem:op-addmm.torch.executor.native.addmm-out-fn/test]
    // also verifies check_addmm_args rejects mat2 with mismatched inner dim (mat1=[2,2] vs wrong_y=[3,1])
    // [spec:et:sem:matmul-ops-util.torch.executor.check-addmm-args-fn/test]
    #[test]
    fn op_addmm_out_test_mismatched_dimensions_dies() {
        let tf = TensorFactory::<i32>::new();

        let self_ = full(&tf, vec![2, 2], 3);
        let x = full(&tf, vec![2, 2], 3);

        let wrong_y = full(&tf, vec![3, 1], 1);
        let right_y = full(&tf, vec![2, 2], 1);

        let out = full(&tf, vec![2, 2], 0);
        let expected = full(&tf, vec![2, 2], 9);

        let mut ctx = context();
        addmm_out(
            &mut ctx,
            &self_,
            &x,
            &wrong_y,
            &Scalar::from_i64(1),
            &Scalar::from_i64(1),
            &out,
        );
        assert_ne!(ctx.failure_state(), Error::Ok);

        let ret = addmm_out(
            &mut ctx,
            &self_,
            &x,
            &right_y,
            &Scalar::from_i64(1),
            &Scalar::from_i64(1),
            &out,
        );
        assert_tensor_eq!(*ret, expected);
    }

    // [spec:et:sem:op-addmm.torch.executor.native.addmm-out-fn/test]
    #[test]
    fn op_addmm_out_test_mismatched_dimension_size_dies() {
        let tf = TensorFactory::<i32>::new();
        let self_ = full(&tf, vec![2, 2], 3);
        let x = full(&tf, vec![2, 2], 3);

        let wrong_y = full(&tf, vec![2, 2, 2], 1);
        let right_y = full(&tf, vec![2, 2], 1);

        let right_out = tf.ones_default(vec![2, 2]);
        let wrong_out = tf.ones_default(vec![2, 2, 3]);

        let mut ctx = context();
        addmm_out(
            &mut ctx,
            &self_,
            &x,
            &right_y,
            &Scalar::from_i64(1),
            &Scalar::from_i64(1),
            &wrong_out,
        );
        assert_ne!(ctx.failure_state(), Error::Ok);

        let mut ctx = context();
        addmm_out(
            &mut ctx,
            &self_,
            &x,
            &wrong_y,
            &Scalar::from_i64(1),
            &Scalar::from_i64(1),
            &right_out,
        );
        assert_ne!(ctx.failure_state(), Error::Ok);
    }

    // [spec:et:sem:op-addmm.torch.executor.native.addmm-out-fn/test]
    #[test]
    fn op_addmm_out_test_wrong_out_shape_dies() {
        let tf = TensorFactory::<i32>::new();
        let self_ = tf.ones_default(vec![10, 4]);
        let x = tf.ones_default(vec![10, 3]);
        let y = tf.ones_default(vec![3, 4]);

        let right_out = tf.ones_default(vec![10, 4]);
        let wrong_out = tf.ones_default(vec![7, 5]);

        let mut ctx = context();
        addmm_out(
            &mut ctx,
            &self_,
            &x,
            &y,
            &Scalar::from_i64(1),
            &Scalar::from_i64(1),
            &wrong_out,
        );
        assert_ne!(ctx.failure_state(), Error::Ok);

        let ret = addmm_out(
            &mut ctx,
            &self_,
            &x,
            &y,
            &Scalar::from_i64(1),
            &Scalar::from_i64(1),
            &right_out,
        );
        assert_tensor_eq!(*ret, full(&tf, vec![10, 4], 4));
    }

    // [spec:et:sem:op-addmm.torch.executor.native.addmm-out-fn/test]
    #[test]
    fn op_addmm_out_test_broadcast_test() {
        let tf = TensorFactory::<i32>::new();

        let self_ = tf.make_default(vec![1], vec![1]);
        let x = tf.make_default(vec![2, 2], vec![1, 2, 3, 4]);
        let y = tf.make_default(vec![2, 2], vec![1, 2, 3, 4]);

        let out = tf.make_default(vec![2, 2], vec![0, 0, 0, 0]);

        let mut ctx = context();
        let ret = addmm_out(
            &mut ctx,
            &self_,
            &x,
            &y,
            &Scalar::from_i64(1),
            &Scalar::from_i64(1),
            &out,
        );
        assert_tensor_eq!(*ret, tf.make_default(vec![2, 2], vec![8, 11, 16, 23]));
    }

    // [spec:et:sem:op-addmm.torch.executor.native.addmm-out-fn/test]
    #[test]
    fn op_addmm_out_test_broadcast_dim_size1() {
        let tf = TensorFactory::<f32>::new();

        let x = tf.make_default(vec![1, 2], vec![0.9937992691993713, 0.7011417150497437]);
        let y = tf.make_default(
            vec![3, 6],
            vec![
                0.3271445035934448,
                0.4104803800582886,
                0.26973772048950195,
                0.29142987728118896,
                0.20096111297607422,
                0.7686975002288818,
                0.07416731119155884,
                0.276896595954895,
                0.43525755405426025,
                0.8261672854423523,
                0.22888076305389404,
                0.042113542556762695,
                0.8771350979804993,
                0.4088439345359802,
                0.0258103609085083,
                0.26305103302001953,
                0.6766068339347839,
                0.3576545715332031,
            ],
        );
        let z = tf.make_default(
            vec![6, 2],
            vec![
                0.5702318549156189,
                0.8886868953704834,
                0.8667161464691162,
                0.7151150107383728,
                0.19591552019119263,
                0.7918031811714172,
                0.8956874012947083,
                0.7162176966667175,
                0.34151601791381836,
                0.16078311204910278,
                0.6722156405448914,
                0.048251569271087646,
            ],
        );
        let expected_result = tf.make_default(
            vec![3, 2],
            vec![
                2.4353551864624023,
                1.7771198749542236,
                2.207819700241089,
                1.9402521848678589,
                2.5604825019836426,
                2.107893466949463,
            ],
        );

        let out = tf.zeros_default(vec![3, 2]);
        let mut ctx = context();
        addmm_out(
            &mut ctx,
            &x,
            &y,
            &z,
            &Scalar::from_i64(1),
            &Scalar::from_i64(1),
            &out,
        );
        assert_tensor_close!(out, expected_result);
    }

    // [spec:et:sem:op-addmm.torch.executor.native.addmm-out-fn/test]
    #[test]
    fn op_addmm_out_test_broadcast_dim_size_missing() {
        let tf = TensorFactory::<f32>::new();

        let x = tf.make_default(vec![2], vec![0.9937992691993713, 0.7011417150497437]);
        let y = tf.make_default(
            vec![3, 6],
            vec![
                0.3271445035934448,
                0.4104803800582886,
                0.26973772048950195,
                0.29142987728118896,
                0.20096111297607422,
                0.7686975002288818,
                0.07416731119155884,
                0.276896595954895,
                0.43525755405426025,
                0.8261672854423523,
                0.22888076305389404,
                0.042113542556762695,
                0.8771350979804993,
                0.4088439345359802,
                0.0258103609085083,
                0.26305103302001953,
                0.6766068339347839,
                0.3576545715332031,
            ],
        );
        let z = tf.make_default(
            vec![6, 2],
            vec![
                0.5702318549156189,
                0.8886868953704834,
                0.8667161464691162,
                0.7151150107383728,
                0.19591552019119263,
                0.7918031811714172,
                0.8956874012947083,
                0.7162176966667175,
                0.34151601791381836,
                0.16078311204910278,
                0.6722156405448914,
                0.048251569271087646,
            ],
        );
        let expected_result = tf.make_default(
            vec![3, 2],
            vec![
                2.4353551864624023,
                1.7771198749542236,
                2.207819700241089,
                1.9402521848678589,
                2.5604825019836426,
                2.107893466949463,
            ],
        );

        let out = tf.zeros_default(vec![3, 2]);
        let mut ctx = context();
        addmm_out(
            &mut ctx,
            &x,
            &y,
            &z,
            &Scalar::from_i64(1),
            &Scalar::from_i64(1),
            &out,
        );
        assert_tensor_close!(out, expected_result);
    }

    // [spec:et:sem:op-addmm.torch.executor.native.addmm-out-fn/test]
    #[test]
    fn op_addmm_out_test_broadcast_dim_size_is_one() {
        let tf = TensorFactory::<f32>::new();

        let x = tf.make_default(vec![1, 2], vec![0.9093303680419922, 0.37621551752090454]);
        let y = tf.make_default(
            vec![3, 6],
            vec![
                0.5741164088249207,
                0.3001101613044739,
                0.6543494462966919,
                0.8815506100654602,
                0.8948686122894287,
                0.3319156765937805,
                0.6683467030525208,
                0.37235790491104126,
                0.15439540147781372,
                0.05733710527420044,
                0.5467379093170166,
                0.9564069509506226,
                0.2915573716163635,
                0.5548340082168579,
                0.20116734504699707,
                0.8199875950813293,
                0.270835816860199,
                0.1414813995361328,
            ],
        );
        let z = tf.make_default(
            vec![6, 2],
            vec![
                0.6883938312530518,
                0.9387704133987427,
                0.6991894841194153,
                0.2945629954338074,
                0.48106586933135986,
                0.932110607624054,
                0.9461215138435364,
                0.7682468295097351,
                0.6223915219306946,
                0.0702824592590332,
                0.9750580787658691,
                0.05068659782409668,
            ],
        );
        let expected_result = tf.make_default(
            vec![3, 2],
            vec![
                3.5438172817230225,
                2.3704721927642822,
                3.0311243534088135,
                1.388188123703003,
                2.6770718097686768,
                1.6570236682891846,
            ],
        );

        let out = tf.zeros_default(vec![3, 2]);
        let mut ctx = context();
        addmm_out(
            &mut ctx,
            &x,
            &y,
            &z,
            &Scalar::from_i64(1),
            &Scalar::from_i64(1),
            &out,
        );
        assert_tensor_close!(out, expected_result);
    }

    // [spec:et:sem:op-addmm.torch.executor.native.addmm-out-fn/test]
    #[test]
    fn op_addmm_out_test_dynamic_shape_upper_bound_same_as_expected() {
        let tf = TensorFactory::<f32>::new();

        let x = tf.make_default(
            vec![3, 2],
            vec![
                0.5024666786193848,
                0.8311734795570374,
                0.17922323942184448,
                0.5711425542831421,
                0.23492926359176636,
                0.6693081259727478,
            ],
        );
        let y = tf.make_default(
            vec![3, 6],
            vec![
                0.8927820920944214,
                0.13490021228790283,
                0.49518370628356934,
                0.027777791023254395,
                0.7909245491027832,
                0.07999932765960693,
                0.9496669173240662,
                0.18807870149612427,
                0.44375330209732056,
                0.761903703212738,
                0.24175149202346802,
                0.31033122539520264,
                0.8609206080436707,
                0.1580638885498047,
                0.2585788369178772,
                0.4787442088127136,
                0.17180007696151733,
                0.2109091877937317,
            ],
        );
        let z = tf.make_default(
            vec![6, 2],
            vec![
                0.06361657381057739,
                0.8065286874771118,
                0.610871434211731,
                0.19808048009872437,
                0.7010428309440613,
                0.904334545135498,
                0.8460395932197571,
                0.34137529134750366,
                0.4836529493331909,
                0.2751874327659607,
                0.22036516666412354,
                0.742312490940094,
            ],
        );
        let expected_result = tf.make_default(
            vec![3, 2],
            vec![
                1.4124772548675537,
                2.3122801780700684,
                1.495530605316162,
                2.3326172828674316,
                1.1021348237991333,
                1.9960856437683105,
            ],
        );

        let out = tf.zeros(vec![3, 2], TensorShapeDynamism::DYNAMIC_BOUND);
        let mut ctx = context();
        addmm_out(
            &mut ctx,
            &x,
            &y,
            &z,
            &Scalar::from_i64(1),
            &Scalar::from_i64(1),
            &out,
        );
        assert_tensor_close!(out, expected_result);
    }

    // [spec:et:sem:op-addmm.torch.executor.native.addmm-out-fn/test]
    #[test]
    fn op_addmm_out_test_dynamic_shape_upper_bound_larger_than_expected() {
        let tf = TensorFactory::<f32>::new();

        let x = tf.make_default(
            vec![3, 2],
            vec![
                0.5024666786193848,
                0.8311734795570374,
                0.17922323942184448,
                0.5711425542831421,
                0.23492926359176636,
                0.6693081259727478,
            ],
        );
        let y = tf.make_default(
            vec![3, 6],
            vec![
                0.8927820920944214,
                0.13490021228790283,
                0.49518370628356934,
                0.027777791023254395,
                0.7909245491027832,
                0.07999932765960693,
                0.9496669173240662,
                0.18807870149612427,
                0.44375330209732056,
                0.761903703212738,
                0.24175149202346802,
                0.31033122539520264,
                0.8609206080436707,
                0.1580638885498047,
                0.2585788369178772,
                0.4787442088127136,
                0.17180007696151733,
                0.2109091877937317,
            ],
        );
        let z = tf.make_default(
            vec![6, 2],
            vec![
                0.06361657381057739,
                0.8065286874771118,
                0.610871434211731,
                0.19808048009872437,
                0.7010428309440613,
                0.904334545135498,
                0.8460395932197571,
                0.34137529134750366,
                0.4836529493331909,
                0.2751874327659607,
                0.22036516666412354,
                0.742312490940094,
            ],
        );
        let expected_result = tf.make_default(
            vec![3, 2],
            vec![
                1.4124772548675537,
                2.3122801780700684,
                1.495530605316162,
                2.3326172828674316,
                1.1021348237991333,
                1.9960856437683105,
            ],
        );

        let out = tf.zeros(vec![10, 10], TensorShapeDynamism::DYNAMIC_BOUND);
        let mut ctx = context();
        addmm_out(
            &mut ctx,
            &x,
            &y,
            &z,
            &Scalar::from_i64(1),
            &Scalar::from_i64(1),
            &out,
        );
        assert_tensor_close!(out, expected_result);
    }

    // PORT-NOTE: `DISABLED_DynamicShapeUnbound` is disabled in the C++ (gtest
    // `DISABLED_` prefix). Ported and `#[ignore]`d to preserve it without running.
    // [spec:et:sem:op-addmm.torch.executor.native.addmm-out-fn/test]
    #[test]
    #[ignore = "DISABLED in C++: Dynamic shape unbound not supported"]
    fn op_addmm_out_test_dynamic_shape_unbound() {
        let tf = TensorFactory::<f32>::new();

        let x = tf.make_default(
            vec![3, 2],
            vec![
                0.754013180732727,
                0.16418755054473877,
                0.8077310919761658,
                0.7187556624412537,
                0.0470539927482605,
                0.2438456416130066,
            ],
        );
        let y = tf.make_default(
            vec![3, 6],
            vec![
                0.5899912118911743,
                0.5052928328514099,
                0.13990312814712524,
                0.22438400983810425,
                0.1697748899459839,
                0.6022286415100098,
                0.08701932430267334,
                0.7246091961860657,
                0.44388288259506226,
                0.9451560974121094,
                0.8658323884010315,
                0.781434953212738,
                0.02855396270751953,
                0.49756181240081787,
                0.506054699420929,
                0.12560266256332397,
                0.7099084854125977,
                0.04813879728317261,
            ],
        );
        let z = tf.make_default(
            vec![6, 2],
            vec![
                0.19827371835708618,
                0.486919641494751,
                0.7659645080566406,
                0.7863746285438538,
                0.032599568367004395,
                0.8414170145988464,
                0.7014893293380737,
                0.2445545196533203,
                0.07429623603820801,
                0.12777382135391235,
                0.39169949293136597,
                0.80079185962677,
            ],
        );
        let expected_result = tf.make_default(
            vec![3, 2],
            vec![
                1.6684993505477905,
                1.5253589153289795,
                2.427912712097168,
                2.6719717979431152,
                0.6100357174873352,
                1.2347958087921143,
            ],
        );

        let out = tf.zeros(vec![1, 1], TensorShapeDynamism::DYNAMIC_UNBOUND);
        let mut ctx = context();
        addmm_out(
            &mut ctx,
            &x,
            &y,
            &z,
            &Scalar::from_i64(1),
            &Scalar::from_i64(1),
            &out,
        );
        assert_tensor_close!(out, expected_result);
    }
}
