//! Literal port of kernels/portable/cpu/op_scatter.cpp.

use crate::kernels::portable::cpu::scalar_utils::internal::check_overflow_scalar_cast;
use crate::kernels::portable::cpu::util::dtype_util::internal::StaticCast;
use crate::kernels::portable::cpu::util::index_util::{
    check_scatter_src_args, check_scatter_value_args,
};
use crate::runtime::core::error::Error;
use crate::runtime::core::exec_aten::util::tensor_util::{
    K_TENSOR_DIMENSION_LIMIT, coordinateToIndex, indexToCoordinate, nonzero_dim, resize_tensor,
};
use crate::runtime::core::portable_type::scalar::Scalar;
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

// PORT-NOTE: the two C++ private-namespace helpers are templates
// (`scatter_src_helper<CTYPE>`, `scatter_value_helper<CTYPE, CTYPE_VAL>`)
// instantiated per-ctype by the switch in the out fns. Rust generic functions
// reproduce those instantiations. The void-typed `memcpy(out_data, in_data,
// in.nbytes())` is a byte-wise `copy_nonoverlapping` over the `CTYPE*` buffers.

// [spec:et:def:op-scatter.torch.executor.native.scatter-src-helper-fn]
// [spec:et:sem:op-scatter.torch.executor.native.scatter-src-helper-fn]
fn scatter_src_helper<CTYPE: Copy>(
    in_: &Tensor,
    mut dim: i64,
    index: &Tensor,
    src: &Tensor,
    out: &Tensor,
) {
    let in_data: *const CTYPE = in_.const_data_ptr::<CTYPE>();
    let index_data: *const i64 = index.const_data_ptr::<i64>();
    let src_data: *const CTYPE = src.const_data_ptr::<CTYPE>();
    let out_data: *mut CTYPE = out.mutable_data_ptr::<CTYPE>();

    unsafe {
        core::ptr::copy_nonoverlapping(in_data as *const u8, out_data as *mut u8, in_.nbytes());
    }

    if dim < 0 {
        dim += nonzero_dim(in_) as i64;
    }

    for ix in 0..index.numel() {
        let mut ix_coord: [usize; K_TENSOR_DIMENSION_LIMIT] = [0; K_TENSOR_DIMENSION_LIMIT];
        unsafe {
            indexToCoordinate(index, ix as usize, ix_coord.as_mut_ptr());
        }

        let src_ix: usize = unsafe { coordinateToIndex(src, ix_coord.as_ptr()) };

        let mut out_coord: [usize; K_TENSOR_DIMENSION_LIMIT] = [0; K_TENSOR_DIMENSION_LIMIT];
        for i in 0..out.dim() {
            if i == dim as isize {
                out_coord[i as usize] = unsafe { *index_data.add(ix as usize) } as usize;
            } else {
                out_coord[i as usize] = ix_coord[i as usize];
            }
        }
        let out_ix: usize = unsafe { coordinateToIndex(out, out_coord.as_ptr()) };

        unsafe {
            *out_data.add(out_ix) = *src_data.add(src_ix);
        }
    }
}

// [spec:et:def:op-scatter.torch.executor.native.scatter-value-helper-fn]
// [spec:et:sem:op-scatter.torch.executor.native.scatter-value-helper-fn]
fn scatter_value_helper<CTYPE, CTYPE_VAL>(
    in_: &Tensor,
    mut dim: i64,
    index: &Tensor,
    val: CTYPE_VAL,
    out: &Tensor,
) where
    CTYPE: Copy + StaticCast<CTYPE_VAL>,
    CTYPE_VAL: Copy,
{
    let in_data: *const CTYPE = in_.const_data_ptr::<CTYPE>();
    let index_data: *const i64 = index.const_data_ptr::<i64>();
    let out_data: *mut CTYPE = out.mutable_data_ptr::<CTYPE>();

    unsafe {
        core::ptr::copy_nonoverlapping(in_data as *const u8, out_data as *mut u8, in_.nbytes());
    }

    if dim < 0 {
        dim += nonzero_dim(in_) as i64;
    }

    for ix in 0..index.numel() {
        let mut ix_coord: [usize; K_TENSOR_DIMENSION_LIMIT] = [0; K_TENSOR_DIMENSION_LIMIT];
        unsafe {
            indexToCoordinate(index, ix as usize, ix_coord.as_mut_ptr());
        }

        let mut out_coord: [usize; K_TENSOR_DIMENSION_LIMIT] = [0; K_TENSOR_DIMENSION_LIMIT];
        for i in 0..out.dim() {
            if i == dim as isize {
                out_coord[i as usize] = unsafe { *index_data.add(ix as usize) } as usize;
            } else {
                out_coord[i as usize] = ix_coord[i as usize];
            }
        }
        let out_ix: usize = unsafe { coordinateToIndex(out, out_coord.as_ptr()) };

        unsafe {
            *out_data.add(out_ix) = <CTYPE as StaticCast<CTYPE_VAL>>::static_cast(val);
        }
    }
}

// [spec:et:def:op-scatter.torch.executor.native.scatter-src-out-fn]
// [spec:et:sem:op-scatter.torch.executor.native.scatter-src-out-fn]
pub fn scatter_src_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    in_: &Tensor,
    dim: i64,
    index: &Tensor,
    src: &Tensor,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    crate::et_kernel_check!(
        ctx,
        check_scatter_src_args(in_, dim, index, src, out),
        InvalidArgument,
        out
    );

    crate::et_kernel_check!(
        ctx,
        resize_tensor(out, in_.sizes()) == Error::Ok,
        InvalidArgument,
        out
    );

    let name = "scatter.src_out";

    crate::et_switch_realhbbf16_types!(in_.scalar_type(), ctx, name, CTYPE, {
        scatter_src_helper::<CTYPE>(in_, dim, index, src, out);
    });

    out
}

// [spec:et:def:op-scatter.torch.executor.native.scatter-value-out-fn]
// [spec:et:sem:op-scatter.torch.executor.native.scatter-value-out-fn]
pub fn scatter_value_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    in_: &Tensor,
    dim: i64,
    index: &Tensor,
    value: &Scalar,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    // (void)ctx;

    crate::et_kernel_check!(
        ctx,
        check_scatter_value_args(in_, dim, index, value, out),
        InvalidArgument,
        out
    );

    crate::et_kernel_check!(
        ctx,
        resize_tensor(out, in_.sizes()) == Error::Ok,
        InvalidArgument,
        out
    );

    let name = "scatter.value_out";

    crate::et_switch_realhbbf16_types!(in_.scalar_type(), ctx, name, CTYPE, {
        let opt_val = check_overflow_scalar_cast::<CTYPE>(value);
        crate::et_kernel_check!(ctx, opt_val.is_some(), InvalidArgument, out);
        let val: CTYPE = opt_val.unwrap();
        scatter_value_helper::<CTYPE, CTYPE>(in_, dim, index, val, out);
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
    use crate::runtime::core::portable_type::{BFloat16, Half};
    use crate::runtime::core::tensor_shape_dynamism::TensorShapeDynamism;

    fn context() -> KernelRuntimeContext<'static> {
        crate::runtime::platform::runtime::runtime_init();
        KernelRuntimeContext::new(
            crate::extension::module::module::null_event_tracer(),
            core::ptr::null_mut::<MemoryAllocator>() as *mut dyn MemoryAllocatorBase,
        )
    }

    fn op_scatter_src_out<'a, 'b>(
        self_: &Tensor,
        dim: i64,
        index: &Tensor,
        src: &Tensor,
        out: &'a Tensor<'b>,
    ) -> &'a Tensor<'b> {
        let mut ctx = context();
        scatter_src_out(&mut ctx, self_, dim, index, src, out)
    }

    fn op_scatter_value_out<'a, 'b>(
        self_: &Tensor,
        dim: i64,
        index: &Tensor,
        value: &Scalar,
        out: &'a Tensor<'b>,
    ) -> &'a Tensor<'b> {
        let mut ctx = context();
        scatter_value_out(&mut ctx, self_, dim, index, value, out)
    }

    trait FromF64Elem: Copy {
        fn from_f64(v: f64) -> Self;
    }
    macro_rules! impl_from_f64_num {
        ($($t:ty),*) => {$(impl FromF64Elem for $t { fn from_f64(v: f64) -> Self { v as $t } })*};
    }
    impl_from_f64_num!(u8, i8, i16, i32, i64, f32, f64);
    impl FromF64Elem for Half {
        fn from_f64(v: f64) -> Self {
            Half::from_f32(v as f32)
        }
    }
    impl FromF64Elem for BFloat16 {
        fn from_f64(v: f64) -> Self {
            BFloat16::from_f32(v as f32)
        }
    }

    fn v<T: FromF64Elem>(vals: &[f64]) -> Vec<T> {
        vals.iter().map(|&x| T::from_f64(x)).collect()
    }

    // test_scatter_src_out<DATA_DTYPE>
    fn test_scatter_src_out<T>()
    where
        T: CppTypeToScalarType + FactoryValue + FromF64Elem,
    {
        let tf_index = TensorFactory::<i64>::new();
        let tf_data = TensorFactory::<T>::new();
        let sizes = vec![3, 5];

        let mut src = tf_data.make_default(
            vec![2, 5],
            v::<T>(&[1., 2., 3., 4., 5., 6., 7., 8., 9., 10.]),
        );
        let in_ = tf_data.zeros_default(sizes.clone());
        let mut out = tf_data.zeros_default(sizes.clone());
        let mut index = tf_index.make_default(vec![2, 3], vec![0, 1, 2, 0, 1, 2]);

        op_scatter_src_out(&in_, 0, &index, &src, &out);
        assert_tensor_eq!(
            out,
            tf_data.make_default(
                sizes.clone(),
                v::<T>(&[6., 0., 0., 0., 0., 0., 7., 0., 0., 0., 0., 0., 8., 0., 0.]),
            )
        );

        op_scatter_src_out(&in_, 1, &index, &src, &out);
        assert_tensor_eq!(
            out,
            tf_data.make_default(
                sizes,
                v::<T>(&[1., 2., 3., 0., 0., 6., 7., 8., 0., 0., 0., 0., 0., 0., 0.]),
            )
        );

        src = tf_data.make_default(
            vec![2, 3, 3],
            v::<T>(&[
                1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18.,
            ]),
        );
        let in_ = tf_data.ones_default(vec![2, 3, 3]);
        out = tf_data.zeros_default(vec![2, 3, 3]);
        index = tf_index.make_default(vec![1, 3, 2], vec![0, 1, 1, 2, 0, 2]);

        op_scatter_src_out(&in_, 1, &index, &src, &out);
        assert_tensor_eq!(
            out,
            tf_data.make_default(
                vec![2, 3, 3],
                v::<T>(&[
                    7., 1., 1., 4., 2., 1., 1., 8., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                ]),
            )
        );

        out = tf_data.zeros_default(vec![2, 3, 3]);
        op_scatter_src_out(&in_, 2, &index, &src, &out);
        assert_tensor_eq!(
            out,
            tf_data.make_default(
                vec![2, 3, 3],
                v::<T>(&[
                    1., 2., 1., 1., 4., 5., 7., 1., 8., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                ]),
            )
        );
    }

    // test_scatter_src_out_invalid_dim<DATA_DTYPE>
    fn test_scatter_src_out_invalid_dim<T>()
    where
        T: CppTypeToScalarType + FactoryValue + FromF64Elem,
    {
        let tf_index = TensorFactory::<i64>::new();
        let tf_data = TensorFactory::<T>::new();
        let sizes = vec![3, 5];

        let mut src = tf_data.make_default(
            vec![2, 5],
            v::<T>(&[1., 2., 3., 4., 5., 6., 7., 8., 9., 10.]),
        );
        let mut index = tf_index.make_default(vec![2, 3], vec![0, 1, 2, 0, 1, 2]);
        let self_ = tf_data.zeros_default(sizes.clone());
        let out = tf_data.zeros_default(sizes);

        let mut ctx = context();
        scatter_src_out(&mut ctx, &self_, -3, &index, &src, &out);
        assert_ne!(ctx.failure_state(), Error::Ok);
        let mut ctx = context();
        scatter_src_out(&mut ctx, &self_, 2, &index, &src, &out);
        assert_ne!(ctx.failure_state(), Error::Ok);

        src = tf_data.zeros_default(vec![2, 2, 2]);
        let mut ctx = context();
        scatter_src_out(&mut ctx, &self_, 0, &index, &src, &out);
        assert_ne!(ctx.failure_state(), Error::Ok);

        src = tf_data.zeros_default(vec![5, 5]);
        index = tf_index.zeros_default(vec![2, 2, 2]);
        let mut ctx = context();
        scatter_src_out(&mut ctx, &self_, 0, &index, &src, &out);
        assert_ne!(ctx.failure_state(), Error::Ok);

        index = tf_index.zeros_default(vec![4, 6]);
        let mut ctx = context();
        scatter_src_out(&mut ctx, &self_, 0, &index, &src, &out);
        assert_ne!(ctx.failure_state(), Error::Ok);

        index = tf_index.zeros_default(vec![4, 5]);
        let mut ctx = context();
        scatter_src_out(&mut ctx, &self_, 1, &index, &src, &out);
        assert_ne!(ctx.failure_state(), Error::Ok);

        index = tf_index.make_default(vec![2, 3], vec![0, 1, 3, 0, 1, 3]);
        let mut ctx = context();
        scatter_src_out(&mut ctx, &self_, 0, &index, &src, &out);
        assert_ne!(ctx.failure_state(), Error::Ok);
    }

    // test_scatter_value_out<DATA_DTYPE>
    fn test_scatter_value_out<T>()
    where
        T: CppTypeToScalarType + FactoryValue + FromF64Elem,
    {
        let tf_index = TensorFactory::<i64>::new();
        let tf_data = TensorFactory::<T>::new();

        let value = Scalar::from_i64(1);

        let sizes = vec![3, 5];
        let self_ = tf_data.zeros_default(sizes.clone());
        let mut out = tf_data.zeros_default(sizes.clone());
        let mut index = tf_index.make_default(vec![2, 3], vec![0, 1, 2, 0, 1, 2]);

        op_scatter_value_out(&self_, 0, &index, &value, &out);
        assert_tensor_eq!(
            out,
            tf_data.make_default(
                sizes.clone(),
                v::<T>(&[1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0.]),
            )
        );

        op_scatter_value_out(&self_, 1, &index, &value, &out);
        assert_tensor_eq!(
            out,
            tf_data.make_default(
                sizes,
                v::<T>(&[1., 1., 1., 0., 0., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0.]),
            )
        );

        let value2 = Scalar::from_i64(2);
        let self_ = tf_data.ones_default(vec![2, 3, 3]);
        out = tf_data.zeros_default(vec![2, 3, 3]);
        index = tf_index.make_default(vec![1, 3, 2], vec![0, 1, 1, 2, 0, 2]);

        op_scatter_value_out(&self_, 1, &index, &value2, &out);
        assert_tensor_eq!(
            out,
            tf_data.make_default(
                vec![2, 3, 3],
                v::<T>(&[
                    2., 1., 1., 2., 2., 1., 1., 2., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                ]),
            )
        );

        out = tf_data.zeros_default(vec![2, 3, 3]);
        op_scatter_value_out(&self_, 2, &index, &value2, &out);
        assert_tensor_eq!(
            out,
            tf_data.make_default(
                vec![2, 3, 3],
                v::<T>(&[
                    2., 2., 1., 1., 2., 2., 2., 1., 2., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                ]),
            )
        );
    }

    // test_scatter_value_out_invalid_dim<DATA_DTYPE>
    fn test_scatter_value_out_invalid_dim<T>()
    where
        T: CppTypeToScalarType + FactoryValue + FromF64Elem,
    {
        let tf_index = TensorFactory::<i64>::new();
        let tf_data = TensorFactory::<T>::new();

        let self_ = tf_data.make_default(
            vec![2, 5],
            v::<T>(&[1., 2., 3., 4., 5., 6., 7., 8., 9., 10.]),
        );
        let sizes = vec![2, 3];
        let mut index = tf_index.make_default(sizes.clone(), vec![0, 1, 0, 1, 0, 1]);
        let value = Scalar::from_i64(1);
        let out = tf_data.zeros_default(sizes);

        let mut ctx = context();
        scatter_value_out(&mut ctx, &self_, -3, &index, &value, &out);
        assert_ne!(ctx.failure_state(), Error::Ok);
        let mut ctx = context();
        scatter_value_out(&mut ctx, &self_, 2, &index, &value, &out);
        assert_ne!(ctx.failure_state(), Error::Ok);

        index = tf_index.zeros_default(vec![2, 2, 2]);
        let mut ctx = context();
        scatter_value_out(&mut ctx, &self_, 0, &index, &value, &out);
        assert_ne!(ctx.failure_state(), Error::Ok);

        index = tf_index.zeros_default(vec![3, 5]);
        let mut ctx = context();
        scatter_value_out(&mut ctx, &self_, 1, &index, &value, &out);
        assert_ne!(ctx.failure_state(), Error::Ok);

        index = tf_index.make_default(vec![2, 3], vec![0, 1, 2, 0, 1, 2]);
        let mut ctx = context();
        scatter_value_out(&mut ctx, &self_, 0, &index, &value, &out);
        assert_ne!(ctx.failure_state(), Error::Ok);
    }

    fn test_dynamic_shape(out_shape: Vec<i32>, dynamism: TensorShapeDynamism) {
        let tf = TensorFactory::<i32>::new();
        let tf_index = TensorFactory::<i64>::new();

        let input = tf.ones_default(vec![2, 3, 4]);
        let index = tf_index.zeros_default(vec![2, 3, 4]);
        let value = Scalar::from_i64(1);
        let expected = tf.ones_default(vec![2, 3, 4]);
        let out = tf.zeros(out_shape, dynamism);

        op_scatter_value_out(&input, 2, &index, &value, &out);
        assert_tensor_eq!(out, expected);
    }

    fn expect_bad_scalar_value_dies<T>(bad_value: Scalar)
    where
        T: CppTypeToScalarType + FactoryValue,
    {
        let tf = TensorFactory::<T>::new();
        let tf_index = TensorFactory::<i64>::new();

        let self_ = tf.ones_default(vec![2, 2]);
        let index = tf_index.zeros_default(vec![2, 2]);
        let out = tf.zeros_default(vec![2, 2]);

        let mut ctx = context();
        scatter_value_out(&mut ctx, &self_, 0, &index, &bad_value, &out);
        assert_ne!(ctx.failure_state(), Error::Ok);
    }

    // OpScatterSrcOutTest.AllValidInputOutputSupport
    // [spec:et:sem:op-scatter.torch.executor.native.scatter-src-out-fn/test]
    // also verifies check_scatter_src_args (delegates to check_scatter_add_args);
    // the _invalid_dim dies test exercises its failure branches.
    // [spec:et:sem:index-util.torch.executor.check-scatter-src-args-fn/test]
    // also verifies scatter_src_helper: the asserted scattered outputs across dims 0/1/2
    // are produced element-by-element by this helper's index-driven copy loop.
    // [spec:et:sem:op-scatter.torch.executor.native.scatter-src-helper-fn/test]
    #[test]
    fn op_scatter_src_out_test_all_valid_input_output_support() {
        // ET_FORALL_REALHBF16_TYPES
        test_scatter_src_out::<u8>();
        test_scatter_src_out::<i8>();
        test_scatter_src_out::<i16>();
        test_scatter_src_out::<i32>();
        test_scatter_src_out::<i64>();
        test_scatter_src_out::<f32>();
        test_scatter_src_out::<f64>();
        test_scatter_src_out::<Half>();
        test_scatter_src_out::<BFloat16>();
    }

    // OpScatterSrcOutTest.InvalidDimensionsDies
    // [spec:et:sem:op-scatter.torch.executor.native.scatter-src-out-fn/test]
    #[test]
    fn op_scatter_src_out_test_invalid_dimensions_dies() {
        // ET_FORALL_REAL_TYPES
        test_scatter_src_out_invalid_dim::<u8>();
        test_scatter_src_out_invalid_dim::<i8>();
        test_scatter_src_out_invalid_dim::<i16>();
        test_scatter_src_out_invalid_dim::<i32>();
        test_scatter_src_out_invalid_dim::<i64>();
        test_scatter_src_out_invalid_dim::<f32>();
        test_scatter_src_out_invalid_dim::<f64>();
    }

    // OpScatterValueOutTest.AllValidInputOutputSupport
    // [spec:et:sem:op-scatter.torch.executor.native.scatter-value-out-fn/test]
    // also verifies check_scatter_value_args (delegates to check_gather_args);
    // the _invalid_dim dies test exercises its failure branches.
    // [spec:et:sem:index-util.torch.executor.check-scatter-value-args-fn/test]
    // also verifies scatter_value_helper: the asserted scattered outputs across dims 0/1/2
    // are produced element-by-element by this helper's index-driven scalar-write loop.
    // [spec:et:sem:op-scatter.torch.executor.native.scatter-value-helper-fn/test]
    #[test]
    fn op_scatter_value_out_test_all_valid_input_output_support() {
        // ET_FORALL_REALHBF16_TYPES
        test_scatter_value_out::<u8>();
        test_scatter_value_out::<i8>();
        test_scatter_value_out::<i16>();
        test_scatter_value_out::<i32>();
        test_scatter_value_out::<i64>();
        test_scatter_value_out::<f32>();
        test_scatter_value_out::<f64>();
        test_scatter_value_out::<Half>();
        test_scatter_value_out::<BFloat16>();
    }

    // OpScatterValueOutTest.InfinityAndNANTest
    // [spec:et:sem:op-scatter.torch.executor.native.scatter-value-out-fn/test]
    #[test]
    fn op_scatter_value_out_test_infinity_and_nan_test() {
        let tf_index = TensorFactory::<i64>::new();
        let tf_data = TensorFactory::<f32>::new();

        let self_ = tf_data.make_default(
            vec![2, 5],
            vec![
                0.0,
                f32::NEG_INFINITY,
                f32::NAN,
                2.33,
                f32::NAN,
                f32::NAN,
                f32::INFINITY,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                2.33,
            ],
        );
        let index = tf_index.make_default(vec![2, 3], vec![0, 1, 0, 1, 0, 1]);
        let value = Scalar::from_double(f64::INFINITY);
        let out = tf_data.zeros_default(vec![2, 5]);

        op_scatter_value_out(&self_, 0, &index, &value, &out);
        assert_tensor_close!(
            out,
            tf_data.make_default(
                vec![2, 5],
                vec![
                    f32::INFINITY,
                    f32::INFINITY,
                    f32::INFINITY,
                    2.33,
                    f32::NAN,
                    f32::INFINITY,
                    f32::INFINITY,
                    f32::INFINITY,
                    f32::NEG_INFINITY,
                    2.33,
                ],
            )
        );
    }

    // OpScatterValueOutTest.InvalidDimensionsDies
    // [spec:et:sem:op-scatter.torch.executor.native.scatter-value-out-fn/test]
    #[test]
    fn op_scatter_value_out_test_invalid_dimensions_dies() {
        // ET_FORALL_REAL_TYPES
        test_scatter_value_out_invalid_dim::<u8>();
        test_scatter_value_out_invalid_dim::<i8>();
        test_scatter_value_out_invalid_dim::<i16>();
        test_scatter_value_out_invalid_dim::<i32>();
        test_scatter_value_out_invalid_dim::<i64>();
        test_scatter_value_out_invalid_dim::<f32>();
        test_scatter_value_out_invalid_dim::<f64>();
    }

    // OpScatterValueOutTest.MismatchedInputDtypesDies
    // [spec:et:sem:op-scatter.torch.executor.native.scatter-value-out-fn/test]
    #[test]
    fn op_scatter_value_out_test_mismatched_input_dtypes_dies() {
        let tf_byte = TensorFactory::<u8>::new();
        let tf_char = TensorFactory::<i8>::new();
        let tf_long = TensorFactory::<i64>::new();

        let mut self_ = tf_char.make_default(vec![2, 5], vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
        let sizes = vec![2, 3];
        let mut index = tf_byte.make_default(sizes.clone(), vec![0, 1, 0, 0, 1, 0]);
        let value = Scalar::from_i64(5);
        let mut out = tf_char.zeros_default(sizes.clone());

        // Types other than long for index should die
        let mut ctx = context();
        scatter_value_out(&mut ctx, &self_, 0, &index, &value, &out);
        assert_ne!(ctx.failure_state(), Error::Ok);

        // Mismatched dtype of self and out should die
        self_ = tf_byte.make_default(vec![2, 5], vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
        index = tf_long.make_default(sizes.clone(), vec![0, 1, 0, 1, 0, 1]);
        out = tf_char.zeros_default(sizes);
        let mut ctx = context();
        scatter_value_out(&mut ctx, &self_, 0, &index, &value, &out);
        assert_ne!(ctx.failure_state(), Error::Ok);
    }

    // OpScatterValueOutTest.DynamicShapeUpperBoundSameAsExpected
    // [spec:et:sem:op-scatter.torch.executor.native.scatter-value-out-fn/test]
    #[test]
    fn op_scatter_value_out_test_dynamic_shape_upper_bound_same_as_expected() {
        test_dynamic_shape(vec![2, 3, 4], TensorShapeDynamism::DYNAMIC_BOUND);
    }

    // OpScatterValueOutTest.DynamicShapeUpperBoundLargerThanExpected
    // [spec:et:sem:op-scatter.torch.executor.native.scatter-value-out-fn/test]
    #[test]
    fn op_scatter_value_out_test_dynamic_shape_upper_bound_larger_than_expected() {
        test_dynamic_shape(vec![10, 10, 10], TensorShapeDynamism::DYNAMIC_BOUND);
    }

    // PORT-NOTE: C++ guards this with ET_SKIP_IF(!output_resize); the portable
    // build does not support DYNAMIC_UNBOUND resize, so the test is #[ignore]d.
    // OpScatterValueOutTest.DynamicShapeUnbound
    // [spec:et:sem:op-scatter.torch.executor.native.scatter-value-out-fn/test]
    #[test]
    #[ignore = "DynamicShapeUnbound: dynamic shape not supported"]
    fn op_scatter_value_out_test_dynamic_shape_unbound() {
        test_dynamic_shape(vec![1, 1, 1], TensorShapeDynamism::DYNAMIC_UNBOUND);
    }

    // OpScatterValueOutTest.EmptyIndex
    // [spec:et:sem:op-scatter.torch.executor.native.scatter-value-out-fn/test]
    #[test]
    fn op_scatter_value_out_test_empty_index() {
        let tf_index = TensorFactory::<i64>::new();
        let tf_data = TensorFactory::<f32>::new();

        let self_ = tf_data.ones_default(vec![2, 5]);
        let index = tf_index.zeros_default(vec![2, 0, 3]);
        let value = Scalar::from_i64(5);
        let out = tf_data.zeros_default(vec![2, 5]);
        op_scatter_value_out(&self_, 0, &index, &value, &out);
        assert_tensor_close!(out, tf_data.ones_default(vec![2, 5]));
    }

    // OpScatterValueOutTest.ValidZeroDim
    // [spec:et:sem:op-scatter.torch.executor.native.scatter-value-out-fn/test]
    #[test]
    fn op_scatter_value_out_test_valid_zero_dim() {
        let tf_index = TensorFactory::<i64>::new();
        let tf_data = TensorFactory::<f32>::new();

        let self_ = tf_data.make_default(vec![], vec![3.14]);
        let index = tf_index.zeros_default(vec![]);
        let value = Scalar::from_i64(5);
        let out = tf_data.zeros_default(vec![]);
        op_scatter_value_out(&self_, 0, &index, &value, &out);
        assert_tensor_close!(out, tf_data.make_default(vec![], vec![5.0]));
    }

    // OpScatterValueOutTest.InvalidZeroDimInput
    // [spec:et:sem:op-scatter.torch.executor.native.scatter-value-out-fn/test]
    #[test]
    fn op_scatter_value_out_test_invalid_zero_dim_input() {
        let tf_index = TensorFactory::<i64>::new();
        let tf_data = TensorFactory::<f32>::new();

        let self_ = tf_data.ones_default(vec![]);
        let index = tf_index.make_default(vec![2, 3], vec![0, 0, 0, 0, 0, 0]);
        let value = Scalar::from_i64(5);
        let out = tf_data.zeros_default(vec![]);
        let mut ctx = context();
        scatter_value_out(&mut ctx, &self_, 0, &index, &value, &out);
        assert_ne!(ctx.failure_state(), Error::Ok);
    }

    // OpScatterValueOutTest.InvalidZeroDimIndex
    // [spec:et:sem:op-scatter.torch.executor.native.scatter-value-out-fn/test]
    #[test]
    fn op_scatter_value_out_test_invalid_zero_dim_index() {
        let tf_index = TensorFactory::<i64>::new();
        let tf_data = TensorFactory::<f32>::new();

        let self_ = tf_data.make_default(vec![2, 3], vec![1., 2., 3., 4., 5., 6.]);
        let index = tf_index.make_default(vec![], vec![2]);
        let value = Scalar::from_i64(5);
        let out = tf_data.zeros_default(vec![2, 3]);
        let mut ctx = context();
        scatter_value_out(&mut ctx, &self_, 1, &index, &value, &out);
        assert_ne!(ctx.failure_state(), Error::Ok);
    }

    // OpScatterValueOutTest.ValidZeroDimInputAndOneDimIndex
    // [spec:et:sem:op-scatter.torch.executor.native.scatter-value-out-fn/test]
    #[test]
    fn op_scatter_value_out_test_valid_zero_dim_input_and_one_dim_index() {
        let tf_index = TensorFactory::<i64>::new();
        let tf_data = TensorFactory::<f32>::new();

        let self_ = tf_data.make_default(vec![], vec![3.14]);
        let index = tf_index.make_default(vec![3], vec![0, 0, 0]);
        let value = Scalar::from_i64(5);
        let out = tf_data.make_default(vec![], vec![2.71]);
        op_scatter_value_out(&self_, 0, &index, &value, &out);
        assert_tensor_close!(out, tf_data.make_default(vec![], vec![5.0]));
    }

    // OpScatterValueOutTest.ValidOneDimInputAndZeroDimIndex
    // [spec:et:sem:op-scatter.torch.executor.native.scatter-value-out-fn/test]
    #[test]
    fn op_scatter_value_out_test_valid_one_dim_input_and_zero_dim_index() {
        let tf_index = TensorFactory::<i64>::new();
        let tf_data = TensorFactory::<f32>::new();

        let self_ = tf_data.make_default(vec![3], vec![10., 20., 30.]);
        let index = tf_index.make_default(vec![], vec![2]);
        let value = Scalar::from_i64(5);
        let out = tf_data.make_default(vec![3], vec![1729., 1729., 1729.]);
        op_scatter_value_out(&self_, 0, &index, &value, &out);
        assert_tensor_close!(out, tf_data.make_default(vec![3], vec![10., 20., 5.]));
    }

    // OpScatterValueOutTest.InvalidZeroDimInputAndOneDimIndex
    // [spec:et:sem:op-scatter.torch.executor.native.scatter-value-out-fn/test]
    #[test]
    fn op_scatter_value_out_test_invalid_zero_dim_input_and_one_dim_index() {
        let tf_index = TensorFactory::<i64>::new();
        let tf_data = TensorFactory::<f32>::new();

        let self_ = tf_data.make_default(vec![], vec![3.14]);
        let index = tf_index.make_default(vec![3], vec![10, 100, 1000]);
        let value = Scalar::from_i64(5);
        let out = tf_data.make_default(vec![], vec![2.71]);
        let mut ctx = context();
        scatter_value_out(&mut ctx, &self_, 0, &index, &value, &out);
        assert_ne!(ctx.failure_state(), Error::Ok);
    }

    // OpScatterValueOutTest.InvalidOneDimInputAndZeroDimIndex
    // [spec:et:sem:op-scatter.torch.executor.native.scatter-value-out-fn/test]
    #[test]
    fn op_scatter_value_out_test_invalid_one_dim_input_and_zero_dim_index() {
        let tf_index = TensorFactory::<i64>::new();
        let tf_data = TensorFactory::<f32>::new();

        let self_ = tf_data.make_default(vec![3], vec![10., 20., 30.]);
        let index = tf_index.make_default(vec![], vec![100]);
        let value = Scalar::from_i64(5);
        let out = tf_data.make_default(vec![3], vec![1729., 1729., 1729.]);
        let mut ctx = context();
        scatter_value_out(&mut ctx, &self_, 0, &index, &value, &out);
        assert_ne!(ctx.failure_state(), Error::Ok);
    }

    // OpScatterSrcOutTest.EmptyIndex
    // [spec:et:sem:op-scatter.torch.executor.native.scatter-src-out-fn/test]
    #[test]
    fn op_scatter_src_out_test_empty_index() {
        let tf_index = TensorFactory::<i64>::new();
        let tf_data = TensorFactory::<f32>::new();

        let self_ = tf_data.ones_default(vec![2, 5]);
        let index = tf_index.zeros_default(vec![2, 0, 3]);
        let src = tf_data.ones_default(vec![1, 1, 4]);
        let out = tf_data.zeros_default(vec![2, 5]);
        op_scatter_src_out(&self_, 0, &index, &src, &out);
        assert_tensor_close!(out, tf_data.ones_default(vec![2, 5]));
    }

    // OpScatterSrcOutTest.ValidZeroDim
    // [spec:et:sem:op-scatter.torch.executor.native.scatter-src-out-fn/test]
    #[test]
    fn op_scatter_src_out_test_valid_zero_dim() {
        let tf_index = TensorFactory::<i64>::new();
        let tf_data = TensorFactory::<f32>::new();

        let self_ = tf_data.make_default(vec![], vec![3.14]);
        let index = tf_index.zeros_default(vec![]);
        let src = tf_data.make_default(vec![], vec![5.0]);
        let out = tf_data.zeros_default(vec![]);
        op_scatter_src_out(&self_, 0, &index, &src, &out);
        assert_tensor_close!(out, tf_data.make_default(vec![], vec![5.0]));
    }

    // OpScatterSrcOutTest.InvalidZeroDimInput
    // [spec:et:sem:op-scatter.torch.executor.native.scatter-src-out-fn/test]
    #[test]
    fn op_scatter_src_out_test_invalid_zero_dim_input() {
        let tf_index = TensorFactory::<i64>::new();
        let tf_data = TensorFactory::<f32>::new();

        let self_ = tf_data.ones_default(vec![]);
        let index = tf_index.make_default(vec![2, 3], vec![0, 0, 0, 0, 0, 0]);
        let src = tf_data.make_default(vec![], vec![5.0]);
        let out = tf_data.zeros_default(vec![]);
        let mut ctx = context();
        scatter_src_out(&mut ctx, &self_, 0, &index, &src, &out);
        assert_ne!(ctx.failure_state(), Error::Ok);
    }

    // OpScatterSrcOutTest.InvalidZeroDimIndex
    // [spec:et:sem:op-scatter.torch.executor.native.scatter-src-out-fn/test]
    #[test]
    fn op_scatter_src_out_test_invalid_zero_dim_index() {
        let tf_index = TensorFactory::<i64>::new();
        let tf_data = TensorFactory::<f32>::new();

        let self_ = tf_data.make_default(vec![2, 3], vec![1., 2., 3., 4., 5., 6.]);
        let index = tf_index.make_default(vec![], vec![2]);
        let src = tf_data.make_default(vec![], vec![5.0]);
        let out = tf_data.zeros_default(vec![2, 3]);
        let mut ctx = context();
        scatter_src_out(&mut ctx, &self_, 1, &index, &src, &out);
        assert_ne!(ctx.failure_state(), Error::Ok);
    }

    // OpScatterSrcOutTest.ValidZeroDimInputAndOneDimIndex
    // [spec:et:sem:op-scatter.torch.executor.native.scatter-src-out-fn/test]
    #[test]
    fn op_scatter_src_out_test_valid_zero_dim_input_and_one_dim_index() {
        let tf_index = TensorFactory::<i64>::new();
        let tf_data = TensorFactory::<f32>::new();

        let self_ = tf_data.make_default(vec![], vec![3.14]);
        let index = tf_index.make_default(vec![3], vec![0, 0, 0]);
        let src = tf_data.make_default(vec![3], vec![5., 5., 5.]);
        let out = tf_data.make_default(vec![], vec![2.71]);
        op_scatter_src_out(&self_, 0, &index, &src, &out);
        assert_tensor_close!(out, tf_data.make_default(vec![], vec![5.0]));
    }

    // OpScatterSrcOutTest.ValidOneDimInputAndZeroDimIndex
    // [spec:et:sem:op-scatter.torch.executor.native.scatter-src-out-fn/test]
    #[test]
    fn op_scatter_src_out_test_valid_one_dim_input_and_zero_dim_index() {
        let tf_index = TensorFactory::<i64>::new();
        let tf_data = TensorFactory::<f32>::new();

        let self_ = tf_data.make_default(vec![3], vec![10., 20., 30.]);
        let index = tf_index.make_default(vec![], vec![2]);
        let src = tf_data.make_default(vec![], vec![5.0]);
        let out = tf_data.make_default(vec![3], vec![1729., 1729., 1729.]);
        op_scatter_src_out(&self_, 0, &index, &src, &out);
        assert_tensor_close!(out, tf_data.make_default(vec![3], vec![10., 20., 5.]));
    }

    // OpScatterSrcOutTest.InvalidZeroDimInputAndOneDimIndex
    // [spec:et:sem:op-scatter.torch.executor.native.scatter-src-out-fn/test]
    #[test]
    fn op_scatter_src_out_test_invalid_zero_dim_input_and_one_dim_index() {
        let tf_index = TensorFactory::<i64>::new();
        let tf_data = TensorFactory::<f32>::new();

        let self_ = tf_data.make_default(vec![], vec![3.14]);
        let index = tf_index.make_default(vec![3], vec![10, 100, 1000]);
        let src = tf_data.make_default(vec![], vec![5.0]);
        let out = tf_data.make_default(vec![], vec![2.71]);
        let mut ctx = context();
        scatter_src_out(&mut ctx, &self_, 0, &index, &src, &out);
        assert_ne!(ctx.failure_state(), Error::Ok);
    }

    // OpScatterSrcOutTest.InvalidOneDimInputAndZeroDimIndex
    // [spec:et:sem:op-scatter.torch.executor.native.scatter-src-out-fn/test]
    #[test]
    fn op_scatter_src_out_test_invalid_one_dim_input_and_zero_dim_index() {
        let tf_index = TensorFactory::<i64>::new();
        let tf_data = TensorFactory::<f32>::new();

        let self_ = tf_data.make_default(vec![3], vec![10., 20., 30.]);
        let index = tf_index.make_default(vec![], vec![100]);
        let src = tf_data.make_default(vec![], vec![5.0]);
        let out = tf_data.make_default(vec![3], vec![1729., 1729., 1729.]);
        let mut ctx = context();
        scatter_src_out(&mut ctx, &self_, 0, &index, &src, &out);
        assert_ne!(ctx.failure_state(), Error::Ok);
    }

    // GENERATE_SCALAR_OVERFLOW_TESTS(OpScatterValueOutTest).
    // [spec:et:sem:op-scatter.torch.executor.native.scatter-value-out-fn/test]
    #[test]
    fn op_scatter_value_out_test_byte_tensor_too_large_scalar_dies() {
        // Cannot be represented by a uint8_t.
        expect_bad_scalar_value_dies::<u8>(Scalar::from_i64(256));
    }
    // [spec:et:sem:op-scatter.torch.executor.native.scatter-value-out-fn/test]
    #[test]
    fn op_scatter_value_out_test_char_tensor_too_small_scalar_dies() {
        // Cannot be represented by a int8_t.
        expect_bad_scalar_value_dies::<i8>(Scalar::from_i64(-129));
    }
    // [spec:et:sem:op-scatter.torch.executor.native.scatter-value-out-fn/test]
    #[test]
    fn op_scatter_value_out_test_short_tensor_too_large_scalar_dies() {
        // Cannot be represented by a int16_t.
        expect_bad_scalar_value_dies::<i16>(Scalar::from_i64(32768));
    }
    // [spec:et:sem:op-scatter.torch.executor.native.scatter-value-out-fn/test]
    #[test]
    fn op_scatter_value_out_test_float_tensor_too_small_scalar_dies() {
        // Cannot be represented by a float.
        expect_bad_scalar_value_dies::<f32>(Scalar::from_double(-3.41e+38));
    }
    // [spec:et:sem:op-scatter.torch.executor.native.scatter-value-out-fn/test]
    #[test]
    fn op_scatter_value_out_test_float_tensor_too_large_scalar_dies() {
        // Cannot be represented by a float.
        expect_bad_scalar_value_dies::<f32>(Scalar::from_double(3.41e+38));
    }
}
