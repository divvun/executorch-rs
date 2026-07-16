//! Literal port of kernels/portable/cpu/op_scatter_add.cpp.

use crate::kernels::portable::cpu::util::index_util::check_scatter_add_args;
use crate::runtime::core::error::Error;
use crate::runtime::core::exec_aten::util::tensor_util::{
    K_TENSOR_DIMENSION_LIMIT, coordinateToIndex, indexToCoordinate, nonempty_size, nonzero_dim,
    resize_tensor, tensor_is_default_dim_order, tensors_have_same_dim_order3,
};
use crate::runtime::core::portable_type::scalar_type::ScalarType;
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

// PORT-NOTE: the C++ private helper `scatter_add_helper<CTYPE>` is a template
// instantiated per-ctype by the switch in `scatter_add_out`. The Rust generic
// function reproduces those instantiations. The void-typed `memcpy(out_data,
// self_data, self.nbytes())` is a byte-wise `copy_nonoverlapping` over the
// `CTYPE*` buffers. `out_data[out_ix] += src_data[src_ix]` needs CTYPE `+`, so
// the `AddCtype` trait reproduces the C++ `+=` for the REALHBBF16 set (Bool via
// integer promotion; Half/BFloat16 via float).

trait AddCtype: Copy {
    fn add_assign_ctype(self, other: Self) -> Self;
}
macro_rules! impl_add_prim {
    ($($t:ty),*) => {$(
        impl AddCtype for $t {
            fn add_assign_ctype(self, other: Self) -> Self { self + other }
        }
    )*};
}
impl_add_prim!(u8, i8, i16, i32, i64, f32, f64);
impl AddCtype for bool {
    fn add_assign_ctype(self, other: Self) -> Self {
        ((self as i32) + (other as i32)) != 0
    }
}
use crate::runtime::core::portable_type::{BFloat16, Half};
macro_rules! impl_add_half {
    ($t:ty) => {
        impl AddCtype for $t {
            fn add_assign_ctype(self, other: Self) -> Self {
                <$t>::from_f32(self.to_f32() + other.to_f32())
            }
        }
    };
}
impl_add_half!(Half);
impl_add_half!(BFloat16);

// [spec:et:def:op-scatter-add.torch.executor.native.scatter-add-helper-fn]
// [spec:et:sem:op-scatter-add.torch.executor.native.scatter-add-helper-fn]
fn scatter_add_helper<CTYPE: AddCtype>(
    src_data: *const CTYPE,
    index_data: *const i64,
    out_data: *mut CTYPE,
    src: &Tensor,
    index: &Tensor,
    out: &Tensor,
    dim: i64,
) {
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
            *out_data.add(out_ix) = (*out_data.add(out_ix)).add_assign_ctype(*src_data.add(src_ix));
        }
    }
}

// [spec:et:def:op-scatter-add.torch.executor.native.scatter-add-out-fn]
// [spec:et:sem:op-scatter-add.torch.executor.native.scatter-add-out-fn]
pub fn scatter_add_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    self_: &Tensor,
    mut dim: i64,
    index: &Tensor,
    src: &Tensor,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    crate::et_kernel_check!(
        ctx,
        check_scatter_add_args(self_, dim, index, src, out),
        InvalidArgument,
        out
    );

    crate::et_kernel_check!(
        ctx,
        tensors_have_same_dim_order3(self_, src, out),
        InvalidArgument,
        out
    );

    crate::et_kernel_check!(
        ctx,
        tensor_is_default_dim_order(index),
        InvalidArgument,
        out
    );

    if dim < 0 {
        dim += nonzero_dim(self_) as i64;
    }

    crate::et_kernel_check!(
        ctx,
        resize_tensor(out, self_.sizes()) == Error::Ok,
        InvalidArgument,
        out
    );

    let self_type: ScalarType = self_.scalar_type();

    crate::et_switch_realhbbf16_types!(self_type, ctx, "scatter_add.out", CTYPE, {
        let self_data: *const CTYPE = self_.const_data_ptr::<CTYPE>();
        let index_data: *const i64 = index.const_data_ptr::<i64>();
        let src_data: *const CTYPE = src.const_data_ptr::<CTYPE>();
        let out_data: *mut CTYPE = out.mutable_data_ptr::<CTYPE>();

        unsafe {
            core::ptr::copy_nonoverlapping(
                self_data as *const u8,
                out_data as *mut u8,
                self_.nbytes(),
            );
        }

        if index.numel() != 0 {
            if self_.dim() == 0 {
                unsafe {
                    *out_data.add(0) = (*out_data.add(0)).add_assign_ctype(
                        <CTYPE as StaticCastFromI64>::static_cast(nonempty_size(index, 0) as i64)
                            .mul_ctype(*src_data.add(0)),
                    );
                }
            } else {
                scatter_add_helper::<CTYPE>(src_data, index_data, out_data, src, index, out, dim);
            }
        }
    });

    out
}

// PORT-NOTE: the scalar-`self` branch computes
// `static_cast<CTYPE>(nonempty_size(index, 0)) * src_data[0]`. `StaticCastFromI64`
// reproduces the `static_cast<CTYPE>(int64_t)` and `mul_ctype` the CTYPE `*`
// (Bool via integer promotion, Half/BFloat16 via float), for the REALHBBF16 set.
trait StaticCastFromI64: Copy {
    fn static_cast(v: i64) -> Self;
    fn mul_ctype(self, other: Self) -> Self;
}
macro_rules! impl_scast_prim {
    ($($t:ty),*) => {$(
        impl StaticCastFromI64 for $t {
            fn static_cast(v: i64) -> Self { v as $t }
            fn mul_ctype(self, other: Self) -> Self { self * other }
        }
    )*};
}
impl_scast_prim!(u8, i8, i16, i32, i64, f32, f64);
impl StaticCastFromI64 for bool {
    fn static_cast(v: i64) -> Self {
        v != 0
    }
    fn mul_ctype(self, other: Self) -> Self {
        ((self as i32) * (other as i32)) != 0
    }
}
macro_rules! impl_scast_half {
    ($t:ty) => {
        impl StaticCastFromI64 for $t {
            fn static_cast(v: i64) -> Self {
                <$t>::from_f32(v as f32)
            }
            fn mul_ctype(self, other: Self) -> Self {
                <$t>::from_f32(self.to_f32() * other.to_f32())
            }
        }
    };
}
impl_scast_half!(Half);
impl_scast_half!(BFloat16);

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
    use crate::runtime::core::tensor_shape_dynamism::TensorShapeDynamism;

    fn context() -> KernelRuntimeContext<'static> {
        crate::runtime::platform::runtime::runtime_init();
        KernelRuntimeContext::new(
            crate::extension::module::module::null_event_tracer(),
            core::ptr::null_mut::<MemoryAllocator>() as *mut dyn MemoryAllocatorBase,
        )
    }

    fn op_scatter_add_out<'a, 'b>(
        self_: &Tensor,
        dim: i64,
        index: &Tensor,
        src: &Tensor,
        out: &'a Tensor<'b>,
    ) -> &'a Tensor<'b> {
        let mut ctx = context();
        scatter_add_out(&mut ctx, self_, dim, index, src, out)
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

    // test_scatter_add_out<DATA_DTYPE>
    fn test_scatter_add_out<T>()
    where
        T: CppTypeToScalarType + FactoryValue + FromF64Elem,
    {
        let tf_index = TensorFactory::<i64>::new();
        let tf_data = TensorFactory::<T>::new();
        let sizes = vec![3, 5];

        let src = tf_data.make_default(
            vec![2, 5],
            v::<T>(&[1., 2., 3., 4., 5., 6., 7., 8., 9., 10.]),
        );
        let self_ = tf_data.zeros_default(sizes.clone());
        let out = tf_data.zeros_default(sizes.clone());
        let index = tf_index.make_default(vec![2, 3], vec![0, 1, 2, 0, 1, 2]);

        op_scatter_add_out(&self_, 0, &index, &src, &out);
        assert_tensor_eq!(
            out,
            tf_data.make_default(
                sizes.clone(),
                v::<T>(&[7., 0., 0., 0., 0., 0., 9., 0., 0., 0., 0., 0., 11., 0., 0.]),
            )
        );

        op_scatter_add_out(&self_, 1, &index, &src, &out);
        assert_tensor_eq!(
            out,
            tf_data.make_default(
                sizes,
                v::<T>(&[1., 2., 3., 0., 0., 6., 7., 8., 0., 0., 0., 0., 0., 0., 0.]),
            )
        );

        let src = tf_data.make_default(
            vec![2, 3, 3],
            v::<T>(&[
                1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18.,
            ]),
        );
        let self_ = tf_data.ones_default(vec![2, 3, 3]);
        let out = tf_data.zeros_default(vec![2, 3, 3]);
        let index = tf_index.make_default(vec![1, 3, 2], vec![0, 1, 1, 2, 0, 2]);

        op_scatter_add_out(&self_, 1, &index, &src, &out);
        assert_tensor_eq!(
            out,
            tf_data.make_default(
                vec![2, 3, 3],
                v::<T>(&[
                    9., 1., 1., 5., 3., 1., 1., 14., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                ]),
            )
        );

        let out = tf_data.zeros_default(vec![2, 3, 3]);
        op_scatter_add_out(&self_, 2, &index, &src, &out);
        assert_tensor_eq!(
            out,
            tf_data.make_default(
                vec![2, 3, 3],
                v::<T>(&[
                    2., 3., 1., 1., 5., 6., 8., 1., 9., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                ]),
            )
        );
    }

    // test_scatter_add_out_invalid_dim<DATA_DTYPE>
    fn test_scatter_add_out_invalid_dim<T>()
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
        scatter_add_out(&mut ctx, &self_, -3, &index, &src, &out);
        assert_ne!(ctx.failure_state(), Error::Ok);
        let mut ctx = context();
        scatter_add_out(&mut ctx, &self_, 2, &index, &src, &out);
        assert_ne!(ctx.failure_state(), Error::Ok);

        src = tf_data.zeros_default(vec![2, 2, 2]);
        let mut ctx = context();
        scatter_add_out(&mut ctx, &self_, 0, &index, &src, &out);
        assert_ne!(ctx.failure_state(), Error::Ok);

        src = tf_data.zeros_default(vec![5, 5]);
        index = tf_index.zeros_default(vec![2, 2, 2]);
        let mut ctx = context();
        scatter_add_out(&mut ctx, &self_, 0, &index, &src, &out);
        assert_ne!(ctx.failure_state(), Error::Ok);

        index = tf_index.zeros_default(vec![4, 6]);
        let mut ctx = context();
        scatter_add_out(&mut ctx, &self_, 0, &index, &src, &out);
        assert_ne!(ctx.failure_state(), Error::Ok);

        index = tf_index.zeros_default(vec![4, 5]);
        let mut ctx = context();
        scatter_add_out(&mut ctx, &self_, 1, &index, &src, &out);
        assert_ne!(ctx.failure_state(), Error::Ok);

        index = tf_index.make_default(vec![2, 3], vec![0, 1, 3, 0, 1, 3]);
        let mut ctx = context();
        scatter_add_out(&mut ctx, &self_, 0, &index, &src, &out);
        assert_ne!(ctx.failure_state(), Error::Ok);
    }

    // test_scatter_add_out_mismatched_shape<DATA_DTYPE>
    fn test_scatter_add_out_mismatched_shape<T>()
    where
        T: CppTypeToScalarType + FactoryValue + FromF64Elem,
    {
        let tf_index = TensorFactory::<i64>::new();
        let tf_data = TensorFactory::<T>::new();

        let src = tf_data.make_default(
            vec![2, 5],
            v::<T>(&[1., 2., 3., 4., 5., 6., 7., 8., 9., 10.]),
        );
        let index = tf_index.make_default(vec![2, 3], vec![0, 1, 2, 0, 1, 2]);
        let self_ = tf_data.zeros_default(vec![3, 5]);
        let out = tf_data.zeros_default(vec![2, 5]);

        let mut ctx = context();
        scatter_add_out(&mut ctx, &self_, 0, &index, &src, &out);
        assert_ne!(ctx.failure_state(), Error::Ok);
    }

    fn test_dynamic_shape(out_shape: Vec<i32>, dynamism: TensorShapeDynamism) {
        let tf = TensorFactory::<i32>::new();
        let tf_index = TensorFactory::<i64>::new();

        let input = tf.make_default(
            vec![2, 3, 4],
            vec![
                4, 9, 3, 0, 3, 9, 7, 3, 7, 3, 1, 6, 6, 9, 8, 6, 6, 8, 4, 3, 6, 9, 1, 4,
            ],
        );
        let index = tf_index.make_default(
            vec![2, 3, 4],
            vec![
                0, 1, 1, 1, 1, 0, 1, 0, 3, 0, 3, 1, 2, 3, 3, 0, 2, 3, 0, 1, 3, 1, 3, 3,
            ],
        );
        let src = tf.make_default(
            vec![2, 3, 4],
            vec![
                2, 1, 0, 9, 3, 1, 1, 0, 3, 6, 6, 7, 9, 6, 3, 4, 5, 0, 8, 2, 8, 2, 7, 5,
            ],
        );
        let expected = tf.make_default(
            vec![2, 3, 4],
            vec![
                6, 19, 3, 0, 4, 13, 7, 3, 13, 10, 1, 15, 10, 9, 17, 15, 14, 10, 9, 3, 6, 11, 1, 24,
            ],
        );
        let out = tf.zeros(out_shape, dynamism);

        op_scatter_add_out(&input, 2, &index, &src, &out);
        assert_tensor_eq!(out, expected);
    }

    // [spec:et:sem:op-scatter-add.torch.executor.native.scatter-add-out-fn/test]
    // also verifies check_scatter_add_args (valid path); the _invalid_dim and
    // _mismatched_shape dies tests exercise its failure branches.
    // [spec:et:sem:index-util.torch.executor.check-scatter-add-args-fn/test]
    // also verifies scatter_add_helper: the asserted accumulated outputs are produced by
    // this helper's index-driven `out[...] += src[...]` loop.
    // [spec:et:sem:op-scatter-add.torch.executor.native.scatter-add-helper-fn/test]
    #[test]
    fn op_scatter_add_out_test_all_valid_input_output_support() {
        // ET_FORALL_REALHBF16_TYPES
        test_scatter_add_out::<u8>();
        test_scatter_add_out::<i8>();
        test_scatter_add_out::<i16>();
        test_scatter_add_out::<i32>();
        test_scatter_add_out::<i64>();
        test_scatter_add_out::<f32>();
        test_scatter_add_out::<f64>();
        test_scatter_add_out::<Half>();
        test_scatter_add_out::<BFloat16>();
    }

    // [spec:et:sem:op-scatter-add.torch.executor.native.scatter-add-out-fn/test]
    #[test]
    fn op_scatter_add_out_test_infinity_and_nan_test() {
        let tf_index = TensorFactory::<i64>::new();
        let tf_data = TensorFactory::<f32>::new();
        let sizes = vec![3, 5];

        let src = tf_data.make_default(
            vec![2, 5],
            vec![
                f32::INFINITY,
                f32::NEG_INFINITY,
                f32::NAN,
                2.33,
                3.14,
                f32::NAN,
                f32::INFINITY,
                f32::NEG_INFINITY,
                3.14,
                2.33,
            ],
        );
        let self_ = tf_data.ones_default(sizes.clone());
        let out = tf_data.zeros_default(sizes.clone());
        let index = tf_index.make_default(vec![2, 3], vec![0, 1, 2, 0, 1, 2]);

        op_scatter_add_out(&self_, 0, &index, &src, &out);
        assert_tensor_close!(
            out,
            tf_data.make_default(
                sizes,
                vec![
                    f32::NAN,
                    1.,
                    1.,
                    1.,
                    1.,
                    1.,
                    f32::NAN,
                    1.,
                    1.,
                    1.,
                    1.,
                    1.,
                    f32::NAN,
                    1.,
                    1.,
                ],
            )
        );
    }

    // [spec:et:sem:op-scatter-add.torch.executor.native.scatter-add-out-fn/test]
    #[test]
    fn op_scatter_add_out_test_invalid_dimensions_dies() {
        // ET_FORALL_REAL_TYPES
        test_scatter_add_out_invalid_dim::<u8>();
        test_scatter_add_out_invalid_dim::<i8>();
        test_scatter_add_out_invalid_dim::<i16>();
        test_scatter_add_out_invalid_dim::<i32>();
        test_scatter_add_out_invalid_dim::<i64>();
        test_scatter_add_out_invalid_dim::<f32>();
        test_scatter_add_out_invalid_dim::<f64>();
    }

    // PORT-NOTE: C++ guards with ET_SKIP_IF(is_aten); ET-mode port runs it.
    // [spec:et:sem:op-scatter-add.torch.executor.native.scatter-add-out-fn/test]
    #[test]
    fn op_scatter_add_out_test_mismatched_shape_dies() {
        // ET_FORALL_REAL_TYPES
        test_scatter_add_out_mismatched_shape::<u8>();
        test_scatter_add_out_mismatched_shape::<i8>();
        test_scatter_add_out_mismatched_shape::<i16>();
        test_scatter_add_out_mismatched_shape::<i32>();
        test_scatter_add_out_mismatched_shape::<i64>();
        test_scatter_add_out_mismatched_shape::<f32>();
        test_scatter_add_out_mismatched_shape::<f64>();
    }

    // [spec:et:sem:op-scatter-add.torch.executor.native.scatter-add-out-fn/test]
    #[test]
    fn op_scatter_add_out_test_mismatched_input_dtypes_dies() {
        let tf_byte = TensorFactory::<u8>::new();
        let tf_char = TensorFactory::<i8>::new();
        let sizes = vec![3, 5];

        let mut src = tf_char.make_default(vec![2, 5], vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
        let index = tf_byte.make_default(vec![2, 3], vec![0, 1, 2, 0, 1, 2]);
        let mut self_ = tf_char.zeros_default(sizes.clone());
        let mut out = tf_char.zeros_default(sizes.clone());

        // Types other than long for index should die
        let mut ctx = context();
        scatter_add_out(&mut ctx, &self_, 0, &index, &src, &out);
        assert_ne!(ctx.failure_state(), Error::Ok);

        // Mismatched dtype of src and self should die
        src = tf_char.make_default(vec![2, 5], vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
        let mut ctx = context();
        scatter_add_out(&mut ctx, &self_, 0, &index, &src, &out);
        assert_ne!(ctx.failure_state(), Error::Ok);

        src = tf_byte.make_default(vec![2, 5], vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
        self_ = tf_byte.zeros_default(sizes.clone());
        out = tf_char.zeros_default(sizes);

        // Mismatched dtype of self and out should die
        let mut ctx = context();
        scatter_add_out(&mut ctx, &self_, 0, &index, &src, &out);
        assert_ne!(ctx.failure_state(), Error::Ok);
    }

    // [spec:et:sem:op-scatter-add.torch.executor.native.scatter-add-out-fn/test]
    #[test]
    fn op_scatter_add_out_test_dynamic_shape_upper_bound_same_as_expected() {
        test_dynamic_shape(vec![2, 3, 4], TensorShapeDynamism::DYNAMIC_BOUND);
    }

    // [spec:et:sem:op-scatter-add.torch.executor.native.scatter-add-out-fn/test]
    #[test]
    fn op_scatter_add_out_test_dynamic_shape_upper_bound_larger_than_expected() {
        test_dynamic_shape(vec![10, 10, 10], TensorShapeDynamism::DYNAMIC_BOUND);
    }

    // PORT-NOTE: C++ guards this with ET_SKIP_IF(!output_resize); the portable
    // build does not support DYNAMIC_UNBOUND resize, so the test is #[ignore]d.
    // [spec:et:sem:op-scatter-add.torch.executor.native.scatter-add-out-fn/test]
    #[test]
    #[ignore = "DynamicShapeUnbound: dynamic shape not supported"]
    fn op_scatter_add_out_test_dynamic_shape_unbound() {
        test_dynamic_shape(vec![1, 1, 1], TensorShapeDynamism::DYNAMIC_UNBOUND);
    }
}
