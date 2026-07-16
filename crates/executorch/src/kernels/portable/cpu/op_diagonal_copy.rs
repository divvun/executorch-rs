//! Literal port of kernels/portable/cpu/op_diagonal_copy.cpp.

use crate::kernels::portable::cpu::util::copy_ops_util::{
    as_strided_copy, check_diagonal_copy_args, get_diagonal_copy_out_target_size,
};
use crate::runtime::core::array_ref::ArrayRef;
use crate::runtime::core::error::Error;
use crate::runtime::core::exec_aten::util::tensor_util::{
    K_TENSOR_DIMENSION_LIMIT, nonzero_dim, resize_tensor_same_type, tensor_is_default_dim_order,
    tensors_have_same_dim_order2,
};
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::core::portable_type::tensor_impl::SizesType;
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

// [spec:et:def:op-diagonal-copy.torch.executor.native.diagonal-copy-impl-fn]
// [spec:et:sem:op-diagonal-copy.torch.executor.native.diagonal-copy-impl-fn]
fn diagonal_copy_impl<CTYPE: Copy>(in_: &Tensor, offset: i64, dim1: i64, dim2: i64, out: &Tensor) {
    if out.numel() == 0 {
        return;
    }

    let mut storage_offset: i64 = 0;
    let diag_size: usize = out.size(out.dim() - 1) as usize;

    if diag_size == 0 {
        // skip
    } else if offset >= 0 {
        storage_offset += offset * (*in_.strides().at(dim2 as usize)) as i64;
    } else {
        storage_offset -= offset * (*in_.strides().at(dim1 as usize)) as i64;
    }

    let new_ndim: usize = out.dim() as usize;
    let mut new_sizes: [i64; K_TENSOR_DIMENSION_LIMIT] = [0; K_TENSOR_DIMENSION_LIMIT];
    for i in 0..new_ndim {
        new_sizes[i] = out.size(i as isize) as i64;
    }

    let mut new_strides: [i64; K_TENSOR_DIMENSION_LIMIT] = [0; K_TENSOR_DIMENSION_LIMIT];
    let mut shift: usize = 0;
    let in_dim: usize = in_.dim() as usize;
    for d in 0..in_dim {
        if (d as i64) == dim1 || (d as i64) == dim2 {
            shift += 1;
        } else {
            new_strides[d - shift] = (*in_.strides().at(d)) as i64;
        }
    }
    new_strides[in_dim - 2] =
        (*in_.strides().at(dim1 as usize)) as i64 + (*in_.strides().at(dim2 as usize)) as i64;

    as_strided_copy::<CTYPE>(
        in_,
        ArrayRef::from_raw_parts(new_sizes.as_ptr(), new_ndim),
        ArrayRef::from_raw_parts(new_strides.as_ptr(), new_ndim),
        storage_offset,
        out,
    );
}

// [spec:et:def:op-diagonal-copy.torch.executor.native.diagonal-copy-out-fn]
// [spec:et:sem:op-diagonal-copy.torch.executor.native.diagonal-copy-out-fn]
pub fn diagonal_copy_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    in_: &Tensor,
    offset: i64,
    mut dim1: i64,
    mut dim2: i64,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    crate::et_kernel_check!(
        ctx,
        check_diagonal_copy_args(in_, dim1, dim2, out),
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

    if dim1 < 0 {
        dim1 += nonzero_dim(in_) as i64;
    }
    if dim2 < 0 {
        dim2 += nonzero_dim(in_) as i64;
    }

    let mut expected_out_size: [SizesType; K_TENSOR_DIMENSION_LIMIT] =
        [0; K_TENSOR_DIMENSION_LIMIT];
    let mut expected_out_dim: usize = 0;
    unsafe {
        get_diagonal_copy_out_target_size(
            in_,
            offset,
            dim1,
            dim2,
            expected_out_size.as_mut_ptr(),
            &mut expected_out_dim,
        );
    }

    crate::et_kernel_check!(
        ctx,
        resize_tensor_same_type(
            out,
            ArrayRef::from_raw_parts(expected_out_size.as_ptr(), expected_out_dim)
        ) == Error::Ok,
        InvalidArgument,
        out
    );

    let op_name = "diagonal_copy.out";

    crate::et_switch_all_types!(in_.scalar_type(), ctx, op_name, CTYPE, {
        diagonal_copy_impl::<CTYPE>(in_, offset, dim1, dim2, out);
    });

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assert_tensor_close;
    use crate::runtime::core::exec_aten::testing_util::tensor_factory::{
        FactoryValue, TensorFactory,
    };
    use crate::runtime::core::exec_aten::util::scalar_type_util::CppTypeToScalarType;
    use crate::runtime::core::memory_allocator::{MemoryAllocator, MemoryAllocatorBase};
    use crate::runtime::core::portable_type::{
        Complex, ComplexDouble, ComplexFloat, ComplexHalf, Half,
    };

    fn setup() {
        crate::runtime::platform::platform::pal_init();
    }

    fn context() -> KernelRuntimeContext<'static> {
        setup();
        KernelRuntimeContext::new(
            crate::extension::module::module::null_event_tracer(),
            core::ptr::null_mut::<MemoryAllocator>() as *mut dyn MemoryAllocatorBase,
        )
    }

    fn op_diagonal_copy_out<'a, 'b>(
        input: &Tensor,
        offset: i64,
        dim1: i64,
        dim2: i64,
        out: &'a Tensor<'b>,
    ) -> &'a Tensor<'b> {
        let mut ctx = context();
        diagonal_copy_out(&mut ctx, input, offset, dim1, dim2, out)
    }

    trait FromI32: Copy {
        fn from_i32(v: i32) -> Self;
    }
    macro_rules! impl_from_i32_num {
        ($($t:ty),*) => {$(impl FromI32 for $t { fn from_i32(v: i32) -> Self { v as $t } })*};
    }
    impl_from_i32_num!(u8, i8, i16, i32, i64, f32, f64);
    impl FromI32 for Half {
        fn from_i32(v: i32) -> Self {
            Half::from_f32(v as f32)
        }
    }
    impl FromI32 for crate::runtime::core::portable_type::BFloat16 {
        fn from_i32(v: i32) -> Self {
            crate::runtime::core::portable_type::BFloat16::from_f32(v as f32)
        }
    }

    fn test_2d_dtype<T>()
    where
        T: CppTypeToScalarType + FactoryValue + FromI32,
    {
        let tf = TensorFactory::<T>::new();
        let input = tf.make_default(vec![3, 4], (1..=12).map(T::from_i32).collect());
        let out = tf.zeros_default(vec![2]);
        let out_expected = tf.make_default(vec![2], vec![T::from_i32(5), T::from_i32(10)]);
        op_diagonal_copy_out(&input, 1, 1, 0, &out);
        assert_tensor_close!(out, out_expected);
    }

    fn run_2d_complex_dtype<R>(mk: impl Fn(f64, f64) -> Complex<R>)
    where
        R: FactoryValue,
        Complex<R>: CppTypeToScalarType + FactoryValue,
    {
        let tf = TensorFactory::<Complex<R>>::new();
        let input = tf.make_default(
            vec![3, 4],
            (1..=12).map(|i| mk(i as f64, i as f64)).collect(),
        );
        let out = tf.make_default(vec![2], vec![mk(0.0, 0.0), mk(0.0, 0.0)]);
        let out_expected = tf.make_default(vec![2], vec![mk(5.0, 5.0), mk(10.0, 10.0)]);
        op_diagonal_copy_out(&input, 1, 1, 0, &out);
        assert_tensor_close!(out, out_expected);
    }

    // [spec:et:sem:op-diagonal-copy.torch.executor.native.diagonal-copy-out-fn/test]
    // also verifies check_diagonal_copy_args (arg gate) and
    // get_diagonal_copy_out_target_size (offset>=0 diagonal-size branch, out {2}).
    // Extracting the offset-1 diagonal [5,10] pins diagonal_copy_impl's strided read.
    // [spec:et:sem:copy-ops-util.torch.executor.check-diagonal-copy-args-fn/test]
    // [spec:et:sem:copy-ops-util.torch.executor.get-diagonal-copy-out-target-size-fn/test]
    // [spec:et:sem:op-diagonal-copy.torch.executor.native.diagonal-copy-impl-fn/test]
    #[test]
    fn op_diagonal_copy_out_test_smoke_test2_d() {
        test_2d_dtype::<u8>();
        test_2d_dtype::<i8>();
        test_2d_dtype::<i16>();
        test_2d_dtype::<i32>();
        test_2d_dtype::<i64>();
        test_2d_dtype::<f32>();
        test_2d_dtype::<f64>();
        test_2d_dtype::<Half>();
        test_2d_dtype::<crate::runtime::core::portable_type::BFloat16>();
    }

    // [spec:et:sem:op-diagonal-copy.torch.executor.native.diagonal-copy-out-fn/test]
    #[test]
    fn op_diagonal_copy_out_test_complex_smoke_test2_d() {
        run_2d_complex_dtype::<Half>(|re, im| ComplexHalf {
            real: Half::from_f64(re),
            imag: Half::from_f64(im),
        });
        run_2d_complex_dtype::<f32>(|re, im| ComplexFloat {
            real: re as f32,
            imag: im as f32,
        });
        run_2d_complex_dtype::<f64>(|re, im| ComplexDouble { real: re, imag: im });
    }

    // [spec:et:sem:op-diagonal-copy.torch.executor.native.diagonal-copy-out-fn/test]
    #[test]
    fn op_diagonal_copy_out_test_smoke_test3_d() {
        let tf_float = TensorFactory::<f32>::new();
        let input = tf_float.make_default(
            vec![2, 3, 2],
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
        );
        let out = tf_float.zeros_default(vec![3, 1]);
        let out_expected = tf_float.make_default(vec![3, 1], vec![7.0, 9.0, 11.0]);
        op_diagonal_copy_out(&input, -1, 0, -1, &out);
        assert_tensor_close!(out, out_expected);
    }

    // [spec:et:sem:op-diagonal-copy.torch.executor.native.diagonal-copy-out-fn/test]
    #[test]
    fn op_diagonal_copy_out_test_smoke_test4_d() {
        let tf_float = TensorFactory::<f32>::new();
        let input = tf_float.make_default(
            vec![2, 1, 2, 3],
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
        );
        let out = tf_float.zeros_default(vec![1, 3, 2]);
        let out_expected =
            tf_float.make_default(vec![1, 3, 2], vec![1.0, 10.0, 2.0, 11.0, 3.0, 12.0]);
        op_diagonal_copy_out(&input, 0, 0, 2, &out);
        assert_tensor_close!(out, out_expected);
    }
}
