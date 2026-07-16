//! Literal port of kernels/portable/cpu/op_split_with_sizes_copy.cpp.

use crate::kernels::portable::cpu::util::broadcast_util::linearize_access_indexes;
use crate::kernels::portable::cpu::util::copy_ops_util::check_split_with_sizes_copy_args;
use crate::kernels::portable::cpu::util::delinearize_index::delinearize_index_tensor;
use crate::kernels::portable::cpu::util::dtype_util::internal::StaticCast;
use crate::runtime::core::array_ref::ArrayRef;
use crate::runtime::core::error::Error;
use crate::runtime::core::exec_aten::util::tensor_util::{
    K_TENSOR_DIMENSION_LIMIT, getLeadingDims, getTrailingDims, resize_tensor,
    tensors_have_same_dim_order2,
};
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::core::portable_type::tensor_impl::{SizesType, StridesType, ssize_t};
use crate::runtime::core::portable_type::{BFloat16, Half};
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

// PORT-NOTE: `TensorList` (kernel_includes.h) is
// `executorch::aten::ArrayRef<Tensor>`.
type TensorList<'a> = ArrayRef<Tensor<'a>>;

// PORT-NOTE: local port of the scalar_type_util `convert<To, From>` template
// (floating source -> integral dest via int64; else plain static_cast).
// Mirrors op_unbind_copy / op_split_copy.
trait Convert<From> {
    fn convert(val: From) -> Self;
}

macro_rules! impl_convert_plain {
    ($to:ty, $from:ty) => {
        impl Convert<$from> for $to {
            #[inline]
            fn convert(val: $from) -> Self {
                <$to as StaticCast<$from>>::static_cast(val)
            }
        }
    };
}
macro_rules! impl_convert_float_to_int {
    ($to:ty, $from:ty) => {
        impl Convert<$from> for $to {
            #[inline]
            fn convert(val: $from) -> Self {
                <$to as StaticCast<i64>>::static_cast(<i64 as StaticCast<$from>>::static_cast(val))
            }
        }
    };
}

macro_rules! impl_convert_row_plain {
    ($to:ty) => {
        impl_convert_plain!($to, u8);
        impl_convert_plain!($to, i8);
        impl_convert_plain!($to, i16);
        impl_convert_plain!($to, i32);
        impl_convert_plain!($to, i64);
        impl_convert_plain!($to, bool);
        impl_convert_plain!($to, f32);
        impl_convert_plain!($to, f64);
        impl_convert_plain!($to, Half);
        impl_convert_plain!($to, BFloat16);
    };
}
impl_convert_row_plain!(f32);
impl_convert_row_plain!(f64);
impl_convert_row_plain!(Half);
impl_convert_row_plain!(BFloat16);

macro_rules! impl_convert_row_int {
    ($to:ty) => {
        impl_convert_plain!($to, u8);
        impl_convert_plain!($to, i8);
        impl_convert_plain!($to, i16);
        impl_convert_plain!($to, i32);
        impl_convert_plain!($to, i64);
        impl_convert_plain!($to, bool);
        impl_convert_float_to_int!($to, f32);
        impl_convert_float_to_int!($to, f64);
        impl_convert_float_to_int!($to, Half);
        impl_convert_float_to_int!($to, BFloat16);
    };
}
impl_convert_row_int!(u8);
impl_convert_row_int!(i8);
impl_convert_row_int!(i16);
impl_convert_row_int!(i32);
impl_convert_row_int!(i64);
impl_convert_row_int!(bool);

// [spec:et:def:op-split-with-sizes-copy.torch.executor.native.split-with-sizes-copy-out-fn]
// [spec:et:sem:op-split-with-sizes-copy.torch.executor.native.split-with-sizes-copy-out-fn]
#[executorch_macros::et_kernel("aten::split_with_sizes_copy.out")]
pub fn split_with_sizes_copy_out(
    ctx: &mut KernelRuntimeContext,
    in_: &Tensor,
    split_sizes: ArrayRef<i64>,
    mut dim: i64,
    out: TensorList,
) {
    let _ = &ctx;
    // Support python-style negative indexing. Note that this op does not accept 0
    // dimensional input tensors.
    if dim < 0 {
        dim += in_.dim() as i64;
    }

    crate::et_kernel_check!(
        ctx,
        check_split_with_sizes_copy_args(in_, split_sizes, dim, out),
        InvalidArgument
    );

    // All output tensors must have the same dim order as the input
    for i in 0..out.size() {
        crate::et_kernel_check!(
            ctx,
            tensors_have_same_dim_order2(in_, out.at(i)),
            InvalidArgument
        );
    }

    // If out is empty, then nothing needs to be done after checking the args.
    // Valid args implies that in.size(dim) == 0 and split_sizes is also empty.
    if out.size() == 0 {
        return;
    }

    // Check that all chunks broadcast to their respective out tensor
    let mut target_out_sizes: [SizesType; K_TENSOR_DIMENSION_LIMIT] = [0; K_TENSOR_DIMENSION_LIMIT];
    let target_out_ndim: usize = in_.dim() as usize;

    for d in 0..in_.dim() {
        target_out_sizes[d as usize] = in_.size(d) as SizesType;
    }

    for i in 0..split_sizes.size() {
        target_out_sizes[dim as usize] = *split_sizes.at(i) as SizesType;
        crate::et_kernel_check!(
            ctx,
            resize_tensor(
                out.at(i),
                ArrayRef::from_raw_parts(target_out_sizes.as_ptr(), target_out_ndim)
            ) == Error::Ok,
            InvalidArgument
        );
    }

    let leading_dims: usize = getLeadingDims(in_, dim);
    let trailing_dims: usize = getTrailingDims(in_, dim);
    let step: usize = in_.size(dim as ssize_t) as usize * trailing_dims;

    let in_type = in_.scalar_type();
    let out_type = out.at(0).scalar_type();

    crate::et_switch_realhbbf16_types!(in_type, ctx, "split_with_sizes_copy_out", CTYPE_IN, {
        crate::et_switch_realhbbf16_types!(
            out_type,
            ctx,
            "split_with_sizes_copy_out",
            CTYPE_OUT,
            {
                let mut in_data: *const CTYPE_IN = in_.const_data_ptr::<CTYPE_IN>();

                // Iterate through list of out tensors
                for i in 0..out.size() {
                    let out_tensor: &Tensor = out.at(i);

                    // If out tensor is empty, no action is required
                    if out_tensor.numel() == 0 {
                        continue;
                    }

                    let chunk_step: usize = *split_sizes.at(i) as usize * trailing_dims;

                    // Update target out shape
                    target_out_sizes[dim as usize] = *split_sizes.at(i) as SizesType;
                    let target_shape: ArrayRef<SizesType> =
                        ArrayRef::from_raw_parts(target_out_sizes.as_ptr(), target_out_ndim);

                    // Check if output involves broadcasting
                    let is_broadcasted: bool = !out_tensor.sizes().equals(target_shape);

                    let mut out_data: *mut CTYPE_OUT = out_tensor.mutable_data_ptr::<CTYPE_OUT>();

                    // Simpler logic if there's no broadcasting
                    if !is_broadcasted {
                        let mut src: *const CTYPE_IN = in_data;
                        for _j in 0..leading_dims {
                            for k in 0..chunk_step {
                                unsafe {
                                    *out_data.add(k) =
                                        <CTYPE_OUT as Convert<CTYPE_IN>>::convert(*src.add(k));
                                }
                            }
                            src = unsafe { src.add(step) };
                            out_data = unsafe { out_data.add(chunk_step) };
                        }
                    } else {
                        // Otherwise, we need to do a copy with broadcasting
                        // Compute target strides
                        let mut target_out_strides: [StridesType; K_TENSOR_DIMENSION_LIMIT] =
                            [0; K_TENSOR_DIMENSION_LIMIT];
                        target_out_strides[(in_.dim() - 1) as usize] = 1;
                        let mut d: ssize_t = in_.dim() - 2;
                        while d >= 0 {
                            target_out_strides[d as usize] = target_out_strides[(d + 1) as usize]
                                * (target_out_sizes[(d + 1) as usize] as StridesType);
                            d -= 1;
                        }
                        let target_strides: ArrayRef<StridesType> =
                            ArrayRef::from_raw_parts(target_out_strides.as_ptr(), target_out_ndim);

                        // For each element in the out tensor, find its corresponding
                        // index in the input tensor and copy it over
                        for ix in 0..out_tensor.numel() {
                            let mut out_coord: [usize; K_TENSOR_DIMENSION_LIMIT] =
                                [0; K_TENSOR_DIMENSION_LIMIT];
                            delinearize_index_tensor(
                                ix as usize,
                                out_tensor,
                                out_coord.as_mut_ptr(),
                                K_TENSOR_DIMENSION_LIMIT,
                            );

                            let in_linear_index: usize = linearize_access_indexes(
                                ArrayRef::from_raw_parts(
                                    out_coord.as_ptr(),
                                    out_tensor.dim() as usize,
                                ),
                                out_tensor.dim(),
                                target_shape,
                                target_strides,
                            );

                            unsafe {
                                *out_data.add(ix as usize) =
                                    <CTYPE_OUT as Convert<CTYPE_IN>>::convert(
                                        *in_data.add(in_linear_index),
                                    );
                            }
                        }
                    }

                    // Move input data pointer
                    in_data = unsafe { in_data.add(chunk_step) };
                }
            }
        );
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assert_tensor_lists_close;
    use crate::runtime::core::exec_aten::testing_util::tensor_factory::TensorFactory;
    use crate::runtime::core::tensor_shape_dynamism::TensorShapeDynamism;

    fn context() -> KernelRuntimeContext<'static> {
        crate::runtime::platform::platform::pal_init();
        KernelRuntimeContext::new(
            crate::extension::module::module::null_event_tracer(),
            core::ptr::null_mut::<crate::runtime::core::memory_allocator::MemoryAllocator>()
                as *mut dyn crate::runtime::core::memory_allocator::MemoryAllocatorBase,
        )
    }

    // Wraps a `&[Tensor]` in the non-owning `ArrayRef<Tensor>` the kernel expects.
    fn tensor_list<'t>(v: &'t [Tensor<'t>]) -> ArrayRef<Tensor<'t>> {
        ArrayRef::from_raw_parts(v.as_ptr(), v.len())
    }

    fn test_tensor_shape_dynamism(dynamism: TensorShapeDynamism) {
        let tf_float = TensorFactory::<f32>::new();

        let self_ = tf_float.make_default(
            vec![2, 6, 3],
            vec![
                -31.25, -92.75, -39.75, -3.25, 53.875, 88.25, -0.625, -1.125, 14.75, 42.0, 89.875,
                -21.125, -8.0, -64.125, 23.0, 37.0, 46.125, -83.25, -58.125, 19.625, -71.125,
                64.75, -1.375, -83.5, -61.375, 13.125, 28.625, -94.0, -67.0, -8.625, -88.875,
                -79.125, 0.375, -61.375, 65.0, -99.375,
            ],
        );
        let split_sizes_vec: Vec<i64> = vec![3, 1, 2];
        let split_sizes: ArrayRef<i64> =
            ArrayRef::from_raw_parts(split_sizes_vec.as_ptr(), split_sizes_vec.len());
        let dim: i64 = 1;

        let out_vec: Vec<Tensor> = if dynamism == TensorShapeDynamism::STATIC {
            vec![
                tf_float.zeros_default(vec![2, 3, 3]),
                tf_float.zeros_default(vec![2, 1, 3]),
                tf_float.zeros_default(vec![2, 2, 3]),
            ]
        } else {
            vec![
                tf_float.zeros(vec![2, 3, 10], TensorShapeDynamism::DYNAMIC_BOUND),
                tf_float.zeros(vec![2, 1, 10], TensorShapeDynamism::DYNAMIC_BOUND),
                tf_float.zeros(vec![2, 2, 10], TensorShapeDynamism::DYNAMIC_BOUND),
            ]
        };
        let out = tensor_list(&out_vec);

        let out_expected_vec: Vec<Tensor> = vec![
            tf_float.make_default(
                vec![2, 3, 3],
                vec![
                    -31.25, -92.75, -39.75, -3.25, 53.875, 88.25, -0.625, -1.125, 14.75, -58.125,
                    19.625, -71.125, 64.75, -1.375, -83.5, -61.375, 13.125, 28.625,
                ],
            ),
            tf_float.make_default(
                vec![2, 1, 3],
                vec![42.0, 89.875, -21.125, -94.0, -67.0, -8.625],
            ),
            tf_float.make_default(
                vec![2, 2, 3],
                vec![
                    -8.0, -64.125, 23.0, 37.0, 46.125, -83.25, -88.875, -79.125, 0.375, -61.375,
                    65.0, -99.375,
                ],
            ),
        ];

        let mut ctx = context();
        split_with_sizes_copy_out(&mut ctx, &self_, split_sizes, dim, out);
        assert_tensor_lists_close!(out_vec, out_expected_vec);
    }

    // [spec:et:sem:op-split-with-sizes-copy.torch.executor.native.split-with-sizes-copy-out-fn/test]
    // also verifies check_split_with_sizes_copy_args (rank/dim gate, split_sizes
    // length == out.size(), each non-negative, sum == in.size(dim))
    // [spec:et:sem:copy-ops-util.torch.executor.check-split-with-sizes-copy-args-fn/test]
    #[test]
    fn op_split_with_sizes_copy_out_test_sanity_check_dim1() {
        test_tensor_shape_dynamism(TensorShapeDynamism::STATIC);
    }

    // [spec:et:sem:op-split-with-sizes-copy.torch.executor.native.split-with-sizes-copy-out-fn/test]
    #[test]
    fn op_split_with_sizes_copy_out_test_dynamic_shape() {
        test_tensor_shape_dynamism(TensorShapeDynamism::DYNAMIC_BOUND);
    }
}
