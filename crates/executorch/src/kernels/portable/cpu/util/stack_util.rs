//! Literal port of kernels/portable/cpu/util/stack_util.cpp.

use crate::extension::tensor::tensor_ptr::NumericCast;
use crate::runtime::core::array_ref::ArrayRef;
use crate::runtime::core::error::Error;
use crate::runtime::core::exec_aten::util::tensor_util::{
    K_TENSOR_DIMENSION_LIMIT, getLeadingDims, getTrailingDims, resize_tensor_same_type,
    tensor_is_default_dim_order, tensors_have_same_dim_order2,
};
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::core::portable_type::tensor_impl::SizesType;
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

use crate::kernels::portable::cpu::util::copy_ops_util::{
    check_stack_args, get_stack_out_target_size,
};

// [spec:et:def:stack-util.torch.executor.native.utils.stack-out-shape-fn]
// [spec:et:sem:stack-util.torch.executor.native.utils.stack-out-shape-fn]
//
// PORT-NOTE: C++ returns `std::tuple<Error, std::array<SizesType,
// kTensorDimensionLimit>, size_t>`. Ported as the equivalent Rust tuple.
pub fn stack_out_shape(
    tensors: ArrayRef<Tensor>,
    dim: i64,
) -> (Error, [SizesType; K_TENSOR_DIMENSION_LIMIT], usize) {
    let mut out_sizes: [SizesType; K_TENSOR_DIMENSION_LIMIT] = [0; K_TENSOR_DIMENSION_LIMIT];
    let mut out_dim: usize = 0;

    // Check if tensors array is empty
    if tensors.size() == 0 {
        return (Error::InvalidArgument, out_sizes, out_dim);
    }

    // Normalize negative dimension
    let mut normalized_dim: i64 = dim;
    if normalized_dim < 0 {
        normalized_dim += tensors.at(0).dim() as i64 + 1;
    }

    // Check if dimension is valid
    if normalized_dim < 0 || normalized_dim > tensors.at(0).dim() as i64 {
        return (Error::InvalidArgument, out_sizes, out_dim);
    }

    // Check that all tensors have the same shape
    for i in 1..tensors.size() {
        if tensors.at(i).dim() != tensors.at(0).dim() {
            return (Error::InvalidArgument, out_sizes, out_dim);
        }
        for d in 0..tensors.at(0).dim() {
            if tensors.at(i).size(d) != tensors.at(0).size(d) {
                return (Error::InvalidArgument, out_sizes, out_dim);
            }
        }
    }

    // Compute output shape using the existing utility
    unsafe {
        get_stack_out_target_size(
            tensors,
            normalized_dim,
            out_sizes.as_mut_ptr(),
            &mut out_dim,
        );
    }

    (Error::Ok, out_sizes, out_dim)
}

// [spec:et:def:stack-util.torch.executor.native.utils.stack-out-impl-fn]
// [spec:et:sem:stack-util.torch.executor.native.utils.stack-out-impl-fn]
//
// PORT-NOTE: C++ returns `Tensor&`; the ported non-owning `Tensor` handle is
// returned by value (it is a cheap view over the same `TensorImpl`), mirroring
// the "returns `out`" contract.
pub fn stack_out_impl<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    tensors: ArrayRef<Tensor>,
    mut dim: i64,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    if dim < 0 {
        dim += out.dim() as i64;
    }

    crate::et_kernel_check!(
        ctx,
        check_stack_args(tensors, dim, out),
        InvalidArgument,
        out
    );

    for i in 0..tensors.size() {
        crate::et_kernel_check!(
            ctx,
            tensors_have_same_dim_order2(tensors.at(i), out),
            InvalidArgument,
            out
        );
    }

    crate::et_kernel_check!(ctx, tensor_is_default_dim_order(out), InvalidArgument, out);

    let mut expected_out_size: [SizesType; K_TENSOR_DIMENSION_LIMIT] =
        [0; K_TENSOR_DIMENSION_LIMIT];
    let mut expected_out_dim: usize = 0;
    unsafe {
        get_stack_out_target_size(
            tensors,
            dim,
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

    let outer: usize = getLeadingDims(out, dim);
    let inner: usize = getTrailingDims(out, dim);
    let ninputs: usize = tensors.size();

    let out_type = out.scalar_type();
    crate::et_switch_realhbbf16_types!(out_type, ctx, "stack.out", CTYPE_OUT, {
        let mut out_ptr: *mut CTYPE_OUT = out.mutable_data_ptr::<CTYPE_OUT>();
        for i in 0..outer {
            for j in 0..ninputs {
                let in_type = tensors.at(j).scalar_type();
                crate::et_switch_realhbbf16_types!(in_type, ctx, "stack.out", CTYPE_IN, {
                    let in_ptr: *const CTYPE_IN =
                        unsafe { tensors.at(j).const_data_ptr::<CTYPE_IN>().add(i * inner) };

                    for k in 0..inner {
                        unsafe {
                            *out_ptr.add(k) =
                                <CTYPE_IN as NumericCast<CTYPE_OUT>>::numeric_cast(*in_ptr.add(k));
                        }
                    }
                    out_ptr = unsafe { out_ptr.add(inner) };
                });
            }
        }
    });

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::core::exec_aten::testing_util::tensor_factory::TensorFactory;

    fn list<'t>(v: &'t [Tensor]) -> ArrayRef<Tensor<'t>> {
        ArrayRef::from_raw_parts(v.as_ptr(), v.len())
    }

    // PORT-NOTE: `stack_out_shape` has no ported op-level caller (op_stack routes
    // through `stack_out_impl`), so it is pinned directly against the C++
    // semantics: for N tensors stacked at `dim`, out_ndim = dim0 + 1 and
    // out_sizes inserts `N` at `dim` (get_stack_out_target_size). Negative dims
    // are normalized by `+ dim0 + 1`; empty list, out-of-range dims, and shape
    // mismatches all return InvalidArgument.
    // [spec:et:sem:stack-util.torch.executor.native.utils.stack-out-shape-fn/test]
    #[test]
    fn stack_out_shape_computes_target() {
        let tf = TensorFactory::<f32>::new();

        let x = tf.ones_default(vec![3, 4]);
        let y = tf.zeros_default(vec![3, 4]);
        let inputs = vec![x, y];
        let ts = list(&inputs);

        // Insert at the middle dim: [3, 4] x2 -> [3, 2, 4].
        let (err, sizes, ndim) = stack_out_shape(ts, 1);
        assert_eq!(err, Error::Ok);
        assert_eq!(ndim, 3);
        assert_eq!(&sizes[..ndim], &[3, 2, 4]);

        // Insert at the front dim -> [2, 3, 4].
        let (err, sizes, ndim) = stack_out_shape(ts, 0);
        assert_eq!(err, Error::Ok);
        assert_eq!(ndim, 3);
        assert_eq!(&sizes[..ndim], &[2, 3, 4]);

        // Negative dim -1 normalizes to dim0 + 1 = 2 (append), -> [3, 4, 2].
        let (err, sizes, ndim) = stack_out_shape(ts, -1);
        assert_eq!(err, Error::Ok);
        assert_eq!(ndim, 3);
        assert_eq!(&sizes[..ndim], &[3, 4, 2]);
    }

    // [spec:et:sem:stack-util.torch.executor.native.utils.stack-out-shape-fn/test]
    #[test]
    fn stack_out_shape_rejects_invalid() {
        let tf = TensorFactory::<f32>::new();

        // Empty tensor list -> InvalidArgument.
        let empty: [Tensor; 0] = [];
        let (err, _sizes, ndim) = stack_out_shape(list(&empty), 0);
        assert_eq!(err, Error::InvalidArgument);
        assert_eq!(ndim, 0);

        // dim out of range (> dim0) -> InvalidArgument.
        let x = tf.ones_default(vec![3, 4]);
        let y = tf.zeros_default(vec![3, 4]);
        let inputs = vec![x, y];
        let (err, _sizes, ndim) = stack_out_shape(list(&inputs), 3);
        assert_eq!(err, Error::InvalidArgument);
        assert_eq!(ndim, 0);

        // Mismatched shapes -> InvalidArgument.
        let a = tf.ones_default(vec![3, 4]);
        let b = tf.zeros_default(vec![3, 5]);
        let mixed = vec![a, b];
        let (err, _sizes, ndim) = stack_out_shape(list(&mixed), 0);
        assert_eq!(err, Error::InvalidArgument);
        assert_eq!(ndim, 0);
    }
}
