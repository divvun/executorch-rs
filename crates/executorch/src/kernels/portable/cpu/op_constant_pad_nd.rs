//! Literal port of kernels/portable/cpu/op_constant_pad_nd.cpp.

use crate::kernels::portable::cpu::scalar_utils::internal::check_overflow_scalar_cast;
use crate::kernels::portable::cpu::util::kernel_ops_util::{
    check_constant_pad_args, resize_constant_pad_output,
};
use crate::runtime::core::array_ref::{ArrayRef, IntArrayRef};
use crate::runtime::core::error::Error;
use crate::runtime::core::exec_aten::util::tensor_util::{
    K_TENSOR_DIMENSION_LIMIT, getTrailingDims, tensors_have_same_dim_order2,
};
use crate::runtime::core::portable_type::scalar::Scalar;
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

// [spec:et:def:op-constant-pad-nd.torch.executor.native.set-all-to-value-fn]
// [spec:et:sem:op-constant-pad-nd.torch.executor.native.set-all-to-value-fn]
fn set_all_to_value<CTYPE: Copy>(out_data: *mut CTYPE, step_len: usize, value: CTYPE) {
    for i in 0..step_len {
        unsafe {
            *out_data.add(i) = value;
        }
    }
}

// [spec:et:def:op-constant-pad-nd.torch.executor.native.apply-padding-to-dim-fn]
// [spec:et:sem:op-constant-pad-nd.torch.executor.native.apply-padding-to-dim-fn]
#[allow(clippy::too_many_arguments)]
fn apply_padding_to_dim<CTYPE: Copy>(
    ctx: &mut KernelRuntimeContext,
    ndim: usize,
    mut self_data: *const CTYPE,
    self_sizes: IntArrayRef,
    self_strides: IntArrayRef,
    mut out_data: *mut CTYPE,
    out_data_end: *mut CTYPE,
    out_sizes: IntArrayRef,
    out_strides: IntArrayRef,
    pad: IntArrayRef,
    value: CTYPE,
    last_padded_dim: usize,
    dim: usize,
) {
    if dim >= ndim {
        return;
    }

    let pad_i: usize = ndim - 1 - dim;

    let mut pad_before: usize = 0;
    let mut pad_after: usize = 0;
    if pad_i < pad.size() / 2 {
        let pb: i64 = *pad.at(2 * pad_i);
        let pa: i64 = *pad.at(2 * pad_i + 1);
        crate::et_kernel_check_msg!(
            ctx,
            pb >= 0 && pa >= 0,
            InvalidArgument,
            (),
            "Padding values must be non-negative."
        );
        pad_before = pb as usize;
        pad_after = pa as usize;
    }

    let out_step_len: usize = *out_strides.at(dim) as usize;
    let in_step_len: usize = *self_strides.at(dim) as usize;

    // Do not copy padding beyond the out tensor bounds.
    // Use division to avoid potential overflow in multiplication.
    if pad_before > 0 {
        crate::et_kernel_check_msg!(
            ctx,
            (out_data as *const CTYPE) <= (out_data_end as *const CTYPE),
            InvalidArgument,
            (),
            "Out data pointer exceeds buffer bounds."
        );
        let remaining: usize = unsafe { out_data_end.offset_from(out_data) } as usize;
        crate::et_kernel_check_msg!(
            ctx,
            out_step_len > 0 && remaining / out_step_len >= pad_before,
            InvalidArgument,
            (),
            "Out tensor is too small for the requested padding."
        );
    }
    for _i in 0..pad_before {
        set_all_to_value(out_data, out_step_len, value);
        out_data = unsafe { out_data.add(out_step_len) };
    }

    // If subsequent dims are not padded, then the whole block of memory can be
    // copied.
    if dim >= last_padded_dim {
        let copy_len: usize = in_step_len * (*self_sizes.at(dim) as usize);
        let copy_nbytes: usize = copy_len * core::mem::size_of::<CTYPE>();

        if copy_nbytes > 0 {
            // Check that out_data and self_data do not overlap.
            crate::et_kernel_check_msg!(
                ctx,
                (out_data as *const CTYPE) != self_data
                    && ((unsafe { out_data.add(copy_len) } as *const CTYPE <= self_data)
                        || (unsafe { self_data.add(copy_len) } <= out_data as *const CTYPE)),
                InvalidArgument,
                (),
                "Out tensor overlaps with the input tensor. This is not supported."
            );
            // Bounds check before memcpy
            crate::et_kernel_check_msg!(
                ctx,
                (out_data as *const CTYPE) <= (out_data_end as *const CTYPE),
                InvalidArgument,
                (),
                "Out data pointer exceeds buffer bounds."
            );
            let remaining: usize = unsafe { out_data_end.offset_from(out_data) } as usize;
            crate::et_kernel_check_msg!(
                ctx,
                remaining >= copy_len,
                InvalidArgument,
                (),
                "Out tensor is too small for the copy operation."
            );
            unsafe {
                core::ptr::copy_nonoverlapping(self_data, out_data, copy_len);
            }
            out_data = unsafe { out_data.add(copy_len) };
            self_data = unsafe { self_data.add(copy_len) };
        }
    }
    // Otherwise, call this function recursively
    else {
        for _i in 0..(*self_sizes.at(dim)) {
            apply_padding_to_dim(
                ctx,
                ndim,
                self_data,
                self_sizes,
                self_strides,
                out_data,
                out_data_end,
                out_sizes,
                out_strides,
                pad,
                value,
                last_padded_dim,
                dim + 1,
            );

            if ctx.failure_state() != Error::Ok {
                return;
            }

            out_data = unsafe { out_data.add(out_step_len) };
            self_data = unsafe { self_data.add(in_step_len) };
        }
    }

    // Do not copy padding beyond the out tensor bounds.
    // Use division to avoid potential overflow in multiplication.
    if pad_after > 0 {
        crate::et_kernel_check_msg!(
            ctx,
            (out_data as *const CTYPE) <= (out_data_end as *const CTYPE),
            InvalidArgument,
            (),
            "Out data pointer exceeds buffer bounds."
        );
        let remaining: usize = unsafe { out_data_end.offset_from(out_data) } as usize;
        crate::et_kernel_check_msg!(
            ctx,
            out_step_len > 0 && remaining / out_step_len >= pad_after,
            InvalidArgument,
            (),
            "Out tensor is too small for the requested padding."
        );
    }
    for _i in 0..pad_after {
        set_all_to_value(out_data, out_step_len, value);
        out_data = unsafe { out_data.add(out_step_len) };
    }
}

// [spec:et:def:op-constant-pad-nd.torch.executor.native.constant-pad-nd-out-impl-fn]
// [spec:et:sem:op-constant-pad-nd.torch.executor.native.constant-pad-nd-out-impl-fn]
fn constant_pad_nd_out_impl<CTYPE: Copy>(
    ctx: &mut KernelRuntimeContext,
    self_: &Tensor,
    pad: IntArrayRef,
    value_v: CTYPE,
    out: &Tensor,
) {
    let self_data: *const CTYPE = self_.const_data_ptr::<CTYPE>();
    let out_data: *mut CTYPE = out.mutable_data_ptr::<CTYPE>();

    let ndim: usize = self_.dim() as usize;

    if ndim == 0 {
        unsafe {
            *out_data.add(0) = *self_data.add(0);
        }
        return;
    }

    let mut self_sizes: [i64; K_TENSOR_DIMENSION_LIMIT] = [0; K_TENSOR_DIMENSION_LIMIT];
    let mut self_strides: [i64; K_TENSOR_DIMENSION_LIMIT] = [0; K_TENSOR_DIMENSION_LIMIT];
    let mut out_sizes: [i64; K_TENSOR_DIMENSION_LIMIT] = [0; K_TENSOR_DIMENSION_LIMIT];
    let mut out_strides: [i64; K_TENSOR_DIMENSION_LIMIT] = [0; K_TENSOR_DIMENSION_LIMIT];

    // Collect sizes and strides of input and output tensors and determine the
    // last padded dimension
    let mut last_padded_dim: usize = 0;
    for i in 0..ndim {
        self_sizes[i] = self_.size(i as isize) as i64;
        self_strides[i] = getTrailingDims(self_, i as i64) as i64;
        out_sizes[i] = out.size(i as isize) as i64;
        out_strides[i] = getTrailingDims(out, i as i64) as i64;

        let pad_i: usize = ndim - 1 - i;
        if pad_i < pad.size() / 2 {
            if *pad.at(2 * pad_i) + *pad.at(2 * pad_i + 1) > 0 {
                last_padded_dim = i;
            }
        }
    }

    let self_sizes_ref = ArrayRef::from_raw_parts(self_sizes.as_ptr(), ndim);
    let self_strides_ref = ArrayRef::from_raw_parts(self_strides.as_ptr(), ndim);
    let out_sizes_ref = ArrayRef::from_raw_parts(out_sizes.as_ptr(), ndim);
    let out_strides_ref = ArrayRef::from_raw_parts(out_strides.as_ptr(), ndim);

    let out_data_end: *mut CTYPE = unsafe { out_data.add(out.numel() as usize) };

    apply_padding_to_dim(
        ctx,
        ndim,
        self_data,
        self_sizes_ref,
        self_strides_ref,
        out_data,
        out_data_end,
        out_sizes_ref,
        out_strides_ref,
        pad,
        value_v,
        last_padded_dim,
        0,
    );
}

// [spec:et:def:op-constant-pad-nd.torch.executor.native.constant-pad-nd-out-fn]
// [spec:et:sem:op-constant-pad-nd.torch.executor.native.constant-pad-nd-out-fn]
#[executorch_macros::et_kernel("aten::constant_pad_nd.out")]
pub fn constant_pad_nd_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    in_: &Tensor,
    pad: IntArrayRef,
    value: &Scalar,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    crate::et_kernel_check!(
        ctx,
        check_constant_pad_args(in_, pad, value, out),
        InvalidArgument,
        out
    );

    crate::et_kernel_check!(
        ctx,
        tensors_have_same_dim_order2(in_, out),
        InvalidArgument,
        out
    );

    // resize out tensor for dynamic shapes
    crate::et_kernel_check_msg!(
        ctx,
        resize_constant_pad_output(in_, pad, out) == Error::Ok,
        InvalidArgument,
        out,
        "Failed to resize output tensor."
    );

    let in_type = in_.scalar_type();

    let op_name = "constant_pad_nd.out";

    crate::et_switch_realhbbf16_types!(in_type, ctx, op_name, CTYPE, {
        let opt_value_casted = check_overflow_scalar_cast::<CTYPE>(value);
        crate::et_kernel_check!(ctx, opt_value_casted.is_some(), InvalidArgument, out);
        let value_casted = opt_value_casted.unwrap();
        constant_pad_nd_out_impl::<CTYPE>(ctx, in_, pad, value_casted, out);
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
    use crate::runtime::core::portable_type::tensor_impl::SizesType;
    use crate::runtime::core::portable_type::{BFloat16, Half};

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

    trait FromI32 {
        fn from_i32(v: i32) -> Self;
    }
    macro_rules! impl_from_i32 {
        ($($t:ty),*) => {$(impl FromI32 for $t { fn from_i32(v: i32) -> Self { v as $t } })*};
    }
    impl_from_i32!(u8, i8, i16, i32, i64, f32, f64);
    impl FromI32 for Half {
        fn from_i32(v: i32) -> Self {
            Half::from_f32(v as f32)
        }
    }
    impl FromI32 for BFloat16 {
        fn from_i32(v: i32) -> Self {
            BFloat16::from_f32(v as f32)
        }
    }

    fn ir(v: &[i64]) -> IntArrayRef {
        IntArrayRef::from_raw_parts(v.as_ptr(), v.len())
    }

    // Shared body for the six per-shape padding cases: all use the same `self`
    // tensor {2,4,4} filled with 1..8 blocks and pad value 7.
    fn run_pad_case<T>(sizes_out: Vec<SizesType>, padding: Vec<i64>, expected_data: Vec<i32>)
    where
        T: CppTypeToScalarType + FactoryValue + FromI32,
    {
        let tf = TensorFactory::<T>::new();

        #[rustfmt::skip]
        let self_ = tf.make_default(
            vec![2, 4, 4],
            [
                1, 2, 3, 4,
                5, 6, 7, 8,
                1, 2, 3, 4,
                5, 6, 7, 8,

                1, 2, 3, 4,
                5, 6, 7, 8,
                1, 2, 3, 4,
                5, 6, 7, 8,
            ]
            .iter()
            .map(|&x| T::from_i32(x))
            .collect(),
        );

        let expected = tf.make_default(
            sizes_out.clone(),
            expected_data.iter().map(|&x| T::from_i32(x)).collect(),
        );

        let out = tf.zeros_default(sizes_out);
        let value = Scalar::from_i64(7);

        let mut ctx = context();
        constant_pad_nd_out(&mut ctx, &self_, ir(&padding), &value, &out);
        assert_tensor_close!(out, expected);
    }

    fn test_dim2<T>()
    where
        T: CppTypeToScalarType + FactoryValue + FromI32,
    {
        #[rustfmt::skip]
        let expected = vec![
            7, 1, 2, 3, 4, 7,
            7, 5, 6, 7, 8, 7,
            7, 1, 2, 3, 4, 7,
            7, 5, 6, 7, 8, 7,

            7, 1, 2, 3, 4, 7,
            7, 5, 6, 7, 8, 7,
            7, 1, 2, 3, 4, 7,
            7, 5, 6, 7, 8, 7,
        ];
        run_pad_case::<T>(vec![2, 4, 6], vec![1, 1], expected);
    }

    fn test_dim1<T>()
    where
        T: CppTypeToScalarType + FactoryValue + FromI32,
    {
        #[rustfmt::skip]
        let expected = vec![
            7, 7, 7, 7,
            7, 7, 7, 7,
            1, 2, 3, 4,
            5, 6, 7, 8,
            1, 2, 3, 4,
            5, 6, 7, 8,

            7, 7, 7, 7,
            7, 7, 7, 7,
            1, 2, 3, 4,
            5, 6, 7, 8,
            1, 2, 3, 4,
            5, 6, 7, 8,
        ];
        run_pad_case::<T>(vec![2, 6, 4], vec![0, 0, 2, 0], expected);
    }

    fn test_dim0<T>()
    where
        T: CppTypeToScalarType + FactoryValue + FromI32,
    {
        #[rustfmt::skip]
        let expected = vec![
            7, 7, 7, 7,
            7, 7, 7, 7,
            7, 7, 7, 7,
            7, 7, 7, 7,

            1, 2, 3, 4,
            5, 6, 7, 8,
            1, 2, 3, 4,
            5, 6, 7, 8,

            1, 2, 3, 4,
            5, 6, 7, 8,
            1, 2, 3, 4,
            5, 6, 7, 8,
        ];
        run_pad_case::<T>(vec![3, 4, 4], vec![0, 0, 0, 0, 1, 0], expected);
    }

    fn test_dim12<T>()
    where
        T: CppTypeToScalarType + FactoryValue + FromI32,
    {
        #[rustfmt::skip]
        let expected = vec![
            7, 7, 1, 2, 3, 4, 7,
            7, 7, 5, 6, 7, 8, 7,
            7, 7, 1, 2, 3, 4, 7,
            7, 7, 5, 6, 7, 8, 7,
            7, 7, 7, 7, 7, 7, 7,
            7, 7, 7, 7, 7, 7, 7,

            7, 7, 1, 2, 3, 4, 7,
            7, 7, 5, 6, 7, 8, 7,
            7, 7, 1, 2, 3, 4, 7,
            7, 7, 5, 6, 7, 8, 7,
            7, 7, 7, 7, 7, 7, 7,
            7, 7, 7, 7, 7, 7, 7,
        ];
        run_pad_case::<T>(vec![2, 6, 7], vec![2, 1, 0, 2], expected);
    }

    fn test_dim02<T>()
    where
        T: CppTypeToScalarType + FactoryValue + FromI32,
    {
        #[rustfmt::skip]
        let expected = vec![
            7, 7, 1, 2, 3, 4, 7,
            7, 7, 5, 6, 7, 8, 7,
            7, 7, 1, 2, 3, 4, 7,
            7, 7, 5, 6, 7, 8, 7,

            7, 7, 1, 2, 3, 4, 7,
            7, 7, 5, 6, 7, 8, 7,
            7, 7, 1, 2, 3, 4, 7,
            7, 7, 5, 6, 7, 8, 7,

            7, 7, 7, 7, 7, 7, 7,
            7, 7, 7, 7, 7, 7, 7,
            7, 7, 7, 7, 7, 7, 7,
            7, 7, 7, 7, 7, 7, 7,
        ];
        run_pad_case::<T>(vec![3, 4, 7], vec![2, 1, 0, 0, 0, 1], expected);
    }

    fn test_dim012<T>()
    where
        T: CppTypeToScalarType + FactoryValue + FromI32,
    {
        #[rustfmt::skip]
        let expected = vec![
            7, 7, 7, 7, 7, 7, 7,
            7, 7, 1, 2, 3, 4, 7,
            7, 7, 5, 6, 7, 8, 7,
            7, 7, 1, 2, 3, 4, 7,
            7, 7, 5, 6, 7, 8, 7,

            7, 7, 7, 7, 7, 7, 7,
            7, 7, 1, 2, 3, 4, 7,
            7, 7, 5, 6, 7, 8, 7,
            7, 7, 1, 2, 3, 4, 7,
            7, 7, 5, 6, 7, 8, 7,

            7, 7, 7, 7, 7, 7, 7,
            7, 7, 7, 7, 7, 7, 7,
            7, 7, 7, 7, 7, 7, 7,
            7, 7, 7, 7, 7, 7, 7,
            7, 7, 7, 7, 7, 7, 7,
        ];
        run_pad_case::<T>(vec![3, 5, 7], vec![2, 1, 1, 0, 0, 1], expected);
    }

    fn expect_bad_scalar_value_dies<T>(bad_value: Scalar)
    where
        T: CppTypeToScalarType + FactoryValue + FromI32,
    {
        let tf = TensorFactory::<T>::new();
        let padding = vec![1i64, 1];

        let self_ = tf.ones_default(vec![2, 2]);
        let out = tf.zeros_default(vec![2, 4]);

        let mut ctx = context();
        et_expect_kernel_failure!(
            ctx,
            constant_pad_nd_out(&mut ctx, &self_, ir(&padding), &bad_value, &out)
        );
    }

    macro_rules! forall_realhbf16 {
        ($f:ident) => {{
            $f::<u8>();
            $f::<i8>();
            $f::<i16>();
            $f::<i32>();
            $f::<i64>();
            $f::<f32>();
            $f::<f64>();
            $f::<Half>();
            $f::<BFloat16>();
        }};
    }

    // [spec:et:sem:op-constant-pad-nd.torch.executor.native.constant-pad-nd-out-fn/test]
    // Numeric pad spans the constant-pad arg check and output resize; a wrong
    // expected size or arg check would change the compared output tensor.
    // Placing the value 7 in the padding and copying the interior pins
    // constant_pad_nd_out_impl, apply_padding_to_dim, and set_all_to_value.
    // [spec:et:sem:kernel-ops-util.torch.executor.check-constant-pad-args-fn/test]
    // [spec:et:sem:kernel-ops-util.torch.executor.resize-constant-pad-output-fn/test]
    // [spec:et:sem:op-constant-pad-nd.torch.executor.native.constant-pad-nd-out-impl-fn/test]
    // [spec:et:sem:op-constant-pad-nd.torch.executor.native.apply-padding-to-dim-fn/test]
    // [spec:et:sem:op-constant-pad-nd.torch.executor.native.set-all-to-value-fn/test]
    #[test]
    fn op_constant_pad_nd_out_test_test_pad_dim2() {
        forall_realhbf16!(test_dim2);
    }

    // [spec:et:sem:op-constant-pad-nd.torch.executor.native.constant-pad-nd-out-fn/test]
    #[test]
    fn op_constant_pad_nd_out_test_test_pad_dim1() {
        forall_realhbf16!(test_dim1);
    }

    // [spec:et:sem:op-constant-pad-nd.torch.executor.native.constant-pad-nd-out-fn/test]
    #[test]
    fn op_constant_pad_nd_out_test_test_pad_dim0() {
        forall_realhbf16!(test_dim0);
    }

    // [spec:et:sem:op-constant-pad-nd.torch.executor.native.constant-pad-nd-out-fn/test]
    #[test]
    fn op_constant_pad_nd_out_test_test_pad_dim1_and2() {
        forall_realhbf16!(test_dim12);
    }

    // [spec:et:sem:op-constant-pad-nd.torch.executor.native.constant-pad-nd-out-fn/test]
    #[test]
    fn op_constant_pad_nd_out_test_test_pad_dim0_and2() {
        forall_realhbf16!(test_dim02);
    }

    // [spec:et:sem:op-constant-pad-nd.torch.executor.native.constant-pad-nd-out-fn/test]
    #[test]
    fn op_constant_pad_nd_out_test_test_pad_dim0_and1_and2() {
        forall_realhbf16!(test_dim012);
    }

    // [spec:et:sem:op-constant-pad-nd.torch.executor.native.constant-pad-nd-out-fn/test]
    #[test]
    fn op_constant_pad_nd_out_test_different_input_output_types_fail() {
        let tf = TensorFactory::<f32>::new();
        let tf_out = TensorFactory::<f64>::new();

        let padding = vec![1i64, 1];

        let self_ = tf.ones_default(vec![1, 4, 4]);
        let out = tf_out.zeros_default(vec![1, 4, 6]);

        let value = Scalar::from_i64(0);
        let mut ctx = context();
        et_expect_kernel_failure!(
            ctx,
            constant_pad_nd_out(&mut ctx, &self_, ir(&padding), &value, &out)
        );
    }

    // [spec:et:sem:op-constant-pad-nd.torch.executor.native.constant-pad-nd-out-fn/test]
    #[test]
    fn op_constant_pad_nd_out_test_odd_number_of_padding_elements_fail() {
        let tf = TensorFactory::<f32>::new();

        let padding = vec![1i64, 1, 0];

        let self_ = tf.ones_default(vec![1, 4, 4]);
        let out = tf.zeros_default(vec![1, 4, 4]);

        let value = Scalar::from_i64(0);
        let mut ctx = context();
        et_expect_kernel_failure!(
            ctx,
            constant_pad_nd_out(&mut ctx, &self_, ir(&padding), &value, &out)
        );
    }

    // [spec:et:sem:op-constant-pad-nd.torch.executor.native.constant-pad-nd-out-fn/test]
    #[test]
    fn op_constant_pad_nd_out_test_too_many_padding_elements_fail() {
        let tf = TensorFactory::<f32>::new();

        let padding = vec![3i64, 2, 1, 1, 2, 1, 1, 0];

        let self_ = tf.ones_default(vec![1, 4, 4]);
        let out = tf.zeros_default(vec![1, 4, 4]);

        let value = Scalar::from_i64(0);
        let mut ctx = context();
        et_expect_kernel_failure!(
            ctx,
            constant_pad_nd_out(&mut ctx, &self_, ir(&padding), &value, &out)
        );
    }

    // PORT-NOTE: C++ guards with `ET_SKIP_IF(is_aten, ...)`; non-ATen build runs.
    // [spec:et:sem:op-constant-pad-nd.torch.executor.native.constant-pad-nd-out-fn/test]
    #[test]
    fn op_constant_pad_nd_out_test_incorrect_output_shape_fail() {
        let tf = TensorFactory::<f32>::new();

        let padding = vec![1i64, 1];

        let self_ = tf.ones_default(vec![1, 4, 4]);
        let out = tf.zeros_default(vec![1, 4, 4]);

        let value = Scalar::from_i64(0);
        let mut ctx = context();
        et_expect_kernel_failure!(
            ctx,
            constant_pad_nd_out(&mut ctx, &self_, ir(&padding), &value, &out)
        );
    }

    // GENERATE_SCALAR_OVERFLOW_TESTS(OpConstantPadNDOutTest)
    // [spec:et:sem:op-constant-pad-nd.torch.executor.native.constant-pad-nd-out-fn/test]
    #[test]
    fn op_constant_pad_nd_out_test_byte_tensor_too_large_scalar_dies() {
        expect_bad_scalar_value_dies::<u8>(Scalar::from_i64(256));
    }

    // [spec:et:sem:op-constant-pad-nd.torch.executor.native.constant-pad-nd-out-fn/test]
    #[test]
    fn op_constant_pad_nd_out_test_char_tensor_too_small_scalar_dies() {
        expect_bad_scalar_value_dies::<i8>(Scalar::from_i64(-129));
    }

    // [spec:et:sem:op-constant-pad-nd.torch.executor.native.constant-pad-nd-out-fn/test]
    #[test]
    fn op_constant_pad_nd_out_test_short_tensor_too_large_scalar_dies() {
        expect_bad_scalar_value_dies::<i16>(Scalar::from_i64(32768));
    }

    // [spec:et:sem:op-constant-pad-nd.torch.executor.native.constant-pad-nd-out-fn/test]
    #[test]
    fn op_constant_pad_nd_out_test_float_tensor_too_small_scalar_dies() {
        expect_bad_scalar_value_dies::<f32>(Scalar::from_double(-3.41e+38));
    }

    // [spec:et:sem:op-constant-pad-nd.torch.executor.native.constant-pad-nd-out-fn/test]
    #[test]
    fn op_constant_pad_nd_out_test_float_tensor_too_large_scalar_dies() {
        expect_bad_scalar_value_dies::<f32>(Scalar::from_double(3.41e+38));
    }
}
