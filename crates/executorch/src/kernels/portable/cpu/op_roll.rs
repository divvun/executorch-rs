//! Literal port of kernels/portable/cpu/op_roll.cpp.

use crate::runtime::core::array_ref::{ArrayRef, IntArrayRef};
use crate::runtime::core::error::Error;
use crate::runtime::core::exec_aten::util::tensor_util::{
    K_TENSOR_DIMENSION_LIMIT, coordinateToIndex, dim_is_valid, indexToCoordinate,
    resize_tensor_same_type, tensor_has_rank_greater_or_equal_to, tensors_have_same_dim_order2,
    tensors_have_same_dtype2,
};
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

// PORT-NOTE: local `et_log_and_return_if_false!` mirroring the C++
// `ET_LOG_AND_RETURN_IF_FALSE(cond)` (the crate-level check macro drops format
// args); same reasoning as repeat_util.rs / padding_util.rs.
macro_rules! et_log_and_return_if_false {
    ($cond:expr) => {{
        if !($cond) {
            $crate::et_log!(Error, "Check failed ({}): ", ::core::stringify!($cond));
            return false;
        }
    }};
}

// [spec:et:def:op-roll.torch.executor.native.check-roll-args-fn]
// [spec:et:sem:op-roll.torch.executor.native.check-roll-args-fn]
fn check_roll_args(in_: &Tensor, shifts: IntArrayRef, dims: IntArrayRef, out: &Tensor) -> bool {
    et_log_and_return_if_false!(tensor_has_rank_greater_or_equal_to(in_, 1));
    if in_.numel() > 0 {
        for i in 0..dims.size() {
            let d = *dims.at(i);
            et_log_and_return_if_false!(dim_is_valid(d, in_.dim() as i64));
        }
    }
    et_log_and_return_if_false!(!shifts.empty());
    et_log_and_return_if_false!(shifts.size() == dims.size());
    et_log_and_return_if_false!(tensors_have_same_dtype2(in_, out));
    true
}

// [spec:et:def:op-roll.torch.executor.native.unshift-flat-ix-fn]
// [spec:et:sem:op-roll.torch.executor.native.unshift-flat-ix-fn]
fn unshift_flat_ix(ix: usize, in_: &Tensor, dim_shifts: IntArrayRef) -> usize {
    let mut ix_coord: [usize; K_TENSOR_DIMENSION_LIMIT] = [0; K_TENSOR_DIMENSION_LIMIT];
    unsafe {
        indexToCoordinate(in_, ix, ix_coord.as_mut_ptr());
    }

    let mut shifted_coord: [usize; K_TENSOR_DIMENSION_LIMIT] = [0; K_TENSOR_DIMENSION_LIMIT];
    for d in 0..in_.dim() {
        let in_size_d = in_.size(d) as usize;
        shifted_coord[d as usize] = (ix_coord[d as usize] + in_size_d
            - *dim_shifts.at(d as usize) as usize % in_size_d)
            % in_size_d;
    }

    unsafe { coordinateToIndex(in_, shifted_coord.as_ptr()) }
}

// [spec:et:def:op-roll.torch.executor.native.roll-out-fn]
// [spec:et:sem:op-roll.torch.executor.native.roll-out-fn]
pub fn roll_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    in_: &Tensor,
    shifts: IntArrayRef,
    dims: IntArrayRef,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    crate::et_kernel_check!(
        ctx,
        resize_tensor_same_type(out, in_.sizes()) == Error::Ok,
        InvalidArgument,
        out
    );

    crate::et_kernel_check!(
        ctx,
        check_roll_args(in_, shifts, dims, out),
        InvalidArgument,
        out
    );

    crate::et_kernel_check!(
        ctx,
        tensors_have_same_dim_order2(in_, out),
        InvalidArgument,
        out
    );

    if in_.numel() == 0 {
        return out;
    }

    let mut dim_shift_array: [i64; K_TENSOR_DIMENSION_LIMIT] = [0; K_TENSOR_DIMENSION_LIMIT];
    for i in 0..in_.dim() {
        dim_shift_array[i as usize] = 0;
    }
    for i in 0..dims.size() {
        let d = if *dims.at(i) < 0 {
            *dims.at(i) + in_.dim() as i64
        } else {
            *dims.at(i)
        };
        dim_shift_array[d as usize] += *shifts.at(i);
    }

    let dim_shift_array_length: usize = in_.dim() as usize;
    let dim_shifts: IntArrayRef =
        ArrayRef::from_raw_parts(dim_shift_array.as_ptr(), dim_shift_array_length);

    let name = "roll.out";

    crate::et_switch_realhbbf16_types!(in_.scalar_type(), ctx, name, CTYPE, {
        let in_data: *const CTYPE = in_.const_data_ptr::<CTYPE>();
        let out_data: *mut CTYPE = out.mutable_data_ptr::<CTYPE>();

        for ix in 0..out.numel() {
            unsafe {
                *out_data.add(ix as usize) =
                    *in_data.add(unshift_flat_ix(ix as usize, in_, dim_shifts));
            }
        }
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
    use crate::runtime::core::portable_type::{BFloat16, Half};

    fn context() -> KernelRuntimeContext<'static> {
        crate::runtime::platform::runtime::runtime_init();
        KernelRuntimeContext::new(
            crate::extension::module::module::null_event_tracer(),
            core::ptr::null_mut::<MemoryAllocator>() as *mut dyn MemoryAllocatorBase,
        )
    }

    fn op_roll_out<'a, 'b>(
        input: &Tensor,
        shifts: IntArrayRef,
        dims: IntArrayRef,
        out: &'a Tensor<'b>,
    ) -> &'a Tensor<'b> {
        let mut ctx = context();
        roll_out(&mut ctx, input, shifts, dims, out)
    }

    fn ir(v: &[i64]) -> IntArrayRef {
        ArrayRef::from_raw_parts(v.as_ptr(), v.len())
    }

    // PORT-NOTE: local `from_f64` bridge (mirrors the op_add.rs test helper).
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

    fn test_dtype<T>()
    where
        T: CppTypeToScalarType + FactoryValue + FromF64Elem,
    {
        let tf = TensorFactory::<T>::new();

        let input = tf.make_default(
            vec![4, 2],
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
                .iter()
                .map(|&x| T::from_f64(x))
                .collect(),
        );
        let shifts_data: [i64; 2] = [2, 1];
        let shifts = ir(&shifts_data);
        let dims_data: [i64; 2] = [0, 1];
        let dims = ir(&dims_data);
        let out = tf.zeros_default(vec![4, 2]);
        let out_expected = tf.make_default(
            vec![4, 2],
            [6.0, 5.0, 8.0, 7.0, 2.0, 1.0, 4.0, 3.0]
                .iter()
                .map(|&x| T::from_f64(x))
                .collect(),
        );
        op_roll_out(&input, shifts, dims, &out);
        assert_tensor_close!(out, out_expected);
    }

    // [spec:et:sem:op-roll.torch.executor.native.roll-out-fn/test]
    // [spec:et:sem:op-roll.torch.executor.native.check-roll-args-fn/test]
    // [spec:et:sem:op-roll.torch.executor.native.unshift-flat-ix-fn/test]
    #[test]
    fn smoke_test() {
        // ET_FORALL_REALHBF16_TYPES: real types + Half + BFloat16.
        // TODO: enable bool test after #7856 lands.
        test_dtype::<u8>();
        test_dtype::<i8>();
        test_dtype::<i16>();
        test_dtype::<i32>();
        test_dtype::<i64>();
        test_dtype::<f32>();
        test_dtype::<f64>();
        test_dtype::<Half>();
        test_dtype::<BFloat16>();
    }
}
