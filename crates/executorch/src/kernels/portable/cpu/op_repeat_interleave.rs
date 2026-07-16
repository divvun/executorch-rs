//! Literal port of kernels/portable/cpu/op_repeat_interleave.cpp.

use crate::runtime::core::array_ref::ArrayRef;
use crate::runtime::core::error::Error;
use crate::runtime::core::exec_aten::util::scalar_type_util::to_string;
use crate::runtime::core::exec_aten::util::tensor_util::{
    resize_tensor_same_type, tensor_is_default_dim_order, tensors_have_same_dim_order2,
    tensors_have_same_dtype2,
};
use crate::runtime::core::portable_type::scalar_type::ScalarType;
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::core::portable_type::tensor_impl::SizesType;
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

// PORT-NOTE: the crate-level check macros drop caller format args; these local
// overrides mirror the C++ `ET_CHECK_OR_RETURN_FALSE` / `ET_LOG_AND_RETURN_IF_FALSE`
// faithfully (same as repeat_util.rs).
macro_rules! et_check_or_return_false {
    ($cond:expr, $fmt:literal $(, $($arg:tt)*)?) => {{
        if !($cond) {
            $crate::et_log!(
                Error,
                ::core::concat!("Check failed ({}): ", $fmt),
                ::core::stringify!($cond)
                $(, $($arg)*)?
            );
            return false;
        }
    }};
}

macro_rules! et_log_and_return_if_false {
    ($cond:expr) => {
        et_check_or_return_false!($cond, "")
    };
}

// [spec:et:def:op-repeat-interleave.torch.executor.native.check-repeat-interleave-args-fn]
// [spec:et:sem:op-repeat-interleave.torch.executor.native.check-repeat-interleave-args-fn]
fn check_repeat_interleave_args(
    repeats: &Tensor,
    output_size_value: i64,
    repeats_sum: i64,
    out: &Tensor,
) -> bool {
    et_check_or_return_false!(
        repeats.scalar_type() == ScalarType::Int || repeats.scalar_type() == ScalarType::Long,
        "repeats must be int or long; repeats.scalar_type() = {}",
        to_string(repeats.scalar_type())
    );
    et_check_or_return_false!(
        repeats.dim() == 1,
        "repeats must be 1-D; repeats.dim() = {}",
        repeats.dim()
    );
    et_check_or_return_false!(
        output_size_value == repeats_sum,
        "output_size, if provided, must be equal to repeats.sum(); output_size_value = {}, repeats_sum = {}",
        output_size_value,
        repeats_sum
    );
    et_log_and_return_if_false!(tensors_have_same_dtype2(repeats, out));

    if repeats.scalar_type() == ScalarType::Long {
        let repeats_data: *const i64 = repeats.const_data_ptr::<i64>();
        for i in 0..repeats.numel() {
            et_check_or_return_false!(
                unsafe { *repeats_data.offset(i as isize) } >= 0,
                "repeats cannot be negative; repeats_data[{}] = {}",
                i,
                unsafe { *repeats_data.offset(i as isize) }
            );
        }
    } else {
        let repeats_data: *const i32 = repeats.const_data_ptr::<i32>();
        for i in 0..repeats.numel() {
            et_check_or_return_false!(
                unsafe { *repeats_data.offset(i as isize) } >= 0,
                "repeats cannot be negative; repeats_data[{}] = {}",
                i,
                unsafe { *repeats_data.offset(i as isize) }
            );
        }
    }

    true
}

// [spec:et:def:op-repeat-interleave.torch.executor.native.repeat-interleave-tensor-out-fn]
// [spec:et:sem:op-repeat-interleave.torch.executor.native.repeat-interleave-tensor-out-fn]
pub fn repeat_interleave_Tensor_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    repeats: &Tensor,
    output_size: Option<i64>,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    let mut repeats_sum: i64 = 0;

    let name = "repeat_interleave.Tensor_out";

    crate::et_switch_two_types!(Int, Long, repeats.scalar_type(), ctx, name, CTYPE, {
        let repeats_data: *const CTYPE = repeats.const_data_ptr::<CTYPE>();
        for ix in 0..repeats.numel() {
            repeats_sum += unsafe { *repeats_data.offset(ix as isize) } as i64;
        }
    });

    let output_size_value: i64 = if output_size.is_some() {
        output_size.unwrap()
    } else {
        repeats_sum
    };

    crate::et_kernel_check!(
        ctx,
        check_repeat_interleave_args(repeats, output_size_value, repeats_sum, out),
        InvalidArgument,
        out
    );

    crate::et_kernel_check!(
        ctx,
        tensors_have_same_dim_order2(repeats, out),
        InvalidArgument,
        out
    );

    crate::et_kernel_check!(
        ctx,
        tensor_is_default_dim_order(repeats),
        InvalidArgument,
        out
    );

    let out_size: SizesType = output_size_value as SizesType;
    crate::et_kernel_check_msg!(
        ctx,
        resize_tensor_same_type(out, ArrayRef::from_raw_parts(&out_size, 1)) == Error::Ok,
        InvalidArgument,
        out,
        "Failed to resize output tensor."
    );

    crate::et_switch_two_types!(Int, Long, repeats.scalar_type(), ctx, name, CTYPE, {
        let repeats_data: *const CTYPE = repeats.const_data_ptr::<CTYPE>();
        let out_data: *mut CTYPE = out.mutable_data_ptr::<CTYPE>();
        let mut out_ix: usize = 0;
        for ix in 0..repeats.numel() {
            let mut i: CTYPE = 0;
            while i < unsafe { *repeats_data.offset(ix as isize) } {
                unsafe {
                    *out_data.add(out_ix) = ix as CTYPE;
                }
                i += 1;
                out_ix += 1;
            }
        }
    });

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assert_tensor_eq;
    use crate::runtime::core::exec_aten::testing_util::tensor_factory::TensorFactory;
    use crate::runtime::core::memory_allocator::{MemoryAllocator, MemoryAllocatorBase};

    fn context() -> KernelRuntimeContext<'static> {
        crate::runtime::platform::runtime::runtime_init();
        KernelRuntimeContext::new(
            crate::extension::module::module::null_event_tracer(),
            core::ptr::null_mut::<MemoryAllocator>() as *mut dyn MemoryAllocatorBase,
        )
    }

    fn op_repeat_out<'a, 'b>(
        repeats: &Tensor,
        output_size: Option<i64>,
        out: &'a Tensor<'b>,
    ) -> &'a Tensor<'b> {
        let mut ctx = context();
        repeat_interleave_Tensor_out(&mut ctx, repeats, output_size, out)
    }

    // [spec:et:sem:op-repeat-interleave.torch.executor.native.repeat-interleave-tensor-out-fn/test]
    // also verifies check_repeat_interleave_args (valid path): Int, 1-D, non-negative
    // repeats with output_size==sum(6) must pass so the op writes the expected output;
    // a wrongly-false result would abort and leave the assertions failing.
    // [spec:et:sem:op-repeat-interleave.torch.executor.native.check-repeat-interleave-args-fn/test]
    #[test]
    fn smoke_test() {
        let tf = TensorFactory::<i32>::new();

        let repeats = tf.make_default(vec![3], vec![2, 3, 1]);

        let out = tf.zeros_default(vec![6]);
        let expected = tf.make_default(vec![6], vec![0, 0, 1, 1, 1, 2]);
        let ret = op_repeat_out(&repeats, Some(6), &out);
        assert_tensor_eq!(*ret, out);
        assert_tensor_eq!(*ret, expected);
    }
}
