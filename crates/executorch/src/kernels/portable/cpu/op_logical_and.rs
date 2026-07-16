//! Literal port of kernels/portable/cpu/op_logical_and.cpp.

use crate::kernels::portable::cpu::pattern::logical_op::logical_tensor_out;
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

// [spec:et:def:op-logical-and.torch.executor.native.logical-and-fn]
// [spec:et:sem:op-logical-and.torch.executor.native.logical-and-fn]
fn logical_and(a: bool, b: bool) -> bool {
    a && b
}

// [spec:et:def:op-logical-and.torch.executor.native.logical-and-out-fn]
// [spec:et:sem:op-logical-and.torch.executor.native.logical-and-out-fn]
pub fn logical_and_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    a: &Tensor,
    b: &Tensor,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    // @lint-ignore CLANGTIDY facebook-hte-CArray
    // op_name = "logical_and.out"
    logical_tensor_out(logical_and, ctx, a, b, out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::portable::cpu::pattern::logical_op::test_harness as h;

    // The C++ `op_reference` bit-reinterprets the incoming `double` to `uint64_t`
    // via `memcpy`, then evaluates `lhs && rhs`.
    fn op_reference(x: f64, y: f64) -> f64 {
        let lhs = x.to_bits();
        let rhs = y.to_bits();
        ((lhs != 0) && (rhs != 0)) as i32 as f64
    }

    // [spec:et:sem:op-logical-and.torch.executor.native.logical-and-out-fn/test]
    // [spec:et:sem:logical-op.torch.executor.native.internal.logical-tensor-out-fn/test]
    // also verifies logical_and: each output element is the `a && b` of the bool-cast operands.
    // [spec:et:sem:op-logical-and.torch.executor.native.logical-and-fn/test]
    #[test]
    fn op_logical_and_test_simple_test_all_types() {
        h::test_all_dtypes(logical_and_out, op_reference);
    }
}
