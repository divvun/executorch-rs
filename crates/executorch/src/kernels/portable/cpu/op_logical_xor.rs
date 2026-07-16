//! Literal port of kernels/portable/cpu/op_logical_xor.cpp.

use crate::kernels::portable::cpu::pattern::logical_op::logical_tensor_out;
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

// PORT-NOTE: the C++ anonymous-namespace `logical_xor` helper is a free `fn`
// here; it is passed by value to `logical_tensor_out` (which takes `fn(bool,
// bool) -> bool`), mirroring the C++ function-pointer decay.

// [spec:et:def:op-logical-xor.torch.executor.native.logical-xor-fn]
// [spec:et:sem:op-logical-xor.torch.executor.native.logical-xor-fn]
fn logical_xor(a: bool, b: bool) -> bool {
    a != b
}

// PORT-NOTE (cross-module): the compile-time `op_name` template parameter of the
// C++ `logical_tensor_out<op_name>` ("logical_xor.out") is dropped — the ported
// `logical_tensor_out` takes no op-name argument.

// [spec:et:def:op-logical-xor.torch.executor.native.logical-xor-out-fn]
// [spec:et:sem:op-logical-xor.torch.executor.native.logical-xor-out-fn]
pub fn logical_xor_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    a: &Tensor,
    b: &Tensor,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    logical_tensor_out(logical_xor, ctx, a, b, out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::portable::cpu::pattern::logical_op::test_harness as h;

    // The C++ `op_reference` bit-reinterprets the incoming `double` to `uint64_t`
    // via `memcpy`, then evaluates `bool(lhs) != bool(rhs)`.
    fn op_reference(x: f64, y: f64) -> f64 {
        let lhs = x.to_bits();
        let rhs = y.to_bits();
        ((lhs != 0) != (rhs != 0)) as i32 as f64
    }

    // [spec:et:sem:op-logical-xor.torch.executor.native.logical-xor-out-fn/test]
    // [spec:et:sem:logical-op.torch.executor.native.internal.logical-tensor-out-fn/test]
    // also verifies logical_xor: the (0,0)/(nz,0)/(0,nz)/(nz,nz) truth-table row set
    // pins each output element to `a != b` of the bool-cast operands.
    // [spec:et:sem:op-logical-xor.torch.executor.native.logical-xor-fn/test]
    #[test]
    fn op_logical_xor_test_simple_test_all_types() {
        h::test_all_dtypes(logical_xor_out, op_reference);
    }
}
