//! Literal port of kernels/portable/cpu/op__conj_physical.cpp.

use crate::kernels::portable::cpu::util::functional_util::apply_unary_map_fn;
use crate::runtime::core::error::Error;
use crate::runtime::core::exec_aten::util::tensor_util::{
    resize_tensor, tensors_have_same_dim_order2, tensors_have_same_dtype2,
};
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::core::portable_type::{Complex, Half};
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

// PORT-NOTE: the C++ writes `CTYPE(val_in.real_, -val_in.imag_)`. The ported
// `Complex<T>` has `real`/`imag` fields but no arithmetic, so this module-local
// trait supplies the per-element-type negation of the imaginary component used to
// build the conjugate — without redesigning `Complex`.
trait Conjugate: Copy {
    fn conj(self) -> Self;
}
impl Conjugate for Complex<Half> {
    fn conj(self) -> Self {
        Complex {
            real: self.real,
            imag: Half::from_f64(-self.imag.to_f64()),
        }
    }
}
impl Conjugate for Complex<f32> {
    fn conj(self) -> Self {
        Complex {
            real: self.real,
            imag: -self.imag,
        }
    }
}
impl Conjugate for Complex<f64> {
    fn conj(self) -> Self {
        Complex {
            real: self.real,
            imag: -self.imag,
        }
    }
}

// [spec:et:def:op-conj-physical.torch.executor.native.conj-physical-out-fn]
// [spec:et:sem:op-conj-physical.torch.executor.native.conj-physical-out-fn]
pub fn _conj_physical_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    in_: &Tensor,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    crate::et_kernel_check_msg!(
        ctx,
        resize_tensor(out, in_.sizes()) == Error::Ok,
        InvalidArgument,
        out,
        "Failed to resize output tensor."
    );

    crate::et_kernel_check!(
        ctx,
        tensors_have_same_dtype2(in_, out),
        InvalidArgument,
        out
    );

    crate::et_kernel_check!(
        ctx,
        tensors_have_same_dim_order2(in_, out),
        InvalidArgument,
        out
    );

    let op_name = "_conj_physical.out";

    crate::et_switch_complexh_types!(in_.scalar_type(), ctx, op_name, CTYPE, {
        apply_unary_map_fn(
            |val_in: CTYPE| -> CTYPE { val_in.conj() },
            in_.const_data_ptr::<CTYPE>(),
            out.mutable_data_ptr::<CTYPE>(),
            in_.numel() as i64,
            1,
        );
    });

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::core::exec_aten::testing_util::tensor_factory::TensorFactory;
    use crate::runtime::core::exec_aten::testing_util::tensor_util::tensors_are_close;
    use crate::runtime::core::memory_allocator::{MemoryAllocator, MemoryAllocatorBase};
    use crate::runtime::core::portable_type::{ComplexDouble, ComplexFloat};

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

    fn op_conj_physical_out<'a, 'b>(in_: &Tensor, out: &'a Tensor<'b>) -> &'a Tensor<'b> {
        let mut ctx = context();
        _conj_physical_out(&mut ctx, in_, out)
    }

    fn cf(re: f32, im: f32) -> ComplexFloat {
        ComplexFloat { real: re, imag: im }
    }
    fn cd(re: f64, im: f64) -> ComplexDouble {
        ComplexDouble { real: re, imag: im }
    }

    // [spec:et:sem:op-conj-physical.torch.executor.native.conj-physical-out-fn/test]
    #[test]
    fn op_conj_physical_out_test_complex_float_basic() {
        let tf = TensorFactory::<ComplexFloat>::new();

        let sizes = vec![2, 2];

        let in_ = tf.make_default(
            sizes.clone(),
            vec![cf(1.0, 2.0), cf(3.0, 4.0), cf(5.0, -6.0), cf(-7.0, 8.0)],
        );

        let out = tf.zeros_default(sizes.clone());

        op_conj_physical_out(&in_, &out);

        let expected = tf.make_default(
            sizes,
            vec![cf(1.0, -2.0), cf(3.0, -4.0), cf(5.0, 6.0), cf(-7.0, -8.0)],
        );

        assert!(tensors_are_close(&out, &expected, 0.0, Some(0.0)));
    }

    // [spec:et:sem:op-conj-physical.torch.executor.native.conj-physical-out-fn/test]
    #[test]
    fn op_conj_physical_out_test_complex_double_basic() {
        let tf = TensorFactory::<ComplexDouble>::new();

        let sizes = vec![3];

        let in_ = tf.make_default(
            sizes.clone(),
            vec![cd(1.5, 2.5), cd(-3.5, 4.5), cd(0.0, -1.0)],
        );

        let out = tf.zeros_default(sizes.clone());

        op_conj_physical_out(&in_, &out);

        let expected = tf.make_default(sizes, vec![cd(1.5, -2.5), cd(-3.5, -4.5), cd(0.0, 1.0)]);

        assert!(tensors_are_close(&out, &expected, 0.0, Some(0.0)));
    }

    // [spec:et:sem:op-conj-physical.torch.executor.native.conj-physical-out-fn/test]
    #[test]
    fn op_conj_physical_out_test_real_part_only() {
        let tf = TensorFactory::<ComplexFloat>::new();

        let sizes = vec![2];

        let in_ = tf.make_default(sizes.clone(), vec![cf(5.0, 0.0), cf(-3.0, 0.0)]);

        let out = tf.zeros_default(sizes);

        op_conj_physical_out(&in_, &out);

        let out_data = out.const_data_ptr::<ComplexFloat>();
        let d0 = unsafe { *out_data.add(0) };
        let d1 = unsafe { *out_data.add(1) };
        assert_eq!(d0.real, 5.0);
        assert_eq!(d0.imag, -0.0);
        assert_eq!(d1.real, -3.0);
        assert_eq!(d1.imag, -0.0);
    }

    // [spec:et:sem:op-conj-physical.torch.executor.native.conj-physical-out-fn/test]
    #[test]
    fn op_conj_physical_out_test_imaginary_part_only() {
        let tf = TensorFactory::<ComplexFloat>::new();

        let sizes = vec![2];

        let in_ = tf.make_default(sizes.clone(), vec![cf(0.0, 5.0), cf(0.0, -3.0)]);

        let out = tf.zeros_default(sizes.clone());

        op_conj_physical_out(&in_, &out);

        let expected = tf.make_default(sizes, vec![cf(0.0, -5.0), cf(0.0, 3.0)]);

        assert!(tensors_are_close(&out, &expected, 0.0, Some(0.0)));
    }

    // [spec:et:sem:op-conj-physical.torch.executor.native.conj-physical-out-fn/test]
    #[test]
    fn op_conj_physical_out_test_empty_tensor() {
        let tf = TensorFactory::<ComplexFloat>::new();

        let sizes = vec![0];

        let in_ = tf.make_default(sizes.clone(), vec![]);
        let out = tf.zeros_default(sizes);

        op_conj_physical_out(&in_, &out);

        assert_eq!(out.numel(), 0);
    }

    // [spec:et:sem:op-conj-physical.torch.executor.native.conj-physical-out-fn/test]
    #[test]
    fn op_conj_physical_out_test_mismatched_dtype_dies() {
        let tf_in = TensorFactory::<ComplexFloat>::new();
        let tf_out = TensorFactory::<ComplexDouble>::new();

        let sizes = vec![2];

        let in_ = tf_in.make_default(sizes.clone(), vec![cf(1.0, 2.0), cf(3.0, 4.0)]);
        let out = tf_out.zeros_default(sizes);

        let mut ctx = context();
        _conj_physical_out(&mut ctx, &in_, &out);
        assert_ne!(ctx.failure_state(), Error::Ok);
    }
}
