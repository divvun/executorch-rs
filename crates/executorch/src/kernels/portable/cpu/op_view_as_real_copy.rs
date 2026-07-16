//! Literal port of kernels/portable/cpu/op_view_as_real_copy.cpp.

use crate::kernels::portable::cpu::util::copy_ops_util::get_view_as_real_copy_out_target_size;
use crate::runtime::core::array_ref::ArrayRef;
use crate::runtime::core::error::Error;
use crate::runtime::core::exec_aten::util::scalar_type_util::is_complex_type;
use crate::runtime::core::exec_aten::util::tensor_util::{
    K_TENSOR_DIMENSION_LIMIT, resize_tensor, tensors_have_same_dim_order2,
};
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::core::portable_type::tensor_impl::SizesType;
use crate::runtime::core::portable_type::{Complex, Half};
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

// PORT-NOTE: `_to_impl` performs `static_cast<OUT_CTYPE>(val_in.real_)` /
// `static_cast<OUT_CTYPE>(val_in.imag_)`, converting a complex component type
// (Half/f32/f64) to the real output element type (Half/f32/f64). The ported
// `Complex<T>` carries no conversions, so this module-local trait reproduces the
// C++ implicit component-to-real cast without redesigning `Complex`.
trait ComponentAs<R> {
    fn component_as(self) -> R;
}
macro_rules! impl_component_as {
    ($comp:ty, $to_f64:expr) => {
        impl ComponentAs<Half> for $comp {
            fn component_as(self) -> Half {
                Half::from_f64($to_f64(self))
            }
        }
        impl ComponentAs<f32> for $comp {
            fn component_as(self) -> f32 {
                $to_f64(self) as f32
            }
        }
        impl ComponentAs<f64> for $comp {
            fn component_as(self) -> f64 {
                $to_f64(self)
            }
        }
    };
}
impl_component_as!(Half, |x: Half| x.to_f64());
impl_component_as!(f32, |x: f32| x as f64);
impl_component_as!(f64, |x: f64| x);

// [spec:et:def:op-view-as-real-copy.torch.executor.native.to-impl-fn]
// [spec:et:sem:op-view-as-real-copy.torch.executor.native.to-impl-fn]
fn _to_impl<SELF_COMP, OUT_CTYPE>(self_: &Tensor, out: &Tensor)
where
    SELF_COMP: Copy + ComponentAs<OUT_CTYPE>,
{
    let self_data = self_.mutable_data_ptr::<Complex<SELF_COMP>>();
    let out_data = out.mutable_data_ptr::<OUT_CTYPE>();

    for i in 0..self_.numel() as usize {
        let val_in = unsafe { *self_data.add(i) };
        unsafe {
            *out_data.add(2 * i) = val_in.real.component_as();
            *out_data.add(2 * i + 1) = val_in.imag.component_as();
        }
    }
}

// view_as_real_copy(Tensor self) -> Tensor
// [spec:et:def:op-view-as-real-copy.torch.executor.native.view-as-real-copy-out-fn]
// [spec:et:sem:op-view-as-real-copy.torch.executor.native.view-as-real-copy-out-fn]
pub fn view_as_real_copy_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    self_: &Tensor,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    let _ = &ctx;

    // Get the output shape
    let mut expected_output_size: [SizesType; K_TENSOR_DIMENSION_LIMIT] =
        [0; K_TENSOR_DIMENSION_LIMIT];
    // PORT-NOTE: C++ message is
    // "Output size buffer is too small. Expected at least %zu, got %zu" with args
    // `self.dim() + 1` and `kTensorDimensionLimit`. `et_kernel_check_msg!` keeps
    // only the condition-format arg (via `__et_first_fmt!`), so the runtime args
    // are dropped; the static text is preserved.
    crate::et_kernel_check_msg!(
        ctx,
        (self_.dim() as usize) < K_TENSOR_DIMENSION_LIMIT,
        InvalidArgument,
        out,
        "Output size buffer is too small."
    );
    unsafe {
        get_view_as_real_copy_out_target_size(self_, expected_output_size.as_mut_ptr());
    }

    // Resize for dynamic shape
    crate::et_kernel_check_msg!(
        ctx,
        resize_tensor(
            out,
            ArrayRef::from_raw_parts(expected_output_size.as_ptr(), out.dim() as usize)
        ) == Error::Ok,
        InvalidArgument,
        out,
        "Failed to resize output tensor."
    );

    // The input tensor must be complex type
    crate::et_kernel_check_msg!(
        ctx,
        is_complex_type(self_.scalar_type()),
        InvalidArgument,
        out,
        "Input tensor must be complex type"
    );

    crate::et_kernel_check!(
        ctx,
        tensors_have_same_dim_order2(self_, out),
        InvalidArgument,
        out
    );

    let op_name = "view_as_real_copy.out";

    crate::et_switch_complexh_types!(self_.scalar_type(), ctx, op_name, CTYPE_IN, {
        crate::et_switch_floath_types!(out.scalar_type(), ctx, op_name, CTYPE_OUT, {
            _to_impl::<<CTYPE_IN as ComplexComponent>::Component, CTYPE_OUT>(self_, out);
        });
    });

    out
}

// PORT-NOTE: the C++ `ET_SWITCH_COMPLEXH_TYPES` binds `CTYPE_IN` to the whole
// complex element type (`c10::complex<T>`) and `_to_impl` templates on it, reading
// `.real_`/`.imag_`. The ported switch binds `CTYPE_IN` to `Complex<T>` too; this
// helper trait recovers the component type `T` so `_to_impl` can be generic over
// the component (mirroring the C++ member access on the complex value).
trait ComplexComponent {
    type Component: Copy + ComponentAs<Half> + ComponentAs<f32> + ComponentAs<f64>;
}
impl ComplexComponent for Complex<Half> {
    type Component = Half;
}
impl ComplexComponent for Complex<f32> {
    type Component = f32;
}
impl ComplexComponent for Complex<f64> {
    type Component = f64;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::core::exec_aten::testing_util::tensor_factory::{
        FactoryValue, TensorFactory,
    };
    use crate::runtime::core::exec_aten::testing_util::tensor_util::tensors_are_close;
    use crate::runtime::core::exec_aten::util::scalar_type_util::{
        CppTypeToScalarType, to_real_value_type,
    };
    use crate::runtime::core::memory_allocator::{MemoryAllocator, MemoryAllocatorBase};
    use crate::runtime::core::portable_type::{ComplexDouble, ComplexFloat, ComplexHalf};

    fn context() -> KernelRuntimeContext<'static> {
        crate::runtime::platform::platform::pal_init();
        KernelRuntimeContext::new(
            crate::extension::module::module::null_event_tracer(),
            core::ptr::null_mut::<MemoryAllocator>() as *mut dyn MemoryAllocatorBase,
        )
    }

    fn view_as_real_copy_out_fn<'a, 'b>(
        ctx: &mut KernelRuntimeContext,
        self_: &Tensor,
        out: &'a Tensor<'b>,
    ) -> &'a Tensor<'b> {
        view_as_real_copy_out(ctx, self_, out)
    }

    // The real element type maps f64 literals to its own representation.
    trait FromF64: Copy {
        fn from_f64(v: f64) -> Self;
    }
    impl FromF64 for f32 {
        fn from_f64(v: f64) -> Self {
            v as f32
        }
    }
    impl FromF64 for f64 {
        fn from_f64(v: f64) -> Self {
            v
        }
    }
    impl FromF64 for Half {
        fn from_f64(v: f64) -> Self {
            Half::from_f32(v as f32)
        }
    }

    fn make_real<R: FromF64>(vals: &[f64]) -> Vec<R> {
        vals.iter().map(|&v| R::from_f64(v)).collect()
    }

    // run_complex_smoke_test<CTYPE, DTYPE>
    fn run_complex_smoke_test<C, R>(mk: impl Fn(f64, f64) -> C)
    where
        C: CppTypeToScalarType + FactoryValue,
        R: CppTypeToScalarType + FactoryValue + FromF64,
    {
        assert_eq!(to_real_value_type(C::VALUE), R::VALUE);
        let tf = TensorFactory::<C>::new();
        let tf_out = TensorFactory::<R>::new();

        let in_ = tf.make_default(
            vec![2, 2],
            vec![mk(3.0, 4.0), mk(-1.7, 7.4), mk(5.0, -12.0), mk(8.3, 0.1)],
        );
        let out = tf_out.zeros_default(vec![2, 2, 2]);
        let expected = tf_out.make_default(
            vec![2, 2, 2],
            make_real::<R>(&[3.0, 4.0, -1.7, 7.4, 5.0, -12.0, 8.3, 0.1]),
        );
        let mut ctx = context();
        let ret = view_as_real_copy_out_fn(&mut ctx, &in_, &out);

        assert!(tensors_are_close(&out, ret, 0.0, Some(0.0)));
        assert!(tensors_are_close(&out, &expected, 0.0, Some(0.0)));
    }

    // Tests on tensors with 0 size
    fn test_empty_input<C, R>()
    where
        C: CppTypeToScalarType + FactoryValue,
        R: CppTypeToScalarType + FactoryValue + FromF64,
    {
        let tf = TensorFactory::<C>::new();
        let tf_out = TensorFactory::<R>::new();

        let in_ = tf.make_default(vec![3, 0, 4], vec![]);
        let out = tf_out.zeros_default(vec![3, 0, 4, 2]);
        let expected = tf_out.make_default(vec![3, 0, 4, 2], vec![]);
        let mut ctx = context();
        let ret = view_as_real_copy_out_fn(&mut ctx, &in_, &out);

        assert!(tensors_are_close(&out, ret, 0.0, Some(0.0)));
        assert!(tensors_are_close(&out, &expected, 0.0, Some(0.0)));
    }

    // Tests on 0-dim input tensors
    fn zero_dim_input<C, R>(mk: impl Fn(f64, f64) -> C)
    where
        C: CppTypeToScalarType + FactoryValue,
        R: CppTypeToScalarType + FactoryValue + FromF64,
    {
        let tf = TensorFactory::<C>::new();
        let tf_out = TensorFactory::<R>::new();

        let in_ = tf.make_default(vec![], vec![mk(0.0, 0.0)]);
        let out = tf_out.zeros_default(vec![2]);
        let expected = tf_out.zeros_default(vec![2]);
        let mut ctx = context();
        let ret = view_as_real_copy_out_fn(&mut ctx, &in_, &out);

        assert!(tensors_are_close(&out, ret, 0.0, Some(0.0)));
        assert!(tensors_are_close(&out, &expected, 0.0, Some(0.0)));
    }

    // [spec:et:sem:op-view-as-real-copy.torch.executor.native.view-as-real-copy-out-fn/test]
    // also verifies get_view_as_real_copy_out_target_size (copies input sizes and
    // appends a trailing 2; zero_dim_input pins the 0-dim -> {2} case)
    // [spec:et:sem:copy-ops-util.torch.executor.get-view-as-real-copy-out-target-size-fn/test]
    // also verifies _to_impl: each complex element's real/imag components are
    // written to out[2*i]/out[2*i+1] (component-to-real cast), pinned by the
    // smoke test's exact 3.0,4.0,-1.7,7.4,... expansion across all complex dtypes.
    // [spec:et:sem:op-view-as-real-copy.torch.executor.native.to-impl-fn/test]
    #[test]
    fn op_view_as_real_test_complex_smoke_test() {
        run_complex_smoke_test::<ComplexHalf, Half>(|re, im| ComplexHalf {
            real: Half::from_f64(re),
            imag: Half::from_f64(im),
        });
        test_empty_input::<ComplexHalf, Half>();
        zero_dim_input::<ComplexHalf, Half>(|re, im| ComplexHalf {
            real: Half::from_f64(re),
            imag: Half::from_f64(im),
        });

        run_complex_smoke_test::<ComplexFloat, f32>(|re, im| ComplexFloat {
            real: re as f32,
            imag: im as f32,
        });
        test_empty_input::<ComplexFloat, f32>();
        zero_dim_input::<ComplexFloat, f32>(|re, im| ComplexFloat {
            real: re as f32,
            imag: im as f32,
        });

        run_complex_smoke_test::<ComplexDouble, f64>(|re, im| ComplexDouble { real: re, imag: im });
        test_empty_input::<ComplexDouble, f64>();
        zero_dim_input::<ComplexDouble, f64>(|re, im| ComplexDouble { real: re, imag: im });
    }
}
