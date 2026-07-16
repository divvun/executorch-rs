//! Literal port of kernels/portable/cpu/op_unbind_copy.cpp.

use crate::kernels::portable::cpu::util::copy_ops_util::check_unbind_copy_args;
use crate::kernels::portable::cpu::util::dtype_util::internal::StaticCast;
use crate::runtime::core::array_ref::ArrayRef;
use crate::runtime::core::exec_aten::util::tensor_util::{
    getLeadingDims, getTrailingDims, tensor_is_default_dim_order, tensors_have_same_dim_order2,
};
use crate::runtime::core::portable_type::scalar_type::ScalarType;
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::core::portable_type::tensor_impl::ssize_t;
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

// PORT-NOTE: C++ `TensorList out` (kernel_includes.h) is
// `executorch::aten::ArrayRef<Tensor>`; the kernel writes into the out tensors
// through the non-owning handles, so `out: ArrayRef<Tensor>` mirrors it.
// PORT-NOTE: `convert<CTYPE_OUT, CTYPE_IN>` is the scalar_type_util `convert`
// template; ported as the `Convert` trait (util/convert.rs) reproducing both
// C++ overloads.

// PORT-NOTE: local port of the scalar_type_util `convert<To, From>` template.
// Two C++ overloads: floating `From` -> integral `To` goes through
// `static_cast<int64_t>` first; every other pair is a plain `static_cast<To>`.
// `StaticCast` reproduces `static_cast` for the ET scalar ctypes.
trait Convert<From> {
    fn convert(val: From) -> Self;
}

// Non-(float->int) pairs: plain static_cast.
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
// float-source -> int-dest: static_cast<To>(static_cast<int64_t>(val)).
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

use crate::runtime::core::portable_type::{BFloat16, Half};

// Integral / Bool / Half / BFloat16 destinations from every REALHBBF16 source,
// and floating destinations from every source: plain static_cast.
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
// float / double / Half / BFloat16 destinations: always plain static_cast
// (these are not integral, so never take the int64-intermediate overload).
impl_convert_row_plain!(f32);
impl_convert_row_plain!(f64);
impl_convert_row_plain!(Half);
impl_convert_row_plain!(BFloat16);

// Integral destinations (`std::is_integral` includes `bool`): plain from
// integral / bool sources, but via int64 from floating (f32/f64/Half/BFloat16)
// sources.
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

// unbind_copy.int_out(Tensor input, int dim=0, *, Tensor(a!)[] out) -> ()
// [spec:et:def:op-unbind-copy.torch.executor.native.unbind-copy-int-out-fn]
// [spec:et:sem:op-unbind-copy.torch.executor.native.unbind-copy-int-out-fn]
pub fn unbind_copy_int_out(
    ctx: &mut KernelRuntimeContext,
    input: &Tensor,
    mut dim: i64,
    out: ArrayRef<Tensor>,
) {
    // (void)ctx;
    // Support python-style negative indexing.
    if dim < 0 {
        dim += input.dim() as i64;
    }

    crate::et_kernel_check!(
        ctx,
        check_unbind_copy_args(input, dim, out),
        InvalidArgument
    );

    for i in 0..out.size() {
        crate::et_kernel_check!(
            ctx,
            tensors_have_same_dim_order2(input, out.at(i)),
            InvalidArgument
        );
    }

    crate::et_kernel_check!(ctx, tensor_is_default_dim_order(input), InvalidArgument);

    if input.numel() == 0 {
        return;
    }

    let leading_dims: usize = getLeadingDims(input, dim);
    let trailing_dims: usize = getTrailingDims(input, dim);
    let step: usize = input.size(dim as ssize_t) as usize * trailing_dims;

    let in_type: ScalarType = input.scalar_type();
    let out_type: ScalarType = out.at(0).scalar_type();

    crate::et_switch_realhbbf16_types!(in_type, ctx, "unbind_copy.int_out", CTYPE_IN, {
        crate::et_switch_realhbbf16_types!(out_type, ctx, "unbind_copy.int_out", CTYPE_OUT, {
            let input_data: *const CTYPE_IN = input.const_data_ptr::<CTYPE_IN>();
            let e = out.size();
            for i in 0..e {
                let mut input_offset: usize = i * trailing_dims;
                let dest: *mut CTYPE_OUT = out.at(i).mutable_data_ptr::<CTYPE_OUT>();
                let mut dest_offset: usize = 0;
                for _j in 0..leading_dims {
                    for k in 0..trailing_dims {
                        unsafe {
                            *dest.add(dest_offset + k) = <CTYPE_OUT as Convert<CTYPE_IN>>::convert(
                                *input_data.add(input_offset + k),
                            );
                        }
                    }
                    input_offset += step;
                    dest_offset += trailing_dims;
                }
            }
        });
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assert_tensor_lists_eq;
    use crate::runtime::core::error::Error;
    use crate::runtime::core::exec_aten::testing_util::tensor_factory::{
        FactoryValue, TensorFactory,
    };
    use crate::runtime::core::exec_aten::util::scalar_type_util::CppTypeToScalarType;
    use crate::runtime::core::memory_allocator::{MemoryAllocator, MemoryAllocatorBase};
    use crate::runtime::core::tensor_shape_dynamism::TensorShapeDynamism;

    fn context() -> KernelRuntimeContext<'static> {
        crate::runtime::platform::platform::pal_init();
        KernelRuntimeContext::new(
            crate::extension::module::module::null_event_tracer(),
            core::ptr::null_mut::<MemoryAllocator>() as *mut dyn MemoryAllocatorBase,
        )
    }

    // PORT-NOTE: `static_cast<CTYPE>(int)` bridge for building integer literal data
    // in the REALHBF16 factory element types used by the templated test helpers.
    trait FromI32Data: Copy {
        fn from_i32(v: i32) -> Self;
    }
    macro_rules! impl_from_i32_data_num {
        ($($t:ty),*) => {$(impl FromI32Data for $t { fn from_i32(v: i32) -> Self { v as $t } })*};
    }
    impl_from_i32_data_num!(u8, i8, i16, i32, i64, f32, f64);
    impl FromI32Data for Half {
        fn from_i32(v: i32) -> Self {
            Half::from_f32(v as f32)
        }
    }
    impl FromI32Data for BFloat16 {
        fn from_i32(v: i32) -> Self {
            BFloat16::from_f32(v as f32)
        }
    }

    trait UnbindElem: CppTypeToScalarType + FactoryValue + FromI32Data {}
    impl<T: CppTypeToScalarType + FactoryValue + FromI32Data> UnbindElem for T {}

    // Wraps a `&[Tensor]` in the non-owning `ArrayRef<Tensor>` the kernel expects.
    // Mirrors the C++ `TensorList out` handle over the output vector.
    fn tensor_list<'t>(v: &'t [Tensor<'t>]) -> ArrayRef<Tensor<'t>> {
        ArrayRef::from_raw_parts(v.as_ptr(), v.len())
    }

    fn make1x2x3<T: UnbindElem>(tf: &TensorFactory<T>) -> Tensor<'_> {
        tf.make_default(vec![1, 2, 3], d::<T>(&[0, 1, 2, 3, 4, 5]))
    }

    fn d<T: FromI32Data>(v: &[i32]) -> Vec<T> {
        v.iter().map(|&x| T::from_i32(x)).collect()
    }

    fn test_unbind_dim0<T: UnbindElem>() {
        let tf = TensorFactory::<T>::new();

        let expected_out = vec![tf.make_default(vec![2, 3], d::<T>(&[0, 1, 2, 3, 4, 5]))];
        let input = make1x2x3(&tf);

        let out = vec![tf.zeros_default(vec![2, 3])];
        let mut ctx = context();
        unbind_copy_int_out(&mut ctx, &input, 0, tensor_list(&out));
        assert_tensor_lists_eq!(expected_out, out);

        let out2 = vec![tf.zeros_default(vec![2, 3])];
        unbind_copy_int_out(&mut ctx, &input, -3, tensor_list(&out2));
        assert_tensor_lists_eq!(expected_out, out2);
    }

    fn test_unbind_dim1<T: UnbindElem>() {
        let tf = TensorFactory::<T>::new();

        let expected_out = vec![
            tf.make_default(vec![1, 3], d::<T>(&[0, 1, 2])),
            tf.make_default(vec![1, 3], d::<T>(&[3, 4, 5])),
        ];
        let input = make1x2x3(&tf);

        let out = vec![tf.zeros_default(vec![1, 3]), tf.zeros_default(vec![1, 3])];
        let mut ctx = context();
        unbind_copy_int_out(&mut ctx, &input, 1, tensor_list(&out));
        assert_tensor_lists_eq!(expected_out, out);

        let out2 = vec![tf.zeros_default(vec![1, 3]), tf.zeros_default(vec![1, 3])];
        unbind_copy_int_out(&mut ctx, &input, -2, tensor_list(&out2));
        assert_tensor_lists_eq!(expected_out, out2);
    }

    fn test_unbind_dim2<T: UnbindElem>() {
        let tf = TensorFactory::<T>::new();

        let expected_out = vec![
            tf.make_default(vec![1, 2], d::<T>(&[0, 3])),
            tf.make_default(vec![1, 2], d::<T>(&[1, 4])),
            tf.make_default(vec![1, 2], d::<T>(&[2, 5])),
        ];
        let input = make1x2x3(&tf);

        let out = vec![
            tf.zeros_default(vec![1, 2]),
            tf.zeros_default(vec![1, 2]),
            tf.zeros_default(vec![1, 2]),
        ];
        let mut ctx = context();
        unbind_copy_int_out(&mut ctx, &input, 2, tensor_list(&out));
        assert_tensor_lists_eq!(expected_out, out);

        let out2 = vec![
            tf.zeros_default(vec![1, 2]),
            tf.zeros_default(vec![1, 2]),
            tf.zeros_default(vec![1, 2]),
        ];
        unbind_copy_int_out(&mut ctx, &input, -1, tensor_list(&out2));
        assert_tensor_lists_eq!(expected_out, out2);
    }

    fn test_dynamic_shape(out_shape: Vec<i32>, dynamism: TensorShapeDynamism) {
        let tf = TensorFactory::<i32>::new();

        let x = tf.make_default(
            vec![2, 3, 4],
            vec![
                4, 9, 3, 0, 3, 9, 7, 3, 7, 3, 1, 6, //
                6, 9, 8, 6, 6, 8, 4, 3, 6, 9, 1, 4,
            ],
        );
        let expected = vec![
            tf.make_default(vec![2, 4], vec![4, 9, 3, 0, 6, 9, 8, 6]),
            tf.make_default(vec![2, 4], vec![3, 9, 7, 3, 6, 8, 4, 3]),
            tf.make_default(vec![2, 4], vec![7, 3, 1, 6, 6, 9, 1, 4]),
        ];

        let out = vec![
            tf.zeros(out_shape.clone(), dynamism),
            tf.zeros(out_shape.clone(), dynamism),
            tf.zeros(out_shape, dynamism),
        ];
        let mut ctx = context();
        unbind_copy_int_out(&mut ctx, &x, 1, tensor_list(&out));
        assert_tensor_lists_eq!(out, expected);
    }

    // [spec:et:sem:op-unbind-copy.torch.executor.native.unbind-copy-int-out-fn/test]
    #[test]
    fn op_unbind_copy_int_out_test_unbind1x2x3_on_dim0_all_real_dtypes() {
        // ET_FORALL_REALHBF16_TYPES
        test_unbind_dim0::<u8>();
        test_unbind_dim0::<i8>();
        test_unbind_dim0::<i16>();
        test_unbind_dim0::<i32>();
        test_unbind_dim0::<i64>();
        test_unbind_dim0::<f32>();
        test_unbind_dim0::<f64>();
        test_unbind_dim0::<Half>();
        test_unbind_dim0::<BFloat16>();
    }

    // [spec:et:sem:op-unbind-copy.torch.executor.native.unbind-copy-int-out-fn/test]
    // also verifies check_unbind_copy_args (in-rank>0, dim valid, out tensorlist
    // length == dim size, each out dim == in.dim()-1 with matching non-dim shapes)
    // [spec:et:sem:copy-ops-util.torch.executor.check-unbind-copy-args-fn/test]
    #[test]
    fn op_unbind_copy_int_out_test_unbind1x2x3_on_dim1_all_real_dtypes() {
        test_unbind_dim1::<u8>();
        test_unbind_dim1::<i8>();
        test_unbind_dim1::<i16>();
        test_unbind_dim1::<i32>();
        test_unbind_dim1::<i64>();
        test_unbind_dim1::<f32>();
        test_unbind_dim1::<f64>();
        test_unbind_dim1::<Half>();
        test_unbind_dim1::<BFloat16>();
    }

    // [spec:et:sem:op-unbind-copy.torch.executor.native.unbind-copy-int-out-fn/test]
    #[test]
    fn op_unbind_copy_int_out_test_unbind1x2x3_on_dim2_all_real_dtypes() {
        test_unbind_dim2::<u8>();
        test_unbind_dim2::<i8>();
        test_unbind_dim2::<i16>();
        test_unbind_dim2::<i32>();
        test_unbind_dim2::<i64>();
        test_unbind_dim2::<f32>();
        test_unbind_dim2::<f64>();
        test_unbind_dim2::<Half>();
        test_unbind_dim2::<BFloat16>();
    }

    // [spec:et:sem:op-unbind-copy.torch.executor.native.unbind-copy-int-out-fn/test]
    #[test]
    fn op_unbind_copy_int_out_test_zero_dimensional_input_tensor_dies() {
        let tf = TensorFactory::<i32>::new();

        let input = tf.ones_default(vec![]);
        let out = vec![tf.zeros_default(vec![])];

        let mut ctx = context();
        unbind_copy_int_out(&mut ctx, &input, 0, tensor_list(&out));
        assert_ne!(ctx.failure_state(), Error::Ok);
    }

    // [spec:et:sem:op-unbind-copy.torch.executor.native.unbind-copy-int-out-fn/test]
    #[test]
    fn op_unbind_copy_int_out_test_unbind_works_with_zero_sized_tensors() {
        let tf = TensorFactory::<i32>::new();

        let input = tf.ones_default(vec![1, 0, 2]);
        assert_eq!(input.numel(), 0);

        // unbind dim 0
        let expected_out = vec![tf.ones_default(vec![0, 2])];
        let out = vec![tf.zeros_default(vec![0, 2])];
        let mut ctx = context();
        unbind_copy_int_out(&mut ctx, &input, 0, tensor_list(&out));
        assert_tensor_lists_eq!(out, expected_out);

        // unbind dim 1
        let expected_out: Vec<Tensor> = vec![];
        let out: Vec<Tensor> = vec![];
        unbind_copy_int_out(&mut ctx, &input, 1, tensor_list(&out));
        assert_tensor_lists_eq!(out, expected_out);

        // unbind dim 2
        let expected_out = vec![tf.ones_default(vec![1, 0]), tf.ones_default(vec![1, 0])];
        let out = vec![tf.zeros_default(vec![1, 0]), tf.zeros_default(vec![1, 0])];
        unbind_copy_int_out(&mut ctx, &input, 2, tensor_list(&out));
        assert_tensor_lists_eq!(out, expected_out);
    }

    // PORT-NOTE: C++ `ET_SKIP_IF(is_aten, ...)`; never ATen in the port.
    // [spec:et:sem:op-unbind-copy.torch.executor.native.unbind-copy-int-out-fn/test]
    #[test]
    fn op_unbind_copy_int_out_test_unbind_fails_with_wrongly_allocated_output() {
        let tf = TensorFactory::<i32>::new();

        let input = tf.ones_default(vec![1, 2, 3]);

        // Die because length of the list should be 2.
        let out = vec![tf.ones_default(vec![1, 3])];
        let mut ctx = context();
        unbind_copy_int_out(&mut ctx, &input, 1, tensor_list(&out));
        assert_ne!(ctx.failure_state(), Error::Ok);

        // Die because output tensors in the list should be of correct sizes.
        let out = vec![tf.ones_default(vec![1, 4]), tf.ones_default(vec![1, 4])];
        let mut ctx = context();
        unbind_copy_int_out(&mut ctx, &input, 1, tensor_list(&out));
        assert_ne!(ctx.failure_state(), Error::Ok);

        // Die because output tensors in the list should have correct number of dims.
        let out = vec![tf.ones_default(vec![1]), tf.ones_default(vec![1])];
        let mut ctx = context();
        unbind_copy_int_out(&mut ctx, &input, 1, tensor_list(&out));
        assert_ne!(ctx.failure_state(), Error::Ok);
    }

    // [spec:et:sem:op-unbind-copy.torch.executor.native.unbind-copy-int-out-fn/test]
    #[test]
    fn op_unbind_copy_int_out_test_unbind_produce_scalar_tensors() {
        let tf = TensorFactory::<i32>::new();

        let input = tf.make_default(vec![3], vec![4, 5, 6]);

        let expected_out = vec![
            tf.make_default(vec![], vec![4]),
            tf.make_default(vec![], vec![5]),
            tf.make_default(vec![], vec![6]),
        ];
        let out = vec![
            tf.zeros_default(vec![]),
            tf.zeros_default(vec![]),
            tf.zeros_default(vec![]),
        ];
        let mut ctx = context();
        unbind_copy_int_out(&mut ctx, &input, 0, tensor_list(&out));
        assert_tensor_lists_eq!(out, expected_out);
    }

    // [spec:et:sem:op-unbind-copy.torch.executor.native.unbind-copy-int-out-fn/test]
    #[test]
    fn op_unbind_copy_int_out_test_unbind_produce_scalar_like_tensors() {
        let tf = TensorFactory::<i32>::new();

        let input = tf.make_default(vec![3, 1], vec![4, 5, 6]);
        let expected_out = vec![
            tf.make_default(vec![1], vec![4]),
            tf.make_default(vec![1], vec![5]),
            tf.make_default(vec![1], vec![6]),
        ];
        let out = vec![
            tf.zeros_default(vec![1]),
            tf.zeros_default(vec![1]),
            tf.zeros_default(vec![1]),
        ];
        let mut ctx = context();
        unbind_copy_int_out(&mut ctx, &input, 0, tensor_list(&out));
        assert_tensor_lists_eq!(out, expected_out);

        let input = tf.make_default(vec![1, 3], vec![4, 5, 6]);
        let expected_out = vec![
            tf.make_default(vec![1], vec![4]),
            tf.make_default(vec![1], vec![5]),
            tf.make_default(vec![1], vec![6]),
        ];
        let out = vec![
            tf.zeros_default(vec![1]),
            tf.zeros_default(vec![1]),
            tf.zeros_default(vec![1]),
        ];
        unbind_copy_int_out(&mut ctx, &input, 1, tensor_list(&out));
        assert_tensor_lists_eq!(out, expected_out);
    }

    // [spec:et:sem:op-unbind-copy.torch.executor.native.unbind-copy-int-out-fn/test]
    #[test]
    fn op_unbind_copy_int_out_test_dynamic_shape_upper_bound_same_as_expected() {
        test_dynamic_shape(vec![2, 4], TensorShapeDynamism::DYNAMIC_BOUND);
    }

    // PORT-NOTE: C++ `DISABLED_DynamicShapeUpperBoundLargerThanExpected` (disabled:
    // dynamic shape not supported). Ported as `#[ignore]`.
    // [spec:et:sem:op-unbind-copy.torch.executor.native.unbind-copy-int-out-fn/test]
    #[test]
    #[ignore = "DISABLED: Dynamic shape not supported"]
    fn op_unbind_copy_int_out_test_dynamic_shape_upper_bound_larger_than_expected() {
        test_dynamic_shape(vec![10, 10], TensorShapeDynamism::DYNAMIC_BOUND);
    }

    // PORT-NOTE: C++ `DISABLED_DynamicShapeUnbound`. Ported as `#[ignore]`.
    // [spec:et:sem:op-unbind-copy.torch.executor.native.unbind-copy-int-out-fn/test]
    #[test]
    #[ignore = "DISABLED: Dynamic shape not supported"]
    fn op_unbind_copy_int_out_test_dynamic_shape_unbound() {
        test_dynamic_shape(vec![1, 1], TensorShapeDynamism::DYNAMIC_UNBOUND);
    }

    // [spec:et:sem:op-unbind-copy.torch.executor.native.unbind-copy-int-out-fn/test]
    #[test]
    fn op_unbind_copy_int_out_test_boolean_tensor_unbind_dim2() {
        let tf = TensorFactory::<bool>::new();

        // Create input tensor of shape (1, 7, 4) with an alternating pattern.
        let in_vec: Vec<bool> = (0..(1 * 7 * 4)).map(|i| (i % 2) == 0).collect();
        let input = tf.make_default(vec![1, 7, 4], in_vec);

        let unbind_dim: i64 = 2;
        let num_outputs = input.size(unbind_dim as ssize_t); // 4

        let out: Vec<Tensor> = (0..num_outputs)
            .map(|_| tf.zeros_default(vec![1, 7]))
            .collect();
        let mut ctx = context();
        unbind_copy_int_out(&mut ctx, &input, unbind_dim, tensor_list(&out));

        for output_idx in 0..num_outputs as usize {
            assert_eq!(out[output_idx].dim(), 2);
            assert_eq!(out[output_idx].size(0), 1);
            assert_eq!(out[output_idx].size(1), 7);

            let out_data = out[output_idx].const_data_ptr::<bool>();
            for i in 0..1usize {
                for j in 0..7usize {
                    let input_idx = i * 7 * 4 + j * 4 + output_idx;
                    let expected = (input_idx % 2) == 0;
                    assert_eq!(unsafe { *out_data.add(i * 7 + j) }, expected);
                }
            }
        }
    }
}
