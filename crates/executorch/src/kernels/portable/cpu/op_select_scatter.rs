//! Literal port of kernels/portable/cpu/op_select_scatter.cpp.

use crate::kernels::portable::cpu::util::dtype_util::internal::StaticCast;
use crate::kernels::portable::cpu::util::index_util::check_select_scatter_args;
use crate::runtime::core::error::Error;
use crate::runtime::core::exec_aten::util::tensor_util::{
    getLeadingDims, getTrailingDims, resize_tensor, tensors_have_same_dim_order3,
};
use crate::runtime::core::portable_type::scalar_type::ScalarType;
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

// PORT-NOTE: `Tensor& out` / returned `Tensor&` become `&'a Tensor` (interior
// mutation through `*mut TensorImpl`). `(void)ctx;` dropped. The void-typed
// `memcpy(out.mutable_data_ptr(), in.const_data_ptr(), in.nbytes())` is a
// byte-wise `copy_nonoverlapping`.

// PORT-NOTE: local port of the scalar_type_util `convert<To, From>` template
// (`convert<CTYPE, CTYPE_SRC>`). Two C++ overloads: floating `From` -> integral
// `To` goes through `static_cast<int64_t>` first; every other pair is a plain
// `static_cast<To>`. Mirrors the `Convert` trait established in op_unbind_copy.
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

use crate::runtime::core::portable_type::{BFloat16, Half};

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

/// aten::select_scatter.out(Tensor self, Tensor src, int dim, SymInt index, *,
/// Tensor(a!) out) -> Tensor(a!)
// [spec:et:def:op-select-scatter.torch.executor.native.select-scatter-out-fn]
// [spec:et:sem:op-select-scatter.torch.executor.native.select-scatter-out-fn]
pub fn select_scatter_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    in_: &Tensor,
    src: &Tensor,
    mut dim: i64,
    mut index: i64,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    // (void)ctx;

    crate::et_kernel_check!(
        ctx,
        resize_tensor(out, in_.sizes()) == Error::Ok,
        InvalidArgument,
        out
    );

    crate::et_kernel_check!(
        ctx,
        tensors_have_same_dim_order3(in_, src, out),
        InvalidArgument,
        out
    );

    // Account for negative indices
    if dim < 0 {
        dim += in_.dim() as i64;
    }

    crate::et_kernel_check!(
        ctx,
        dim >= 0 && dim < in_.dim() as i64,
        InvalidArgument,
        out
    );

    if index < 0 {
        index += in_.size(dim as isize) as i64;
    }

    // Check args
    crate::et_kernel_check!(
        ctx,
        check_select_scatter_args(in_, src, dim, index, out),
        InvalidArgument,
        out
    );

    // If the input is an empty tensor, no other operation could be done. We just
    // return the output.
    if in_.numel() == 0 {
        return out;
    }

    // To start, copy the input into the output. Input will not be empty due to
    // the checks performed above.
    unsafe {
        core::ptr::copy_nonoverlapping(
            in_.const_data_ptr_typed() as *const u8,
            out.mutable_data_ptr_typed() as *mut u8,
            in_.nbytes(),
        );
    }

    // Strides to help with memory address arithmetic
    let leading_dims: usize = getLeadingDims(in_, dim);
    let trailing_stride: usize = getTrailingDims(in_, dim);
    let start_offset: usize = index as usize * trailing_stride;
    let out_step: usize = in_.size(dim as isize) as usize * trailing_stride;

    let in_type: ScalarType = in_.scalar_type();
    let src_type: ScalarType = src.scalar_type();

    crate::et_switch_realhbbf16_types!(in_type, ctx, "select_scatter.out", CTYPE, {
        crate::et_switch_realhbbf16_types!(src_type, ctx, "select_scatter.out", CTYPE_SRC, {
            let out_data: *mut CTYPE = out.mutable_data_ptr::<CTYPE>();
            let src_data: *const CTYPE_SRC = src.const_data_ptr::<CTYPE_SRC>();

            for i in 0..leading_dims {
                for j in 0..trailing_stride {
                    unsafe {
                        *out_data.add(start_offset + i * out_step + j) =
                            <CTYPE as Convert<CTYPE_SRC>>::convert(
                                *src_data.add(i * trailing_stride + j),
                            );
                    }
                }
            }
        });
    });

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assert_tensor_close;
    use crate::assert_tensor_eq;
    use crate::runtime::core::exec_aten::testing_util::tensor_factory::{
        FactoryValue, TensorFactory,
    };
    use crate::runtime::core::exec_aten::util::scalar_type_util::CppTypeToScalarType;
    use crate::runtime::core::memory_allocator::{MemoryAllocator, MemoryAllocatorBase};
    use crate::runtime::core::portable_type::tensor::Tensor as PtTensor;
    use crate::runtime::core::tensor_shape_dynamism::TensorShapeDynamism;

    fn context() -> KernelRuntimeContext<'static> {
        crate::runtime::platform::runtime::runtime_init();
        KernelRuntimeContext::new(
            crate::extension::module::module::null_event_tracer(),
            core::ptr::null_mut::<MemoryAllocator>() as *mut dyn MemoryAllocatorBase,
        )
    }

    fn op_select_scatter_out<'a, 'b>(
        self_: &Tensor,
        src: &Tensor,
        dim: i64,
        index: i64,
        out: &'a Tensor<'b>,
    ) -> &'a Tensor<'b> {
        let mut ctx = context();
        select_scatter_out(&mut ctx, self_, src, dim, index, out)
    }

    trait FromF64Elem: Copy {
        fn from_f64(v: f64) -> Self;
    }
    macro_rules! impl_from_f64_num {
        ($($t:ty),*) => {$(impl FromF64Elem for $t { fn from_f64(v: f64) -> Self { v as $t } })*};
    }
    impl_from_f64_num!(u8, i8, i16, i32, i64, f32, f64);
    impl FromF64Elem for bool {
        fn from_f64(v: f64) -> Self {
            v != 0.0
        }
    }
    use crate::runtime::core::portable_type::{BFloat16, Half};
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

    fn v<T: FromF64Elem>(vals: &[f64]) -> Vec<T> {
        vals.iter().map(|&x| T::from_f64(x)).collect()
    }

    // test_dtype<CTYPE, DTYPE>
    fn test_dtype<T>()
    where
        T: CppTypeToScalarType + FactoryValue + FromF64Elem,
    {
        let tf = TensorFactory::<T>::new();

        let x = tf.make_default(
            vec![3, 2, 4],
            v::<T>(&[
                1., 1., 1., 1., 0., 0., 0., 0., 1., 1., 1., 1., 0., 0., 0., 0., 1., 1., 1., 1., 0.,
                0., 0., 0.,
            ]),
        );

        let src_ones = tf.make_default(
            vec![3, 4],
            v::<T>(&[1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]),
        );

        let src_zeros = tf.make_default(
            vec![3, 4],
            v::<T>(&[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
        );

        let out_0 = tf.zeros_default(vec![3, 2, 4]);
        let out_1 = tf.ones_default(vec![3, 2, 4]);
        let ret_0 = op_select_scatter_out(&x, &src_zeros, 1, 0, &out_0);
        let ret_1 = op_select_scatter_out(&x, &src_ones, 1, 1, &out_1);

        assert_tensor_eq!(*ret_0, out_0);
        assert_tensor_eq!(*ret_1, out_1);

        assert_tensor_eq!(*ret_0, tf.zeros_default(vec![3, 2, 4]));
        assert_tensor_eq!(*ret_1, tf.ones_default(vec![3, 2, 4]));
    }

    fn run_test_cases(x: &Tensor, src: &Tensor, dim: isize, expected: &[PtTensor]) {
        let tf = TensorFactory::<f64>::new();

        let out_size: Vec<i32> = (0..expected[0].sizes().size())
            .map(|i| *expected[0].sizes().at(i))
            .collect();
        let out = tf.zeros_default(out_size);

        for idx in 0..x.size(dim) {
            let ret = op_select_scatter_out(x, src, dim as i64, idx as i64, &out);
            assert_tensor_eq!(out, *ret);
            assert_tensor_eq!(out, expected[idx as usize]);

            let ret = op_select_scatter_out(x, src, dim as i64, (idx - x.size(dim)) as i64, &out);
            assert_tensor_eq!(out, *ret);
            assert_tensor_eq!(out, expected[idx as usize]);
        }
    }

    fn test_dynamic_shape(out_shape: Vec<i32>, dynamism: TensorShapeDynamism) {
        let tf = TensorFactory::<i32>::new();

        let x = tf.make_default(vec![2, 3, 2], vec![4, 9, 3, 0, 3, 9, 7, 3, 7, 3, 1, 6]);
        let y = tf.make_default(vec![3, 2], vec![6, 9, 8, 6, 6, 8]);
        let expected = tf.make_default(vec![2, 3, 2], vec![4, 9, 3, 0, 3, 9, 6, 9, 8, 6, 6, 8]);

        let out = tf.zeros(out_shape, dynamism);
        op_select_scatter_out(&x, &y, 0, 1, &out);
        assert_tensor_close!(out, expected);
    }

    // [spec:et:sem:op-select-scatter.torch.executor.native.select-scatter-out-fn/test]
    // also verifies check_select_scatter_args (valid path); the _dim_out_of_bound,
    // _index_out_of_bound, _mismatched_dtypes and _src_*_dies tests exercise its
    // failure branches.
    // [spec:et:sem:index-util.torch.executor.check-select-scatter-args-fn/test]
    #[test]
    fn op_select_scatter_out_test_select_front_dim_all_indexes() {
        let tf = TensorFactory::<f64>::new();

        let x = tf.make_default(
            vec![2, 3, 4],
            vec![
                1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., -1., -2., -3., -4., -5., -6.,
                -7., -8., -9., -10., -11., -12.,
            ],
        );

        let src = tf.make_default(
            vec![3, 4],
            vec![1., 4., 1., 4., 1., 4., 1., 4., 1., 4., 1., 4.],
        );

        let out_size = vec![2, 3, 4];

        let expected_rets = vec![
            tf.make_default(
                out_size.clone(),
                vec![
                    1., 4., 1., 4., 1., 4., 1., 4., 1., 4., 1., 4., -1., -2., -3., -4., -5., -6.,
                    -7., -8., -9., -10., -11., -12.,
                ],
            ),
            tf.make_default(
                out_size,
                vec![
                    1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 1., 4., 1., 4., 1., 4., 1.,
                    4., 1., 4., 1., 4.,
                ],
            ),
        ];

        run_test_cases(&x, &src, 0, &expected_rets);
    }

    // [spec:et:sem:op-select-scatter.torch.executor.native.select-scatter-out-fn/test]
    #[test]
    fn op_select_scatter_out_test_select_middle_dim_all_indexes() {
        let tf = TensorFactory::<f64>::new();

        let x = tf.make_default(
            vec![2, 3, 4],
            vec![
                1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., -1., -2., -3., -4., -5., -6.,
                -7., -8., -9., -10., -11., -12.,
            ],
        );

        let src = tf.make_default(vec![2, 4], vec![1., 4., 1., 4., 1., 4., 1., 4.]);

        let out_size = vec![2, 3, 4];

        let expected_rets = vec![
            tf.make_default(
                out_size.clone(),
                vec![
                    1., 4., 1., 4., 5., 6., 7., 8., 9., 10., 11., 12., 1., 4., 1., 4., -5., -6.,
                    -7., -8., -9., -10., -11., -12.,
                ],
            ),
            tf.make_default(
                out_size.clone(),
                vec![
                    1., 2., 3., 4., 1., 4., 1., 4., 9., 10., 11., 12., -1., -2., -3., -4., 1., 4.,
                    1., 4., -9., -10., -11., -12.,
                ],
            ),
            tf.make_default(
                out_size,
                vec![
                    1., 2., 3., 4., 5., 6., 7., 8., 1., 4., 1., 4., -1., -2., -3., -4., -5., -6.,
                    -7., -8., 1., 4., 1., 4.,
                ],
            ),
        ];

        run_test_cases(&x, &src, 1, &expected_rets);
    }

    // [spec:et:sem:op-select-scatter.torch.executor.native.select-scatter-out-fn/test]
    #[test]
    fn op_select_scatter_out_test_select_end_dim_all_indexes() {
        let tf = TensorFactory::<f64>::new();

        let x = tf.make_default(
            vec![2, 3, 4],
            vec![
                1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., -1., -2., -3., -4., -5., -6.,
                -7., -8., -9., -10., -11., -12.,
            ],
        );

        let src = tf.make_default(vec![2, 3], vec![1., 4., 1., 1., 4., 1.]);

        let out_size = vec![2, 3, 4];

        let expected_rets = vec![
            tf.make_default(
                out_size.clone(),
                vec![
                    1., 2., 3., 4., 4., 6., 7., 8., 1., 10., 11., 12., 1., -2., -3., -4., 4., -6.,
                    -7., -8., 1., -10., -11., -12.,
                ],
            ),
            tf.make_default(
                out_size.clone(),
                vec![
                    1., 1., 3., 4., 5., 4., 7., 8., 9., 1., 11., 12., -1., 1., -3., -4., -5., 4.,
                    -7., -8., -9., 1., -11., -12.,
                ],
            ),
            tf.make_default(
                out_size.clone(),
                vec![
                    1., 2., 1., 4., 5., 6., 4., 8., 9., 10., 1., 12., -1., -2., 1., -4., -5., -6.,
                    4., -8., -9., -10., 1., -12.,
                ],
            ),
            tf.make_default(
                out_size,
                vec![
                    1., 2., 3., 1., 5., 6., 7., 4., 9., 10., 11., 1., -1., -2., -3., 1., -5., -6.,
                    -7., 4., -9., -10., -11., 1.,
                ],
            ),
        ];

        run_test_cases(&x, &src, 2, &expected_rets);
    }

    // #ifndef USE_ATEN_LIB — runs in the ET port.
    // [spec:et:sem:op-select-scatter.torch.executor.native.select-scatter-out-fn/test]
    #[test]
    fn op_select_scatter_out_test_output_dynamic_shape() {
        let tf = TensorFactory::<f64>::new();

        let x = tf.make_default(
            vec![2, 3, 4],
            vec![
                1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., -1., -2., -3., -4., -5., -6.,
                -7., -8., -9., -10., -11., -12.,
            ],
        );

        let src = tf.make_default(vec![2, 3], vec![1., 4., 1., 1., 4., 1.]);

        let out_size = vec![2, 6, 2];
        let actual_out_size = vec![2, 3, 4];

        let out = tf.zeros(out_size, TensorShapeDynamism::DYNAMIC_BOUND);

        let expected_ret = tf.make_default(
            actual_out_size,
            vec![
                1., 2., 3., 4., 4., 6., 7., 8., 1., 10., 11., 12., 1., -2., -3., -4., 4., -6., -7.,
                -8., 1., -10., -11., -12.,
            ],
        );

        let ret = op_select_scatter_out(&x, &src, 2, 0, &out);
        assert_tensor_eq!(out, *ret);
        assert_tensor_eq!(out, expected_ret);
    }

    // [spec:et:sem:op-select-scatter.torch.executor.native.select-scatter-out-fn/test]
    #[test]
    fn op_select_scatter_out_test_all_dtypes_supported() {
        // ET_FORALL_REALHBBF16_TYPES
        test_dtype::<u8>();
        test_dtype::<i8>();
        test_dtype::<i16>();
        test_dtype::<i32>();
        test_dtype::<i64>();
        test_dtype::<f32>();
        test_dtype::<f64>();
        test_dtype::<bool>();
        test_dtype::<Half>();
        test_dtype::<BFloat16>();
    }

    // [spec:et:sem:op-select-scatter.torch.executor.native.select-scatter-out-fn/test]
    #[test]
    fn op_select_scatter_out_test_empty_tensor_non_zero_n_dims_input_supported() {
        let tf = TensorFactory::<i32>::new();

        let x = tf.make_default(vec![3, 0, 10, 3], vec![]);
        assert_eq!(x.numel(), 0);

        let src = tf.make_default(vec![3, 0, 3], vec![]);

        let out = tf.make_default(vec![3, 0, 10, 3], vec![]);
        assert_eq!(out.numel(), 0);

        let ret = op_select_scatter_out(&x, &src, 2, 3, &out);
        assert_eq!(ret.numel(), 0);
    }

    // [spec:et:sem:op-select-scatter.torch.executor.native.select-scatter-out-fn/test]
    #[test]
    fn op_select_scatter_out_test_empty_tensor_zero_n_dims_input_dies() {
        let tf = TensorFactory::<i32>::new();

        let x = tf.make_default(vec![0], vec![]);
        assert_eq!(x.numel(), 0);

        let src = tf.make_default(vec![0], vec![]);
        assert_eq!(src.numel(), 0);

        let out = tf.make_default(vec![], vec![0]);
        assert_eq!(out.numel(), 1);

        let mut ctx = context();
        select_scatter_out(&mut ctx, &x, &src, 0, 0, &out);
        assert_ne!(ctx.failure_state(), Error::Ok);
    }

    // [spec:et:sem:op-select-scatter.torch.executor.native.select-scatter-out-fn/test]
    #[test]
    fn op_select_scatter_out_test_dim_out_of_bound_dies() {
        let tf = TensorFactory::<i32>::new();

        let x = tf.ones_default(vec![1, 1, 1]);
        let src = tf.ones_default(vec![1, 1]);

        let out = tf.zeros_default(vec![1, 1, 1]);

        let invalid_dims: Vec<i64> = vec![3, 4, 5, -4, -5, -6];
        for dim in invalid_dims {
            let mut ctx = context();
            select_scatter_out(&mut ctx, &x, &src, dim, 0, &out);
            assert_ne!(ctx.failure_state(), Error::Ok);
        }
    }

    // [spec:et:sem:op-select-scatter.torch.executor.native.select-scatter-out-fn/test]
    #[test]
    fn op_select_scatter_out_test_index_out_of_bound_dies() {
        let tf = TensorFactory::<i32>::new();

        let x = tf.ones_default(vec![1, 1, 1]);
        let src = tf.ones_default(vec![1, 1]);

        let out = tf.zeros_default(vec![1, 1, 1]);

        let invalid_indices: Vec<i64> = vec![3, 4, 5, -4, -5, -6];
        for idx in invalid_indices {
            let mut ctx = context();
            select_scatter_out(&mut ctx, &x, &src, 0, idx, &out);
            assert_ne!(ctx.failure_state(), Error::Ok);
        }
    }

    // [spec:et:sem:op-select-scatter.torch.executor.native.select-scatter-out-fn/test]
    #[test]
    fn op_select_scatter_out_test_mismatched_dtypes_dies() {
        let tf_int = TensorFactory::<i32>::new();
        let tf_float = TensorFactory::<f32>::new();
        let x = tf_int.zeros_default(vec![1, 2, 2]);
        let src = tf_int.zeros_default(vec![2, 2]);

        let out = tf_float.ones_default(vec![1, 2, 2]);

        let mut ctx = context();
        select_scatter_out(&mut ctx, &x, &src, 0, 0, &out);
        assert_ne!(ctx.failure_state(), Error::Ok);
    }

    // [spec:et:sem:op-select-scatter.torch.executor.native.select-scatter-out-fn/test]
    #[test]
    fn op_select_scatter_out_test_src_match_numel_lack_dim_at_end_dies() {
        let tf = TensorFactory::<i32>::new();
        let x = tf.zeros_default(vec![1, 2, 2, 1]);
        let src = tf.zeros_default(vec![2, 2]);

        let out = tf.ones_default(vec![1, 2, 2, 1]);

        let mut ctx = context();
        select_scatter_out(&mut ctx, &x, &src, 0, 0, &out);
        assert_ne!(ctx.failure_state(), Error::Ok);
    }

    // [spec:et:sem:op-select-scatter.torch.executor.native.select-scatter-out-fn/test]
    #[test]
    fn op_select_scatter_out_test_src_match_numel_extra_dim_at_front_dies() {
        let tf = TensorFactory::<i32>::new();
        let x = tf.zeros_default(vec![2, 2]);
        let src = tf.zeros_default(vec![1, 2]);

        let out = tf.ones_default(vec![2, 2]);

        let mut ctx = context();
        select_scatter_out(&mut ctx, &x, &src, 0, 0, &out);
        assert_ne!(ctx.failure_state(), Error::Ok);
    }

    // [spec:et:sem:op-select-scatter.torch.executor.native.select-scatter-out-fn/test]
    #[test]
    fn op_select_scatter_out_test_src_size_mismatch_dim_dies() {
        let tf = TensorFactory::<i32>::new();

        let x = tf.zeros_default(vec![2, 4, 7, 5]);
        let src = tf.zeros_default(vec![2, 4, 7]);

        let out = tf.zeros_default(vec![2, 4, 7, 5]);

        let mut ctx = context();
        select_scatter_out(&mut ctx, &x, &src, 2, 3, &out);
        assert_ne!(ctx.failure_state(), Error::Ok);
    }

    // [spec:et:sem:op-select-scatter.torch.executor.native.select-scatter-out-fn/test]
    #[test]
    fn op_select_scatter_out_test_dynamic_shape_upper_bound_same_as_expected() {
        test_dynamic_shape(vec![2, 3, 2], TensorShapeDynamism::DYNAMIC_BOUND);
    }

    // [spec:et:sem:op-select-scatter.torch.executor.native.select-scatter-out-fn/test]
    #[test]
    fn op_select_scatter_out_test_dynamic_shape_upper_bound_larger_than_expected() {
        test_dynamic_shape(vec![10, 10, 10], TensorShapeDynamism::DYNAMIC_BOUND);
    }

    // PORT-NOTE: C++ guards this with ET_SKIP_IF(!output_resize); the portable
    // build does not support DYNAMIC_UNBOUND resize, so the test is #[ignore]d.
    // [spec:et:sem:op-select-scatter.torch.executor.native.select-scatter-out-fn/test]
    #[test]
    #[ignore = "DynamicShapeUnbound: dynamic shape not supported"]
    fn op_select_scatter_out_test_dynamic_shape_unbound() {
        test_dynamic_shape(vec![1, 1, 1], TensorShapeDynamism::DYNAMIC_UNBOUND);
    }
}
