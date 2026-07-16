//! Literal port of kernels/portable/cpu/op_bmm.cpp.

use crate::kernels::portable::cpu::util::matmul_ops_util::internal::bmm_out_impl;
use crate::kernels::portable::cpu::util::matmul_ops_util::{
    check_bmm_args, get_bmm_out_target_size,
};
use crate::runtime::core::array_ref::ArrayRef;
use crate::runtime::core::error::Error;
use crate::runtime::core::exec_aten::util::scalar_type_util::is_complex_type;
use crate::runtime::core::exec_aten::util::tensor_util::{
    K_TENSOR_DIMENSION_LIMIT, resize_tensor_same_type, tensor_is_default_dim_order,
    tensors_have_same_dim_order3,
};
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::core::portable_type::tensor_impl::SizesType;
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

// PORT-NOTE: `Tensor& out` / returned `Tensor&` become `&'a Tensor` (interior
// mutation through `*mut TensorImpl`).
//
// PORT-NOTE (cross-module): the C++ dispatches complex input dtypes over
// `ET_SWITCH_COMPLEXH_TYPES` and calls `internal::bmm_out_impl<CTYPE>` on the
// complex CTYPE. `et_switch_complexh_types!` exists, but
// `matmul_ops_util::internal::BmmCtype` is implemented only for real dtypes
// (not `ComplexHalf`/`ComplexFloat`/`ComplexDouble`), so `bmm_out_impl::<CTYPE>`
// cannot be instantiated for complex CTYPE. The complex branch therefore fails
// with InvalidArgument as a placeholder; the real branch (REALHBF16) is ported
// faithfully. Unresolved cross-module reference: add `BmmCtype` impls for the
// complex portable types (fixer).

// [spec:et:def:op-bmm.torch.executor.native.bmm-out-fn]
// [spec:et:sem:op-bmm.torch.executor.native.bmm-out-fn]
pub fn bmm_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    in_: &Tensor,
    mat2: &Tensor,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    crate::et_kernel_check!(ctx, check_bmm_args(in_, mat2, out), InvalidArgument, out);

    crate::et_kernel_check!(
        ctx,
        tensors_have_same_dim_order3(in_, mat2, out),
        InvalidArgument,
        out
    );

    crate::et_kernel_check!(ctx, tensor_is_default_dim_order(in_), InvalidArgument, out);

    let mut output_ndim: usize = 0;
    let mut output_sizes: [SizesType; K_TENSOR_DIMENSION_LIMIT] = [0; K_TENSOR_DIMENSION_LIMIT];
    unsafe {
        get_bmm_out_target_size(in_, mat2, output_sizes.as_mut_ptr(), &mut output_ndim);
    }
    crate::et_kernel_check!(
        ctx,
        resize_tensor_same_type(
            out,
            ArrayRef::from_raw_parts(output_sizes.as_ptr(), output_ndim)
        ) == Error::Ok,
        InvalidArgument,
        out
    );

    let in_type = in_.scalar_type();

    if is_complex_type(in_type) {
        // PORT-NOTE: unrepresentable complex branch — see module note.
        // C++: ET_SWITCH_COMPLEXH_TYPES(in_type, ..., bmm_out_impl<CTYPE>(in, mat2, out));
        crate::et_kernel_check!(ctx, false, InvalidArgument, out);
    } else {
        crate::et_switch_realhbf16_types!(in_type, ctx, "bmm.out", CTYPE, {
            bmm_out_impl::<CTYPE>(in_, mat2, out);
        });
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::core::exec_aten::testing_util::tensor_factory::{
        FactoryValue, TensorFactory,
    };
    use crate::runtime::core::exec_aten::util::scalar_type_util::CppTypeToScalarType;
    use crate::runtime::core::memory_allocator::{MemoryAllocator, MemoryAllocatorBase};
    use crate::runtime::core::portable_type::{BFloat16, Half};
    use crate::runtime::core::tensor_shape_dynamism::TensorShapeDynamism;
    use crate::{assert_tensor_close, assert_tensor_eq};

    // Mirrors `OperatorTest::SetUp()` which runs `runtime_init()` (initializes
    // the PAL) then holds a default-constructed `KernelRuntimeContext`.
    fn context() -> KernelRuntimeContext<'static> {
        crate::runtime::platform::runtime::runtime_init();
        KernelRuntimeContext::new(
            crate::extension::module::module::null_event_tracer(),
            core::ptr::null_mut::<MemoryAllocator>() as *mut dyn MemoryAllocatorBase,
        )
    }

    // Mirrors `ET_EXPECT_KERNEL_FAILURE(context_, statement)`: run the statement,
    // then assert the context recorded a non-Ok failure state.
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

    fn test_dtype<T>()
    where
        T: CppTypeToScalarType + FactoryValue + FromI32,
    {
        let tf = TensorFactory::<T>::new();

        // Gives 4 * 2 * 3 = 24, shape (10, 3, 5)
        let x = tf.full(vec![10, 3, 4], T::from_i32(2), TensorShapeDynamism::STATIC);
        let y = tf.full(vec![10, 4, 5], T::from_i32(3), TensorShapeDynamism::STATIC);

        let out = tf.zeros_default(vec![10, 3, 5]);
        let mut ctx = context();
        bmm_out(&mut ctx, &x, &y, &out);

        let expected = tf.full(vec![10, 3, 5], T::from_i32(24), TensorShapeDynamism::STATIC);

        assert_tensor_eq!(out, expected);
    }

    // [spec:et:sem:op-bmm.torch.executor.native.bmm-out-fn/test]
    // also verifies get_bmm_out_target_size (out resized to [10,3,5] = [mat1.size(0),mat1.size(1),mat2.size(2)])
    // [spec:et:sem:matmul-ops-util.torch.executor.get-bmm-out-target-size-fn/test]
    #[test]
    fn op_bmm_out_test_output_dim() {
        let tf = TensorFactory::<i32>::new();

        let x = tf.ones_default(vec![10, 3, 4]);
        let y = tf.ones_default(vec![10, 4, 5]);

        let out = tf.zeros_default(vec![10, 3, 5]);

        let mut ctx = context();
        let ret = bmm_out(&mut ctx, &x, &y, &out);

        assert_tensor_eq!(*ret, out);

        let expected = tf.full(vec![10, 3, 5], 4, TensorShapeDynamism::STATIC);

        assert_tensor_eq!(out, expected);
    }

    // [spec:et:sem:op-bmm.torch.executor.native.bmm-out-fn/test]
    // also verifies internal::bmm_out_impl (per-batch matmul products asserted exactly)
    // [spec:et:sem:matmul-ops-util.torch.executor.internal.bmm-out-impl-fn/test]
    #[test]
    fn op_bmm_out_test_output_dim_float() {
        let tf = TensorFactory::<f32>::new();

        #[rustfmt::skip]
        let x = tf.make_default(
            vec![2, 4, 5],
            vec![
                4., 3., 1., 1., 1.,
                3., 1., 4., 4., 2.,
                1., 1., 1., 3., 3.,
                4., 2., 2., 2., 3.,

                1., 3., 1., 4., 4.,
                1., 1., 2., 4., 3.,
                4., 3., 4., 1., 2.,
                1., 4., 4., 4., 4.,
            ],
        );

        #[rustfmt::skip]
        let y = tf.make_default(
            vec![2, 5, 3],
            vec![
                4., 4., 4.,
                2., 3., 1.,
                1., 4., 4.,
                3., 1., 2.,
                1., 4., 3.,

                1., 4., 4.,
                4., 4., 4.,
                2., 1., 4.,
                1., 4., 3.,
                1., 4., 4.,
            ],
        );

        let out = tf.zeros_default(vec![2, 4, 3]);

        let mut ctx = context();
        let ret = bmm_out(&mut ctx, &x, &y, &out);

        assert_tensor_eq!(*ret, out);

        #[rustfmt::skip]
        let expected = tf.make_default(
            vec![2, 4, 3],
            vec![
                27., 34., 28.,
                32., 43., 43.,
                19., 26., 24.,
                31., 44., 39.,

                23., 49., 48.,
                16., 38., 40.,
                27., 44., 55.,
                33., 56., 64.,
            ],
        );

        assert_tensor_eq!(out, expected);
    }

    // [spec:et:sem:op-bmm.torch.executor.native.bmm-out-fn/test]
    #[test]
    fn op_bmm_out_test_all_real_dtypes_supported() {
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

    // PORT-NOTE: `AllComplexDtypesSupported` exercises complex element dtypes via
    // `test_complex_dtype`. The Rust `TensorFactory` has no complex element type
    // and `op_bmm`'s complex branch is an unimplemented placeholder (see module
    // note). Ported but ignored until complex support lands.
    // [spec:et:sem:op-bmm.torch.executor.native.bmm-out-fn/test]
    #[test]
    #[ignore = "complex dtypes unsupported by TensorFactory / op_bmm complex branch"]
    fn op_bmm_out_test_all_complex_dtypes_supported() {
        // Unrepresentable: complex CTYPE (2, 2, 3) x (2, 3, 2) -> (2, 2, 2).
    }

    // [spec:et:sem:op-bmm.torch.executor.native.bmm-out-fn/test]
    #[test]
    fn op_bmm_out_test_empty_input_with_empty_out_tensor_passes() {
        let tf = TensorFactory::<i32>::new();

        let x = tf.full(vec![2, 2, 2], 3, TensorShapeDynamism::STATIC);
        let y = tf.make_default(vec![2, 2, 0], vec![]);

        let out = tf.make_default(vec![2, 2, 0], vec![]);

        assert_eq!(out.numel(), 0);

        let mut ctx = context();
        bmm_out(&mut ctx, &x, &y, &out);

        assert_eq!(out.numel(), 0);
    }

    // [spec:et:sem:op-bmm.torch.executor.native.bmm-out-fn/test]
    // also verifies check_bmm_args rejects batch/inner-dim mismatch (x=[2,10,3] vs wrong_y=[3,7,4])
    // [spec:et:sem:matmul-ops-util.torch.executor.check-bmm-args-fn/test]
    #[test]
    fn op_bmm_out_test_mismatched_dimensions_dies() {
        let tf = TensorFactory::<i32>::new();

        let x = tf.ones_default(vec![2, 10, 3]);

        let wrong_y = tf.ones_default(vec![3, 7, 4]);
        let right_y = tf.ones_default(vec![2, 3, 4]);

        let out = tf.ones_default(vec![2, 10, 4]);

        let mut ctx = context();
        et_expect_kernel_failure!(ctx, bmm_out(&mut ctx, &x, &wrong_y, &out));

        let mut ctx = context();
        assert_tensor_eq!(
            *bmm_out(&mut ctx, &x, &right_y, &out),
            tf.full(vec![2, 10, 4], 3, TensorShapeDynamism::STATIC)
        );
    }

    // PORT-NOTE: guarded in C++ by `ET_SKIP_IF(is_aten, ...)`; this is the
    // non-ATen (portable) build, so it always runs.
    // [spec:et:sem:op-bmm.torch.executor.native.bmm-out-fn/test]
    #[test]
    fn op_bmm_out_test_mismatched_dimension_size_dies() {
        let tf = TensorFactory::<i32>::new();

        let x = tf.ones_default(vec![2, 10, 3]);

        let y = tf.ones_default(vec![2, 3, 4]);

        let wrong_y = tf.ones_default(vec![7, 4]);
        let right_y = tf.ones_default(vec![2, 3, 4]);

        let right_out = tf.ones_default(vec![2, 10, 4]);
        let wrong_out = tf.ones_default(vec![7, 5]);

        let _ = (&y, &right_y);

        let mut ctx = context();
        et_expect_kernel_failure!(ctx, bmm_out(&mut ctx, &x, &right_y, &wrong_out));

        let mut ctx = context();
        et_expect_kernel_failure!(ctx, bmm_out(&mut ctx, &x, &wrong_y, &right_out));
    }

    // PORT-NOTE: guarded in C++ by `ET_SKIP_IF(is_aten, ...)`; non-ATen build runs.
    // [spec:et:sem:op-bmm.torch.executor.native.bmm-out-fn/test]
    #[test]
    fn op_bmm_out_test_wrong_out_shape_dies() {
        let tf = TensorFactory::<i32>::new();

        let x = tf.ones_default(vec![2, 10, 3]);

        let y = tf.ones_default(vec![2, 3, 4]);

        let right_out = tf.ones_default(vec![2, 10, 4]);
        let wrong_out = tf.ones_default(vec![3, 7, 5]);

        let mut ctx = context();
        et_expect_kernel_failure!(ctx, bmm_out(&mut ctx, &x, &y, &wrong_out));

        let mut ctx = context();
        assert_tensor_eq!(
            *bmm_out(&mut ctx, &x, &y, &right_out),
            tf.full(vec![2, 10, 4], 3, TensorShapeDynamism::STATIC)
        );
    }

    #[rustfmt::skip]
    fn dynamic_shape_inputs(tf: &TensorFactory<f32>) -> (Tensor<'_>, Tensor<'_>, Tensor<'_>) {
        let x = tf.make_default(
            vec![3, 3, 6],
            vec![
                0.7231091856956482,    0.7423362731933594,  0.5262957811355591,
                0.24365824460983276,   0.584592342376709,   0.033152639865875244,
                0.13871687650680542,   0.242235004901886,   0.815468966960907,
                0.793160617351532,     0.2782524824142456,  0.48195880651474,
                0.8197803497314453,    0.9970665574073792,  0.6984410881996155,
                0.5675464272499084,    0.8352431654930115,  0.2055988311767578,
                0.593172013759613,     0.11234724521636963, 0.1534569263458252,
                0.24170821905136108,   0.7262365221977234,  0.7010802030563354,
                0.2038237452507019,    0.6510535478591919,  0.7744860053062439,
                0.4368913173675537,    0.5190907716751099,  0.6158523559570312,
                0.8101882934570312,    0.9800970554351807,  0.1146882176399231,
                0.3167651295661926,    0.6965049505233765,  0.9142746925354004,
                0.9351036548614502,    0.9411783814430237,  0.5995072722434998,
                0.06520867347717285,   0.5459962487220764,  0.18719732761383057,
                0.03402292728424072,   0.944246232509613,   0.8801798820495605,
                0.0012360215187072754, 0.5935860276222229,  0.4157699942588806,
                0.41771942377090454,   0.2711215615272522,  0.6922780871391296,
                0.2038482427597046,    0.6832956671714783,  0.75285404920578,
            ],
        );
        let y = tf.make_default(
            vec![3, 6, 2],
            vec![
                0.8579357862472534,   0.6869555711746216,  0.0051323771476745605,
                0.17565155029296875,  0.7496575117111206,  0.6046506762504578,
                0.1099579930305481,   0.21209025382995605, 0.9703746438026428,
                0.8369089365005493,   0.28198742866516113, 0.3741576075553894,
                0.023700952529907227, 0.49101293087005615, 0.12347054481506348,
                0.11432164907455444,  0.4724501967430115,  0.5750725269317627,
                0.2952348589897156,   0.7966887950897217,  0.19573044776916504,
                0.9536850452423096,   0.8426499366760254,  0.07835853099822998,
                0.3755578398704529,   0.5225613117218018,  0.572950541973114,
                0.6185871362686157,   0.6962141394615173,  0.5299500823020935,
                0.25603562593460083,  0.7365944981575012,  0.020375549793243408,
                0.20364665985107422,  0.3748350739479065,  0.2564433217048645,
            ],
        );
        let expected_result = tf.make_default(
            vec![3, 3, 2],
            vec![
                1.6221470832824707,
                1.498693823814392,
                1.224705696105957,
                1.2123372554779053,
                2.1629090309143066,
                2.05692195892334,
                0.9047035574913025,
                1.3324503898620605,
                1.2006582021713257,
                1.5112680196762085,
                1.1946606636047363,
                1.5640640258789062,
                1.405808448791504,
                1.5957869291305542,
                1.3348338603973389,
                1.2967426776885986,
                1.1425018310546875,
                1.2352378368377686,
            ],
        );
        (x, y, expected_result)
    }

    // [spec:et:sem:op-bmm.torch.executor.native.bmm-out-fn/test]
    #[test]
    fn op_bmm_out_test_dynamic_shape_upper_bound_same_as_expected() {
        let tf = TensorFactory::<f32>::new();
        let (x, y, expected_result) = dynamic_shape_inputs(&tf);

        let out = tf.zeros(vec![3, 3, 2], TensorShapeDynamism::DYNAMIC_BOUND);
        let mut ctx = context();
        let _ret = bmm_out(&mut ctx, &x, &y, &out);
        assert_tensor_close!(out, expected_result);
    }

    // [spec:et:sem:op-bmm.torch.executor.native.bmm-out-fn/test]
    #[test]
    fn op_bmm_out_test_dynamic_shape_upper_bound_larger_than_expected() {
        let tf = TensorFactory::<f32>::new();
        let (x, y, expected_result) = dynamic_shape_inputs(&tf);

        let out = tf.zeros(vec![6, 6, 4], TensorShapeDynamism::DYNAMIC_BOUND);
        let mut ctx = context();
        let _ret = bmm_out(&mut ctx, &x, &y, &out);
        assert_tensor_close!(out, expected_result);
    }

    // PORT-NOTE: DISABLED in C++ (`DISABLED_DynamicShapeUnbound`): dynamic shape
    // unbound not supported. Ported and ignored to match.
    // [spec:et:sem:op-bmm.torch.executor.native.bmm-out-fn/test]
    #[test]
    #[ignore = "DISABLED_DynamicShapeUnbound: dynamic shape unbound not supported"]
    fn op_bmm_out_test_dynamic_shape_unbound() {
        let tf = TensorFactory::<f32>::new();
        let (x, y, expected_result) = dynamic_shape_inputs(&tf);

        let out = tf.zeros(vec![1, 1, 1], TensorShapeDynamism::DYNAMIC_UNBOUND);
        let mut ctx = context();
        let _ret = bmm_out(&mut ctx, &x, &y, &out);
        assert_tensor_close!(out, expected_result);
    }
}
