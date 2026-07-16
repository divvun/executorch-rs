//! Literal port of kernels/portable/cpu/op_embedding.cpp.

use crate::kernels::portable::cpu::util::kernel_ops_util::{
    check_embedding_args, resize_embedding_output,
};
use crate::runtime::core::error::Error;
use crate::runtime::core::exec_aten::util::tensor_util::{
    tensor_is_default_dim_order, tensors_have_same_dim_order3,
};
use crate::runtime::core::portable_type::scalar_type::ScalarType;
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

// PORT-NOTE: `Tensor& out` / returned `Tensor&` become `&'a Tensor` (interior
// mutation through `*mut TensorImpl`). The `(void)`-casts are dropped.
//
// PORT-NOTE: local `et_check_msg!` mirroring the C++ fatal `ET_CHECK_MSG` for the
// `ix_type` guard (message dropped, a fatal abort follows), matching the
// established per-module definitions (see tensor_util.rs / scalar_type_util.rs).
macro_rules! et_check_msg {
    ($cond:expr, $($arg:tt)*) => {
        if !($cond) {
            crate::runtime::platform::abort::runtime_abort();
        }
    };
}

// A simple lookup table that looks up embeddings in a fixed dictionary and
// size.

// PORT-NOTE: `CTYPE` (the index element type) is a type parameter here rather
// than a template; the `embedding_out` dispatch monomorphizes it to `i64` (Long)
// or `i32` (Int). Raw byte pointers (`char*` in C++) are `*const u8` / `*mut u8`.
// The `ET_KERNEL_CHECK_MSG` format args are dropped (message-only), matching the
// ported macro; on failure the kernel returns void, leaving `out` partially
// written — same as C++.
// [spec:et:def:op-embedding.torch.executor.native.embedding-kernel-fn]
// [spec:et:sem:op-embedding.torch.executor.native.embedding-kernel-fn]
fn embedding_kernel<CTYPE>(
    ctx: &mut KernelRuntimeContext,
    weight: &Tensor,
    indices: &Tensor,
    out: &Tensor,
) where
    CTYPE: Copy + PartialOrd + Zero + AsI64,
{
    let nbytes_per_entry: i64 = weight.size(1) as i64 * weight.element_size() as i64;
    let w_data: *const u8 = weight.const_data_ptr::<u8>();
    let mut out_data: *mut u8 = out.mutable_data_ptr::<u8>();
    let indices_ptr: *const CTYPE = indices.const_data_ptr::<CTYPE>();
    let weight_height: isize = weight.size(0) as isize;
    let indices_numel = indices.numel();
    for i in 0..indices_numel {
        let idx: CTYPE = unsafe { *indices_ptr.offset(i as isize) };
        // Ensure index is larger than 0 and smaller than weight.size(0)
        crate::et_kernel_check_msg!(
            ctx,
            (idx.as_i64() as isize) < weight_height,
            InvalidArgument,
            (),
            "indices_ptr[i] >= weight.size(0)"
        );
        crate::et_kernel_check_msg!(
            ctx,
            !(idx < CTYPE::zero()),
            InvalidArgument,
            (),
            "indices_ptr[i] < 0"
        );
        if !w_data.is_null() {
            unsafe {
                core::ptr::copy_nonoverlapping(
                    w_data.offset((nbytes_per_entry * idx.as_i64()) as isize),
                    out_data,
                    nbytes_per_entry as usize,
                );
            }
        }
        out_data = unsafe { out_data.offset(nbytes_per_entry as isize) };
    }
}

// PORT-NOTE: the index element types are `i64`/`i32`; `Zero` supplies the `0`
// literal for the `>= 0` check and `AsI64` reproduces the `static_cast<long>` /
// pointer-arithmetic widening (`nbytes_per_entry * indices_ptr[i]`).
trait Zero {
    fn zero() -> Self;
}
trait AsI64 {
    fn as_i64(self) -> i64;
}
impl Zero for i64 {
    fn zero() -> Self {
        0
    }
}
impl Zero for i32 {
    fn zero() -> Self {
        0
    }
}
impl AsI64 for i64 {
    fn as_i64(self) -> i64 {
        self
    }
}
impl AsI64 for i32 {
    fn as_i64(self) -> i64 {
        self as i64
    }
}

// embedding.out(Tensor weight, Tensor indices, int padding_idx=-1, bool
// scale_grad_by_freq=False, bool sparse=False, *, Tensor(a!) out) -> Tensor(a!)
// [spec:et:def:op-embedding.torch.executor.native.embedding-out-fn]
// [spec:et:sem:op-embedding.torch.executor.native.embedding-out-fn]
#[executorch_macros::et_kernel("aten::embedding.out")]
pub fn embedding_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    weight: &Tensor,
    indices: &Tensor,
    _padding_idx: i64,
    _scale_grad_by_freq: bool,
    _sparse: bool,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    crate::et_kernel_check!(
        ctx,
        check_embedding_args(weight, indices, out),
        InvalidArgument,
        out
    );

    crate::et_kernel_check!(
        ctx,
        resize_embedding_output(weight, indices, out) == Error::Ok,
        InvalidArgument,
        out
    );

    crate::et_kernel_check_msg!(
        ctx,
        out.size(out.dim() - 1) == weight.size(1),
        InvalidArgument,
        out,
        "out.size(...) != weight.size(1)"
    );

    crate::et_kernel_check!(
        ctx,
        tensors_have_same_dim_order3(weight, indices, out),
        InvalidArgument,
        out
    );

    crate::et_kernel_check!(
        ctx,
        tensor_is_default_dim_order(weight),
        InvalidArgument,
        out
    );

    let ix_type: ScalarType = indices.scalar_type();
    et_check_msg!(
        ix_type == ScalarType::Long || ix_type == ScalarType::Int,
        "Expected indices tensor to have Long or Int scalar types"
    );

    let op_name = "op_embedding.out";

    crate::et_switch_two_types!(Long, Int, ix_type, ctx, op_name, CTYPE, {
        embedding_kernel::<CTYPE>(ctx, weight, indices, out);
    });

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assert_tensor_eq;
    use crate::runtime::core::exec_aten::testing_util::tensor_factory::{
        FactoryValue, TensorFactory,
    };
    use crate::runtime::core::exec_aten::util::scalar_type_util::CppTypeToScalarType;
    use crate::runtime::core::memory_allocator::{MemoryAllocator, MemoryAllocatorBase};
    use crate::runtime::core::tensor_shape_dynamism::TensorShapeDynamism;

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

    fn op_embedding_out<'a, 'b>(
        weight: &Tensor,
        indices: &Tensor,
        padding_idx: i64,
        scale_grad_by_freq: bool,
        sparse: bool,
        out: &'a Tensor<'b>,
    ) -> &'a Tensor<'b> {
        let mut ctx = context();
        embedding_out(
            &mut ctx,
            weight,
            indices,
            padding_idx,
            scale_grad_by_freq,
            sparse,
            out,
        )
    }

    trait FromI32: Copy {
        fn from_i32(v: i32) -> Self;
    }
    macro_rules! impl_from_i32_num {
        ($($t:ty),*) => {$(impl FromI32 for $t { fn from_i32(v: i32) -> Self { v as $t } })*};
    }
    impl_from_i32_num!(u8, i8, i16, i32, i64, f32, f64);

    fn test_dtype<T>()
    where
        T: CppTypeToScalarType + FactoryValue + FromI32,
    {
        let tf = TensorFactory::<T>::new();
        let tfl = TensorFactory::<i64>::new();
        let weight = tf.make_default(
            vec![3, 2],
            vec![
                T::from_i32(1),
                T::from_i32(2),
                T::from_i32(3),
                T::from_i32(4),
                T::from_i32(5),
                T::from_i32(6),
            ],
        );
        let indices = tfl.make_default(vec![1, 2], vec![0, 2]);
        let out = tf.zeros_default(vec![1, 2, 2]);
        let actual = op_embedding_out(&weight, &indices, 0, false, false, &out);

        let expected = tf.make_default(
            vec![1, 2, 2],
            vec![
                T::from_i32(1),
                T::from_i32(2),
                T::from_i32(5),
                T::from_i32(6),
            ],
        );

        assert_tensor_eq!(actual, out);
        assert_tensor_eq!(out, expected);
    }

    fn test_dynamic_shape(out_shape: Vec<i32>, dynamism: TensorShapeDynamism) {
        let tf_weight = TensorFactory::<f32>::new();
        let tf_indices = TensorFactory::<i64>::new();

        let weight = tf_weight.make_default(
            vec![10, 3],
            vec![
                0.49625658988952637,
                0.7682217955589294,
                0.08847743272781372,
                0.13203048706054688,
                0.30742281675338745,
                0.6340786814689636,
                0.4900934100151062,
                0.8964447379112244,
                0.455627977848053,
                0.6323062777519226,
                0.3488934636116028,
                0.40171730518341064,
                0.022325754165649414,
                0.16885894536972046,
                0.2938884496688843,
                0.518521785736084,
                0.6976675987243652,
                0.800011396408081,
                0.16102945804595947,
                0.28226858377456665,
                0.6816085577011108,
                0.9151939749717712,
                0.39709991216659546,
                0.8741558790206909,
                0.41940832138061523,
                0.5529070496559143,
                0.9527381062507629,
                0.036164820194244385,
                0.1852310299873352,
                0.37341737747192383,
            ],
        );
        let indices = tf_indices.make_default(vec![2, 4], vec![1, 2, 4, 5, 4, 3, 2, 9]);
        let expected = tf_weight.make_default(
            vec![2, 4, 3],
            vec![
                0.13203048706054688,
                0.30742281675338745,
                0.6340786814689636,
                0.4900934100151062,
                0.8964447379112244,
                0.455627977848053,
                0.022325754165649414,
                0.16885894536972046,
                0.2938884496688843,
                0.518521785736084,
                0.6976675987243652,
                0.800011396408081,
                0.022325754165649414,
                0.16885894536972046,
                0.2938884496688843,
                0.6323062777519226,
                0.3488934636116028,
                0.40171730518341064,
                0.4900934100151062,
                0.8964447379112244,
                0.455627977848053,
                0.036164820194244385,
                0.1852310299873352,
                0.37341737747192383,
            ],
        );
        let out = tf_weight.zeros(out_shape, dynamism);

        op_embedding_out(&weight, &indices, 0, false, false, &out);
        assert!(
            crate::runtime::core::exec_aten::testing_util::tensor_util::tensors_are_close(
                &out,
                &expected,
                crate::runtime::core::exec_aten::testing_util::tensor_util::internal::K_DEFAULT_RTOL,
                None
            )
        );
    }

    // [spec:et:sem:op-embedding.torch.executor.native.embedding-out-fn/test]
    // Gathering weight row 1 -> [0.5, 0.6] pins embedding_kernel's per-index copy.
    // [spec:et:sem:op-embedding.torch.executor.native.embedding-out-fn/test]
    // [spec:et:sem:op-embedding.torch.executor.native.embedding-kernel-fn/test]
    #[test]
    fn op_embedding_out_test_smoke() {
        let tff = TensorFactory::<f32>::new();
        let weight = tff.make_default(vec![2, 2], vec![1.0, 2.0, 0.5, 0.6]);
        let out = tff.zeros_default(vec![1, 2]);
        let tfl = TensorFactory::<i64>::new();
        let indices = tfl.make_default(vec![1], vec![1]);
        let actual = op_embedding_out(&weight, &indices, 0, false, false, &out);
        assert_tensor_eq!(actual, out);
        assert_tensor_eq!(out, tff.make_default(vec![1, 2], vec![0.5, 0.6]));
    }

    // [spec:et:sem:op-embedding.torch.executor.native.embedding-out-fn/test]
    #[test]
    fn op_embedding_out_test_all_dtypes_supported() {
        test_dtype::<u8>();
        test_dtype::<i8>();
        test_dtype::<i16>();
        test_dtype::<i32>();
        test_dtype::<i64>();
        test_dtype::<f32>();
        test_dtype::<f64>();
    }

    // [spec:et:sem:op-embedding.torch.executor.native.embedding-out-fn/test]
    // Multi-dim indices exercise the embedding arg check and output resize
    // (out shape = indices dims + embedding_dim); wrong helper -> wrong out shape.
    // [spec:et:sem:kernel-ops-util.torch.executor.check-embedding-args-fn/test]
    // [spec:et:sem:kernel-ops-util.torch.executor.resize-embedding-output-fn/test]
    #[test]
    fn op_embedding_out_test_indices_multi_dims() {
        let tff = TensorFactory::<f32>::new();
        let weight = tff.make_default(
            vec![5, 2],
            vec![1.0, 2.0, 0.5, 0.6, 0.1, 0.2, 3.0, 4.0, 5.0, 6.0],
        );
        let out = tff.zeros_default(vec![1, 2, 3, 2]);
        let tfl = TensorFactory::<i64>::new();
        let indices = tfl.make_default(vec![1, 2, 3], vec![1, 0, 2, 3, 4, 0]);
        let actual = op_embedding_out(&weight, &indices, 0, false, false, &out);
        assert_tensor_eq!(actual, out);
        assert_tensor_eq!(
            out,
            tff.make_default(
                vec![1, 2, 3, 2],
                vec![
                    0.5, 0.6, // weight[1]
                    1.0, 2.0, // weight[0]
                    0.1, 0.2, // weight[2]
                    3.0, 4.0, // weight[3]
                    5.0, 6.0, // weight[4]
                    1.0, 2.0, // weight[0]
                ]
            )
        );
    }

    // [spec:et:sem:op-embedding.torch.executor.native.embedding-out-fn/test]
    #[test]
    fn op_embedding_out_test_weight_wrong_dimensions_dies() {
        let tff = TensorFactory::<f32>::new();
        let weight = tff.make_default(vec![2, 2, 2], vec![1.0, 2.0, 0.5, 0.6, 0.1, 0.2, 3.0, 4.0]);
        let out = tff.zeros_default(vec![2, 2, 2]);
        let tfl = TensorFactory::<i64>::new();
        let indices = tfl.make_default(vec![2, 2], vec![1, 0, 2, 3]);
        let mut ctx = context();
        embedding_out(&mut ctx, &weight, &indices, 0, false, false, &out);
        assert_ne!(ctx.failure_state(), Error::Ok);
    }

    // PORT-NOTE: C++ `ET_SKIP_IF(is_aten, ...)`: never ATen, so the failure runs.
    // [spec:et:sem:op-embedding.torch.executor.native.embedding-out-fn/test]
    #[test]
    fn op_embedding_out_test_wrong_out_shape_dies() {
        let tff = TensorFactory::<f32>::new();
        let weight = tff.make_default(
            vec![5, 2],
            vec![1.0, 2.0, 0.5, 0.6, 0.1, 0.2, 3.0, 4.0, 5.0, 6.0],
        );
        let tfl = TensorFactory::<i64>::new();
        let indices = tfl.make_default(vec![2, 2], vec![1, 0, 2, 3]);

        let wrong_out0 = tff.zeros_default(vec![4, 3]);
        let wrong_out1 = tff.zeros_default(vec![4, 2]);
        let wrong_out2 = tff.zeros_default(vec![4, 2, 2]);
        for wrong_out in [&wrong_out0, &wrong_out1, &wrong_out2] {
            let mut ctx = context();
            embedding_out(&mut ctx, &weight, &indices, 0, false, false, wrong_out);
            assert_ne!(ctx.failure_state(), Error::Ok);
        }
    }

    // [spec:et:sem:op-embedding.torch.executor.native.embedding-out-fn/test]
    #[test]
    fn op_embedding_out_test_unmatched_out_type_die() {
        let tff = TensorFactory::<f32>::new();
        let tfl = TensorFactory::<i64>::new();
        let weight = tff.make_default(
            vec![5, 2],
            vec![1.0, 2.0, 0.5, 0.6, 0.1, 0.2, 3.0, 4.0, 5.0, 6.0],
        );
        let wrong_out = tfl.zeros_default(vec![2, 2, 2]);
        let indices = tfl.make_default(vec![2, 2], vec![1, 0, 2, 3]);
        let mut ctx = context();
        embedding_out(&mut ctx, &weight, &indices, 0, false, false, &wrong_out);
        assert_ne!(ctx.failure_state(), Error::Ok);
    }

    // [spec:et:sem:op-embedding.torch.executor.native.embedding-out-fn/test]
    #[test]
    fn op_embedding_out_test_out_of_bound_indices_dies() {
        let tff = TensorFactory::<f32>::new();
        let weight = tff.make_default(
            vec![5, 2],
            vec![1.0, 2.0, 0.5, 0.6, 0.1, 0.2, 3.0, 4.0, 5.0, 6.0],
        );
        let out = tff.zeros_default(vec![2, 2, 2]);
        let tfl = TensorFactory::<i64>::new();

        let neg_indices = tfl.make_default(vec![2, 2], vec![-1, 0, 2, 4]);
        let overflow_indices = tfl.make_default(vec![2, 2], vec![1, 0, 2, 8]);

        let mut ctx = context();
        embedding_out(&mut ctx, &weight, &neg_indices, 0, false, false, &out);
        assert_ne!(ctx.failure_state(), Error::Ok);

        let mut ctx = context();
        embedding_out(&mut ctx, &weight, &overflow_indices, 0, false, false, &out);
        assert_ne!(ctx.failure_state(), Error::Ok);
    }

    // [spec:et:sem:op-embedding.torch.executor.native.embedding-out-fn/test]
    #[test]
    fn op_embedding_out_test_empty_weight_supported() {
        let tff = TensorFactory::<f32>::new();
        let weight = tff.make_default(vec![5, 0], vec![]);
        let out = tff.ones_default(vec![2, 2, 0]);
        let tfl = TensorFactory::<i64>::new();
        let indices = tfl.make_default(vec![2, 2], vec![2, 0, 2, 4]);
        let actual = op_embedding_out(&weight, &indices, 0, false, false, &out);
        assert_tensor_eq!(actual, out);
        assert_tensor_eq!(actual, tff.zeros_default(vec![2, 2, 0]));
    }

    // [spec:et:sem:op-embedding.torch.executor.native.embedding-out-fn/test]
    #[test]
    fn op_embedding_out_test_zero_dim_indices_supported() {
        let tff = TensorFactory::<f32>::new();
        let weight = tff.make_default(
            vec![5, 2],
            vec![1.0, 2.0, 0.5, 0.6, 0.1, 0.2, 3.0, 4.0, 5.0, 6.0],
        );
        let out = tff.zeros_default(vec![2]);
        let tfl = TensorFactory::<i64>::new();
        let indices = tfl.make_default(vec![], vec![3]);
        let expected = tff.make_default(vec![2], vec![3.0, 4.0]);
        let actual = op_embedding_out(&weight, &indices, 0, false, false, &out);
        assert_tensor_eq!(actual, out);
        assert_tensor_eq!(out, expected);
    }

    // [spec:et:sem:op-embedding.torch.executor.native.embedding-out-fn/test]
    #[test]
    fn op_embedding_out_test_empty_dim_indices_supported() {
        let tff = TensorFactory::<f32>::new();
        let weight = tff.make_default(
            vec![5, 2],
            vec![1.0, 2.0, 0.5, 0.6, 0.1, 0.2, 3.0, 4.0, 5.0, 6.0],
        );
        let out = tff.zeros_default(vec![3, 0, 2]);
        let tfl = TensorFactory::<i64>::new();
        let indices = tfl.make_default(vec![3, 0], vec![]);
        let expected = tff.make_default(vec![3, 0, 2], vec![]);
        let actual = op_embedding_out(&weight, &indices, 0, false, false, &out);
        assert_tensor_eq!(actual, out);
        assert_tensor_eq!(out, expected);
    }

    // [spec:et:sem:op-embedding.torch.executor.native.embedding-out-fn/test]
    #[test]
    fn op_embedding_out_test_dynamic_shape_upper_bound_same_as_expected() {
        test_dynamic_shape(vec![2, 4, 3], TensorShapeDynamism::DYNAMIC_BOUND);
    }

    // [spec:et:sem:op-embedding.torch.executor.native.embedding-out-fn/test]
    #[test]
    fn op_embedding_out_test_dynamic_shape_upper_bound_larger_than_expected() {
        test_dynamic_shape(vec![10, 10, 10], TensorShapeDynamism::DYNAMIC_BOUND);
    }

    // PORT-NOTE: C++ `ET_SKIP_IF(!output_resize, ...)`: portable's `output_resize`
    // SupportedFeature is false, so this test is skipped in the portable build.
    // [spec:et:sem:op-embedding.torch.executor.native.embedding-out-fn/test]
    #[test]
    #[ignore = "SKIP_IF(!output_resize): portable kernel does not support output resize"]
    fn op_embedding_out_test_dynamic_shape_unbound() {
        test_dynamic_shape(vec![1, 1, 1], TensorShapeDynamism::DYNAMIC_UNBOUND);
    }
}
