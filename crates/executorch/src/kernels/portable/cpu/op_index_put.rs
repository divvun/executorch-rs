//! Literal port of kernels/portable/cpu/op_index_put.cpp.

use crate::kernels::portable::cpu::util::advanced_index_util::{
    TensorOptList, check_index_args, compute_dim_map, compute_index_map, count_index_blocks,
    get_in_coord, get_index_out_target_size, get_indices_broadcast_ndim,
    get_num_leading_null_indices,
};
use crate::kernels::portable::cpu::util::broadcast_util::{
    apply_binary_elementwise_fn, linearize_access_indexes_tensor, tensor_is_broadcastable_to,
};
use crate::kernels::portable::cpu::util::delinearize_index::delinearize_index;
use crate::runtime::core::array_ref::ArrayRef;
use crate::runtime::core::error::Error;
use crate::runtime::core::exec_aten::util::tensor_util::{
    K_TENSOR_DIMENSION_LIMIT, coordinateToIndex, getLeadingDims, getTrailingDims, resize_tensor,
    tensor_has_expected_size, tensor_is_default_dim_order, tensors_have_same_dim_order2,
    tensors_have_same_dtype2,
};
use crate::runtime::core::portable_type::scalar_type::ScalarType;
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::core::portable_type::tensor_impl::{SizesType, ssize_t};
use crate::runtime::core::portable_type::{BFloat16, Half};
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

// PORT-NOTE: C++ `val_in + val` / `out_data[in_ix] += ...` run over REALHBBF16
// (including `bool`). `bool + bool` is not valid in Rust; mirroring op_cumsum.rs's
// `CumsumAdd`, a local `IndexPutAdd` trait reproduces the C++ `a + b` promote-and-
// narrow semantics per ctype (bool: promote to int, add, narrow-to-bool).
trait IndexPutAdd: Copy {
    fn index_put_add(self, b: Self) -> Self;
}
macro_rules! impl_index_put_add {
    ($t:ty) => {
        impl IndexPutAdd for $t {
            fn index_put_add(self, b: Self) -> Self {
                self + b
            }
        }
    };
}
impl_index_put_add!(u8);
impl_index_put_add!(i8);
impl_index_put_add!(i16);
impl_index_put_add!(i32);
impl_index_put_add!(i64);
impl_index_put_add!(f32);
impl_index_put_add!(f64);
impl_index_put_add!(Half);
impl_index_put_add!(BFloat16);
impl IndexPutAdd for bool {
    fn index_put_add(self, b: Self) -> Self {
        (self as i32 + b as i32) != 0
    }
}

// PORT-NOTE: local `et_check_or_return_false!` mirroring the C++
// `ET_CHECK_OR_RETURN_FALSE` (crate-level check macros drop format args), per the
// established per-module definitions.
macro_rules! et_check_or_return_false {
    ($cond:expr, $($arg:tt)*) => {{
        if !($cond) {
            $crate::et_log!(Error, $($arg)*);
            return false;
        }
    }};
}

// [spec:et:def:op-index-put.torch.executor.native.index-put-out-fn]
// [spec:et:sem:op-index-put.torch.executor.native.index-put-out-fn]
pub fn index_put_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    in_: &Tensor,
    indices: TensorOptList,
    values: &Tensor,
    accumulate: bool,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    // (void)ctx;

    crate::et_kernel_check!(
        ctx,
        check_index_args(in_, indices, out),
        InvalidArgument,
        out
    );

    crate::et_kernel_check!(
        ctx,
        tensors_have_same_dtype2(in_, values),
        InvalidArgument,
        out
    );

    crate::et_kernel_check!(
        ctx,
        tensors_have_same_dim_order2(in_, out),
        InvalidArgument,
        out
    );

    crate::et_kernel_check!(ctx, tensor_is_default_dim_order(in_), InvalidArgument, out);

    let in_type: ScalarType = in_.scalar_type();
    let block_count: usize = count_index_blocks(indices);

    // If indices list is empty or all indices are null, then the operation is
    // performed over then entire input tensor. So, this is equivalent to
    // out = values when accumulate is false. Otherwise, the operation is
    // out = in + values where accumulate is true.
    if block_count == 0 {
        crate::et_kernel_check!(
            ctx,
            resize_tensor(out, in_.sizes()) == Error::Ok,
            InvalidArgument,
            out
        );

        // Check that values tensors can be broadcasted to out
        crate::et_kernel_check!(
            ctx,
            tensor_is_broadcastable_to(values.sizes(), out.sizes()),
            InvalidArgument,
            out
        );

        crate::et_switch_realhbbf16_types!(in_type, ctx, "index_put.out", CTYPE, {
            apply_binary_elementwise_fn::<CTYPE, CTYPE, CTYPE, _>(
                |val_in: CTYPE, val: CTYPE| -> CTYPE {
                    if accumulate {
                        val_in.index_put_add(val)
                    } else {
                        val
                    }
                },
                in_,
                values,
                out,
            );
        });
        return out;
    }

    // The index output shape depends on whether all the non-null indices are
    // adjacent or not.
    let adjacent: bool = block_count == 1;

    // Compute the expected index output shape.
    let mut x_sizes: [SizesType; K_TENSOR_DIMENSION_LIMIT] = [0; K_TENSOR_DIMENSION_LIMIT];
    let mut x_dim: usize = 0;
    crate::et_kernel_check!(
        ctx,
        unsafe {
            get_index_out_target_size(in_, indices, adjacent, x_sizes.as_mut_ptr(), &mut x_dim)
        },
        InvalidArgument,
        out
    );

    // Check that values tensors can be broadcasted to indexing result
    crate::et_kernel_check!(
        ctx,
        tensor_is_broadcastable_to(
            values.sizes(),
            ArrayRef::from_raw_parts(x_sizes.as_ptr(), x_dim)
        ),
        InvalidArgument,
        out
    );

    crate::et_kernel_check!(
        ctx,
        resize_tensor(out, in_.sizes()) == Error::Ok,
        InvalidArgument,
        out
    );

    // No further action if the input is empty
    if in_.numel() == 0 {
        return out;
    }

    // To start, copy the input data into the out tensor
    unsafe {
        core::ptr::copy_nonoverlapping(
            in_.const_data_ptr::<u8>(),
            out.mutable_data_ptr::<u8>(),
            in_.nbytes(),
        );
    }

    // In what follows, `x = in[indices]`. This tensor is implicit, and it would
    // be much easier to be able to allocate memory, and then call index.Tensor
    // to compute `x`. But since we can't do that, we have to keep track of its
    // shape, number of dimensions, number of elements, and use it to translate
    // coordinates from `x` to `in`.

    // Compute the dim_map and ix_map needed for `x -> in` coordinate translation
    let mut dim_map: [i32; K_TENSOR_DIMENSION_LIMIT] = [0; K_TENSOR_DIMENSION_LIMIT];
    let mut ix_map: [i32; K_TENSOR_DIMENSION_LIMIT] = [0; K_TENSOR_DIMENSION_LIMIT];
    let mut start: usize = 0;

    if adjacent {
        start = get_num_leading_null_indices(indices);
    }
    let bc_ndim: usize = get_indices_broadcast_ndim(indices);
    unsafe {
        compute_dim_map(in_, indices, dim_map.as_mut_ptr(), block_count == 1);
        compute_index_map(in_, indices, ix_map.as_mut_ptr());
    }

    // Compute the number of elements in the indexed space
    let mut x_numel: usize = 1;
    for i in 0..x_dim {
        x_numel *= x_sizes[i] as usize;
    }

    crate::et_switch_realhbbf16_types!(in_type, ctx, "index_put.out", CTYPE, {
        let values_data: *const CTYPE = values.const_data_ptr::<CTYPE>();
        let out_data: *mut CTYPE = out.mutable_data_ptr::<CTYPE>();

        for x_ix in 0..x_numel {
            let in_ix: usize;

            let mut x_coord: [usize; K_TENSOR_DIMENSION_LIMIT] = [0; K_TENSOR_DIMENSION_LIMIT];
            delinearize_index(
                x_ix,
                ArrayRef::from_raw_parts(x_sizes.as_ptr(), x_dim),
                x_coord.as_mut_ptr(),
                K_TENSOR_DIMENSION_LIMIT,
            );

            let mut in_coord: [usize; K_TENSOR_DIMENSION_LIMIT] = [0; K_TENSOR_DIMENSION_LIMIT];

            crate::et_kernel_check!(
                ctx,
                unsafe {
                    get_in_coord(
                        in_,
                        indices,
                        start,
                        bc_ndim,
                        dim_map.as_mut_ptr(),
                        ix_map.as_mut_ptr(),
                        x_coord.as_mut_ptr(),
                        in_coord.as_mut_ptr(),
                    )
                },
                InvalidArgument,
                out
            );

            in_ix = unsafe { coordinateToIndex(in_, in_coord.as_ptr()) };

            // Braodcast values
            let val_ix: usize = linearize_access_indexes_tensor(
                ArrayRef::from_raw_parts(x_coord.as_ptr(), x_dim),
                x_dim as ssize_t,
                values,
            );
            if accumulate {
                unsafe {
                    *out_data.add(in_ix) =
                        (*out_data.add(in_ix)).index_put_add(*values_data.add(val_ix));
                }
            } else {
                unsafe {
                    *out_data.add(in_ix) = *values_data.add(val_ix);
                }
            }
        }
    });

    out
}

// [spec:et:def:op-index-put.torch.executor.native.check-special-case-in-place-args-fn]
// [spec:et:sem:op-index-put.torch.executor.native.check-special-case-in-place-args-fn]
fn check_special_case_in_place_args(
    ctx: &mut KernelRuntimeContext,
    in_: &Tensor,
    indices: TensorOptList,
    values: &Tensor,
    accumulate: bool,
    dim: &mut usize,
) -> bool {
    et_check_or_return_false!(
        !accumulate,
        "Special case in-place index_put does not support accumulate"
    );

    et_check_or_return_false!(
        indices.size() as isize <= in_.dim(),
        "Indexing too many dimensions"
    );

    let mut found_index = false;
    for i in 0..indices.size() {
        if unsafe { indices.index(i) }.is_some() {
            *dim = i;
            et_check_or_return_false!(
                !found_index,
                "Special case in-place index_put only supports a single non-null index tensor"
            );
            found_index = true;
            let index: &Tensor = unsafe { indices.index(i) }.as_ref().unwrap();
            let ix_type: ScalarType = index.scalar_type();
            et_check_or_return_false!(
                ix_type == ScalarType::Long || ix_type == ScalarType::Int,
                "Special case in-place index_put only supports Long or Int index tensors; got {}",
                ix_type as i32
            );
            et_check_or_return_false!(
                index.dim() == 1,
                "Special case in-place index_put only supports 1-dimensional index tensors; got {}",
                ix_type as i32
            );
        }
    }

    et_check_or_return_false!(
        found_index,
        "Special case in-place index_put needs at least one non-null index tensor"
    );

    let index: &Tensor = unsafe { indices.index(*dim) }.as_ref().unwrap();

    let mut is_valid_index = true;
    crate::et_switch_two_types!(Long, Int, index.scalar_type(), ctx, "index_put_", CTYPE, {
        let index_arr: *const CTYPE = index.const_data_ptr::<CTYPE>();
        for i in 0..index.numel() {
            if unsafe { *index_arr.add(i as usize) } < 0 as CTYPE
                || unsafe { *index_arr.add(i as usize) } >= in_.size(*dim as isize) as CTYPE
            {
                crate::et_log!(
                    Error,
                    "Index {} out of range for tensor with size {} at dimension {}",
                    unsafe { *index_arr.add(i as usize) } as i64,
                    in_.size(*dim as isize),
                    *dim
                );
                is_valid_index = false;
                break;
            }
        }
    });

    et_check_or_return_false!(
        is_valid_index,
        "Some index values are not within bounds of input tensor at indexed dim"
    );

    et_check_or_return_false!(
        values.size(*dim as isize) == index.size(0),
        "Special case in-place index_put requires values to match index length at the indexed dim; values.size({}) = {}, index_length = {}",
        *dim,
        values.size(*dim as isize),
        index.size(0)
    );

    let mut expected_values_size: [SizesType; K_TENSOR_DIMENSION_LIMIT] =
        [0; K_TENSOR_DIMENSION_LIMIT];
    let in_ndim: usize = in_.dim() as usize;
    for i in 0..in_ndim {
        if i != *dim {
            expected_values_size[i] = in_.size(i as isize) as SizesType;
        }
    }
    expected_values_size[*dim] = index.size(0) as SizesType;

    // PORT-NOTE: the C++ `#if ET_LOG_ENABLED` branch formats input/values shape
    // strings for the message; the ported check macro drops the message, so the
    // shape-string computation is elided (both branches call the same check).
    et_check_or_return_false!(
        tensor_has_expected_size(
            values,
            ArrayRef::from_raw_parts(expected_values_size.as_ptr(), in_ndim)
        ),
        "Special case in-place index_put requires values to match input shape except for indexed dim"
    );

    true
}

// [spec:et:def:op-index-put.torch.executor.native.index-put-fn]
// [spec:et:sem:op-index-put.torch.executor.native.index-put-fn]
pub fn index_put_<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    in_: &'a Tensor<'b>,
    indices: TensorOptList,
    values: &Tensor,
    accumulate: bool,
) -> &'a Tensor<'b> {
    // (void)ctx;

    crate::et_kernel_check!(
        ctx,
        tensors_have_same_dtype2(in_, values),
        InvalidArgument,
        in_
    );

    crate::et_kernel_check!(
        ctx,
        tensors_have_same_dim_order2(in_, values),
        InvalidArgument,
        in_
    );

    crate::et_kernel_check!(ctx, tensor_is_default_dim_order(in_), InvalidArgument, in_);

    let mut dim: usize = 0;
    crate::et_kernel_check!(
        ctx,
        check_special_case_in_place_args(ctx, in_, indices, values, accumulate, &mut dim),
        InvalidArgument,
        in_
    );

    let index: &Tensor = unsafe { indices.index(dim) }.as_ref().unwrap();
    let index_type: ScalarType = index.scalar_type();

    if in_.dim() == 0 {
        unsafe {
            core::ptr::copy_nonoverlapping(
                values.const_data_ptr::<u8>(),
                in_.mutable_data_ptr::<u8>(),
                in_.nbytes(),
            );
        }
        return in_;
    }

    let leading_dims: usize = getLeadingDims(in_, dim as i64);
    let trailing_dims: usize = getTrailingDims(in_, dim as i64);

    if leading_dims == 0 || trailing_dims == 0 {
        return in_;
    }

    let values_dim_length: usize = values.size(dim as isize) as usize;
    let in_dim_length: usize = in_.size(dim as isize) as usize;

    let length_per_step: usize = trailing_dims * in_.element_size() as usize;

    let values_data: *const u8 = values.const_data_ptr::<u8>();
    let in_data: *mut u8 = in_.mutable_data_ptr::<u8>();

    crate::et_switch_two_types!(Long, Int, index_type, ctx, "index_put_", CTYPE, {
        let index_arr: *const CTYPE = index.const_data_ptr::<CTYPE>();
        for i in 0..leading_dims {
            let src: *const u8 =
                unsafe { values_data.add(i * values_dim_length * length_per_step) };
            let dest: *mut u8 = unsafe { in_data.add(i * in_dim_length * length_per_step) };
            for j in 0..values_dim_length {
                let copy_src: *const u8 = unsafe { src.add(j * length_per_step) };
                let index_val_j: CTYPE = unsafe { *index_arr.add(j) };
                let copy_dest: *mut u8 =
                    unsafe { dest.add(index_val_j as usize * length_per_step) };
                unsafe {
                    core::ptr::copy_nonoverlapping(copy_src, copy_dest, length_per_step);
                }
            }
        }
    });

    in_
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

    fn context() -> KernelRuntimeContext<'static> {
        crate::runtime::platform::runtime::runtime_init();
        KernelRuntimeContext::new(
            crate::extension::module::module::null_event_tracer(),
            core::ptr::null_mut::<MemoryAllocator>() as *mut dyn MemoryAllocatorBase,
        )
    }

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

    // Builds a `TensorOptList` viewing `v` (which the caller must keep alive).
    fn oil<'a, 'b>(v: &'a [Option<Tensor<'b>>]) -> TensorOptList<'b>
    where
        'a: 'b,
    {
        ArrayRef::from_raw_parts(v.as_ptr(), v.len())
    }

    trait FromI32: Copy {
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
    impl FromI32 for bool {
        fn from_i32(v: i32) -> Self {
            v != 0
        }
    }

    fn ii<T: FromI32>(v: &[i32]) -> Vec<T> {
        v.iter().map(|&x| T::from_i32(x)).collect()
    }

    // ---- OpIndexPutOutTest ----

    fn test_dtype_out<INPUT, INDICES>()
    where
        INPUT: CppTypeToScalarType + FactoryValue + FromI32,
        INDICES: CppTypeToScalarType + FactoryValue + FromI32,
    {
        let tf = TensorFactory::<INPUT>::new();
        let tfl = TensorFactory::<INDICES>::new();
        let tfb = TensorFactory::<bool>::new();

        let x = tf.make_default(
            vec![3, 2, 4],
            ii::<INPUT>(&[
                1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0,
            ]),
        );

        // indices [0, 1, :], [1, 1, :], [2, 1, :]
        let indices = [
            Some(tfl.make_default(vec![1, 3], ii::<INDICES>(&[0, 1, 2]))),
            Some(tfl.make_default(vec![1, 3], ii::<INDICES>(&[1, 1, 1]))),
        ];
        // bool representation of the same index list
        let indices_bool = [
            Some(tfb.make_default(vec![3], vec![true, true, true])),
            Some(tfb.make_default(vec![2], vec![false, true])),
        ];

        let values = tf.ones_default(vec![3, 4]);
        let out_size = vec![3, 2, 4];

        let out = tf.zeros_default(out_size.clone());
        let mut ctx = context();
        let ret = index_put_out(&mut ctx, &x, oil(&indices), &values, false, &out);
        assert_tensor_eq!(*ret, out);
        assert_tensor_eq!(*ret, tf.ones_default(out_size.clone()));

        let out_with_bool = tf.zeros_default(out_size.clone());
        let ret_with_bool = index_put_out(
            &mut ctx,
            &x,
            oil(&indices_bool),
            &values,
            false,
            &out_with_bool,
        );
        assert_tensor_eq!(*ret_with_bool, out_with_bool);
        assert_tensor_eq!(*ret_with_bool, tf.ones_default(out_size.clone()));

        // indices [0, 1, :], [1, 0, :], [2, 0, :]
        let indices_alt = [
            Some(tfl.make_default(vec![1, 3], ii::<INDICES>(&[0, 1, 2]))),
            Some(tfl.make_default(vec![1, 3], ii::<INDICES>(&[0, 0, 0]))),
        ];
        let indices_alt_bool = [
            Some(tfb.make_default(vec![3], vec![true, true, true])),
            Some(tfb.make_default(vec![2], vec![true, false])),
        ];
        let values_alt = tf.zeros_default(vec![3, 4]);

        let out_alt = tf.ones_default(out_size.clone());
        let ret_alt = index_put_out(
            &mut ctx,
            &x,
            oil(&indices_alt),
            &values_alt,
            false,
            &out_alt,
        );
        assert_tensor_eq!(*ret_alt, out_alt);
        assert_tensor_eq!(*ret_alt, tf.zeros_default(out_size.clone()));

        let out_alt_with_bool = tf.ones_default(out_size.clone());
        let ret_alt_with_bool = index_put_out(
            &mut ctx,
            &x,
            oil(&indices_alt_bool),
            &values_alt,
            false,
            &out_alt_with_bool,
        );
        assert_tensor_eq!(*ret_alt_with_bool, out_alt_with_bool);
        assert_tensor_eq!(*ret_alt_with_bool, tf.zeros_default(out_size));
    }

    // Runs value-put test cases (out dtype Double).
    fn run_test_cases(
        x: &Tensor,
        indices: TensorOptList,
        values: &Tensor,
        expected: &Tensor,
        expected_accum: &Tensor,
    ) {
        let tf = TensorFactory::<f64>::new();

        let expected_sizes = expected.sizes();
        let out_size: Vec<i32> = (0..expected_sizes.size())
            .map(|i| *expected_sizes.at(i))
            .collect();
        let out = tf.ones_default(out_size.clone());

        let mut ctx = context();
        let ret = index_put_out(&mut ctx, x, indices, values, false, &out);
        assert_tensor_eq!(out, *ret);
        assert_tensor_eq!(*ret, *expected);

        let out_accum = tf.ones_default(out_size);
        let ret_accum = index_put_out(&mut ctx, x, indices, values, true, &out_accum);
        assert_tensor_eq!(out_accum, *ret_accum);
        assert_tensor_eq!(*ret_accum, *expected_accum);
    }

    fn test_dynamic_shape(out_shape: Vec<i32>, dynamism: TensorShapeDynamism) {
        let tf = TensorFactory::<f32>::new();
        let tf_indices = TensorFactory::<i64>::new();

        let input = tf.make_default(
            vec![2, 3, 4],
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
            ],
        );
        let indices = [
            Some(tf_indices.make_default(vec![1], vec![1])),
            Some(tf_indices.make_default(vec![1], vec![0])),
            Some(tf_indices.make_default(vec![2], vec![1, 2])),
        ];
        let values = tf.make_default(vec![2], vec![0.41940832138061523, 0.5529070496559143]);
        let expected = tf.make_default(
            vec![2, 3, 4],
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
                0.41940832138061523,
                0.5529070496559143,
                0.518521785736084,
                0.6976675987243652,
                0.800011396408081,
                0.16102945804595947,
                0.28226858377456665,
                0.6816085577011108,
                0.9151939749717712,
                0.39709991216659546,
                0.8741558790206909,
            ],
        );
        let out = tf.zeros(out_shape, dynamism);

        let mut ctx = context();
        index_put_out(&mut ctx, &input, oil(&indices), &values, false, &out);
        assert_tensor_eq!(out, expected);
    }

    // [spec:et:sem:op-index-put.torch.executor.native.index-put-out-fn/test]
    #[test]
    fn op_index_put_out_test_index_put_mask() {
        let tf = TensorFactory::<f64>::new();
        let tfb = TensorFactory::<bool>::new();
        let x = tf.make_default(
            vec![2, 3, 4],
            vec![
                1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., -1., -2., -3., -4., -5., -6.,
                -7., -8., -9., -10., -11., -12.,
            ],
        );
        let indices = tfb.make_default(
            vec![2, 3, 4],
            vec![
                true, false, false, false, false, false, true, false, false, false, false, false,
                false, true, false, false, false, false, false, false, false, false, true, false,
            ],
        );
        let values = tf.make_default(vec![4], vec![10., 20., 30., 40.]);
        let expected = tf.make_default(
            vec![2, 3, 4],
            vec![
                10., 2., 3., 4., 5., 6., 20., 8., 9., 10., 11., 12., -1., 30., -3., -4., -5., -6.,
                -7., -8., -9., -10., 40., -12.,
            ],
        );
        let expected_accum = tf.make_default(
            vec![2, 3, 4],
            vec![
                11., 2., 3., 4., 5., 6., 27., 8., 9., 10., 11., 12., -1., 28., -3., -4., -5., -6.,
                -7., -8., -9., -10., 29., -12.,
            ],
        );
        let indices_v = [Some(indices)];
        run_test_cases(&x, oil(&indices_v), &values, &expected, &expected_accum);
    }

    // [spec:et:sem:op-index-put.torch.executor.native.index-put-out-fn/test]
    #[test]
    fn op_index_put_out_test_index_put_mask_broadcast() {
        let tf = TensorFactory::<f64>::new();
        let tfb = TensorFactory::<bool>::new();
        let x = tf.make_default(
            vec![2, 3, 4],
            vec![
                1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., -1., -2., -3., -4., -5., -6.,
                -7., -8., -9., -10., -11., -12.,
            ],
        );
        let indices = tfb.make_default(
            vec![2, 3, 4],
            vec![
                true, false, false, false, false, false, true, false, false, false, false, false,
                false, true, false, false, false, false, false, false, false, false, true, false,
            ],
        );
        let values = tf.make_default(vec![1], vec![10.]);
        let expected = tf.make_default(
            vec![2, 3, 4],
            vec![
                10., 2., 3., 4., 5., 6., 10., 8., 9., 10., 11., 12., -1., 10., -3., -4., -5., -6.,
                -7., -8., -9., -10., 10., -12.,
            ],
        );
        let expected_accum = tf.make_default(
            vec![2, 3, 4],
            vec![
                11., 2., 3., 4., 5., 6., 17., 8., 9., 10., 11., 12., -1., 8., -3., -4., -5., -6.,
                -7., -8., -9., -10., -1., -12.,
            ],
        );
        let indices_v = [Some(indices)];
        run_test_cases(&x, oil(&indices_v), &values, &expected, &expected_accum);
    }

    // [spec:et:sem:op-index-put.torch.executor.native.index-put-out-fn/test]
    #[test]
    fn op_index_put_out_test_put_front_dim_all_indexes() {
        let tf = TensorFactory::<f64>::new();
        let tfi = TensorFactory::<i32>::new();
        let tfl = TensorFactory::<i64>::new();
        let tfb = TensorFactory::<bool>::new();
        let x = tf.make_default(
            vec![2, 3, 4],
            vec![
                1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., -1., -2., -3., -4., -5., -6.,
                -7., -8., -9., -10., -11., -12.,
            ],
        );

        let indices_long = [
            Some(tfl.make_default(vec![1], vec![1])),
            Some(tfl.make_default(vec![1], vec![0])),
            Some(tfl.make_default(vec![2], vec![1, 2])),
        ];
        let indices_int = [
            Some(tfi.make_default(vec![1], vec![1])),
            Some(tfi.make_default(vec![1], vec![0])),
            Some(tfi.make_default(vec![2], vec![1, 2])),
        ];
        let indices_negative = [
            Some(tfl.make_default(vec![1], vec![-1])),
            Some(tfl.make_default(vec![1], vec![0])),
            Some(tfl.make_default(vec![2], vec![-3, -2])),
        ];
        let indices_bool = [
            Some(tfb.make_default(vec![2], vec![false, true])),
            Some(tfb.make_default(vec![3], vec![true, false, false])),
            Some(tfl.make_default(vec![2], vec![-3, -2])),
        ];
        let indices_mixed = [
            Some(tfb.make_default(vec![2], vec![false, true])),
            Some(tfl.make_default(vec![1], vec![0])),
            Some(tfl.make_default(vec![2], vec![-3, -2])),
        ];

        let values = tf.make_default(vec![2], vec![10., 20.]);
        let expected = tf.make_default(
            vec![2, 3, 4],
            vec![
                1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., -1., 10., 20., -4., -5., -6.,
                -7., -8., -9., -10., -11., -12.,
            ],
        );
        let expected_accum = tf.make_default(
            vec![2, 3, 4],
            vec![
                1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., -1., 8., 17., -4., -5., -6.,
                -7., -8., -9., -10., -11., -12.,
            ],
        );

        run_test_cases(&x, oil(&indices_long), &values, &expected, &expected_accum);
        run_test_cases(&x, oil(&indices_int), &values, &expected, &expected_accum);
        run_test_cases(
            &x,
            oil(&indices_negative),
            &values,
            &expected,
            &expected_accum,
        );
        run_test_cases(&x, oil(&indices_bool), &values, &expected, &expected_accum);
        run_test_cases(&x, oil(&indices_mixed), &values, &expected, &expected_accum);
    }

    // [spec:et:sem:op-index-put.torch.executor.native.index-put-out-fn/test]
    #[test]
    fn op_index_put_out_test_put_two_values_at_same_index() {
        let tf = TensorFactory::<f64>::new();
        let tfl = TensorFactory::<i64>::new();
        let x = tf.make_default(
            vec![2, 3, 4],
            vec![
                1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., -1., -2., -3., -4., -5., -6.,
                -7., -8., -9., -10., -11., -12.,
            ],
        );
        let indices = [
            Some(tfl.make_default(vec![1, 2], vec![0, 0])),
            Some(tfl.make_default(vec![1, 2], vec![1, 1])),
            Some(tfl.make_default(vec![1, 2], vec![2, 2])),
        ];
        let values = tf.make_default(vec![1], vec![10.]);
        let expected = tf.make_default(
            vec![2, 3, 4],
            vec![
                1., 2., 3., 4., 5., 6., 10., 8., 9., 10., 11., 12., -1., -2., -3., -4., -5., -6.,
                -7., -8., -9., -10., -11., -12.,
            ],
        );
        let expected_accum = tf.make_default(
            vec![2, 3, 4],
            vec![
                1., 2., 3., 4., 5., 6., 27., 8., 9., 10., 11., 12., -1., -2., -3., -4., -5., -6.,
                -7., -8., -9., -10., -11., -12.,
            ],
        );
        run_test_cases(&x, oil(&indices), &values, &expected, &expected_accum);
    }

    // [spec:et:sem:op-index-put.torch.executor.native.index-put-out-fn/test]
    #[test]
    fn op_index_put_out_test_indices_fewer_than_input_dim_supported() {
        let tf = TensorFactory::<f64>::new();
        let tfi = TensorFactory::<i32>::new();
        let tfl = TensorFactory::<i64>::new();
        let tfb = TensorFactory::<bool>::new();
        let x = tf.make_default(
            vec![2, 3, 4],
            vec![
                1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., -1., -2., -3., -4., -5., -6.,
                -7., -8., -9., -10., -11., -12.,
            ],
        );

        let indices_long = [
            Some(tfl.make_default(vec![1], vec![1])),
            Some(tfl.make_default(vec![2], vec![0, 1])),
        ];
        let indices_mixed = [
            Some(tfi.make_default(vec![1], vec![-1])),
            Some(tfb.make_default(vec![3], vec![true, true, false])),
        ];

        let values = tf.make_default(vec![2, 4], vec![10., 20., 30., 40., -10., -20., -30., -40.]);
        let expected = tf.make_default(
            vec![2, 3, 4],
            vec![
                1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 10., 20., 30., 40., -10., -20.,
                -30., -40., -9., -10., -11., -12.,
            ],
        );
        let expected_accum = tf.make_default(
            vec![2, 3, 4],
            vec![
                1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 9., 18., 27., 36., -15., -26.,
                -37., -48., -9., -10., -11., -12.,
            ],
        );

        run_test_cases(&x, oil(&indices_long), &values, &expected, &expected_accum);
        run_test_cases(&x, oil(&indices_mixed), &values, &expected, &expected_accum);
    }

    // [spec:et:sem:op-index-put.torch.executor.native.index-put-out-fn/test]
    #[test]
    fn op_index_put_out_test_indices_fewer_than_input_dim_supported_same_value() {
        let tf = TensorFactory::<f64>::new();
        let tfl = TensorFactory::<i64>::new();
        let x = tf.make_default(
            vec![2, 3, 4],
            vec![
                1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., -1., -2., -3., -4., -5., -6.,
                -7., -8., -9., -10., -11., -12.,
            ],
        );
        let indices = [
            Some(tfl.make_default(vec![1], vec![1])),
            Some(tfl.make_default(vec![2], vec![0, 1])),
        ];
        let values = tf.make_default(vec![1], vec![10.]);
        let expected = tf.make_default(
            vec![2, 3, 4],
            vec![
                1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 10., 10., 10., 10., 10., 10.,
                10., 10., -9., -10., -11., -12.,
            ],
        );
        let expected_accum = tf.make_default(
            vec![2, 3, 4],
            vec![
                1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 9., 8., 7., 6., 5., 4., 3., 2.,
                -9., -10., -11., -12.,
            ],
        );
        run_test_cases(&x, oil(&indices), &values, &expected, &expected_accum);
    }

    // [spec:et:sem:op-index-put.torch.executor.native.index-put-out-fn/test]
    #[test]
    fn op_index_put_out_test_all_dtypes_supported_for_input() {
        // ET_FORALL_REALHBBF16_TYPES x Long
        test_dtype_out::<u8, i64>();
        test_dtype_out::<i8, i64>();
        test_dtype_out::<i16, i64>();
        test_dtype_out::<i32, i64>();
        test_dtype_out::<i64, i64>();
        test_dtype_out::<Half, i64>();
        test_dtype_out::<f32, i64>();
        test_dtype_out::<f64, i64>();
        test_dtype_out::<bool, i64>();
        test_dtype_out::<BFloat16, i64>();
    }

    // [spec:et:sem:op-index-put.torch.executor.native.index-put-out-fn/test]
    #[test]
    fn op_index_put_out_test_all_dtypes_supported_for_indices_list() {
        test_dtype_out::<f32, i64>();
        test_dtype_out::<f32, i32>();
    }

    // [spec:et:sem:op-index-put.torch.executor.native.index-put-out-fn/test]
    #[test]
    fn op_index_put_out_test_index_out_of_bound_dies() {
        let tf = TensorFactory::<f32>::new();
        let tfl = TensorFactory::<i64>::new();
        let x = tf.ones_default(vec![1, 1, 1]);
        let out = tf.zeros_default(vec![1, 1, 1]);
        let index = tfl.make_default(vec![1], vec![5]);
        let values = tf.make_default(vec![1], vec![10.]);
        let indices = [Some(index)];
        let mut ctx = context();
        et_expect_kernel_failure!(
            ctx,
            index_put_out(&mut ctx, &x, oil(&indices), &values, false, &out)
        );
    }

    // [spec:et:sem:op-index-put.torch.executor.native.index-put-out-fn/test]
    #[test]
    fn op_index_put_out_test_negative_index_out_of_bound_dies() {
        let tf = TensorFactory::<f32>::new();
        let tfl = TensorFactory::<i64>::new();
        let x = tf.ones_default(vec![1, 1, 1]);
        let out = tf.zeros_default(vec![1, 1, 1]);
        let index = tfl.make_default(vec![1], vec![-5]);
        let values = tf.make_default(vec![1], vec![10.]);
        let indices = [Some(index)];
        let mut ctx = context();
        et_expect_kernel_failure!(
            ctx,
            index_put_out(&mut ctx, &x, oil(&indices), &values, false, &out)
        );
    }

    // [spec:et:sem:op-index-put.torch.executor.native.index-put-out-fn/test]
    #[test]
    fn op_index_put_out_test_too_many_boolean_index_count_dies() {
        let tf = TensorFactory::<f32>::new();
        let tfb = TensorFactory::<bool>::new();
        let x = tf.ones_default(vec![1, 1, 1]);
        let out = tf.zeros_default(vec![1, 1, 1]);
        let index = tfb.make_default(vec![3], vec![true, true, false]);
        let values = tf.make_default(vec![1], vec![10.]);
        let indices = [Some(index)];
        let mut ctx = context();
        et_expect_kernel_failure!(
            ctx,
            index_put_out(&mut ctx, &x, oil(&indices), &values, false, &out)
        );
    }

    // [spec:et:sem:op-index-put.torch.executor.native.index-put-out-fn/test]
    #[test]
    fn op_index_put_out_test_too_few_boolean_index_count_dies() {
        let tf = TensorFactory::<f32>::new();
        let tfb = TensorFactory::<bool>::new();
        let x = tf.ones_default(vec![4]);
        let out = tf.zeros_default(vec![4]);
        let index = tfb.make_default(vec![1], vec![true]);
        let values = tf.make_default(vec![1], vec![10.]);
        let indices = [Some(index)];
        let mut ctx = context();
        et_expect_kernel_failure!(
            ctx,
            index_put_out(&mut ctx, &x, oil(&indices), &values, false, &out)
        );
    }

    // [spec:et:sem:op-index-put.torch.executor.native.index-put-out-fn/test]
    #[test]
    fn op_index_put_out_test_mismatched_index_mask_dies() {
        let tf = TensorFactory::<f32>::new();
        let tfb = TensorFactory::<bool>::new();
        let x = tf.ones_default(vec![4, 4]);
        let out = tf.zeros_default(vec![4, 4]);
        let index = tfb.ones_default(vec![3, 3]);
        let values = tf.make_default(vec![1], vec![10.]);
        let indices = [Some(index)];
        let mut ctx = context();
        et_expect_kernel_failure!(
            ctx,
            index_put_out(&mut ctx, &x, oil(&indices), &values, false, &out)
        );
    }

    // [spec:et:sem:op-index-put.torch.executor.native.index-put-out-fn/test]
    #[test]
    fn op_index_put_out_test_mismatched_output_dtypes_dies() {
        let tf_float = TensorFactory::<f32>::new();
        let tf_double = TensorFactory::<f64>::new();
        let tf_long = TensorFactory::<i64>::new();
        let x = tf_float.zeros_default(vec![1, 2, 2]);
        let out = tf_double.ones_default(vec![1, 2, 2]);
        let index = tf_long.make_default(vec![1], vec![0]);
        let values = tf_float.make_default(vec![1], vec![10.]);
        let indices = [Some(index)];
        let mut ctx = context();
        et_expect_kernel_failure!(
            ctx,
            index_put_out(&mut ctx, &x, oil(&indices), &values, false, &out)
        );
    }

    // [spec:et:sem:op-index-put.torch.executor.native.index-put-out-fn/test]
    #[test]
    fn op_index_put_out_test_mismatched_values_dtypes_dies() {
        let tf_float = TensorFactory::<f32>::new();
        let tf_double = TensorFactory::<f64>::new();
        let tf_long = TensorFactory::<i64>::new();
        let x = tf_float.zeros_default(vec![1, 2, 2]);
        let out = tf_float.ones_default(vec![1, 2, 2]);
        let index = tf_long.make_default(vec![1], vec![0]);
        let values = tf_double.make_default(vec![1], vec![10.]);
        let indices = [Some(index)];
        let mut ctx = context();
        et_expect_kernel_failure!(
            ctx,
            index_put_out(&mut ctx, &x, oil(&indices), &values, false, &out)
        );
    }

    // [spec:et:sem:op-index-put.torch.executor.native.index-put-out-fn/test]
    #[test]
    fn op_index_put_out_test_values_size_mismatch_dim_dies() {
        let tf = TensorFactory::<f32>::new();
        let tfl = TensorFactory::<i64>::new();
        let x = tf.zeros_default(vec![2, 4, 7, 5]);
        let index = tfl.make_default(vec![1], vec![1]);
        let out = tf.ones_default(vec![2, 4, 7, 5]);
        let values = tf.make_default(vec![1, 2], vec![10., 10.]);
        let indices = [Some(index)];
        let mut ctx = context();
        et_expect_kernel_failure!(
            ctx,
            index_put_out(&mut ctx, &x, oil(&indices), &values, false, &out)
        );
    }

    // [spec:et:sem:op-index-put.torch.executor.native.index-put-out-fn/test]
    #[test]
    fn op_index_put_out_test_invalid_indices_dtype_dies() {
        let tf = TensorFactory::<f32>::new();
        let tff = TensorFactory::<f32>::new();
        let x = tf.zeros_default(vec![2, 4, 7, 5]);
        let indices = [
            Some(tff.make_default(vec![3], vec![1., 1., 1.])),
            Some(tff.make_default(vec![2], vec![1., 2.])),
        ];
        let out = tf.ones_default(vec![2, 4, 7, 5]);
        let values = tf.make_default(vec![1], vec![10.]);
        let mut ctx = context();
        et_expect_kernel_failure!(
            ctx,
            index_put_out(&mut ctx, &x, oil(&indices), &values, false, &out)
        );
    }

    // [spec:et:sem:op-index-put.torch.executor.native.index-put-out-fn/test]
    #[test]
    fn op_index_put_out_test_invalid_indices_shapes_dies() {
        let tf = TensorFactory::<f32>::new();
        let tfl = TensorFactory::<i64>::new();
        let x = tf.zeros_default(vec![2, 4, 7, 5]);
        let indices = [
            Some(tfl.make_default(vec![3], vec![1, 1, 1])),
            Some(tfl.make_default(vec![2], vec![1, 2])),
        ];
        let out = tf.ones_default(vec![2, 4, 7, 5]);
        let values = tf.make_default(vec![1, 2], vec![10., 10.]);
        let mut ctx = context();
        et_expect_kernel_failure!(
            ctx,
            index_put_out(&mut ctx, &x, oil(&indices), &values, false, &out)
        );
    }

    // [spec:et:sem:op-index-put.torch.executor.native.index-put-out-fn/test]
    #[test]
    fn op_index_put_out_test_non_linear_indices() {
        let tf = TensorFactory::<f32>::new();
        let tfl = TensorFactory::<i64>::new();
        let x = tf.zeros_default(vec![4, 4]);
        let indices = [
            Some(tfl.make_default(vec![2, 2], vec![1, 1, 1, 1])),
            Some(tfl.make_default(vec![1, 2], vec![3, 0])),
        ];
        let out = tf.ones_default(vec![4, 4]);
        let values = tf.make_default(vec![1], vec![10.]);
        let expected = tf.make_default(
            vec![4, 4],
            vec![
                0., 0., 0., 0., 10., 0., 0., 10., 0., 0., 0., 0., 0., 0., 0., 0.,
            ],
        );
        let mut ctx = context();
        let ret = index_put_out(&mut ctx, &x, oil(&indices), &values, false, &out);
        assert_tensor_eq!(*ret, out);
        assert_tensor_eq!(*ret, expected);
    }

    // [spec:et:sem:op-index-put.torch.executor.native.index-put-out-fn/test]
    #[test]
    fn op_index_put_out_test_dynamic_shape_upper_bound_same_as_expected() {
        test_dynamic_shape(vec![2, 3, 4], TensorShapeDynamism::DYNAMIC_BOUND);
    }

    // PORT-NOTE: C++ guards with `ET_SKIP_IF(!output_resize, ...)`; the portable
    // kernel's `output_resize` supported-feature defaults to false, so the C++
    // test is skipped. Ported + ignored.
    // [spec:et:sem:op-index-put.torch.executor.native.index-put-out-fn/test]
    #[test]
    #[ignore]
    fn op_index_put_out_test_dynamic_shape_upper_bound_larger_than_expected() {
        test_dynamic_shape(vec![10, 10, 10], TensorShapeDynamism::DYNAMIC_BOUND);
    }

    // PORT-NOTE: C++ guards with `ET_SKIP_IF(!output_resize, ...)`; skipped for
    // portable. Ported + ignored.
    // [spec:et:sem:op-index-put.torch.executor.native.index-put-out-fn/test]
    #[test]
    #[ignore]
    fn op_index_put_out_test_dynamic_shape_unbound() {
        test_dynamic_shape(vec![1, 1, 1], TensorShapeDynamism::DYNAMIC_UNBOUND);
    }

    // ---- OpIndexPutInplaceTest ----

    fn test_dtype_inplace<INPUT, INDICES>()
    where
        INPUT: CppTypeToScalarType + FactoryValue + FromI32,
        INDICES: CppTypeToScalarType + FactoryValue + FromI32,
    {
        let tf = TensorFactory::<INPUT>::new();
        let tfl = TensorFactory::<INDICES>::new();

        let x = tf.make_default(
            vec![2, 3, 4],
            ii::<INPUT>(&[
                1, 1, 1, 1, 0, 0, 0, 0, 2, 2, 2, 2, 3, 3, 3, 3, 0, 0, 0, 0, 5, 5, 5, 5,
            ]),
        );

        let indices = [
            None,
            Some(tfl.make_default(vec![2], ii::<INDICES>(&[0, 2]))),
        ];

        let values = tf.make_default(
            vec![2, 2, 4],
            ii::<INPUT>(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]),
        );

        let expected = tf.make_default(
            vec![2, 3, 4],
            ii::<INPUT>(&[
                1, 2, 3, 4, 0, 0, 0, 0, 5, 6, 7, 8, 9, 10, 11, 12, 0, 0, 0, 0, 13, 14, 15, 16,
            ]),
        );

        let mut ctx = context();
        let ret = index_put_(&mut ctx, &x, oil(&indices), &values, false);
        assert_tensor_eq!(*ret, x);
        assert_tensor_eq!(*ret, expected);
    }

    // [spec:et:sem:op-index-put.torch.executor.native.index-put-fn/test]
    // also verifies check_special_case_in_place_args (valid path): a single non-null 1-D
    // Long index at dim=1 with matching non-accumulate values must pass, or index_put_
    // aborts and the in-place result never matches expected.
    // [spec:et:sem:op-index-put.torch.executor.native.check-special-case-in-place-args-fn/test]
    #[test]
    fn op_index_put_inplace_test_all_dtypes_supported_for_input() {
        // ET_FORALL_REALHBBF16_TYPES x Long
        test_dtype_inplace::<u8, i64>();
        test_dtype_inplace::<i8, i64>();
        test_dtype_inplace::<i16, i64>();
        test_dtype_inplace::<i32, i64>();
        test_dtype_inplace::<i64, i64>();
        test_dtype_inplace::<Half, i64>();
        test_dtype_inplace::<f32, i64>();
        test_dtype_inplace::<f64, i64>();
        test_dtype_inplace::<bool, i64>();
        test_dtype_inplace::<BFloat16, i64>();
    }

    // [spec:et:sem:op-index-put.torch.executor.native.index-put-fn/test]
    #[test]
    fn op_index_put_inplace_test_all_dtypes_supported_for_indices_list() {
        test_dtype_inplace::<f32, i64>();
        test_dtype_inplace::<f32, i32>();
    }
}
