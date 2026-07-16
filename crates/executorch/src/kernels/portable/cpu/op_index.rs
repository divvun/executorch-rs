//! Literal port of kernels/portable/cpu/op_index.cpp.

use crate::kernels::portable::cpu::util::advanced_index_util::{
    TensorOptList, check_index_args, compute_dim_map, compute_index_map, count_index_blocks,
    get_in_ix, get_index_out_target_size, get_indices_broadcast_ndim, get_num_leading_null_indices,
};
use crate::runtime::core::array_ref::ArrayRef;
use crate::runtime::core::error::Error;
use crate::runtime::core::exec_aten::util::tensor_util::{
    K_TENSOR_DIMENSION_LIMIT, getLeadingDims, getTrailingDims, resize_tensor,
    tensor_is_default_dim_order, tensors_have_same_dim_order2, tensors_have_same_dtype2,
};
use crate::runtime::core::portable_type::scalar_type::ScalarType;
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::core::portable_type::tensor_impl::{SizesType, ssize_t};
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

// PORT-NOTE: local `et_log_and_return_if_false!` / `et_check_or_return_false!`
// mirroring the C++ macros (format args dropped by the crate-level check macros),
// following the established per-module definitions.
macro_rules! et_log_and_return_if_false {
    ($cond:expr) => {{
        if !($cond) {
            $crate::et_log!(Error, "Check failed ({}): ", ::core::stringify!($cond));
            return false;
        }
    }};
}
macro_rules! et_check_or_return_false {
    ($cond:expr, $($arg:tt)*) => {{
        if !($cond) {
            $crate::et_log!(Error, $($arg)*);
            return false;
        }
    }};
}

// [spec:et:def:op-index.torch.executor.native.check-fast-path-conditions-fn]
// [spec:et:sem:op-index.torch.executor.native.check-fast-path-conditions-fn]
fn check_fast_path_conditions(_in: &Tensor, indices: TensorOptList, dim: &mut usize) -> bool {
    let mut found_index = false;
    for i in 0..indices.size() {
        if unsafe { indices.index(i) }.is_some() {
            *dim = i;
            // Fast path only supports a single non-null index tensor
            if found_index {
                return false;
            }
            found_index = true;
            let index: &Tensor = unsafe { indices.index(i) }.as_ref().unwrap();
            let ix_type: ScalarType = index.scalar_type();
            // Fast path only supports Long or Int index tensors
            if ix_type != ScalarType::Long && ix_type != ScalarType::Int {
                return false;
            }
            // Fast path only supports a 1-dimensional index tensor
            if index.dim() != 1 {
                return false;
            }
        }
    }

    // Fast path needs at least one non-null index tensor
    if !found_index {
        return false;
    }

    true
}

// [spec:et:def:op-index.torch.executor.native.check-fast-path-args-fn]
// [spec:et:sem:op-index.torch.executor.native.check-fast-path-args-fn]
fn check_fast_path_args(
    ctx: &mut KernelRuntimeContext,
    in_: &Tensor,
    indices: TensorOptList,
    dim: usize,
    out: &Tensor,
) -> bool {
    et_log_and_return_if_false!(tensors_have_same_dtype2(in_, out));

    et_check_or_return_false!(
        indices.size() as isize <= in_.dim(),
        "Indexing too many dimensions"
    );

    let index: &Tensor = unsafe { indices.index(dim) }.as_ref().unwrap();

    let mut is_valid_index = true;
    crate::et_switch_two_types!(
        Long,
        Int,
        index.scalar_type(),
        ctx,
        "index.Tensor",
        CTYPE,
        {
            let index_arr: *const CTYPE = index.const_data_ptr::<CTYPE>();
            for i in 0..index.numel() {
                let mut index_val: CTYPE = unsafe { *index_arr.add(i as usize) };
                let dim_size: CTYPE = in_.size(dim as isize) as CTYPE;
                index_val = if index_val < 0 as CTYPE {
                    index_val + dim_size
                } else {
                    index_val
                };
                if index_val < 0 as CTYPE || index_val >= dim_size {
                    crate::et_log!(
                        Error,
                        "Index {} out of range for tensor with size {} at dimension {}",
                        unsafe { *index_arr.add(i as usize) } as i64,
                        in_.size(dim as isize),
                        dim
                    );
                    is_valid_index = false;
                    break;
                }
            }
        }
    );

    et_check_or_return_false!(
        is_valid_index,
        "Some index values are not within bounds of input tensor at indexed dim"
    );

    true
}

// [spec:et:def:op-index.torch.executor.native.get-fast-path-index-out-target-size-fn]
// [spec:et:sem:op-index.torch.executor.native.get-fast-path-index-out-target-size-fn]
fn get_fast_path_index_out_target_size(
    in_: &Tensor,
    indices: TensorOptList,
    dim: usize,
    out_sizes: *mut SizesType,
    out_ndim: *mut usize,
) {
    unsafe {
        *out_ndim = in_.dim() as usize;
    }

    for d in 0..(in_.dim() as usize) {
        if d != dim {
            unsafe {
                *out_sizes.add(d) = in_.size(d as isize) as SizesType;
            }
        } else {
            unsafe {
                *out_sizes.add(d) = indices.index(dim).as_ref().unwrap().numel() as SizesType;
            }
        }
    }
}

// [spec:et:def:op-index.torch.executor.native.fast-path-fn]
// [spec:et:sem:op-index.torch.executor.native.fast-path-fn]
fn fast_path<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    in_: &Tensor,
    indices: TensorOptList,
    dim: usize,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    crate::et_kernel_check!(
        ctx,
        check_fast_path_args(ctx, in_, indices, dim, out),
        InvalidArgument,
        out
    );

    let index: &Tensor = unsafe { indices.index(dim) }.as_ref().unwrap();
    let index_type: ScalarType = index.scalar_type();

    let mut expected_size: [SizesType; K_TENSOR_DIMENSION_LIMIT] = [0; K_TENSOR_DIMENSION_LIMIT];
    let mut expected_ndim: usize = 0;
    get_fast_path_index_out_target_size(
        in_,
        indices,
        dim,
        expected_size.as_mut_ptr(),
        &mut expected_ndim,
    );

    crate::et_kernel_check!(
        ctx,
        resize_tensor(
            out,
            ArrayRef::from_raw_parts(expected_size.as_ptr(), expected_ndim)
        ) == Error::Ok,
        InvalidArgument,
        out
    );

    if out.dim() == 0 {
        unsafe {
            core::ptr::copy_nonoverlapping(
                in_.const_data_ptr::<u8>(),
                out.mutable_data_ptr::<u8>(),
                out.nbytes(),
            );
        }
        return out;
    }

    let leading_dims: usize = getLeadingDims(in_, dim as i64);
    let trailing_dims: usize = getTrailingDims(in_, dim as i64);

    if leading_dims == 0 || trailing_dims == 0 {
        return out;
    }

    let in_dim_length: usize = in_.size(dim as isize) as usize;
    let out_dim_length: usize = out.size(dim as isize) as usize;

    let length_per_step: usize = trailing_dims * in_.element_size() as usize;

    let in_data: *const u8 = in_.const_data_ptr::<u8>();
    let out_data: *mut u8 = out.mutable_data_ptr::<u8>();

    let op_name = "index.Tensor_out";

    crate::et_switch_two_types!(Long, Int, index_type, ctx, op_name, CTYPE, {
        let index_arr: *const CTYPE = index.const_data_ptr::<CTYPE>();
        let dim_size: CTYPE = in_.size(dim as isize) as CTYPE;
        for i in 0..leading_dims {
            let src: *const u8 = unsafe { in_data.add(i * in_dim_length * length_per_step) };
            let dest: *mut u8 = unsafe { out_data.add(i * out_dim_length * length_per_step) };
            for j in 0..out_dim_length {
                let index_val: CTYPE = if unsafe { *index_arr.add(j) } < 0 as CTYPE {
                    (unsafe { *index_arr.add(j) }) + dim_size
                } else {
                    unsafe { *index_arr.add(j) }
                };
                let copy_src: *const u8 = unsafe { src.add(index_val as usize * length_per_step) };
                let copy_dest: *mut u8 = unsafe { dest.add(j * length_per_step) };
                unsafe {
                    core::ptr::copy_nonoverlapping(copy_src, copy_dest, length_per_step);
                }
            }
        }
    });

    out
}

// [spec:et:def:op-index.torch.executor.native.index-tensor-out-fn]
// [spec:et:sem:op-index.torch.executor.native.index-tensor-out-fn]
pub fn index_Tensor_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    in_: &Tensor,
    indices: TensorOptList,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    crate::et_kernel_check!(
        ctx,
        tensors_have_same_dim_order2(in_, out),
        InvalidArgument,
        out
    );

    crate::et_kernel_check!(ctx, tensor_is_default_dim_order(in_), InvalidArgument, out);

    let mut dim: usize = 0;
    let is_fast_path = check_fast_path_conditions(in_, indices, &mut dim);
    if is_fast_path {
        return fast_path(ctx, in_, indices, dim, out);
    }

    crate::et_kernel_check!(
        ctx,
        check_index_args(in_, indices, out),
        InvalidArgument,
        out
    );

    let in_type: ScalarType = in_.scalar_type();
    let block_count: usize = count_index_blocks(indices);

    // If indices list is empty or all indices are null, just copy the input to
    // output and return early.
    if block_count == 0 {
        crate::et_kernel_check!(
            ctx,
            resize_tensor(out, in_.sizes()) == Error::Ok,
            InvalidArgument,
            out
        );
        crate::et_switch_realhbbf16_types!(in_type, ctx, "index.Tensor_out", CTYPE, {
            let in_data: *const CTYPE = in_.const_data_ptr::<CTYPE>();
            let out_data: *mut CTYPE = out.mutable_data_ptr::<CTYPE>();
            unsafe {
                core::ptr::copy_nonoverlapping(
                    in_data as *const u8,
                    out_data as *mut u8,
                    in_.nbytes(),
                );
            }
        });
        return out;
    }

    // The output shape depends on whether all the non-null indices are adjacent
    // or not.
    let adjacent: bool = block_count == 1;

    let mut expected_size: [SizesType; K_TENSOR_DIMENSION_LIMIT] = [0; K_TENSOR_DIMENSION_LIMIT];
    let mut expected_ndim: usize = 0;

    crate::et_kernel_check!(
        ctx,
        unsafe {
            get_index_out_target_size(
                in_,
                indices,
                adjacent,
                expected_size.as_mut_ptr(),
                &mut expected_ndim,
            )
        },
        InvalidArgument,
        out
    );

    crate::et_kernel_check!(
        ctx,
        resize_tensor(
            out,
            ArrayRef::from_raw_parts(expected_size.as_ptr(), expected_ndim)
        ) == Error::Ok,
        InvalidArgument,
        out
    );

    if out.numel() == 0 {
        return out;
    }

    let mut dim_map: [i32; K_TENSOR_DIMENSION_LIMIT] = [0; K_TENSOR_DIMENSION_LIMIT];
    let mut ix_map: [i32; K_TENSOR_DIMENSION_LIMIT] = [0; K_TENSOR_DIMENSION_LIMIT];
    let mut start: usize = 0;
    let mut xdim: usize = 0;

    if adjacent {
        start = get_num_leading_null_indices(indices);
    }
    xdim = get_indices_broadcast_ndim(indices);
    unsafe {
        compute_dim_map(in_, indices, dim_map.as_mut_ptr(), block_count == 1);
        compute_index_map(in_, indices, ix_map.as_mut_ptr());
    }

    crate::et_switch_realhbbf16_types!(in_type, ctx, "index.Tensor_out", CTYPE, {
        let in_data: *const CTYPE = in_.const_data_ptr::<CTYPE>();
        let out_data: *mut CTYPE = out.mutable_data_ptr::<CTYPE>();

        let mut out_ix: ssize_t = 0;
        while out_ix < out.numel() {
            let (in_ix, success): (usize, bool) = unsafe {
                get_in_ix(
                    in_,
                    indices,
                    out,
                    out_ix as usize,
                    start,
                    xdim,
                    dim_map.as_mut_ptr(),
                    ix_map.as_mut_ptr(),
                )
            };
            crate::et_kernel_check!(ctx, success, InvalidArgument, out);
            unsafe {
                *out_data.add(out_ix as usize) = *in_data.add(in_ix);
            }
            out_ix += 1;
        }
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
    use crate::runtime::core::portable_type::{BFloat16, Half};
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

    fn test_dtype<INPUT, OUTPUT>()
    where
        INPUT: CppTypeToScalarType + FactoryValue,
        OUTPUT: CppTypeToScalarType + FactoryValue,
    {
        let tf = TensorFactory::<INPUT>::new();
        let tfl = TensorFactory::<i64>::new();
        let tfo = TensorFactory::<OUTPUT>::new();

        let x = tf.make_default(
            vec![3, 2, 4],
            vec![
                INPUT::one(),
                INPUT::one(),
                INPUT::one(),
                INPUT::one(), // [0, 0, :]
                INPUT::zero(),
                INPUT::zero(),
                INPUT::zero(),
                INPUT::zero(), // [0, 1, :]
                INPUT::one(),
                INPUT::one(),
                INPUT::one(),
                INPUT::one(), // [1, 0, :]
                INPUT::zero(),
                INPUT::zero(),
                INPUT::zero(),
                INPUT::zero(), // [1, 1, :]
                INPUT::one(),
                INPUT::one(),
                INPUT::one(),
                INPUT::one(), // [2, 0, :]
                INPUT::zero(),
                INPUT::zero(),
                INPUT::zero(),
                INPUT::zero(), // [2, 1, :]
            ],
        );

        // indices [0, 1], [1, 0], [2, 3]
        let indices_v = [
            Some(tfl.make_default(vec![2], vec![0, 1])),
            Some(tfl.make_default(vec![2], vec![1, 0])),
            Some(tfl.make_default(vec![2], vec![2, 3])),
        ];

        let out_size = vec![2];

        let out_0 = tfo.zeros_default(out_size.clone());
        let mut ctx = context();
        let ret_0 = index_Tensor_out(&mut ctx, &x, oil(&indices_v), &out_0);

        assert_tensor_eq!(*ret_0, out_0);
        assert_tensor_eq!(
            *ret_0,
            tfo.make_default(out_size.clone(), vec![OUTPUT::zero(), OUTPUT::one()])
        );

        // Repeat the test with the same indices representation.
        let out_0_with_mixed = tfo.zeros_default(out_size.clone());
        let ret_0_with_mixed = index_Tensor_out(&mut ctx, &x, oil(&indices_v), &out_0_with_mixed);

        assert_tensor_eq!(*ret_0_with_mixed, out_0_with_mixed);
        assert_tensor_eq!(
            *ret_0_with_mixed,
            tfo.make_default(out_size, vec![OUTPUT::zero(), OUTPUT::one()])
        );
    }

    fn test_dtype_enumerate_in_types() {
        // ET_FORALL_REALHBF16_TYPES with Long index and matching output dtype.
        test_dtype::<u8, u8>();
        test_dtype::<i8, i8>();
        test_dtype::<i16, i16>();
        test_dtype::<i32, i32>();
        test_dtype::<i64, i64>();
        test_dtype::<f32, f32>();
        test_dtype::<f64, f64>();
        test_dtype::<Half, Half>();
        test_dtype::<BFloat16, BFloat16>();
    }

    fn test_indices_with_only_null_tensors_supported<INPUT>()
    where
        INPUT: CppTypeToScalarType + FactoryValue + core::ops::Add<Output = INPUT>,
    {
        let tf = TensorFactory::<INPUT>::new();

        let x = tf.make_default(
            vec![2, 3],
            vec![
                INPUT::one(),
                two::<INPUT>(),
                three::<INPUT>(),
                four::<INPUT>(),
                five::<INPUT>(),
                six::<INPUT>(),
            ],
        );
        let out = tf.zeros_default(vec![2, 3]);

        let indices1 = [None];
        let mut ctx = context();
        index_Tensor_out(&mut ctx, &x, oil(&indices1), &out);
        assert_tensor_eq!(out, x);

        let out = tf.zeros_default(vec![2, 3]);
        let indices2 = [None, None];
        index_Tensor_out(&mut ctx, &x, oil(&indices2), &out);
        assert_tensor_eq!(out, x);
    }

    // Materialize small integer constants at the element type (implicit int->CTYPE
    // conversion in the C++ initializer lists).
    fn two<T: FactoryValue + core::ops::Add<Output = T>>() -> T {
        T::one() + T::one()
    }
    fn three<T: FactoryValue + core::ops::Add<Output = T>>() -> T {
        two::<T>() + T::one()
    }
    fn four<T: FactoryValue + core::ops::Add<Output = T>>() -> T {
        three::<T>() + T::one()
    }
    fn five<T: FactoryValue + core::ops::Add<Output = T>>() -> T {
        four::<T>() + T::one()
    }
    fn six<T: FactoryValue + core::ops::Add<Output = T>>() -> T {
        five::<T>() + T::one()
    }

    fn test_indices_with_only_null_tensors_enumerate_in_types() {
        // ET_FORALL_REALHBF16_TYPES.
        test_indices_with_only_null_tensors_supported::<u8>();
        test_indices_with_only_null_tensors_supported::<i8>();
        test_indices_with_only_null_tensors_supported::<i16>();
        test_indices_with_only_null_tensors_supported::<i32>();
        test_indices_with_only_null_tensors_supported::<i64>();
        test_indices_with_only_null_tensors_supported::<f32>();
        test_indices_with_only_null_tensors_supported::<f64>();
        test_indices_with_only_null_tensors_supported::<Half>();
        test_indices_with_only_null_tensors_supported::<BFloat16>();
    }

    // Run the test by selecting elements in input. `expected` is a Double tensor.
    fn run_test_cases(x: &Tensor, indices: &[Option<Tensor>], expected: &Tensor) {
        let tf = TensorFactory::<f64>::new();

        let expected_sizes = expected.sizes();
        let out_size: Vec<i32> = (0..expected_sizes.size())
            .map(|i| *expected_sizes.at(i))
            .collect();
        let out = tf.ones_default(out_size);

        let mut ctx = context();
        let ret = index_Tensor_out(&mut ctx, x, oil(indices), &out);
        assert_tensor_eq!(out, *ret);
        assert_tensor_eq!(*ret, *expected);
    }

    fn make_x_2_3_4(tf: &TensorFactory<f64>) -> Tensor<'_> {
        tf.make_default(
            vec![2, 3, 4],
            vec![
                1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., // [0, :, :]
                -1., -2., -3., -4., -5., -6., -7., -8., -9., -10., -11., -12., // [1, :, :]
            ],
        )
    }

    //
    // Correctness Tests
    //

    // Bool mask index takes the advanced-index slow path: is_mask_index picks the
    // mask branch, check_mask_indices validates shape, count_trues_in_mask_index
    // (=4) drives the output size via get_indices_broadcast_shape /
    // get_index_out_target_size, and get_in_ix -> get_in_coord -> query_mask_index
    // resolves each selected element (values 1,7,-2,-11). check_index_args,
    // count_index_blocks, get_indices_broadcast_ndim, get_num_indexed_dims,
    // get_num_null_indices, compute_dim_map and compute_index_map are all on the path.
    // [spec:et:sem:op-index.torch.executor.native.index-tensor-out-fn/test]
    // [spec:et:sem:advanced-index-util.torch.executor.is-mask-index-fn/test]
    // [spec:et:sem:advanced-index-util.torch.executor.check-mask-indices-fn/test]
    // [spec:et:sem:advanced-index-util.torch.executor.count-trues-in-mask-index-fn/test]
    // [spec:et:sem:advanced-index-util.torch.executor.query-mask-index-fn/test]
    // [spec:et:sem:advanced-index-util.torch.executor.check-index-args-fn/test]
    // [spec:et:sem:advanced-index-util.torch.executor.count-index-blocks-fn/test]
    // [spec:et:sem:advanced-index-util.torch.executor.get-index-out-target-size-fn/test]
    // [spec:et:sem:advanced-index-util.torch.executor.get-indices-broadcast-shape-fn/test]
    // [spec:et:sem:advanced-index-util.torch.executor.get-indices-broadcast-ndim-fn/test]
    // [spec:et:sem:advanced-index-util.torch.executor.get-num-indexed-dims-fn/test]
    // [spec:et:sem:advanced-index-util.torch.executor.get-num-null-indices-fn/test]
    // [spec:et:sem:advanced-index-util.torch.executor.compute-dim-map-fn/test]
    // [spec:et:sem:advanced-index-util.torch.executor.compute-index-map-fn/test]
    // [spec:et:sem:advanced-index-util.torch.executor.get-in-ix-fn/test]
    // [spec:et:sem:advanced-index-util.torch.executor.get-in-coord-fn/test]
    #[test]
    fn op_index_tensor_out_test_index_mask() {
        let tf = TensorFactory::<f64>::new();
        let tfb = TensorFactory::<bool>::new();
        let x = make_x_2_3_4(&tf);

        let indices = tfb.make_default(
            vec![2, 3, 4],
            vec![
                true, false, false, false, false, false, true, false, false, false, false,
                false, // [0, :, :]
                false, true, false, false, false, false, false, false, false, false, true,
                false, // [1, :, :]
            ],
        );

        let expected = tf.make_default(vec![4], vec![1., 7., -2., -11.]);

        run_test_cases(&x, &[Some(indices)], &expected);
    }

    // Three integral index tensors (Long/Int, plus negative and bool/mixed
    // variants) drive the slow path: check_indices_dtypes accepts Long/Int/Bool,
    // query_integral_index reads each index value, and get_in_coord applies the
    // negative-index wrap (indices_negative) and out-of-bounds check. Same
    // expected result -2,-3 across all five index encodings pins the arithmetic.
    // [spec:et:sem:op-index.torch.executor.native.index-tensor-out-fn/test]
    // [spec:et:sem:advanced-index-util.torch.executor.check-indices-dtypes-fn/test]
    // [spec:et:sem:advanced-index-util.torch.executor.query-integral-index-fn/test]
    #[test]
    fn op_index_tensor_out_test_select_front_dim_all_indexes() {
        let tf = TensorFactory::<f64>::new();
        let tfi = TensorFactory::<i32>::new();
        let tfl = TensorFactory::<i64>::new();
        let tfb = TensorFactory::<bool>::new();
        let x = make_x_2_3_4(&tf);

        let indices = [
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

        let out_size = vec![2];
        let expected = tf.make_default(out_size, vec![-2., -3.]);

        run_test_cases(&x, &indices, &expected);
        run_test_cases(&x, &indices_int, &expected);
        run_test_cases(&x, &indices_negative, &expected);
        run_test_cases(&x, &indices_bool, &expected);
        run_test_cases(&x, &indices_mixed, &expected);
    }

    // [spec:et:sem:op-index.torch.executor.native.index-tensor-out-fn/test]
    #[test]
    fn op_index_tensor_out_test_select_two_values_at_same_index() {
        let tf = TensorFactory::<f64>::new();
        let tfl = TensorFactory::<i64>::new();
        let x = make_x_2_3_4(&tf);

        let indices = [
            Some(tfl.make_default(vec![1, 2], vec![0, 0])),
            Some(tfl.make_default(vec![1, 2], vec![1, 1])),
            Some(tfl.make_default(vec![1, 2], vec![2, 2])),
        ];

        let out_size = vec![1, 2];
        let expected = tf.make_default(out_size, vec![7., 7.]);

        run_test_cases(&x, &indices, &expected);
    }

    // [spec:et:sem:op-index.torch.executor.native.index-tensor-out-fn/test]
    #[test]
    fn op_index_tensor_out_test_indices_fewer_than_input_dim_supported() {
        let tf = TensorFactory::<f64>::new();
        let tfi = TensorFactory::<i32>::new();
        let tfl = TensorFactory::<i64>::new();
        let tfb = TensorFactory::<bool>::new();
        let x = make_x_2_3_4(&tf);

        let indices = [
            Some(tfl.make_default(vec![1], vec![1])),
            Some(tfl.make_default(vec![2], vec![0, 1])),
        ];
        let indices_mixed = [
            Some(tfi.make_default(vec![1], vec![-1])),
            Some(tfb.make_default(vec![3], vec![true, true, false])),
        ];

        let out_size = vec![2, 4];
        let expected = tf.make_default(out_size, vec![-1., -2., -3., -4., -5., -6., -7., -8.]);

        run_test_cases(&x, &indices, &expected);
        run_test_cases(&x, &indices_mixed, &expected);
    }

    // A leading None ([None, Some, Some]) exercises get_num_leading_null_indices
    // (=1) in the adjacent output-layout path; the [None,Some,Some] /
    // [Some,None,Some] / [Some,Some,None] placements move the indexed block and
    // pin the null-index bookkeeping via the differing expected shapes/values.
    // [spec:et:sem:op-index.torch.executor.native.index-tensor-out-fn/test]
    // [spec:et:sem:advanced-index-util.torch.executor.get-num-leading-null-indices-fn/test]
    #[test]
    fn op_index_tensor_out_test_indices_with_null_tensors_supported() {
        let tf = TensorFactory::<f64>::new();
        let tfl = TensorFactory::<i64>::new();
        let x = make_x_2_3_4(&tf);

        let indices0 = [
            None,
            Some(tfl.make_default(vec![1], vec![1])),
            Some(tfl.make_default(vec![2], vec![0, 1])),
        ];
        let expected0 = tf.make_default(vec![2, 2], vec![5., 6., -5., -6.]);
        run_test_cases(&x, &indices0, &expected0);

        let indices1 = [
            Some(tfl.make_default(vec![1], vec![1])),
            None,
            Some(tfl.make_default(vec![2], vec![0, 1])),
        ];
        let expected1 = tf.make_default(vec![2, 3], vec![-1., -5., -9., -2., -6., -10.]);
        run_test_cases(&x, &indices1, &expected1);

        let indices2 = [
            Some(tfl.make_default(vec![1], vec![1])),
            Some(tfl.make_default(vec![2], vec![0, 1])),
            None,
        ];
        let expected2 = tf.make_default(vec![2, 4], vec![-1., -2., -3., -4., -5., -6., -7., -8.]);
        run_test_cases(&x, &indices2, &expected2);
    }

    // PORT-NOTE: guarded in C++ by `ET_SKIP_IF(is_aten, ...)`; non-ATen build runs.
    // [spec:et:sem:op-index.torch.executor.native.index-tensor-out-fn/test]
    #[test]
    fn op_index_tensor_out_test_indices_with_only_null_tensors_supported() {
        test_indices_with_only_null_tensors_enumerate_in_types();
    }

    // [spec:et:sem:op-index.torch.executor.native.index-tensor-out-fn/test]
    #[test]
    fn op_index_tensor_out_test_too_many_null_indices() {
        let tf = TensorFactory::<f64>::new();
        let x = tf.make_default(vec![2, 3], vec![1., 2., 3., 4., 5., 6.]);
        let indices = [None, None, None];
        let out = tf.ones_default(vec![2, 3]);
        let mut ctx = context();
        et_expect_kernel_failure!(ctx, index_Tensor_out(&mut ctx, &x, oil(&indices), &out));
    }

    // PORT-NOTE: guarded in C++ by `ET_SKIP_IF(is_aten, ...)`; non-ATen build runs.
    // [spec:et:sem:op-index.torch.executor.native.index-tensor-out-fn/test]
    #[test]
    fn op_index_tensor_out_test_empty_indices_supported() {
        let tf = TensorFactory::<f32>::new();

        let x = tf.make_default(vec![2], vec![1., 2.]);
        let out = tf.zeros_default(vec![2]);

        let indices: [Option<Tensor>; 0] = [];
        let mut ctx = context();
        index_Tensor_out(&mut ctx, &x, oil(&indices), &out);
        assert_tensor_eq!(out, x);
    }

    //
    // Test that all dtypes are supported
    //

    // [spec:et:sem:op-index.torch.executor.native.index-tensor-out-fn/test]
    #[test]
    fn op_index_tensor_out_test_all_dtypes_supported_for_input() {
        test_dtype_enumerate_in_types();
    }

    // [spec:et:sem:op-index.torch.executor.native.index-tensor-out-fn/test]
    #[test]
    fn op_index_tensor_out_test_all_dtypes_supported_for_index() {
        // Double input, Long / Int index, Double output.
        test_dtype::<f64, f64>();
        test_dtype_int_index::<f64, f64>();
    }

    // Variant of `test_dtype` using an Int index tensor.
    fn test_dtype_int_index<INPUT, OUTPUT>()
    where
        INPUT: CppTypeToScalarType + FactoryValue,
        OUTPUT: CppTypeToScalarType + FactoryValue,
    {
        let tf = TensorFactory::<INPUT>::new();
        let tfi = TensorFactory::<i32>::new();
        let tfo = TensorFactory::<OUTPUT>::new();

        let x = tf.make_default(
            vec![3, 2, 4],
            vec![
                INPUT::one(),
                INPUT::one(),
                INPUT::one(),
                INPUT::one(),
                INPUT::zero(),
                INPUT::zero(),
                INPUT::zero(),
                INPUT::zero(),
                INPUT::one(),
                INPUT::one(),
                INPUT::one(),
                INPUT::one(),
                INPUT::zero(),
                INPUT::zero(),
                INPUT::zero(),
                INPUT::zero(),
                INPUT::one(),
                INPUT::one(),
                INPUT::one(),
                INPUT::one(),
                INPUT::zero(),
                INPUT::zero(),
                INPUT::zero(),
                INPUT::zero(),
            ],
        );

        let indices_v = [
            Some(tfi.make_default(vec![2], vec![0, 1])),
            Some(tfi.make_default(vec![2], vec![1, 0])),
            Some(tfi.make_default(vec![2], vec![2, 3])),
        ];

        let out_size = vec![2];

        let out_0 = tfo.zeros_default(out_size.clone());
        let mut ctx = context();
        let ret_0 = index_Tensor_out(&mut ctx, &x, oil(&indices_v), &out_0);

        assert_tensor_eq!(*ret_0, out_0);
        assert_tensor_eq!(
            *ret_0,
            tfo.make_default(out_size, vec![OUTPUT::zero(), OUTPUT::one()])
        );
    }

    // [spec:et:sem:op-index.torch.executor.native.index-tensor-out-fn/test]
    #[test]
    fn op_index_tensor_out_test_negative_index_supported_for_long() {
        let tf = TensorFactory::<f32>::new();
        let tfl = TensorFactory::<i64>::new();

        let x = tf.make_default(vec![3], vec![1., 2., 3.]);
        let out = tf.zeros_default(vec![1]);
        let expected = tf.make_default(vec![1], vec![3.]);

        let indices = [Some(tfl.make_default(vec![1], vec![-1]))];

        let mut ctx = context();
        let ret = index_Tensor_out(&mut ctx, &x, oil(&indices), &out);
        assert_tensor_eq!(*ret, expected);
    }

    // [spec:et:sem:op-index.torch.executor.native.index-tensor-out-fn/test]
    #[test]
    fn op_index_tensor_out_test_negative_index_supported_for_int() {
        let tf = TensorFactory::<f32>::new();
        let tfi = TensorFactory::<i32>::new();

        let x = tf.make_default(vec![3], vec![1., 2., 3.]);
        let out = tf.zeros_default(vec![1]);
        let expected = tf.make_default(vec![1], vec![3.]);

        let indices = [Some(tfi.make_default(vec![1], vec![-1]))];

        let mut ctx = context();
        let ret = index_Tensor_out(&mut ctx, &x, oil(&indices), &out);
        assert_tensor_eq!(*ret, expected);
    }

    //
    // Death Tests
    //

    // [spec:et:sem:op-index.torch.executor.native.index-tensor-out-fn/test]
    #[test]
    fn op_index_tensor_out_test_index_out_of_bound_dies() {
        let tf = TensorFactory::<i32>::new();
        let tfl = TensorFactory::<i64>::new();

        let x = tf.ones_default(vec![1, 1, 1]);
        let out = tf.zeros_default(vec![1, 1, 1]);
        let index = tfl.make_default(vec![1], vec![5]);

        let indices = [Some(index)];
        let mut ctx = context();
        et_expect_kernel_failure!(ctx, index_Tensor_out(&mut ctx, &x, oil(&indices), &out));
    }

    // [spec:et:sem:op-index.torch.executor.native.index-tensor-out-fn/test]
    #[test]
    fn op_index_tensor_out_test_negative_index_out_of_bound_dies() {
        let tf = TensorFactory::<i32>::new();
        let tfl = TensorFactory::<i64>::new();

        let x = tf.ones_default(vec![1, 1, 1]);
        let out = tf.zeros_default(vec![1, 1, 1]);
        let index = tfl.make_default(vec![1], vec![-5]);

        let indices = [Some(index)];
        let mut ctx = context();
        et_expect_kernel_failure!(ctx, index_Tensor_out(&mut ctx, &x, oil(&indices), &out));
    }

    // [spec:et:sem:op-index.torch.executor.native.index-tensor-out-fn/test]
    #[test]
    fn op_index_tensor_out_test_too_many_boolean_index_count_dies() {
        let tf = TensorFactory::<f32>::new();
        let tfb = TensorFactory::<bool>::new();

        let x = tf.ones_default(vec![1, 1, 1]);
        let out = tf.zeros_default(vec![1, 1, 1]);
        let index = tfb.make_default(vec![3], vec![true, false, false]);

        let indices = [Some(index)];
        let mut ctx = context();
        et_expect_kernel_failure!(ctx, index_Tensor_out(&mut ctx, &x, oil(&indices), &out));
    }

    // [spec:et:sem:op-index.torch.executor.native.index-tensor-out-fn/test]
    #[test]
    fn op_index_tensor_out_test_too_few_boolean_index_count_dies() {
        let tf = TensorFactory::<f32>::new();
        let tfb = TensorFactory::<bool>::new();

        let x = tf.ones_default(vec![4]);
        let out = tf.zeros_default(vec![1]);
        let index = tfb.make_default(vec![1], vec![true]);

        let indices = [Some(index)];
        let mut ctx = context();
        et_expect_kernel_failure!(ctx, index_Tensor_out(&mut ctx, &x, oil(&indices), &out));
    }

    // [spec:et:sem:op-index.torch.executor.native.index-tensor-out-fn/test]
    #[test]
    fn op_index_tensor_out_test_mismatched_index_mask_dies() {
        let tf = TensorFactory::<f32>::new();
        let tfb = TensorFactory::<bool>::new();

        let x = tf.ones_default(vec![4, 4]);
        let out = tf.zeros_default(vec![9]);
        let index = tfb.ones_default(vec![3, 3]);

        let indices = [Some(index)];
        let mut ctx = context();
        et_expect_kernel_failure!(ctx, index_Tensor_out(&mut ctx, &x, oil(&indices), &out));
    }

    // [spec:et:sem:op-index.torch.executor.native.index-tensor-out-fn/test]
    #[test]
    fn op_index_tensor_out_test_mismatched_output_dim_dies() {
        let tf = TensorFactory::<i32>::new();
        let tfl = TensorFactory::<i64>::new();

        let x = tf.zeros_default(vec![2, 4, 7, 5]);
        let index = tfl.make_default(vec![1], vec![3]);

        // Should be {1, 4, 7, 5}.
        let out = tf.zeros_default(vec![2, 4]);

        let indices = [Some(index)];
        let mut ctx = context();
        et_expect_kernel_failure!(ctx, index_Tensor_out(&mut ctx, &x, oil(&indices), &out));
    }

    // [spec:et:sem:op-index.torch.executor.native.index-tensor-out-fn/test]
    #[test]
    fn op_index_tensor_out_test_invalid_indices_dtype_dies() {
        let tf = TensorFactory::<i32>::new();
        let tff = TensorFactory::<f32>::new();

        let x = tf.zeros_default(vec![2, 4, 7, 5]);
        let index = tff.make_default(vec![1], vec![3.0]);

        let out = tf.zeros_default(vec![1, 4, 7, 5]);

        let indices = [Some(index)];
        let mut ctx = context();
        et_expect_kernel_failure!(ctx, index_Tensor_out(&mut ctx, &x, oil(&indices), &out));
    }

    // [spec:et:sem:op-index.torch.executor.native.index-tensor-out-fn/test]
    #[test]
    fn op_index_tensor_out_test_invalid_indices_shapes_dies() {
        let tf = TensorFactory::<f32>::new();
        let tfl = TensorFactory::<i64>::new();

        let x = tf.zeros_default(vec![2, 4, 7, 5]);
        let indices = [
            Some(tfl.make_default(vec![3], vec![1, 1, 1])),
            Some(tfl.make_default(vec![2], vec![1, 2])),
        ];

        let out = tf.ones_default(vec![3, 7, 5]);

        let mut ctx = context();
        et_expect_kernel_failure!(ctx, index_Tensor_out(&mut ctx, &x, oil(&indices), &out));
    }

    // PORT-NOTE: guarded in C++ by `ET_SKIP_IF(is_aten, "")`; non-ATen build runs.
    // [spec:et:sem:op-index.torch.executor.native.index-tensor-out-fn/test]
    #[test]
    fn op_index_tensor_out_test_invalid_indices_shape_dies2() {
        let tf = TensorFactory::<f32>::new();
        let tfl = TensorFactory::<i64>::new();

        let x = tf.zeros_default(vec![4, 4]);
        let indices = [
            Some(tfl.make_default(vec![2, 2], vec![1, 1, 1, 1])),
            Some(tfl.make_default(vec![1, 2], vec![3, 0])),
        ];

        let out = tf.ones_default(vec![4]);

        let mut ctx = context();
        et_expect_kernel_failure!(ctx, index_Tensor_out(&mut ctx, &x, oil(&indices), &out));
    }

    //
    // Dynamic Shape Tests
    //

    // [spec:et:sem:op-index.torch.executor.native.index-tensor-out-fn/test]
    #[test]
    fn op_index_tensor_out_test_upper_bound_out_tensor() {
        let tf = TensorFactory::<f64>::new();
        let tfl = TensorFactory::<i64>::new();
        let x = make_x_2_3_4(&tf);

        let indices = [
            Some(tfl.make_default(vec![1, 2], vec![0, 1])),
            Some(tfl.make_default(vec![1, 2], vec![2, 1])),
            Some(tfl.make_default(vec![1, 2], vec![2, 2])),
        ];

        let out = tf.zeros(vec![5, 5], TensorShapeDynamism::DYNAMIC_BOUND);
        let expected = tf.make_default(vec![1, 2], vec![11., -7.]);

        let mut ctx = context();
        let ret = index_Tensor_out(&mut ctx, &x, oil(&indices), &out);
        assert_tensor_eq!(out, *ret);
        assert_tensor_eq!(*ret, expected);
    }

    //
    // Fast Path Tests
    //

    fn make_x_2_3_4_f32(tf: &TensorFactory<f32>) -> Tensor<'_> {
        tf.make_default(
            vec![2, 3, 4],
            vec![
                1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., // [0, :, :]
                -1., -2., -3., -4., -5., -6., -7., -8., -9., -10., -11., -12., // [1, :, :]
            ],
        )
    }

    // A single non-null 1-D Long index selects the fast path via
    // check_fast_path_conditions (dim=0 found); check_fast_path_args validates the
    // in-bounds index values; get_fast_path_index_out_target_size replaces dim-0 size
    // with the index numel (3), giving the asserted {3,3,4} output the copy loop fills.
    // [spec:et:sem:op-index.torch.executor.native.fast-path-fn/test]
    // [spec:et:sem:op-index.torch.executor.native.index-tensor-out-fn/test]
    // [spec:et:sem:op-index.torch.executor.native.check-fast-path-conditions-fn/test]
    // [spec:et:sem:op-index.torch.executor.native.check-fast-path-args-fn/test]
    // [spec:et:sem:op-index.torch.executor.native.get-fast-path-index-out-target-size-fn/test]
    #[test]
    fn op_index_tensor_out_test_fast_path_first_dim() {
        let tf = TensorFactory::<f32>::new();
        let tfl = TensorFactory::<i64>::new();
        let x = make_x_2_3_4_f32(&tf);

        let indices = [Some(tfl.make_default(vec![3], vec![1, 0, 1])), None, None];

        let out = tf.zeros_default(vec![3, 3, 4]);
        let expected = tf.make_default(
            vec![3, 3, 4],
            vec![
                -1., -2., -3., -4., -5., -6., -7., -8., -9., -10., -11., -12., // [1, :, :]
                1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., // [0, :, :]
                -1., -2., -3., -4., -5., -6., -7., -8., -9., -10., -11., -12., // [1, :, :]
            ],
        );

        let mut ctx = context();
        index_Tensor_out(&mut ctx, &x, oil(&indices), &out);
        assert_tensor_eq!(out, expected);
    }

    // [spec:et:sem:op-index.torch.executor.native.fast-path-fn/test]
    // [spec:et:sem:op-index.torch.executor.native.index-tensor-out-fn/test]
    #[test]
    fn op_index_tensor_out_test_fast_path_middle_dim() {
        let tf = TensorFactory::<f32>::new();
        let tfl = TensorFactory::<i64>::new();
        let x = make_x_2_3_4_f32(&tf);

        let indices = [None, Some(tfl.make_default(vec![5], vec![2, 0, 1, 0, 2]))];

        let out = tf.zeros_default(vec![2, 5, 4]);
        let expected = tf.make_default(
            vec![2, 5, 4],
            vec![
                9., 10., 11., 12., 1., 2., 3., 4., 5., 6., 7., 8., 1., 2., 3., 4., 9., 10., 11.,
                12., // [0, :, :]
                -9., -10., -11., -12., -1., -2., -3., -4., -5., -6., -7., -8., -1., -2., -3., -4.,
                -9., -10., -11., -12., // [1, :, :]
            ],
        );

        let mut ctx = context();
        index_Tensor_out(&mut ctx, &x, oil(&indices), &out);
        assert_tensor_eq!(out, expected);
    }

    // [spec:et:sem:op-index.torch.executor.native.fast-path-fn/test]
    // [spec:et:sem:op-index.torch.executor.native.index-tensor-out-fn/test]
    #[test]
    fn op_index_tensor_out_test_fast_path_last_dim() {
        let tf = TensorFactory::<f32>::new();
        let tfl = TensorFactory::<i64>::new();
        let x = make_x_2_3_4_f32(&tf);

        let indices = [None, None, Some(tfl.make_default(vec![3], vec![2, 0, 1]))];

        let out = tf.zeros_default(vec![2, 3, 3]);
        let expected = tf.make_default(
            vec![2, 3, 3],
            vec![
                3., 1., 2., 7., 5., 6., 11., 9., 10., //
                -3., -1., -2., -7., -5., -6., -11., -9., -10.,
            ],
        );

        let mut ctx = context();
        index_Tensor_out(&mut ctx, &x, oil(&indices), &out);
        assert_tensor_eq!(out, expected);
    }

    // [spec:et:sem:op-index.torch.executor.native.fast-path-fn/test]
    // [spec:et:sem:op-index.torch.executor.native.index-tensor-out-fn/test]
    #[test]
    fn op_index_tensor_out_test_fast_path_zero_dim() {
        let tf = TensorFactory::<f32>::new();
        let tfl = TensorFactory::<i64>::new();

        let x = tf.ones_default(vec![0]);
        let indices = [Some(tfl.zeros_default(vec![0]))];
        let out = tf.zeros_default(vec![0]);
        let expected = tf.ones_default(vec![0]);

        let mut ctx = context();
        index_Tensor_out(&mut ctx, &x, oil(&indices), &out);
        assert_tensor_eq!(out, expected);
    }

    // [spec:et:sem:op-index.torch.executor.native.fast-path-fn/test]
    // [spec:et:sem:op-index.torch.executor.native.index-tensor-out-fn/test]
    #[test]
    fn op_index_tensor_out_test_fast_path1_d_less_elements() {
        let tf = TensorFactory::<f32>::new();
        let tfl = TensorFactory::<i64>::new();

        let x = tf.make_default(vec![5], vec![1., 2., 3., 4., 5.]);
        let indices = [Some(tfl.make_default(vec![3], vec![2, 0, 1]))];
        let out = tf.zeros_default(vec![3]);
        let expected = tf.make_default(vec![3], vec![3., 1., 2.]);

        let mut ctx = context();
        index_Tensor_out(&mut ctx, &x, oil(&indices), &out);
        assert_tensor_eq!(out, expected);
    }

    // [spec:et:sem:op-index.torch.executor.native.fast-path-fn/test]
    // [spec:et:sem:op-index.torch.executor.native.index-tensor-out-fn/test]
    #[test]
    fn op_index_tensor_out_test_fast_path1_d_more_elements() {
        let tf = TensorFactory::<f32>::new();
        let tfl = TensorFactory::<i64>::new();

        let x = tf.make_default(vec![5], vec![1., 2., 3., 4., 5.]);
        let indices = [Some(tfl.make_default(vec![7], vec![2, 0, 1, 3, 3, 4, 1]))];
        let out = tf.zeros_default(vec![7]);
        let expected = tf.make_default(vec![7], vec![3., 1., 2., 4., 4., 5., 2.]);

        let mut ctx = context();
        index_Tensor_out(&mut ctx, &x, oil(&indices), &out);
        assert_tensor_eq!(out, expected);
    }

    // [spec:et:sem:op-index.torch.executor.native.fast-path-fn/test]
    // [spec:et:sem:op-index.torch.executor.native.index-tensor-out-fn/test]
    #[test]
    fn op_index_tensor_out_test_fast_path_upper_bound_out_tensor() {
        let tf = TensorFactory::<f32>::new();
        let tfl = TensorFactory::<i64>::new();
        let x = make_x_2_3_4_f32(&tf);

        let indices = [
            None,
            Some(tfl.make_default(vec![5], vec![2, 0, 1, 0, 2])),
            None,
        ];

        let out = tf.zeros(vec![5, 5, 5], TensorShapeDynamism::DYNAMIC_BOUND);
        let expected = tf.make_default(
            vec![2, 5, 4],
            vec![
                9., 10., 11., 12., 1., 2., 3., 4., 5., 6., 7., 8., 1., 2., 3., 4., 9., 10., 11.,
                12., // [0, :, :]
                -9., -10., -11., -12., -1., -2., -3., -4., -5., -6., -7., -8., -1., -2., -3., -4.,
                -9., -10., -11., -12., // [1, :, :]
            ],
        );

        let mut ctx = context();
        index_Tensor_out(&mut ctx, &x, oil(&indices), &out);
        assert_tensor_eq!(out, expected);
    }

    // [spec:et:sem:op-index.torch.executor.native.fast-path-fn/test]
    // [spec:et:sem:op-index.torch.executor.native.index-tensor-out-fn/test]
    #[test]
    fn op_index_tensor_out_test_fast_path_empty_input() {
        let tf = TensorFactory::<f32>::new();
        let tfl = TensorFactory::<i64>::new();

        let x = tf.ones_default(vec![2, 3, 0, 4]);
        let indices = [
            None,
            Some(tfl.make_default(vec![5], vec![2, 0, 1, 0, 2])),
            None,
        ];
        let out = tf.zeros(vec![5, 5, 5, 5], TensorShapeDynamism::DYNAMIC_BOUND);
        let expected = tf.ones_default(vec![2, 5, 0, 4]);

        let mut ctx = context();
        index_Tensor_out(&mut ctx, &x, oil(&indices), &out);
        assert_tensor_eq!(out, expected);
    }

    // [spec:et:sem:op-index.torch.executor.native.fast-path-fn/test]
    // [spec:et:sem:op-index.torch.executor.native.index-tensor-out-fn/test]
    #[test]
    fn op_index_tensor_out_test_fast_path_negative_index() {
        let tf = TensorFactory::<f32>::new();
        let tfl = TensorFactory::<i64>::new();
        let x = make_x_2_3_4_f32(&tf);

        // Use negative indices in the first dimension: -1, 0, -2.
        let indices = [Some(tfl.make_default(vec![3], vec![-1, 0, -2])), None, None];

        let out = tf.zeros_default(vec![3, 3, 4]);
        let expected = tf.make_default(
            vec![3, 3, 4],
            vec![
                -1., -2., -3., -4., -5., -6., -7., -8., -9., -10., -11., -12., // [1, :, :]
                1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., // [0, :, :]
                1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11.,
                12., // [0, :, :] (-2 wraps to 0)
            ],
        );

        let mut ctx = context();
        index_Tensor_out(&mut ctx, &x, oil(&indices), &out);
        assert_tensor_eq!(out, expected);
    }
}
