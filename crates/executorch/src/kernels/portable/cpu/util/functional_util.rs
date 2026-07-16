//! Literal port of kernels/portable/cpu/util/functional_util.h.

use crate::runtime::kernel::thread_parallel_interface::{internal::GRAIN_SIZE, parallel_for};

//
// Reduction
//

/// For `size` elements of `data_in`, accumulates the modified values into a
/// value using `reduce_fun`, and returns the accumulated value. The `stride` can
/// also be defined; by default it is set to 1.
// [spec:et:def:functional-util.torch.executor.apply-unary-reduce-fn-fn]
// [spec:et:sem:functional-util.torch.executor.apply-unary-reduce-fn-fn]
//
// PORT-NOTE: raw input pointer mirrors the C++ `const CTYPE* const data_in`;
// the caller guarantees `size >= 1` and a `size * stride`-addressable buffer.
pub fn apply_unary_reduce_fn<CTYPE, ReduceOp>(
    reduce_fun: ReduceOp,
    data_in: *const CTYPE,
    size: i64,
    stride: i64,
) -> CTYPE
where
    CTYPE: Copy,
    ReduceOp: Fn(CTYPE, CTYPE) -> CTYPE,
{
    let mut acc_val: CTYPE = unsafe { *data_in };
    for i in 1..size {
        acc_val = reduce_fun(unsafe { *data_in.offset((i * stride) as isize) }, acc_val);
    }
    acc_val
}

//
// Mapping
//

/// Applies `map_fun` to `size` elements of `data_in`, writing results to
/// `data_out`. The `stride` can also be defined; by default it is set to 1.
// [spec:et:def:functional-util.torch.executor.apply-unary-map-fn-fn]
// [spec:et:sem:functional-util.torch.executor.apply-unary-map-fn-fn]
//
// PORT-NOTE: `data_in`/`data_out` are raw pointers as in C++; the parallel_for
// closure borrows them by value (raw pointers are Copy), reproducing the C++
// capture-by-reference semantics.
pub fn apply_unary_map_fn<CTYPE_IN, CTYPE_OUT, MapOp>(
    map_fun: MapOp,
    data_in: *const CTYPE_IN,
    data_out: *mut CTYPE_OUT,
    size: i64,
    stride: i64,
) where
    CTYPE_IN: Copy,
    MapOp: Fn(CTYPE_IN) -> CTYPE_OUT,
{
    parallel_for(0, size, GRAIN_SIZE, &|begin, end| {
        for i in begin..end {
            unsafe {
                *data_out.offset((i * stride) as isize) =
                    map_fun(*data_in.offset((i * stride) as isize));
            }
        }
    });
}

//
// Mapping + Reduction
//

/// Applies `map_fun` to `size` elements of `data_in`, accumulates the modified
/// values into a value using `reduce_fun`, and returns the accumulated value.
/// The `stride` can also be defined; by default it is set to 1.
// [spec:et:def:functional-util.torch.executor.apply-unary-map-reduce-fn-fn]
// [spec:et:sem:functional-util.torch.executor.apply-unary-map-reduce-fn-fn]
//
// PORT-NOTE: sequential left fold (not parallelized), matching the C++; the
// caller guarantees `size >= 1`.
pub fn apply_unary_map_reduce_fn<CTYPE_IN, CTYPE_OUT, MapOp, ReduceOp>(
    map_fun: MapOp,
    reduce_fun: ReduceOp,
    data_in: *const CTYPE_IN,
    size: i64,
    stride: i64,
) -> CTYPE_OUT
where
    CTYPE_IN: Copy,
    CTYPE_OUT: Copy,
    MapOp: Fn(CTYPE_IN) -> CTYPE_OUT,
    ReduceOp: Fn(CTYPE_OUT, CTYPE_OUT) -> CTYPE_OUT,
{
    let mut acc_val: CTYPE_OUT = map_fun(unsafe { *data_in });
    for i in 1..size {
        acc_val = reduce_fun(
            map_fun(unsafe { *data_in.offset((i * stride) as isize) }),
            acc_val,
        );
    }
    acc_val
}
