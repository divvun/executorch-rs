//! Literal port of kernels/optimized/vec/functional.h.
//!
//! DEVIATION: the C++ operates on `at::vec::Vectorized<scalar_t>` SIMD lanes
//! (chunked main loop of `Vec::size()` plus a masked remainder tail). Per the
//! optimized-kernels substitution rules the SIMD lane type collapses to the
//! scalar element type, so each inner loop becomes a plain per-element scalar
//! loop and the `Op` becomes a scalar binary function
//! `Fn(scalar_t, scalar_t) -> scalar_t`. The blocked outer/broadcast structure
//! and pointer arithmetic are preserved verbatim.

// This function implements broadcasting binary operation on two tensors
// where lhs tensor is treated to be of shape [outer_size, broadcast_size, inner_size]
// and rhs tensor is treated to be of shape [outer_size, 1, inner_size]
// And this 1st dimension is considered broadcasting dimension
// This formula can map broadcasting on any dim=broadcast_dim
// for any two N dimensional tensors, where 0 < braodcast_dim < N-1
// [spec:et:def:functional.executorch.vec.broadcasting-map-3d-and-unsqueezed-3d-fn]
// [spec:et:sem:functional.executorch.vec.broadcasting-map-3d-and-unsqueezed-3d-fn]
///
/// # Safety
/// `output_data`/`lhs` must be valid for `outer_size * broadcast_size *
/// inner_size` elements and `rhs` for `outer_size * inner_size` elements, with
/// `output_data` non-aliasing the inputs, matching the C++ raw-pointer
/// contract at the call sites.
#[inline]
pub unsafe fn broadcasting_map_3d_and_unsqueezed_3d<ScalarT, Op>(
    vec_fun: &Op,
    output_data: *mut ScalarT,
    lhs: *const ScalarT,
    rhs: *const ScalarT,
    outer_size: i64,
    broadcast_size: i64,
    inner_size: i64,
) where
    ScalarT: Copy,
    Op: Fn(ScalarT, ScalarT) -> ScalarT,
{
    let outer_stride_lhs: i64 = inner_size * broadcast_size;
    let outer_stride_rhs: i64 = inner_size;
    let broadcast_stride_lhs: i64 = inner_size;
    let mut outer_idx: i64 = 0;
    while outer_idx < outer_size {
        let lhs_outer = unsafe { lhs.offset((outer_idx * outer_stride_lhs) as isize) };
        let output_data_row =
            unsafe { output_data.offset((outer_idx * outer_stride_lhs) as isize) };
        let rhs_outer = unsafe { rhs.offset((outer_idx * outer_stride_rhs) as isize) };
        let mut broadcast_idx: i64 = 0;
        while broadcast_idx < broadcast_size {
            let lhs_outer_2 =
                unsafe { lhs_outer.offset((broadcast_idx * broadcast_stride_lhs) as isize) };
            let output_data_row_2 =
                unsafe { output_data_row.offset((broadcast_idx * broadcast_stride_lhs) as isize) };
            let mut inner_idx: i64 = 0;
            while inner_idx < inner_size {
                let data = unsafe { *lhs_outer_2.offset(inner_idx as isize) };
                let data2 = unsafe { *rhs_outer.offset(inner_idx as isize) };
                let output = vec_fun(data, data2);
                unsafe {
                    *output_data_row_2.offset(inner_idx as isize) = output;
                }
                inner_idx += 1;
            }
            broadcast_idx += 1;
        }
        outer_idx += 1;
    }
}

// [spec:et:def:functional.executorch.vec.broadcasting-map-2d-by-1d-fn]
// [spec:et:sem:functional.executorch.vec.broadcasting-map-2d-by-1d-fn]
///
/// # Safety
/// Same raw-pointer validity/non-aliasing contract as
/// [`broadcasting_map_3d_and_unsqueezed_3d`], with `input_data` of shape
/// `[size, size2]` and `input_data2` of shape `[size2]`.
#[inline]
pub unsafe fn broadcasting_map_2d_by_1d<ScalarT, Op>(
    vec_fun: &Op,
    output_data: *mut ScalarT,
    input_data: *const ScalarT,
    input_data2: *const ScalarT,
    size: i64,
    size2: i64,
) where
    ScalarT: Copy,
    Op: Fn(ScalarT, ScalarT) -> ScalarT,
{
    unsafe {
        broadcasting_map_3d_and_unsqueezed_3d(
            vec_fun,
            output_data,
            input_data,
            input_data2,
            1,
            size,
            size2,
        );
    }
}

// Following function is used to implement broadcasting binary operation on two tensors
// where lhs tensor is treated to be of shape [outer_size, broadcast_size] and
// rhs tensor is treated to be of shape [outer_size, 1]
// Any two N dimensional tensors can be mapped to this formula
// when lhs size = [lhs0, lhs1, ..., lhsN-1] and rhs size = [rhs0, rhs1, ..., 1]
// by viewing the two tensors as
// lhs size = [lsh0 * lsh1 * ... * lshN-2, lhsN-1]
// rhs size = [rsh0 * rsh1 * ... * rshN-2, 1]
// [spec:et:def:functional.executorch.vec.broadcasting-map-broadcast-last-dim-fn]
// [spec:et:sem:functional.executorch.vec.broadcasting-map-broadcast-last-dim-fn]
///
/// # Safety
/// `output_data`/`lhs` must be valid for `outer_size * broadcast_size`
/// elements and `rhs` for `outer_size` elements, with `output_data`
/// non-aliasing the inputs, matching the C++ raw-pointer contract.
#[inline]
pub unsafe fn broadcasting_map_broadcast_last_dim<ScalarT, Op>(
    vec_fun: &Op,
    output_data: *mut ScalarT,
    lhs: *const ScalarT,
    rhs: *const ScalarT,
    outer_size: i64,
    broadcast_size: i64,
) where
    ScalarT: Copy,
    Op: Fn(ScalarT, ScalarT) -> ScalarT,
{
    let outer_stride_lhs: i64 = broadcast_size;
    let mut outer_idx: i64 = 0;
    while outer_idx < outer_size {
        let lhs_outer = unsafe { lhs.offset((outer_idx * outer_stride_lhs) as isize) };
        let output_data_row =
            unsafe { output_data.offset((outer_idx * outer_stride_lhs) as isize) };
        let mut inner_idx: i64 = 0;
        let data2 = unsafe { *rhs.offset(outer_idx as isize) };
        while inner_idx < broadcast_size {
            let data = unsafe { *lhs_outer.offset(inner_idx as isize) };
            let output = vec_fun(data, data2);
            unsafe {
                *output_data_row.offset(inner_idx as isize) = output;
            }
            inner_idx += 1;
        }
        outer_idx += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::{
        broadcasting_map_2d_by_1d, broadcasting_map_3d_and_unsqueezed_3d,
        broadcasting_map_broadcast_last_dim,
    };

    // lhs [outer=2, broadcast=2, inner=3] against rhs [outer=2, 1, inner=3]:
    // out[o][b][i] = f(lhs[o][b][i], rhs[o][i]). A non-commutative op
    // (2*a + b) pins the (lhs, rhs) argument order.
    // [spec:et:sem:functional.executorch.vec.broadcasting-map-3d-and-unsqueezed-3d-fn/test]
    #[test]
    fn broadcasting_map_3d_and_unsqueezed_3d_broadcasts_middle_dim() {
        let lhs: Vec<f32> = (1..=12).map(|v| v as f32).collect();
        let rhs: Vec<f32> = vec![10.0, 20.0, 30.0, 100.0, 200.0, 300.0];
        let mut out = vec![0.0f32; 12];
        let f = |a: f32, b: f32| a * 2.0 + b;
        unsafe {
            broadcasting_map_3d_and_unsqueezed_3d(
                &f,
                out.as_mut_ptr(),
                lhs.as_ptr(),
                rhs.as_ptr(),
                2,
                2,
                3,
            );
        }
        let expected: Vec<f32> = vec![
            12.0, 24.0, 36.0, // o=0, b=0: 2*[1,2,3] + [10,20,30]
            18.0, 30.0, 42.0, // o=0, b=1: 2*[4,5,6] + [10,20,30]
            114.0, 216.0, 318.0, // o=1, b=0: 2*[7,8,9] + [100,200,300]
            120.0, 222.0, 324.0, // o=1, b=1: 2*[10,11,12] + [100,200,300]
        ];
        assert_eq!(out, expected);

        // Same bookkeeping over an integer element type, broadcast_size=1
        // degenerate case: pure elementwise over [outer, 1, inner].
        let lhs_i: Vec<i32> = vec![1, 2, 3, 4];
        let rhs_i: Vec<i32> = vec![10, 20, 30, 40];
        let mut out_i = vec![0i32; 4];
        let add = |a: i32, b: i32| a + b;
        unsafe {
            broadcasting_map_3d_and_unsqueezed_3d(
                &add,
                out_i.as_mut_ptr(),
                lhs_i.as_ptr(),
                rhs_i.as_ptr(),
                2,
                1,
                2,
            );
        }
        assert_eq!(out_i, vec![11, 22, 33, 44]);
    }

    // [size, size2] x [size2]: each row of the 2d input is combined with the
    // single 1d row, i.e. the 3d mapping with outer_size=1.
    // [spec:et:sem:functional.executorch.vec.broadcasting-map-2d-by-1d-fn/test]
    #[test]
    fn broadcasting_map_2d_by_1d_applies_row_to_each() {
        let input: Vec<i32> = (1..=12).collect();
        let input2: Vec<i32> = vec![2, 3, 4, 5];
        let mut out = vec![0i32; 12];
        let mul = |a: i32, b: i32| a * b;
        unsafe {
            broadcasting_map_2d_by_1d(
                &mul,
                out.as_mut_ptr(),
                input.as_ptr(),
                input2.as_ptr(),
                3,
                4,
            );
        }
        let expected: Vec<i32> = vec![
            2, 6, 12, 20, // [1,2,3,4] * [2,3,4,5]
            10, 18, 28, 40, // [5,6,7,8] * [2,3,4,5]
            18, 30, 44, 60, // [9,10,11,12] * [2,3,4,5]
        ];
        assert_eq!(out, expected);
    }

    // lhs [outer, broadcast] against rhs [outer, 1]: every element of row o is
    // combined with the scalar rhs[o]. Subtraction pins the argument order
    // f(lhs_elem, rhs[o]).
    // [spec:et:sem:functional.executorch.vec.broadcasting-map-broadcast-last-dim-fn/test]
    #[test]
    fn broadcasting_map_broadcast_last_dim_scalar_per_row() {
        let lhs: Vec<i32> = (1..=12).collect();
        let rhs: Vec<i32> = vec![1, 10, 100];
        let mut out = vec![0i32; 12];
        let sub = |a: i32, b: i32| a - b;
        unsafe {
            broadcasting_map_broadcast_last_dim(
                &sub,
                out.as_mut_ptr(),
                lhs.as_ptr(),
                rhs.as_ptr(),
                3,
                4,
            );
        }
        let expected: Vec<i32> = vec![
            0, 1, 2, 3, // [1,2,3,4] - 1
            -5, -4, -3, -2, // [5,6,7,8] - 10
            -91, -90, -89, -88, // [9,10,11,12] - 100
        ];
        assert_eq!(out, expected);

        // f32 with a single row (outer_size=1).
        let lhs_f: Vec<f32> = vec![0.5, 1.5, 2.5];
        let rhs_f: Vec<f32> = vec![2.0];
        let mut out_f = vec![0.0f32; 3];
        let div = |a: f32, b: f32| a / b;
        unsafe {
            broadcasting_map_broadcast_last_dim(
                &div,
                out_f.as_mut_ptr(),
                lhs_f.as_ptr(),
                rhs_f.as_ptr(),
                1,
                3,
            );
        }
        assert_eq!(out_f, vec![0.25, 0.75, 1.25]);
    }
}
