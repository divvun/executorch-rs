# kernels/optimized/vec/functional.h

> [spec:et:def:functional.executorch.vec.broadcasting-map-2d-by-1d-fn]
> inline void broadcasting_map_2d_by_1d( const Op& vec_fun, scalar_t* output_data, const scalar_t* input_data, const scalar_t* input_data2, int64_t size, int64_t size2)

> [spec:et:sem:functional.executorch.vec.broadcasting-map-2d-by-1d-fn]
> Convenience wrapper that applies a binary elementwise operation between a
> 2D lhs of shape [size, size2] and a 1D rhs of shape [size2] (broadcast over
> the first dimension). It simply forwards to
> broadcasting_map_3d_and_unsqueezed_3d with outer_size = 1,
> broadcast_size = size, inner_size = size2, passing vec_fun, output_data,
> input_data (as lhs) and input_data2 (as rhs) unchanged.

> [spec:et:def:functional.executorch.vec.broadcasting-map-3d-and-unsqueezed-3d-fn]
> inline void broadcasting_map_3d_and_unsqueezed_3d( const Op& vec_fun, scalar_t* output_data, const scalar_t* lhs, const scalar_t* rhs, int64_t outer_size, int64_t broadcast_size, int64_t inner_size)

> [spec:et:sem:functional.executorch.vec.broadcasting-map-3d-and-unsqueezed-3d-fn]
> Apply a binary elementwise operation vec_fun over lhs viewed as
> [outer_size, broadcast_size, inner_size] and rhs viewed as
> [outer_size, 1, inner_size], broadcasting rhs along the middle
> (broadcast) dimension; the result has lhs's shape and is written to
> output_data. Compute strides: outer_stride_lhs = inner_size * broadcast_size,
> outer_stride_rhs = inner_size, broadcast_stride_lhs = inner_size. For each
> outer_idx in [0, outer_size): let lhs_outer = lhs + outer_idx*outer_stride_lhs,
> output_data_row = output_data + outer_idx*outer_stride_lhs,
> rhs_outer = rhs + outer_idx*outer_stride_rhs. For each broadcast_idx in
> [0, broadcast_size): let lhs_outer_2 = lhs_outer + broadcast_idx*broadcast_stride_lhs
> and output_data_row_2 = output_data_row + broadcast_idx*broadcast_stride_lhs.
> Then, for each inner_idx in [0, inner_size), compute
> output_data_row_2[inner_idx] = vec_fun(lhs_outer_2[inner_idx], rhs_outer[inner_idx]).
> (In the C++ this inner loop is SIMD-vectorized in chunks of Vec::size() with a
> masked remainder tail; the scalar Rust port collapses the lane type to the
> scalar element and iterates one element at a time — semantically identical.)

> [spec:et:def:functional.executorch.vec.broadcasting-map-broadcast-last-dim-fn]
> inline void broadcasting_map_broadcast_last_dim( const Op& vec_fun, scalar_t* output_data, const scalar_t* lhs, const scalar_t* rhs, int64_t outer_size, int64_t broadcast_size)

> [spec:et:sem:functional.executorch.vec.broadcasting-map-broadcast-last-dim-fn]
> Apply a binary elementwise operation vec_fun over lhs viewed as
> [outer_size, broadcast_size] and rhs viewed as [outer_size, 1], broadcasting
> the single rhs element of each outer row across the last dimension; the
> result has lhs's shape and is written to output_data. Set
> outer_stride_lhs = broadcast_size. For each outer_idx in [0, outer_size): let
> lhs_outer = lhs + outer_idx*outer_stride_lhs,
> output_data_row = output_data + outer_idx*outer_stride_lhs, and take the
> scalar rhs value rhs[outer_idx]. Then for each inner_idx in [0, broadcast_size),
> compute output_data_row[inner_idx] = vec_fun(lhs_outer[inner_idx], rhs[outer_idx]).
> (In the C++ rhs[outer_idx] is splatted into a Vec constant reused across the
> row and the last dimension is processed with SIMD chunks plus a masked
> remainder tail; the scalar Rust port collapses to a per-element loop —
> semantically identical.)
