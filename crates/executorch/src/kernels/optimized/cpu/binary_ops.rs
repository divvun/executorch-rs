//! Literal port of kernels/optimized/cpu/binary_ops.cpp + kernels/optimized/cpu/binary_ops.h.
//!
//! DEVIATION (group): the C++ optimized binary ops dispatch a `Vectorized<T>`
//! lambda through `executorch::vec::broadcasting_map_*`. Following
//! PORTING.md's "Optimized kernels" substitution table, the SIMD lane type
//! collapses to the scalar element type: `vec_fun` here is `Fn(CTYPE, CTYPE) ->
//! CTYPE` and the `broadcasting_map_*` helpers become plain scalar loops with
//! the identical blocked (outer/broadcast/inner) structure. The op files supply
//! scalar lambdas instead of `Vec` lambdas.

use crate::kernels::portable::cpu::util::broadcast_indexes_range::{
    arrayref_begin_ignoring_leading_1s, sizes_match_ignoring_leading_1s,
};
use crate::runtime::core::error::Error;
use crate::runtime::core::exec_aten::exec_aten::SizesType;
use crate::runtime::core::exec_aten::util::tensor_util::{getLeadingDims, resize_tensor};
use crate::runtime::core::portable_type::scalar::Scalar;
use crate::runtime::core::portable_type::scalar_type::ScalarType;
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

// PORT-NOTE: `ET_CHECK_MSG` is a C++ fatal check; mirrored with a local
// `runtime_abort` on failure, matching broadcast_util.rs / tensor_util.rs.
// Format arguments are dropped since a fatal abort follows.
macro_rules! et_check_msg {
    ($cond:expr, $($arg:tt)*) => {
        if !($cond) {
            crate::runtime::platform::abort::runtime_abort();
        }
    };
}

// [spec:et:def:binary-ops.torch.executor.elementwise-optimized-path]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ElementwiseOptimizedPath {
    KNone,
    KTreatAs1d,
    KBroadcast2dBy1d,
    KBroadcast2dBy1dReverseArguments,
    KBroadcastNdByNd,
    KBroadcastNdByNdReverseArguments,
    KBroadcastLastDim,
    KBroadcastLastDimReverseArguments,
}

pub mod internal {
    use super::*;

    /*
      Given two tensors, this function returns the broadcast dim if it exists.
      Returns 0 if no broadcast dim is found.
      Else negative index is used to indicate broadcast dim
      e.g. if size = [a, b, c, 1, e, f] then broadcast dim is -3

      This path aims to handle broadcast of the following form
      A = [a1, a2,., 1, .., an]
      B = [b1, b2,., bm, .., bn]
      OR
      A = [a1, a2,., am, .., an]
      B = [b1, b2,., 1, .., bn]
      Note that this way of determining broadcast dim also works
      when broadcast dim is the last dim.
    */
    // [spec:et:def:binary-ops.torch.executor.internal.get-broadcast-dim-fn]
    // [spec:et:sem:binary-ops.torch.executor.internal.get-broadcast-dim-fn]
    pub fn get_broadcast_dim(lhs: &Tensor, rhs: &Tensor) -> i32 {
        let lhs_begin = arrayref_begin_ignoring_leading_1s(lhs.sizes());
        let lhs_end = lhs.sizes().end();

        let rhs_begin = arrayref_begin_ignoring_leading_1s(rhs.sizes());
        let rhs_end = rhs.sizes().end();

        let lhs_size = unsafe { lhs_end.offset_from(lhs_begin) };
        let rhs_size = unsafe { rhs_end.offset_from(rhs_begin) };

        // Following example is not handled at the moment
        // [1, 3, 4, 5]
        // [2, 3, 4, 5]
        if lhs_size != rhs_size {
            return 0;
        }

        let mut broadcast_dim: i32 = 0;
        // Check
        // 1. if any dim value is 1 (it constitutes a broadcast dim)
        // 2. If more than one dim value is 1 (we cannot handle)
        // 3. If non-1 dim values are equal
        let mut lhs_end = unsafe { lhs_end.sub(1) };
        let mut rhs_end = unsafe { rhs_end.sub(1) };
        while lhs_end != lhs_begin {
            if unsafe { *lhs_end == 1 || *rhs_end == 1 } {
                // If more than one broadcast dim is found, return 0.
                if broadcast_dim != 0 {
                    return 0;
                }
                // negative index is used
                broadcast_dim = unsafe { lhs_end.offset_from(lhs.sizes().end()) } as i32;
            } else if unsafe { *lhs_end != *rhs_end } {
                // If non-1 dim values are not equal, return 0.
                return 0;
            }
            lhs_end = unsafe { lhs_end.sub(1) };
            rhs_end = unsafe { rhs_end.sub(1) };
        }
        broadcast_dim
    }

    // [spec:et:def:binary-ops.torch.executor.internal.select-broadcast-optimized-path-fn]
    // [spec:et:sem:binary-ops.torch.executor.internal.select-broadcast-optimized-path-fn]
    pub fn select_broadcast_optimized_path(lhs: &Tensor, rhs: &Tensor) -> ElementwiseOptimizedPath {
        let lhs_begin = arrayref_begin_ignoring_leading_1s(lhs.sizes());
        let lhs_end = lhs.sizes().end();

        let rhs_begin = arrayref_begin_ignoring_leading_1s(rhs.sizes());
        let rhs_end = rhs.sizes().end();

        let lhs_size = unsafe { lhs_end.offset_from(lhs_begin) };
        let rhs_size = unsafe { rhs_end.offset_from(rhs_begin) };
        if lhs_size == 2 && rhs_size == 1 && unsafe { *lhs_begin.add(1) == *rhs_begin.add(0) } {
            return ElementwiseOptimizedPath::KBroadcast2dBy1d;
        }

        if lhs_size == 1 && rhs_size == 2 && unsafe { *rhs_begin.add(1) == *lhs_begin.add(0) } {
            return ElementwiseOptimizedPath::KBroadcast2dBy1dReverseArguments;
        }

        let broadcast_dim = get_broadcast_dim(lhs, rhs);
        // Right now we dont handle last dim broadcast
        if broadcast_dim < -1 {
            if count_ones_in_range(rhs_begin, rhs_end) == 1 {
                ElementwiseOptimizedPath::KBroadcastNdByNd
            } else {
                ElementwiseOptimizedPath::KBroadcastNdByNdReverseArguments
            }
        } else if broadcast_dim == -1 {
            if count_ones_in_range(lhs_begin, lhs_end) == 1 {
                ElementwiseOptimizedPath::KBroadcastLastDimReverseArguments
            } else {
                ElementwiseOptimizedPath::KBroadcastLastDim
            }
        } else {
            ElementwiseOptimizedPath::KNone
        }
    }

    // Literal stand-in for `std::count_if(begin, end, [](x){ return x == 1; })`.
    fn count_ones_in_range(begin: *const SizesType, end: *const SizesType) -> isize {
        let mut count: isize = 0;
        let mut p = begin;
        while p != end {
            if unsafe { *p } == 1 {
                count += 1;
            }
            p = unsafe { p.add(1) };
        }
        count
    }

    // [spec:et:def:binary-ops.torch.executor.internal.broadcast-elementwise-plan]
    pub struct BroadcastElementwisePlan<'a, 'b> {
        pub lhs: &'a Tensor<'b>,
        pub rhs: &'a Tensor<'b>,
        pub outer_size: i64,
        pub broadcast_size: i64,
        pub inner_size: i64,
    }

    // [spec:et:def:binary-ops.torch.executor.internal.plan-broadcast-elementwise-fn]
    // [spec:et:sem:binary-ops.torch.executor.internal.plan-broadcast-elementwise-fn]
    pub fn plan_broadcast_elementwise<'a, 'b>(
        ctx: &mut KernelRuntimeContext,
        a: &'a Tensor<'b>,
        b: &'a Tensor<'b>,
        out: &Tensor,
        selected_optimized_path: ElementwiseOptimizedPath,
    ) -> Option<BroadcastElementwisePlan<'a, 'b>> {
        let lhs: &'a Tensor<'b>;
        let rhs: &'a Tensor<'b>;
        if selected_optimized_path == ElementwiseOptimizedPath::KBroadcast2dBy1dReverseArguments
            || selected_optimized_path == ElementwiseOptimizedPath::KBroadcastNdByNdReverseArguments
        {
            lhs = b;
            rhs = a;
        } else {
            // Catch failure to update logic when adding new broadcasting possibility.
            debug_assert!(
                selected_optimized_path == ElementwiseOptimizedPath::KBroadcast2dBy1d
                    || selected_optimized_path == ElementwiseOptimizedPath::KBroadcastNdByNd
            );
            lhs = a;
            rhs = b;
        }
        let error = resize_tensor(out, lhs.sizes());
        crate::et_kernel_check_msg!(
            ctx,
            error == Error::Ok,
            InvalidArgument,
            None,
            "Failed to resize output tensor."
        );
        let mut outer_size: i64 = 1;
        let broadcast_size: i64;
        let inner_size: i64;
        if selected_optimized_path == ElementwiseOptimizedPath::KBroadcastNdByNd
            || selected_optimized_path == ElementwiseOptimizedPath::KBroadcastNdByNdReverseArguments
        {
            let broadcast_dim = get_broadcast_dim(lhs, rhs);
            let broadcast_dim_lhs = lhs.dim() as i32 + broadcast_dim;
            let normalized_tensor_size_lhs = get_normalized_tensor_size(lhs, broadcast_dim_lhs);
            outer_size = normalized_tensor_size_lhs[0] as i64;
            broadcast_size = normalized_tensor_size_lhs[1] as i64;
            inner_size = normalized_tensor_size_lhs[2] as i64;
        } else {
            broadcast_size = *lhs.sizes().at((lhs.dim() - 2) as usize) as i64;
            inner_size = *lhs.sizes().at((lhs.dim() - 1) as usize) as i64;
        }
        Some(BroadcastElementwisePlan {
            lhs,
            rhs,
            outer_size,
            broadcast_size,
            inner_size,
        })
    }
}

// [spec:et:def:binary-ops.torch.executor.select-optimized-path-fn]
// [spec:et:sem:binary-ops.torch.executor.select-optimized-path-fn]
pub fn select_optimized_path(a: &Tensor, b: &Tensor, out: &Tensor) -> ElementwiseOptimizedPath {
    let a_type = a.scalar_type();
    let b_type = b.scalar_type();
    let out_type = out.scalar_type();

    if a_type != b_type
        || a_type != out_type
        || a_type == ScalarType::Half
        || a_type == ScalarType::BFloat16
    {
        return ElementwiseOptimizedPath::KNone;
    }
    if a.sizes().equals(b.sizes())
        || (a.numel() == b.numel()
            && (a.numel() == out.numel() || sizes_match_ignoring_leading_1s(a.sizes(), b.sizes())))
    {
        return ElementwiseOptimizedPath::KTreatAs1d;
    }
    internal::select_broadcast_optimized_path(a, b)
}

// [spec:et:def:binary-ops.torch.executor.get-normalized-tensor-size-fn]
// [spec:et:sem:binary-ops.torch.executor.get-normalized-tensor-size-fn]
pub fn get_normalized_tensor_size(a: &Tensor, broadcast_dim: i32) -> [i32; 3] {
    et_check_msg!(
        a.dim() > broadcast_dim as isize,
        "Size of tensor: {}, must be larger than broadcast_dim: {}",
        a.dim(),
        broadcast_dim
    );
    let mut normalized_tensor_size: [i32; 3] = [0; 3];
    normalized_tensor_size[0] = 1;
    normalized_tensor_size[1] = a.size(broadcast_dim as isize) as i32;
    normalized_tensor_size[2] = 1;
    let mut i: i32 = 0;
    while i < broadcast_dim {
        normalized_tensor_size[0] *= a.size(i as isize) as i32;
        i += 1;
    }
    let mut i: i32 = broadcast_dim + 1;
    while (i as isize) < a.dim() {
        normalized_tensor_size[2] *= a.size(i as isize) as i32;
        i += 1;
    }
    normalized_tensor_size
}

// [spec:et:def:binary-ops.torch.executor.handle-last-dim-broadcast-elementwise-fn]
// [spec:et:sem:binary-ops.torch.executor.handle-last-dim-broadcast-elementwise-fn]
pub fn handle_last_dim_broadcast_elementwise<'a, 'b, CTYPE, Op>(
    ctx: &mut KernelRuntimeContext,
    vec_fun: &Op,
    a: &'a Tensor<'b>,
    b: &'a Tensor<'b>,
    out: &'a Tensor<'b>,
    selected_optimized_path: ElementwiseOptimizedPath,
) -> &'a Tensor<'b>
where
    CTYPE: Copy,
    Op: Fn(CTYPE, CTYPE) -> CTYPE,
{
    let lhs: &'a Tensor<'b>;
    let rhs: &'a Tensor<'b>;
    if selected_optimized_path == ElementwiseOptimizedPath::KBroadcastLastDimReverseArguments {
        lhs = b;
        rhs = a;
    } else {
        lhs = a;
        rhs = b;
    }
    let error = resize_tensor(out, lhs.sizes());
    crate::et_kernel_check_msg!(
        ctx,
        error == Error::Ok,
        InvalidArgument,
        out,
        "Failed to resize output tensor."
    );
    let outer_size = getLeadingDims(out, out.dim() as i64 - 1);
    let broadcast_size = out.size(out.dim() - 1);
    // DEVIATION: `executorch::vec::broadcasting_map_broadcast_last_dim` becomes
    // this scalar loop; rhs contributes one element per outer row, lhs
    // contributes `broadcast_size` contiguous elements per row.
    let out_data = out.mutable_data_ptr::<CTYPE>();
    let lhs_data = lhs.const_data_ptr::<CTYPE>();
    let rhs_data = rhs.const_data_ptr::<CTYPE>();
    for outer in 0..outer_size {
        let base = outer * (broadcast_size as usize);
        let rhs_val = unsafe { *rhs_data.add(outer) };
        for j in 0..(broadcast_size as usize) {
            unsafe {
                *out_data.add(base + j) = vec_fun(*lhs_data.add(base + j), rhs_val);
            }
        }
    }
    out
}

// [spec:et:def:binary-ops.torch.executor.handle-broadcast-elementwise-fn]
// [spec:et:sem:binary-ops.torch.executor.handle-broadcast-elementwise-fn]
pub fn handle_broadcast_elementwise<'a, 'b, CTYPE, Op>(
    ctx: &mut KernelRuntimeContext,
    vec_fun: &Op,
    a: &'a Tensor<'b>,
    b: &'a Tensor<'b>,
    out: &'a Tensor<'b>,
    selected_optimized_path: ElementwiseOptimizedPath,
    _alpha: Option<Scalar>,
) -> &'a Tensor<'b>
where
    CTYPE: Copy,
    Op: Fn(CTYPE, CTYPE) -> CTYPE,
{
    if selected_optimized_path == ElementwiseOptimizedPath::KBroadcastLastDim
        || selected_optimized_path == ElementwiseOptimizedPath::KBroadcastLastDimReverseArguments
    {
        return handle_last_dim_broadcast_elementwise::<CTYPE, Op>(
            ctx,
            vec_fun,
            a,
            b,
            out,
            selected_optimized_path,
        );
    }

    let opt_plan = internal::plan_broadcast_elementwise(ctx, a, b, out, selected_optimized_path);
    let opt_plan = match opt_plan {
        Some(plan) => plan,
        None => return out,
    };
    // DEVIATION: `executorch::vec::broadcasting_map_3d_and_unsqueezed_3d` becomes
    // this scalar triple loop; `lhs` is the full [outer, broadcast, inner] block
    // and `rhs` is the unsqueezed [outer, 1, inner] block broadcast along the
    // broadcast axis.
    let out_data = out.mutable_data_ptr::<CTYPE>();
    let lhs_data = opt_plan.lhs.const_data_ptr::<CTYPE>();
    let rhs_data = opt_plan.rhs.const_data_ptr::<CTYPE>();
    let outer_size = opt_plan.outer_size as usize;
    let broadcast_size = opt_plan.broadcast_size as usize;
    let inner_size = opt_plan.inner_size as usize;
    for outer in 0..outer_size {
        let lhs_outer = outer * broadcast_size * inner_size;
        let rhs_outer = outer * inner_size;
        for bcast in 0..broadcast_size {
            let lhs_row = lhs_outer + bcast * inner_size;
            for inner in 0..inner_size {
                unsafe {
                    *out_data.add(lhs_row + inner) = vec_fun(
                        *lhs_data.add(lhs_row + inner),
                        *rhs_data.add(rhs_outer + inner),
                    );
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::internal::{
        get_broadcast_dim, plan_broadcast_elementwise, select_broadcast_optimized_path,
    };
    use super::*;
    use crate::runtime::core::exec_aten::testing_util::tensor_factory::TensorFactory;
    use crate::runtime::core::exec_aten::testing_util::tensor_util::{internal, tensors_are_close};
    use crate::runtime::core::memory_allocator::{MemoryAllocator, MemoryAllocatorBase};
    use crate::runtime::core::portable_type::Half;

    fn context() -> KernelRuntimeContext<'static> {
        crate::runtime::platform::platform::pal_init();
        KernelRuntimeContext::new(
            crate::extension::module::module::null_event_tracer(),
            core::ptr::null_mut::<MemoryAllocator>() as *mut dyn MemoryAllocatorBase,
        )
    }

    // [spec:et:sem:binary-ops.torch.executor.internal.get-broadcast-dim-fn/test]
    #[test]
    fn internal_get_broadcast_dim() {
        let tf = TensorFactory::<f32>::new();

        // rhs has a single broadcast (=1) dim at axis 1 -> negative offset -3.
        let a = tf.zeros_default(vec![2, 3, 4, 5]);
        let b = tf.zeros_default(vec![2, 1, 4, 5]);
        assert_eq!(get_broadcast_dim(&a, &b), -3);

        // lhs has the broadcast dim instead (symmetric).
        assert_eq!(get_broadcast_dim(&b, &a), -3);

        // Last-dim broadcast -> -1.
        let a = tf.zeros_default(vec![3, 4]);
        let b = tf.zeros_default(vec![3, 1]);
        assert_eq!(get_broadcast_dim(&a, &b), -1);

        // Equal non-1 shapes: no broadcast dim -> 0.
        let a = tf.zeros_default(vec![2, 3, 4]);
        let b = tf.zeros_default(vec![2, 3, 4]);
        assert_eq!(get_broadcast_dim(&a, &b), 0);

        // Two broadcast dims are unsupported -> 0.
        let a = tf.zeros_default(vec![2, 1, 4, 1]);
        let b = tf.zeros_default(vec![2, 3, 4, 5]);
        assert_eq!(get_broadcast_dim(&a, &b), 0);

        // Unequal non-1 dims -> 0.
        let a = tf.zeros_default(vec![2, 3, 4]);
        let b = tf.zeros_default(vec![2, 3, 7]);
        assert_eq!(get_broadcast_dim(&a, &b), 0);

        // Mismatched effective rank (after ignoring leading 1s) -> 0.
        let a = tf.zeros_default(vec![3, 4, 5]);
        let b = tf.zeros_default(vec![4, 5]);
        assert_eq!(get_broadcast_dim(&a, &b), 0);
    }

    // [spec:et:sem:binary-ops.torch.executor.internal.select-broadcast-optimized-path-fn/test]
    #[test]
    fn internal_select_broadcast_optimized_path() {
        let tf = TensorFactory::<f32>::new();

        // 2d-by-1d: lhs [3,4], rhs [4] with lhs_begin[1]==rhs_begin[0].
        let lhs = tf.zeros_default(vec![3, 4]);
        let rhs = tf.zeros_default(vec![4]);
        assert_eq!(
            select_broadcast_optimized_path(&lhs, &rhs),
            ElementwiseOptimizedPath::KBroadcast2dBy1d
        );

        // Reversed: lhs [4], rhs [3,4].
        assert_eq!(
            select_broadcast_optimized_path(&rhs, &lhs),
            ElementwiseOptimizedPath::KBroadcast2dBy1dReverseArguments
        );

        // NdByNd: broadcast_dim < -1 and rhs has exactly one dim==1.
        let a = tf.zeros_default(vec![2, 3, 4, 5]);
        let b = tf.zeros_default(vec![2, 1, 4, 5]);
        assert_eq!(
            select_broadcast_optimized_path(&a, &b),
            ElementwiseOptimizedPath::KBroadcastNdByNd
        );

        // NdByNdReverse: rhs (=a here) has no dim==1.
        assert_eq!(
            select_broadcast_optimized_path(&b, &a),
            ElementwiseOptimizedPath::KBroadcastNdByNdReverseArguments
        );

        // Last-dim broadcast: broadcast_dim == -1, lhs has no dim==1.
        let a = tf.zeros_default(vec![3, 4]);
        let b = tf.zeros_default(vec![3, 1]);
        assert_eq!(
            select_broadcast_optimized_path(&a, &b),
            ElementwiseOptimizedPath::KBroadcastLastDim
        );

        // Reversed last-dim: lhs (=b here) has exactly one dim==1.
        assert_eq!(
            select_broadcast_optimized_path(&b, &a),
            ElementwiseOptimizedPath::KBroadcastLastDimReverseArguments
        );

        // No broadcast pattern -> kNone.
        let a = tf.zeros_default(vec![3, 4]);
        let b = tf.zeros_default(vec![3, 4]);
        assert_eq!(
            select_broadcast_optimized_path(&a, &b),
            ElementwiseOptimizedPath::KNone
        );
    }

    // [spec:et:sem:binary-ops.torch.executor.select-optimized-path-fn/test]
    #[test]
    fn select_optimized_path_dispatch() {
        let tf = TensorFactory::<f32>::new();
        let tf_i = TensorFactory::<i32>::new();
        let tf_h = TensorFactory::<Half>::new();

        // Mismatched dtype -> kNone.
        let a = tf.zeros_default(vec![3, 4]);
        let b_i = tf_i.zeros_default(vec![3, 4]);
        let out = tf.zeros_default(vec![3, 4]);
        assert_eq!(
            select_optimized_path(&a, &b_i, &out),
            ElementwiseOptimizedPath::KNone
        );

        // Half type -> kNone even with equal shapes.
        let a_h = tf_h.zeros_default(vec![3, 4]);
        let b_h = tf_h.zeros_default(vec![3, 4]);
        let out_h = tf_h.zeros_default(vec![3, 4]);
        assert_eq!(
            select_optimized_path(&a_h, &b_h, &out_h),
            ElementwiseOptimizedPath::KNone
        );

        // Equal sizes -> kTreatAs1d.
        let a = tf.zeros_default(vec![3, 4]);
        let b = tf.zeros_default(vec![3, 4]);
        let out = tf.zeros_default(vec![3, 4]);
        assert_eq!(
            select_optimized_path(&a, &b, &out),
            ElementwiseOptimizedPath::KTreatAs1d
        );

        // Same numel, same out numel (different shapes) -> kTreatAs1d.
        let a = tf.zeros_default(vec![2, 6]);
        let b = tf.zeros_default(vec![3, 4]);
        let out = tf.zeros_default(vec![4, 3]);
        assert_eq!(
            select_optimized_path(&a, &b, &out),
            ElementwiseOptimizedPath::KTreatAs1d
        );

        // sizes_match_ignoring_leading_1s branch: a.numel()==b.numel() but
        // a.numel()!=out.numel(), so classification falls to the leading-1s check.
        let a = tf.zeros_default(vec![1, 3, 4]);
        let b = tf.zeros_default(vec![3, 4]);
        let out = tf.zeros_default(vec![24]);
        assert_eq!(
            select_optimized_path(&a, &b, &out),
            ElementwiseOptimizedPath::KTreatAs1d
        );

        // Otherwise delegates to the broadcast classifier.
        let a = tf.zeros_default(vec![3, 4]);
        let b = tf.zeros_default(vec![4]);
        let out = tf.zeros_default(vec![3, 4]);
        assert_eq!(
            select_optimized_path(&a, &b, &out),
            ElementwiseOptimizedPath::KBroadcast2dBy1d
        );
    }

    // [spec:et:sem:binary-ops.torch.executor.get-normalized-tensor-size-fn/test]
    #[test]
    fn get_normalized_tensor_size_collapse() {
        let tf = TensorFactory::<f32>::new();
        let a = tf.zeros_default(vec![2, 3, 4, 5]);

        // broadcast_dim 1 -> [outer=2, mid=3, inner=4*5=20].
        assert_eq!(get_normalized_tensor_size(&a, 1), [2, 3, 20]);
        // broadcast_dim 0 -> [outer=1, mid=2, inner=3*4*5=60].
        assert_eq!(get_normalized_tensor_size(&a, 0), [1, 2, 60]);
        // broadcast_dim 3 (last) -> [outer=2*3*4=24, mid=5, inner=1].
        assert_eq!(get_normalized_tensor_size(&a, 3), [24, 5, 1]);
    }

    // PORT-NOTE: `ET_CHECK_MSG` death test. `runtime_abort` calls
    // `libc::abort()`, which terminates the process rather than unwinding, so
    // `#[should_panic]` cannot catch it; ported and `#[ignore]`d per the
    // established convention (see kernels/portable/cpu/util/broadcast_util.rs).
    #[test]
    #[should_panic]
    #[ignore]
    fn get_normalized_tensor_size_check_fails() {
        let tf = TensorFactory::<f32>::new();
        let a = tf.zeros_default(vec![2, 3]);
        // a.dim() == 2, broadcast_dim == 2 violates a.dim() > broadcast_dim.
        let _ = get_normalized_tensor_size(&a, 2);
    }

    // [spec:et:sem:binary-ops.torch.executor.internal.plan-broadcast-elementwise-fn/test]
    #[test]
    fn internal_plan_broadcast_elementwise_2d_by_1d() {
        let tf = TensorFactory::<f32>::new();
        let a = tf.zeros_default(vec![3, 4]);
        let b = tf.zeros_default(vec![4]);
        let out = tf.zeros_default(vec![3, 4]);
        let mut ctx = context();
        let plan = plan_broadcast_elementwise(
            &mut ctx,
            &a,
            &b,
            &out,
            ElementwiseOptimizedPath::KBroadcast2dBy1d,
        )
        .expect("plan should be present");
        // lhs = a, rhs = b; outer stays 1, broadcast = sizes[dim-2], inner = sizes[dim-1].
        assert!(core::ptr::eq(plan.lhs, &a));
        assert!(core::ptr::eq(plan.rhs, &b));
        assert_eq!(plan.outer_size, 1);
        assert_eq!(plan.broadcast_size, 3);
        assert_eq!(plan.inner_size, 4);
    }

    // [spec:et:sem:binary-ops.torch.executor.internal.plan-broadcast-elementwise-fn/test]
    #[test]
    fn internal_plan_broadcast_elementwise_nd_by_nd_reverse() {
        let tf = TensorFactory::<f32>::new();
        // a=[2,1,4,5], b=[2,3,4,5] -> ReverseArguments: lhs=b, rhs=a.
        let a = tf.zeros_default(vec![2, 1, 4, 5]);
        let b = tf.zeros_default(vec![2, 3, 4, 5]);
        let out = tf.zeros_default(vec![2, 3, 4, 5]);
        let mut ctx = context();
        let plan = plan_broadcast_elementwise(
            &mut ctx,
            &a,
            &b,
            &out,
            ElementwiseOptimizedPath::KBroadcastNdByNdReverseArguments,
        )
        .expect("plan should be present");
        assert!(core::ptr::eq(plan.lhs, &b));
        assert!(core::ptr::eq(plan.rhs, &a));
        // broadcast_dim(lhs=b, rhs=a) = -3 -> abs axis 1: outer=2, mid=3, inner=20.
        assert_eq!(plan.outer_size, 2);
        assert_eq!(plan.broadcast_size, 3);
        assert_eq!(plan.inner_size, 20);
    }

    // [spec:et:sem:binary-ops.torch.executor.handle-last-dim-broadcast-elementwise-fn/test]
    #[test]
    fn handle_last_dim_broadcast_elementwise_add() {
        let tf = TensorFactory::<f32>::new();
        // lhs [2,3], rhs [2,1] broadcast along last dim; out += per-row scalar.
        let a = tf.make_default(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = tf.make_default(vec![2, 1], vec![10.0, 20.0]);
        let out = tf.zeros_default(vec![2, 3]);
        let add = |x: f32, y: f32| x + y;
        let mut ctx = context();
        handle_last_dim_broadcast_elementwise::<f32, _>(
            &mut ctx,
            &add,
            &a,
            &b,
            &out,
            ElementwiseOptimizedPath::KBroadcastLastDim,
        );
        let expected = tf.make_default(vec![2, 3], vec![11.0, 12.0, 13.0, 24.0, 25.0, 26.0]);
        assert!(tensors_are_close(
            &out,
            &expected,
            internal::K_DEFAULT_RTOL,
            None
        ));
    }

    // [spec:et:sem:binary-ops.torch.executor.handle-last-dim-broadcast-elementwise-fn/test]
    #[test]
    fn handle_last_dim_broadcast_elementwise_reverse() {
        let tf = TensorFactory::<f32>::new();
        // ReverseArguments: lhs=b, rhs=a. Here a=[2,1] (per-row scalar), b=[2,3].
        let a = tf.make_default(vec![2, 1], vec![100.0, 200.0]);
        let b = tf.make_default(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let out = tf.zeros_default(vec![2, 3]);
        let sub = |x: f32, y: f32| x - y;
        let mut ctx = context();
        handle_last_dim_broadcast_elementwise::<f32, _>(
            &mut ctx,
            &sub,
            &a,
            &b,
            &out,
            ElementwiseOptimizedPath::KBroadcastLastDimReverseArguments,
        );
        // lhs=b elems minus rhs=a per-row scalar.
        let expected = tf.make_default(
            vec![2, 3],
            vec![-99.0, -98.0, -97.0, -196.0, -195.0, -194.0],
        );
        assert!(tensors_are_close(
            &out,
            &expected,
            internal::K_DEFAULT_RTOL,
            None
        ));
    }

    // [spec:et:sem:binary-ops.torch.executor.handle-broadcast-elementwise-fn/test]
    #[test]
    fn handle_broadcast_elementwise_2d_by_1d() {
        let tf = TensorFactory::<f32>::new();
        // lhs [2,3], rhs broadcast [3] over the broadcast axis (rows).
        let a = tf.make_default(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = tf.make_default(vec![3], vec![10.0, 20.0, 30.0]);
        let out = tf.zeros_default(vec![2, 3]);
        let add = |x: f32, y: f32| x + y;
        let mut ctx = context();
        handle_broadcast_elementwise::<f32, _>(
            &mut ctx,
            &add,
            &a,
            &b,
            &out,
            ElementwiseOptimizedPath::KBroadcast2dBy1d,
            None,
        );
        // outer=1, broadcast=2, inner=3: rhs indexed by inner only.
        let expected = tf.make_default(vec![2, 3], vec![11.0, 22.0, 33.0, 14.0, 25.0, 36.0]);
        assert!(tensors_are_close(
            &out,
            &expected,
            internal::K_DEFAULT_RTOL,
            None
        ));
    }

    // [spec:et:sem:binary-ops.torch.executor.handle-broadcast-elementwise-fn/test]
    #[test]
    fn handle_broadcast_elementwise_nd_by_nd() {
        let tf = TensorFactory::<f32>::new();
        // a=[2,2,2] full block, b=[2,1,2] broadcast along axis 1.
        let a = tf.make_default(vec![2, 2, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let b = tf.make_default(vec![2, 1, 2], vec![10.0, 20.0, 30.0, 40.0]);
        let out = tf.zeros_default(vec![2, 2, 2]);
        let add = |x: f32, y: f32| x + y;
        let mut ctx = context();
        handle_broadcast_elementwise::<f32, _>(
            &mut ctx,
            &add,
            &a,
            &b,
            &out,
            ElementwiseOptimizedPath::KBroadcastNdByNd,
            None,
        );
        // broadcast_dim(a,b) = -2 -> abs axis 1: outer=2, broadcast=2, inner=2.
        // rhs indexed by (outer*inner + inner) => the [outer,1,inner] block is
        // reused across the broadcast axis.
        let expected = tf.make_default(
            vec![2, 2, 2],
            vec![11.0, 22.0, 13.0, 24.0, 35.0, 46.0, 37.0, 48.0],
        );
        assert!(tensors_are_close(
            &out,
            &expected,
            internal::K_DEFAULT_RTOL,
            None
        ));
    }

    // [spec:et:sem:binary-ops.torch.executor.handle-broadcast-elementwise-fn/test]
    #[test]
    fn handle_broadcast_elementwise_delegates_last_dim() {
        let tf = TensorFactory::<f32>::new();
        // The last-dim paths must be forwarded to handle_last_dim_broadcast_elementwise.
        let a = tf.make_default(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = tf.make_default(vec![2, 1], vec![10.0, 20.0]);
        let out = tf.zeros_default(vec![2, 3]);
        let add = |x: f32, y: f32| x + y;
        let mut ctx = context();
        handle_broadcast_elementwise::<f32, _>(
            &mut ctx,
            &add,
            &a,
            &b,
            &out,
            ElementwiseOptimizedPath::KBroadcastLastDim,
            None,
        );
        let expected = tf.make_default(vec![2, 3], vec![11.0, 12.0, 13.0, 24.0, 25.0, 26.0]);
        assert!(tensors_are_close(
            &out,
            &expected,
            internal::K_DEFAULT_RTOL,
            None
        ));
    }
}
