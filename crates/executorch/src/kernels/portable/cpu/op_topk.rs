//! Literal port of kernels/portable/cpu/op_topk.cpp.

use crate::runtime::core::array_ref::ArrayRef;
use crate::runtime::core::error::Error;
use crate::runtime::core::exec_aten::util::tensor_util::{
    K_TENSOR_DIMENSION_LIMIT, getLeadingDims, nonempty_size, nonzero_dim, resize_tensor,
    tensor_has_dim, tensors_have_same_dtype2,
};
use crate::runtime::core::memory_allocator::MemoryAllocator;
use crate::runtime::core::portable_type::scalar_type::ScalarType;
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::core::portable_type::tensor_impl::{SizesType, ssize_t};
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

// PORT-NOTE: local `ET_LOG_AND_RETURN_IF_FALSE(cond)` mirror (crate check macro
// drops format args); same pattern as transpose_util.rs / op_flip.rs.
macro_rules! et_log_and_return_if_false {
    ($cond:expr) => {{
        if !($cond) {
            $crate::et_log!(Error, "Check failed ({}): ", ::core::stringify!($cond));
            return false;
        }
    }};
}

// PORT-NOTE: local `ET_CHECK_OR_RETURN_FALSE` mirror that forwards the full
// message + args (the crate-level macro drops args after the leading literal);
// same override as copy_ops_util.rs / tensor_util.rs.
macro_rules! et_check_or_return_false {
    ($cond:expr, $fmt:literal $(, $($arg:tt)*)?) => {{
        if !($cond) {
            $crate::et_log!(
                Error,
                ::core::concat!("Check failed ({}): ", $fmt),
                ::core::stringify!($cond)
                $(, $($arg)*)?
            );
            return false;
        }
    }};
}

// PORT-NOTE: `std::tuple<Tensor&, Tensor&>` becomes `(&Tensor, &Tensor)`;
// `Tensor&` is `&'a Tensor` (interior mutation through `*mut TensorImpl`).
// PORT-NOTE: the scratch `elem_t = std::pair<CTYPE, int64_t>` becomes the tuple
// `(CTYPE, i64)`; `.first` -> `.0`, `.second` -> `.1`. The temp buffer from
// `allocate_temp_memory` is reinterpreted as a `&mut [(CTYPE, i64)]` slice.
// PORT-NOTE: `std::partial_sort` / `std::nth_element` / `std::sort` are mapped to
// `select_nth_unstable_by` + `sort_unstable_by` over the queue slice, matching
// the C++ semantics (first-k sorted / nth partition / sort first k-1). Neither
// C++ nor these Rust selectors are stable, so tie order is unspecified as in C++.

// [spec:et:def:op-topk.torch.executor.native.float-less-than-fn]
// [spec:et:sem:op-topk.torch.executor.native.float-less-than-fn]
trait FloatLessThan: Copy {
    fn float_less_than(x: Self, y: Self) -> bool;
}
macro_rules! impl_float_less_than_int {
    ($t:ty) => {
        impl FloatLessThan for $t {
            #[inline]
            fn float_less_than(x: Self, y: Self) -> bool {
                x < y
            }
        }
    };
}
impl_float_less_than_int!(u8);
impl_float_less_than_int!(i8);
impl_float_less_than_int!(i16);
impl_float_less_than_int!(i32);
impl_float_less_than_int!(i64);
macro_rules! impl_float_less_than_float {
    ($t:ty) => {
        impl FloatLessThan for $t {
            #[inline]
            fn float_less_than(x: Self, y: Self) -> bool {
                use crate::kernels::portable::cpu::util::math_util::isnan_override;
                (!isnan_override(x) && isnan_override(y)) || x < y
            }
        }
    };
}
impl_float_less_than_float!(f32);
impl_float_less_than_float!(f64);
impl_float_less_than_float!(crate::runtime::core::portable_type::Half);
impl_float_less_than_float!(crate::runtime::core::portable_type::BFloat16);

// [spec:et:def:op-topk.torch.executor.native.check-topk-args-fn]
// [spec:et:sem:op-topk.torch.executor.native.check-topk-args-fn]
fn check_topk_args(in_: &Tensor, k: i64, mut dim: i64, values: &Tensor, indices: &Tensor) -> bool {
    et_log_and_return_if_false!(tensors_have_same_dtype2(in_, values));
    et_log_and_return_if_false!(indices.scalar_type() == ScalarType::Long);
    et_log_and_return_if_false!(tensor_has_dim(in_, dim));
    if dim < 0 {
        dim += nonzero_dim(in_) as i64;
    }
    et_check_or_return_false!(
        k >= 0 && k <= nonempty_size(in_, dim as ssize_t) as i64,
        "selected index k out of range; k = {}, dim = {}, in.dim() = {}, nonempty_size(in, dim) = {}",
        k,
        dim,
        in_.dim(),
        nonempty_size(in_, dim as ssize_t)
    );
    true
}

// [spec:et:def:op-topk.torch.executor.native.get-topk-target-size-fn]
// [spec:et:sem:op-topk.torch.executor.native.get-topk-target-size-fn]
//
// # Safety
// `target_size` must point to at least `in.dim()` writable elements.
unsafe fn get_topk_target_size(
    in_: &Tensor,
    k: i64,
    dim: i64,
    target_size: *mut SizesType,
    target_dim: &mut usize,
) -> bool {
    *target_dim = in_.dim() as usize;
    for i in 0..*target_dim {
        if i as i64 == dim {
            unsafe { *target_size.add(i) = k as SizesType };
        } else {
            unsafe { *target_size.add(i) = in_.size(i as ssize_t) as SizesType };
        }
    }
    true
}

// [spec:et:def:op-topk.torch.executor.native.perform-topk-fn]
// [spec:et:sem:op-topk.torch.executor.native.perform-topk-fn]
#[allow(clippy::too_many_arguments)]
fn perform_topk<CTYPE: FloatLessThan>(
    in_: &Tensor,
    k: i64,
    dim: i64,
    largest: bool,
    sorted: bool,
    values: &Tensor,
    indices: &Tensor,
    queue: &mut [(CTYPE, i64)],
) {
    let in_data: *const CTYPE = in_.const_data_ptr::<CTYPE>();
    let values_data: *mut CTYPE = values.mutable_data_ptr::<CTYPE>();
    let indices_data: *mut i64 = indices.mutable_data_ptr::<i64>();

    if in_.dim() == 0 {
        unsafe {
            *values_data = *in_data;
            *indices_data = 0;
        }
        return;
    }

    if k == 0 {
        return;
    }

    let outer_size: usize = getLeadingDims(in_, dim);

    let dim_size: usize = in_.size(dim as ssize_t) as usize;
    let dim_stride: usize = *in_.strides().at(dim as usize) as usize;

    let outer_stride_in: usize = dim_size * dim_stride;
    let outer_stride_out: usize = k as usize * dim_stride;

    let use_partial_sort: bool = k * 64 <= dim_size as i64;

    // Comparators built from `float_less_than`.
    let elem_greater = |x: &(CTYPE, i64), y: &(CTYPE, i64)| -> core::cmp::Ordering {
        if CTYPE::float_less_than(y.0, x.0) {
            core::cmp::Ordering::Less
        } else {
            core::cmp::Ordering::Greater
        }
    };
    let elem_less = |x: &(CTYPE, i64), y: &(CTYPE, i64)| -> core::cmp::Ordering {
        if CTYPE::float_less_than(x.0, y.0) {
            core::cmp::Ordering::Less
        } else {
            core::cmp::Ordering::Greater
        }
    };

    // Loop through all outer dimensions
    for outer_idx in 0..outer_size {
        let outer_in: usize = outer_idx * outer_stride_in;
        let outer_out: usize = outer_idx * outer_stride_out;
        // Loop through all inner dimensions
        for inner_idx in 0..dim_stride {
            let base_in: usize = outer_in + inner_idx;
            let base_out: usize = outer_out + inner_idx;

            // Populate the queue with the values from the input tensor
            for i in 0..dim_size {
                let in_ix: usize = base_in + i * dim_stride;
                queue[i].0 = unsafe { *in_data.add(in_ix) };
                queue[i].1 = i as i64;
            }

            // Perform topk on the queue.
            let ku: usize = k as usize;
            if use_partial_sort {
                // std::partial_sort(queue, queue+k, queue+dim_size, cmp): first k
                // sorted and are the k smallest per cmp.
                queue[..dim_size].select_nth_unstable_by(ku - 1, |a, b| {
                    if largest {
                        elem_greater(a, b)
                    } else {
                        elem_less(a, b)
                    }
                });
                queue[..ku].sort_unstable_by(|a, b| {
                    if largest {
                        elem_greater(a, b)
                    } else {
                        elem_less(a, b)
                    }
                });
            } else {
                // std::nth_element(queue, queue+k-1, queue+dim_size, cmp)
                queue[..dim_size].select_nth_unstable_by(ku - 1, |a, b| {
                    if largest {
                        elem_greater(a, b)
                    } else {
                        elem_less(a, b)
                    }
                });
                if sorted {
                    // std::sort(queue, queue+k-1, cmp): sorts only first k-1,
                    // leaving element k-1 in its nth_element position.
                    queue[..ku - 1].sort_unstable_by(|a, b| {
                        if largest {
                            elem_greater(a, b)
                        } else {
                            elem_less(a, b)
                        }
                    });
                }
            }

            // Write the topk values and indices to the output tensors
            for i in 0..ku {
                let out_ix: usize = base_out + i * dim_stride;
                unsafe {
                    *values_data.add(out_ix) = queue[i].0;
                    *indices_data.add(out_ix) = queue[i].1;
                }
            }
        }
    }
}

// [spec:et:def:op-topk.torch.executor.native.allocate-temp-memory-fn]
// [spec:et:sem:op-topk.torch.executor.native.allocate-temp-memory-fn]
fn allocate_temp_memory(ctx: &mut KernelRuntimeContext, size: usize) -> *mut core::ffi::c_void {
    let temp_mem_res = ctx.allocate_temp(size, MemoryAllocator::K_DEFAULT_ALIGNMENT);
    match temp_mem_res {
        Ok(p) => p,
        Err(_) => core::ptr::null_mut(),
    }
}

// [spec:et:def:op-topk.torch.executor.native.topk-values-fn]
// [spec:et:sem:op-topk.torch.executor.native.topk-values-fn]
#[allow(clippy::too_many_arguments)]
pub fn topk_values<'a, 'b, 'c, 'd>(
    ctx: &mut KernelRuntimeContext,
    in_: &Tensor,
    k: i64,
    mut dim: i64,
    largest: bool,
    sorted: bool,
    values: &'a Tensor<'b>,
    indices: &'c Tensor<'d>,
) -> (&'a Tensor<'b>, &'c Tensor<'d>) {
    let out = (values, indices);

    crate::et_kernel_check!(
        ctx,
        check_topk_args(in_, k, dim, values, indices),
        InvalidArgument,
        out
    );

    if dim < 0 {
        dim += nonzero_dim(in_) as i64;
    }

    let mut target_size: [SizesType; K_TENSOR_DIMENSION_LIMIT] = [0; K_TENSOR_DIMENSION_LIMIT];
    let mut target_dim: usize = 0;
    unsafe {
        get_topk_target_size(in_, k, dim, target_size.as_mut_ptr(), &mut target_dim);
    }

    crate::et_kernel_check!(
        ctx,
        resize_tensor(
            values,
            ArrayRef::from_raw_parts(target_size.as_ptr(), target_dim)
        ) == Error::Ok,
        InvalidArgument,
        out
    );

    crate::et_kernel_check!(
        ctx,
        resize_tensor(
            indices,
            ArrayRef::from_raw_parts(target_size.as_ptr(), target_dim)
        ) == Error::Ok,
        InvalidArgument,
        out
    );

    let name = "topk.values";

    if in_.numel() == 0 || (k == 0 && in_.dim() > 0) {
        return out;
    }

    let mut temp_mem_allocated = false;

    crate::et_switch_realhbf16_types!(in_.scalar_type(), ctx, name, CTYPE, {
        // using elem_t = std::pair<CTYPE, int64_t>;
        let temp_mem_size: usize =
            nonempty_size(in_, dim as ssize_t) as usize * core::mem::size_of::<(CTYPE, i64)>();

        let queue_ptr: *mut core::ffi::c_void = allocate_temp_memory(ctx, temp_mem_size);
        if queue_ptr.is_null() {
            // return from the switch body; leaves temp_mem_allocated == false.
        } else {
            temp_mem_allocated = true;

            let queue: &mut [(CTYPE, i64)] = unsafe {
                core::slice::from_raw_parts_mut(
                    queue_ptr as *mut (CTYPE, i64),
                    nonempty_size(in_, dim as ssize_t) as usize,
                )
            };

            perform_topk::<CTYPE>(in_, k, dim, largest, sorted, values, indices, queue);
        }
    });

    crate::et_kernel_check!(ctx, temp_mem_allocated, MemoryAllocationFailed, out);

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::core::exec_aten::testing_util::tensor_factory::{
        FactoryValue, TensorFactory,
    };
    use crate::runtime::core::exec_aten::util::scalar_type_util::CppTypeToScalarType;
    use crate::runtime::core::memory_allocator::MemoryAllocatorBase;
    use crate::runtime::core::portable_type::{BFloat16, Half};
    use crate::{assert_tensor_close, assert_tensor_eq};

    // PORT-NOTE: the C++ test builds a `TempMemoryAllocator` (a malloc-backed
    // `MemoryAllocator` subclass) fresh inside the `op_topk_values` free function
    // for every call. The ported kernel calls `ctx.allocate_temp`, so the test
    // context must carry a real allocator. This holder owns a heap buffer and a
    // bump `MemoryAllocator` over it, mirroring the fresh-per-call C++ allocator;
    // the buffer is generously sized for the tiny test tensors.
    struct TempAllocatorHolder {
        _buffer: Box<[u8]>,
        allocator: MemoryAllocator,
    }

    impl TempAllocatorHolder {
        fn new() -> Self {
            const CAP: usize = 1 << 16;
            let mut buffer = vec![0u8; CAP].into_boxed_slice();
            let base = buffer.as_mut_ptr();
            let allocator = MemoryAllocator::new(CAP as u32, base);
            TempAllocatorHolder {
                _buffer: buffer,
                allocator,
            }
        }
    }

    // Mirrors the C++ `op_topk_values` free function: allocates a fresh temp
    // allocator per call and invokes the kernel.
    #[allow(clippy::too_many_arguments)]
    fn op_topk_values<'a, 'b, 'c, 'd>(
        input: &Tensor,
        k: i64,
        dim: i64,
        largest: bool,
        sorted: bool,
        values: &'a Tensor<'b>,
        indices: &'c Tensor<'d>,
    ) -> (&'a Tensor<'b>, &'c Tensor<'d>) {
        crate::runtime::platform::runtime::runtime_init();
        let mut holder = TempAllocatorHolder::new();
        let mut context = KernelRuntimeContext::new(
            crate::extension::module::module::null_event_tracer(),
            &mut holder.allocator as *mut dyn MemoryAllocatorBase,
        );
        topk_values(
            &mut context,
            input,
            k,
            dim,
            largest,
            sorted,
            values,
            indices,
        )
    }

    // PORT-NOTE: `static_cast<CTYPE>(int)` bridge for the REALHBF16 factory element
    // types used by the templated smoke test.
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

    fn d<T: FromI32Data>(v: &[i32]) -> Vec<T> {
        v.iter().map(|&x| T::from_i32(x)).collect()
    }

    fn run_smoke_test<T>()
    where
        T: CppTypeToScalarType + FactoryValue + FromI32Data,
    {
        let tf_dtype = TensorFactory::<T>::new();
        let tf_long = TensorFactory::<i64>::new();

        let input = tf_dtype.make_default(
            vec![3, 2, 2],
            d::<T>(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
        );
        let k: i64 = 2;
        let dim: i64 = 0;
        let mut largest = true;
        let sorted = true;
        let values = tf_dtype.zeros_default(vec![2, 2, 2]);
        let indices = tf_long.zeros_default(vec![2, 2, 2]);
        let values_expected =
            tf_dtype.make_default(vec![2, 2, 2], d::<T>(&[9, 10, 11, 12, 5, 6, 7, 8]));
        let indices_expected = tf_long.make_default(vec![2, 2, 2], vec![2, 2, 2, 2, 1, 1, 1, 1]);
        op_topk_values(&input, k, dim, largest, sorted, &values, &indices);
        assert_tensor_close!(values, values_expected);
        assert_tensor_eq!(indices, indices_expected);

        largest = false;
        let values_expected =
            tf_dtype.make_default(vec![2, 2, 2], d::<T>(&[1, 2, 3, 4, 5, 6, 7, 8]));
        let indices_expected = tf_long.make_default(vec![2, 2, 2], vec![0, 0, 0, 0, 1, 1, 1, 1]);
        op_topk_values(&input, k, dim, largest, sorted, &values, &indices);
        assert_tensor_close!(values, values_expected);
        assert_tensor_eq!(indices, indices_expected);
    }

    // [spec:et:sem:op-topk.torch.executor.native.topk-values-fn/test]
    // [spec:et:sem:op-topk.torch.executor.native.perform-topk-fn/test]
    // also verifies: check_topk_args (valid k=2/dim=0 path must pass or the op aborts);
    // get_topk_target_size (k=2 replaces dim-0 size 3, giving the asserted {2,2,2} shape);
    // float_less_than (the largest/smallest sorted orderings [9,10,11,12,...] vs [1,2,...]
    // depend on the comparator); and allocate_temp_memory (a null return would abort with
    // MemoryAllocationFailed instead of producing the expected output).
    // [spec:et:sem:op-topk.torch.executor.native.check-topk-args-fn/test]
    // [spec:et:sem:op-topk.torch.executor.native.get-topk-target-size-fn/test]
    // [spec:et:sem:op-topk.torch.executor.native.float-less-than-fn/test]
    // [spec:et:sem:op-topk.torch.executor.native.allocate-temp-memory-fn/test]
    #[test]
    fn op_topk_values_test_smoke_test() {
        // ET_FORALL_REALHBF16_TYPES
        run_smoke_test::<u8>();
        run_smoke_test::<i8>();
        run_smoke_test::<i16>();
        run_smoke_test::<i32>();
        run_smoke_test::<i64>();
        run_smoke_test::<f32>();
        run_smoke_test::<f64>();
        run_smoke_test::<Half>();
        run_smoke_test::<BFloat16>();
    }

    // [spec:et:sem:op-topk.torch.executor.native.topk-values-fn/test]
    // [spec:et:sem:op-topk.torch.executor.native.perform-topk-fn/test]
    #[test]
    fn op_topk_values_test_non_partial_sort() {
        let tf_float = TensorFactory::<f32>::new();
        let tf_long = TensorFactory::<i64>::new();

        // std::iota(0..100).
        let data: Vec<f32> = (0..100).map(|i| i as f32).collect();

        for largest in [true, false] {
            let input = tf_float.make_default(vec![data.len() as i32], data.clone());
            let values = tf_float.zeros_default(vec![1]);
            let indices = tf_long.zeros_default(vec![1]);
            let values_expected = tf_float.make_default(
                vec![1],
                vec![if largest {
                    *data.last().unwrap()
                } else {
                    data[0]
                }],
            );
            let indices_expected = tf_long.make_default(
                vec![1],
                vec![if largest { data.len() as i64 - 1 } else { 0 }],
            );
            op_topk_values(&input, 1, 0, largest, true, &values, &indices);
            assert_tensor_close!(values, values_expected);
            assert_tensor_eq!(indices, indices_expected);
        }
    }
}
