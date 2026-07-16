//! Literal port of kernels/portable/cpu/util/delinearize_index.cpp.

use crate::runtime::core::array_ref::ArrayRef;
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::core::portable_type::tensor_impl::SizesType;

// PORT-NOTE: `ET_CHECK` is the C++ fatal check; mirrored with a local
// `runtime_abort` on failure, matching the established pattern in
// tensor_util.rs / scalar_type_util.rs.
macro_rules! et_check {
    ($cond:expr) => {
        if !($cond) {
            crate::runtime::platform::abort::runtime_abort();
        }
    };
}

// [spec:et:def:delinearize-index.torch.executor.delinearize-index-fn]
// [spec:et:sem:delinearize-index.torch.executor.delinearize-index-fn]
//
// PORT-NOTE: `out_indexes` is a caller-owned raw buffer written through a raw
// pointer, mirroring the C++ `size_t*` out-parameter (pointer identity /
// partial-fill semantics survive).
pub fn delinearize_index(
    mut linear_index: usize,
    shape: ArrayRef<SizesType>,
    out_indexes: *mut usize,
    out_indexes_len: usize,
) {
    et_check!(shape.size() <= out_indexes_len);
    for i in 0..shape.size() {
        let dim = shape.size() - 1 - i;
        let dim_size = *shape.at(dim) as usize;
        unsafe {
            *out_indexes.add(dim) = linear_index % dim_size;
        }
        linear_index /= dim_size;
    }
}

// [spec:et:def:delinearize-index.torch.executor.delinearize-index-fn]
// [spec:et:sem:delinearize-index.torch.executor.delinearize-index-fn]
pub fn delinearize_index_tensor(
    linear_index: usize,
    t: &Tensor,
    out_indexes: *mut usize,
    out_indexes_len: usize,
) {
    delinearize_index(linear_index, t.sizes(), out_indexes, out_indexes_len);
}
