//! Literal port of runtime/core/tensor_layout.cpp + runtime/core/tensor_layout.h.

use crate::runtime::core::error::Error;
use crate::runtime::core::exec_aten::util::scalar_type_util::element_size;
use crate::runtime::core::portable_type::scalar_type::ScalarType;
use crate::runtime::core::result::Result;
use crate::runtime::core::result::ResultExt;
use crate::runtime::core::span::Span;

// [spec:et:def:tensor-layout.executorch.et-runtime-namespace.calculate-nbytes-fn]
// [spec:et:sem:tensor-layout.executorch.et-runtime-namespace.calculate-nbytes-fn]
fn calculate_nbytes(sizes: &Span<i32>, scalar_type: &ScalarType) -> Result<usize> {
    let mut n: usize = 1;
    for i in 0..sizes.size() {
        if unsafe { *sizes.index(i) } < 0 {
            return Err(Error::InvalidArgument);
        }
        let next: usize;
        // PORT-NOTE: C++ uses `c10::mul_overflows(n, x, &next)`, returning true
        // on overflow. `checked_mul` returns None on overflow, mirroring the
        // wraparound-detected branch.
        match n.checked_mul(unsafe { *sizes.index(i) } as usize) {
            None => {
                return Err(Error::InvalidArgument);
            }
            Some(v) => {
                next = v;
            }
        }
        n = next;
    }
    // Use the full namespace to disambiguate from c10::elementSize.
    let elem_size: usize = element_size(*scalar_type) as usize;
    let total: usize;
    match n.checked_mul(elem_size) {
        None => {
            return Err(Error::InvalidArgument);
        }
        Some(v) => {
            total = v;
        }
    }
    Ok(total)
}

/// Describes the layout of a tensor.
// [spec:et:def:tensor-layout.executorch.et-runtime-namespace.tensor-layout]
pub struct TensorLayout {
    /// The sizes of the tensor.
    sizes_: Span<i32>,

    /// The dim order of the tensor.
    dim_order_: Span<u8>,

    /// The scalar type of the tensor.
    scalar_type_: ScalarType,

    /// The size in bytes of the tensor.
    nbytes_: usize,
}

impl TensorLayout {
    // TensorLayout() = delete; — no default constructor is provided.

    /// Creates a TensorLayout from the given parameters.
    ///
    /// `sizes`: The sizes of the tensor. Note: the span passed here must
    /// outlive the TensorLayout and all copies of it.
    /// `dim_order`: The dim order of the tensor. Note: the span passed here
    /// must outlive the TensorLayout and all copies of it.
    /// `scalar_type`: The scalar type of the tensor.
    // [spec:et:def:tensor-layout.executorch.et-runtime-namespace.tensor-layout.create-fn]
    // [spec:et:sem:tensor-layout.executorch.et-runtime-namespace.tensor-layout.create-fn]
    pub fn create(
        sizes: Span<i32>,
        dim_order: Span<u8>,
        scalar_type: ScalarType,
    ) -> Result<TensorLayout> {
        let nbytes = calculate_nbytes(&sizes, &scalar_type);
        if !ResultExt::ok(&nbytes) {
            return Err(ResultExt::error(&nbytes));
        }

        if dim_order.size() != sizes.size() {
            return Err(Error::InvalidArgument);
        }

        for i in 0..dim_order.size() {
            if unsafe { *dim_order.index(i) } as usize >= sizes.size() {
                return Err(Error::InvalidArgument);
            }
        }
        Ok(TensorLayout::new(
            sizes,
            dim_order,
            scalar_type,
            *nbytes.get(),
        ))
    }

    /// Returns the sizes of the tensor.
    ///
    /// NOTE: The TensorLayout must outlive the spans returned here.
    // [spec:et:def:tensor-layout.executorch.et-runtime-namespace.tensor-layout.sizes-fn]
    // [spec:et:sem:tensor-layout.executorch.et-runtime-namespace.tensor-layout.sizes-fn]
    pub fn sizes(&self) -> Span<i32> {
        self.sizes_
    }

    /// Returns the dim order of the tensor.
    ///
    /// NOTE: The TensorLayout must outlive the spans returned here.
    // [spec:et:def:tensor-layout.executorch.et-runtime-namespace.tensor-layout.dim-order-fn]
    // [spec:et:sem:tensor-layout.executorch.et-runtime-namespace.tensor-layout.dim-order-fn]
    pub fn dim_order(&self) -> Span<u8> {
        self.dim_order_
    }

    /// Returns the scalar type of the tensor.
    // [spec:et:def:tensor-layout.executorch.et-runtime-namespace.tensor-layout.scalar-type-fn]
    // [spec:et:sem:tensor-layout.executorch.et-runtime-namespace.tensor-layout.scalar-type-fn]
    pub fn scalar_type(&self) -> ScalarType {
        self.scalar_type_
    }

    /// Returns the size of the tensor in bytes.
    // [spec:et:def:tensor-layout.executorch.et-runtime-namespace.tensor-layout.nbytes-fn]
    // [spec:et:sem:tensor-layout.executorch.et-runtime-namespace.tensor-layout.nbytes-fn]
    pub fn nbytes(&self) -> usize {
        self.nbytes_
    }

    // [spec:et:def:tensor-layout.executorch.et-runtime-namespace.tensor-layout.tensor-layout-fn]
    // [spec:et:sem:tensor-layout.executorch.et-runtime-namespace.tensor-layout.tensor-layout-fn]
    fn new(sizes: Span<i32>, dim_order: Span<u8>, scalar_type: ScalarType, nbytes: usize) -> Self {
        TensorLayout {
            sizes_: sizes,
            dim_order_: dim_order,
            scalar_type_: scalar_type,
            nbytes_: nbytes,
        }
    }
}

// Literal port of runtime/core/test/tensor_layout_test.cpp.
#[cfg(test)]
mod tests {
    use super::*;

    // [spec:et:sem:tensor-layout.executorch.et-runtime-namespace.tensor-layout.create-fn/test]
    // [spec:et:sem:tensor-layout.executorch.et-runtime-namespace.tensor-layout.scalar-type-fn/test]
    // [spec:et:sem:tensor-layout.executorch.et-runtime-namespace.tensor-layout.sizes-fn/test]
    // [spec:et:sem:tensor-layout.executorch.et-runtime-namespace.tensor-layout.dim-order-fn/test]
    // [spec:et:sem:tensor-layout.executorch.et-runtime-namespace.tensor-layout.nbytes-fn/test]
    // [spec:et:sem:tensor-layout.executorch.et-runtime-namespace.calculate-nbytes-fn/test]
    // also verifies TensorLayout::new (the private ctor create() delegates to)
    // by checking every field it stores, and result::check_ok via ResultExt::get.
    // [spec:et:sem:tensor-layout.executorch.et-runtime-namespace.tensor-layout.tensor-layout-fn/test]
    // [spec:et:sem:result.executorch.runtime.result.check-ok-fn/test]
    #[test]
    fn test_tensor_layout_ctor() {
        let mut sizes: [i32; 2] = [1, 2];
        let mut dim_order: [u8; 2] = [0, 1];
        let sizes_span = Span::from_raw_parts(sizes.as_mut_ptr(), sizes.len());
        let dim_order_span = Span::from_raw_parts(dim_order.as_mut_ptr(), dim_order.len());

        let layout_res = TensorLayout::create(sizes_span, dim_order_span, ScalarType::Float);
        assert!(ResultExt::ok(&layout_res));

        let layout = ResultExt::get(&layout_res);
        assert_eq!(layout.scalar_type(), ScalarType::Float);

        assert_eq!(layout.sizes().size(), sizes_span.size());
        assert_eq!(unsafe { *layout.sizes().index(0) }, unsafe {
            *sizes_span.index(0)
        });
        assert_eq!(unsafe { *layout.sizes().index(1) }, unsafe {
            *sizes_span.index(1)
        });

        assert_eq!(layout.dim_order().size(), dim_order_span.size());
        assert_eq!(unsafe { *layout.dim_order().index(0) }, unsafe {
            *dim_order_span.index(0)
        });
        assert_eq!(unsafe { *layout.dim_order().index(1) }, unsafe {
            *dim_order_span.index(1)
        });

        assert_eq!(layout.nbytes(), 8);
    }

    // [spec:et:sem:tensor-layout.executorch.et-runtime-namespace.tensor-layout.create-fn/test]
    #[test]
    fn test_tensor_layout_ctor_invalid_dim_order() {
        let mut sizes: [i32; 1] = [2];
        let mut dim_order: [u8; 1] = [1];
        let sizes_span = Span::from_raw_parts(sizes.as_mut_ptr(), sizes.len());
        let dim_order_span = Span::from_raw_parts(dim_order.as_mut_ptr(), dim_order.len());

        let layout_res = TensorLayout::create(sizes_span, dim_order_span, ScalarType::Float);
        assert_eq!(ResultExt::error(&layout_res), Error::InvalidArgument);
    }

    // [spec:et:sem:tensor-layout.executorch.et-runtime-namespace.tensor-layout.create-fn/test]
    // [spec:et:sem:tensor-layout.executorch.et-runtime-namespace.calculate-nbytes-fn/test]
    #[test]
    fn test_tensor_layout_ctor_invalid_sizes() {
        let mut sizes: [i32; 1] = [-1];
        let mut dim_order: [u8; 1] = [0];
        let sizes_span = Span::from_raw_parts(sizes.as_mut_ptr(), sizes.len());
        let dim_order_span = Span::from_raw_parts(dim_order.as_mut_ptr(), dim_order.len());

        let layout_res = TensorLayout::create(sizes_span, dim_order_span, ScalarType::Float);
        assert_eq!(ResultExt::error(&layout_res), Error::InvalidArgument);
    }

    // [spec:et:sem:tensor-layout.executorch.et-runtime-namespace.tensor-layout.create-fn/test]
    #[test]
    fn test_tensor_layout_ctor_sizes_dim_order_mismatch() {
        let mut sizes: [i32; 1] = [2];
        let mut dim_order: [u8; 2] = [0, 1];
        let sizes_span = Span::from_raw_parts(sizes.as_mut_ptr(), sizes.len());
        let dim_order_span = Span::from_raw_parts(dim_order.as_mut_ptr(), dim_order.len());

        let layout_res = TensorLayout::create(sizes_span, dim_order_span, ScalarType::Float);
        assert_eq!(ResultExt::error(&layout_res), Error::InvalidArgument);
    }
}
