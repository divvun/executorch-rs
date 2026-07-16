//! Literal port of extension/tensor/tensor_accessor.h.

use crate::et_log;
use crate::runtime::core::array_ref::ArrayRef;
use crate::runtime::core::error::{Error, Result};
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::core::portable_type::tensor_impl::{
    DimOrderType, SizesType, StridesType, ssize_t,
};

// PORT-NOTE: `ET_CHECK_MSG` (runtime/platform/assert.h) has no ported target
// yet; the two bounds checks below inline its fatal-abort semantics (via the PAL
// abort path), matching the pattern used in tensor_impl.rs. Unresolved
// cross-module reference.

pub mod internal {
    use super::{SizesType, StridesType, ssize_t};

    /// Base class template storing the underlying data with size and stride
    /// helpers. Inherited by TensorAccessor<> which requires specialization on
    /// rank.
    ///
    /// PORT-NOTE: C++ `TensorAccessorBase<T, N>` carries the compile-time rank
    /// `N` as a template parameter but also stores it as the runtime member
    /// `dim_`. Stable Rust (no `generic_const_exprs`) cannot form the
    /// `TensorAccessor<T, N-1>` return type of `operator[]`, so the rank is
    /// tracked through the runtime `dim_` field alone and `N` is carried as a
    /// const generic solely to keep the "generic over rank" type contract at
    /// the entry point. Pointer arithmetic and behavior are identical.
    // [spec:et:def:tensor-accessor.executorch.extension.internal.tensor-accessor-base]
    pub struct TensorAccessorBase<T, const N: usize> {
        pub(super) data_: *mut T,
        pub(super) sizes_: *const SizesType,
        pub(super) strides_: *const StridesType,
        pub(super) dim_: ssize_t,
    }

    impl<T, const N: usize> TensorAccessorBase<T, N> {
        /// Returns the size of the underlying tensor at the given dimension.
        // [spec:et:def:tensor-accessor.executorch.extension.internal.tensor-accessor-base.size-fn]
        // [spec:et:sem:tensor-accessor.executorch.extension.internal.tensor-accessor-base.size-fn]
        pub fn size(&self, i: ssize_t) -> SizesType {
            // ET_CHECK_MSG(i < dim_ && i >= 0, "Dimension outside of [0, dim_-1], got i")
            if !(i < self.dim_ && i >= 0) {
                crate::runtime::platform::abort::runtime_abort();
            }
            unsafe { *self.sizes_.offset(i) }
        }

        /// Returns the stride of the underlying tensor at the given dimension.
        // [spec:et:def:tensor-accessor.executorch.extension.internal.tensor-accessor-base.stride-fn]
        // [spec:et:sem:tensor-accessor.executorch.extension.internal.tensor-accessor-base.stride-fn]
        pub fn stride(&self, i: ssize_t) -> StridesType {
            // ET_CHECK_MSG(i < dim_ && i >= 0, "Dimension outside of [0, dim_-1], got i")
            if !(i < self.dim_ && i >= 0) {
                crate::runtime::platform::abort::runtime_abort();
            }
            unsafe { *self.strides_.offset(i) }
        }

        // [spec:et:def:tensor-accessor.executorch.extension.internal.tensor-accessor-base.tensor-accessor-base-fn]
        // [spec:et:sem:tensor-accessor.executorch.extension.internal.tensor-accessor-base.tensor-accessor-base-fn]
        pub(super) fn new(
            data: *mut T,
            sizes: *const SizesType,
            strides: *const StridesType,
            dim: ssize_t,
        ) -> Self {
            TensorAccessorBase {
                data_: data,
                sizes_: sizes,
                strides_: strides,
                dim_: dim,
            }
        }
    }
}

/// TensorAccessor template with data type and rank as template parameters. No
/// public constructors, can only be created using make_tensor_accessor from a
/// given executorch::aten::Tensor. Use operator[] to index and obtain a lower
/// rank accessor or the underlying scalar value.
// [spec:et:def:tensor-accessor.executorch.extension.tensor-accessor]
// [spec:et:def:tensor-accessor.executorch.extension.tensor-accessor-t-1]
pub struct TensorAccessor<T, const N: usize> {
    base: internal::TensorAccessorBase<T, N>,
}

impl<T, const N: usize> core::ops::Deref for TensorAccessor<T, N> {
    type Target = internal::TensorAccessorBase<T, N>;
    fn deref(&self) -> &Self::Target {
        &self.base
    }
}

impl<T, const N: usize> TensorAccessor<T, N> {
    /// Index into the the outer most dimension.
    ///
    /// If N > 1, a TensorAccessor with N-1 dimensions. If N == 1, use
    /// `scalar`/`scalar_mut` to obtain a reference to the underlying scalar.
    ///
    /// PORT-NOTE: The C++ `operator[]` returns `TensorAccessor<T, N-1>`. Stable
    /// Rust cannot compute `N-1` in a return type, so the same `N` const
    /// generic is reused and the decremented rank is carried in the runtime
    /// `dim_` field (`N - 1`). The pointer arithmetic and metadata advance are
    /// identical to the C++.
    // [spec:et:def:tensor-accessor.executorch.extension.tensor-accessor.operator-fn]
    // [spec:et:sem:tensor-accessor.executorch.extension.tensor-accessor.operator-fn]
    pub fn index(&self, i: ssize_t) -> TensorAccessor<T, N> {
        TensorAccessor {
            base: internal::TensorAccessorBase::new(
                unsafe {
                    self.base
                        .data_
                        .offset((*self.base.strides_.offset(0)) as isize * i)
                },
                unsafe { self.base.sizes_.offset(1) },
                unsafe { self.base.strides_.offset(1) },
                (N as ssize_t) - 1,
            ),
        }
    }

    /// Index into the single dimension of a rank-1 accessor, returning a
    /// reference to the addressed scalar.
    ///
    /// PORT-NOTE: This is the `TensorAccessor<T, 1>::operator[]` specialization.
    /// Because stable Rust reuses one `TensorAccessor<T, N>` type for every
    /// rank, the scalar-returning overload is exposed as `scalar`/`scalar_mut`
    /// instead of a specialized `operator[]`.
    // [spec:et:def:tensor-accessor.executorch.extension.tensor-accessor-t-1.operator-fn]
    // [spec:et:sem:tensor-accessor.executorch.extension.tensor-accessor-t-1.operator-fn]
    pub fn scalar(&self, i: ssize_t) -> &T {
        unsafe {
            &*self
                .base
                .data_
                .offset((*self.base.strides_.offset(0)) as isize * i)
        }
    }

    pub fn scalar_mut(&mut self, i: ssize_t) -> &mut T {
        unsafe {
            &mut *self
                .base
                .data_
                .offset((*self.base.strides_.offset(0)) as isize * i)
        }
    }

    // [spec:et:def:tensor-accessor.executorch.extension.tensor-accessor.tensor-accessor-fn]
    // [spec:et:sem:tensor-accessor.executorch.extension.tensor-accessor.tensor-accessor-fn]
    // [spec:et:def:tensor-accessor.executorch.extension.tensor-accessor-t-1.tensor-accessor-fn]
    // [spec:et:sem:tensor-accessor.executorch.extension.tensor-accessor-t-1.tensor-accessor-fn]
    fn new(
        data: *mut T,
        sizes: *const SizesType,
        strides: *const StridesType,
        dim: ssize_t,
    ) -> Self {
        TensorAccessor {
            base: internal::TensorAccessorBase::new(data, sizes, strides, dim),
        }
    }
}

/// Creates a TensorAccessor<T, N> from the given tensor. The number of dimension
/// N and the data type T's size must match those of the input tensor. For
/// Executorch tensors, non-trivial dimension order is not supported.
///
/// PORT-NOTE: `T` is const-vs-mutable at the C++ template level to pick
/// `const_data_ptr` vs `mutable_data_ptr`; Rust has no const-qualified generic
/// type, so this port always takes the mutable data pointer (the
/// `std::is_const_v<T>` branch collapses to the mutable one). The scalar-type
/// size check uses `size_of::<T>()`.
// [spec:et:def:tensor-accessor.executorch.extension.make-tensor-accessor-fn]
// [spec:et:sem:tensor-accessor.executorch.extension.make-tensor-accessor-fn]
// [spec:et:def:tensor-accessor.executorch.extension.tensor-accessor.make-tensor-accessor-fn]
// [spec:et:sem:tensor-accessor.executorch.extension.tensor-accessor.make-tensor-accessor-fn]
// [spec:et:def:tensor-accessor.executorch.extension.tensor-accessor-t-1.make-tensor-accessor-fn]
// [spec:et:sem:tensor-accessor.executorch.extension.tensor-accessor-t-1.make-tensor-accessor-fn]
pub fn make_tensor_accessor<T, const N: usize>(tensor: &Tensor) -> Result<TensorAccessor<T, N>> {
    // static_assert(N > 0, ...)
    const {
        assert!(
            N > 0,
            "TensorAccessor is used for indexing tensors, for scalar use *_data_ptr<T>()"
        )
    };

    if (N as ssize_t) != tensor.dim() {
        et_log!(
            Error,
            "Expecting {} dimensions but tensor has {}.",
            N as ssize_t,
            tensor.dim() as ssize_t
        );
        return Err(Error::InvalidArgument);
    }

    if core::mem::size_of::<T>() as ssize_t != tensor.element_size() {
        et_log!(
            Error,
            "Size of data type template argument ({}) not equal to tensor element size ({})",
            core::mem::size_of::<T>() as ssize_t,
            tensor.element_size() as ssize_t
        );
        return Err(Error::InvalidArgument);
    }

    // #ifndef USE_ATEN_LIB
    let dim_order: ArrayRef<DimOrderType> = tensor.dim_order();
    for i in 0..dim_order.size() {
        if *dim_order.at(i) as usize != i {
            et_log!(Error, "Non-trival dim_order not supported.");
            return Err(Error::NotSupported);
        }
    }
    // #endif

    let ptr: *mut T = tensor.mutable_data_ptr::<T>();
    Ok(TensorAccessor::<T, N>::new(
        ptr,
        tensor.sizes().data(),
        tensor.strides().data(),
        N as ssize_t,
    ))
}

#[cfg(test)]
mod tests {
    // Literal port of extension/tensor/test/tensor_accessor_test.cpp
    // (non-ATen `TensorAccessorTest` fixture).
    use super::*;
    use crate::extension::tensor::tensor_ptr::make_tensor_ptr_simple;
    use crate::runtime::core::portable_type::device::{Device, DeviceType};
    use crate::runtime::core::tensor_shape_dynamism::TensorShapeDynamism;
    use alloc::vec::Vec;
    extern crate alloc;

    // Mirrors `TensorAccessorTest::SetUpTestSuite()`'s `runtime_init()`.
    fn setup() {
        crate::runtime::platform::runtime::runtime_init();
    }

    // PORT-NOTE: the C++ `make_tensor_ptr(sizes, data.data(), type)` raw-pointer
    // overload maps to `make_tensor_ptr_simple` (default DYNAMIC_BOUND, no
    // deleter, CPU). Callers own the backing `Vec` for the tensor's lifetime.
    fn make_ptr(
        sizes: Vec<SizesType>,
        data: *mut core::ffi::c_void,
        type_: crate::runtime::core::portable_type::scalar_type::ScalarType,
    ) -> crate::extension::tensor::tensor_ptr::TensorPtr {
        make_tensor_ptr_simple(
            sizes,
            data,
            type_,
            TensorShapeDynamism::DYNAMIC_BOUND,
            None,
            Device::from_type(DeviceType::CPU),
        )
    }

    // [spec:et:sem:tensor-accessor.executorch.extension.make-tensor-accessor-fn/test]
    // [spec:et:sem:tensor-accessor.executorch.extension.tensor-accessor-t-1.operator-fn/test]
    #[test]
    fn tensor_accessor_test_from1d_tensor() {
        setup();
        const K_N: i32 = 16;
        let mut data: Vec<u8> = alloc::vec![0u8; K_N as usize];
        for i in 0..K_N {
            data[i as usize] = i as u8;
        }

        let tensor = make_ptr(
            alloc::vec![K_N],
            data.as_mut_ptr() as *mut core::ffi::c_void,
            crate::runtime::core::portable_type::scalar_type::ScalarType::Byte,
        );
        let tensor_accessor = make_tensor_accessor::<u8, 1>(&tensor.tensor());
        assert!(tensor_accessor.is_ok());
        let accessor = tensor_accessor.unwrap();
        for i in 0..K_N {
            assert_eq!(*accessor.scalar(i as ssize_t), i as u8);
        }
    }

    fn value_at_pos_in_4d_int_tensor(n: i32, c: i32, h: i32, w: i32) -> i32 {
        // just encode the position into the value, assuming dimensions fit in 8 bits
        (n << 24) | (c << 16) | (h << 8) | w
    }

    fn check_4d_int_tensor_accessor(
        accessor: TensorAccessor<i32, 4>,
        n_dim: i32,
        c_dim: i32,
        h_dim: i32,
        w_dim: i32,
    ) {
        for n in 0..n_dim {
            for c in 0..c_dim {
                for h in 0..h_dim {
                    for w in 0..w_dim {
                        let v = *accessor
                            .index(n as ssize_t)
                            .index(c as ssize_t)
                            .index(h as ssize_t)
                            .scalar(w as ssize_t);
                        assert_eq!(v, value_at_pos_in_4d_int_tensor(n, c, h, w));
                    }
                }
            }
        }
    }

    // [spec:et:sem:tensor-accessor.executorch.extension.make-tensor-accessor-fn/test]
    // [spec:et:sem:tensor-accessor.executorch.extension.tensor-accessor.make-tensor-accessor-fn/test]
    // [spec:et:sem:tensor-accessor.executorch.extension.tensor-accessor-t-1.make-tensor-accessor-fn/test]
    // also verifies the private accessor constructors and base constructor:
    // [spec:et:sem:tensor-accessor.executorch.extension.tensor-accessor.tensor-accessor-fn/test]
    // [spec:et:sem:tensor-accessor.executorch.extension.tensor-accessor-t-1.tensor-accessor-fn/test]
    // [spec:et:sem:tensor-accessor.executorch.extension.internal.tensor-accessor-base.tensor-accessor-base-fn/test]
    // [spec:et:sem:tensor-accessor.executorch.extension.tensor-accessor.operator-fn/test]
    // [spec:et:sem:tensor-accessor.executorch.extension.tensor-accessor-t-1.operator-fn/test]
    #[test]
    fn tensor_accessor_test_from4d_tensor() {
        setup();
        const K_N: i32 = 2;
        const K_C: i32 = 8;
        const K_H: i32 = 4;
        const K_W: i32 = 6;
        let mut data: Vec<i32> = alloc::vec![0i32; (K_N * K_C * K_H * K_W) as usize];
        let mut idx = 0usize;
        for n in 0..K_N {
            for c in 0..K_C {
                for h in 0..K_H {
                    for w in 0..K_W {
                        data[idx] = value_at_pos_in_4d_int_tensor(n, c, h, w);
                        idx += 1;
                    }
                }
            }
        }

        let tensor = make_ptr(
            alloc::vec![K_N, K_C, K_H, K_W],
            data.as_mut_ptr() as *mut core::ffi::c_void,
            crate::runtime::core::portable_type::scalar_type::ScalarType::Int,
        );
        let accessor = make_tensor_accessor::<i32, 4>(&tensor.tensor());
        assert!(accessor.is_ok());
        check_4d_int_tensor_accessor(accessor.unwrap(), K_N, K_C, K_H, K_W);
    }

    // PORT-NOTE: `TensorAccessorTest::FromNonContiguousTensor` is `#ifdef
    // USE_ATEN_LIB` (non-contiguous tensors are ATen-only); this port is the
    // portable (`!USE_ATEN_LIB`) branch, so the case is not ported.

    // PORT-NOTE: the C++ `tensor_accessor_test.cpp` suite never exercises
    // `TensorAccessorBase::size`/`stride` directly (it only indexes via
    // `operator[]`). This focused unit test pins those two pure accessors
    // against the sem rules: `size(i)` returns `sizes_[i]`, `stride(i)` returns
    // `strides_[i]` (element strides), for a rank-4 contiguous tensor.
    // [spec:et:sem:tensor-accessor.executorch.extension.internal.tensor-accessor-base.size-fn/test]
    // [spec:et:sem:tensor-accessor.executorch.extension.internal.tensor-accessor-base.stride-fn/test]
    #[test]
    fn tensor_accessor_test_size_and_stride() {
        setup();
        const K_N: i32 = 2;
        const K_C: i32 = 8;
        const K_H: i32 = 4;
        const K_W: i32 = 6;
        let mut data: Vec<i32> = alloc::vec![0i32; (K_N * K_C * K_H * K_W) as usize];
        let tensor = make_ptr(
            alloc::vec![K_N, K_C, K_H, K_W],
            data.as_mut_ptr() as *mut core::ffi::c_void,
            crate::runtime::core::portable_type::scalar_type::ScalarType::Int,
        );
        let accessor = make_tensor_accessor::<i32, 4>(&tensor.tensor()).unwrap();

        // size(i) returns the i-th entry of the size array.
        assert_eq!(accessor.size(0), K_N as SizesType);
        assert_eq!(accessor.size(1), K_C as SizesType);
        assert_eq!(accessor.size(2), K_H as SizesType);
        assert_eq!(accessor.size(3), K_W as SizesType);

        // stride(i) returns the i-th entry of the (contiguous, element-wise)
        // stride array: [C*H*W, H*W, W, 1].
        assert_eq!(accessor.stride(0), (K_C * K_H * K_W) as StridesType);
        assert_eq!(accessor.stride(1), (K_H * K_W) as StridesType);
        assert_eq!(accessor.stride(2), K_W as StridesType);
        assert_eq!(accessor.stride(3), 1 as StridesType);
    }

    // [spec:et:sem:tensor-accessor.executorch.extension.make-tensor-accessor-fn/test]
    #[test]
    fn tensor_accessor_test_fail_on_incorrect_dtype_or_rank() {
        setup();
        const K_N: i32 = 16;
        let mut data: Vec<f32> = alloc::vec![0.0f32; K_N as usize];
        let tensor = make_ptr(
            alloc::vec![K_N],
            data.as_mut_ptr() as *mut core::ffi::c_void,
            crate::runtime::core::portable_type::scalar_type::ScalarType::Float,
        );

        // Tensor has rank 1 but creating accessor with rank 2.
        let fail1 = make_tensor_accessor::<f32, 2>(&tensor.tensor());
        assert!(fail1.is_err());

        // Tensor has dtype float but creating accessor with dtype uint8_t.
        let fail2 = make_tensor_accessor::<u8, 1>(&tensor.tensor());
        assert!(fail2.is_err());
    }

    // #ifndef USE_ATEN_LIB — Dim order is only defined for portable Tensor.
    // [spec:et:sem:tensor-accessor.executorch.extension.make-tensor-accessor-fn/test]
    #[test]
    fn tensor_accessor_test_fail_on_non_trivial_dim_order() {
        setup();
        const K_N: i32 = 8;
        const K_M: i32 = 16;
        let mut data: Vec<f32> = alloc::vec![0.0f32; (K_N * K_M) as usize];
        let tensor = crate::extension::tensor::tensor_ptr::make_tensor_ptr(
            alloc::vec![K_N, K_M],
            data.as_mut_ptr() as *mut core::ffi::c_void,
            /*dim_order=*/ alloc::vec![1, 0],
            /*strides=*/ alloc::vec![1, K_N],
            crate::runtime::core::portable_type::scalar_type::ScalarType::Float,
            TensorShapeDynamism::DYNAMIC_BOUND,
            None,
            Device::from_type(DeviceType::CPU),
        );

        // Non trivial dim order is not supported.
        let fail = make_tensor_accessor::<f32, 2>(&tensor.tensor());
        assert!(fail.is_err());
    }
}
