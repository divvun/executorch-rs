//! Literal port of runtime/core/portable_type/tensor.h.

use crate::runtime::core::array_ref::ArrayRef;
use crate::runtime::core::portable_type::device::{Device, DeviceIndex, DeviceType};
use crate::runtime::core::portable_type::scalar_type::ScalarType;
use crate::runtime::core::portable_type::tensor_impl::{
    DimOrderType, SizesType, StridesType, TensorImpl, ssize_t,
};
use crate::runtime::core::tensor_shape_dynamism::TensorShapeDynamism;

/// A minimal Tensor type whose API is a source compatible subset of at::Tensor.
///
/// NOTE: Instances of this class do not own the TensorImpl given to it, which
/// means that the caller must guarantee that the TensorImpl lives longer than
/// any Tensor instances that point to it.
///
/// See the documention on TensorImpl for details about the return/parameter
/// types used here and how they relate to at::Tensor.
// [spec:et:def:tensor.executorch.runtime.etensor.tensor]
// PORT-NOTE: C++ `Tensor` holds a non-owning `TensorImpl*`. Per PORTING.md the
// non-owning handle is modeled as a raw pointer carried with a lifetime marker;
// `set_data` is `const` on the Tensor yet mutates the impl (which the Tensor
// does not own), so a raw `*mut TensorImpl` is retained (mirroring the impl
// aliasing pattern) rather than a `&'a` shared reference.
pub struct Tensor<'a> {
    impl_: *mut TensorImpl,
    _marker: core::marker::PhantomData<&'a TensorImpl>,
}

/// The type used for elements of `sizes()`.
pub type TensorSizesType = SizesType;
/// The type used for elements of `dim_order()`.
pub type TensorDimOrderType = DimOrderType;
/// The type used for elements of `strides()`.
pub type TensorStridesType = StridesType;

impl<'a> Tensor<'a> {
    // Tensor() = delete;
    // explicit constexpr Tensor(TensorImpl* impl) : impl_(impl) {}
    // [spec:et:def:tensor.executorch.runtime.etensor.tensor.tensor-fn]
    // [spec:et:sem:tensor.executorch.runtime.etensor.tensor.tensor-fn]
    // PORT-NOTE: the deleted default ctor has no Rust analog (there is simply no
    // parameterless constructor). `new` stores the given `TensorImpl*` with no
    // other work: no allocation, copy, validation, or null check.
    pub fn new(impl_: *mut TensorImpl) -> Self {
        Tensor {
            impl_,
            _marker: core::marker::PhantomData,
        }
    }

    /// Returns a pointer to the underlying TensorImpl.
    ///
    /// NOTE: Clients should be wary of operating on the TensorImpl directly
    /// instead of the Tensor. It is easy to break things.
    // [spec:et:def:tensor.executorch.runtime.etensor.tensor.unsafe-get-tensor-impl-fn]
    // [spec:et:sem:tensor.executorch.runtime.etensor.tensor.unsafe-get-tensor-impl-fn]
    pub fn unsafe_get_tensor_impl(&self) -> *mut TensorImpl {
        // TODO(T154114015): See if we can make this api private with friends.
        self.impl_
    }

    /// Returns the size of the tensor in bytes.
    ///
    /// NOTE: Only the alive space is returned not the total capacity of the
    /// underlying data blob.
    // [spec:et:def:tensor.executorch.runtime.etensor.tensor.nbytes-fn]
    // [spec:et:sem:tensor.executorch.runtime.etensor.tensor.nbytes-fn]
    pub fn nbytes(&self) -> usize {
        unsafe { (*self.impl_).nbytes() }
    }

    /// Returns the size of the tensor at the given dimension.
    ///
    /// NOTE: that size() intentionally does not return SizeType even though it
    /// returns an element of an array of SizeType. This is to help make calls
    /// of this method more compatible with at::Tensor.
    // [spec:et:def:tensor.executorch.runtime.etensor.tensor.size-fn]
    // [spec:et:sem:tensor.executorch.runtime.etensor.tensor.size-fn]
    pub fn size(&self, dim: ssize_t) -> ssize_t {
        unsafe { (*self.impl_).size(dim) }
    }

    /// Returns the tensor's number of dimensions.
    // [spec:et:def:tensor.executorch.runtime.etensor.tensor.dim-fn]
    // [spec:et:sem:tensor.executorch.runtime.etensor.tensor.dim-fn]
    pub fn dim(&self) -> ssize_t {
        unsafe { (*self.impl_).dim() }
    }

    /// Returns the number of elements in the tensor.
    // [spec:et:def:tensor.executorch.runtime.etensor.tensor.numel-fn]
    // [spec:et:sem:tensor.executorch.runtime.etensor.tensor.numel-fn]
    pub fn numel(&self) -> ssize_t {
        unsafe { (*self.impl_).numel() }
    }

    /// Returns the type of the elements in the tensor (int32, float, bool, etc).
    // [spec:et:def:tensor.executorch.runtime.etensor.tensor.scalar-type-fn]
    // [spec:et:sem:tensor.executorch.runtime.etensor.tensor.scalar-type-fn]
    pub fn scalar_type(&self) -> ScalarType {
        unsafe { (*self.impl_).scalar_type() }
    }

    // [spec:et:def:tensor.executorch.runtime.etensor.tensor.dtype-fn]
    // [spec:et:sem:tensor.executorch.runtime.etensor.tensor.dtype-fn]
    pub fn dtype(&self) -> ScalarType {
        self.scalar_type()
    }

    /// Returns the size in bytes of one element of the tensor.
    // [spec:et:def:tensor.executorch.runtime.etensor.tensor.element-size-fn]
    // [spec:et:sem:tensor.executorch.runtime.etensor.tensor.element-size-fn]
    pub fn element_size(&self) -> ssize_t {
        unsafe { (*self.impl_).element_size() }
    }

    /// Returns the sizes of the tensor at each dimension.
    // [spec:et:def:tensor.executorch.runtime.etensor.tensor.sizes-fn]
    // [spec:et:sem:tensor.executorch.runtime.etensor.tensor.sizes-fn]
    pub fn sizes(&self) -> ArrayRef<SizesType> {
        unsafe { (*self.impl_).sizes() }
    }

    /// Returns the order the dimensions are laid out in memory.
    // [spec:et:def:tensor.executorch.runtime.etensor.tensor.dim-order-fn]
    // [spec:et:sem:tensor.executorch.runtime.etensor.tensor.dim-order-fn]
    pub fn dim_order(&self) -> ArrayRef<DimOrderType> {
        unsafe { (*self.impl_).dim_order() }
    }

    /// Returns the strides of the tensor at each dimension.
    // [spec:et:def:tensor.executorch.runtime.etensor.tensor.strides-fn]
    // [spec:et:sem:tensor.executorch.runtime.etensor.tensor.strides-fn]
    pub fn strides(&self) -> ArrayRef<StridesType> {
        unsafe { (*self.impl_).strides() }
    }

    /// Returns the mutability of the shape of the tensor.
    // [spec:et:def:tensor.executorch.runtime.etensor.tensor.shape-dynamism-fn]
    // [spec:et:sem:tensor.executorch.runtime.etensor.tensor.shape-dynamism-fn]
    pub fn shape_dynamism(&self) -> TensorShapeDynamism {
        unsafe { (*self.impl_).shape_dynamism() }
    }

    /// Returns the device where tensor data resides.
    // [spec:et:def:tensor.executorch.runtime.etensor.tensor.device-fn]
    // [spec:et:sem:tensor.executorch.runtime.etensor.tensor.device-fn]
    pub fn device(&self) -> Device {
        unsafe { (*self.impl_).device() }
    }

    /// Returns the type of device where tensor data resides.
    // [spec:et:def:tensor.executorch.runtime.etensor.tensor.device-type-fn]
    // [spec:et:sem:tensor.executorch.runtime.etensor.tensor.device-type-fn]
    pub fn device_type(&self) -> DeviceType {
        unsafe { (*self.impl_).device_type() }
    }

    /// Returns the device index, or 0 if default/unspecified.
    // [spec:et:def:tensor.executorch.runtime.etensor.tensor.device-index-fn]
    // [spec:et:sem:tensor.executorch.runtime.etensor.tensor.device-index-fn]
    pub fn device_index(&self) -> DeviceIndex {
        unsafe { (*self.impl_).device_index() }
    }

    /// Returns a pointer of type T to the constant underlying data blob.
    // [spec:et:def:tensor.executorch.runtime.etensor.tensor.const-data-ptr-fn]
    // [spec:et:sem:tensor.executorch.runtime.etensor.tensor.const-data-ptr-fn]
    pub fn const_data_ptr<T>(&self) -> *const T {
        unsafe { (*self.impl_).data::<T>() }
    }

    /// Returns a pointer to the constant underlying data blob.
    pub fn const_data_ptr_typed(&self) -> *const core::ffi::c_void {
        unsafe { (*self.impl_).data_typed() }
    }

    /// Returns a pointer of type T to the mutable underlying data blob.
    // [spec:et:def:tensor.executorch.runtime.etensor.tensor.mutable-data-ptr-fn]
    // [spec:et:sem:tensor.executorch.runtime.etensor.tensor.mutable-data-ptr-fn]
    pub fn mutable_data_ptr<T>(&self) -> *mut T {
        unsafe { (*self.impl_).mutable_data::<T>() }
    }

    /// Returns a pointer to the mutable underlying data blob.
    pub fn mutable_data_ptr_typed(&self) -> *mut core::ffi::c_void {
        unsafe { (*self.impl_).mutable_data_typed() }
    }

    /// DEPRECATED: Use const_data_ptr or mutable_data_ptr instead.
    // [spec:et:def:tensor.executorch.runtime.etensor.tensor.data-ptr-fn]
    // [spec:et:sem:tensor.executorch.runtime.etensor.tensor.data-ptr-fn]
    #[deprecated]
    pub fn data_ptr<T>(&self) -> *mut T {
        unsafe { (*self.impl_).mutable_data::<T>() }
    }

    /// DEPRECATED: Use const_data_ptr or mutable_data_ptr instead.
    #[deprecated]
    pub fn data_ptr_typed(&self) -> *mut core::ffi::c_void {
        unsafe { (*self.impl_).mutable_data_typed() }
    }

    /// DEPRECATED: Changes the data_ptr the tensor aliases. Does not free the
    /// previously pointed to data, does not assume ownership semantics of the
    /// new ptr. This api does not exist in at::Tensor so kernel developers
    /// should avoid it.
    // [spec:et:def:tensor.executorch.runtime.etensor.tensor.set-data-fn]
    // [spec:et:sem:tensor.executorch.runtime.etensor.tensor.set-data-fn]
    // PORT-NOTE: C++ method is `const` on the Tensor yet mutates the (non-owned)
    // impl. The impl is held as `*mut TensorImpl`, so this takes `&self` and
    // calls the impl's `set_data` through the raw pointer, matching the C++
    // const-on-handle / mutate-through-pointer semantics.
    #[deprecated]
    pub fn set_data(&self, ptr: *mut core::ffi::c_void) {
        unsafe { (*self.impl_).set_data(ptr) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::platform::runtime::runtime_init;

    // TEST_F fixture SetUp(): the tests trigger ET_LOG, so the PAL must be
    // initialized first.
    fn setup() {
        runtime_init();
    }

    fn as_void<T>(p: *mut T) -> *mut core::ffi::c_void {
        p as *mut core::ffi::c_void
    }

    // Helper mirroring `TensorImpl(type, dim, sizes, data, dim_order, strides)`
    // (STATIC dynamism, CPU/0 device) — the common C++ construction with default
    // arguments filled in explicitly.
    fn make_impl(
        type_: ScalarType,
        dim: ssize_t,
        sizes: *mut SizesType,
        data: *mut core::ffi::c_void,
        dim_order: *mut DimOrderType,
        strides: *mut StridesType,
    ) -> TensorImpl {
        TensorImpl::new(
            type_,
            dim,
            sizes,
            data,
            dim_order,
            strides,
            TensorShapeDynamism::STATIC,
            DeviceType::CPU,
            0,
        )
    }

    // PORT-NOTE: `ET_EXPECT_DEATH` death test — constructing a TensorImpl with an
    // invalid/undefined scalar type aborts. `runtime_abort` ->
    // `std::process::abort()` terminates the process, so `#[should_panic]` cannot
    // catch it; ported and `#[ignore]`d per the established death-test convention.
    // Each C++ ET_EXPECT_DEATH line becomes its own ignored death test.
    // [spec:et:sem:tensor.executorch.runtime.etensor.tensor.tensor-fn/test]
    #[test]
    #[should_panic]
    #[ignore]
    fn tensor_test_invalid_scalar_type_undefined() {
        setup();
        let mut sizes: [SizesType; 1] = [1];
        let _y = make_impl(
            ScalarType::Undefined,
            1,
            sizes.as_mut_ptr(),
            core::ptr::null_mut(),
            core::ptr::null_mut(),
            core::ptr::null_mut(),
        );
    }

    // [spec:et:sem:tensor.executorch.runtime.etensor.tensor.tensor-fn/test]
    #[test]
    #[should_panic]
    #[ignore]
    fn tensor_test_invalid_scalar_type_num_options() {
        setup();
        let mut sizes: [SizesType; 1] = [1];
        let _y = make_impl(
            ScalarType::NumOptions,
            1,
            sizes.as_mut_ptr(),
            core::ptr::null_mut(),
            core::ptr::null_mut(),
            core::ptr::null_mut(),
        );
    }

    // [spec:et:sem:tensor.executorch.runtime.etensor.tensor.tensor-fn/test]
    #[test]
    #[should_panic]
    #[ignore]
    fn tensor_test_invalid_scalar_type_127() {
        setup();
        let mut sizes: [SizesType; 1] = [1];
        let _y = make_impl(
            unsafe { core::mem::transmute::<i8, ScalarType>(127) },
            1,
            sizes.as_mut_ptr(),
            core::ptr::null_mut(),
            core::ptr::null_mut(),
            core::ptr::null_mut(),
        );
    }

    // [spec:et:sem:tensor.executorch.runtime.etensor.tensor.tensor-fn/test]
    #[test]
    #[should_panic]
    #[ignore]
    fn tensor_test_invalid_scalar_type_neg1() {
        setup();
        let mut sizes: [SizesType; 1] = [1];
        let _y = make_impl(
            unsafe { core::mem::transmute::<i8, ScalarType>(-1) },
            1,
            sizes.as_mut_ptr(),
            core::ptr::null_mut(),
            core::ptr::null_mut(),
            core::ptr::null_mut(),
        );
    }

    // [spec:et:sem:tensor.executorch.runtime.etensor.tensor.const-data-ptr-fn/test]
    // [spec:et:sem:tensor.executorch.runtime.etensor.tensor.set-data-fn/test]
    #[test]
    #[allow(deprecated)]
    fn tensor_test_set_data() {
        setup();
        let mut sizes: [SizesType; 1] = [5];
        let mut dim_order: [DimOrderType; 1] = [0];
        let mut data: [i32; 5] = [0, 0, 1, 0, 0];
        let mut a_impl = make_impl(
            ScalarType::Int,
            1,
            sizes.as_mut_ptr(),
            as_void(data.as_mut_ptr()),
            dim_order.as_mut_ptr(),
            core::ptr::null_mut(),
        );
        let a = Tensor::new(&raw mut a_impl);
        assert_eq!(a.const_data_ptr::<i32>(), data.as_ptr());
        a.set_data(core::ptr::null_mut());
        assert_eq!(a.const_data_ptr::<i32>(), core::ptr::null());
    }

    // PORT-NOTE: the C++ tensor_test.cpp does not directly exercise the thin
    // forwarders `dim/size/sizes/dim_order/shape_dynamism/dtype/data_ptr/
    // unsafe_get_tensor_impl`. Focused unit test pinning each against the
    // underlying TensorImpl per the sem rules (each forwarder just delegates).
    // [spec:et:sem:tensor.executorch.runtime.etensor.tensor.dim-fn/test]
    // [spec:et:sem:tensor.executorch.runtime.etensor.tensor.size-fn/test]
    // [spec:et:sem:tensor.executorch.runtime.etensor.tensor.sizes-fn/test]
    // [spec:et:sem:tensor.executorch.runtime.etensor.tensor.dim-order-fn/test]
    // [spec:et:sem:tensor.executorch.runtime.etensor.tensor.shape-dynamism-fn/test]
    // [spec:et:sem:tensor.executorch.runtime.etensor.tensor.dtype-fn/test]
    // [spec:et:sem:tensor.executorch.runtime.etensor.tensor.data-ptr-fn/test]
    // [spec:et:sem:tensor.executorch.runtime.etensor.tensor.unsafe-get-tensor-impl-fn/test]
    #[test]
    #[allow(deprecated)]
    fn tensor_test_forwarders() {
        setup();
        let mut sizes: [SizesType; 2] = [3, 2];
        let mut dim_order: [DimOrderType; 2] = [0, 1];
        let mut strides: [StridesType; 2] = [2, 1];
        let mut data: [i32; 6] = [1, 2, 3, 4, 5, 6];
        let mut a_impl = TensorImpl::new(
            ScalarType::Int,
            2,
            sizes.as_mut_ptr(),
            as_void(data.as_mut_ptr()),
            dim_order.as_mut_ptr(),
            strides.as_mut_ptr(),
            TensorShapeDynamism::DYNAMIC_BOUND,
            DeviceType::CPU,
            0,
        );
        let a = Tensor::new(&raw mut a_impl);

        // unsafe_get_tensor_impl returns the exact pointer passed to new().
        assert_eq!(a.unsafe_get_tensor_impl(), &raw mut a_impl);

        // Each forwarder must agree with the underlying TensorImpl.
        assert_eq!(a.dim(), a_impl.dim());
        assert_eq!(a.dim(), 2);
        assert_eq!(a.size(0), a_impl.size(0));
        assert_eq!(a.size(0), 3);
        assert_eq!(a.size(1), 2);
        assert_eq!(a.dtype(), a_impl.dtype());
        assert_eq!(a.dtype(), ScalarType::Int);
        assert_eq!(a.shape_dynamism(), a_impl.shape_dynamism());
        assert_eq!(a.shape_dynamism(), TensorShapeDynamism::DYNAMIC_BOUND);

        assert_eq!(a.sizes().data(), a_impl.sizes().data());
        assert_eq!(a.sizes().size(), 2);
        assert_eq!(a.dim_order().data(), a_impl.dim_order().data());
        assert_eq!(a.dim_order().size(), 2);

        // Deprecated data_ptr forwards to the mutable data pointer.
        assert_eq!(a.data_ptr::<i32>(), data.as_mut_ptr());
    }

    // [spec:et:sem:tensor.executorch.runtime.etensor.tensor.scalar-type-fn/test]
    // [spec:et:sem:tensor.executorch.runtime.etensor.tensor.const-data-ptr-fn/test]
    // [spec:et:sem:tensor.executorch.runtime.etensor.tensor.strides-fn/test]
    #[test]
    fn tensor_test_strides() {
        setup();
        let mut sizes: [SizesType; 2] = [2, 2];
        let mut dim_order: [DimOrderType; 2] = [0, 1];
        let mut data: [i32; 4] = [0, 0, 1, 1];
        let mut strides: [StridesType; 2] = [2, 1];
        let mut a_impl = make_impl(
            ScalarType::Int,
            2,
            sizes.as_mut_ptr(),
            as_void(data.as_mut_ptr()),
            dim_order.as_mut_ptr(),
            strides.as_mut_ptr(),
        );
        let a = Tensor::new(&raw mut a_impl);

        assert_eq!(a_impl.scalar_type(), ScalarType::Int);
        assert_eq!(a.scalar_type(), ScalarType::Int);
        assert_eq!(unsafe { *a.const_data_ptr::<i32>().add(0) }, 0);
        assert_eq!(
            unsafe {
                *a.const_data_ptr::<i32>()
                    .add((0 + *a.strides().index(0)) as usize)
            },
            1
        );
    }

    // [spec:et:sem:tensor.executorch.runtime.etensor.tensor.scalar-type-fn/test]
    // [spec:et:sem:tensor.executorch.runtime.etensor.tensor.const-data-ptr-fn/test]
    // [spec:et:sem:tensor.executorch.runtime.etensor.tensor.mutable-data-ptr-fn/test]
    #[test]
    fn tensor_test_modify_data_of_const_tensor() {
        setup();
        let mut sizes: [SizesType; 1] = [1];
        let mut dim_order: [DimOrderType; 2] = [0, 0];
        let mut data: [i32; 1] = [1];
        let mut a_impl = make_impl(
            ScalarType::Int,
            1,
            sizes.as_mut_ptr(),
            as_void(data.as_mut_ptr()),
            dim_order.as_mut_ptr(),
            core::ptr::null_mut(),
        );
        // C++ `const Tensor a(&a_impl)`: const on the handle still permits
        // mutating the (non-owned) impl's data through mutable_data_ptr.
        let a = Tensor::new(&raw mut a_impl);
        unsafe {
            *a.mutable_data_ptr::<i32>().add(0) = 0;
        }

        assert_eq!(a_impl.scalar_type(), ScalarType::Int);
        assert_eq!(a.scalar_type(), ScalarType::Int);
        assert_eq!(unsafe { *a.const_data_ptr::<i32>().add(0) }, 0);
    }

    // [spec:et:sem:tensor.executorch.runtime.etensor.tensor.device-type-fn/test]
    // [spec:et:sem:tensor.executorch.runtime.etensor.tensor.device-index-fn/test]
    // [spec:et:sem:tensor.executorch.runtime.etensor.tensor.device-fn/test]
    #[test]
    fn tensor_test_device_forwarders_default_cpu() {
        setup();
        let mut sizes: [SizesType; 1] = [1];
        let mut dim_order: [DimOrderType; 1] = [0];
        let mut data: [i32; 1] = [0];
        // TensorImpl ctor defaults device to CPU/0 when not specified.
        let mut a_impl = make_impl(
            ScalarType::Int,
            1,
            sizes.as_mut_ptr(),
            as_void(data.as_mut_ptr()),
            dim_order.as_mut_ptr(),
            core::ptr::null_mut(),
        );
        let a = Tensor::new(&raw mut a_impl);

        assert_eq!(a.device_type(), DeviceType::CPU);
        assert_eq!(a.device_index(), 0 as DeviceIndex);
        assert_eq!(a.device(), Device::new(DeviceType::CPU, 0));
    }

    // [spec:et:sem:tensor.executorch.runtime.etensor.tensor.device-type-fn/test]
    // [spec:et:sem:tensor.executorch.runtime.etensor.tensor.device-index-fn/test]
    // [spec:et:sem:tensor.executorch.runtime.etensor.tensor.device-fn/test]
    #[test]
    fn tensor_test_device_forwarders_non_cpu() {
        setup();
        let mut sizes: [SizesType; 1] = [1];
        let mut dim_order: [DimOrderType; 1] = [0];
        let mut data: [i32; 1] = [0];
        let mut a_impl = TensorImpl::new(
            ScalarType::Int,
            1,
            sizes.as_mut_ptr(),
            as_void(data.as_mut_ptr()),
            dim_order.as_mut_ptr(),
            /*strides=*/ core::ptr::null_mut(),
            TensorShapeDynamism::STATIC,
            DeviceType::CUDA,
            /*device_index=*/ 3,
        );
        let a = Tensor::new(&raw mut a_impl);

        // Each forwarder must agree with the underlying TensorImpl.
        assert_eq!(a.device_type(), a_impl.device_type());
        assert_eq!(a.device_index(), a_impl.device_index());
        assert_eq!(a.device(), a_impl.device());

        assert_eq!(a.device_type(), DeviceType::CUDA);
        assert_eq!(a.device_index(), 3 as DeviceIndex);
    }
}
