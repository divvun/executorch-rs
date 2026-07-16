//! Literal port of kernels/portable/cpu/op__device_copy.cpp.
//!
//! Runtime kernels for et_copy._h2d_copy and et_copy._d2h_copy ops.
//!
//! These ops transfer tensor data between CPU and device memory using
//! the DeviceAllocator interface. The device type is inferred from the
//! tensor metadata (out.device_type() for H2D, self.device_type() for D2H),
//! which was set during AOT serialization by PropagateDevicePass.

use crate::runtime::core::device_allocator::get_device_allocator;
use crate::runtime::core::error::Error;
use crate::runtime::core::exec_aten::util::tensor_util::resize_tensor;
use crate::runtime::core::portable_type::device::DeviceType;
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

/// Copies tensor data from host (CPU) memory to device memory.
///
/// self: source tensor on CPU
/// out:  destination tensor on device (memory-planned by runtime)
///
/// The device type and index are inferred from out's TensorImpl metadata.
// [spec:et:def:op-device-copy.torch.executor.native.h2d-copy-out-fn]
// [spec:et:sem:op-device-copy.torch.executor.native.h2d-copy-out-fn]
pub fn _h2d_copy_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    self_: &Tensor,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    let device_type = unsafe { (*out.unsafe_get_tensor_impl()).device_type() };
    let device_index = unsafe { (*out.unsafe_get_tensor_impl()).device_index() };

    crate::et_kernel_check_msg!(
        ctx,
        unsafe { (*self_.unsafe_get_tensor_impl()).device_type() } == DeviceType::CPU,
        InvalidArgument,
        out,
        "_h2d_copy: source tensor must be on CPU"
    );

    crate::et_kernel_check_msg!(
        ctx,
        device_type != DeviceType::CPU,
        InvalidArgument,
        out,
        "_h2d_copy: destination tensor must be on a non-CPU device"
    );

    crate::et_kernel_check_msg!(
        ctx,
        resize_tensor(out, self_.sizes()) == Error::Ok,
        InvalidArgument,
        out,
        "_h2d_copy: cannot resize out to self sizes (self.nbytes() exceeds out planned capacity?)"
    );
    let nbytes = self_.nbytes();

    let allocator = get_device_allocator(device_type);
    crate::et_kernel_check_msg!(
        ctx,
        !allocator.is_null(),
        NotFound,
        out,
        "_h2d_copy: no device allocator registered for device_type"
    );

    let err = unsafe {
        (*allocator).copy_host_to_device(
            out.mutable_data_ptr_typed(),
            self_.const_data_ptr_typed(),
            nbytes,
            device_index,
        )
    };
    crate::et_kernel_check_msg!(
        ctx,
        err == Error::Ok,
        Internal,
        out,
        "_h2d_copy: copy_host_to_device failed"
    );

    out
}

/// Copies tensor data from device memory to host (CPU) memory.
///
/// self: source tensor on device
/// out:  destination tensor on CPU (memory-planned by runtime)
///
/// The device type and index are inferred from self's TensorImpl metadata.
// [spec:et:def:op-device-copy.torch.executor.native.d2h-copy-out-fn]
// [spec:et:sem:op-device-copy.torch.executor.native.d2h-copy-out-fn]
pub fn _d2h_copy_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    self_: &Tensor,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    let device_type = unsafe { (*self_.unsafe_get_tensor_impl()).device_type() };
    let device_index = unsafe { (*self_.unsafe_get_tensor_impl()).device_index() };

    crate::et_kernel_check_msg!(
        ctx,
        device_type != DeviceType::CPU,
        InvalidArgument,
        out,
        "_d2h_copy: source tensor must be on a non-CPU device"
    );

    crate::et_kernel_check_msg!(
        ctx,
        unsafe { (*out.unsafe_get_tensor_impl()).device_type() } == DeviceType::CPU,
        InvalidArgument,
        out,
        "_d2h_copy: destination tensor must be on CPU"
    );

    crate::et_kernel_check_msg!(
        ctx,
        resize_tensor(out, self_.sizes()) == Error::Ok,
        InvalidArgument,
        out,
        "_d2h_copy: cannot resize out to self sizes (self.nbytes() exceeds out planned capacity?)"
    );
    let nbytes = self_.nbytes();

    let allocator = get_device_allocator(device_type);
    crate::et_kernel_check_msg!(
        ctx,
        !allocator.is_null(),
        NotFound,
        out,
        "_d2h_copy: no device allocator registered for device_type"
    );

    let err = unsafe {
        (*allocator).copy_device_to_host(
            out.mutable_data_ptr_typed(),
            self_.const_data_ptr_typed(),
            nbytes,
            device_index,
        )
    };
    crate::et_kernel_check_msg!(
        ctx,
        err == Error::Ok,
        Internal,
        out,
        "_d2h_copy: copy_device_to_host failed"
    );

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::core::device_allocator::{DeviceAllocator, DeviceAllocatorRegistry};
    use crate::runtime::core::error::Result;
    use crate::runtime::core::memory_allocator::{MemoryAllocator, MemoryAllocatorBase};
    use crate::runtime::core::portable_type::device::{DeviceIndex, DeviceType};
    use crate::runtime::core::portable_type::scalar_type::ScalarType;
    use crate::runtime::core::portable_type::tensor_impl::TensorImpl;
    use crate::runtime::core::tensor_shape_dynamism::TensorShapeDynamism;

    // PORT-NOTE: port of runtime/core/test/mock_cuda_allocator.h. Uses host
    // memory (memcpy) to simulate device memory and tracks copy calls. gtest runs
    // this fixture's cases serially with a single registered allocator and a
    // per-test counter reset; `cargo test` runs tests in parallel, so the whole
    // suite is serialized through `SUITE_LOCK` and each test resets the shared
    // counters while holding it, preserving the C++ SetUp/SetUpTestSuite contract.
    struct MockCudaAllocator {
        h2d_count: i32,
        d2h_count: i32,
        last_h2d_size: usize,
        last_d2h_size: usize,
        last_h2d_index: DeviceIndex,
        last_d2h_index: DeviceIndex,
    }

    impl MockCudaAllocator {
        const fn new() -> Self {
            MockCudaAllocator {
                h2d_count: 0,
                d2h_count: 0,
                last_h2d_size: 0,
                last_d2h_size: 0,
                last_h2d_index: -1,
                last_d2h_index: -1,
            }
        }
        fn reset(&mut self) {
            self.h2d_count = 0;
            self.d2h_count = 0;
            self.last_h2d_size = 0;
            self.last_d2h_size = 0;
            self.last_h2d_index = -1;
            self.last_d2h_index = -1;
        }
    }

    impl DeviceAllocator for MockCudaAllocator {
        fn allocate(
            &mut self,
            nbytes: usize,
            _index: DeviceIndex,
            _alignment: usize,
        ) -> Result<*mut core::ffi::c_void> {
            // Not exercised by these tests, but mirror the mock's malloc path.
            let mut v = vec![0u8; nbytes];
            let ptr = v.as_mut_ptr() as *mut core::ffi::c_void;
            core::mem::forget(v);
            Ok(ptr)
        }
        fn deallocate(&mut self, _ptr: *mut core::ffi::c_void, _index: DeviceIndex) {}
        fn copy_host_to_device(
            &mut self,
            dst: *mut core::ffi::c_void,
            src: *const core::ffi::c_void,
            nbytes: usize,
            index: DeviceIndex,
        ) -> Error {
            unsafe {
                core::ptr::copy_nonoverlapping(src as *const u8, dst as *mut u8, nbytes);
            }
            self.h2d_count += 1;
            self.last_h2d_size = nbytes;
            self.last_h2d_index = index;
            Error::Ok
        }
        fn copy_device_to_host(
            &mut self,
            dst: *mut core::ffi::c_void,
            src: *const core::ffi::c_void,
            nbytes: usize,
            index: DeviceIndex,
        ) -> Error {
            unsafe {
                core::ptr::copy_nonoverlapping(src as *const u8, dst as *mut u8, nbytes);
            }
            self.d2h_count += 1;
            self.last_d2h_size = nbytes;
            self.last_d2h_index = index;
            Error::Ok
        }
        fn device_type(&self) -> DeviceType {
            DeviceType::CUDA
        }
    }

    static mut G_MOCK_CUDA: MockCudaAllocator = MockCudaAllocator::new();

    // PORT-NOTE (cross-suite registry serialization): this suite originally used a
    // private `SUITE_LOCK` + `Once` register-if-empty scheme, which assumed its
    // mock stayed in the process-wide CUDA slot for the whole run. Other
    // registry-dependent suites (device_allocator, tensor_ptr's device_tests)
    // clear + re-register the same slot under the shared `DEVICE_REGISTRY_TEST_LOCK`,
    // which would evict this mock and make these tests dispatch to a foreign mock
    // (counter assertions then fail). Switched to the shared lock and a
    // clear+register on every `setup()` so all registry-mutating suites serialize
    // and each installs its own mock atomically, matching the pattern in
    // device_allocator.rs.
    fn setup() -> std::sync::MutexGuard<'static, ()> {
        crate::runtime::platform::platform::pal_init();
        let ptr =
            &raw mut G_MOCK_CUDA as *mut MockCudaAllocator as *mut (dyn DeviceAllocator + 'static);
        let guard = DeviceAllocatorRegistry::install_for_test(ptr);
        unsafe {
            (*(&raw mut G_MOCK_CUDA)).reset();
        }
        guard
    }

    fn context() -> KernelRuntimeContext<'static> {
        KernelRuntimeContext::new(
            crate::extension::module::module::null_event_tracer(),
            core::ptr::null_mut::<MemoryAllocator>() as *mut dyn MemoryAllocatorBase,
        )
    }

    fn mock() -> &'static mut MockCudaAllocator {
        unsafe { &mut *(&raw mut G_MOCK_CUDA) }
    }

    #[allow(clippy::too_many_arguments)]
    fn make_impl(
        dim: isize,
        sizes: *mut i32,
        data: *mut f32,
        dim_order: *mut u8,
        strides: *mut i32,
        dynamism: TensorShapeDynamism,
        device_type: DeviceType,
        device_index: DeviceIndex,
    ) -> TensorImpl {
        TensorImpl::new(
            ScalarType::Float,
            dim,
            sizes,
            data as *mut core::ffi::c_void,
            dim_order,
            strides,
            dynamism,
            device_type,
            device_index,
        )
    }

    // [spec:et:sem:op-device-copy.torch.executor.native.h2d-copy-out-fn/test]
    #[test]
    fn op_device_copy_test_h2d_copy_copies_data_and_calls_allocator() {
        let _guard = setup();

        let mut src_data: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
        let mut sizes: [i32; 1] = [4];
        let mut dim_order: [u8; 1] = [0];
        let mut strides: [i32; 1] = [1];
        let mut src_impl = make_impl(
            1,
            sizes.as_mut_ptr(),
            src_data.as_mut_ptr(),
            dim_order.as_mut_ptr(),
            strides.as_mut_ptr(),
            TensorShapeDynamism::STATIC,
            DeviceType::CPU,
            0,
        );
        let src = Tensor::new(&mut src_impl as *mut TensorImpl);

        let mut dst_data: [f32; 4] = [0.0; 4];
        let mut dst_impl = make_impl(
            1,
            sizes.as_mut_ptr(),
            dst_data.as_mut_ptr(),
            dim_order.as_mut_ptr(),
            strides.as_mut_ptr(),
            TensorShapeDynamism::STATIC,
            DeviceType::CUDA,
            0,
        );
        let dst = Tensor::new(&mut dst_impl as *mut TensorImpl);

        let mut ctx = context();
        let result = _h2d_copy_out(&mut ctx, &src, &dst);

        assert_eq!(mock().h2d_count, 1);
        assert_eq!(mock().last_h2d_size, 4 * core::mem::size_of::<f32>());
        assert_eq!(mock().last_h2d_index, 0);

        assert_eq!(dst_data[0], 1.0);
        assert_eq!(dst_data[1], 2.0);
        assert_eq!(dst_data[2], 3.0);
        assert_eq!(dst_data[3], 4.0);

        assert_eq!(
            result.unsafe_get_tensor_impl(),
            dst.unsafe_get_tensor_impl()
        );
    }

    // [spec:et:sem:op-device-copy.torch.executor.native.d2h-copy-out-fn/test]
    #[test]
    fn op_device_copy_test_d2h_copy_copies_data_and_calls_allocator() {
        let _guard = setup();

        let mut src_data: [f32; 4] = [5.0, 6.0, 7.0, 8.0];
        let mut sizes: [i32; 1] = [4];
        let mut dim_order: [u8; 1] = [0];
        let mut strides: [i32; 1] = [1];
        let mut src_impl = make_impl(
            1,
            sizes.as_mut_ptr(),
            src_data.as_mut_ptr(),
            dim_order.as_mut_ptr(),
            strides.as_mut_ptr(),
            TensorShapeDynamism::STATIC,
            DeviceType::CUDA,
            0,
        );
        let src = Tensor::new(&mut src_impl as *mut TensorImpl);

        let mut dst_data: [f32; 4] = [0.0; 4];
        let mut dst_impl = make_impl(
            1,
            sizes.as_mut_ptr(),
            dst_data.as_mut_ptr(),
            dim_order.as_mut_ptr(),
            strides.as_mut_ptr(),
            TensorShapeDynamism::STATIC,
            DeviceType::CPU,
            0,
        );
        let dst = Tensor::new(&mut dst_impl as *mut TensorImpl);

        let mut ctx = context();
        let result = _d2h_copy_out(&mut ctx, &src, &dst);

        assert_eq!(mock().d2h_count, 1);
        assert_eq!(mock().last_d2h_size, 4 * core::mem::size_of::<f32>());
        assert_eq!(mock().last_d2h_index, 0);

        assert_eq!(dst_data[0], 5.0);
        assert_eq!(dst_data[1], 6.0);
        assert_eq!(dst_data[2], 7.0);
        assert_eq!(dst_data[3], 8.0);

        assert_eq!(
            result.unsafe_get_tensor_impl(),
            dst.unsafe_get_tensor_impl()
        );
    }

    // [spec:et:sem:op-device-copy.torch.executor.native.h2d-copy-out-fn/test]
    #[test]
    fn op_device_copy_test_h2d_copy_with_device_index1() {
        let _guard = setup();

        let mut src_data: [f32; 1] = [1.0];
        let mut dst_data: [f32; 1] = [0.0];
        let mut sizes: [i32; 1] = [1];
        let mut dim_order: [u8; 1] = [0];
        let mut strides: [i32; 1] = [1];

        let mut src_impl = make_impl(
            1,
            sizes.as_mut_ptr(),
            src_data.as_mut_ptr(),
            dim_order.as_mut_ptr(),
            strides.as_mut_ptr(),
            TensorShapeDynamism::STATIC,
            DeviceType::CPU,
            0,
        );
        let src = Tensor::new(&mut src_impl as *mut TensorImpl);

        let mut dst_impl = make_impl(
            1,
            sizes.as_mut_ptr(),
            dst_data.as_mut_ptr(),
            dim_order.as_mut_ptr(),
            strides.as_mut_ptr(),
            TensorShapeDynamism::STATIC,
            DeviceType::CUDA,
            1,
        );
        let dst = Tensor::new(&mut dst_impl as *mut TensorImpl);

        let mut ctx = context();
        _h2d_copy_out(&mut ctx, &src, &dst);

        assert_eq!(mock().h2d_count, 1);
        assert_eq!(mock().last_h2d_index, 1);
    }

    // [spec:et:sem:op-device-copy.torch.executor.native.h2d-copy-out-fn/test]
    #[test]
    fn op_device_copy_test_h2d_copy_multidimensional_tensor() {
        let _guard = setup();

        let mut src_data: [f32; 6] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut dst_data: [f32; 6] = [0.0; 6];
        let mut sizes: [i32; 2] = [2, 3];
        let mut dim_order: [u8; 2] = [0, 1];
        let mut strides: [i32; 2] = [3, 1];

        let mut src_impl = make_impl(
            2,
            sizes.as_mut_ptr(),
            src_data.as_mut_ptr(),
            dim_order.as_mut_ptr(),
            strides.as_mut_ptr(),
            TensorShapeDynamism::STATIC,
            DeviceType::CPU,
            0,
        );
        let src = Tensor::new(&mut src_impl as *mut TensorImpl);

        let mut dst_impl = make_impl(
            2,
            sizes.as_mut_ptr(),
            dst_data.as_mut_ptr(),
            dim_order.as_mut_ptr(),
            strides.as_mut_ptr(),
            TensorShapeDynamism::STATIC,
            DeviceType::CUDA,
            0,
        );
        let dst = Tensor::new(&mut dst_impl as *mut TensorImpl);

        let mut ctx = context();
        _h2d_copy_out(&mut ctx, &src, &dst);

        assert_eq!(mock().h2d_count, 1);
        assert_eq!(mock().last_h2d_size, 6 * core::mem::size_of::<f32>());

        for i in 0..6 {
            assert_eq!(dst_data[i], src_data[i]);
        }
    }

    // [spec:et:sem:op-device-copy.torch.executor.native.h2d-copy-out-fn/test]
    #[test]
    fn op_device_copy_test_h2d_copy_dynamic_shape_resizes_out_down_to_input() {
        let _guard = setup();

        let mut src_data: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
        let mut src_sizes: [i32; 1] = [4];
        let mut src_dim_order: [u8; 1] = [0];
        let mut src_strides: [i32; 1] = [1];
        let mut src_impl = make_impl(
            1,
            src_sizes.as_mut_ptr(),
            src_data.as_mut_ptr(),
            src_dim_order.as_mut_ptr(),
            src_strides.as_mut_ptr(),
            TensorShapeDynamism::STATIC,
            DeviceType::CPU,
            0,
        );
        let src = Tensor::new(&mut src_impl as *mut TensorImpl);

        let mut dst_data: [f32; 8] = [0.0; 8];
        let mut dst_sizes: [i32; 1] = [8];
        let mut dst_dim_order: [u8; 1] = [0];
        let mut dst_strides: [i32; 1] = [1];
        let mut dst_impl = make_impl(
            1,
            dst_sizes.as_mut_ptr(),
            dst_data.as_mut_ptr(),
            dst_dim_order.as_mut_ptr(),
            dst_strides.as_mut_ptr(),
            TensorShapeDynamism::DYNAMIC_BOUND,
            DeviceType::CUDA,
            0,
        );
        let dst = Tensor::new(&mut dst_impl as *mut TensorImpl);

        let mut ctx = context();
        let result = _h2d_copy_out(&mut ctx, &src, &dst);

        assert_eq!(dst.dim(), 1);
        assert_eq!(dst.size(0), 4);
        assert_eq!(dst.numel(), 4);

        assert_eq!(mock().h2d_count, 1);
        assert_eq!(mock().last_h2d_size, 4 * core::mem::size_of::<f32>());

        assert_eq!(dst_data[0], 1.0);
        assert_eq!(dst_data[1], 2.0);
        assert_eq!(dst_data[2], 3.0);
        assert_eq!(dst_data[3], 4.0);

        assert_eq!(
            result.unsafe_get_tensor_impl(),
            dst.unsafe_get_tensor_impl()
        );
    }

    // [spec:et:sem:op-device-copy.torch.executor.native.d2h-copy-out-fn/test]
    #[test]
    fn op_device_copy_test_d2h_copy_dynamic_shape_resizes_out_down_to_input() {
        let _guard = setup();

        let mut src_data: [f32; 4] = [5.0, 6.0, 7.0, 8.0];
        let mut src_sizes: [i32; 1] = [4];
        let mut src_dim_order: [u8; 1] = [0];
        let mut src_strides: [i32; 1] = [1];
        let mut src_impl = make_impl(
            1,
            src_sizes.as_mut_ptr(),
            src_data.as_mut_ptr(),
            src_dim_order.as_mut_ptr(),
            src_strides.as_mut_ptr(),
            TensorShapeDynamism::STATIC,
            DeviceType::CUDA,
            0,
        );
        let src = Tensor::new(&mut src_impl as *mut TensorImpl);

        let mut dst_data: [f32; 8] = [0.0; 8];
        let mut dst_sizes: [i32; 1] = [8];
        let mut dst_dim_order: [u8; 1] = [0];
        let mut dst_strides: [i32; 1] = [1];
        let mut dst_impl = make_impl(
            1,
            dst_sizes.as_mut_ptr(),
            dst_data.as_mut_ptr(),
            dst_dim_order.as_mut_ptr(),
            dst_strides.as_mut_ptr(),
            TensorShapeDynamism::DYNAMIC_BOUND,
            DeviceType::CPU,
            0,
        );
        let dst = Tensor::new(&mut dst_impl as *mut TensorImpl);

        let mut ctx = context();
        let result = _d2h_copy_out(&mut ctx, &src, &dst);

        assert_eq!(dst.dim(), 1);
        assert_eq!(dst.size(0), 4);
        assert_eq!(dst.numel(), 4);

        assert_eq!(mock().d2h_count, 1);
        assert_eq!(mock().last_d2h_size, 4 * core::mem::size_of::<f32>());

        assert_eq!(dst_data[0], 5.0);
        assert_eq!(dst_data[1], 6.0);
        assert_eq!(dst_data[2], 7.0);
        assert_eq!(dst_data[3], 8.0);

        assert_eq!(
            result.unsafe_get_tensor_impl(),
            dst.unsafe_get_tensor_impl()
        );
    }

    // [spec:et:sem:op-device-copy.torch.executor.native.h2d-copy-out-fn/test]
    #[test]
    fn op_device_copy_test_h2d_copy_fails_when_input_exceeds_out_capacity() {
        let _guard = setup();

        let mut src_data: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
        let mut src_sizes: [i32; 1] = [4];
        let mut src_dim_order: [u8; 1] = [0];
        let mut src_strides: [i32; 1] = [1];
        let mut src_impl = make_impl(
            1,
            src_sizes.as_mut_ptr(),
            src_data.as_mut_ptr(),
            src_dim_order.as_mut_ptr(),
            src_strides.as_mut_ptr(),
            TensorShapeDynamism::STATIC,
            DeviceType::CPU,
            0,
        );
        let src = Tensor::new(&mut src_impl as *mut TensorImpl);

        let mut dst_data: [f32; 2] = [0.0; 2];
        let mut dst_sizes: [i32; 1] = [2];
        let mut dst_dim_order: [u8; 1] = [0];
        let mut dst_strides: [i32; 1] = [1];
        let mut dst_impl = make_impl(
            1,
            dst_sizes.as_mut_ptr(),
            dst_data.as_mut_ptr(),
            dst_dim_order.as_mut_ptr(),
            dst_strides.as_mut_ptr(),
            TensorShapeDynamism::DYNAMIC_BOUND,
            DeviceType::CUDA,
            0,
        );
        let dst = Tensor::new(&mut dst_impl as *mut TensorImpl);

        let mut ctx = context();
        _h2d_copy_out(&mut ctx, &src, &dst);

        assert_eq!(ctx.failure_state(), Error::InvalidArgument);
        // The kernel bailed before copying.
        assert_eq!(mock().h2d_count, 0);
    }

    // [spec:et:sem:op-device-copy.torch.executor.native.d2h-copy-out-fn/test]
    #[test]
    fn op_device_copy_test_d2h_copy_fails_when_input_exceeds_out_capacity() {
        let _guard = setup();

        let mut src_data: [f32; 4] = [5.0, 6.0, 7.0, 8.0];
        let mut src_sizes: [i32; 1] = [4];
        let mut src_dim_order: [u8; 1] = [0];
        let mut src_strides: [i32; 1] = [1];
        let mut src_impl = make_impl(
            1,
            src_sizes.as_mut_ptr(),
            src_data.as_mut_ptr(),
            src_dim_order.as_mut_ptr(),
            src_strides.as_mut_ptr(),
            TensorShapeDynamism::STATIC,
            DeviceType::CUDA,
            0,
        );
        let src = Tensor::new(&mut src_impl as *mut TensorImpl);

        let mut dst_data: [f32; 2] = [0.0; 2];
        let mut dst_sizes: [i32; 1] = [2];
        let mut dst_dim_order: [u8; 1] = [0];
        let mut dst_strides: [i32; 1] = [1];
        let mut dst_impl = make_impl(
            1,
            dst_sizes.as_mut_ptr(),
            dst_data.as_mut_ptr(),
            dst_dim_order.as_mut_ptr(),
            dst_strides.as_mut_ptr(),
            TensorShapeDynamism::DYNAMIC_BOUND,
            DeviceType::CPU,
            0,
        );
        let dst = Tensor::new(&mut dst_impl as *mut TensorImpl);

        let mut ctx = context();
        _d2h_copy_out(&mut ctx, &src, &dst);

        assert_eq!(ctx.failure_state(), Error::InvalidArgument);
        assert_eq!(mock().d2h_count, 0);
    }

    // [spec:et:sem:op-device-copy.torch.executor.native.h2d-copy-out-fn/test]
    #[test]
    fn op_device_copy_test_h2d_copy_dynamic_bound_equal_size_still_copies() {
        let _guard = setup();

        let mut src_data: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
        let mut sizes: [i32; 1] = [4];
        let mut dim_order: [u8; 1] = [0];
        let mut strides: [i32; 1] = [1];
        let mut src_impl = make_impl(
            1,
            sizes.as_mut_ptr(),
            src_data.as_mut_ptr(),
            dim_order.as_mut_ptr(),
            strides.as_mut_ptr(),
            TensorShapeDynamism::STATIC,
            DeviceType::CPU,
            0,
        );
        let src = Tensor::new(&mut src_impl as *mut TensorImpl);

        let mut dst_data: [f32; 4] = [0.0; 4];
        let mut dst_impl = make_impl(
            1,
            sizes.as_mut_ptr(),
            dst_data.as_mut_ptr(),
            dim_order.as_mut_ptr(),
            strides.as_mut_ptr(),
            TensorShapeDynamism::DYNAMIC_BOUND,
            DeviceType::CUDA,
            0,
        );
        let dst = Tensor::new(&mut dst_impl as *mut TensorImpl);

        let mut ctx = context();
        _h2d_copy_out(&mut ctx, &src, &dst);

        assert_eq!(dst.size(0), 4);
        assert_eq!(mock().h2d_count, 1);
        assert_eq!(mock().last_h2d_size, 4 * core::mem::size_of::<f32>());
        for i in 0..4 {
            assert_eq!(dst_data[i], src_data[i]);
        }
    }
}
