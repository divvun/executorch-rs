# kernels/portable/cpu/op__device_copy.cpp

> [spec:et:def:op-device-copy.torch.executor.native.d2h-copy-out-fn]
> Tensor&

> [spec:et:sem:op-device-copy.torch.executor.native.d2h-copy-out-fn]
> Implements `_d2h_copy_out(ctx, self, out)`: copies tensor data from device
> memory (`self`) into host/CPU memory (`out`). Device type and index come from
> `self`'s TensorImpl metadata: `device_type = self.unsafeGetTensorImpl()->
> device_type()`, `device_index = self.unsafeGetTensorImpl()->device_index()`.
>
> Validation, in order (each `ET_KERNEL_CHECK_MSG`; on failure sets the stated
> context Error and returns `out` unmodified):
> 1. `device_type != DeviceType::CPU` → `InvalidArgument` ("source tensor must
>    be on a non-CPU device"). Source must be on a device.
> 2. `out.unsafeGetTensorImpl()->device_type() == DeviceType::CPU` →
>    `InvalidArgument` ("destination tensor must be on CPU"). Destination must be
>    CPU.
> 3. `resize_tensor(out, self.sizes()) == Error::Ok` → `InvalidArgument`
>    (message notes `self.nbytes()` may exceed `out`'s planned capacity). Then
>    `nbytes = self.nbytes()`.
> 4. `allocator = get_device_allocator(device_type)` must be non-null, else
>    `NotFound` ("no device allocator registered").
>
> Then calls `allocator->copy_device_to_host(out.mutable_data_ptr(),
> self.const_data_ptr(), nbytes, device_index)`; if the returned `Error` is not
> `Error::Ok`, sets context Error to `Internal` ("copy_device_to_host failed")
> and returns `out`. On success returns `out`. No dtype dispatch — raw byte copy
> of `nbytes` bytes.

> [spec:et:def:op-device-copy.torch.executor.native.h2d-copy-out-fn]
> Tensor&

> [spec:et:sem:op-device-copy.torch.executor.native.h2d-copy-out-fn]
> Implements `_h2d_copy_out(ctx, self, out)`: copies tensor data from host/CPU
> memory (`self`) into device memory (`out`). Device type and index come from
> `out`'s TensorImpl metadata: `device_type = out.unsafeGetTensorImpl()->
> device_type()`, `device_index = out.unsafeGetTensorImpl()->device_index()`.
>
> Validation, in order (each `ET_KERNEL_CHECK_MSG`; on failure sets the stated
> context Error and returns `out` unmodified):
> 1. `self.unsafeGetTensorImpl()->device_type() == DeviceType::CPU` →
>    `InvalidArgument` ("source tensor must be on CPU"). Source must be CPU.
> 2. `device_type != DeviceType::CPU` → `InvalidArgument` ("destination tensor
>    must be on a non-CPU device"). Destination must be on a device.
> 3. `resize_tensor(out, self.sizes()) == Error::Ok` → `InvalidArgument`
>    (message notes `self.nbytes()` may exceed `out`'s planned capacity). Then
>    `nbytes = self.nbytes()`.
> 4. `allocator = get_device_allocator(device_type)` must be non-null, else
>    `NotFound` ("no device allocator registered").
>
> Then calls `allocator->copy_host_to_device(out.mutable_data_ptr(),
> self.const_data_ptr(), nbytes, device_index)`; if the returned `Error` is not
> `Error::Ok`, sets context Error to `Internal` ("copy_host_to_device failed")
> and returns `out`. On success returns `out`. No dtype dispatch — raw byte copy
> of `nbytes` bytes.

