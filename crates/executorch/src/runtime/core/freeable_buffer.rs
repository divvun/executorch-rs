//! Literal port of runtime/core/freeable_buffer.h.

use crate::runtime::core::error::Error;
use crate::runtime::core::result::Result;

// Callback signature for the function that does the freeing.
pub type FreeFn = unsafe extern "C" fn(
    context: *mut core::ffi::c_void,
    data: *mut core::ffi::c_void,
    size: usize,
);
pub type FreeUInt64Fn =
    unsafe extern "C" fn(context: *mut core::ffi::c_void, data_uint64: u64, size: usize);

// [spec:et:def:freeable-buffer.executorch.runtime.freeable-buffer.pointer-data]
#[derive(Clone, Copy)]
struct PointerData {
    data_: *const core::ffi::c_void,
    free_fn_: Option<FreeFn>,
}

// [spec:et:def:freeable-buffer.executorch.runtime.freeable-buffer.u-int64-data]
#[derive(Clone, Copy)]
struct UInt64Data {
    // A pointer value cast to uint64_t.
    data_: u64,
    free_fn_: Option<FreeUInt64Fn>,
}

/// The `std::variant<PointerData, UInt64Data>` alternatives.
#[derive(Clone, Copy)]
enum Data {
    Pointer(PointerData),
    UInt64(UInt64Data),
}

/// A read-only buffer than can be freed.
// [spec:et:def:freeable-buffer.executorch.runtime.freeable-buffer]
pub struct FreeableBuffer {
    // This stores either a PointerData or a UInt64Data structure. Most users
    // should use the PointerData variant and the void* ctor.
    data_: Data,
    free_fn_context_: *mut core::ffi::c_void,
    size_: usize,
}

impl FreeableBuffer {
    /// Creates an empty FreeableBuffer with size zero and a null data pointer.
    pub fn new() -> Self {
        FreeableBuffer {
            data_: Data::Pointer(PointerData {
                data_: core::ptr::null(),
                free_fn_: None,
            }),
            free_fn_context_: core::ptr::null_mut(),
            size_: 0,
        }
    }

    /// Creates a FreeableBuffer with an optional free function.
    ///
    /// `free_fn` is guaranteed to be called exactly once before the
    /// FreeableBuffer is destroyed. NOTE: This function must be thread-safe.
    pub fn from_pointer(
        data: *const core::ffi::c_void,
        size: usize,
        free_fn: Option<FreeFn>,
        free_fn_context: *mut core::ffi::c_void,
    ) -> Self {
        FreeableBuffer {
            data_: Data::Pointer(PointerData {
                data_: data,
                free_fn_: free_fn,
            }),
            free_fn_context_: free_fn_context,
            size_: size,
        }
    }

    /// Creates a FreeableBuffer with an optional free function.
    ///
    /// NOTE: most users should use the other ctor with FreeFn. This variant
    /// exists for situations where the FreeableBuffer points to memory on a
    /// different core whose pointer value is larger than the local core's void*.
    pub fn from_uint64(
        data_uint64: u64,
        size: usize,
        free_fn: Option<FreeUInt64Fn>,
        free_fn_context: *mut core::ffi::c_void,
    ) -> Self {
        FreeableBuffer {
            data_: Data::UInt64(UInt64Data {
                data_: data_uint64,
                free_fn_: free_fn,
            }),
            free_fn_context_: free_fn_context,
            size_: size,
        }
    }

    /// Move ctor. Takes the ownership of the data previously owned by `rhs`,
    /// leaving `rhs` pointing to nullptr.
    // [spec:et:def:freeable-buffer.executorch.runtime.freeable-buffer.freeable-buffer-fn]
    // [spec:et:sem:freeable-buffer.executorch.runtime.freeable-buffer.freeable-buffer-fn]
    //
    // PORT-NOTE: Rust moves are destructive; a value moved out of `rhs` cannot
    // be dropped again. To match the C++ contract byte-for-byte (rhs reset to a
    // null-but-same-variant, freed state) this takes `&mut rhs` and manually
    // resets it, so a Rust move-plus-later-Drop of `rhs` is a genuine no-op.
    pub fn from_move(rhs: &mut FreeableBuffer) -> Self {
        let new = FreeableBuffer {
            data_: rhs.data_,
            free_fn_context_: rhs.free_fn_context_,
            size_: rhs.size_,
        };
        match rhs.data_ {
            Data::Pointer(_) => {
                rhs.data_ = Data::Pointer(PointerData {
                    data_: core::ptr::null(),
                    free_fn_: None,
                });
            }
            Data::UInt64(_) => {
                rhs.data_ = Data::UInt64(UInt64Data {
                    data_: 0,
                    free_fn_: None,
                });
            }
        }
        rhs.free_fn_context_ = core::ptr::null_mut();
        rhs.size_ = 0;
        new
    }

    /// Frees the data if not already free. Safe to call multiple times.
    // [spec:et:def:freeable-buffer.executorch.runtime.freeable-buffer.free-fn]
    // [spec:et:sem:freeable-buffer.executorch.runtime.freeable-buffer.free-fn]
    pub fn free(&mut self) {
        match &mut self.data_ {
            Data::Pointer(ptr_data) => {
                if !ptr_data.data_.is_null() && ptr_data.free_fn_.is_some() {
                    // Do not need to check for truncation here, as free_fn_ is
                    // only set using the void* ctor.
                    unsafe {
                        (ptr_data.free_fn_.unwrap())(
                            self.free_fn_context_,
                            ptr_data.data_ as *mut core::ffi::c_void,
                            self.size_,
                        );
                    }
                }
                ptr_data.data_ = core::ptr::null();
                self.size_ = 0;
            }
            Data::UInt64(int64_data) => {
                if int64_data.data_ != 0 && int64_data.free_fn_.is_some() {
                    unsafe {
                        (int64_data.free_fn_.unwrap())(
                            self.free_fn_context_,
                            int64_data.data_,
                            self.size_,
                        );
                    }
                }
                int64_data.data_ = 0u64;
                self.size_ = 0;
            }
        }
    }

    /// Size of the data in bytes. Returns 0 if the data has been freed.
    // [spec:et:def:freeable-buffer.executorch.runtime.freeable-buffer.size-fn]
    // [spec:et:sem:freeable-buffer.executorch.runtime.freeable-buffer.size-fn]
    pub fn size(&self) -> usize {
        self.size_
    }

    /// Pointer to the data. Returns nullptr if the data has been freed.
    // [spec:et:def:freeable-buffer.executorch.runtime.freeable-buffer.data-fn]
    // [spec:et:sem:freeable-buffer.executorch.runtime.freeable-buffer.data-fn]
    pub fn data(&self) -> *const core::ffi::c_void {
        // PORT-NOTE: C++ `ET_CHECK_MSG(holds_alternative<PointerData>, ...)`
        // fatally aborts if backed by a uint64_t; `assert!` stands in for the
        // platform assert group's fatal check (still a stub).
        match &self.data_ {
            Data::Pointer(ptr_data) => ptr_data.data_,
            Data::UInt64(_) => {
                assert!(
                    false,
                    "FreeableBuffer is backed by an uint64_t, please use the data_uint64_type() API."
                );
                core::ptr::null()
            }
        }
    }

    /// Pointer to the data. Returns nullptr if the data has been freed.
    /// Safe version of data() API that returns an Error if the data is backed
    /// by int64_t instead of void*.
    // [spec:et:def:freeable-buffer.executorch.runtime.freeable-buffer.data-safe-fn]
    // [spec:et:sem:freeable-buffer.executorch.runtime.freeable-buffer.data-safe-fn]
    pub fn data_safe(&self) -> Result<*const core::ffi::c_void> {
        crate::et_check_or_return_error!(
            matches!(self.data_, Data::Pointer(_)),
            InvalidType,
            "FreeableBuffer is backed by an uint64_t, please use the data_uint64_type() API."
        );
        match &self.data_ {
            Data::Pointer(ptr_data) => Ok(ptr_data.data_),
            Data::UInt64(_) => unreachable!(),
        }
    }

    /// Data address as a uint64_t. Returns zero if the data has been freed.
    /// Most users should use data(). data_uint64_type() is only helpful in
    /// situations where the FreeableBuffer points to memory on a different core
    /// whose pointer value is larger than the local core's void*.
    // [spec:et:def:freeable-buffer.executorch.runtime.freeable-buffer.data-uint64-type-fn]
    // [spec:et:sem:freeable-buffer.executorch.runtime.freeable-buffer.data-uint64-type-fn]
    pub fn data_uint64_type(&self) -> Result<u64> {
        crate::et_check_or_return_error!(
            matches!(self.data_, Data::UInt64(_)),
            InvalidType,
            "FreeableBuffer is backed by a void*, please use the data() API."
        );
        match &self.data_ {
            Data::UInt64(int64_data) => Ok(int64_data.data_),
            Data::Pointer(_) => unreachable!(),
        }
    }
}

impl Drop for FreeableBuffer {
    fn drop(&mut self) {
        self.free();
    }
}

// [spec:et:def:freeable-buffer.executorch.runtime.freeable-buffer.operator-fn]
// [spec:et:sem:freeable-buffer.executorch.runtime.freeable-buffer.operator-fn]
//
// PORT-NOTE: copy ctor, copy-assign, and move-assign are all `= delete`d in
// C++. Modeled here as a move-only, non-reassignable owner: `FreeableBuffer`
// derives no `Clone`, and there is no assignment operator to implement — any
// attempt to duplicate it is a compile error, matching the deleted methods.

impl Default for FreeableBuffer {
    fn default() -> Self {
        FreeableBuffer::new()
    }
}

// Silence unused-import lint on `Error` in builds where the check macro path is
// the only consumer; it names `Error` through the macro expansion.
#[allow(unused_imports)]
use Error as _;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::core::result::ResultExt;

    // struct FreeCallArgs { size_t calls; variant<const void*, uint64_t> data; size_t size; }
    //
    // The C++ `std::variant<const void*, uint64_t>` tracks whichever free
    // callback fired; modeled as two separate fields (only one is written per
    // buffer) since each test only inspects the variant it expects.
    struct FreeCallArgs {
        calls: usize,
        data_ptr: *const core::ffi::c_void,
        data_uint64: u64,
        size: usize,
    }

    impl FreeCallArgs {
        fn new() -> Self {
            FreeCallArgs {
                calls: 0,
                data_ptr: core::ptr::null(),
                data_uint64: 0,
                size: 0,
            }
        }
    }

    // void RecordFree(void* context, void* data, size_t size)
    unsafe extern "C" fn record_free(
        context: *mut core::ffi::c_void,
        data: *mut core::ffi::c_void,
        size: usize,
    ) {
        let call = unsafe { &mut *(context as *mut FreeCallArgs) };
        call.calls += 1;
        call.data_ptr = data;
        call.size = size;
    }

    // void RecordInt64Free(void* context, uint64_t data, size_t size)
    unsafe extern "C" fn record_int64_free(
        context: *mut core::ffi::c_void,
        data: u64,
        size: usize,
    ) {
        let call = unsafe { &mut *(context as *mut FreeCallArgs) };
        call.calls += 1;
        call.data_uint64 = data;
        call.size = size;
    }

    // [spec:et:sem:freeable-buffer.executorch.runtime.freeable-buffer.data-fn/test]
    // [spec:et:sem:freeable-buffer.executorch.runtime.freeable-buffer.data-safe-fn/test]
    // [spec:et:sem:freeable-buffer.executorch.runtime.freeable-buffer.size-fn/test]
    #[test]
    fn freeable_buffer_test_empty_test() {
        let fb = FreeableBuffer::new();
        assert_eq!(fb.data(), core::ptr::null());
        assert_eq!(fb.data_safe().error(), Error::Ok);
        assert_eq!(*fb.data_safe().get(), core::ptr::null());
        assert_eq!(fb.size(), 0);
    }

    // [spec:et:sem:freeable-buffer.executorch.runtime.freeable-buffer.data-fn/test]
    // [spec:et:sem:freeable-buffer.executorch.runtime.freeable-buffer.data-safe-fn/test]
    // [spec:et:sem:freeable-buffer.executorch.runtime.freeable-buffer.data-uint64-type-fn/test]
    // [spec:et:sem:freeable-buffer.executorch.runtime.freeable-buffer.free-fn/test]
    #[test]
    fn freeable_buffer_test_data_and_size_test() {
        let i: i32 = 0;
        let mut fb = FreeableBuffer::from_pointer(
            &i as *const i32 as *const core::ffi::c_void,
            core::mem::size_of::<i32>(),
            None,
            core::ptr::null_mut(),
        );

        // It should return the ctor params unmodified.
        assert_eq!(fb.size(), core::mem::size_of::<i32>());
        assert_eq!(fb.data(), &i as *const i32 as *const core::ffi::c_void);
        assert_eq!(fb.data_safe().error(), Error::Ok);
        assert_eq!(
            *fb.data_safe().get(),
            &i as *const i32 as *const core::ffi::c_void
        );

        // Freeing should clear them, even though free_fn is nullptr.
        fb.free();
        assert_eq!(fb.size(), 0);
        assert_eq!(fb.data(), core::ptr::null());
        assert_eq!(fb.data_safe().error(), Error::Ok);
        assert_eq!(*fb.data_safe().get(), core::ptr::null());

        // Use uint64_t constructor.
        let i64: u64 = 1;
        let mut fb2 = FreeableBuffer::from_uint64(
            i64,
            core::mem::size_of::<u64>(),
            None,
            core::ptr::null_mut(),
        );

        // It should return the ctor params unmodified.
        assert_eq!(fb2.size(), core::mem::size_of::<u64>());
        assert_eq!(fb2.data_uint64_type().error(), Error::Ok);
        assert_eq!(*fb2.data_uint64_type().get(), i64);

        // Freeing should clear them, even though free_fn is nullptr.
        fb2.free();
        assert_eq!(fb2.size(), 0);
        assert_eq!(fb2.data_uint64_type().error(), Error::Ok);
        assert_eq!(*fb2.data_uint64_type().get(), 0);
    }

    // [spec:et:sem:freeable-buffer.executorch.runtime.freeable-buffer.free-fn/test]
    #[test]
    fn freeable_buffer_test_free_test() {
        // Updated when record_free() is called.
        let mut call = FreeCallArgs::new();

        {
            // Create a FreeableBuffer with a free_fn that records when it's called.
            let i: i32 = 0;
            let mut fb = FreeableBuffer::from_pointer(
                &i as *const i32 as *const core::ffi::c_void,
                core::mem::size_of::<i32>(),
                Some(record_free),
                &mut call as *mut FreeCallArgs as *mut core::ffi::c_void,
            );

            // Not called during construction.
            assert_eq!(call.calls, 0);

            // Called once during Free() with the expected data/size.
            fb.free();
            assert_eq!(call.calls, 1);
            assert_eq!(call.data_ptr, &i as *const i32 as *const core::ffi::c_void);
            assert_eq!(call.size, core::mem::size_of::<i32>());

            // A second call to Free() should not call the function again.
            fb.free();
            assert_eq!(call.calls, 1);
        }

        // The destructor should not have called the function again.
        assert_eq!(call.calls, 1);

        // Test with uint64_t constructor and free function.
        let mut call2 = FreeCallArgs::new();
        {
            let i64: u64 = 1;
            let mut fb = FreeableBuffer::from_uint64(
                i64,
                core::mem::size_of::<u64>(),
                Some(record_int64_free),
                &mut call2 as *mut FreeCallArgs as *mut core::ffi::c_void,
            );

            // Not called during construction.
            assert_eq!(call2.calls, 0);

            // Called once during Free() with the expected data/size.
            fb.free();
            assert_eq!(call2.calls, 1);
            assert_eq!(call2.data_uint64, i64);
            assert_eq!(call2.size, core::mem::size_of::<u64>());

            // A second call to Free() should not call the function again.
            fb.free();
            assert_eq!(call2.calls, 1);
        }
        assert_eq!(call2.calls, 1);
    }

    // [spec:et:sem:freeable-buffer.executorch.runtime.freeable-buffer.free-fn/test]
    #[test]
    fn freeable_buffer_test_destructor_test() {
        // Updated when record_free() is called.
        let mut call = FreeCallArgs::new();
        let i: i32 = 0;

        {
            // Create a FreeableBuffer with a free_fn that records when it's called.
            let _fb = FreeableBuffer::from_pointer(
                &i as *const i32 as *const core::ffi::c_void,
                core::mem::size_of::<i32>(),
                Some(record_free),
                &mut call as *mut FreeCallArgs as *mut core::ffi::c_void,
            );

            // Not called during construction.
            assert_eq!(call.calls, 0);
        }

        // The destructor should have freed the data.
        assert_eq!(call.calls, 1);
        assert_eq!(call.data_ptr, &i as *const i32 as *const core::ffi::c_void);
        assert_eq!(call.size, core::mem::size_of::<i32>());

        // Test with uint64_t constructor and free function.
        let mut call2 = FreeCallArgs::new();
        let i64: u64 = 1;
        {
            let _fb2 = FreeableBuffer::from_uint64(
                i64,
                core::mem::size_of::<i32>(),
                Some(record_int64_free),
                &mut call2 as *mut FreeCallArgs as *mut core::ffi::c_void,
            );
            assert_eq!(call2.calls, 0);
        }
        // The destructor should have freed the data.
        assert_eq!(call2.calls, 1);
        assert_eq!(call2.data_uint64, i64);
        assert_eq!(call2.size, core::mem::size_of::<i32>());
    }

    // [spec:et:sem:freeable-buffer.executorch.runtime.freeable-buffer.freeable-buffer-fn/test]
    // [spec:et:sem:freeable-buffer.executorch.runtime.freeable-buffer.free-fn/test]
    #[test]
    fn freeable_buffer_test_move_test() {
        // Updated when record_free() is called.
        let mut call = FreeCallArgs::new();
        let i: i32 = 0;

        // Create a FreeableBuffer with some data.
        let mut fb_src = FreeableBuffer::from_pointer(
            &i as *const i32 as *const core::ffi::c_void,
            core::mem::size_of::<i32>(),
            Some(record_free),
            &mut call as *mut FreeCallArgs as *mut core::ffi::c_void,
        );
        assert_eq!(fb_src.size(), core::mem::size_of::<i32>());
        assert_eq!(fb_src.data(), &i as *const i32 as *const core::ffi::c_void);

        // Move it into a second FreeableBuffer.
        let mut fb_dst = FreeableBuffer::from_move(&mut fb_src);

        // The source FreeableBuffer should now be empty.
        assert_eq!(fb_src.size(), 0);
        assert_eq!(fb_src.data(), core::ptr::null());

        // The destination FreeableBuffer should have the data.
        assert_eq!(fb_dst.size(), core::mem::size_of::<i32>());
        assert_eq!(fb_dst.data(), &i as *const i32 as *const core::ffi::c_void);
        // Freeing the source FreeableBuffer should not call the free function.
        fb_src.free();
        assert_eq!(call.calls, 0);

        // Freeing the destination FreeableBuffer should call the free function.
        fb_dst.free();
        assert_eq!(call.calls, 1);
        assert_eq!(call.size, core::mem::size_of::<i32>());

        // Test with uint64_t constructor and free function.
        let mut call2 = FreeCallArgs::new();
        let i64: u64 = 1;
        let mut fb_src2 = FreeableBuffer::from_uint64(
            i64,
            core::mem::size_of::<u64>(),
            Some(record_int64_free),
            &mut call2 as *mut FreeCallArgs as *mut core::ffi::c_void,
        );
        assert_eq!(fb_src2.size(), core::mem::size_of::<u64>());
        assert_eq!(fb_src2.data_uint64_type().error(), Error::Ok);
        assert_eq!(*fb_src2.data_uint64_type().get(), i64);

        // Move it into a second FreeableBuffer.
        let mut fb_dst2 = FreeableBuffer::from_move(&mut fb_src2);

        // The source FreeableBuffer should now be empty.
        assert_eq!(fb_src2.size(), 0);
        assert_eq!(fb_src2.data_uint64_type().error(), Error::Ok);
        assert_eq!(*fb_src2.data_uint64_type().get(), 0);

        // The destination FreeableBuffer should have the data.
        assert_eq!(fb_dst2.size(), core::mem::size_of::<u64>());
        assert_eq!(fb_dst2.data_uint64_type().error(), Error::Ok);
        assert_eq!(*fb_dst2.data_uint64_type().get(), i64);
        // Freeing the source FreeableBuffer should not call the free function.
        fb_src2.free();
        assert_eq!(call2.calls, 0);

        // Freeing the destination FreeableBuffer should call the free function.
        fb_dst2.free();
        assert_eq!(call2.calls, 1);
        assert_eq!(call2.size, core::mem::size_of::<u64>());
    }

    // [spec:et:sem:freeable-buffer.executorch.runtime.freeable-buffer.data-uint64-type-fn/test]
    // [spec:et:sem:freeable-buffer.executorch.runtime.freeable-buffer.data-safe-fn/test]
    // [spec:et:sem:freeable-buffer.executorch.runtime.freeable-buffer.data-fn/test]
    //
    // PORT-NOTE: the C++ `APIMisuseDeathTest` splits into two checks. The
    // non-fatal parts (`data_uint64_type()`/`data_safe()` returning
    // `Error::InvalidType`) run here; the `ET_EXPECT_DEATH(fb2.data(), ...)`
    // part — which aborts via `runtime_abort()` (`libc::abort()`, not an unwind)
    // — becomes the separate `#[should_panic]`+`#[ignore]` test below, mirroring
    // the tensor_factory death-test convention.
    #[test]
    fn freeable_buffer_test_api_misuse_death_test() {
        crate::runtime::platform::platform::pal_init();
        let i: i32 = 0;
        let fb = FreeableBuffer::from_pointer(
            &i as *const i32 as *const core::ffi::c_void,
            core::mem::size_of::<i32>(),
            None,
            core::ptr::null_mut(),
        );
        assert_eq!(fb.data_uint64_type().error(), Error::InvalidType);

        let i64: u64 = 1;
        let fb2 = FreeableBuffer::from_uint64(
            i64,
            core::mem::size_of::<u64>(),
            None,
            core::ptr::null_mut(),
        );
        assert_eq!(fb2.data_safe().error(), Error::InvalidType);
    }

    // The `ET_EXPECT_DEATH(fb2.data(), ".*")` arm of APIMisuseDeathTest: calling
    // data() on a uint64-backed buffer aborts (Rust: `assert!(false, ...)`).
    #[test]
    #[should_panic]
    #[ignore]
    fn freeable_buffer_test_api_misuse_death_test_data_aborts() {
        let i64: u64 = 1;
        let fb2 = FreeableBuffer::from_uint64(
            i64,
            core::mem::size_of::<u64>(),
            None,
            core::ptr::null_mut(),
        );
        let _ = fb2.data();
    }
}
