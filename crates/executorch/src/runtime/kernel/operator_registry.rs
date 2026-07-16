//! Literal port of runtime/kernel/operator_registry.cpp + runtime/kernel/operator_registry.h.

use crate::runtime::core::error::{Error, Result};
use crate::runtime::core::evalue::EValue;
use crate::runtime::core::exec_aten::exec_aten::{DimOrderType, ScalarType};
use crate::runtime::core::result::ResultExt;
use crate::runtime::core::span::Span;
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;
use crate::runtime::platform::platform::et_pal_init;
use crate::runtime::platform::system::et_pal_get_shared_library_name;

// ET_LOG_KERNEL_KEY(k): logs the kernel key string and its is_fallback flag.
//
// PORT-NOTE: the C++ `k.data()` is a NUL-terminated C string (or nullptr for a
// fallback key). This macro renders it via `KernelKey::data_str()`, which
// converts the borrowed C string to a `&str` (empty for a null pointer).
macro_rules! et_log_kernel_key {
    ($k:expr) => {{
        crate::et_log!(
            Info,
            "key: {}, is_fallback: {}",
            $k.data_str(),
            if $k.is_fallback() { "true" } else { "false" }
        );
    }};
}

// ET_LOG_TENSOR_META(meta_list): logs the dtype + dim order of each TensorMeta.
macro_rules! et_log_tensor_meta {
    ($meta_list:expr) => {{
        let meta_list = $meta_list;
        let mut mi: usize = 0;
        while mi < meta_list.size() {
            let meta = unsafe { *meta_list.index(mi) };
            crate::et_log!(Info, "dtype: {} | dim order: [", (meta.dtype_ as i32));
            let mut i: usize = 0;
            while i < meta.dim_order_.size() {
                crate::et_log!(Info, "{},", unsafe { *meta.dim_order_.index(i) } as i32);
                i += 1;
            }
            crate::et_log!(Info, "]");
            mi += 1;
        }
    }};
}

// class KernelRuntimeContext; // Forward declaration
// PORT-NOTE: `OpFunction = void (*)(KernelRuntimeContext&, Span<EValue*>)`. The
// callback takes references/spans that carry a lifetime; modeled as a
// higher-ranked function pointer so a single `OpFunction` type can be stored in
// the registry regardless of the concrete argument lifetime.
pub type OpFunction = for<'a, 'b> fn(&'b mut KernelRuntimeContext<'a>, Span<*mut EValue<'a>>);

/// Dtype and dim order metadata for a Tensor argument to an operator.
/// Used by the Executor to hold the tensor metadata info and retrieve kernel.
// [spec:et:def:operator-registry.executorch.et-runtime-namespace.tensor-meta]
#[derive(Clone, Copy)]
pub struct TensorMeta {
    pub dtype_: ScalarType,
    pub dim_order_: Span<DimOrderType>,
}

impl TensorMeta {
    // TensorMeta() = default;
    //
    // PORT-NOTE: `ScalarType` has no zero-initialized default; the C++ defaulted
    // aggregate leaves `dtype_` value-initialized (which for the underlying
    // scalar enum is 0 == `ScalarType::Byte`) and `dim_order_` an empty span.
    pub fn default_new() -> Self {
        TensorMeta {
            dtype_: ScalarType::Byte,
            dim_order_: Span::new(),
        }
    }

    // [spec:et:def:operator-registry.executorch.et-runtime-namespace.tensor-meta.tensor-meta-fn]
    // [spec:et:sem:operator-registry.executorch.et-runtime-namespace.tensor-meta.tensor-meta-fn]
    pub fn new(dtype: ScalarType, order: Span<DimOrderType>) -> Self {
        TensorMeta {
            dtype_: dtype,
            dim_order_: order,
        }
    }

    // [spec:et:def:operator-registry.executorch.et-runtime-namespace.tensor-meta.equals-fn]
    // [spec:et:sem:operator-registry.executorch.et-runtime-namespace.tensor-meta.equals-fn]
    pub fn equals(&self, other: &TensorMeta) -> bool {
        if self.dtype_ != other.dtype_ {
            return false;
        }
        if self.dim_order_.size() != other.dim_order_.size() {
            return false;
        }
        for i in 0..self.dim_order_.size() {
            if unsafe { *self.dim_order_.index(i) } != unsafe { *other.dim_order_.index(i) } {
                return false;
            }
        }
        true
    }
}

// [spec:et:def:operator-registry.executorch.et-runtime-namespace.tensor-meta.operator-fn]
// [spec:et:sem:operator-registry.executorch.et-runtime-namespace.tensor-meta.operator-fn]
impl PartialEq for TensorMeta {
    fn eq(&self, other: &Self) -> bool {
        self.equals(other)
    }

    fn ne(&self, other: &Self) -> bool {
        !self.equals(other)
    }
}

/// Describes which dtype & dim order specialized kernel to be bound to an
/// operator.
///
/// Kernel key data is a string with the format:
///
/// ```text
/// "v<version>/<tensor_meta>|<tensor_meta>..."
/// ```
///
/// The version is v1 for now. Each tensor_meta has the following format:
/// "<dtype>;<dim_order,...>".
///
/// IMPORTANT:
/// Users should not construct a kernel key manually. Instead, it should be
/// generated from kernel yaml.
// [spec:et:def:operator-registry.executorch.et-runtime-namespace.kernel-key]
//
// PORT-NOTE: `kernel_key_data_` is a borrowed `const char*` (NUL-terminated,
// or nullptr for a fallback key). Mirrored as a raw `*const c_char` so the
// null-pointer-is-fallback and pointer-not-owned semantics survive.
// PORT-NOTE: `Debug` is derived (over the raw `*const c_char`) purely so the
// ported `assert_eq!`/`assert_ne!` tests can format a KernelKey on failure; the
// C++ `EXPECT_EQ`/`EXPECT_NE` needs no such formatter. Not a behavioral change.
#[derive(Clone, Copy, Debug)]
pub struct KernelKey {
    kernel_key_data_: *const core::ffi::c_char,
}

impl KernelKey {
    /// Creates a fallback (non-specialized) kernel key.
    // constexpr KernelKey() = default;
    pub const fn new_default() -> Self {
        KernelKey {
            kernel_key_data_: core::ptr::null(),
        }
    }

    /// Creates a specialized (non-fallback) kernel key that matches a specific
    /// set of input tensor dtypes and dim orders.
    // [spec:et:def:operator-registry.executorch.et-runtime-namespace.kernel-key.kernel-key-fn]
    // [spec:et:sem:operator-registry.executorch.et-runtime-namespace.kernel-key.kernel-key-fn]
    pub const fn new(kernel_key_data: *const core::ffi::c_char) -> Self {
        KernelKey {
            kernel_key_data_: kernel_key_data,
        }
    }

    // [spec:et:def:operator-registry.executorch.et-runtime-namespace.kernel-key.equals-fn]
    // [spec:et:sem:operator-registry.executorch.et-runtime-namespace.kernel-key.equals-fn]
    pub fn equals(&self, other: &KernelKey) -> bool {
        if self.is_fallback() != other.is_fallback() {
            return false;
        }
        if self.is_fallback() {
            return true;
        }
        strcmp(self.kernel_key_data_, other.kernel_key_data_) == 0
    }

    // [spec:et:def:operator-registry.executorch.et-runtime-namespace.kernel-key.is-fallback-fn]
    // [spec:et:sem:operator-registry.executorch.et-runtime-namespace.kernel-key.is-fallback-fn]
    pub fn is_fallback(&self) -> bool {
        self.kernel_key_data_.is_null()
    }

    // [spec:et:def:operator-registry.executorch.et-runtime-namespace.kernel-key.data-fn]
    // [spec:et:sem:operator-registry.executorch.et-runtime-namespace.kernel-key.data-fn]
    pub fn data(&self) -> *const core::ffi::c_char {
        self.kernel_key_data_
    }

    // Renders the borrowed C string as a `&str` for logging (empty for a null
    // fallback pointer). Not a C++ member; supports `et_log_kernel_key!`.
    fn data_str(&self) -> &str {
        if self.kernel_key_data_.is_null() {
            ""
        } else {
            unsafe { core::ffi::CStr::from_ptr(self.kernel_key_data_) }
                .to_str()
                .unwrap_or("")
        }
    }
}

// [spec:et:def:operator-registry.executorch.et-runtime-namespace.kernel-key.operator-fn]
// [spec:et:sem:operator-registry.executorch.et-runtime-namespace.kernel-key.operator-fn]
impl PartialEq for KernelKey {
    fn eq(&self, other: &Self) -> bool {
        self.equals(other)
    }

    fn ne(&self, other: &Self) -> bool {
        !self.equals(other)
    }
}

/// Struct that bundles a kernel key, a function and an op name together.
// [spec:et:def:operator-registry.executorch.et-runtime-namespace.kernel]
//
// PORT-NOTE: `name_` is a borrowed `const char*`; mirrored as `*const c_char`.
// `op_` is an `OpFunction` (nullable in the defaulted ctor), so it is stored as
// `Option<OpFunction>` — `None` mirrors the null function pointer used only to
// zero-fill the registry backing store.
#[derive(Clone, Copy)]
pub struct Kernel {
    pub name_: *const core::ffi::c_char,
    // String representation of kernel key. Data is not owned by the Kernel struct.
    pub kernel_key_: KernelKey,
    pub op_: Option<OpFunction>,
}

impl Kernel {
    // [spec:et:def:operator-registry.executorch.et-runtime-namespace.kernel.kernel-fn]
    // [spec:et:sem:operator-registry.executorch.et-runtime-namespace.kernel.kernel-fn]
    pub const fn new_fallback(name: *const core::ffi::c_char, func: OpFunction) -> Self {
        Kernel {
            name_: name,
            kernel_key_: KernelKey::new_default(),
            op_: Some(func),
        }
    }

    // constexpr explicit Kernel(const char* name, KernelKey key, OpFunction func)
    pub const fn new(name: *const core::ffi::c_char, key: KernelKey, func: OpFunction) -> Self {
        Kernel {
            name_: name,
            kernel_key_: key,
            op_: Some(func),
        }
    }

    // constexpr Kernel() : name_(nullptr), op_(nullptr) {}
    pub const fn new_default() -> Self {
        Kernel {
            name_: core::ptr::null(),
            kernel_key_: KernelKey::new_default(),
            op_: None,
        }
    }
}

pub mod internal {
    use super::{TensorMeta, copy_char_as_number_to_buf};
    use crate::runtime::core::error::Error;
    use crate::runtime::core::span::Span;

    /// A make_kernel_key_string buffer size that is large enough to hold a
    /// kernel key string with 16 tensors of 16 dimensions, plus the trailing
    /// NUL byte.
    pub const K_KERNEL_KEY_BUF_SIZE: usize = 659;

    // [spec:et:def:operator-registry.executorch.et-runtime-namespace.internal.make-kernel-key-string-fn]
    // [spec:et:sem:operator-registry.executorch.et-runtime-namespace.internal.make-kernel-key-string-fn]
    //
    // PORT-NOTE: the C++ writes into a raw `char* buf` advancing a pointer and
    // decrementing a remaining-size counter. The Rust port keeps the same
    // pointer-walking control flow over a `*mut c_char` so the byte-identical
    // output and every size check survive verbatim.
    pub fn make_kernel_key_string(
        key: Span<TensorMeta>,
        buf: *mut core::ffi::c_char,
        buf_size: usize,
    ) -> Error {
        let mut buf = buf;
        let mut buf_size = buf_size;

        if key.empty() {
            // If no tensor is present in an op, kernel key does not apply.
            if buf_size > 0 {
                unsafe {
                    *buf = 0;
                }
            }
            return Error::Ok;
        }

        // Reserve one byte for null terminator.
        if buf_size < 1 {
            return Error::InvalidArgument;
        }
        buf_size -= 1;

        // Add prefix.
        if buf_size < 3 {
            return Error::InvalidArgument;
        }
        unsafe {
            core::ptr::copy_nonoverlapping(b"v1/".as_ptr() as *const core::ffi::c_char, buf, 3);
            buf = buf.add(3);
        }
        buf_size -= 3;

        // Add tensor meta.
        for i in 0..key.size() {
            let meta = unsafe { key.index(i) };

            // Add dtype.
            let mut n = copy_char_as_number_to_buf(meta.dtype_ as i32, buf, buf_size);
            if n < 0 {
                return Error::InvalidArgument;
            }
            unsafe {
                buf = buf.add(n as usize);
            }
            buf_size -= n as usize;

            // Add separator between dtype and dim order.
            if buf_size < 1 {
                return Error::InvalidArgument;
            }
            unsafe {
                *buf = b';' as core::ffi::c_char;
                buf = buf.add(1);
            }
            buf_size -= 1;

            // Add dim order.
            for j in 0..meta.dim_order_.size() {
                n = copy_char_as_number_to_buf(
                    unsafe { *meta.dim_order_.index(j) } as i32,
                    buf,
                    buf_size,
                );
                if n < 0 {
                    return Error::InvalidArgument;
                }
                unsafe {
                    buf = buf.add(n as usize);
                }
                buf_size -= n as usize;

                if j < meta.dim_order_.size() - 1 {
                    if buf_size < 1 {
                        return Error::InvalidArgument;
                    }
                    unsafe {
                        *buf = b',' as core::ffi::c_char;
                        buf = buf.add(1);
                    }
                    buf_size -= 1;
                }
            }
            if i < key.size() - 1 {
                if buf_size < 1 {
                    return Error::InvalidArgument;
                }
                unsafe {
                    *buf = b'|' as core::ffi::c_char;
                    buf = buf.add(1);
                }
                buf_size -= 1;
            }
        }
        unsafe {
            *buf = 0; // Space for this was reserved above.
        }
        Error::Ok
    }
}

// Maximum number of operators and their associated kernels that can be
// registered. Resolution order:
//   1. User-defined -DMAX_KERNEL_NUM wins.
//   2. Otherwise, EXECUTORCH_SELECTED_MAX_KERNEL_NUM from selective build.
//   3. Otherwise, fall back to a default of 2000 slots.
//
// PORT-NOTE: `MAX_KERNEL_NUM` and `EXECUTORCH_SELECTED_MAX_KERNEL_NUM` are C++
// preprocessor defines chosen at compile time. Only the default fallback path
// (250 * 8 = 2000) is ported; the selective-build overrides are an unresolved
// cross-module reference (would map to build-time cfg values).
const K_MAX_OPERATORS: u32 = 250;
const K_MAX_KERNELS_PER_OP: u32 = 8;
const K_MAX_REGISTERED_KERNELS: u32 = K_MAX_OPERATORS * K_MAX_KERNELS_PER_OP;

// Data that backs the kernel table.
// [spec:et:def:operator-registry.executorch.et-runtime-namespace.kernel-buffer]
//
// PORT-NOTE: the C++ `KernelBuffer` is a `sizeof(Kernel)`-byte, `alignas(Kernel)`
// blob used only to zero-fill the backing store without running `Kernel`'s
// non-trivial default ctor. In Rust a `Kernel` array can be initialized with
// `Kernel::new_default()` (all fields null/None), which is the equivalent
// "zeroed" state, so the table is a `[Kernel; N]` directly and the separate
// `KernelBuffer` reinterpret-cast dance is unnecessary.
struct KernelRegistry {
    // @lint-ignore CLANGTIDY facebook-hte-CArray
    registered_kernels: [Kernel; K_MAX_REGISTERED_KERNELS as usize],
    // The number of kernels registered in the table.
    num_registered_kernels: usize,
}

// PORT-NOTE: the C++ globals (`registered_kernels`, `num_registered_kernels`)
// are unsynchronized process-global mutable state populated at
// static-initialization time. Mirrored here as a single `static mut`
// accessed through unsafe, reproducing the same lack of synchronization. All
// access is funneled through `registry()` / `registry_mut()`.
static mut REGISTRY: KernelRegistry = KernelRegistry {
    registered_kernels: [Kernel::new_default(); K_MAX_REGISTERED_KERNELS as usize],
    num_registered_kernels: 0,
};

#[allow(static_mut_refs)]
unsafe fn registry() -> &'static KernelRegistry {
    unsafe { &REGISTRY }
}

#[allow(static_mut_refs)]
unsafe fn registry_mut() -> &'static mut KernelRegistry {
    unsafe { &mut REGISTRY }
}

/// Test-only: empties the process-global kernel table so an in-process test
/// suite can register a fresh set of kernels. The C++ has no such API because
/// each gtest binary starts with a zero-initialized static registry; the Rust
/// test binary shares one process-wide `REGISTRY` across every suite, so the
/// registry-dependent suites reset it (under `OPERATOR_REGISTRY_TEST_LOCK`) to
/// stay isolated, mirroring the sibling device_allocator pattern.
#[cfg(test)]
pub(crate) fn clear_registry_for_test() {
    unsafe {
        let reg = registry_mut();
        reg.num_registered_kernels = 0;
    }
}

/// Test-only serialization lock shared by every operator-registry test suite in
/// this module. The process-wide `REGISTRY` is a shared mutable global with no
/// reset in the C++; holding this lock lets a suite clear + re-register kernels
/// and read them back without racing another suite that also mutates the table.
#[cfg(test)]
pub(crate) static OPERATOR_REGISTRY_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

// strcmp over two NUL-terminated C strings; mirrors the C++ `strcmp` used for
// kernel-key and op-name comparison. Returns 0 iff byte-for-byte identical.
fn strcmp(a: *const core::ffi::c_char, b: *const core::ffi::c_char) -> i32 {
    unsafe {
        let mut i = 0isize;
        loop {
            let ca = *a.offset(i) as u8;
            let cb = *b.offset(i) as u8;
            if ca != cb {
                return ca as i32 - cb as i32;
            }
            if ca == 0 {
                return 0;
            }
            i += 1;
        }
    }
}

// Registers the kernels, but may return an error.
// [spec:et:def:operator-registry.executorch.et-runtime-namespace.register-kernels-internal-fn]
// [spec:et:sem:operator-registry.executorch.et-runtime-namespace.register-kernels-internal-fn]
fn register_kernels_internal(kernels: Span<Kernel>) -> Error {
    // Operator registration happens in static initialization time before or
    // after PAL init, so call it here. It is safe to call multiple times.
    unsafe {
        et_pal_init();
    }

    let num_registered_kernels = unsafe { registry().num_registered_kernels };
    if kernels.size() + num_registered_kernels > K_MAX_REGISTERED_KERNELS as usize {
        crate::et_log!(
            Error,
            "The total number of kernels to be registered is larger than the limit {}. {} kernels are already registered and we're trying to register another {} kernels.",
            K_MAX_REGISTERED_KERNELS,
            num_registered_kernels as u32,
            kernels.size() as u32
        );
        crate::et_log!(Error, "======== Kernels already in the registry: ========");
        for i in 0..num_registered_kernels {
            let k = unsafe { registry().registered_kernels[i] };
            crate::et_log!(Error, "{}", c_str_display(k.name_));
            et_log_kernel_key!(k.kernel_key_);
        }
        crate::et_log!(Error, "======== Kernels being registered: ========");
        for i in 0..kernels.size() {
            let k = unsafe { *kernels.index(i) };
            crate::et_log!(Error, "{}", c_str_display(k.name_));
            et_log_kernel_key!(k.kernel_key_);
        }
        return Error::RegistrationExceedingMaxKernels;
    }
    // for debugging purpose
    let lib_name = et_pal_get_shared_library_name(kernels.data() as *const core::ffi::c_void);

    for kernel_idx in 0..kernels.size() {
        let kernel = unsafe { *kernels.index(kernel_idx) };
        // Linear search. This is fine if the number of kernels is small.
        let num_registered_kernels = unsafe { registry().num_registered_kernels };
        for i in 0..num_registered_kernels {
            let k = unsafe { registry().registered_kernels[i] };
            if strcmp(kernel.name_, k.name_) == 0 && kernel.kernel_key_ == k.kernel_key_ {
                crate::et_log!(
                    Error,
                    "Re-registering {}, from {}",
                    c_str_display(k.name_),
                    c_str_display(lib_name)
                );
                et_log_kernel_key!(k.kernel_key_);
                // ET_CHECK_MSG(false, ...): fatal abort in this variant.
                //
                // PORT-NOTE: `ET_CHECK_MSG` is an unresolved cross-module
                // reference (no shared macro yet); mirror its log-then-abort
                // semantics inline. The trailing `return` is unreachable but
                // preserved to match the C++ source.
                crate::et_log!(
                    Fatal,
                    "Kernel registration failed with error {}, Re-registering {}, from {}",
                    Error::RegistrationAlreadyRegistered as u32,
                    c_str_display(k.name_),
                    c_str_display(lib_name)
                );
                crate::runtime::platform::abort::runtime_abort();
                #[allow(unreachable_code)]
                return Error::RegistrationAlreadyRegistered;
            }
        }
        unsafe {
            let reg = registry_mut();
            reg.registered_kernels[reg.num_registered_kernels] = kernel;
            reg.num_registered_kernels += 1;
        }
    }
    crate::et_log!(
        Debug,
        "Successfully registered all kernels from shared library: {}",
        c_str_display(lib_name)
    );

    Error::Ok
}

// Registers the kernels, but panics if an error occurs. Always returns Ok.
// [spec:et:def:operator-registry.executorch.et-runtime-namespace.register-kernels-fn]
// [spec:et:sem:operator-registry.executorch.et-runtime-namespace.register-kernels-fn]
#[must_use]
pub fn register_kernels(kernels: Span<Kernel>) -> Error {
    let success = register_kernels_internal(kernels);
    if success == Error::RegistrationAlreadyRegistered
        || success == Error::RegistrationExceedingMaxKernels
    {
        // ET_CHECK_MSG(false, ...): unconditional fatal abort.
        //
        // PORT-NOTE: `ET_CHECK_MSG` is an unresolved cross-module reference;
        // mirror its log-then-abort semantics inline.
        crate::et_log!(
            Fatal,
            "Kernel registration failed with error {}, see error log for details.",
            success as u32
        );
        crate::runtime::platform::abort::runtime_abort();
    }
    success
}

/// Registers a single kernel.
// [spec:et:def:operator-registry.executorch.et-runtime-namespace.register-kernel-fn]
// [spec:et:sem:operator-registry.executorch.et-runtime-namespace.register-kernel-fn]
#[must_use]
pub fn register_kernel(kernel: &Kernel) -> Error {
    register_kernels(Span::from_raw_parts(
        kernel as *const Kernel as *mut Kernel,
        1,
    ))
}

// Writes `num` as a decimal string to `buf` and returns the number of bytes
// written. Returns -1 if `buf` is too small or if `num` is not supported.
// [spec:et:def:operator-registry.executorch.et-runtime-namespace.copy-char-as-number-to-buf-fn]
// [spec:et:sem:operator-registry.executorch.et-runtime-namespace.copy-char-as-number-to-buf-fn]
fn copy_char_as_number_to_buf(num: i32, buf: *mut core::ffi::c_char, buf_size: usize) -> i32 {
    if num < 0 {
        return -1;
    }
    if num < 10 {
        if buf_size < 1 {
            return -1;
        }
        unsafe {
            *buf = (b'0' as i32 + num) as core::ffi::c_char;
        }
        return 1;
    }
    if num < 100 {
        if buf_size < 2 {
            return -1;
        }
        unsafe {
            *buf = (b'0' as core::ffi::c_char).wrapping_add((num as u8 as core::ffi::c_char) / 10);
            *buf.add(1) =
                (b'0' as core::ffi::c_char).wrapping_add((num as u8 as core::ffi::c_char) % 10);
        }
        return 2;
    }
    -1
}

// [spec:et:def:operator-registry.executorch.et-runtime-namespace.registry-has-op-function-fn]
// [spec:et:sem:operator-registry.executorch.et-runtime-namespace.registry-has-op-function-fn]
//
// PORT-NOTE: C++ defaults `meta_list` to an empty span; Rust callers pass
// `Span::new()` explicitly for that case.
pub fn registry_has_op_function(
    name: *const core::ffi::c_char,
    meta_list: Span<TensorMeta>,
) -> bool {
    ResultExt::ok(&get_op_function_from_registry_2(name, meta_list))
}

// [spec:et:def:operator-registry.executorch.et-runtime-namespace.get-op-function-from-registry-fn]
// [spec:et:sem:operator-registry.executorch.et-runtime-namespace.get-op-function-from-registry-fn]
pub fn get_op_function_from_registry(
    name: *const core::ffi::c_char,
    meta_list: Span<TensorMeta>,
    kernel_list: Span<Kernel>,
) -> Result<OpFunction> {
    let mut key_string = [0 as core::ffi::c_char; internal::K_KERNEL_KEY_BUF_SIZE];
    let err =
        internal::make_kernel_key_string(meta_list, key_string.as_mut_ptr(), key_string.len());
    if err != Error::Ok {
        crate::et_log!(Error, "Failed to make kernel key string");
        return Err(err);
    }
    let kernel_key = KernelKey::new(key_string.as_ptr());

    let mut fallback_idx: i32 = -1;
    for idx in 0..kernel_list.size() {
        let entry = unsafe { *kernel_list.index(idx) };
        if strcmp(entry.name_, name) == 0 {
            if entry.kernel_key_ == kernel_key {
                return Ok(entry.op_.unwrap());
            }
            if entry.kernel_key_.is_fallback() {
                fallback_idx = idx as i32;
            }
        }
    }
    if fallback_idx != -1 {
        let entry = unsafe { *kernel_list.index(fallback_idx as usize) };
        return Ok(entry.op_.unwrap());
    }
    crate::et_log!(Error, "kernel '{}' not found.", c_str_display(name));
    et_log_tensor_meta!(meta_list);
    Err(Error::OperatorMissing)
}

// The two-argument overload that searches the global registry.
//
// PORT-NOTE: Rust has no function overloading; the C++ two-arg overload
// `get_op_function_from_registry(name, meta_list)` is named
// `get_op_function_from_registry_2` here.
pub fn get_op_function_from_registry_2(
    name: *const core::ffi::c_char,
    meta_list: Span<TensorMeta>,
) -> Result<OpFunction> {
    get_op_function_from_registry(name, meta_list, get_registered_kernels())
}

// [spec:et:def:operator-registry.executorch.et-runtime-namespace.get-registered-kernels-fn]
// [spec:et:sem:operator-registry.executorch.et-runtime-namespace.get-registered-kernels-fn]
pub fn get_registered_kernels() -> Span<Kernel> {
    unsafe {
        let reg = registry();
        Span::from_raw_parts(
            reg.registered_kernels.as_ptr() as *mut Kernel,
            reg.num_registered_kernels,
        )
    }
}

// Renders a borrowed NUL-terminated C string for logging (empty if null).
fn c_str_display(s: *const core::ffi::c_char) -> &'static str {
    if s.is_null() {
        ""
    } else {
        unsafe { core::ffi::CStr::from_ptr(s) }
            .to_str()
            .unwrap_or("")
    }
}

// TODO(T197294990): Remove these deprecated aliases once all users have moved to
// the new `::executorch` namespaces.
pub mod torch_executor {
    use super::{
        Kernel, OpFunction, TensorMeta, get_op_function_from_registry_2, get_registered_kernels,
        register_kernels as register_kernels_impl, registry_has_op_function,
    };
    use crate::runtime::core::array_ref::ArrayRef;
    use crate::runtime::core::error::Error;
    use crate::runtime::core::result::ResultExt;
    use crate::runtime::core::span::Span;

    // [spec:et:def:operator-registry.torch.executor.register-kernels-fn]
    // [spec:et:sem:operator-registry.torch.executor.register-kernels-fn]
    #[must_use]
    pub fn register_kernels(kernels: ArrayRef<Kernel>) -> Error {
        register_kernels_impl(Span::from_raw_parts(
            kernels.data() as *mut Kernel,
            kernels.size(),
        ))
    }

    // [spec:et:def:operator-registry.torch.executor.get-ops-fn-fn]
    // [spec:et:sem:operator-registry.torch.executor.get-ops-fn-fn]
    //
    // PORT-NOTE: C++ defaults `meta_list` to `{}`; Rust callers pass an empty
    // `ArrayRef` explicitly.
    pub fn get_ops_fn(
        name: *const core::ffi::c_char,
        meta_list: ArrayRef<TensorMeta>,
    ) -> OpFunction {
        let result = get_op_function_from_registry_2(
            name,
            Span::from_raw_parts(meta_list.data() as *mut TensorMeta, meta_list.size()),
        );
        // ET_CHECK(result.ok()): get_op_function_from_registry() logs details.
        //
        // PORT-NOTE: `ET_CHECK` is an unresolved cross-module reference; mirror
        // its abort-on-false semantics.
        if !ResultExt::ok(&result) {
            crate::runtime::platform::abort::runtime_abort();
        }
        *ResultExt::get(&result)
    }

    // [spec:et:def:operator-registry.torch.executor.has-ops-fn-fn]
    // [spec:et:sem:operator-registry.torch.executor.has-ops-fn-fn]
    pub fn has_ops_fn(name: *const core::ffi::c_char, meta_list: ArrayRef<TensorMeta>) -> bool {
        registry_has_op_function(
            name,
            Span::from_raw_parts(meta_list.data() as *mut TensorMeta, meta_list.size()),
        )
    }

    // [spec:et:def:operator-registry.torch.executor.get-kernels-fn]
    // [spec:et:sem:operator-registry.torch.executor.get-kernels-fn]
    pub fn get_kernels() -> ArrayRef<Kernel> {
        let kernels: Span<Kernel> = get_registered_kernels();
        ArrayRef::from_raw_parts(kernels.data(), kernels.size())
    }
}

// Ports of:
//   runtime/kernel/test/operator_registry_test.cpp
//   runtime/kernel/test/operator_registry_max_kernel_num_test.cpp
//   runtime/kernel/test/kernel_double_registration_test.cpp
//   runtime/kernel/test/test_generated_lib_and_aten.cpp
//   runtime/kernel/test/test_kernel_manual_registration.cpp
//
// PORT-NOTE: every C++ suite here runs in its own gtest process against a fresh
// static registry. The Rust test binary shares one process-wide `REGISTRY`
// across all suites and runs tests in parallel, so each test serializes on
// `OPERATOR_REGISTRY_TEST_LOCK` and calls `setup()`, which resets the registry
// via `clear_registry_for_test()`. That reset also removes the cross-suite op
// name collisions (`foo` is registered by both `OperatorRegistryTest::Basic`
// and `OperatorRegistryMaxKernelNumTest::RegisterOneOp`, `test::boo`/etc.) that
// would otherwise re-register a name and abort.
#[cfg(test)]
mod tests {
    use super::internal::{K_KERNEL_KEY_BUF_SIZE, make_kernel_key_string};
    use super::{
        Kernel, KernelKey, OPERATOR_REGISTRY_TEST_LOCK, OpFunction, TensorMeta,
        clear_registry_for_test, get_op_function_from_registry, get_op_function_from_registry_2,
        register_kernels, registry_has_op_function,
    };
    use crate::runtime::core::error::Error;
    use crate::runtime::core::evalue::EValue;
    use crate::runtime::core::exec_aten::exec_aten::{DimOrderType, ScalarType};
    use crate::runtime::core::memory_allocator::MemoryAllocatorBase;
    use crate::runtime::core::portable_type::scalar::Scalar;
    use crate::runtime::core::result::ResultExt;
    use crate::runtime::core::span::Span;
    use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;
    use crate::runtime::platform::runtime::runtime_init;

    // Mirrors the C++ default-constructed `KernelRuntimeContext context{};` used
    // to invoke kernels in these tests: no event tracer, no temp allocator. Null
    // `dyn` fat pointers are built per the established
    // `null_event_tracer()` / `null_mut::<Concrete>() as *mut dyn Trait` pattern.
    fn null_context() -> KernelRuntimeContext<'static> {
        KernelRuntimeContext::new(
            crate::extension::module::module::null_event_tracer(),
            core::ptr::null_mut::<crate::runtime::core::memory_allocator::MemoryAllocator>()
                as *mut dyn MemoryAllocatorBase,
        )
    }

    // Mirrors every fixture's `SetUp()` (`runtime_init()`), plus the shared-
    // registry reset described in the module PORT-NOTE. Must hold
    // OPERATOR_REGISTRY_TEST_LOCK.
    fn setup() {
        runtime_init();
        clear_registry_for_test();
    }

    // Port of runtime/kernel/test/test_util.h `make_kernel_key`: builds a
    // `Span<TensorMeta>` from (dtype, dim_order) pairs and forwards to
    // `make_kernel_key_string`. The `dim_order` vectors must outlive the call
    // because the `TensorMeta` spans borrow them, matching the C++ where the
    // input vector owns the dim-order storage for the duration of the call.
    fn make_kernel_key(
        tensors: &[(ScalarType, Vec<DimOrderType>)],
        buf: *mut core::ffi::c_char,
        buf_size: usize,
    ) -> Error {
        let mut meta: Vec<TensorMeta> = Vec::new();
        for t in tensors {
            let dim_order = Span::from_raw_parts(t.1.as_ptr() as *mut DimOrderType, t.1.len());
            meta.push(TensorMeta::new(t.0, dim_order));
        }
        let metadata = Span::from_raw_parts(meta.as_ptr() as *mut TensorMeta, meta.len());
        make_kernel_key_string(metadata, buf, buf_size)
    }

    // No-op kernel: `[](KernelRuntimeContext&, Span<EValue*>) {}`.
    fn noop_kernel(_context: &mut KernelRuntimeContext, _stack: Span<*mut EValue>) {}

    // `[](KernelRuntimeContext& context, Span<EValue*> stack) { *(stack[0]) = Scalar(100); }`
    fn write_100_kernel(_context: &mut KernelRuntimeContext, stack: Span<*mut EValue>) {
        let mut tmp = EValue::from_scalar(Scalar::from_i64(100));
        unsafe {
            (**stack.index(0)).assign_move(&mut tmp);
        }
    }

    // `[](KernelRuntimeContext& context, Span<EValue*> stack) { *(stack[0]) = Scalar(50); }`
    fn write_50_kernel(_context: &mut KernelRuntimeContext, stack: Span<*mut EValue>) {
        let mut tmp = EValue::from_scalar(Scalar::from_i64(50));
        unsafe {
            (**stack.index(0)).assign_move(&mut tmp);
        }
    }

    //
    // Tests for make_kernel_key_string
    //

    // Helper for testing make_kernel_key_string.
    fn test_make_kernel_key_string(
        tensors: &[(ScalarType, Vec<DimOrderType>)],
        expected_key: &str,
    ) {
        let min_buf_size = expected_key.len() + 1;

        // Sweep across too-small buffer sizes, exercising all possible failure
        // checks.
        for buf_size in 0..min_buf_size {
            let mut actual_key = vec![0x55u8 as core::ffi::c_char; buf_size];
            let err = make_kernel_key(
                tensors,
                // nullptr should be valid for buf_size == 0 because it won't be
                // written to.
                if buf_size == 0 {
                    core::ptr::null_mut()
                } else {
                    actual_key.as_mut_ptr()
                },
                actual_key.len(),
            );
            assert_ne!(err, Error::Ok);
        }

        // Demonstrate that it succeeds for buffers of exactly the right size or
        // larger.
        for buf_size in min_buf_size..min_buf_size + 1 {
            let mut actual_key = vec![0x55u8 as core::ffi::c_char; buf_size];
            let err = make_kernel_key(tensors, actual_key.as_mut_ptr(), actual_key.len());
            assert_eq!(err, Error::Ok);
            assert_eq!(cstr_to_string(actual_key.as_ptr()), expected_key);
        }
    }

    // Reads a NUL-terminated C string out of a buffer, mirroring `EXPECT_STREQ`.
    fn cstr_to_string(ptr: *const core::ffi::c_char) -> String {
        unsafe { core::ffi::CStr::from_ptr(ptr).to_str().unwrap().to_string() }
    }

    // [spec:et:sem:operator-registry.executorch.et-runtime-namespace.internal.make-kernel-key-string-fn/test]
    #[test]
    fn make_kernel_key_string_test_zero_tensor_success_with_null_buffer() {
        let err = make_kernel_key(&[], core::ptr::null_mut(), 0);
        assert_eq!(err, Error::Ok);
    }

    // [spec:et:sem:operator-registry.executorch.et-runtime-namespace.internal.make-kernel-key-string-fn/test]
    #[test]
    fn make_kernel_key_string_test_zero_tensor_success_makes_empty_string() {
        let mut buf: core::ffi::c_char = 0x55u8 as core::ffi::c_char;
        let err = make_kernel_key(&[], &mut buf, 1);
        assert_eq!(err, Error::Ok);
        assert_eq!(buf, 0);
    }

    // [spec:et:sem:operator-registry.executorch.et-runtime-namespace.internal.make-kernel-key-string-fn/test]
    // [spec:et:sem:operator-registry.executorch.et-runtime-namespace.copy-char-as-number-to-buf-fn/test]
    #[test]
    fn make_kernel_key_string_test_one_tensor_success() {
        test_make_kernel_key_string(&[(ScalarType::Long, vec![0, 1, 2, 3])], "v1/4;0,1,2,3");
    }

    // [spec:et:sem:operator-registry.executorch.et-runtime-namespace.internal.make-kernel-key-string-fn/test]
    #[test]
    fn make_kernel_key_string_test_two_tensor_success() {
        test_make_kernel_key_string(
            &[
                (ScalarType::Long, vec![0, 1, 2, 3]),
                (ScalarType::Double, vec![3, 2, 1, 0]),
            ],
            "v1/4;0,1,2,3|7;3,2,1,0",
        );
    }

    // [spec:et:sem:operator-registry.executorch.et-runtime-namespace.internal.make-kernel-key-string-fn/test]
    #[test]
    fn make_kernel_key_string_test_three_tensor_success() {
        test_make_kernel_key_string(
            &[
                (ScalarType::Long, vec![0, 1, 2, 3]),
                (ScalarType::Double, vec![3, 2, 1, 0]),
                (ScalarType::Byte, vec![2, 1, 3, 0]),
            ],
            "v1/4;0,1,2,3|7;3,2,1,0|0;2,1,3,0",
        );
    }

    // [spec:et:sem:operator-registry.executorch.et-runtime-namespace.internal.make-kernel-key-string-fn/test]
    // [spec:et:sem:operator-registry.executorch.et-runtime-namespace.copy-char-as-number-to-buf-fn/test]
    #[test]
    fn make_kernel_key_string_test_two_digit_dim_order_success() {
        test_make_kernel_key_string(&[(ScalarType::Long, vec![0, 10, 2, 99])], "v1/4;0,10,2,99");
    }

    // [spec:et:sem:operator-registry.executorch.et-runtime-namespace.copy-char-as-number-to-buf-fn/test]
    #[test]
    fn make_kernel_key_string_test_three_digit_dim_order_failure() {
        let mut actual_key = vec![0x55u8 as core::ffi::c_char; 1024]; // Large enough for any key.
        let err = make_kernel_key(
            // Cannot represent a dim order entry with more than two digits.
            &[(ScalarType::Long, vec![0, 100, 2, 255])],
            actual_key.as_mut_ptr(),
            actual_key.len(),
        );
        assert_ne!(err, Error::Ok);
    }

    // PORT-NOTE: the C++ test passes `(ScalarType)-1` — legal because C++
    // `ScalarType` is an `int8_t` that can hold -1. The Rust port models
    // `ScalarType` as a `#[repr(i8)]` enum crate-wide, and `TensorMeta.dtype_`
    // is typed `ScalarType`, so a -1 discriminant is not a valid enum value:
    // constructing it (via transmute) is instant UB and aborts under debug
    // enum-validity checks before `make_kernel_key_string`'s `dtype_ as i32 < 0`
    // guard can fire. Reproducing the negative-dtype path would require
    // widening `TensorMeta.dtype_` to a raw integer — a cross-module redesign of
    // the ScalarType port. Ported and `#[ignore]`d; the guard itself
    // (`num < 0 => -1`) is still exercised indirectly by
    // copy_char_as_number_to_buf's contract.
    // [spec:et:sem:operator-registry.executorch.et-runtime-namespace.copy-char-as-number-to-buf-fn/test]
    #[test]
    #[ignore]
    fn make_kernel_key_string_test_negative_scalar_type_failure() {
        let mut actual_key = vec![0x55u8 as core::ffi::c_char; 1024]; // Large enough for any key.
        // Cannot represent a ScalarType (aka int8_t) with a negative value.
        let neg_scalar_type: ScalarType = unsafe { core::mem::transmute::<i8, ScalarType>(-1i8) };
        let err = make_kernel_key(
            &[(neg_scalar_type, vec![0, 1, 2, 3])],
            actual_key.as_mut_ptr(),
            actual_key.len(),
        );
        assert_ne!(err, Error::Ok);
    }

    // [spec:et:sem:operator-registry.executorch.et-runtime-namespace.internal.make-kernel-key-string-fn/test]
    #[test]
    fn make_kernel_key_string_test_key_buf_size_meets_assumptions() {
        // Create the longest key that fits in the assumptions of
        // K_KERNEL_KEY_BUF_SIZE: 16 tensors, 16 dims, with two-digit
        // ScalarTypes.
        let mut tensors: Vec<(ScalarType, Vec<DimOrderType>)> = Vec::with_capacity(16);
        for _ in 0..16 {
            let mut dims: Vec<DimOrderType> = Vec::with_capacity(16);
            for j in 0..16 {
                dims.push(j as DimOrderType);
            }
            tensors.push((
                unsafe { core::mem::transmute::<i8, ScalarType>(10i8) },
                dims,
            ));
        }

        let mut actual_key = vec![0x55u8 as core::ffi::c_char; K_KERNEL_KEY_BUF_SIZE];
        let err = make_kernel_key(&tensors, actual_key.as_mut_ptr(), actual_key.len());
        assert_eq!(err, Error::Ok);
        assert_eq!(
            cstr_to_string(actual_key.as_ptr()),
            concat!(
                "v1/",
                "10;0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15|",
                "10;0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15|",
                "10;0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15|",
                "10;0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15|",
                "10;0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15|",
                "10;0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15|",
                "10;0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15|",
                "10;0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15|",
                "10;0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15|",
                "10;0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15|",
                "10;0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15|",
                "10;0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15|",
                "10;0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15|",
                "10;0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15|",
                "10;0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15|",
                "10;0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15"
            )
        );
        assert!(cstr_to_string(actual_key.as_ptr()).len() + 1 <= K_KERNEL_KEY_BUF_SIZE);
    }

    //
    // Test for TensorMeta::equals (and TensorMeta::new)
    //

    // No upstream test exercises `TensorMeta::equals`: the registry lookup path
    // compares `KernelKey`s, never `TensorMeta`s directly. Pin its contract
    // literally per the sem rule: equal iff same dtype AND same dim-order length
    // AND element-wise-equal dim order; any single mismatch (dtype, length, or an
    // element) makes it unequal. Also exercises `TensorMeta::new` storing the
    // dtype and dim-order span verbatim.
    // [spec:et:sem:operator-registry.executorch.et-runtime-namespace.tensor-meta.equals-fn/test]
    // [spec:et:sem:operator-registry.executorch.et-runtime-namespace.tensor-meta.tensor-meta-fn/test]
    #[test]
    fn tensor_meta_equals() {
        let dims_a: [DimOrderType; 4] = [0, 1, 2, 3];
        let dims_b: [DimOrderType; 4] = [0, 1, 2, 3];
        let dims_channel_first: [DimOrderType; 4] = [0, 3, 1, 2];
        let dims_short: [DimOrderType; 3] = [0, 1, 2];

        let span =
            |d: &[DimOrderType]| Span::from_raw_parts(d.as_ptr() as *mut DimOrderType, d.len());

        let long_a = TensorMeta::new(ScalarType::Long, span(&dims_a));
        // Same dtype, same dim order (distinct backing storage): equal.
        let long_b = TensorMeta::new(ScalarType::Long, span(&dims_b));
        assert!(long_a.equals(&long_b));

        // Different dtype: unequal.
        let float_a = TensorMeta::new(ScalarType::Float, span(&dims_a));
        assert!(!long_a.equals(&float_a));

        // Same dtype, same length, different element: unequal.
        let long_channel_first = TensorMeta::new(ScalarType::Long, span(&dims_channel_first));
        assert!(!long_a.equals(&long_channel_first));

        // Same dtype, different dim-order length: unequal.
        let long_short = TensorMeta::new(ScalarType::Long, span(&dims_short));
        assert!(!long_a.equals(&long_short));

        // Reflexive.
        assert!(long_a.equals(&long_a));
    }

    // `operator==` (and the sibling `operator!=`) for TensorMeta delegate
    // entirely to `equals`: `==`/`!=` must agree with the dtype + dim-order
    // comparison above on both the equal and every unequal case.
    // [spec:et:sem:operator-registry.executorch.et-runtime-namespace.tensor-meta.operator-fn/test]
    #[test]
    fn tensor_meta_operator_eq_ne() {
        let dims_a: [DimOrderType; 4] = [0, 1, 2, 3];
        let dims_b: [DimOrderType; 4] = [0, 1, 2, 3];
        let dims_channel_first: [DimOrderType; 4] = [0, 3, 1, 2];
        let dims_short: [DimOrderType; 3] = [0, 1, 2];

        let span =
            |d: &[DimOrderType]| Span::from_raw_parts(d.as_ptr() as *mut DimOrderType, d.len());

        let long_a = TensorMeta::new(ScalarType::Long, span(&dims_a));
        let long_b = TensorMeta::new(ScalarType::Long, span(&dims_b));
        let float_a = TensorMeta::new(ScalarType::Float, span(&dims_a));
        let long_channel_first = TensorMeta::new(ScalarType::Long, span(&dims_channel_first));
        let long_short = TensorMeta::new(ScalarType::Long, span(&dims_short));

        // Same dtype + same dim order (distinct backing storage): ==.
        assert!(long_a == long_b);
        assert!(!(long_a != long_b));

        // Different dtype: !=.
        assert!(long_a != float_a);
        assert!(!(long_a == float_a));

        // Same dtype, same length, different element: !=.
        assert!(long_a != long_channel_first);

        // Same dtype, different dim-order length: !=.
        assert!(long_a != long_short);
    }

    //
    // Tests for public operator registry APIs
    //

    // [spec:et:sem:operator-registry.executorch.et-runtime-namespace.register-kernels-fn/test]
    // [spec:et:sem:operator-registry.executorch.et-runtime-namespace.registry-has-op-function-fn/test]
    #[test]
    fn operator_registry_test_basic() {
        let _guard = OPERATOR_REGISTRY_TEST_LOCK.lock().unwrap();
        setup();
        let kernels = [Kernel::new_fallback(c"foo".as_ptr(), noop_kernel)];
        let kernels_span = Span::from_raw_parts(kernels.as_ptr() as *mut Kernel, kernels.len());
        let err = register_kernels(kernels_span);
        assert_eq!(err, Error::Ok);
        assert!(!registry_has_op_function(c"fpp".as_ptr(), Span::new()));
        assert!(registry_has_op_function(c"foo".as_ptr(), Span::new()));
    }

    // PORT-NOTE: `ET_EXPECT_DEATH` death test — registering the same op name +
    // fallback key twice aborts inside `register_kernels`. `runtime_abort`
    // terminates the process rather than unwinding, so `#[should_panic]` cannot
    // catch it; ported and `#[ignore]`d per the established death-test
    // convention. Cannot hold the lock without deadlocking a subsequent run, but
    // the test aborts regardless.
    // [spec:et:sem:operator-registry.executorch.et-runtime-namespace.register-kernels-fn/test]
    #[test]
    #[should_panic]
    #[ignore]
    fn operator_registry_test_register_ops_more_than_once_die() {
        setup();
        let kernels = [
            Kernel::new_fallback(c"foo".as_ptr(), noop_kernel),
            Kernel::new_fallback(c"foo".as_ptr(), noop_kernel),
        ];
        let kernels_span = Span::from_raw_parts(kernels.as_ptr() as *mut Kernel, kernels.len());
        let _ = register_kernels(kernels_span);
    }

    // Building keys from `make_kernel_key`, reading their backing string via
    // `KernelKey::data()`, and comparing with `==`/`!=` exercises `data` (feeds
    // `KernelKey::new(long_contiguous.data())`), `equals` (backing the `==`/`!=`
    // that must distinguish long vs float vs channel-first keys), and its
    // non-fallback `is_fallback()` branch (all keys here are non-fallback, so
    // equality falls through to the byte-for-byte strcmp).
    // [spec:et:sem:operator-registry.executorch.et-runtime-namespace.kernel-key.operator-fn/test]
    // [spec:et:sem:operator-registry.executorch.et-runtime-namespace.kernel-key.data-fn/test]
    // [spec:et:sem:operator-registry.executorch.et-runtime-namespace.kernel-key.equals-fn/test]
    // [spec:et:sem:operator-registry.executorch.et-runtime-namespace.kernel-key.is-fallback-fn/test]
    // [spec:et:sem:operator-registry.executorch.et-runtime-namespace.tensor-meta.tensor-meta-fn/test]
    // [spec:et:sem:operator-registry.executorch.et-runtime-namespace.internal.make-kernel-key-string-fn/test]
    #[test]
    fn operator_registry_test_kernel_key_equals() {
        let _guard = OPERATOR_REGISTRY_TEST_LOCK.lock().unwrap();
        setup();
        let mut buf_long_contiguous = [0 as core::ffi::c_char; K_KERNEL_KEY_BUF_SIZE];
        let err = make_kernel_key(
            &[(ScalarType::Long, vec![0, 1, 2, 3])],
            buf_long_contiguous.as_mut_ptr(),
            buf_long_contiguous.len(),
        );
        assert_eq!(err, Error::Ok);
        let long_contiguous = KernelKey::new(buf_long_contiguous.as_ptr());

        let long_key_1 = KernelKey::new(long_contiguous.data());
        let long_key_2 = KernelKey::new(long_contiguous.data());

        assert_eq!(long_key_1, long_key_2);

        let mut buf_float_contiguous = [0 as core::ffi::c_char; K_KERNEL_KEY_BUF_SIZE];
        let err = make_kernel_key(
            &[(ScalarType::Float, vec![0, 1, 2, 3])],
            buf_float_contiguous.as_mut_ptr(),
            buf_float_contiguous.len(),
        );
        assert_eq!(err, Error::Ok);
        let float_key = KernelKey::new(buf_float_contiguous.as_ptr());

        assert_ne!(long_key_1, float_key);

        let mut buf_channel_first = [0 as core::ffi::c_char; K_KERNEL_KEY_BUF_SIZE];
        let err = make_kernel_key(
            &[(ScalarType::Long, vec![0, 3, 1, 2])],
            buf_channel_first.as_mut_ptr(),
            buf_channel_first.len(),
        );
        assert_eq!(err, Error::Ok);
        let long_key_3 = KernelKey::new(buf_channel_first.as_ptr());

        assert_ne!(long_key_1, long_key_3);
    }

    // [spec:et:sem:operator-registry.executorch.et-runtime-namespace.get-op-function-from-registry-fn/test]
    #[test]
    fn operator_registry_test_get_op_fails_for_long_kernel_key() {
        let _guard = OPERATOR_REGISTRY_TEST_LOCK.lock().unwrap();
        setup();
        // Looking up a way-too-long kernel key should fail with an error.
        // 1000 is a lot of tensors.
        let mut tensors: Vec<(ScalarType, Vec<DimOrderType>)> = Vec::with_capacity(1000);
        for _ in 0..1000 {
            let mut dims: Vec<DimOrderType> = Vec::with_capacity(16);
            for j in 0..16 {
                dims.push(j as DimOrderType);
            }
            tensors.push((
                unsafe { core::mem::transmute::<i8, ScalarType>(10i8) },
                dims,
            ));
        }
        let mut meta: Vec<TensorMeta> = Vec::new();
        for t in &tensors {
            let dim_order = Span::from_raw_parts(t.1.as_ptr() as *mut DimOrderType, t.1.len());
            meta.push(TensorMeta::new(t.0, dim_order));
        }
        let metadata = Span::from_raw_parts(meta.as_ptr() as *mut TensorMeta, meta.len());

        let op = get_op_function_from_registry_2(c"test::not-real".as_ptr(), metadata);
        assert_ne!(ResultExt::error(&op), Error::Ok);
        assert_ne!(ResultExt::error(&op), Error::OperatorMissing);
        // The lookup failed, but not because the operator is missing.
    }

    // Registers a specialized kernel into the global registry and looks it up:
    // this exercises register_kernels_internal (invoked by register_kernels),
    // get_registered_kernels (the span the two-arg lookup searches), and
    // Kernel::new (the specialized-kernel constructor whose stored name/key/op
    // the successful lookup and `val == 100` assertion depend on).
    // [spec:et:sem:operator-registry.executorch.et-runtime-namespace.register-kernels-fn/test]
    // [spec:et:sem:operator-registry.executorch.et-runtime-namespace.register-kernels-internal-fn/test]
    // [spec:et:sem:operator-registry.executorch.et-runtime-namespace.get-registered-kernels-fn/test]
    // [spec:et:sem:operator-registry.executorch.et-runtime-namespace.kernel.kernel-fn/test]
    // [spec:et:sem:operator-registry.executorch.et-runtime-namespace.registry-has-op-function-fn/test]
    // [spec:et:sem:operator-registry.executorch.et-runtime-namespace.get-op-function-from-registry-fn/test]
    #[test]
    fn operator_registry_test_register_kernels() {
        let _guard = OPERATOR_REGISTRY_TEST_LOCK.lock().unwrap();
        setup();
        let mut buf_long_contiguous = [0 as core::ffi::c_char; K_KERNEL_KEY_BUF_SIZE];
        let err = make_kernel_key(
            &[(ScalarType::Long, vec![0, 1, 2, 3])],
            buf_long_contiguous.as_mut_ptr(),
            buf_long_contiguous.len(),
        );
        assert_eq!(err, Error::Ok);
        let key = KernelKey::new(buf_long_contiguous.as_ptr());

        let kernel_1 = Kernel::new(c"test::boo".as_ptr(), key, write_100_kernel);
        let mut kernel_1_arr = [kernel_1];
        let err = register_kernels(Span::from_raw_parts(kernel_1_arr.as_mut_ptr(), 1));
        assert_eq!(err, Error::Ok);

        let dims: [DimOrderType; 4] = [0, 1, 2, 3];
        let dim_order_type = Span::from_raw_parts(dims.as_ptr() as *mut DimOrderType, 4);
        let meta = [TensorMeta::new(ScalarType::Long, dim_order_type)];
        let user_kernel_key = Span::from_raw_parts(meta.as_ptr() as *mut TensorMeta, meta.len());

        // no fallback kernel is registered
        assert!(!registry_has_op_function(
            c"test::boo".as_ptr(),
            Span::new()
        ));
        let fallback_func = get_op_function_from_registry_2(c"test::boo".as_ptr(), Span::new());
        assert_ne!(ResultExt::error(&fallback_func), Error::Ok);

        assert!(registry_has_op_function(
            c"test::boo".as_ptr(),
            user_kernel_key
        ));
        let func = get_op_function_from_registry_2(c"test::boo".as_ptr(), user_kernel_key);
        assert_eq!(ResultExt::error(&func), Error::Ok);

        let mut values = [EValue::from_scalar(Scalar::from_i64(0))];
        let mut kernels: [*mut EValue; 1] = [&mut values[0] as *mut EValue];
        let mut context = null_context();
        (ResultExt::get(&func))(&mut context, Span::from_raw_parts(kernels.as_mut_ptr(), 1));

        let val = values[0].to_scalar().to_i64();
        assert_eq!(val, 100);
    }

    // [spec:et:sem:operator-registry.executorch.et-runtime-namespace.register-kernels-fn/test]
    // [spec:et:sem:operator-registry.executorch.et-runtime-namespace.get-op-function-from-registry-fn/test]
    #[test]
    fn operator_registry_test_register_two_kernels() {
        let _guard = OPERATOR_REGISTRY_TEST_LOCK.lock().unwrap();
        setup();
        let mut buf_long_contiguous = [0 as core::ffi::c_char; K_KERNEL_KEY_BUF_SIZE];
        let err = make_kernel_key(
            &[(ScalarType::Long, vec![0, 1, 2, 3])],
            buf_long_contiguous.as_mut_ptr(),
            buf_long_contiguous.len(),
        );
        assert_eq!(err, Error::Ok);
        let key_1 = KernelKey::new(buf_long_contiguous.as_ptr());

        let mut buf_float_contiguous = [0 as core::ffi::c_char; K_KERNEL_KEY_BUF_SIZE];
        let err = make_kernel_key(
            &[(ScalarType::Float, vec![0, 1, 2, 3])],
            buf_float_contiguous.as_mut_ptr(),
            buf_float_contiguous.len(),
        );
        assert_eq!(err, Error::Ok);
        let key_2 = KernelKey::new(buf_float_contiguous.as_ptr());
        let kernel_1 = Kernel::new(c"test::bar".as_ptr(), key_1, write_100_kernel);
        let kernel_2 = Kernel::new(c"test::bar".as_ptr(), key_2, write_50_kernel);
        let mut kernels_arr = [kernel_1, kernel_2];
        let err = register_kernels(Span::from_raw_parts(
            kernels_arr.as_mut_ptr(),
            kernels_arr.len(),
        ));
        assert_eq!(err, Error::Ok);

        // has both kernels
        let dims: [DimOrderType; 4] = [0, 1, 2, 3];
        let dim_order_type = Span::from_raw_parts(dims.as_ptr() as *mut DimOrderType, 4);
        let meta = [TensorMeta::new(ScalarType::Long, dim_order_type)];
        let user_kernel_key_1 = Span::from_raw_parts(meta.as_ptr() as *mut TensorMeta, meta.len());

        let meta_2 = [TensorMeta::new(ScalarType::Float, dim_order_type)];
        let user_kernel_key_2 =
            Span::from_raw_parts(meta_2.as_ptr() as *mut TensorMeta, meta_2.len());

        // no fallback kernel is registered
        assert!(!registry_has_op_function(
            c"test::bar".as_ptr(),
            Span::new()
        ));
        let fallback_func = get_op_function_from_registry_2(c"test::bar".as_ptr(), Span::new());
        assert_ne!(ResultExt::error(&fallback_func), Error::Ok);

        let mut values = [EValue::from_scalar(Scalar::from_i64(0))];
        let mut evalues: [*mut EValue; 1] = [&mut values[0] as *mut EValue];
        let mut context = null_context();

        // test kernel_1
        assert!(registry_has_op_function(
            c"test::bar".as_ptr(),
            user_kernel_key_1
        ));
        let func_1 = get_op_function_from_registry_2(c"test::bar".as_ptr(), user_kernel_key_1);
        assert_eq!(ResultExt::error(&func_1), Error::Ok);
        (ResultExt::get(&func_1))(&mut context, Span::from_raw_parts(evalues.as_mut_ptr(), 1));

        let val_1 = values[0].to_scalar().to_i64();
        assert_eq!(val_1, 100);

        // test kernel_2
        assert!(registry_has_op_function(
            c"test::bar".as_ptr(),
            user_kernel_key_2
        ));
        let func_2 = get_op_function_from_registry_2(c"test::bar".as_ptr(), user_kernel_key_2);
        assert_eq!(ResultExt::error(&func_2), Error::Ok);
        let mut zero = EValue::from_scalar(Scalar::from_i64(0));
        values[0].assign_move(&mut zero);
        evalues[0] = &mut values[0] as *mut EValue;
        (ResultExt::get(&func_2))(&mut context, Span::from_raw_parts(evalues.as_mut_ptr(), 1));

        let val_2 = values[0].to_scalar().to_i64();
        assert_eq!(val_2, 50);
    }

    // [spec:et:sem:operator-registry.executorch.et-runtime-namespace.get-op-function-from-registry-fn/test]
    #[test]
    fn operator_registry_test_get_op_function_uses_provided_kernel_list() {
        let _guard = OPERATOR_REGISTRY_TEST_LOCK.lock().unwrap();
        setup();
        let mut buf = [0 as core::ffi::c_char; K_KERNEL_KEY_BUF_SIZE];
        let err = make_kernel_key(
            &[(ScalarType::Long, vec![0, 1, 2, 3])],
            buf.as_mut_ptr(),
            buf.len(),
        );
        assert_eq!(err, Error::Ok);
        let long_key = KernelKey::new(buf.as_ptr());

        let kernels = [
            Kernel::new(
                c"test::provided_kernel_list".as_ptr(),
                KernelKey::new_default(),
                write_50_kernel,
            ),
            Kernel::new(
                c"test::provided_kernel_list".as_ptr(),
                long_key,
                write_100_kernel,
            ),
        ];
        let kernels_span = Span::from_raw_parts(kernels.as_ptr() as *mut Kernel, kernels.len());

        let dims: [DimOrderType; 4] = [0, 1, 2, 3];
        let dim_order_type = Span::from_raw_parts(dims.as_ptr() as *mut DimOrderType, dims.len());
        let long_meta = [TensorMeta::new(ScalarType::Long, dim_order_type)];
        let long_kernel_key =
            Span::from_raw_parts(long_meta.as_ptr() as *mut TensorMeta, long_meta.len());

        let run_kernel = |func: OpFunction| -> i64 {
            let mut value = EValue::from_scalar(Scalar::from_i64(0));
            let mut stack: [*mut EValue; 1] = [&mut value as *mut EValue];
            let mut context = null_context();
            func(
                &mut context,
                Span::from_raw_parts(stack.as_mut_ptr(), stack.len()),
            );
            value.to_scalar().to_i64()
        };

        let specialized_func = get_op_function_from_registry(
            c"test::provided_kernel_list".as_ptr(),
            long_kernel_key,
            kernels_span,
        );
        assert_eq!(ResultExt::error(&specialized_func), Error::Ok);
        assert_eq!(run_kernel(*ResultExt::get(&specialized_func)), 100);

        let float_meta = [TensorMeta::new(ScalarType::Float, dim_order_type)];
        let float_kernel_key =
            Span::from_raw_parts(float_meta.as_ptr() as *mut TensorMeta, float_meta.len());
        let fallback_func = get_op_function_from_registry(
            c"test::provided_kernel_list".as_ptr(),
            float_kernel_key,
            kernels_span,
        );
        assert_eq!(ResultExt::error(&fallback_func), Error::Ok);
        assert_eq!(run_kernel(*ResultExt::get(&fallback_func)), 50);
    }

    // [spec:et:sem:operator-registry.executorch.et-runtime-namespace.get-op-function-from-registry-fn/test]
    // [spec:et:sem:operator-registry.executorch.et-runtime-namespace.register-kernels-fn/test]
    #[test]
    fn operator_registry_test_provided_kernel_list_miss_can_fall_back_to_global() {
        let _guard = OPERATOR_REGISTRY_TEST_LOCK.lock().unwrap();
        setup();
        let mut buf = [0 as core::ffi::c_char; K_KERNEL_KEY_BUF_SIZE];
        let err = make_kernel_key(
            &[(ScalarType::Long, vec![0, 1, 2, 3])],
            buf.as_mut_ptr(),
            buf.len(),
        );
        assert_eq!(err, Error::Ok);
        let long_key = KernelKey::new(buf.as_ptr());

        let global_kernel = Kernel::new(
            c"test::provided_kernel_list_global_fallback".as_ptr(),
            KernelKey::new_default(),
            write_50_kernel,
        );
        let mut global_kernel_arr = [global_kernel];
        let err = register_kernels(Span::from_raw_parts(global_kernel_arr.as_mut_ptr(), 1));
        assert_eq!(err, Error::Ok);

        let scoped_kernel = Kernel::new(
            c"test::provided_kernel_list_global_fallback".as_ptr(),
            long_key,
            write_100_kernel,
        );
        let scoped_kernel_arr = [scoped_kernel];
        let scoped_registry = Span::from_raw_parts(scoped_kernel_arr.as_ptr() as *mut Kernel, 1);

        let dims: [DimOrderType; 4] = [0, 1, 2, 3];
        let dim_order_type = Span::from_raw_parts(dims.as_ptr() as *mut DimOrderType, dims.len());
        let long_meta = [TensorMeta::new(ScalarType::Long, dim_order_type)];
        let long_kernel_key =
            Span::from_raw_parts(long_meta.as_ptr() as *mut TensorMeta, long_meta.len());

        let float_meta = [TensorMeta::new(ScalarType::Float, dim_order_type)];
        let float_kernel_key =
            Span::from_raw_parts(float_meta.as_ptr() as *mut TensorMeta, float_meta.len());

        let run_kernel = |func: OpFunction| -> i64 {
            let mut value = EValue::from_scalar(Scalar::from_i64(0));
            let mut stack: [*mut EValue; 1] = [&mut value as *mut EValue];
            let mut context = null_context();
            func(
                &mut context,
                Span::from_raw_parts(stack.as_mut_ptr(), stack.len()),
            );
            value.to_scalar().to_i64()
        };

        let scoped_func = get_op_function_from_registry(
            c"test::provided_kernel_list_global_fallback".as_ptr(),
            long_kernel_key,
            scoped_registry,
        );
        assert_eq!(ResultExt::error(&scoped_func), Error::Ok);
        assert_eq!(run_kernel(*ResultExt::get(&scoped_func)), 100);

        let scoped_miss = get_op_function_from_registry(
            c"test::provided_kernel_list_global_fallback".as_ptr(),
            float_kernel_key,
            scoped_registry,
        );
        assert_eq!(ResultExt::error(&scoped_miss), Error::OperatorMissing);

        let global_func = get_op_function_from_registry_2(
            c"test::provided_kernel_list_global_fallback".as_ptr(),
            float_kernel_key,
        );
        assert_eq!(ResultExt::error(&global_func), Error::Ok);
        assert_eq!(run_kernel(*ResultExt::get(&global_func)), 50);
    }

    // PORT-NOTE: `ET_EXPECT_DEATH` death test — registering the same op name +
    // key twice aborts. See the death-test note on
    // `operator_registry_test_register_ops_more_than_once_die`.
    // [spec:et:sem:operator-registry.executorch.et-runtime-namespace.register-kernels-fn/test]
    #[test]
    #[should_panic]
    #[ignore]
    fn operator_registry_test_double_register_kernels_dies() {
        setup();
        let mut buf_long_contiguous = [0 as core::ffi::c_char; K_KERNEL_KEY_BUF_SIZE];
        let err = make_kernel_key(
            &[(ScalarType::Long, vec![0, 1, 2, 3])],
            buf_long_contiguous.as_mut_ptr(),
            buf_long_contiguous.len(),
        );
        assert_eq!(err, Error::Ok);
        let key = KernelKey::new(buf_long_contiguous.as_ptr());

        let kernel_1 = Kernel::new(c"test::baz".as_ptr(), key, write_100_kernel);
        let kernel_2 = Kernel::new(c"test::baz".as_ptr(), key, write_50_kernel);
        let mut kernels = [kernel_1, kernel_2];
        let _ = register_kernels(Span::from_raw_parts(kernels.as_mut_ptr(), kernels.len()));
    }

    // [spec:et:sem:operator-registry.executorch.et-runtime-namespace.registry-has-op-function-fn/test]
    // [spec:et:sem:operator-registry.executorch.et-runtime-namespace.register-kernels-fn/test]
    #[test]
    fn operator_registry_test_executor_checks_kernel() {
        let _guard = OPERATOR_REGISTRY_TEST_LOCK.lock().unwrap();
        setup();
        let mut buf_long_contiguous = [0 as core::ffi::c_char; K_KERNEL_KEY_BUF_SIZE];
        let err = make_kernel_key(
            &[(ScalarType::Long, vec![0, 1, 2, 3])],
            buf_long_contiguous.as_mut_ptr(),
            buf_long_contiguous.len(),
        );
        assert_eq!(err, Error::Ok);
        let key = KernelKey::new(buf_long_contiguous.as_ptr());

        let kernel_1 = Kernel::new(c"test::qux".as_ptr(), key, write_100_kernel);
        let mut kernel_1_arr = [kernel_1];
        let err = register_kernels(Span::from_raw_parts(kernel_1_arr.as_mut_ptr(), 1));
        assert_eq!(err, Error::Ok);

        let dims: [DimOrderType; 4] = [0, 1, 2, 3];
        let dim_order_type = Span::from_raw_parts(dims.as_ptr() as *mut DimOrderType, 4);
        let meta = [TensorMeta::new(ScalarType::Long, dim_order_type)];
        let user_kernel_key_1 = Span::from_raw_parts(meta.as_ptr() as *mut TensorMeta, meta.len());
        assert!(registry_has_op_function(
            c"test::qux".as_ptr(),
            user_kernel_key_1
        ));

        let dims_channel_first: [DimOrderType; 4] = [0, 3, 1, 2];
        let dim_order_type_channel_first =
            Span::from_raw_parts(dims_channel_first.as_ptr() as *mut DimOrderType, 4);
        let meta_channel_first = [TensorMeta::new(
            ScalarType::Long,
            dim_order_type_channel_first,
        )];
        let user_kernel_key_2 = Span::from_raw_parts(
            meta_channel_first.as_ptr() as *mut TensorMeta,
            meta_channel_first.len(),
        );
        assert!(!registry_has_op_function(
            c"test::qux".as_ptr(),
            user_kernel_key_2
        ));

        let meta_float = [TensorMeta::new(ScalarType::Float, dim_order_type)];
        let user_kernel_key_3 =
            Span::from_raw_parts(meta_float.as_ptr() as *mut TensorMeta, meta_float.len());
        assert!(!registry_has_op_function(
            c"test::qux".as_ptr(),
            user_kernel_key_3
        ));
    }

    // [spec:et:sem:operator-registry.executorch.et-runtime-namespace.registry-has-op-function-fn/test]
    // [spec:et:sem:operator-registry.executorch.et-runtime-namespace.get-op-function-from-registry-fn/test]
    // [spec:et:sem:operator-registry.executorch.et-runtime-namespace.register-kernels-fn/test]
    #[test]
    fn operator_registry_test_executor_uses_kernel() {
        let _guard = OPERATOR_REGISTRY_TEST_LOCK.lock().unwrap();
        setup();
        let mut buf_long_contiguous = [0 as core::ffi::c_char; K_KERNEL_KEY_BUF_SIZE];
        let err = make_kernel_key(
            &[(ScalarType::Long, vec![0, 1, 2, 3])],
            buf_long_contiguous.as_mut_ptr(),
            buf_long_contiguous.len(),
        );
        assert_eq!(err, Error::Ok);
        let key = KernelKey::new(buf_long_contiguous.as_ptr());

        let kernel_1 = Kernel::new(c"test::quux".as_ptr(), key, write_100_kernel);
        let mut kernel_1_arr = [kernel_1];
        let err = register_kernels(Span::from_raw_parts(kernel_1_arr.as_mut_ptr(), 1));
        assert_eq!(err, Error::Ok);

        let dims: [DimOrderType; 4] = [0, 1, 2, 3];
        let dim_order_type = Span::from_raw_parts(dims.as_ptr() as *mut DimOrderType, 4);
        let meta = [TensorMeta::new(ScalarType::Long, dim_order_type)];
        let user_kernel_key_1 = Span::from_raw_parts(meta.as_ptr() as *mut TensorMeta, meta.len());

        assert!(registry_has_op_function(
            c"test::quux".as_ptr(),
            user_kernel_key_1
        ));
        let func = get_op_function_from_registry_2(c"test::quux".as_ptr(), user_kernel_key_1);
        assert_eq!(ResultExt::error(&func), Error::Ok);

        let mut values = [EValue::from_scalar(Scalar::from_i64(0))];
        let mut kernels: [*mut EValue; 1] = [&mut values[0] as *mut EValue];
        let mut context = null_context();
        (ResultExt::get(&func))(&mut context, Span::from_raw_parts(kernels.as_mut_ptr(), 1));

        let val = values[0].to_scalar().to_i64();
        assert_eq!(val, 100);
    }

    // [spec:et:sem:operator-registry.executorch.et-runtime-namespace.registry-has-op-function-fn/test]
    // [spec:et:sem:operator-registry.executorch.et-runtime-namespace.get-op-function-from-registry-fn/test]
    // [spec:et:sem:operator-registry.executorch.et-runtime-namespace.register-kernels-fn/test]
    #[test]
    fn operator_registry_test_executor_uses_fallback_kernel() {
        let _guard = OPERATOR_REGISTRY_TEST_LOCK.lock().unwrap();
        setup();
        let kernel_1 = Kernel::new(
            c"test::corge".as_ptr(),
            KernelKey::new_default(),
            write_100_kernel,
        );
        let mut kernel_1_arr = [kernel_1];
        let err = register_kernels(Span::from_raw_parts(kernel_1_arr.as_mut_ptr(), 1));
        assert_eq!(err, Error::Ok);

        assert!(registry_has_op_function(
            c"test::corge".as_ptr(),
            Span::new()
        ));
        assert!(registry_has_op_function(
            c"test::corge".as_ptr(),
            Span::new()
        ));

        let func = get_op_function_from_registry_2(c"test::corge".as_ptr(), Span::new());
        assert_eq!(ResultExt::error(&func), Error::Ok);

        let mut values = [EValue::from_scalar(Scalar::from_i64(0))];
        let mut kernels: [*mut EValue; 1] = [&mut values[0] as *mut EValue];
        let mut context = null_context();
        (ResultExt::get(&func))(&mut context, Span::from_raw_parts(kernels.as_mut_ptr(), 1));

        let val = values[0].to_scalar().to_i64();
        assert_eq!(val, 100);
    }

    //
    // Tests for the deprecated torch::executor compatibility aliases
    //

    // The `torch::executor` aliases have no dedicated upstream test (they are
    // inline forwarders). This registers a fallback kernel through
    // `torch_executor::register_kernels` (ArrayRef -> Span forward), then checks
    // it via `has_ops_fn`, resolves + invokes it via `get_ops_fn`, and confirms
    // `get_kernels` re-wraps the live registry span (same pointer, same length)
    // as an ArrayRef. Each assertion depends on the alias forwarding faithfully
    // to the `executorch` namespace function.
    // [spec:et:sem:operator-registry.torch.executor.register-kernels-fn/test]
    // [spec:et:sem:operator-registry.torch.executor.has-ops-fn-fn/test]
    // [spec:et:sem:operator-registry.torch.executor.get-ops-fn-fn/test]
    // [spec:et:sem:operator-registry.torch.executor.get-kernels-fn/test]
    #[test]
    fn torch_executor_aliases_forward_to_executorch_namespace() {
        use super::torch_executor;
        use crate::runtime::core::array_ref::ArrayRef;

        let _guard = OPERATOR_REGISTRY_TEST_LOCK.lock().unwrap();
        setup();

        let kernels = [Kernel::new_fallback(
            c"test::grault".as_ptr(),
            write_100_kernel,
        )];
        let err = torch_executor::register_kernels(ArrayRef::from_raw_parts(
            kernels.as_ptr(),
            kernels.len(),
        ));
        assert_eq!(err, Error::Ok);

        // has_ops_fn forwards to registry_has_op_function.
        let empty_meta: ArrayRef<TensorMeta> = ArrayRef::from_raw_parts(core::ptr::null(), 0);
        assert!(torch_executor::has_ops_fn(
            c"test::grault".as_ptr(),
            empty_meta
        ));
        assert!(!torch_executor::has_ops_fn(
            c"test::not-registered".as_ptr(),
            empty_meta
        ));

        // get_ops_fn forwards to get_op_function_from_registry and returns the
        // OpFunction directly; invoke it and confirm it is the write_100 kernel.
        let func = torch_executor::get_ops_fn(c"test::grault".as_ptr(), empty_meta);
        let mut value = EValue::from_scalar(Scalar::from_i64(0));
        let mut stack: [*mut EValue; 1] = [&mut value as *mut EValue];
        let mut context = null_context();
        func(
            &mut context,
            Span::from_raw_parts(stack.as_mut_ptr(), stack.len()),
        );
        assert_eq!(value.to_scalar().to_i64(), 100);

        // get_kernels re-wraps get_registered_kernels() as an ArrayRef with the
        // same pointer and length.
        let registered = super::get_registered_kernels();
        let kernels_ref = torch_executor::get_kernels();
        assert_eq!(kernels_ref.data(), registered.data() as *const Kernel);
        assert_eq!(kernels_ref.size(), registered.size());
    }

    //
    // Port of runtime/kernel/test/operator_registry_max_kernel_num_test.cpp
    //
    // PORT-NOTE: the C++ suite links against
    // `operator_registry_MAX_NUM_KERNELS_TEST_ONLY`, built with a compile-time
    // `MAX_KERNEL_NUM=1`. The Rust port fixes `K_MAX_REGISTERED_KERNELS` at 2000
    // (only the default selective-build fallback is ported; the build-time
    // override is an unresolved cross-module reference — see the
    // `K_MAX_OPERATORS` PORT-NOTE above). `RegisterOneOp` still passes under the
    // larger limit; `RegisterTwoOpsFail` cannot reproduce the limit-of-1 abort,
    // so it is ported and `#[ignore]`d.
    //

    // [spec:et:sem:operator-registry.executorch.et-runtime-namespace.register-kernels-fn/test]
    // [spec:et:sem:operator-registry.executorch.et-runtime-namespace.registry-has-op-function-fn/test]
    #[test]
    fn operator_registry_max_kernel_num_test_register_one_op() {
        let _guard = OPERATOR_REGISTRY_TEST_LOCK.lock().unwrap();
        setup();
        let kernels = [Kernel::new_fallback(c"foo".as_ptr(), noop_kernel)];
        let s1 = register_kernels(Span::from_raw_parts(
            kernels.as_ptr() as *mut Kernel,
            kernels.len(),
        ));
        assert_eq!(s1, Error::Ok);
        assert!(!registry_has_op_function(c"fpp".as_ptr(), Span::new()));
        assert!(registry_has_op_function(c"foo".as_ptr(), Span::new()));
    }

    // PORT-NOTE: `ET_EXPECT_DEATH` expecting the max-kernel-limit abort, which
    // only fires when `MAX_KERNEL_NUM=1`. With the ported limit of 2000 this
    // registers successfully instead of dying, so it is `#[ignore]`d (build-time
    // cfg dependency). See the max-kernel-num PORT-NOTE above.
    // [spec:et:sem:operator-registry.executorch.et-runtime-namespace.register-kernels-fn/test]
    #[test]
    #[should_panic]
    #[ignore]
    fn operator_registry_max_kernel_num_test_register_two_ops_fail() {
        setup();
        let kernels = [
            Kernel::new_fallback(c"foo1".as_ptr(), noop_kernel),
            Kernel::new_fallback(c"foo2".as_ptr(), noop_kernel),
        ];
        let _ = register_kernels(Span::from_raw_parts(
            kernels.as_ptr() as *mut Kernel,
            kernels.len(),
        ));
    }

    //
    // Port of runtime/kernel/test/kernel_double_registration_test.cpp
    //

    // PORT-NOTE: `ET_EXPECT_DEATH` — this suite links against the generated
    // `specialized_kernel_generated_lib`, which registers `aten::add.out` at
    // static-init; re-registering the same op+key then aborts with
    // `RegistrationAlreadyRegistered`. The Rust port has no generated lib
    // registering `aten::add.out` (codegen is a cross-module gap), so the abort
    // would not fire for the right reason. Ported and `#[ignore]`d as a death
    // test plus generated-lib dependency.
    // [spec:et:sem:operator-registry.executorch.et-runtime-namespace.register-kernels-fn/test]
    #[test]
    #[should_panic]
    #[ignore]
    fn kernel_double_registration_test_basic() {
        setup();
        let kernels = [Kernel::new(
            c"aten::add.out".as_ptr(),
            KernelKey::new(c"v1/7;0,1,2,3|7;0,1,2,3|7;0,1,2,3".as_ptr()),
            noop_kernel,
        )];
        let _ = register_kernels(Span::from_raw_parts(
            kernels.as_ptr() as *mut Kernel,
            kernels.len(),
        ));
    }

    //
    // Port of runtime/kernel/test/test_generated_lib_and_aten.cpp
    //

    // PORT-NOTE: depends on the codegen-generated ATen-mode registry
    // (`executorch::runtime::aten::registry_has_op_function` /
    // `get_op_function_from_registry`) plus a `generated_lib` that registers
    // `aten::add.out`. Neither the ATen-mode registry namespace nor the
    // generated lib exists in the Wave-2 Rust port (codegen is a cross-module
    // gap), so this is ported and `#[ignore]`d until they are wired up.
    // [spec:et:sem:operator-registry.executorch.et-runtime-namespace.registry-has-op-function-fn/test]
    // [spec:et:sem:operator-registry.executorch.et-runtime-namespace.get-op-function-from-registry-fn/test]
    #[test]
    #[ignore]
    fn generated_lib_and_aten_test_get_kernels_from_aten_registry() {
        setup();
        // Check if the kernel exists in the ATen registry.
        let has_kernel = registry_has_op_function(c"aten::add.out".as_ptr(), Span::new());
        assert!(
            has_kernel,
            "Kernel 'aten::add.out' not found in ATen registry"
        );

        // Get the kernel from the ATen registry.
        let result = get_op_function_from_registry_2(c"aten::add.out".as_ptr(), Span::new());
        assert_eq!(
            ResultExt::error(&result),
            Error::Ok,
            "Failed to get kernel from ATen registry"
        );
        assert!(
            ResultExt::ok(&result),
            "Kernel function from ATen registry is null"
        );
    }

    //
    // Port of runtime/kernel/test/test_kernel_manual_registration.cpp
    //

    // PORT-NOTE: depends on the codegen-generated
    // `torch::executor::register_all_kernels()` (from `RegisterKernels.h`),
    // which does not exist in the Wave-2 Rust port (codegen is a cross-module
    // gap). Ported and `#[ignore]`d until the generated registration function is
    // available.
    // [spec:et:sem:operator-registry.executorch.et-runtime-namespace.registry-has-op-function-fn/test]
    #[test]
    #[ignore]
    fn kernel_manual_registration_test_manual_register() {
        let _guard = OPERATOR_REGISTRY_TEST_LOCK.lock().unwrap();
        setup();
        // Before registering, we can't find the add operator.
        assert!(!registry_has_op_function(
            c"aten::add.out".as_ptr(),
            Span::new()
        ));

        // Call the generated registration function.
        // PORT-NOTE: `torch::executor::register_all_kernels()` is unavailable; the
        // call is left as a compile-time gap marker. This test is `#[ignore]`d.
        // let result = torch::executor::register_all_kernels();
        // assert_eq!(result, Error::Ok);

        // We can now find the registered add operator.
        // assert!(registry_has_op_function(c"aten::add.out".as_ptr(), Span::new()));

        // We can't find a random other operator.
        // assert!(!registry_has_op_function(c"fpp".as_ptr(), Span::new()));
    }
}
