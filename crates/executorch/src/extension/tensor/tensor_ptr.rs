//! Literal port of extension/tensor/tensor_ptr.cpp + extension/tensor/tensor_ptr.h.

extern crate alloc;

use alloc::sync::Arc;
use alloc::vec::Vec;

use crate::runtime::core::array_ref::ArrayRef;
use crate::runtime::core::device_allocator::get_device_allocator;
use crate::runtime::core::error::Error;
use crate::runtime::core::exec_aten::util::dim_order_util::dim_order_to_stride;
use crate::runtime::core::exec_aten::util::scalar_type_util::{
    CppTypeToScalarType, can_cast, element_size,
};
use crate::runtime::core::exec_aten::util::tensor_util::resize_tensor_same_type;
use crate::runtime::core::portable_type::device::{Device, DeviceType};
use crate::runtime::core::portable_type::scalar_type::ScalarType;
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::core::portable_type::tensor_impl::{
    DimOrderType, SizesType, StridesType, TensorImpl, safe_numel, ssize_t,
};
use crate::runtime::core::result::ResultExt;
use crate::runtime::core::tensor_shape_dynamism::TensorShapeDynamism;

// PORT-NOTE: `ET_CHECK_MSG` (runtime/platform/assert.h) has no ported target
// yet; this local macro mirrors its semantics (fatal abort via the PAL abort
// path), matching the pattern used across the tier-1 modules. Unresolved
// cross-module reference.
macro_rules! et_check_msg {
    ($cond:expr, $($arg:tt)*) => {
        if !($cond) {
            crate::runtime::platform::abort::runtime_abort();
        }
    };
}

// PORT-NOTE: `ET_SWITCH_REALHBBF16_AND_UINT_TYPES` (scalar_type_util.h) is not
// yet ported into scalar_type_util.rs (which owns the switch macros). Its type
// set is the REALHBBF16 set (Byte, Char, Short, Int, Long, Float, Double, Half,
// Bool, BFloat16) plus the unsigned integer types (UInt16, UInt32, UInt64).
// It is reproduced locally here as a full `match` expression (rather than via
// the existing `et_internal_switch_case!` helper, which composes match arms
// through nested macro calls — a shape Rust does not accept in arm position).
// The default arm ports the `ctx.fail(...)` fail-fn behavior, then logs and
// yields `Default::default()` exactly like `et_internal_switch!`. Should move to
// scalar_type_util.rs once that module gains the macro. Unresolved cross-module
// reference.
macro_rules! et_switch_realhbbf16_and_uint_types {
    ($type:expr, $ctx:expr, $name:expr, $ctype_alias:ident, $body:block) => {{
        let _st = $type;
        let et_switch_name = $name;
        let _ = et_switch_name;
        match _st {
            $crate::runtime::core::portable_type::scalar_type::ScalarType::Byte => {
                #[allow(non_camel_case_types, dead_code)]
                type $ctype_alias = u8;
                $body
            }
            $crate::runtime::core::portable_type::scalar_type::ScalarType::Char => {
                #[allow(non_camel_case_types, dead_code)]
                type $ctype_alias = i8;
                $body
            }
            $crate::runtime::core::portable_type::scalar_type::ScalarType::Short => {
                #[allow(non_camel_case_types, dead_code)]
                type $ctype_alias = i16;
                $body
            }
            $crate::runtime::core::portable_type::scalar_type::ScalarType::Int => {
                #[allow(non_camel_case_types, dead_code)]
                type $ctype_alias = i32;
                $body
            }
            $crate::runtime::core::portable_type::scalar_type::ScalarType::Long => {
                #[allow(non_camel_case_types, dead_code)]
                type $ctype_alias = i64;
                $body
            }
            $crate::runtime::core::portable_type::scalar_type::ScalarType::Float => {
                #[allow(non_camel_case_types, dead_code)]
                type $ctype_alias = f32;
                $body
            }
            $crate::runtime::core::portable_type::scalar_type::ScalarType::Double => {
                #[allow(non_camel_case_types, dead_code)]
                type $ctype_alias = f64;
                $body
            }
            $crate::runtime::core::portable_type::scalar_type::ScalarType::Half => {
                #[allow(non_camel_case_types, dead_code)]
                type $ctype_alias = $crate::runtime::core::portable_type::Half;
                $body
            }
            $crate::runtime::core::portable_type::scalar_type::ScalarType::Bool => {
                #[allow(non_camel_case_types, dead_code)]
                type $ctype_alias = bool;
                $body
            }
            $crate::runtime::core::portable_type::scalar_type::ScalarType::BFloat16 => {
                #[allow(non_camel_case_types, dead_code)]
                type $ctype_alias = $crate::runtime::core::portable_type::BFloat16;
                $body
            }
            $crate::runtime::core::portable_type::scalar_type::ScalarType::UInt16 => {
                #[allow(non_camel_case_types, dead_code)]
                type $ctype_alias = u16;
                $body
            }
            $crate::runtime::core::portable_type::scalar_type::ScalarType::UInt32 => {
                #[allow(non_camel_case_types, dead_code)]
                type $ctype_alias = u32;
                $body
            }
            $crate::runtime::core::portable_type::scalar_type::ScalarType::UInt64 => {
                #[allow(non_camel_case_types, dead_code)]
                type $ctype_alias = u64;
                $body
            }
            _ => {
                $ctx.fail($crate::runtime::core::error::Error::InvalidArgument);
                $crate::et_log!(
                    Error,
                    "Unhandled dtype {} for {}",
                    $crate::runtime::core::exec_aten::util::scalar_type_util::to_string(_st),
                    et_switch_name
                );
                ::core::default::Default::default()
            }
        }
    }};
}
pub(crate) use et_switch_realhbbf16_and_uint_types;

// PORT-NOTE: `c10::mul_overflows(a, b, &out)` returns true on overflow and
// writes the wrapped product to `out`. Ported inline via `checked_mul` (same
// pattern as tensor_impl.rs).
fn mul_overflows_usize(a: usize, b: usize, out: &mut usize) -> bool {
    match a.checked_mul(b) {
        Some(product) => {
            *out = product;
            false
        }
        None => {
            *out = a.wrapping_mul(b);
            true
        }
    }
}

// [spec:et:sem:tensor-ptr.executorch.extension.fail-fn]
// PORT-NOTE: The C++ `[[noreturn]] void fail(Error)` local context struct fed to
// ET_SWITCH becomes the `ctx.fail(...)` object the switch machinery invokes on
// an unhandled dtype. The three distinct message strings from the C++ sites are
// preserved by constructing the context with the right message. `fail` aborts
// and never returns.
pub struct FailCtx {
    message: &'static str,
}
impl FailCtx {
    pub fn new(message: &'static str) -> Self {
        FailCtx { message }
    }
    // [spec:et:def:tensor-ptr.executorch.extension.fail-fn]
    pub fn fail(&self, _error: Error) {
        let _ = self.message;
        et_check_msg!(false, "{}", self.message);
    }
}

/// A structure that consolidates the metadata (sizes, dim_order, strides) and
/// the data buffer associated with a Tensor. Since Tensor does not own the
/// memory for these metadata arrays or the data itself, this structure ensures
/// that they are managed together and have the same lifetime as the Tensor.
///
/// PORT-NOTE: The C++ `Storage` co-locates the `TensorImpl` with a `Tensor`
/// member that aliases `&this->tensor_impl`, made stable via `shared_ptr`. Here
/// `TensorPtr` is `Arc<Storage>`, `Storage` is heap-pinned by the Arc, and the
/// aliasing `Tensor` is produced on demand from `&self.tensor_impl` rather than
/// stored as a self-referential field (which Rust cannot express safely). The
/// owning metadata vectors are `Vec`s whose heap buffers stay put across the
/// move into `Storage`, so the raw pointers `TensorImpl` holds into them remain
/// valid. `Storage` is neither `Clone` nor movable-after-publish.
// [spec:et:def:tensor-ptr.executorch.extension.storage]
pub struct Storage {
    tensor_impl: TensorImpl,
    #[allow(dead_code)]
    sizes: Vec<SizesType>,
    #[allow(dead_code)]
    dim_order: Vec<DimOrderType>,
    #[allow(dead_code)]
    strides: Vec<StridesType>,
    deleter: Option<Deleter>,
}

/// PORT-NOTE: `std::function<void(void*)>` maps to a boxed `FnMut(*mut c_void)`.
pub type Deleter = alloc::boxed::Box<dyn FnMut(*mut core::ffi::c_void)>;

impl Storage {
    // [spec:et:def:tensor-ptr.executorch.extension.storage.storage-fn]
    // [spec:et:sem:tensor-ptr.executorch.extension.storage.storage-fn]
    fn new(
        tensor_impl: TensorImpl,
        sizes: Vec<SizesType>,
        dim_order: Vec<DimOrderType>,
        strides: Vec<StridesType>,
        deleter: Option<Deleter>,
    ) -> Self {
        Storage {
            tensor_impl,
            sizes,
            dim_order,
            strides,
            deleter,
        }
    }
}

// [spec:et:def:tensor-ptr.executorch.extension.storage.operator-fn]
// [spec:et:sem:tensor-ptr.executorch.extension.storage.operator-fn]
// PORT-NOTE: C++ deletes copy/move ctors and copy/move assignment on `Storage`
// (self-referential). Rust achieves the same by never deriving `Clone`/`Copy`
// and only ever handing `Storage` out through an `Arc`.
impl Drop for Storage {
    fn drop(&mut self) {
        if let Some(deleter) = self.deleter.as_mut() {
            deleter(self.tensor_impl.mutable_data_typed());
        }
    }
}

/// A smart pointer type for managing the lifecycle of a Tensor.
///
/// PORT-NOTE: C++ `TensorPtr = std::shared_ptr<Tensor>` built via the aliasing
/// constructor over a `shared_ptr<Storage>`. Modeled as `Arc<Storage>`; the
/// aliased `Tensor` is obtained via `tensor()`/`tensor_mut()`.
// [spec:et:def:tensor-ptr.executorch.extension.tensor-ptr] (using TensorPtr)
#[derive(Clone)]
pub struct TensorPtr {
    storage: Arc<Storage>,
}

impl TensorPtr {
    /// Returns the managed `Tensor`, aliasing this storage's `TensorImpl`.
    pub fn tensor(&self) -> Tensor<'_> {
        Tensor::new(&self.storage.tensor_impl as *const TensorImpl as *mut TensorImpl)
    }

    /// Returns the underlying storage handle.
    pub fn storage(&self) -> &Arc<Storage> {
        &self.storage
    }
}

// [spec:et:def:tensor-ptr.executorch.extension.make-tensor-ptr-fn]
// [spec:et:sem:tensor-ptr.executorch.extension.make-tensor-ptr-fn]
pub fn make_tensor_ptr(
    sizes: Vec<SizesType>,
    data: *mut core::ffi::c_void,
    dim_order: Vec<DimOrderType>,
    strides: Vec<StridesType>,
    type_: ScalarType,
    dynamism: TensorShapeDynamism,
    deleter: Option<Deleter>,
    device: Device,
) -> TensorPtr {
    let mut dim_order = dim_order;
    let mut strides = strides;
    let mut sizes = sizes;
    let dim = sizes.len();
    et_check_msg!(
        dim_order.is_empty() || dim_order.len() == dim,
        "dim_order size must match sizes or be empty."
    );
    et_check_msg!(
        strides.is_empty() || strides.len() == dim,
        "strides size must match sizes or be empty."
    );

    if dim_order.is_empty() {
        dim_order.resize(dim, 0);
        for (i, v) in dim_order.iter_mut().enumerate() {
            *v = i as DimOrderType;
        }
        if !strides.is_empty() {
            // std::sort comparator: strides[a] > strides[b] => a before b.
            // PORT-NOTE: C++ uses (non-stable) std::sort; `sort_unstable_by`
            // matches.
            dim_order.sort_unstable_by(|&a, &b| {
                let sa = strides[a as usize];
                let sb = strides[b as usize];
                sb.cmp(&sa)
            });
        }
    }
    let mut computed_strides: Vec<StridesType> = alloc::vec![0; dim];

    let error = unsafe {
        dim_order_to_stride(
            sizes.as_ptr(),
            dim_order.as_ptr(),
            dim,
            computed_strides.as_mut_ptr(),
        )
    };
    et_check_msg!(error == Error::Ok, "Failed to compute strides.");

    if !strides.is_empty() {
        for i in 0..dim {
            et_check_msg!(
                strides[i] == computed_strides[i] || sizes[i] == 1,
                "invalid strides for dim {}: {} != {} while its size is {} != 1",
                i,
                strides[i],
                computed_strides[i],
                sizes[i]
            );
        }
    }

    strides = computed_strides;

    // #ifndef USE_ATEN_LIB
    let tensor_impl = TensorImpl::new(
        type_,
        dim as ssize_t,
        sizes.as_mut_ptr(),
        data,
        dim_order.as_mut_ptr(),
        strides.as_mut_ptr(),
        if dim > 0 {
            dynamism
        } else {
            TensorShapeDynamism::STATIC
        },
        device.type_(),
        device.index(),
    );
    let storage = Arc::new(Storage::new(
        tensor_impl,
        sizes,
        dim_order,
        strides,
        deleter,
    ));
    TensorPtr { storage }
    // #endif // USE_ATEN_LIB
}

/// Convenience overload for the primary factory.
pub fn make_tensor_ptr_simple(
    sizes: Vec<SizesType>,
    data: *mut core::ffi::c_void,
    type_: ScalarType,
    dynamism: TensorShapeDynamism,
    deleter: Option<Deleter>,
    device: Device,
) -> TensorPtr {
    make_tensor_ptr(
        sizes,
        data,
        Vec::new(),
        Vec::new(),
        type_,
        dynamism,
        deleter,
        device,
    )
}

// [spec:et:sem:tensor-ptr.executorch.extension.casted-data-fn] (see below)
/// Template overload: builds a TensorPtr from a `Vec<T>`, deducing the scalar
/// type from `T` and casting to `type_` if it differs.
pub fn make_tensor_ptr_from_vec<T>(
    sizes: Vec<SizesType>,
    data: Vec<T>,
    dim_order: Vec<DimOrderType>,
    strides: Vec<StridesType>,
    type_: ScalarType,
    dynamism: TensorShapeDynamism,
) -> TensorPtr
where
    T: CppTypeToScalarType
        + Copy
        + 'static
        + NumericCast<u8>
        + NumericCast<i8>
        + NumericCast<i16>
        + NumericCast<i32>
        + NumericCast<i64>
        + NumericCast<f32>
        + NumericCast<f64>
        + NumericCast<crate::runtime::core::portable_type::Half>
        + NumericCast<bool>
        + NumericCast<crate::runtime::core::portable_type::BFloat16>
        + NumericCast<u16>
        + NumericCast<u32>
        + NumericCast<u64>,
{
    let deduced_type = <T as CppTypeToScalarType>::VALUE;
    let numel_result = safe_numel(sizes.as_ptr(), sizes.len() as ssize_t);
    et_check_msg!(
        ResultExt::ok(&numel_result),
        "safe_numel failed: {}",
        ResultExt::error(&numel_result) as i32
    );
    et_check_msg!(
        data.len() == *ResultExt::get(&numel_result) as usize,
        "Data size does not match tensor size."
    );
    if type_ != deduced_type {
        et_check_msg!(
            can_cast(deduced_type, type_),
            "Cannot cast deduced type to specified type."
        );
        let mut casted_bytes: usize = 0;
        et_check_msg!(
            !mul_overflows_usize(data.len(), element_size(type_), &mut casted_bytes),
            "casted_data size overflow: {} elements * {} bytes/element",
            data.len(),
            element_size(type_)
        );
        // [spec:et:def:tensor-ptr.executorch.extension.casted-data-fn]
        // [spec:et:sem:tensor-ptr.executorch.extension.casted-data-fn]
        let mut casted_data: Vec<u8> = alloc::vec![0u8; casted_bytes];

        let ctx = FailCtx {
            message: "Unsupported dtype in make_tensor_ptr",
        };

        et_switch_realhbbf16_and_uint_types!(type_, ctx, "make_tensor_ptr", CTYPE, {
            let dst = casted_data.as_mut_ptr() as *mut CTYPE;
            for (idx, val) in data.iter().enumerate() {
                unsafe {
                    *dst.add(idx) = <T as NumericCast<CTYPE>>::numeric_cast(*val);
                }
            }
        });
        // PORT-NOTE: C++ passes `casted_data.data()` (a `std::vector`); on an
        // empty vector that is `nullptr` (libstdc++/libc++), which the
        // `TensorOwningEmptyData` test relies on. Rust's `Vec::as_ptr()` returns
        // a non-null dangling pointer for an empty Vec, so null it explicitly to
        // mirror `std::vector::data()`.
        let raw_data_ptr = if casted_data.is_empty() {
            core::ptr::null_mut()
        } else {
            casted_data.as_ptr() as *mut core::ffi::c_void
        };
        let data_ptr = Arc::new(casted_data);
        return make_tensor_ptr(
            sizes,
            raw_data_ptr,
            dim_order,
            strides,
            type_,
            dynamism,
            Some(alloc::boxed::Box::new(move |_| {
                let _keep = &data_ptr;
            })),
            Device::from_type(DeviceType::CPU),
        );
    }
    // PORT-NOTE: mirror `std::vector::data()` — nullptr for an empty vector (see
    // the cast-branch note above).
    let raw_data_ptr = if data.is_empty() {
        core::ptr::null_mut()
    } else {
        data.as_ptr() as *mut core::ffi::c_void
    };
    let data_ptr = Arc::new(data);
    make_tensor_ptr(
        sizes,
        raw_data_ptr,
        dim_order,
        strides,
        type_,
        dynamism,
        Some(alloc::boxed::Box::new(move |_| {
            let _keep = &data_ptr;
        })),
        Device::from_type(DeviceType::CPU),
    )
}

/// PORT-NOTE: The C++ per-element `static_cast<CTYPE>(val)` inside the
/// ET_SWITCH bodies is a compile-time cast from the source element type to the
/// switch-selected target type. Rust has no numeric `static_cast` over open
/// generics; `NumericCast<To>` supplies the concrete per-pair cast paths used by
/// the switch bodies (mirroring `static_cast`'s truncation/wrap/narrow
/// semantics). The C++ ctype set is {u8, i8, i16, i32, i64, f32, f64, Half,
/// bool, BFloat16, u16, u32, u64}; a full NxN cast matrix is generated below.
/// `Half`/`BFloat16` bridge through `f32` exactly as `c10::Half`'s
/// `static_cast` does. Unresolved cross-module reference: a shared numeric-cast
/// helper would live alongside scalar_type_util once ported.
pub trait NumericCast<To> {
    fn numeric_cast(self) -> To;
}

mod numeric_cast_impls {
    use super::NumericCast;
    use crate::runtime::core::portable_type::{BFloat16, Half};

    // Bridge every ctype to/from f32 and f64 (the value channels used to
    // reconstruct static_cast on Half/BFloat16 and to funnel bool).
    trait ToF32 {
        fn to_f32_bridge(self) -> f32;
    }
    trait ToF64 {
        fn to_f64_bridge(self) -> f64;
    }
    trait FromF32 {
        fn from_f32_bridge(v: f32) -> Self;
    }
    trait FromF64 {
        fn from_f64_bridge(v: f64) -> Self;
    }

    macro_rules! prim_bridge {
        ($($t:ty),*) => {$(
            impl ToF32 for $t { fn to_f32_bridge(self) -> f32 { self as f32 } }
            impl ToF64 for $t { fn to_f64_bridge(self) -> f64 { self as f64 } }
            impl FromF32 for $t { fn from_f32_bridge(v: f32) -> Self { v as $t } }
            impl FromF64 for $t { fn from_f64_bridge(v: f64) -> Self { v as $t } }
        )*};
    }
    prim_bridge!(u8, i8, i16, i32, i64, f32, f64, u16, u32, u64);

    impl ToF32 for bool {
        fn to_f32_bridge(self) -> f32 {
            self as u8 as f32
        }
    }
    impl ToF64 for bool {
        fn to_f64_bridge(self) -> f64 {
            self as u8 as f64
        }
    }
    impl FromF32 for bool {
        fn from_f32_bridge(v: f32) -> Self {
            v != 0.0
        }
    }
    impl FromF64 for bool {
        fn from_f64_bridge(v: f64) -> Self {
            v != 0.0
        }
    }

    impl ToF32 for Half {
        fn to_f32_bridge(self) -> f32 {
            self.to_f32()
        }
    }
    impl ToF64 for Half {
        fn to_f64_bridge(self) -> f64 {
            self.to_f64()
        }
    }
    impl FromF32 for Half {
        fn from_f32_bridge(v: f32) -> Self {
            Half::from_f32(v)
        }
    }
    impl FromF64 for Half {
        fn from_f64_bridge(v: f64) -> Self {
            Half::from_f64(v)
        }
    }

    impl ToF32 for BFloat16 {
        fn to_f32_bridge(self) -> f32 {
            self.to_f32()
        }
    }
    impl ToF64 for BFloat16 {
        fn to_f64_bridge(self) -> f64 {
            self.to_f64()
        }
    }
    impl FromF32 for BFloat16 {
        fn from_f32_bridge(v: f32) -> Self {
            BFloat16::from_f32(v)
        }
    }
    impl FromF64 for BFloat16 {
        fn from_f64_bridge(v: f64) -> Self {
            BFloat16::from_f64(v)
        }
    }

    // Primitive-integer-and-float source -> any target: direct `as` when both
    // are primitive; otherwise bridge through f64 (or bool via != 0). We favor
    // f64 to preserve integer magnitudes up to 2^53; wider ints lose precision
    // exactly as `static_cast<Half/BFloat16>` would through float.
    macro_rules! cast_prim_to {
        ($from:ty => $($to:ty),*) => {$(
            impl NumericCast<$to> for $from {
                fn numeric_cast(self) -> $to { self as $to }
            }
        )*};
    }
    macro_rules! cast_prim {
        ($($from:ty),*) => {$(
            cast_prim_to!($from => u8, i8, i16, i32, i64, f32, f64, u16, u32, u64);
            impl NumericCast<bool> for $from { fn numeric_cast(self) -> bool { self != (0 as $from) } }
            impl NumericCast<Half> for $from { fn numeric_cast(self) -> Half { Half::from_f64(self as f64) } }
            impl NumericCast<BFloat16> for $from { fn numeric_cast(self) -> BFloat16 { BFloat16::from_f64(self as f64) } }
        )*};
    }
    cast_prim!(u8, i8, i16, i32, i64, u16, u32, u64);

    // Float sources: `x != 0.0` for bool, `as` for primitives, bridge for half.
    macro_rules! cast_float {
        ($($from:ty),*) => {$(
            cast_prim_to!($from => u8, i8, i16, i32, i64, f32, f64, u16, u32, u64);
            impl NumericCast<bool> for $from { fn numeric_cast(self) -> bool { self != (0 as $from) } }
            impl NumericCast<Half> for $from { fn numeric_cast(self) -> Half { Half::from_f64(self as f64) } }
            impl NumericCast<BFloat16> for $from { fn numeric_cast(self) -> BFloat16 { BFloat16::from_f64(self as f64) } }
        )*};
    }
    cast_float!(f32, f64);

    // bool source: static_cast<To>(bool) treats true as 1, false as 0.
    macro_rules! cast_bool_to_prim {
        ($($to:ty),*) => {$(
            impl NumericCast<$to> for bool { fn numeric_cast(self) -> $to { self as u8 as $to } }
        )*};
    }
    cast_bool_to_prim!(u8, i8, i16, i32, i64, f32, f64, u16, u32, u64);
    impl NumericCast<bool> for bool {
        fn numeric_cast(self) -> bool {
            self
        }
    }
    impl NumericCast<Half> for bool {
        fn numeric_cast(self) -> Half {
            Half::from_f32(self as u8 as f32)
        }
    }
    impl NumericCast<BFloat16> for bool {
        fn numeric_cast(self) -> BFloat16 {
            BFloat16::from_f32(self as u8 as f32)
        }
    }

    // Half / BFloat16 sources: value-bridge through f32 (mirrors c10::Half).
    macro_rules! cast_half_src {
        ($from:ty) => {
            impl NumericCast<u8> for $from {
                fn numeric_cast(self) -> u8 {
                    self.to_f32() as u8
                }
            }
            impl NumericCast<i8> for $from {
                fn numeric_cast(self) -> i8 {
                    self.to_f32() as i8
                }
            }
            impl NumericCast<i16> for $from {
                fn numeric_cast(self) -> i16 {
                    self.to_f32() as i16
                }
            }
            impl NumericCast<i32> for $from {
                fn numeric_cast(self) -> i32 {
                    self.to_f32() as i32
                }
            }
            impl NumericCast<i64> for $from {
                fn numeric_cast(self) -> i64 {
                    self.to_f64() as i64
                }
            }
            impl NumericCast<u16> for $from {
                fn numeric_cast(self) -> u16 {
                    self.to_f32() as u16
                }
            }
            impl NumericCast<u32> for $from {
                fn numeric_cast(self) -> u32 {
                    self.to_f64() as u32
                }
            }
            impl NumericCast<u64> for $from {
                fn numeric_cast(self) -> u64 {
                    self.to_f64() as u64
                }
            }
            impl NumericCast<f32> for $from {
                fn numeric_cast(self) -> f32 {
                    self.to_f32()
                }
            }
            impl NumericCast<f64> for $from {
                fn numeric_cast(self) -> f64 {
                    self.to_f64()
                }
            }
            impl NumericCast<bool> for $from {
                fn numeric_cast(self) -> bool {
                    self.to_f32() != 0.0
                }
            }
        };
    }
    cast_half_src!(Half);
    cast_half_src!(BFloat16);
    impl NumericCast<Half> for Half {
        fn numeric_cast(self) -> Half {
            self
        }
    }
    impl NumericCast<BFloat16> for BFloat16 {
        fn numeric_cast(self) -> BFloat16 {
            self
        }
    }
    impl NumericCast<BFloat16> for Half {
        fn numeric_cast(self) -> BFloat16 {
            BFloat16::from_f32(self.to_f32())
        }
    }
    impl NumericCast<Half> for BFloat16 {
        fn numeric_cast(self) -> Half {
            Half::from_f32(self.to_f32())
        }
    }

    // PORT-NOTE: identity same-type complex copy for `_to_dim_order_copy_impl<
    // CTYPE, CTYPE>` on complex dtypes (`ET_SWITCH_COMPLEXH_TYPES`), which copies
    // element-wise with no value conversion.
    use crate::runtime::core::portable_type::{Complex, ComplexDouble, ComplexFloat, ComplexHalf};
    impl NumericCast<ComplexHalf> for ComplexHalf {
        fn numeric_cast(self) -> ComplexHalf {
            self
        }
    }
    impl NumericCast<ComplexFloat> for ComplexFloat {
        fn numeric_cast(self) -> ComplexFloat {
            self
        }
    }
    impl NumericCast<ComplexDouble> for ComplexDouble {
        fn numeric_cast(self) -> ComplexDouble {
            self
        }
    }
    #[allow(dead_code)]
    fn _use_complex_cast(a: Complex<f32>) -> Complex<f32> {
        a.numeric_cast()
    }

    // Silence unused-trait warnings for the bridge scaffolding.
    #[allow(dead_code)]
    fn _use_bridges() {
        let _ = 0u8.to_f32_bridge();
        let _ = 0u8.to_f64_bridge();
        let _ = <u8 as FromF32>::from_f32_bridge(0.0);
        let _ = <u8 as FromF64>::from_f64_bridge(0.0);
    }
}

// [spec:et:def:tensor-ptr.executorch.extension.make-tensor-ptr-fn] (raw-buffer overload)
/// Raw-buffer overload: manages a `Vec<u8>` and interprets it as `type_`.
pub fn make_tensor_ptr_from_bytes(
    sizes: Vec<SizesType>,
    data: Vec<u8>,
    dim_order: Vec<DimOrderType>,
    strides: Vec<StridesType>,
    type_: ScalarType,
    dynamism: TensorShapeDynamism,
) -> TensorPtr {
    let numel_result = safe_numel(sizes.as_ptr(), sizes.len() as ssize_t);
    et_check_msg!(
        ResultExt::ok(&numel_result),
        "safe_numel failed: {}",
        ResultExt::error(&numel_result) as i32
    );
    let numel: ssize_t = *ResultExt::get(&numel_result);
    let mut nbytes: usize = 0;
    et_check_msg!(
        !mul_overflows_usize(numel as usize, element_size(type_), &mut nbytes),
        "Overflow computing nbytes: numel={} element_size={}",
        numel,
        element_size(type_)
    );
    et_check_msg!(
        data.len() == nbytes,
        "Data size ({}) does not match tensor size ({}).",
        data.len(),
        nbytes
    );
    let data_ptr = data.as_ptr() as *mut core::ffi::c_void;
    let data_keep = Arc::new(data);
    make_tensor_ptr(
        sizes,
        data_ptr,
        dim_order,
        strides,
        type_,
        dynamism,
        // Data is moved into the deleter and is destroyed together with Storage.
        Some(alloc::boxed::Box::new(move |_| {
            let _keep = &data_keep;
        })),
        Device::from_type(DeviceType::CPU),
    )
}

// [spec:et:def:tensor-ptr.executorch.extension.clone-tensor-ptr-fn]
// [spec:et:sem:tensor-ptr.executorch.extension.clone-tensor-ptr-fn]
pub fn clone_tensor_ptr(tensor: &Tensor, type_: ScalarType) -> TensorPtr {
    // #ifndef USE_ATEN_LIB
    et_check_msg!(
        tensor.device_type() == DeviceType::CPU,
        "clone_tensor_ptr only supports CPU tensors; use clone_tensor_ptr_to with a CPU target first."
    );
    let sizes: Vec<SizesType> = arrayref_to_vec(&tensor.sizes());
    // #ifndef USE_ATEN_LIB
    let dim_order: Vec<DimOrderType> = arrayref_to_vec(&tensor.dim_order());
    // #endif // USE_ATEN_LIB
    let strides: Vec<StridesType> = arrayref_to_vec(&tensor.strides());
    // PORT-NOTE: the default is unconditionally overwritten in the portable
    // build below (the C++ `#ifndef USE_ATEN_LIB` branch); the dead initial
    // assignment is kept for literal correspondence.
    #[allow(unused_assignments)]
    let mut dynamism = TensorShapeDynamism::DYNAMIC_BOUND;
    // #ifndef USE_ATEN_LIB
    dynamism = tensor.shape_dynamism();
    // #endif // USE_ATEN_LIB
    let tensor_data = tensor.const_data_ptr_typed();
    if tensor_data.is_null() {
        return make_tensor_ptr(
            sizes,
            core::ptr::null_mut(),
            dim_order,
            strides,
            type_,
            dynamism,
            None,
            Device::from_type(DeviceType::CPU),
        );
    }
    let tensor_type = tensor.scalar_type();
    if tensor_type == type_ {
        let nbytes = tensor.nbytes();
        let bytes: Vec<u8> =
            unsafe { core::slice::from_raw_parts(tensor_data as *const u8, nbytes).to_vec() };
        return make_tensor_ptr_from_bytes(sizes, bytes, dim_order, strides, tensor_type, dynamism);
    }
    et_check_msg!(
        can_cast(tensor_type, type_),
        "Cannot cast tensor type to desired type."
    );
    let tensor_numel = tensor.numel() as usize;
    let mut clone_nbytes: usize = 0;
    et_check_msg!(
        !mul_overflows_usize(tensor_numel, element_size(type_), &mut clone_nbytes),
        "Overflow computing clone nbytes: numel={} element_size={}",
        tensor_numel,
        element_size(type_)
    );
    let mut data: Vec<u8> = alloc::vec![0u8; clone_nbytes];

    let ctx = FailCtx {
        message: "Unsupported dtype in clone_tensor_ptr",
    };

    et_switch_realhbbf16_and_uint_types!(
        tensor_type,
        ctx,
        "clone_tensor_ptr_cast_from",
        CtypeFrom,
        {
            let tensor_data_ptr = tensor_data as *const CtypeFrom;
            et_switch_realhbbf16_and_uint_types!(
                type_,
                ctx,
                "clone_tensor_ptr_cast_to",
                CtypeTo,
                {
                    let data_ptr = data.as_mut_ptr() as *mut CtypeTo;
                    for i in 0..tensor_numel {
                        unsafe {
                            *data_ptr.add(i) = <CtypeFrom as NumericCast<CtypeTo>>::numeric_cast(
                                *tensor_data_ptr.add(i),
                            );
                        }
                    }
                }
            );
        }
    );
    make_tensor_ptr_from_bytes(sizes, data, dim_order, strides, type_, dynamism)
}

// [spec:et:def:tensor-ptr.executorch.extension.resize-tensor-ptr-fn]
// [spec:et:sem:tensor-ptr.executorch.extension.resize-tensor-ptr-fn]
// [spec:et:def:tensor-ptr.executorch.extension.runtime.error-resize-tensor-ptr-fn]
// [spec:et:sem:tensor-ptr.executorch.extension.runtime.error-resize-tensor-ptr-fn]
#[must_use]
pub fn resize_tensor_ptr(tensor: &TensorPtr, sizes: &[SizesType]) -> Error {
    resize_tensor_same_type(
        &tensor.tensor(),
        ArrayRef::from_raw_parts(sizes.as_ptr(), sizes.len()),
    )
}

// ---- Device tensor helper ----
//
// This helper relies on the ExecuTorch DeviceAllocator and the portable tensor
// metadata APIs, which have no equivalent in USE_ATEN_LIB builds.

// #ifndef USE_ATEN_LIB
// [spec:et:def:tensor-ptr.executorch.extension.clone-tensor-ptr-to-fn]
// [spec:et:sem:tensor-ptr.executorch.extension.clone-tensor-ptr-to-fn]
pub fn clone_tensor_ptr_to(tensor: &TensorPtr, target: Device) -> TensorPtr {
    let src_tensor = tensor.tensor();
    let source = src_tensor.device();
    et_check_msg!(
        !(source.is_cpu() && target.is_cpu()),
        "clone_tensor_ptr_to does not copy CPU-to-CPU; use clone_tensor_ptr."
    );
    et_check_msg!(
        source.is_cpu() || target.is_cpu(),
        "Device-to-device copy is not supported; route through CPU."
    );

    let nbytes = src_tensor.nbytes();
    let src_data = src_tensor.const_data_ptr_typed();
    et_check_msg!(!src_data.is_null(), "Source tensor has no data.");

    // Whichever end is not CPU provides the allocator.
    let device = if target.is_cpu() { source } else { target };
    let allocator = get_device_allocator(device.type_());
    et_check_msg!(
        !allocator.is_null(),
        "No device allocator registered for device type {}",
        device.type_() as i32
    );
    let allocator = unsafe { &mut *allocator };

    let sizes: Vec<SizesType> = arrayref_to_vec(&src_tensor.sizes());
    let dim_order: Vec<DimOrderType> = arrayref_to_vec(&src_tensor.dim_order());
    let strides: Vec<StridesType> = arrayref_to_vec(&src_tensor.strides());

    if target.is_cpu() {
        let mut cpu_data: Vec<u8> = alloc::vec![0u8; nbytes];
        let err = allocator.copy_device_to_host(
            cpu_data.as_mut_ptr() as *mut core::ffi::c_void,
            src_data,
            nbytes,
            source.index(),
        );
        et_check_msg!(
            err == Error::Ok,
            "Device-to-host copy failed: error {}",
            err as i32
        );
        return make_tensor_ptr_from_bytes(
            sizes,
            cpu_data,
            dim_order,
            strides,
            src_tensor.scalar_type(),
            src_tensor.shape_dynamism(),
        );
    }

    let result = allocator.allocate(
        nbytes,
        target.index(),
        <dyn crate::runtime::core::device_allocator::DeviceAllocator>::K_DEFAULT_ALIGNMENT,
    );
    et_check_msg!(
        ResultExt::ok(&result),
        "Failed to allocate device memory: error {}",
        ResultExt::error(&result) as i32
    );
    let device_data = *ResultExt::get(&result);
    let err = allocator.copy_host_to_device(device_data, src_data, nbytes, target.index());
    et_check_msg!(
        err == Error::Ok,
        "Host-to-device copy failed: error {}",
        err as i32
    );
    let allocator_ptr: *mut (
        dyn crate::runtime::core::device_allocator::DeviceAllocator + 'static
    ) = allocator;
    make_tensor_ptr(
        sizes,
        device_data,
        dim_order,
        strides,
        src_tensor.scalar_type(),
        src_tensor.shape_dynamism(),
        Some(alloc::boxed::Box::new(
            move |ptr: *mut core::ffi::c_void| {
                unsafe { (*allocator_ptr).deallocate(ptr, target.index()) };
            },
        )),
        target,
    )
}
// #endif // USE_ATEN_LIB

// PORT-NOTE: helper for the repeated `std::vector<T>(arr.begin(), arr.end())`
// metadata copies from an `ArrayRef<T>`.
fn arrayref_to_vec<T: Copy>(arr: &ArrayRef<T>) -> Vec<T> {
    let mut v = Vec::with_capacity(arr.size());
    for i in 0..arr.size() {
        v.push(*arr.at(i));
    }
    v
}

#[cfg(test)]
mod tests {
    // Literal port of extension/tensor/test/tensor_ptr_test.cpp
    // (`TensorPtrTest` fixture).
    //
    // PORT-NOTE (missing wave-2 surface): the C++ `tensor_ptr.h` declares a large
    // family of `make_tensor_ptr` convenience overloads. The wave-2 Rust port
    // implements only a subset:
    //   * `make_tensor_ptr` (full: sizes, raw ptr, dim_order, strides, type,
    //      dynamism, deleter, device) and `make_tensor_ptr_simple`;
    //   * `make_tensor_ptr_from_vec<T>` (owning `Vec<T>`, deduced/cast type);
    //   * `make_tensor_ptr_from_bytes` (owning `Vec<u8>` raw buffer);
    //   * `clone_tensor_ptr(&Tensor, type)` (2-arg) and `resize_tensor_ptr`.
    // NOT ported (no Rust surface to bind to):
    //   * `make_tensor_ptr(T value)` single-scalar overload;
    //   * `make_tensor_ptr(vector<T>|initializer_list<T>[, type])` data-only
    //      (rank-1 deducing) overloads;
    //   * the aliasing view overloads `make_tensor_ptr(const Tensor&, sizes,
    //      dim_order, strides, deleter)` and `make_tensor_ptr(const TensorPtr&,
    //      ...)`;
    //   * the single-arg `clone_tensor_ptr(tensor)` (same-type) convenience.
    // Every C++ case that binds ONLY to one of the un-ported overloads is recorded
    // in the `PORT-NOTE: unportable ...` comment blocks below rather than ported,
    // to be added once those overloads land in tensor_ptr.rs. Cases that also (or
    // instead) exercise a ported overload ARE ported here.
    use super::*;
    use crate::runtime::core::exec_aten::util::scalar_type_util::element_size;
    use crate::runtime::core::portable_type::device::DeviceType;
    use core::sync::atomic::{AtomicBool, Ordering};

    fn setup() {
        crate::runtime::platform::runtime::runtime_init();
    }

    fn cpu() -> Device {
        Device::from_type(DeviceType::CPU)
    }

    // `make_tensor_ptr(sizes, raw_ptr, type)` raw-pointer convenience (default
    // DYNAMIC_BOUND, no deleter, CPU).
    fn mk_raw(sizes: Vec<SizesType>, data: *mut core::ffi::c_void, type_: ScalarType) -> TensorPtr {
        make_tensor_ptr_simple(
            sizes,
            data,
            type_,
            TensorShapeDynamism::DYNAMIC_BOUND,
            None,
            cpu(),
        )
    }

    // [spec:et:sem:tensor-ptr.executorch.extension.make-tensor-ptr-fn/test]
    // make_tensor_ptr publishes the tensor through Storage::new, which stores
    // the metadata Vecs the TensorImpl aliases; the shape/data assertions below
    // fail if that construction is wrong:
    // [spec:et:sem:tensor-ptr.executorch.extension.storage.storage-fn/test]
    #[test]
    fn tensor_ptr_test_scalar_tensor_creation() {
        setup();
        let mut scalar_data: f32 = 3.14;
        let tensor = mk_raw(
            alloc::vec![],
            &mut scalar_data as *mut f32 as *mut core::ffi::c_void,
            ScalarType::Float,
        );
        let t = tensor.tensor();
        assert_eq!(t.numel(), 1);
        assert_eq!(t.dim(), 0);
        assert_eq!(t.sizes().size(), 0);
        assert_eq!(t.strides().size(), 0);
        assert_eq!(t.const_data_ptr::<f32>(), &scalar_data as *const f32);
        assert_eq!(unsafe { *t.const_data_ptr::<f32>().add(0) }, 3.14f32);
    }

    // [spec:et:sem:tensor-ptr.executorch.extension.make-tensor-ptr-fn/test]
    // [spec:et:sem:tensor-ptr.executorch.extension.casted-data-fn/test]
    #[test]
    fn tensor_ptr_test_scalar_tensor_owning_data() {
        setup();
        let tensor = make_tensor_ptr_from_vec(
            alloc::vec![],
            alloc::vec![3.14f32],
            Vec::new(),
            Vec::new(),
            ScalarType::Float,
            TensorShapeDynamism::DYNAMIC_BOUND,
        );
        let t = tensor.tensor();
        assert_eq!(t.numel(), 1);
        assert_eq!(t.dim(), 0);
        assert_eq!(t.sizes().size(), 0);
        assert_eq!(t.strides().size(), 0);
        assert_eq!(unsafe { *t.const_data_ptr::<f32>().add(0) }, 3.14f32);
    }

    // PORT-NOTE: unportable — `ScalarTensorSingleValueCreation` binds solely to
    // the `make_tensor_ptr(T value)` single-scalar overload (not ported). Records
    // Float/Int32/Double/Int64 single-value construction + dtype deduction.

    // [spec:et:sem:tensor-ptr.executorch.extension.make-tensor-ptr-fn/test]
    #[test]
    fn tensor_ptr_test_create_tensor_with_strides_and_dim_order() {
        setup();
        let mut data: [f32; 20] = [0.0; 20];
        data[0] = 2.0;
        let tensor = make_tensor_ptr(
            alloc::vec![4, 5],
            data.as_mut_ptr() as *mut core::ffi::c_void,
            alloc::vec![0, 1],
            alloc::vec![5, 1],
            ScalarType::Float,
            TensorShapeDynamism::DYNAMIC_BOUND,
            None,
            cpu(),
        );
        let t = tensor.tensor();
        assert_eq!(t.dim(), 2);
        assert_eq!(t.size(0), 4);
        assert_eq!(t.size(1), 5);
        assert_eq!(*t.strides().at(0), 5);
        assert_eq!(*t.strides().at(1), 1);
        assert_eq!(t.const_data_ptr::<f32>(), data.as_ptr());
        assert_eq!(unsafe { *t.const_data_ptr::<f32>().add(0) }, 2.0);
    }

    // [spec:et:sem:tensor-ptr.executorch.extension.make-tensor-ptr-fn/test]
    #[test]
    fn tensor_ptr_test_tensor_sharing_impl() {
        setup();
        let mut data: [f32; 20] = [0.0; 20];
        data[0] = 2.0;
        let tensor1 = mk_raw(
            alloc::vec![4, 5],
            data.as_mut_ptr() as *mut core::ffi::c_void,
            ScalarType::Float,
        );
        // C++ `auto tensor2 = tensor1;` copies the shared_ptr; Rust clones the Arc.
        let tensor2 = tensor1.clone();
        // Both wrap the same Storage / TensorImpl.
        assert_eq!(
            Arc::as_ptr(tensor1.storage()),
            Arc::as_ptr(tensor2.storage())
        );
        assert_eq!(
            tensor1.tensor().unsafe_get_tensor_impl(),
            tensor2.tensor().unsafe_get_tensor_impl()
        );
    }

    // PORT-NOTE: `TensorLifetime` pins the default-constructed `TensorPtr tensor;`
    // (== nullptr) then reassignment. The ported `TensorPtr` is `Arc<Storage>`
    // with no null/default state (the C++ `nullptr` shared_ptr has no analog), so
    // the `EXPECT_EQ(tensor, nullptr)` half is unportable. The lifetime half — a
    // tensor built from a stack buffer in an inner scope, read after the buffer's
    // C++ scope — relies on the raw pointer outliving its `float data[20]`, which
    // in Rust would be a use-after-free; not ported.

    // [spec:et:sem:tensor-ptr.executorch.extension.make-tensor-ptr-fn/test]
    #[test]
    fn tensor_ptr_test_tensor_with_zero_dimension_and_elements() {
        setup();
        let mut data: [f32; 20] = [0.0; 20];
        data[0] = 2.0;
        let tensor = mk_raw(
            alloc::vec![],
            data.as_mut_ptr() as *mut core::ffi::c_void,
            ScalarType::Float,
        );
        assert_eq!(tensor.tensor().dim(), 0);
        assert_eq!(tensor.tensor().numel(), 1);
        let tensor = mk_raw(
            alloc::vec![0, 5],
            data.as_mut_ptr() as *mut core::ffi::c_void,
            ScalarType::Float,
        );
        assert_eq!(tensor.tensor().dim(), 2);
        assert_eq!(tensor.tensor().numel(), 0);
    }

    // [spec:et:sem:tensor-ptr.executorch.extension.resize-tensor-ptr-fn/test]
    #[test]
    fn tensor_ptr_test_tensor_resize() {
        setup();
        let mut data: [f32; 20] = [0.0; 20];
        data[0] = 2.0;
        let tensor = make_tensor_ptr(
            alloc::vec![4, 5],
            data.as_mut_ptr() as *mut core::ffi::c_void,
            Vec::new(),
            Vec::new(),
            ScalarType::Float,
            TensorShapeDynamism::DYNAMIC_UNBOUND,
            None,
            cpu(),
        );
        assert_eq!(resize_tensor_ptr(&tensor, &[5, 4]), Error::Ok);
        assert_eq!(tensor.tensor().size(0), 5);
        assert_eq!(tensor.tensor().size(1), 4);
    }

    // [spec:et:sem:tensor-ptr.executorch.extension.make-tensor-ptr-fn/test]
    #[test]
    fn tensor_ptr_test_tensor_data_access() {
        setup();
        let mut data: [f32; 6] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = mk_raw(
            alloc::vec![2, 3],
            data.as_mut_ptr() as *mut core::ffi::c_void,
            ScalarType::Float,
        );
        assert_eq!(
            unsafe { *tensor.tensor().const_data_ptr::<f32>().add(0) },
            1.0
        );
        assert_eq!(
            unsafe { *tensor.tensor().const_data_ptr::<f32>().add(5) },
            6.0
        );
        unsafe {
            *tensor.tensor().mutable_data_ptr::<f32>().add(0) = 10.0;
        }
        assert_eq!(
            unsafe { *tensor.tensor().const_data_ptr::<f32>().add(0) },
            10.0
        );
    }

    // [spec:et:sem:tensor-ptr.executorch.extension.make-tensor-ptr-fn/test]
    #[test]
    fn tensor_ptr_test_tensor_with_custom_data_deleter() {
        setup();
        static DELETER_CALLED: AtomicBool = AtomicBool::new(false);
        DELETER_CALLED.store(false, Ordering::SeqCst);
        let data: *mut f32 = Box::into_raw(alloc::vec![0.0f32; 20].into_boxed_slice()) as *mut f32;
        let tensor = make_tensor_ptr(
            alloc::vec![4, 5],
            data as *mut core::ffi::c_void,
            Vec::new(),
            Vec::new(),
            ScalarType::Float,
            TensorShapeDynamism::DYNAMIC_BOUND,
            Some(alloc::boxed::Box::new(
                move |ptr: *mut core::ffi::c_void| {
                    DELETER_CALLED.store(true, Ordering::SeqCst);
                    unsafe {
                        drop(Box::from_raw(core::slice::from_raw_parts_mut(
                            ptr as *mut f32,
                            20,
                        )));
                    }
                },
            )),
            cpu(),
        );
        drop(tensor);
        assert!(DELETER_CALLED.load(Ordering::SeqCst));
    }

    // [spec:et:sem:tensor-ptr.executorch.extension.make-tensor-ptr-fn/test]
    #[test]
    fn tensor_ptr_test_tensor_manages_moved_vector() {
        setup();
        static DELETER_CALLED: AtomicBool = AtomicBool::new(false);
        DELETER_CALLED.store(false, Ordering::SeqCst);
        let data: Vec<f32> = alloc::vec![3.0f32; 20];
        let data_ptr = data.as_ptr() as *mut f32;
        let tensor = make_tensor_ptr(
            alloc::vec![4, 5],
            data_ptr as *mut core::ffi::c_void,
            Vec::new(),
            Vec::new(),
            ScalarType::Float,
            TensorShapeDynamism::DYNAMIC_BOUND,
            Some(alloc::boxed::Box::new(move |_| {
                let _moved_data = &data;
                DELETER_CALLED.store(true, Ordering::SeqCst);
            })),
            cpu(),
        );
        assert_eq!(tensor.tensor().data_ptr::<f32>(), data_ptr);
        drop(tensor);
        assert!(DELETER_CALLED.load(Ordering::SeqCst));
    }

    // [spec:et:sem:tensor-ptr.executorch.extension.make-tensor-ptr-fn/test]
    #[test]
    fn tensor_ptr_test_tensor_deleter_releases_captured_shared_ptr() {
        setup();
        static DELETER_CALLED: AtomicBool = AtomicBool::new(false);
        DELETER_CALLED.store(false, Ordering::SeqCst);
        let data_ptr: Arc<[f32]> = Arc::from(alloc::vec![0.0f32; 10]);
        let raw = data_ptr.as_ptr() as *mut f32;
        let captured = data_ptr.clone();
        let tensor = make_tensor_ptr(
            alloc::vec![4, 5],
            raw as *mut core::ffi::c_void,
            Vec::new(),
            Vec::new(),
            ScalarType::Float,
            TensorShapeDynamism::DYNAMIC_BOUND,
            Some(alloc::boxed::Box::new(move |_| {
                let _keep = &captured;
                DELETER_CALLED.store(true, Ordering::SeqCst);
            })),
            cpu(),
        );
        assert_eq!(Arc::strong_count(&data_ptr), 2);
        drop(tensor);
        assert!(DELETER_CALLED.load(Ordering::SeqCst));
        assert_eq!(Arc::strong_count(&data_ptr), 1);
    }

    // [spec:et:sem:tensor-ptr.executorch.extension.casted-data-fn/test]
    // make_tensor_ptr_from_vec calls safe_numel(sizes) and requires
    // data.len() == numel; a [2,5] shape over a 10-element vec only constructs
    // if safe_numel computes the product (10) correctly.
    // [spec:et:sem:exec-aten.executorch.aten.safe-numel-fn/test]
    #[test]
    fn tensor_ptr_test_tensor_owning_data() {
        setup();
        let tensor = make_tensor_ptr_from_vec(
            alloc::vec![2, 5],
            alloc::vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            alloc::vec![1, 0],
            alloc::vec![1, 2],
            ScalarType::Float,
            TensorShapeDynamism::DYNAMIC_BOUND,
        );
        let t = tensor.tensor();
        assert_eq!(t.dim(), 2);
        assert_eq!(t.size(0), 2);
        assert_eq!(t.size(1), 5);
        assert_eq!(*t.strides().at(0), 1);
        assert_eq!(*t.strides().at(1), 2);
        assert_eq!(unsafe { *t.const_data_ptr::<f32>().add(0) }, 1.0);
        assert_eq!(unsafe { *t.const_data_ptr::<f32>().add(9) }, 10.0);
    }

    // [spec:et:sem:tensor-ptr.executorch.extension.casted-data-fn/test]
    #[test]
    fn tensor_ptr_test_tensor_owning_empty_data() {
        setup();
        let tensor = make_tensor_ptr_from_vec(
            alloc::vec![0, 5],
            Vec::<f32>::new(),
            Vec::new(),
            Vec::new(),
            ScalarType::Float,
            TensorShapeDynamism::DYNAMIC_BOUND,
        );
        let t = tensor.tensor();
        assert_eq!(t.dim(), 2);
        assert_eq!(t.size(0), 0);
        assert_eq!(t.size(1), 5);
        assert_eq!(*t.strides().at(0), 5);
        assert_eq!(*t.strides().at(1), 1);
        assert!(t.data_ptr::<f32>().is_null());
        assert_eq!(t.scalar_type(), ScalarType::Float);
    }

    // PORT-NOTE: unportable — the data-only overloads. `TensorDataOnly`,
    // `TensorDataOnlyDoubleType`, `TensorDataOnlyInt32Type`,
    // `TensorDataOnlyInt64Type`, `TensorDataOnlyUint8Type`,
    // `TensorDataOnlyUInt16Type`, `TensorDataOnlyUInt32Type`,
    // `TensorDataOnlyUInt64Type` all bind solely to
    // `make_tensor_ptr(vector<T>)` (rank-1, deduced dtype), not ported.

    // [spec:et:sem:tensor-ptr.executorch.extension.casted-data-fn/test]
    #[test]
    fn tensor_ptr_test_tensor_ambiguity_with_mixed_vectors() {
        setup();
        let tensor = make_tensor_ptr_from_vec(
            alloc::vec![2, 2],
            alloc::vec![1.0f32, 2.0, 3.0, 4.0],
            Vec::new(),
            Vec::new(),
            ScalarType::Float,
            TensorShapeDynamism::DYNAMIC_BOUND,
        );
        let t = tensor.tensor();
        assert_eq!(t.dim(), 2);
        assert_eq!(t.size(0), 2);
        assert_eq!(t.size(1), 2);
        assert_eq!(*t.strides().at(0), 2);
        assert_eq!(*t.strides().at(1), 1);
        assert_eq!(unsafe { *t.const_data_ptr::<f32>().add(0) }, 1.0);
        assert_eq!(unsafe { *t.const_data_ptr::<f32>().add(3) }, 4.0);

        let tensor2 = make_tensor_ptr_from_vec(
            alloc::vec![2, 2],
            alloc::vec![1.0f32, 2.0, 3.0, 4.0],
            Vec::new(),
            Vec::new(),
            ScalarType::Float,
            TensorShapeDynamism::DYNAMIC_BOUND,
        );
        let t2 = tensor2.tensor();
        assert_eq!(t2.dim(), 2);
        assert_eq!(t2.size(0), 2);
        assert_eq!(t2.size(1), 2);
        assert_eq!(*t2.strides().at(0), 2);
        assert_eq!(*t2.strides().at(1), 1);
        assert_eq!(unsafe { *t2.const_data_ptr::<f32>().add(0) }, 1.0);
        assert_eq!(unsafe { *t2.const_data_ptr::<f32>().add(3) }, 4.0);
    }

    // [spec:et:sem:tensor-ptr.executorch.extension.casted-data-fn/test]
    // [spec:et:sem:tensor-ptr.executorch.extension.resize-tensor-ptr-fn/test]
    #[test]
    fn tensor_ptr_test_tensor_sharing_impl_modifies_shared_data_vector() {
        setup();
        let tensor1 = make_tensor_ptr_from_vec(
            alloc::vec![2, 3],
            alloc::vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
            Vec::new(),
            Vec::new(),
            ScalarType::Float,
            TensorShapeDynamism::DYNAMIC_BOUND,
        );
        let tensor2 = tensor1.clone();

        unsafe {
            *tensor1.tensor().mutable_data_ptr::<f32>().add(0) = 10.0;
        }
        assert_eq!(
            unsafe { *tensor2.tensor().const_data_ptr::<f32>().add(0) },
            10.0
        );

        unsafe {
            *tensor2.tensor().mutable_data_ptr::<f32>().add(5) = 20.0;
        }
        assert_eq!(
            unsafe { *tensor1.tensor().const_data_ptr::<f32>().add(5) },
            20.0
        );
    }

    // [spec:et:sem:tensor-ptr.executorch.extension.resize-tensor-ptr-fn/test]
    #[test]
    fn tensor_ptr_test_tensor_sharing_impl_resizing_affects_both_vector() {
        setup();
        let tensor1 = make_tensor_ptr_from_vec(
            alloc::vec![3, 4],
            (1..=12).map(|x| x as f32).collect(),
            Vec::new(),
            Vec::new(),
            ScalarType::Float,
            TensorShapeDynamism::DYNAMIC_BOUND,
        );
        let tensor2 = tensor1.clone();

        assert_eq!(resize_tensor_ptr(&tensor1, &[2, 6]), Error::Ok);
        assert_eq!(tensor2.tensor().size(0), 2);
        assert_eq!(tensor2.tensor().size(1), 6);

        assert_eq!(resize_tensor_ptr(&tensor2, &[4, 3]), Error::Ok);
        assert_eq!(tensor1.tensor().size(0), 4);
        assert_eq!(tensor1.tensor().size(1), 3);
    }

    // PORT-NOTE: unportable — the aliasing view overloads
    // `make_tensor_ptr(const Tensor&, ...)` / `make_tensor_ptr(const TensorPtr&,
    // ...)` are not ported, so the following cases (which bind solely to them)
    // are recorded here rather than ported:
    //   MakeTensorPtrFromExistingTensorInt32, MakeViewOverrideSizesRankIncrease,
    //   MakeViewOverrideSizesSameRankRecomputesStrides,
    //   MakeViewOverrideDimOrderOnly, MakeViewOverrideStridesOnlyInfersDimOrder,
    //   MakeViewReuseMetadataWhenShapeSame,
    //   MakeViewShapeChangeWithExplicitOldStridesExpectDeath,
    //   MakeViewInvalidDimOrderExpectDeath,
    //   MakeViewFromTensorPtrConvenienceOverload, MakeViewRankDecreaseFlatten,
    //   MakeViewFromScalarAliasAnd1D, MakeViewExplicitDimOrderAndStridesShapeChange,
    //   MakeView3DDimOrderOnly, MakeViewDynamismPropagationResizeAlias,
    //   MakeViewSameRankShapeChangeCopiesDimOrder,
    //   MakeTensorPtrFromTensorPtrInt32/Double/Int64/Null,
    //   MakeTensorPtrFromExistingTensorDouble/Int64/UInt32,
    //   MakeViewFromTensorPtrKeepsSourceAlive,
    //   MakeViewFromTensorDoesNotKeepAliveByDefault,
    //   MakeViewFromTensorWithDeleterKeepsAlive.

    // [spec:et:sem:tensor-ptr.executorch.extension.make-tensor-ptr-fn/test]
    #[test]
    fn tensor_ptr_test_tensor_uint8data_int16_type() {
        setup();
        let int16_values: [i16; 4] = [-1, 2, -3, 4];
        let byte_pointer = int16_values.as_ptr() as *const u8;
        let byte_data: Vec<u8> = unsafe {
            core::slice::from_raw_parts(
                byte_pointer,
                int16_values.len() * core::mem::size_of::<i16>(),
            )
        }
        .to_vec();
        let tensor = make_tensor_ptr_from_bytes(
            alloc::vec![4],
            byte_data,
            Vec::new(),
            Vec::new(),
            ScalarType::Short,
            TensorShapeDynamism::DYNAMIC_BOUND,
        );
        let t = tensor.tensor();
        assert_eq!(t.dim(), 1);
        assert_eq!(t.size(0), 4);
        let int16_data = t.const_data_ptr::<i16>();
        assert_eq!(unsafe { *int16_data.add(0) }, -1);
        assert_eq!(unsafe { *int16_data.add(1) }, 2);
        assert_eq!(unsafe { *int16_data.add(2) }, -3);
        assert_eq!(unsafe { *int16_data.add(3) }, 4);
    }

    // ---- clone_tensor_ptr (2-arg cast overload) ----
    // PORT-NOTE: the single-arg `clone_tensor_ptr(tensor)` is not ported; the
    // cases below use the 2-arg cast form `clone_tensor_ptr(*tensor, type)`.

    // [spec:et:sem:tensor-ptr.executorch.extension.clone-tensor-ptr-fn/test]
    #[test]
    fn tensor_ptr_test_clone_tensor_ptr_cast_int32_to_float() {
        setup();
        let tensor = make_tensor_ptr_from_vec(
            alloc::vec![2, 2],
            alloc::vec![1i32, 2, 3, 4],
            Vec::new(),
            Vec::new(),
            ScalarType::Int,
            TensorShapeDynamism::DYNAMIC_BOUND,
        );
        let cloned_tensor = clone_tensor_ptr(&tensor.tensor(), ScalarType::Float);

        let ct = cloned_tensor.tensor();
        assert_eq!(ct.dim(), 2);
        assert_eq!(ct.size(0), 2);
        assert_eq!(ct.size(1), 2);
        assert_eq!(ct.scalar_type(), ScalarType::Float);
        let ptr = ct.const_data_ptr::<f32>();
        assert_eq!(unsafe { *ptr.add(0) }, 1.0);
        assert_eq!(unsafe { *ptr.add(1) }, 2.0);
        assert_eq!(unsafe { *ptr.add(2) }, 3.0);
        assert_eq!(unsafe { *ptr.add(3) }, 4.0);
    }

    // [spec:et:sem:tensor-ptr.executorch.extension.clone-tensor-ptr-fn/test]
    #[test]
    fn tensor_ptr_test_clone_tensor_ptr_cast_float_to_bfloat16() {
        setup();
        let tensor = make_tensor_ptr_from_vec(
            alloc::vec![3],
            alloc::vec![1.0f32, 2.0, 3.5],
            Vec::new(),
            Vec::new(),
            ScalarType::Float,
            TensorShapeDynamism::DYNAMIC_BOUND,
        );
        let cloned_tensor = clone_tensor_ptr(&tensor.tensor(), ScalarType::BFloat16);

        let ct = cloned_tensor.tensor();
        assert_eq!(ct.dim(), 1);
        assert_eq!(ct.size(0), 3);
        assert_eq!(ct.scalar_type(), ScalarType::BFloat16);
        let ptr = ct.const_data_ptr::<crate::runtime::core::portable_type::BFloat16>();
        assert!((unsafe { *ptr.add(0) }.to_f32() - 1.0).abs() < 0.01);
        assert!((unsafe { *ptr.add(1) }.to_f32() - 2.0).abs() < 0.01);
        assert!((unsafe { *ptr.add(2) }.to_f32() - 3.5).abs() < 0.01);
    }

    // [spec:et:sem:tensor-ptr.executorch.extension.clone-tensor-ptr-fn/test]
    #[test]
    fn tensor_ptr_test_clone_tensor_ptr_cast_keeps_metadata() {
        setup();
        let data: Vec<u8> = alloc::vec![0u8; 6 * element_size(ScalarType::Float)];
        let tensor = make_tensor_ptr_from_bytes(
            alloc::vec![2, 3],
            data,
            Vec::new(),
            Vec::new(),
            ScalarType::Float,
            TensorShapeDynamism::DYNAMIC_BOUND,
        );
        let cloned_tensor = clone_tensor_ptr(&tensor.tensor(), ScalarType::Float);

        let ct = cloned_tensor.tensor();
        assert_eq!(ct.dim(), 2);
        assert_eq!(ct.size(0), 2);
        assert_eq!(ct.size(1), 3);
        assert_eq!(*ct.strides().at(0), 3);
        assert_eq!(*ct.strides().at(1), 1);
        assert_eq!(ct.scalar_type(), ScalarType::Float);
    }

    // [spec:et:sem:tensor-ptr.executorch.extension.clone-tensor-ptr-fn/test]
    #[test]
    fn tensor_ptr_test_clone_tensor_ptr_cast_null_data() {
        setup();
        let tensor = make_tensor_ptr(
            alloc::vec![2, 2],
            core::ptr::null_mut(),
            Vec::new(),
            Vec::new(),
            ScalarType::Float,
            TensorShapeDynamism::DYNAMIC_BOUND,
            None,
            cpu(),
        );
        let cloned_tensor = clone_tensor_ptr(&tensor.tensor(), ScalarType::Int);

        let ct = cloned_tensor.tensor();
        assert_eq!(ct.dim(), 2);
        assert_eq!(ct.size(0), 2);
        assert_eq!(ct.size(1), 2);
        assert!(ct.const_data_ptr_typed().is_null());
        assert_eq!(ct.scalar_type(), ScalarType::Int);
    }

    // PORT-NOTE: `ET_EXPECT_DEATH` death test; `runtime_abort` -> `libc::abort()`
    // terminates the process, so ported + `#[ignore]`d.
    // [spec:et:sem:tensor-ptr.executorch.extension.clone-tensor-ptr-fn/test]
    #[test]
    #[should_panic]
    #[ignore]
    fn tensor_ptr_test_clone_tensor_ptr_cast_invalid_expect_death() {
        setup();
        let tensor = make_tensor_ptr_from_vec(
            alloc::vec![2],
            alloc::vec![1.0f32, 2.0],
            Vec::new(),
            Vec::new(),
            ScalarType::Float,
            TensorShapeDynamism::DYNAMIC_BOUND,
        );
        let _ = clone_tensor_ptr(&tensor.tensor(), ScalarType::Int);
    }

    // [spec:et:sem:tensor-ptr.executorch.extension.casted-data-fn/test]
    #[test]
    fn tensor_ptr_test_tensor_data_casting_from_int_to_float() {
        setup();
        let tensor = make_tensor_ptr_from_vec(
            alloc::vec![2, 3],
            alloc::vec![1i32, 2, 3, 4, 5, 6],
            Vec::new(),
            Vec::new(),
            ScalarType::Float,
            TensorShapeDynamism::DYNAMIC_BOUND,
        );
        let t = tensor.tensor();
        assert_eq!(t.dim(), 2);
        assert_eq!(t.size(0), 2);
        assert_eq!(t.size(1), 3);
        assert_eq!(t.scalar_type(), ScalarType::Float);
        let data_ptr = t.const_data_ptr::<f32>();
        assert_eq!(unsafe { *data_ptr.add(0) }, 1.0);
        assert_eq!(unsafe { *data_ptr.add(5) }, 6.0);
    }

    // PORT-NOTE: unportable — `TensorDataCastingFromIntToDouble`,
    // `TensorDataCastingFromFloatToHalf`, `TensorDataCastingFromDoubleToFloat`,
    // `TensorDataCastingFromInt64ToInt32`, `TensorDataCastingFromFloatToBFloat16`,
    // `TensorDataCastingFromInt32ToUInt16`, `TensorDataCastingFromUInt32ToFloat`,
    // `TensorDataCastingFromFloatToUInt32`, and the initializer-list casts
    // `InitializerListDoubleToHalf`/`InitializerListInt8ToInt64` all bind solely
    // to the 2-arg data-only overload `make_tensor_ptr(vector<T>, type)` /
    // `make_tensor_ptr<T>({list}, type)` (rank-1 deducing), not ported.

    // [spec:et:sem:tensor-ptr.executorch.extension.make-tensor-ptr-fn/test]
    #[test]
    fn tensor_ptr_test_tensor_inferred_dim_order_and_strides() {
        setup();
        let mut data: [f32; 12] = [0.0; 12];
        let tensor = make_tensor_ptr(
            alloc::vec![3, 4],
            data.as_mut_ptr() as *mut core::ffi::c_void,
            Vec::new(),
            alloc::vec![4, 1],
            ScalarType::Float,
            TensorShapeDynamism::DYNAMIC_BOUND,
            None,
            cpu(),
        );
        let t = tensor.tensor();
        assert_eq!(t.dim(), 2);
        assert_eq!(t.size(0), 3);
        assert_eq!(t.size(1), 4);
        assert_eq!(*t.strides().at(0), 4);
        assert_eq!(*t.strides().at(1), 1);
        assert_eq!(
            t.const_data_ptr_typed(),
            data.as_ptr() as *const core::ffi::c_void
        );
    }

    // [spec:et:sem:tensor-ptr.executorch.extension.make-tensor-ptr-fn/test]
    #[test]
    fn tensor_ptr_test_tensor_inferred_dim_order_custom_strides() {
        setup();
        let mut data: [f32; 12] = [0.0; 12];
        let tensor = make_tensor_ptr(
            alloc::vec![3, 4],
            data.as_mut_ptr() as *mut core::ffi::c_void,
            Vec::new(),
            alloc::vec![1, 3],
            ScalarType::Float,
            TensorShapeDynamism::DYNAMIC_BOUND,
            None,
            cpu(),
        );
        let t = tensor.tensor();
        assert_eq!(t.dim(), 2);
        assert_eq!(t.size(0), 3);
        assert_eq!(t.size(1), 4);
        assert_eq!(*t.strides().at(0), 1);
        assert_eq!(*t.strides().at(1), 3);
    }

    // [spec:et:sem:tensor-ptr.executorch.extension.make-tensor-ptr-fn/test]
    #[test]
    fn tensor_ptr_test_tensor_default_dim_order_and_strides() {
        setup();
        let mut data: [f32; 24] = [0.0; 24];
        let tensor = mk_raw(
            alloc::vec![2, 3, 4],
            data.as_mut_ptr() as *mut core::ffi::c_void,
            ScalarType::Float,
        );
        let t = tensor.tensor();
        assert_eq!(t.dim(), 3);
        assert_eq!(t.size(0), 2);
        assert_eq!(t.size(1), 3);
        assert_eq!(t.size(2), 4);
        assert_eq!(*t.strides().at(0), 12);
        assert_eq!(*t.strides().at(1), 4);
        assert_eq!(*t.strides().at(2), 1);
    }

    // PORT-NOTE: `ET_EXPECT_DEATH`; ported + `#[ignore]`d.
    // [spec:et:sem:tensor-ptr.executorch.extension.make-tensor-ptr-fn/test]
    #[test]
    #[should_panic]
    #[ignore]
    fn tensor_ptr_test_tensor_mismatch_strides_and_dim_order() {
        setup();
        let mut data: [f32; 12] = [0.0; 12];
        let _ = make_tensor_ptr(
            alloc::vec![3, 4],
            data.as_mut_ptr() as *mut core::ffi::c_void,
            alloc::vec![1, 0],
            alloc::vec![1, 4],
            ScalarType::Float,
            TensorShapeDynamism::DYNAMIC_BOUND,
            None,
            cpu(),
        );
    }

    // [spec:et:sem:tensor-ptr.executorch.extension.make-tensor-ptr-fn/test]
    #[test]
    fn tensor_ptr_test_tensor_custom_dim_order_and_strides() {
        setup();
        let mut data: [f32; 12] = [0.0; 12];
        let tensor = make_tensor_ptr(
            alloc::vec![3, 4],
            data.as_mut_ptr() as *mut core::ffi::c_void,
            alloc::vec![1, 0],
            alloc::vec![1, 3],
            ScalarType::Float,
            TensorShapeDynamism::DYNAMIC_BOUND,
            None,
            cpu(),
        );
        let t = tensor.tensor();
        assert_eq!(t.dim(), 2);
        assert_eq!(t.size(0), 3);
        assert_eq!(t.size(1), 4);
        assert_eq!(*t.strides().at(0), 1);
        assert_eq!(*t.strides().at(1), 3);
    }

    // PORT-NOTE: `ET_EXPECT_DEATH`; ported + `#[ignore]`d.
    // [spec:et:sem:tensor-ptr.executorch.extension.make-tensor-ptr-fn/test]
    #[test]
    #[should_panic]
    #[ignore]
    fn tensor_ptr_test_tensor_invalid_dim_order() {
        setup();
        let mut data: [f32; 20] = [0.0; 20];
        data[0] = 2.0;
        let _ = make_tensor_ptr(
            alloc::vec![4, 5],
            data.as_mut_ptr() as *mut core::ffi::c_void,
            alloc::vec![2, 1],
            alloc::vec![1, 4],
            ScalarType::Float,
            TensorShapeDynamism::DYNAMIC_BOUND,
            None,
            cpu(),
        );
    }

    // [spec:et:sem:tensor-ptr.executorch.extension.make-tensor-ptr-fn/test]
    #[test]
    fn tensor_ptr_test_tensor_custom_deleter() {
        setup();
        let mut data: [f32; 20] = [0.0; 20];
        data[0] = 4.0;
        let tensor = mk_raw(
            alloc::vec![4, 5],
            data.as_mut_ptr() as *mut core::ffi::c_void,
            ScalarType::Float,
        );

        // C++ `TensorPtr copied_tensor = tensor;` -> Arc clone.
        let copied_tensor = tensor.clone();
        assert_eq!(
            Arc::strong_count(tensor.storage()),
            Arc::strong_count(copied_tensor.storage())
        );

        drop(tensor);
        assert_eq!(Arc::strong_count(copied_tensor.storage()), 1);
    }

    // [spec:et:sem:tensor-ptr.executorch.extension.make-tensor-ptr-fn/test]
    #[test]
    fn tensor_ptr_test_tensor_data_deleter_releases_captured_shared_ptr() {
        setup();
        static DELETER_CALLED: AtomicBool = AtomicBool::new(false);
        DELETER_CALLED.store(false, Ordering::SeqCst);
        let data_ptr: Arc<[f32]> = Arc::from(alloc::vec![0.0f32; 10]);
        let raw = data_ptr.as_ptr() as *mut f32;
        let captured = data_ptr.clone();
        let tensor = make_tensor_ptr(
            alloc::vec![4, 5],
            raw as *mut core::ffi::c_void,
            Vec::new(),
            Vec::new(),
            ScalarType::Float,
            TensorShapeDynamism::DYNAMIC_BOUND,
            Some(alloc::boxed::Box::new(move |_| {
                let _keep = &captured;
                DELETER_CALLED.store(true, Ordering::SeqCst);
            })),
            cpu(),
        );
        assert_eq!(Arc::strong_count(&data_ptr), 2);
        drop(tensor);
        assert!(DELETER_CALLED.load(Ordering::SeqCst));
        assert_eq!(Arc::strong_count(&data_ptr), 1);
    }

    // [spec:et:sem:tensor-ptr.executorch.extension.make-tensor-ptr-fn/test]
    #[test]
    fn tensor_ptr_test_shared_data_management() {
        setup();
        // std::make_shared<std::vector<float>>(100, 1.0f).
        let data: Arc<Vec<f32>> = Arc::new(alloc::vec![1.0f32; 100]);
        let tensor1 = mk_raw(
            alloc::vec![10, 10],
            data.as_ptr() as *mut core::ffi::c_void,
            ScalarType::Float,
        );
        let tensor2 = tensor1.clone();

        assert_eq!(
            Arc::as_ptr(tensor1.storage()),
            Arc::as_ptr(tensor2.storage())
        );
        assert_eq!(Arc::strong_count(tensor1.storage()), 2);
        assert_eq!(
            unsafe { *tensor1.tensor().const_data_ptr::<f32>().add(0) },
            1.0
        );

        unsafe {
            *tensor1.tensor().mutable_data_ptr::<f32>().add(0) = 2.0;
        }
        assert_eq!(
            unsafe { *tensor1.tensor().const_data_ptr::<f32>().add(0) },
            2.0
        );

        drop(tensor1);
        // tensor2 still valid.
        assert_eq!(Arc::strong_count(tensor2.storage()), 1);
        assert_eq!(
            unsafe { *tensor2.tensor().const_data_ptr::<f32>().add(0) },
            2.0
        );
        let _ = &data;
    }

    // [spec:et:sem:tensor-ptr.executorch.extension.make-tensor-ptr-fn/test]
    //
    // The deleted copy/move ops on `Storage` collapse onto "handed out only via
    // Arc, never duplicated" in Rust: the deleter runs exactly once, when the
    // sole Storage drops with the last TensorPtr, and the captured Arc count
    // returns to 1 (a copied Storage would run the deleter twice).
    // [spec:et:sem:tensor-ptr.executorch.extension.storage.operator-fn/test]
    #[test]
    fn tensor_ptr_test_custom_deleter_with_shared_data() {
        setup();
        let data: Arc<Vec<f32>> = Arc::new(alloc::vec![1.0f32; 100]);
        static DELETER_CALLED: AtomicBool = AtomicBool::new(false);
        DELETER_CALLED.store(false, Ordering::SeqCst);
        {
            let captured = data.clone();
            let data_ptr = data.as_ptr() as *mut core::ffi::c_void;
            let tensor = make_tensor_ptr(
                alloc::vec![10, 10],
                data_ptr,
                Vec::new(),
                Vec::new(),
                ScalarType::Float,
                TensorShapeDynamism::DYNAMIC_BOUND,
                Some(alloc::boxed::Box::new(move |_| {
                    DELETER_CALLED.store(true, Ordering::SeqCst);
                    let _drop = &captured;
                })),
                cpu(),
            );
            assert_eq!(Arc::strong_count(&data), 2);
            assert!(!DELETER_CALLED.load(Ordering::SeqCst));
            drop(tensor);
        }
        assert!(DELETER_CALLED.load(Ordering::SeqCst));
        assert_eq!(Arc::strong_count(&data), 1);
    }

    // [spec:et:sem:tensor-ptr.executorch.extension.casted-data-fn/test]
    #[test]
    fn tensor_ptr_test_tensor_deduced_scalar_type() {
        setup();
        let tensor = make_tensor_ptr_from_vec(
            alloc::vec![2, 2],
            alloc::vec![1.0f64, 2.0, 3.0, 4.0],
            Vec::new(),
            Vec::new(),
            ScalarType::Double,
            TensorShapeDynamism::DYNAMIC_BOUND,
        );
        let t = tensor.tensor();
        assert_eq!(t.dim(), 2);
        assert_eq!(t.size(0), 2);
        assert_eq!(t.size(1), 2);
        assert_eq!(*t.strides().at(0), 2);
        assert_eq!(*t.strides().at(1), 1);
        assert_eq!(unsafe { *t.const_data_ptr::<f64>().add(0) }, 1.0);
        assert_eq!(unsafe { *t.const_data_ptr::<f64>().add(3) }, 4.0);
    }

    // [spec:et:sem:tensor-ptr.executorch.extension.make-tensor-ptr-fn/test]
    #[test]
    fn tensor_ptr_test_tensor_uint8data_with_float_scalar_type() {
        setup();
        let mut data: Vec<u8> = alloc::vec![0u8; 4 * element_size(ScalarType::Float)];
        {
            let float_data = data.as_mut_ptr() as *mut f32;
            unsafe {
                *float_data.add(0) = 1.0;
                *float_data.add(1) = 2.0;
                *float_data.add(2) = 3.0;
                *float_data.add(3) = 4.0;
            }
        }
        let tensor = make_tensor_ptr_from_bytes(
            alloc::vec![2, 2],
            data,
            Vec::new(),
            Vec::new(),
            ScalarType::Float,
            TensorShapeDynamism::DYNAMIC_BOUND,
        );
        let t = tensor.tensor();
        assert_eq!(t.dim(), 2);
        assert_eq!(t.size(0), 2);
        assert_eq!(t.size(1), 2);
        assert_eq!(*t.strides().at(0), 2);
        assert_eq!(*t.strides().at(1), 1);
        assert_eq!(unsafe { *t.const_data_ptr::<f32>().add(0) }, 1.0);
        assert_eq!(unsafe { *t.const_data_ptr::<f32>().add(1) }, 2.0);
        assert_eq!(unsafe { *t.const_data_ptr::<f32>().add(2) }, 3.0);
        assert_eq!(unsafe { *t.const_data_ptr::<f32>().add(3) }, 4.0);
    }

    // PORT-NOTE: `ET_EXPECT_DEATH`; ported + `#[ignore]`d.
    // [spec:et:sem:tensor-ptr.executorch.extension.make-tensor-ptr-fn/test]
    #[test]
    #[should_panic]
    #[ignore]
    fn tensor_ptr_test_tensor_uint8data_too_small_expect_death() {
        setup();
        let data: Vec<u8> = alloc::vec![0u8; 2 * element_size(ScalarType::Float)];
        let _ = make_tensor_ptr_from_bytes(
            alloc::vec![2, 2],
            data,
            Vec::new(),
            Vec::new(),
            ScalarType::Float,
            TensorShapeDynamism::DYNAMIC_BOUND,
        );
    }

    // PORT-NOTE: `ET_EXPECT_DEATH`; ported + `#[ignore]`d.
    // [spec:et:sem:tensor-ptr.executorch.extension.make-tensor-ptr-fn/test]
    #[test]
    #[should_panic]
    #[ignore]
    fn tensor_ptr_test_tensor_uint8data_too_large_expect_death() {
        setup();
        let data: Vec<u8> = alloc::vec![0u8; 5 * element_size(ScalarType::Float)];
        let _ = make_tensor_ptr_from_bytes(
            alloc::vec![2, 2],
            data,
            Vec::new(),
            Vec::new(),
            ScalarType::Float,
            TensorShapeDynamism::DYNAMIC_BOUND,
        );
    }

    // PORT-NOTE: `ET_EXPECT_DEATH`; ported + `#[ignore]`d.
    // [spec:et:sem:tensor-ptr.executorch.extension.casted-data-fn/test]
    #[test]
    #[should_panic]
    #[ignore]
    fn tensor_ptr_test_vector_float_too_small_expect_death() {
        setup();
        let data: Vec<f32> = alloc::vec![1.0f32; 9];
        let _ = make_tensor_ptr_from_vec(
            alloc::vec![2, 5],
            data,
            Vec::new(),
            Vec::new(),
            ScalarType::Float,
            TensorShapeDynamism::DYNAMIC_BOUND,
        );
    }

    // PORT-NOTE: `ET_EXPECT_DEATH`; ported + `#[ignore]`d.
    // [spec:et:sem:tensor-ptr.executorch.extension.casted-data-fn/test]
    #[test]
    #[should_panic]
    #[ignore]
    fn tensor_ptr_test_vector_float_too_large_expect_death() {
        setup();
        let data: Vec<f32> = alloc::vec![1.0f32; 11];
        let _ = make_tensor_ptr_from_vec(
            alloc::vec![2, 5],
            data,
            Vec::new(),
            Vec::new(),
            ScalarType::Float,
            TensorShapeDynamism::DYNAMIC_BOUND,
        );
    }

    // PORT-NOTE: `ET_EXPECT_DEATH`; ported + `#[ignore]`d.
    // [spec:et:sem:tensor-ptr.executorch.extension.casted-data-fn/test]
    #[test]
    #[should_panic]
    #[ignore]
    fn tensor_ptr_test_vector_int_to_float_cast_too_small_expect_death() {
        setup();
        let data: Vec<i32> = alloc::vec![1i32; 9];
        let _ = make_tensor_ptr_from_vec(
            alloc::vec![2, 5],
            data,
            Vec::new(),
            Vec::new(),
            ScalarType::Float,
            TensorShapeDynamism::DYNAMIC_BOUND,
        );
    }

    // PORT-NOTE: `ET_EXPECT_DEATH`; ported + `#[ignore]`d.
    // [spec:et:sem:tensor-ptr.executorch.extension.casted-data-fn/test]
    #[test]
    #[should_panic]
    #[ignore]
    fn tensor_ptr_test_vector_int_to_float_cast_too_large_expect_death() {
        setup();
        let data: Vec<i32> = alloc::vec![1i32; 11];
        let _ = make_tensor_ptr_from_vec(
            alloc::vec![2, 5],
            data,
            Vec::new(),
            Vec::new(),
            ScalarType::Float,
            TensorShapeDynamism::DYNAMIC_BOUND,
        );
    }

    // PORT-NOTE: `ET_EXPECT_DEATH` (two death sites); ported + `#[ignore]`d. Both
    // sites drive the `make_tensor_ptr` strides/dim_order size-mismatch check.
    // [spec:et:sem:tensor-ptr.executorch.extension.make-tensor-ptr-fn/test]
    #[test]
    #[should_panic]
    #[ignore]
    fn tensor_ptr_test_strides_and_dim_order_must_match_sizes() {
        setup();
        let mut data: [f32; 12] = [0.0; 12];
        // First death site: strides len 1 for 2D sizes.
        let _ = make_tensor_ptr(
            alloc::vec![3, 4],
            data.as_mut_ptr() as *mut core::ffi::c_void,
            Vec::new(),
            alloc::vec![1],
            ScalarType::Float,
            TensorShapeDynamism::DYNAMIC_BOUND,
            None,
            cpu(),
        );
    }

    // PORT-NOTE: second death site of `StridesAndDimOrderMustMatchSizes`
    // (dim_order len 1 for 2D sizes) — split out because a single process abort
    // ends the test. Ported + `#[ignore]`d.
    // [spec:et:sem:tensor-ptr.executorch.extension.make-tensor-ptr-fn/test]
    #[test]
    #[should_panic]
    #[ignore]
    fn tensor_ptr_test_strides_and_dim_order_must_match_sizes_dim_order() {
        setup();
        let mut data: [f32; 12] = [0.0; 12];
        let _ = make_tensor_ptr(
            alloc::vec![3, 4],
            data.as_mut_ptr() as *mut core::ffi::c_void,
            alloc::vec![0],
            alloc::vec![4, 1],
            ScalarType::Float,
            TensorShapeDynamism::DYNAMIC_BOUND,
            None,
            cpu(),
        );
    }

    // PORT-NOTE: `ET_EXPECT_DEATH`; ported + `#[ignore]`d. `TensorDataCastingInvalidCast`
    // binds to `make_tensor_ptr(vector<float>, ScalarType::Int)` (data-only cast),
    // an un-ported overload; the equivalent cast-and-size path is driven here via
    // `make_tensor_ptr_from_vec`.
    // [spec:et:sem:tensor-ptr.executorch.extension.casted-data-fn/test]
    #[test]
    #[should_panic]
    #[ignore]
    fn tensor_ptr_test_tensor_data_casting_invalid_cast() {
        setup();
        let float_data: Vec<f32> = alloc::vec![1.0f32, 2.0, 3.0];
        let _ = make_tensor_ptr_from_vec(
            alloc::vec![3],
            float_data,
            Vec::new(),
            Vec::new(),
            ScalarType::Int,
            TensorShapeDynamism::DYNAMIC_BOUND,
        );
    }

    // PORT-NOTE: `ET_EXPECT_DEATH`; ported + `#[ignore]`d.
    // [spec:et:sem:tensor-ptr.executorch.extension.make-tensor-ptr-fn/test]
    #[test]
    #[should_panic]
    #[ignore]
    fn tensor_ptr_test_tensor_uint8data_size_mismatch_uint32_expect_death() {
        setup();
        let data: Vec<u8> = alloc::vec![0u8; 3 * element_size(ScalarType::UInt32) - 1];
        let _ = make_tensor_ptr_from_bytes(
            alloc::vec![3],
            data,
            Vec::new(),
            Vec::new(),
            ScalarType::UInt32,
            TensorShapeDynamism::DYNAMIC_BOUND,
        );
    }

    // PORT-NOTE: `ET_EXPECT_DEATH`; ported + `#[ignore]`d.
    // [spec:et:sem:tensor-ptr.executorch.extension.make-tensor-ptr-fn/test]
    #[test]
    #[should_panic]
    #[ignore]
    fn tensor_ptr_test_tensor_uint8data_size_mismatch_uint64_expect_death() {
        setup();
        let data: Vec<u8> = alloc::vec![0u8; 2 * element_size(ScalarType::UInt64) + 1];
        let _ = make_tensor_ptr_from_bytes(
            alloc::vec![2],
            data,
            Vec::new(),
            Vec::new(),
            ScalarType::UInt64,
            TensorShapeDynamism::DYNAMIC_BOUND,
        );
    }

    // [spec:et:sem:tensor-ptr.executorch.extension.make-tensor-ptr-fn/test]
    #[test]
    fn tensor_ptr_test_tensor_uint8data_uint32_type() {
        setup();
        let values: [u32; 3] = [1, 4000000000, 123];
        let bytes = values.as_ptr() as *const u8;
        let raw: Vec<u8> = unsafe {
            core::slice::from_raw_parts(bytes, values.len() * core::mem::size_of::<u32>())
        }
        .to_vec();
        let tensor = make_tensor_ptr_from_bytes(
            alloc::vec![3],
            raw,
            Vec::new(),
            Vec::new(),
            ScalarType::UInt32,
            TensorShapeDynamism::DYNAMIC_BOUND,
        );
        let t = tensor.tensor();
        assert_eq!(t.dim(), 1);
        assert_eq!(t.size(0), 3);
        assert_eq!(t.scalar_type(), ScalarType::UInt32);
        let ptr = t.const_data_ptr::<u32>();
        assert_eq!(unsafe { *ptr.add(0) }, 1);
        assert_eq!(unsafe { *ptr.add(1) }, 4000000000);
        assert_eq!(unsafe { *ptr.add(2) }, 123);
    }

    // [spec:et:sem:tensor-ptr.executorch.extension.make-tensor-ptr-fn/test]
    #[test]
    fn tensor_ptr_test_tensor_uint8data_uint64_type() {
        setup();
        let values: [u64; 3] = [0, 42, 9000000000000000000];
        let bytes = values.as_ptr() as *const u8;
        let raw: Vec<u8> = unsafe {
            core::slice::from_raw_parts(bytes, values.len() * core::mem::size_of::<u64>())
        }
        .to_vec();
        let tensor = make_tensor_ptr_from_bytes(
            alloc::vec![3],
            raw,
            Vec::new(),
            Vec::new(),
            ScalarType::UInt64,
            TensorShapeDynamism::DYNAMIC_BOUND,
        );
        let t = tensor.tensor();
        assert_eq!(t.dim(), 1);
        assert_eq!(t.size(0), 3);
        assert_eq!(t.scalar_type(), ScalarType::UInt64);
        let ptr = t.const_data_ptr::<u64>();
        assert_eq!(unsafe { *ptr.add(0) }, 0);
        assert_eq!(unsafe { *ptr.add(1) }, 42);
        assert_eq!(unsafe { *ptr.add(2) }, 9000000000000000000);
    }

    // [spec:et:sem:tensor-ptr.executorch.extension.casted-data-fn/test]
    #[test]
    fn tensor_ptr_test_tensor2d_uint16_owning_data() {
        setup();
        let tensor = make_tensor_ptr_from_vec(
            alloc::vec![2, 3],
            alloc::vec![1u16, 2, 3, 4, 5, 6],
            Vec::new(),
            Vec::new(),
            ScalarType::UInt16,
            TensorShapeDynamism::DYNAMIC_BOUND,
        );
        let t = tensor.tensor();
        assert_eq!(t.dim(), 2);
        assert_eq!(t.size(0), 2);
        assert_eq!(t.size(1), 3);
        assert_eq!(*t.strides().at(0), 3);
        assert_eq!(*t.strides().at(1), 1);
        assert_eq!(t.scalar_type(), ScalarType::UInt16);
        let ptr = t.const_data_ptr::<u16>();
        assert_eq!(unsafe { *ptr.add(0) }, 1);
        assert_eq!(unsafe { *ptr.add(5) }, 6);
    }
}

#[cfg(test)]
mod device_tests {
    // Literal port of extension/tensor/test/tensor_ptr_device_test.cpp
    // (non-ATen `TensorPtrDeviceTest` fixture).
    //
    // PORT-NOTE: the C++ registers a single static `MockCudaAllocator` in the
    // CUDA slot at program start. The Rust test binary shares one process-wide
    // registry across all suites, so — mirroring the device_allocator suite —
    // these tests serialize on `DEVICE_REGISTRY_TEST_LOCK`, clear the registry,
    // and register this suite's mock afresh in `setup()`.
    use super::*;
    use crate::runtime::core::device_allocator::{DeviceAllocator, DeviceAllocatorRegistry};
    use crate::runtime::core::error::Result;
    use crate::runtime::core::portable_type::device::{DeviceIndex, DeviceType};

    // PORT-NOTE: `MockCudaAllocator` (runtime/core/test/mock_cuda_allocator.h)
    // simulates device memory with host malloc/free/memcpy so data roundtrips
    // can be verified without CUDA. It is a test-only mock and is reproduced here
    // rather than ported to its own module (the C++ header lives under
    // runtime/core/test and is only consumed by this test).
    struct MockCudaAllocator {
        allocate_count_: i32,
        deallocate_count_: i32,
        h2d_count_: i32,
        d2h_count_: i32,
    }

    impl MockCudaAllocator {
        const fn new() -> Self {
            MockCudaAllocator {
                allocate_count_: 0,
                deallocate_count_: 0,
                h2d_count_: 0,
                d2h_count_: 0,
            }
        }
    }

    impl DeviceAllocator for MockCudaAllocator {
        fn allocate(
            &mut self,
            nbytes: usize,
            _index: DeviceIndex,
            _alignment: usize,
        ) -> Result<*mut core::ffi::c_void> {
            let ptr = unsafe { libc::malloc(nbytes) };
            if ptr.is_null() {
                return Err(Error::MemoryAllocationFailed);
            }
            self.allocate_count_ += 1;
            Ok(ptr)
        }

        fn deallocate(&mut self, ptr: *mut core::ffi::c_void, _index: DeviceIndex) {
            self.deallocate_count_ += 1;
            unsafe { libc::free(ptr) };
        }

        fn copy_host_to_device(
            &mut self,
            dst: *mut core::ffi::c_void,
            src: *const core::ffi::c_void,
            nbytes: usize,
            _index: DeviceIndex,
        ) -> Error {
            unsafe { libc::memcpy(dst, src, nbytes) };
            self.h2d_count_ += 1;
            Error::Ok
        }

        fn copy_device_to_host(
            &mut self,
            dst: *mut core::ffi::c_void,
            src: *const core::ffi::c_void,
            nbytes: usize,
            _index: DeviceIndex,
        ) -> Error {
            unsafe { libc::memcpy(dst, src, nbytes) };
            self.d2h_count_ += 1;
            Error::Ok
        }

        fn device_type(&self) -> DeviceType {
            DeviceType::CUDA
        }
    }

    // static MockCudaAllocator g_mock_cuda;
    static mut G_MOCK_CUDA: MockCudaAllocator = MockCudaAllocator::new();

    fn g_mock_cuda() -> *mut MockCudaAllocator {
        &raw mut G_MOCK_CUDA
    }

    // Mirrors SetUpTestSuite() (runtime_init + register) + SetUp() (reset the
    // four counters). Locks, clears and registers atomically via
    // install_for_test; callers hold the returned guard for the test body.
    fn setup() -> std::sync::MutexGuard<'static, ()> {
        crate::runtime::platform::runtime::runtime_init();
        let guard = DeviceAllocatorRegistry::install_for_test(
            g_mock_cuda() as *mut (dyn DeviceAllocator + 'static)
        );
        unsafe {
            (*g_mock_cuda()).allocate_count_ = 0;
            (*g_mock_cuda()).deallocate_count_ = 0;
            (*g_mock_cuda()).h2d_count_ = 0;
            (*g_mock_cuda()).d2h_count_ = 0;
        }
        guard
    }

    // ---- make_tensor_ptr convenience shims ----
    //
    // PORT-NOTE: the C++ `make_tensor_ptr(sizes, {float initializer list})` and
    // `make_tensor_ptr(sizes, vector<T>)` data-owning overloads map to the ported
    // `make_tensor_ptr_from_vec` (deduced type, contiguous, DYNAMIC_BOUND). The
    // raw-pointer overload maps to `make_tensor_ptr_simple`.
    fn make_owning_f32(sizes: Vec<SizesType>, data: Vec<f32>) -> TensorPtr {
        make_tensor_ptr_from_vec(
            sizes,
            data,
            Vec::new(),
            Vec::new(),
            ScalarType::Float,
            TensorShapeDynamism::DYNAMIC_BOUND,
        )
    }

    fn make_raw_f32(sizes: Vec<SizesType>, data: *mut f32) -> TensorPtr {
        make_tensor_ptr_simple(
            sizes,
            data as *mut core::ffi::c_void,
            ScalarType::Float,
            TensorShapeDynamism::DYNAMIC_BOUND,
            None,
            Device::from_type(DeviceType::CPU),
        )
    }

    // [spec:et:sem:tensor-ptr.executorch.extension.clone-tensor-ptr-to-fn/test]
    #[test]
    fn tensor_ptr_device_test_cpu_to_device_tensor() {
        let _guard = setup();
        let cpu_tensor =
            make_owning_f32(alloc::vec![2, 3], alloc::vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let device_tensor = clone_tensor_ptr_to(&cpu_tensor, Device::from_type(DeviceType::CUDA));

        let dt = device_tensor.tensor();
        assert_eq!(dt.dim(), 2);
        assert_eq!(dt.size(0), 2);
        assert_eq!(dt.size(1), 3);
        assert_eq!(dt.scalar_type(), ScalarType::Float);
        assert!(!dt.const_data_ptr_typed().is_null());
        assert_ne!(
            dt.const_data_ptr_typed(),
            cpu_tensor.tensor().const_data_ptr_typed()
        );

        assert_eq!(
            unsafe { (*dt.unsafe_get_tensor_impl()).device_type() },
            DeviceType::CUDA
        );
        assert_eq!(unsafe { (*dt.unsafe_get_tensor_impl()).device_index() }, 0);

        assert_eq!(unsafe { (*g_mock_cuda()).allocate_count_ }, 1);
        assert_eq!(unsafe { (*g_mock_cuda()).h2d_count_ }, 1);
    }

    // [spec:et:sem:tensor-ptr.executorch.extension.clone-tensor-ptr-to-fn/test]
    #[test]
    fn tensor_ptr_device_test_cpu_to_device_from_raw_data() {
        let _guard = setup();
        let mut data: [f32; 4] = [10.0, 20.0, 30.0, 40.0];
        let cpu_tensor = make_raw_f32(alloc::vec![2, 2], data.as_mut_ptr());
        let device_tensor = clone_tensor_ptr_to(&cpu_tensor, Device::from_type(DeviceType::CUDA));

        let dt = device_tensor.tensor();
        assert_eq!(dt.dim(), 2);
        assert_eq!(dt.size(0), 2);
        assert_eq!(dt.size(1), 2);
        assert_eq!(dt.scalar_type(), ScalarType::Float);
        assert!(!dt.const_data_ptr_typed().is_null());
        assert_ne!(
            dt.const_data_ptr_typed(),
            data.as_ptr() as *const core::ffi::c_void
        );

        assert_eq!(
            unsafe { (*dt.unsafe_get_tensor_impl()).device_type() },
            DeviceType::CUDA
        );

        assert_eq!(unsafe { (*g_mock_cuda()).allocate_count_ }, 1);
        assert_eq!(unsafe { (*g_mock_cuda()).h2d_count_ }, 1);
    }

    // [spec:et:sem:tensor-ptr.executorch.extension.clone-tensor-ptr-to-fn/test]
    #[test]
    fn tensor_ptr_device_test_device_to_cpu_tensor() {
        let _guard = setup();
        let cpu_tensor =
            make_owning_f32(alloc::vec![2, 3], alloc::vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let device_tensor = clone_tensor_ptr_to(&cpu_tensor, Device::from_type(DeviceType::CUDA));
        let result_tensor = clone_tensor_ptr_to(&device_tensor, Device::from_type(DeviceType::CPU));

        let rt = result_tensor.tensor();
        assert_eq!(rt.dim(), 2);
        assert_eq!(rt.size(0), 2);
        assert_eq!(rt.size(1), 3);
        assert_eq!(rt.scalar_type(), ScalarType::Float);

        let result_data = rt.const_data_ptr::<f32>();
        let original_data = cpu_tensor.tensor().const_data_ptr::<f32>();
        for i in 0..6 {
            assert_eq!(unsafe { *result_data.add(i) }, unsafe {
                *original_data.add(i)
            });
        }

        assert_eq!(unsafe { (*g_mock_cuda()).d2h_count_ }, 1);
    }

    // [spec:et:sem:tensor-ptr.executorch.extension.clone-tensor-ptr-to-fn/test]
    #[test]
    fn tensor_ptr_device_test_device_to_cpu_preserves_shape_dynamism() {
        let _guard = setup();
        let cpu_tensor = make_tensor_ptr_from_vec(
            alloc::vec![2],
            alloc::vec![1.0f32, 2.0],
            Vec::new(),
            Vec::new(),
            ScalarType::Float,
            TensorShapeDynamism::STATIC,
        );
        let device_tensor = clone_tensor_ptr_to(&cpu_tensor, Device::from_type(DeviceType::CUDA));
        let result_tensor = clone_tensor_ptr_to(&device_tensor, Device::from_type(DeviceType::CPU));

        assert_eq!(
            result_tensor.tensor().shape_dynamism(),
            TensorShapeDynamism::STATIC
        );
    }

    // [spec:et:sem:tensor-ptr.executorch.extension.clone-tensor-ptr-to-fn/test]
    #[test]
    fn tensor_ptr_device_test_roundtrip_cpu_device_cpu() {
        let _guard = setup();
        let original: Vec<f32> = alloc::vec![1.5, 2.5, 3.5, 4.5, 5.5, 6.5];
        let cpu_tensor = make_owning_f32(alloc::vec![2, 3], original.clone());

        let device_tensor = clone_tensor_ptr_to(&cpu_tensor, Device::from_type(DeviceType::CUDA));
        let roundtrip_tensor =
            clone_tensor_ptr_to(&device_tensor, Device::from_type(DeviceType::CPU));

        assert_ne!(
            roundtrip_tensor.tensor().const_data_ptr_typed(),
            cpu_tensor.tensor().const_data_ptr_typed()
        );
        assert_ne!(
            roundtrip_tensor.tensor().const_data_ptr_typed(),
            device_tensor.tensor().const_data_ptr_typed()
        );

        let result_data = roundtrip_tensor.tensor().const_data_ptr::<f32>();
        for i in 0..original.len() {
            assert_eq!(unsafe { *result_data.add(i) }, original[i]);
        }

        assert_eq!(roundtrip_tensor.tensor().dim(), cpu_tensor.tensor().dim());
        assert_eq!(
            roundtrip_tensor.tensor().size(0),
            cpu_tensor.tensor().size(0)
        );
        assert_eq!(
            roundtrip_tensor.tensor().size(1),
            cpu_tensor.tensor().size(1)
        );
        assert_eq!(
            roundtrip_tensor.tensor().scalar_type(),
            cpu_tensor.tensor().scalar_type()
        );
    }

    // [spec:et:sem:tensor-ptr.executorch.extension.clone-tensor-ptr-to-fn/test]
    #[test]
    fn tensor_ptr_device_test_roundtrip_int32() {
        let _guard = setup();
        let cpu_tensor = make_tensor_ptr_from_vec(
            alloc::vec![4],
            alloc::vec![10i32, 20, 30, 40],
            Vec::new(),
            Vec::new(),
            ScalarType::Int,
            TensorShapeDynamism::DYNAMIC_BOUND,
        );

        let device_tensor = clone_tensor_ptr_to(&cpu_tensor, Device::from_type(DeviceType::CUDA));
        let roundtrip = clone_tensor_ptr_to(&device_tensor, Device::from_type(DeviceType::CPU));

        assert_eq!(roundtrip.tensor().scalar_type(), ScalarType::Int);
        let expected: Vec<i32> = alloc::vec![10, 20, 30, 40];
        let data = roundtrip.tensor().const_data_ptr::<i32>();
        for i in 0..expected.len() {
            assert_eq!(unsafe { *data.add(i) }, expected[i]);
        }
    }

    // [spec:et:sem:tensor-ptr.executorch.extension.clone-tensor-ptr-to-fn/test]
    #[test]
    fn tensor_ptr_device_test_device_index_propagation() {
        let _guard = setup();
        let cpu_tensor = make_owning_f32(alloc::vec![2], alloc::vec![1.0, 2.0]);
        let device_tensor =
            clone_tensor_ptr_to(&cpu_tensor, Device::new(DeviceType::CUDA, /*index=*/ 1));

        assert_eq!(
            unsafe { (*device_tensor.tensor().unsafe_get_tensor_impl()).device_index() },
            1
        );

        let roundtrip = clone_tensor_ptr_to(&device_tensor, Device::from_type(DeviceType::CPU));
        assert_eq!(
            unsafe { *roundtrip.tensor().const_data_ptr::<f32>().add(0) },
            1.0
        );
        assert_eq!(
            unsafe { *roundtrip.tensor().const_data_ptr::<f32>().add(1) },
            2.0
        );
    }

    // [spec:et:sem:tensor-ptr.executorch.extension.clone-tensor-ptr-to-fn/test]
    #[test]
    fn tensor_ptr_device_test_device_memory_cleanup() {
        let _guard = setup();
        {
            let cpu_tensor = make_owning_f32(alloc::vec![2], alloc::vec![1.0, 2.0]);
            let _device_tensor =
                clone_tensor_ptr_to(&cpu_tensor, Device::from_type(DeviceType::CUDA));
            assert_eq!(unsafe { (*g_mock_cuda()).allocate_count_ }, 1);
            assert_eq!(unsafe { (*g_mock_cuda()).deallocate_count_ }, 0);
        }
        assert_eq!(unsafe { (*g_mock_cuda()).deallocate_count_ }, 1);
    }

    // [spec:et:sem:tensor-ptr.executorch.extension.clone-tensor-ptr-to-fn/test]
    #[test]
    fn tensor_ptr_device_test_scalar_tensor_roundtrip() {
        let _guard = setup();
        let cpu_tensor = make_owning_f32(alloc::vec![], alloc::vec![42.0]);
        let device_tensor = clone_tensor_ptr_to(&cpu_tensor, Device::from_type(DeviceType::CUDA));

        assert_eq!(device_tensor.tensor().dim(), 0);
        assert_eq!(device_tensor.tensor().numel(), 1);

        let roundtrip = clone_tensor_ptr_to(&device_tensor, Device::from_type(DeviceType::CPU));
        assert_eq!(roundtrip.tensor().dim(), 0);
        assert_eq!(roundtrip.tensor().numel(), 1);
        assert_eq!(
            unsafe { *roundtrip.tensor().const_data_ptr::<f32>().add(0) },
            42.0
        );
    }

    // [spec:et:sem:tensor-ptr.executorch.extension.clone-tensor-ptr-to-fn/test]
    #[test]
    fn tensor_ptr_device_test_raw_data_roundtrip() {
        let _guard = setup();
        let mut raw_data: [f32; 3] = [100.0, 200.0, 300.0];
        let cpu_tensor = make_raw_f32(alloc::vec![3], raw_data.as_mut_ptr());
        let device_tensor = clone_tensor_ptr_to(&cpu_tensor, Device::from_type(DeviceType::CUDA));
        let roundtrip = clone_tensor_ptr_to(&device_tensor, Device::from_type(DeviceType::CPU));

        assert_eq!(roundtrip.tensor().dim(), 1);
        assert_eq!(roundtrip.tensor().size(0), 3);
        let data = roundtrip.tensor().const_data_ptr::<f32>();
        assert_eq!(unsafe { *data.add(0) }, 100.0);
        assert_eq!(unsafe { *data.add(1) }, 200.0);
        assert_eq!(unsafe { *data.add(2) }, 300.0);
    }

    // PORT-NOTE: `ET_EXPECT_DEATH` death test. `runtime_abort` -> `libc::abort()`
    // terminates the process (not an unwind), so `#[should_panic]` cannot catch
    // it; ported and `#[ignore]`d, matching the death-test convention used across
    // the port.
    // [spec:et:sem:tensor-ptr.executorch.extension.clone-tensor-ptr-to-fn/test]
    #[test]
    #[should_panic]
    #[ignore]
    fn tensor_ptr_device_test_error_cpu_to_cpu() {
        let _guard = setup();
        let cpu_tensor = make_owning_f32(alloc::vec![2], alloc::vec![1.0, 2.0]);
        let _ = clone_tensor_ptr_to(&cpu_tensor, Device::from_type(DeviceType::CPU));
    }

    // [spec:et:sem:tensor-ptr.executorch.extension.clone-tensor-ptr-to-fn/test]
    #[test]
    #[should_panic]
    #[ignore]
    fn tensor_ptr_device_test_error_null_cpu_tensor_data() {
        let _guard = setup();
        let null_tensor = make_tensor_ptr_simple(
            alloc::vec![2, 2],
            core::ptr::null_mut(),
            ScalarType::Float,
            TensorShapeDynamism::DYNAMIC_BOUND,
            None,
            Device::from_type(DeviceType::CPU),
        );
        let _ = clone_tensor_ptr_to(&null_tensor, Device::from_type(DeviceType::CUDA));
    }

    // [spec:et:sem:tensor-ptr.executorch.extension.clone-tensor-ptr-to-fn/test]
    #[test]
    #[should_panic]
    #[ignore]
    fn tensor_ptr_device_test_error_device_to_device() {
        let _guard = setup();
        let cpu_tensor = make_owning_f32(alloc::vec![2], alloc::vec![1.0, 2.0]);
        let device_tensor = clone_tensor_ptr_to(&cpu_tensor, Device::from_type(DeviceType::CUDA));
        let _ = clone_tensor_ptr_to(&device_tensor, Device::new(DeviceType::CUDA, /*index=*/ 1));
    }

    // [spec:et:sem:tensor-ptr.executorch.extension.clone-tensor-ptr-to-fn/test]
    #[test]
    fn tensor_ptr_device_test_make_tensor_ptr_vector_to_device() {
        let _guard = setup();
        let cpu_tensor = make_owning_f32(alloc::vec![2, 2], alloc::vec![1.0, 2.0, 3.0, 4.0]);
        let device_tensor = clone_tensor_ptr_to(&cpu_tensor, Device::from_type(DeviceType::CUDA));

        let dt = device_tensor.tensor();
        assert_eq!(dt.dim(), 2);
        assert_eq!(dt.size(0), 2);
        assert_eq!(dt.size(1), 2);
        assert_eq!(dt.scalar_type(), ScalarType::Float);
        assert_eq!(
            unsafe { (*dt.unsafe_get_tensor_impl()).device_type() },
            DeviceType::CUDA
        );
        assert_eq!(unsafe { (*g_mock_cuda()).allocate_count_ }, 1);
        assert_eq!(unsafe { (*g_mock_cuda()).h2d_count_ }, 1);

        let roundtrip = clone_tensor_ptr_to(&device_tensor, Device::from_type(DeviceType::CPU));
        let data = roundtrip.tensor().const_data_ptr::<f32>();
        assert_eq!(unsafe { *data.add(0) }, 1.0);
        assert_eq!(unsafe { *data.add(1) }, 2.0);
        assert_eq!(unsafe { *data.add(2) }, 3.0);
        assert_eq!(unsafe { *data.add(3) }, 4.0);
    }

    // [spec:et:sem:tensor-ptr.executorch.extension.clone-tensor-ptr-to-fn/test]
    #[test]
    fn tensor_ptr_device_test_make_tensor_ptr_raw_pointer_to_device() {
        let _guard = setup();
        let mut raw: [f32; 3] = [5.0, 6.0, 7.0];
        let cpu_tensor = make_raw_f32(alloc::vec![3], raw.as_mut_ptr());
        let device_tensor = clone_tensor_ptr_to(&cpu_tensor, Device::from_type(DeviceType::CUDA));

        let dt = device_tensor.tensor();
        assert_eq!(dt.dim(), 1);
        assert_eq!(dt.size(0), 3);
        assert_eq!(
            unsafe { (*dt.unsafe_get_tensor_impl()).device_type() },
            DeviceType::CUDA
        );
        assert_ne!(
            dt.const_data_ptr_typed(),
            raw.as_ptr() as *const core::ffi::c_void
        );
        assert_eq!(unsafe { (*g_mock_cuda()).allocate_count_ }, 1);
        assert_eq!(unsafe { (*g_mock_cuda()).h2d_count_ }, 1);

        let roundtrip = clone_tensor_ptr_to(&device_tensor, Device::from_type(DeviceType::CPU));
        let data = roundtrip.tensor().const_data_ptr::<f32>();
        assert_eq!(unsafe { *data.add(0) }, 5.0);
        assert_eq!(unsafe { *data.add(1) }, 6.0);
        assert_eq!(unsafe { *data.add(2) }, 7.0);
    }

    // [spec:et:sem:tensor-ptr.executorch.extension.clone-tensor-ptr-to-fn/test]
    #[test]
    fn tensor_ptr_device_test_clone_to_cpu_verifies_cpu_device_metadata() {
        let _guard = setup();
        let cpu_tensor = make_owning_f32(alloc::vec![3], alloc::vec![1.0, 2.0, 3.0]);
        let device_tensor = clone_tensor_ptr_to(&cpu_tensor, Device::from_type(DeviceType::CUDA));
        let result = clone_tensor_ptr_to(&device_tensor, Device::from_type(DeviceType::CPU));

        assert_eq!(
            unsafe { (*result.tensor().unsafe_get_tensor_impl()).device_type() },
            DeviceType::CPU
        );
        assert_eq!(
            unsafe { (*result.tensor().unsafe_get_tensor_impl()).device_index() },
            0
        );
    }

    // [spec:et:sem:tensor-ptr.executorch.extension.clone-tensor-ptr-to-fn/test]
    #[test]
    fn tensor_ptr_device_test_multiple_clones_from_same_source() {
        let _guard = setup();
        let cpu_tensor = make_owning_f32(alloc::vec![3], alloc::vec![1.0, 2.0, 3.0]);
        let device1 = clone_tensor_ptr_to(&cpu_tensor, Device::from_type(DeviceType::CUDA));
        let device2 = clone_tensor_ptr_to(&cpu_tensor, Device::from_type(DeviceType::CUDA));

        assert_ne!(
            device1.tensor().const_data_ptr_typed(),
            device2.tensor().const_data_ptr_typed()
        );
        assert_eq!(unsafe { (*g_mock_cuda()).allocate_count_ }, 2);
        assert_eq!(unsafe { (*g_mock_cuda()).h2d_count_ }, 2);
    }

    // [spec:et:sem:tensor-ptr.executorch.extension.clone-tensor-ptr-to-fn/test]
    #[test]
    fn tensor_ptr_device_test_high_dimensional_tensor_roundtrip() {
        let _guard = setup();
        let mut data: Vec<f32> = alloc::vec![0.0f32; 24];
        for i in 0..24 {
            data[i] = i as f32;
        }
        let cpu_tensor = make_owning_f32(alloc::vec![2, 3, 4], data.clone());
        let device_tensor = clone_tensor_ptr_to(&cpu_tensor, Device::from_type(DeviceType::CUDA));

        let dt = device_tensor.tensor();
        assert_eq!(dt.dim(), 3);
        assert_eq!(dt.size(0), 2);
        assert_eq!(dt.size(1), 3);
        assert_eq!(dt.size(2), 4);

        let roundtrip = clone_tensor_ptr_to(&device_tensor, Device::from_type(DeviceType::CPU));
        let result = roundtrip.tensor().const_data_ptr::<f32>();
        for i in 0..24 {
            assert_eq!(unsafe { *result.add(i) }, i as f32);
        }
    }

    // [spec:et:sem:tensor-ptr.executorch.extension.clone-tensor-ptr-to-fn/test]
    #[test]
    fn tensor_ptr_device_test_roundtrip_double() {
        let _guard = setup();
        let cpu_tensor = make_tensor_ptr_from_vec(
            alloc::vec![3],
            alloc::vec![1.1f64, 2.2, 3.3],
            Vec::new(),
            Vec::new(),
            ScalarType::Double,
            TensorShapeDynamism::DYNAMIC_BOUND,
        );
        let device_tensor = clone_tensor_ptr_to(&cpu_tensor, Device::from_type(DeviceType::CUDA));
        let roundtrip = clone_tensor_ptr_to(&device_tensor, Device::from_type(DeviceType::CPU));

        assert_eq!(roundtrip.tensor().scalar_type(), ScalarType::Double);
        let data = roundtrip.tensor().const_data_ptr::<f64>();
        assert_eq!(unsafe { *data.add(0) }, 1.1);
        assert_eq!(unsafe { *data.add(1) }, 2.2);
        assert_eq!(unsafe { *data.add(2) }, 3.3);
    }

    // [spec:et:sem:tensor-ptr.executorch.extension.clone-tensor-ptr-to-fn/test]
    #[test]
    fn tensor_ptr_device_test_roundtrip_int64() {
        let _guard = setup();
        let cpu_tensor = make_tensor_ptr_from_vec(
            alloc::vec![3],
            alloc::vec![100i64, 200, 300],
            Vec::new(),
            Vec::new(),
            ScalarType::Long,
            TensorShapeDynamism::DYNAMIC_BOUND,
        );
        let device_tensor = clone_tensor_ptr_to(&cpu_tensor, Device::from_type(DeviceType::CUDA));
        let roundtrip = clone_tensor_ptr_to(&device_tensor, Device::from_type(DeviceType::CPU));

        assert_eq!(roundtrip.tensor().scalar_type(), ScalarType::Long);
        let data = roundtrip.tensor().const_data_ptr::<i64>();
        assert_eq!(unsafe { *data.add(0) }, 100);
        assert_eq!(unsafe { *data.add(1) }, 200);
        assert_eq!(unsafe { *data.add(2) }, 300);
    }

    // [spec:et:sem:tensor-ptr.executorch.extension.clone-tensor-ptr-to-fn/test]
    #[test]
    fn tensor_ptr_device_test_large_tensor_roundtrip() {
        let _guard = setup();
        let n: usize = 10000;
        let mut data: Vec<f32> = alloc::vec![0.0f32; n];
        for i in 0..n {
            data[i] = (i as f32) * 0.1;
        }
        let cpu_tensor = make_owning_f32(alloc::vec![n as i32], data.clone());
        let device_tensor = clone_tensor_ptr_to(&cpu_tensor, Device::from_type(DeviceType::CUDA));
        let roundtrip = clone_tensor_ptr_to(&device_tensor, Device::from_type(DeviceType::CPU));

        let result = roundtrip.tensor().const_data_ptr::<f32>();
        for i in 0..n {
            assert_eq!(unsafe { *result.add(i) }, data[i]);
        }
    }

    // PORT-NOTE: no C++ counterpart test; death test (`fail` calls
    // `ET_CHECK_MSG(false, ...)`, a fatal abort), so `#[should_panic]` +
    // `#[ignore]`d like the other death tests in this file. Drives the
    // `ET_SWITCH_REALHBBF16_AND_UINT_TYPES` default arm: `can_cast(Float,
    // ComplexFloat)` is true, but ComplexFloat is outside the switch's type set,
    // so the switch invokes `ctx.fail(InvalidArgument)`, which never returns.
    // [spec:et:sem:tensor-ptr.executorch.extension.fail-fn/test]
    #[test]
    #[should_panic]
    #[ignore]
    fn tensor_ptr_test_unsupported_dtype_fail_ctx_aborts() {
        let _guard = setup();
        let _ = make_tensor_ptr_from_vec(
            alloc::vec![2],
            alloc::vec![1.0f32, 2.0],
            Vec::new(),
            Vec::new(),
            ScalarType::ComplexFloat,
            TensorShapeDynamism::DYNAMIC_BOUND,
        );
    }
}
