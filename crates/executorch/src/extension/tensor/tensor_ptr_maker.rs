//! Literal port of extension/tensor/tensor_ptr_maker.cpp + extension/tensor/tensor_ptr_maker.h.

extern crate alloc;

use alloc::vec::Vec;

use crate::extension::tensor::tensor_ptr::{FailCtx, TensorPtr, make_tensor_ptr};
use crate::runtime::core::exec_aten::util::scalar_type_util::element_size;
use crate::runtime::core::portable_type::device::{Device, DeviceType};
use crate::runtime::core::portable_type::scalar::Scalar;
use crate::runtime::core::portable_type::scalar_type::ScalarType;
use crate::runtime::core::portable_type::tensor_impl::{
    DimOrderType, SizesType, StridesType, safe_numel, ssize_t,
};
use crate::runtime::core::portable_type::{BFloat16, Half};
use crate::runtime::core::result::ResultExt;
use crate::runtime::core::tensor_shape_dynamism::TensorShapeDynamism;

// PORT-NOTE: `ET_CHECK_MSG` local mirror (see tensor_ptr.rs). Fatal abort.
macro_rules! et_check_msg {
    ($cond:expr, $($arg:tt)*) => {
        if !($cond) {
            crate::runtime::platform::abort::runtime_abort();
        }
    };
}

// PORT-NOTE: reuse the local REALHBBF16_AND_UINT switch macro defined in
// tensor_ptr.rs (see the PORT-NOTE there). Unresolved cross-module reference:
// belongs in scalar_type_util.rs.
use crate::extension::tensor::tensor_ptr::et_switch_realhbbf16_and_uint_types;

// PORT-NOTE: `c10::mul_overflows` inline port (see tensor_ptr.rs).
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

// ---- extract_scalar ----
//
// PORT-NOTE: The three SFINAE-selected `extract_scalar(Scalar, T*)` overloads
// (integral non-bool, floating/Half/BFloat16, bool) are modeled by the
// `ExtractScalar` trait with one method per target ctype. `ET_EXTRACT_SCALAR`
// aborts on a `false` return.

pub trait ExtractScalar: Sized {
    fn extract_scalar(scalar: Scalar, out_val: &mut Self) -> bool;
}

// Integral non-bool overload.
macro_rules! extract_scalar_int {
    ($($t:ty),*) => {$(
        impl ExtractScalar for $t {
            fn extract_scalar(scalar: Scalar, out_val: &mut Self) -> bool {
                if !scalar.is_integral(/*include_bool=*/ false) {
                    return false;
                }
                let val: i64 = scalar.to_i64();
                if val < <$t>::MIN as i64 || val > <$t>::MAX as i64 {
                    return false;
                }
                *out_val = val as $t;
                true
            }
        }
    )*};
}
extract_scalar_int!(u8, i8, i16, i32, i64, u16, u32, u64);

// Floating-point / Half / BFloat16 overload.
// [spec:et:def:tensor-ptr-maker.executorch.extension.extract-scalar-fn]
// [spec:et:sem:tensor-ptr-maker.executorch.extension.extract-scalar-fn]
macro_rules! extract_scalar_float {
    ($t:ty, $lowest:expr, $max:expr, $from_f64:expr) => {
        impl ExtractScalar for $t {
            fn extract_scalar(scalar: Scalar, out_val: &mut Self) -> bool {
                let val: f64;
                if scalar.is_floating_point() {
                    val = scalar.to_f64();
                    if val.is_finite() && (val < ($lowest) || val > ($max)) {
                        return false;
                    }
                } else if scalar.is_integral(/*include_bool=*/ false) {
                    val = scalar.to_i64() as f64;
                } else {
                    return false;
                }
                *out_val = ($from_f64)(val);
                true
            }
        }
    };
}
extract_scalar_float!(f32, f32::MIN as f64, f32::MAX as f64, |v: f64| v as f32);
extract_scalar_float!(f64, f64::MIN, f64::MAX, |v: f64| v);
extract_scalar_float!(Half, Half::MIN.to_f64(), Half::MAX.to_f64(), |v: f64| {
    Half::from_f64(v)
});
extract_scalar_float!(
    BFloat16,
    BFloat16::MIN.to_f64(),
    BFloat16::MAX.to_f64(),
    |v: f64| BFloat16::from_f64(v)
);

// Bool overload.
impl ExtractScalar for bool {
    fn extract_scalar(scalar: Scalar, out_val: &mut Self) -> bool {
        if scalar.is_integral(/*include_bool=*/ false) {
            *out_val = scalar.to_i64() != 0;
            return true;
        }
        if scalar.is_boolean() {
            *out_val = scalar.to_bool_val();
            return true;
        }
        false
    }
}

// #define ET_EXTRACT_SCALAR(scalar, out_val)
macro_rules! et_extract_scalar {
    ($scalar:expr, $out_val:expr) => {
        et_check_msg!(
            ExtractScalar::extract_scalar($scalar, &mut $out_val),
            "scalar could not be extracted: wrong type or out of range"
        );
    };
}

// ---- RNG ----
//
// PORT-NOTE: The C++ uses `std::default_random_engine gen{std::random_device{}()}`
// plus `std::uniform_real_distribution<float>`, `std::normal_distribution<float>`,
// and `std::uniform_int_distribution<int64_t>`. Rust std ships no RNG, so the
// engine and distributions are ported inline. `std::default_random_engine` in
// libstdc++ is `minstd_rand0` (a linear-congruential engine, a=16807, c=0,
// m=2^31-1); it is reproduced as `Minstd0`. The distributions are literal
// reimplementations (canonical real, rejection int, Box-Muller normal). Exact
// bit-for-bit parity with libstdc++ is not guaranteed but the statistical
// contracts (ranges, distributions) match. Seeded non-deterministically per
// call from an OS/time entropy source, as the C++ `std::random_device` is.

struct Minstd0 {
    state: u64,
}
impl Minstd0 {
    const A: u64 = 16807;
    const M: u64 = 2147483647; // 2^31 - 1
    fn new(seed: u32) -> Self {
        // libstdc++ seeds so the state is in [1, m-1].
        let mut s = (seed as u64) % Self::M;
        if s == 0 {
            s = 1;
        }
        Minstd0 { state: s }
    }
    fn next_u32(&mut self) -> u32 {
        self.state = (self.state * Self::A) % Self::M;
        self.state as u32
    }
    const MIN: u64 = 1;
    const MAX: u64 = Self::M - 1;
    // Canonical real in [0, 1).
    fn canonical_f32(&mut self) -> f32 {
        let range = (Self::MAX - Self::MIN + 1) as f64; // m - 1
        let x = (self.next_u32() as u64 - Self::MIN) as f64;
        (x / range) as f32
    }
}

// PORT-NOTE: `std::random_device{}()` — a non-deterministic 32-bit seed. Derived
// here from the current time and a stack address; no user-controllable seed,
// matching the C++.
fn random_device_seed() -> u32 {
    use core::sync::atomic::{AtomicU32, Ordering};
    static COUNTER: AtomicU32 = AtomicU32::new(0);
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u32)
        .unwrap_or(0);
    let stack_marker = 0u8;
    let addr = (&stack_marker as *const u8) as usize as u32;
    now ^ addr ^ COUNTER.fetch_add(0x9E3779B9, Ordering::Relaxed)
}

trait Distribution {
    fn sample(&mut self, rng: &mut Minstd0) -> f64;
}

struct UniformRealDistribution {
    a: f32,
    b: f32,
}
impl Distribution for UniformRealDistribution {
    fn sample(&mut self, rng: &mut Minstd0) -> f64 {
        (self.a + (self.b - self.a) * rng.canonical_f32()) as f64
    }
}

struct NormalDistribution {
    mean: f32,
    stddev: f32,
    cached: Option<f32>,
}
impl Distribution for NormalDistribution {
    fn sample(&mut self, rng: &mut Minstd0) -> f64 {
        if let Some(v) = self.cached.take() {
            return (self.mean + self.stddev * v) as f64;
        }
        // Box-Muller.
        loop {
            let u1 = rng.canonical_f32();
            let u2 = rng.canonical_f32();
            if u1 <= f32::EPSILON {
                continue;
            }
            let mag = (-2.0f32 * u1.ln()).sqrt();
            let z0 = mag * (core::f32::consts::TAU * u2).cos();
            let z1 = mag * (core::f32::consts::TAU * u2).sin();
            self.cached = Some(z1);
            return (self.mean + self.stddev * z0) as f64;
        }
    }
}

struct UniformIntDistribution {
    low: i64,
    high: i64, // inclusive upper bound (already high-1 at construction)
}
impl Distribution for UniformIntDistribution {
    fn sample(&mut self, rng: &mut Minstd0) -> f64 {
        let range = (self.high as i128 - self.low as i128 + 1) as u128;
        // Rejection sampling over the 32-bit engine output range.
        let engine_range = (Minstd0::MAX - Minstd0::MIN + 1) as u128;
        if range == 0 {
            return self.low as f64;
        }
        let limit = engine_range - (engine_range % range);
        loop {
            let r = (rng.next_u32() as u64 - Minstd0::MIN) as u128;
            if r < limit {
                let v = self.low as i128 + (r % range) as i128;
                return v as f64;
            }
        }
    }
}

// PORT-NOTE: The switch-selected cast `static_cast<CTYPE>(distribution(rng))`
// takes the `f64` sample and narrows to the element ctype. `FromSample`
// supplies the per-ctype cast, mirroring `static_cast` truncation/wrap.
trait FromSample {
    fn from_sample(v: f64) -> Self;
}
macro_rules! from_sample_prim {
    ($($t:ty),*) => {$(
        impl FromSample for $t { fn from_sample(v: f64) -> Self { v as $t } }
    )*};
}
from_sample_prim!(u8, i8, i16, i32, i64, f32, f64, u16, u32, u64);
impl FromSample for bool {
    fn from_sample(v: f64) -> Self {
        v != 0.0
    }
}
impl FromSample for Half {
    fn from_sample(v: f64) -> Self {
        Half::from_f64(v)
    }
}
impl FromSample for BFloat16 {
    fn from_sample(v: f64) -> Self {
        BFloat16::from_f64(v)
    }
}

// [spec:et:def:tensor-ptr-maker.executorch.extension.random-strided-fn]
// [spec:et:sem:tensor-ptr-maker.executorch.extension.random-strided-fn]
fn random_strided<D: Distribution>(
    sizes: Vec<SizesType>,
    strides: Vec<StridesType>,
    type_: ScalarType,
    dynamism: TensorShapeDynamism,
    mut distribution: D,
) -> TensorPtr {
    let tensor = empty_strided(sizes, strides, type_, dynamism);
    let mut rng = Minstd0::new(random_device_seed());

    let ctx = FailCtx::new("Unsupported dtype in random_strided");

    et_switch_realhbbf16_and_uint_types!(type_, ctx, "random_strided", CTYPE, {
        let t = tensor.tensor();
        let ptr = t.mutable_data_ptr::<CTYPE>();
        let numel = t.numel();
        for i in 0..numel {
            unsafe {
                *ptr.offset(i) = <CTYPE as FromSample>::from_sample(distribution.sample(&mut rng));
            }
        }
    });
    tensor
}

// [spec:et:def:tensor-ptr-maker.executorch.extension.empty-strided-fn]
// [spec:et:sem:tensor-ptr-maker.executorch.extension.empty-strided-fn]
pub fn empty_strided(
    sizes: Vec<SizesType>,
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
    let data: Vec<u8> = alloc::vec![0u8; nbytes];
    crate::extension::tensor::tensor_ptr::make_tensor_ptr_from_bytes(
        sizes,
        data,
        Vec::new(),
        strides,
        type_,
        dynamism,
    )
}

// [spec:et:def:tensor-ptr-maker.executorch.extension.full-strided-fn]
// [spec:et:sem:tensor-ptr-maker.executorch.extension.full-strided-fn]
pub fn full_strided(
    sizes: Vec<SizesType>,
    strides: Vec<StridesType>,
    fill_value: Scalar,
    type_: ScalarType,
    dynamism: TensorShapeDynamism,
) -> TensorPtr {
    let tensor = empty_strided(sizes, strides, type_, dynamism);
    let ctx = FailCtx::new("Unsupported data type in full_strided");

    et_switch_realhbbf16_and_uint_types!(type_, ctx, "full_strided", CTYPE, {
        let mut value: CTYPE = <CTYPE as DefaultFill>::default_fill();
        et_extract_scalar!(fill_value, value);
        let t = tensor.tensor();
        let ptr = t.mutable_data_ptr::<CTYPE>();
        let numel = t.numel();
        for i in 0..numel {
            unsafe {
                *ptr.offset(i) = value;
            }
        }
    });
    tensor
}

// PORT-NOTE: `CTYPE value;` in C++ is default-uninitialized; Rust requires a
// value before `ET_EXTRACT_SCALAR` overwrites it. `DefaultFill` supplies a
// placeholder zero of the ctype.
trait DefaultFill: Copy {
    fn default_fill() -> Self;
}
macro_rules! default_fill_prim {
    ($($t:ty = $z:expr),*) => {$(
        impl DefaultFill for $t { fn default_fill() -> Self { $z } }
    )*};
}
default_fill_prim!(
    u8 = 0,
    i8 = 0,
    i16 = 0,
    i32 = 0,
    i64 = 0,
    f32 = 0.0,
    f64 = 0.0,
    u16 = 0,
    u32 = 0,
    u64 = 0,
    bool = false
);
impl DefaultFill for Half {
    fn default_fill() -> Self {
        Half::from_f32(0.0)
    }
}
impl DefaultFill for BFloat16 {
    fn default_fill() -> Self {
        BFloat16::from_f32(0.0)
    }
}

// [spec:et:def:tensor-ptr-maker.executorch.extension.rand-strided-fn]
// [spec:et:sem:tensor-ptr-maker.executorch.extension.rand-strided-fn]
pub fn rand_strided(
    sizes: Vec<SizesType>,
    strides: Vec<StridesType>,
    type_: ScalarType,
    dynamism: TensorShapeDynamism,
) -> TensorPtr {
    let mut upper_bound = 1.0f32;
    // Adjusts the upper bound to prevent rounding to 1.0 when converting to
    // lower-precision types.
    if type_ == ScalarType::Half {
        upper_bound -= f32::from(Half::EPSILON) / 2.0;
    } else if type_ == ScalarType::BFloat16 {
        upper_bound -= f32::from(BFloat16::EPSILON) / 2.0;
    }
    random_strided(
        sizes,
        strides,
        type_,
        dynamism,
        UniformRealDistribution {
            a: 0.0,
            b: upper_bound,
        },
    )
}

// [spec:et:def:tensor-ptr-maker.executorch.extension.randn-strided-fn]
// [spec:et:sem:tensor-ptr-maker.executorch.extension.randn-strided-fn]
pub fn randn_strided(
    sizes: Vec<SizesType>,
    strides: Vec<StridesType>,
    type_: ScalarType,
    dynamism: TensorShapeDynamism,
) -> TensorPtr {
    random_strided(
        sizes,
        strides,
        type_,
        dynamism,
        NormalDistribution {
            mean: 0.0,
            stddev: 1.0,
            cached: None,
        },
    )
}

// [spec:et:def:tensor-ptr-maker.executorch.extension.randint-strided-fn]
// [spec:et:sem:tensor-ptr-maker.executorch.extension.randint-strided-fn]
pub fn randint_strided(
    low: i64,
    high: i64,
    sizes: Vec<SizesType>,
    strides: Vec<StridesType>,
    type_: ScalarType,
    dynamism: TensorShapeDynamism,
) -> TensorPtr {
    random_strided(
        sizes,
        strides,
        type_,
        dynamism,
        UniformIntDistribution {
            low,
            high: high - 1,
        },
    )
}

// ==== tensor_ptr_maker.h ====

/// A helper class for creating TensorPtr instances from raw data and tensor
/// properties. The TensorPtr created by this class does not own the data, so
/// the data must outlive the TensorPtr.
// [spec:et:def:tensor-ptr-maker.executorch.extension.tensor-ptr-maker]
// [spec:et:def:tensor-ptr-maker.executorch.extension.tensor-ptr-maker.operator-fn]
// [spec:et:sem:tensor-ptr-maker.executorch.extension.tensor-ptr-maker.operator-fn]
// PORT-NOTE: C++ deletes copy/assign and keeps move-only; Rust builder is move-
// only by not deriving `Clone`.
pub struct TensorPtrMaker {
    sizes_: Vec<SizesType>,
    strides_: Vec<StridesType>,
    dim_order_: Vec<DimOrderType>,
    deleter_: Option<crate::extension::tensor::tensor_ptr::Deleter>,
    data_: *mut core::ffi::c_void,
    type_: ScalarType,
    dynamism_: TensorShapeDynamism,
}

impl TensorPtrMaker {
    /// Sets the scalar type of the tensor elements.
    // [spec:et:def:tensor-ptr-maker.executorch.extension.tensor-ptr-maker.type-fn]
    // [spec:et:sem:tensor-ptr-maker.executorch.extension.tensor-ptr-maker.type-fn]
    pub fn type_(mut self, type_: ScalarType) -> Self {
        self.type_ = type_;
        self
    }

    /// Sets the order of dimensions in memory.
    // [spec:et:def:tensor-ptr-maker.executorch.extension.tensor-ptr-maker.dim-order-fn]
    // [spec:et:sem:tensor-ptr-maker.executorch.extension.tensor-ptr-maker.dim-order-fn]
    pub fn dim_order(mut self, dim_order: Vec<DimOrderType>) -> Self {
        self.dim_order_ = dim_order;
        self
    }

    /// Sets the strides for each dimension of the tensor.
    // [spec:et:def:tensor-ptr-maker.executorch.extension.tensor-ptr-maker.strides-fn]
    // [spec:et:sem:tensor-ptr-maker.executorch.extension.tensor-ptr-maker.strides-fn]
    pub fn strides(mut self, strides: Vec<StridesType>) -> Self {
        self.strides_ = strides;
        self
    }

    /// Sets the shape dynamism of the tensor.
    // [spec:et:def:tensor-ptr-maker.executorch.extension.tensor-ptr-maker.dynamism-fn]
    // [spec:et:sem:tensor-ptr-maker.executorch.extension.tensor-ptr-maker.dynamism-fn]
    pub fn dynamism(mut self, dynamism: TensorShapeDynamism) -> Self {
        self.dynamism_ = dynamism;
        self
    }

    /// Sets a custom deleter function to manage the lifetime of the data buffer.
    // [spec:et:def:tensor-ptr-maker.executorch.extension.tensor-ptr-maker.deleter-fn]
    // [spec:et:sem:tensor-ptr-maker.executorch.extension.tensor-ptr-maker.deleter-fn]
    pub fn deleter(mut self, deleter: crate::extension::tensor::tensor_ptr::Deleter) -> Self {
        self.deleter_ = Some(deleter);
        self
    }

    /// Creates and returns a TensorPtr instance using the properties set in this
    /// TensorPtrMaker.
    // [spec:et:def:tensor-ptr-maker.executorch.extension.tensor-ptr-maker.make-tensor-ptr-fn]
    // [spec:et:sem:tensor-ptr-maker.executorch.extension.tensor-ptr-maker.make-tensor-ptr-fn]
    pub fn make_tensor_ptr(self) -> TensorPtr {
        make_tensor_ptr(
            self.sizes_,
            self.data_,
            self.dim_order_,
            self.strides_,
            self.type_,
            self.dynamism_,
            self.deleter_,
            Device::from_type(DeviceType::CPU),
        )
    }

    // Private constructor, callable only by for_blob.
    // [spec:et:def:tensor-ptr-maker.executorch.extension.tensor-ptr-maker.tensor-ptr-maker-fn]
    // [spec:et:sem:tensor-ptr-maker.executorch.extension.tensor-ptr-maker.tensor-ptr-maker-fn]
    fn new(data: *mut core::ffi::c_void, sizes: Vec<SizesType>, type_: ScalarType) -> Self {
        TensorPtrMaker {
            sizes_: sizes,
            strides_: Vec::new(),
            dim_order_: Vec::new(),
            deleter_: None,
            data_: data,
            type_,
            dynamism_: TensorShapeDynamism::DYNAMIC_BOUND,
        }
    }
}

/// PORT-NOTE: C++ `operator TensorPtr() &&` maps to `From<TensorPtrMaker>`.
// [spec:et:sem:tensor-ptr-maker.executorch.extension.tensor-ptr-maker.make-tensor-ptr-fn]
impl From<TensorPtrMaker> for TensorPtr {
    fn from(maker: TensorPtrMaker) -> TensorPtr {
        maker.make_tensor_ptr()
    }
}

/// Creates a TensorPtrMaker instance for building a TensorPtr from a raw data
/// pointer and tensor sizes.
// [spec:et:def:tensor-ptr-maker.executorch.extension.for-blob-fn]
// [spec:et:sem:tensor-ptr-maker.executorch.extension.for-blob-fn]
// [spec:et:def:tensor-ptr-maker.executorch.extension.tensor-ptr-maker.for-blob-fn]
// [spec:et:sem:tensor-ptr-maker.executorch.extension.tensor-ptr-maker.for-blob-fn]
pub fn for_blob(
    data: *mut core::ffi::c_void,
    sizes: Vec<SizesType>,
    type_: ScalarType,
) -> TensorPtrMaker {
    TensorPtrMaker::new(data, sizes, type_)
}

/// Creates a TensorPtr from a raw data pointer and tensor sizes.
pub fn from_blob(
    data: *mut core::ffi::c_void,
    sizes: Vec<SizesType>,
    type_: ScalarType,
    dynamism: TensorShapeDynamism,
) -> TensorPtr {
    for_blob(data, sizes, type_)
        .dynamism(dynamism)
        .make_tensor_ptr()
}

/// Creates a TensorPtr from a raw data pointer, tensor sizes, and strides.
pub fn from_blob_strided(
    data: *mut core::ffi::c_void,
    sizes: Vec<SizesType>,
    strides: Vec<StridesType>,
    type_: ScalarType,
    dynamism: TensorShapeDynamism,
) -> TensorPtr {
    for_blob(data, sizes, type_)
        .strides(strides)
        .dynamism(dynamism)
        .make_tensor_ptr()
}

/// Creates a TensorPtr from a raw data pointer, tensor sizes, and a deleter.
pub fn from_blob_deleter(
    data: *mut core::ffi::c_void,
    sizes: Vec<SizesType>,
    type_: ScalarType,
    deleter: crate::extension::tensor::tensor_ptr::Deleter,
    dynamism: TensorShapeDynamism,
) -> TensorPtr {
    for_blob(data, sizes, type_)
        .deleter(deleter)
        .dynamism(dynamism)
        .make_tensor_ptr()
}

/// Creates a TensorPtr from a raw data pointer, tensor sizes, strides, and a
/// deleter.
// [spec:et:def:tensor-ptr-maker.executorch.extension.from-blob-fn]
// [spec:et:sem:tensor-ptr-maker.executorch.extension.from-blob-fn]
pub fn from_blob_strided_deleter(
    data: *mut core::ffi::c_void,
    sizes: Vec<SizesType>,
    strides: Vec<StridesType>,
    type_: ScalarType,
    deleter: crate::extension::tensor::tensor_ptr::Deleter,
    dynamism: TensorShapeDynamism,
) -> TensorPtr {
    for_blob(data, sizes, type_)
        .strides(strides)
        .deleter(deleter)
        .dynamism(dynamism)
        .make_tensor_ptr()
}

/// Creates an empty TensorPtr with the same size and properties as the given
/// tensor.
// [spec:et:def:tensor-ptr-maker.executorch.extension.empty-like-fn]
// [spec:et:sem:tensor-ptr-maker.executorch.extension.empty-like-fn]
pub fn empty_like(
    other: &TensorPtr,
    type_: ScalarType,
    dynamism: TensorShapeDynamism,
) -> TensorPtr {
    let mut type_ = type_;
    if type_ == ScalarType::Undefined {
        type_ = other.tensor().scalar_type();
    }
    let t = other.tensor();
    empty_strided(
        arrayref_to_vec_sizes(&t),
        arrayref_to_vec_strides(&t),
        type_,
        dynamism,
    )
}

/// Creates an empty TensorPtr with the specified sizes and properties.
// [spec:et:def:tensor-ptr-maker.executorch.extension.empty-fn]
// [spec:et:sem:tensor-ptr-maker.executorch.extension.empty-fn]
pub fn empty(sizes: Vec<SizesType>, type_: ScalarType, dynamism: TensorShapeDynamism) -> TensorPtr {
    empty_strided(sizes, Vec::new(), type_, dynamism)
}

/// Creates a TensorPtr filled with the specified value, same size/props as
/// another tensor.
// [spec:et:def:tensor-ptr-maker.executorch.extension.full-like-fn]
// [spec:et:sem:tensor-ptr-maker.executorch.extension.full-like-fn]
pub fn full_like(
    other: &TensorPtr,
    fill_value: Scalar,
    type_: ScalarType,
    dynamism: TensorShapeDynamism,
) -> TensorPtr {
    let mut type_ = type_;
    if type_ == ScalarType::Undefined {
        type_ = other.tensor().scalar_type();
    }
    let t = other.tensor();
    full_strided(
        arrayref_to_vec_sizes(&t),
        arrayref_to_vec_strides(&t),
        fill_value,
        type_,
        dynamism,
    )
}

/// Creates a TensorPtr filled with the specified value.
// [spec:et:def:tensor-ptr-maker.executorch.extension.full-fn]
// [spec:et:sem:tensor-ptr-maker.executorch.extension.full-fn]
pub fn full(
    sizes: Vec<SizesType>,
    fill_value: Scalar,
    type_: ScalarType,
    dynamism: TensorShapeDynamism,
) -> TensorPtr {
    full_strided(sizes, Vec::new(), fill_value, type_, dynamism)
}

/// Creates a TensorPtr holding a scalar value.
// [spec:et:def:tensor-ptr-maker.executorch.extension.scalar-tensor-fn]
// [spec:et:sem:tensor-ptr-maker.executorch.extension.scalar-tensor-fn]
pub fn scalar_tensor(value: Scalar, type_: ScalarType) -> TensorPtr {
    full(Vec::new(), value, type_, TensorShapeDynamism::DYNAMIC_BOUND)
}

/// Creates a TensorPtr filled with ones, same size/props as another tensor.
// [spec:et:def:tensor-ptr-maker.executorch.extension.ones-like-fn]
// [spec:et:sem:tensor-ptr-maker.executorch.extension.ones-like-fn]
pub fn ones_like(other: &TensorPtr, type_: ScalarType, dynamism: TensorShapeDynamism) -> TensorPtr {
    full_like(other, Scalar::from_i64(1), type_, dynamism)
}

/// Creates a TensorPtr filled with ones.
// [spec:et:def:tensor-ptr-maker.executorch.extension.ones-fn]
// [spec:et:sem:tensor-ptr-maker.executorch.extension.ones-fn]
pub fn ones(sizes: Vec<SizesType>, type_: ScalarType, dynamism: TensorShapeDynamism) -> TensorPtr {
    full(sizes, Scalar::from_i64(1), type_, dynamism)
}

/// Creates a TensorPtr filled with zeros, same size/props as another tensor.
// [spec:et:def:tensor-ptr-maker.executorch.extension.zeros-like-fn]
// [spec:et:sem:tensor-ptr-maker.executorch.extension.zeros-like-fn]
pub fn zeros_like(
    other: &TensorPtr,
    type_: ScalarType,
    dynamism: TensorShapeDynamism,
) -> TensorPtr {
    full_like(other, Scalar::from_i64(0), type_, dynamism)
}

/// Creates a TensorPtr filled with zeros.
// [spec:et:def:tensor-ptr-maker.executorch.extension.zeros-fn]
// [spec:et:sem:tensor-ptr-maker.executorch.extension.zeros-fn]
pub fn zeros(sizes: Vec<SizesType>, type_: ScalarType, dynamism: TensorShapeDynamism) -> TensorPtr {
    full(sizes, Scalar::from_i64(0), type_, dynamism)
}

/// Creates a TensorPtr filled with random values between 0 and 1, same size as
/// another tensor.
// [spec:et:def:tensor-ptr-maker.executorch.extension.rand-like-fn]
// [spec:et:sem:tensor-ptr-maker.executorch.extension.rand-like-fn]
pub fn rand_like(other: &TensorPtr, type_: ScalarType, dynamism: TensorShapeDynamism) -> TensorPtr {
    let mut type_ = type_;
    if type_ == ScalarType::Undefined {
        type_ = other.tensor().scalar_type();
    }
    let t = other.tensor();
    rand_strided(
        arrayref_to_vec_sizes(&t),
        arrayref_to_vec_strides(&t),
        type_,
        dynamism,
    )
}

/// Creates a TensorPtr filled with random values between 0 and 1.
// [spec:et:def:tensor-ptr-maker.executorch.extension.rand-fn]
// [spec:et:sem:tensor-ptr-maker.executorch.extension.rand-fn]
pub fn rand(sizes: Vec<SizesType>, type_: ScalarType, dynamism: TensorShapeDynamism) -> TensorPtr {
    rand_strided(sizes, Vec::new(), type_, dynamism)
}

/// Creates a TensorPtr filled with normal random values, same size as another
/// tensor.
// [spec:et:def:tensor-ptr-maker.executorch.extension.randn-like-fn]
// [spec:et:sem:tensor-ptr-maker.executorch.extension.randn-like-fn]
pub fn randn_like(
    other: &TensorPtr,
    type_: ScalarType,
    dynamism: TensorShapeDynamism,
) -> TensorPtr {
    let mut type_ = type_;
    if type_ == ScalarType::Undefined {
        type_ = other.tensor().scalar_type();
    }
    let t = other.tensor();
    randn_strided(
        arrayref_to_vec_sizes(&t),
        arrayref_to_vec_strides(&t),
        type_,
        dynamism,
    )
}

/// Creates a TensorPtr filled with normal random values.
// [spec:et:def:tensor-ptr-maker.executorch.extension.randn-fn]
// [spec:et:sem:tensor-ptr-maker.executorch.extension.randn-fn]
pub fn randn(sizes: Vec<SizesType>, type_: ScalarType, dynamism: TensorShapeDynamism) -> TensorPtr {
    randn_strided(sizes, Vec::new(), type_, dynamism)
}

/// Creates a TensorPtr filled with random integers, same size as another
/// tensor.
// [spec:et:def:tensor-ptr-maker.executorch.extension.randint-like-fn]
// [spec:et:sem:tensor-ptr-maker.executorch.extension.randint-like-fn]
pub fn randint_like(
    other: &TensorPtr,
    low: i64,
    high: i64,
    type_: ScalarType,
    dynamism: TensorShapeDynamism,
) -> TensorPtr {
    let mut type_ = type_;
    if type_ == ScalarType::Undefined {
        type_ = other.tensor().scalar_type();
    }
    let t = other.tensor();
    randint_strided(
        low,
        high,
        arrayref_to_vec_sizes(&t),
        arrayref_to_vec_strides(&t),
        type_,
        dynamism,
    )
}

/// Creates a TensorPtr filled with random integers in the given range.
// [spec:et:def:tensor-ptr-maker.executorch.extension.randint-fn]
// [spec:et:sem:tensor-ptr-maker.executorch.extension.randint-fn]
pub fn randint(
    low: i64,
    high: i64,
    sizes: Vec<SizesType>,
    type_: ScalarType,
    dynamism: TensorShapeDynamism,
) -> TensorPtr {
    randint_strided(low, high, sizes, Vec::new(), type_, dynamism)
}

// PORT-NOTE: helpers for the `{other->sizes().begin(), other->sizes().end()}`
// metadata copies.
fn arrayref_to_vec_sizes(
    t: &crate::runtime::core::portable_type::tensor::Tensor,
) -> Vec<SizesType> {
    let arr = t.sizes();
    let mut v = Vec::with_capacity(arr.size());
    for i in 0..arr.size() {
        v.push(*arr.at(i));
    }
    v
}
fn arrayref_to_vec_strides(
    t: &crate::runtime::core::portable_type::tensor::Tensor,
) -> Vec<StridesType> {
    let arr = t.strides();
    let mut v = Vec::with_capacity(arr.size());
    for i in 0..arr.size() {
        v.push(*arr.at(i));
    }
    v
}

#[cfg(test)]
mod tests {
    // Literal port of extension/tensor/test/tensor_ptr_maker_test.cpp
    // (non-ATen `TensorPtrMakerTest` fixture).
    use super::*;
    use core::sync::atomic::{AtomicBool, Ordering};

    // Mirrors `TensorPtrMakerTest::SetUpTestSuite()`'s `runtime_init()`.
    fn setup() {
        crate::runtime::platform::runtime::runtime_init();
    }

    // [spec:et:sem:tensor-ptr-maker.executorch.extension.for-blob-fn/test]
    // [spec:et:sem:tensor-ptr-maker.executorch.extension.tensor-ptr-maker.for-blob-fn/test]
    // [spec:et:sem:tensor-ptr-maker.executorch.extension.tensor-ptr-maker.tensor-ptr-maker-fn/test]
    // [spec:et:sem:tensor-ptr-maker.executorch.extension.tensor-ptr-maker.dim-order-fn/test]
    // [spec:et:sem:tensor-ptr-maker.executorch.extension.tensor-ptr-maker.strides-fn/test]
    // [spec:et:sem:tensor-ptr-maker.executorch.extension.tensor-ptr-maker.dynamism-fn/test]
    // [spec:et:sem:tensor-ptr-maker.executorch.extension.tensor-ptr-maker.make-tensor-ptr-fn/test]
    //
    // The deleted copy-assign (`operator=`) collapses onto the move-only builder
    // in Rust (no `Copy`/`Clone`): every setter in this chain consumes the maker
    // by value (a move, C++'s defaulted move-assign path) and the final moved-in
    // state is what make_tensor_ptr publishes.
    // [spec:et:sem:tensor-ptr-maker.executorch.extension.tensor-ptr-maker.operator-fn/test]
    #[test]
    fn tensor_ptr_maker_test_create_tensor_using_tensor_maker() {
        setup();
        let mut data: [f32; 20] = [0.0; 20];
        data[0] = 2.0;
        let tensor = for_blob(
            data.as_mut_ptr() as *mut core::ffi::c_void,
            alloc::vec![4, 5],
            ScalarType::Float,
        )
        .dim_order(alloc::vec![0, 1])
        .strides(alloc::vec![5, 1])
        .dynamism(TensorShapeDynamism::DYNAMIC_BOUND)
        .make_tensor_ptr();

        let t = tensor.tensor();
        assert_eq!(t.dim(), 2);
        assert_eq!(t.size(0), 4);
        assert_eq!(t.size(1), 5);
        assert_eq!(*t.strides().at(0), 5);
        assert_eq!(*t.strides().at(1), 1);
        assert_eq!(t.const_data_ptr::<f32>(), data.as_ptr());
        assert_eq!(unsafe { *t.const_data_ptr::<f32>().add(0) }, 2.0);
    }

    // PORT-NOTE: the C++ `PerfectForwardingLValue`/`PerfectForwardingRValue`
    // tests pin `std::vector`'s copy-vs-move forwarding behavior of `for_blob`'s
    // `sizes` and the builder's `dim_order`/`strides` setters (checking the
    // caller's vector is left intact on lvalue-forward, emptied on
    // move-forward). Rust `Vec` arguments are always moved by value, so the
    // caller no longer owns them either way and the post-call `.size()` checks
    // have no Rust surface to bind to. The tensor-shape assertions common to both
    // are exercised by `create_tensor_using_tensor_maker`; the forwarding checks
    // themselves are not portable and are recorded here.

    // [spec:et:sem:tensor-ptr-maker.executorch.extension.from-blob-fn/test]
    #[test]
    fn tensor_ptr_maker_test_create_tensor_from_blob() {
        setup();
        let mut data: [f32; 20] = [0.0; 20];
        data[0] = 2.0;
        let tensor = from_blob(
            data.as_mut_ptr() as *mut core::ffi::c_void,
            alloc::vec![4, 5],
            ScalarType::Float,
            TensorShapeDynamism::DYNAMIC_BOUND,
        );

        let t = tensor.tensor();
        assert_eq!(t.dim(), 2);
        assert_eq!(t.size(0), 4);
        assert_eq!(t.size(1), 5);
        assert_eq!(*t.strides().at(0), 5);
        assert_eq!(*t.strides().at(1), 1);
        assert_eq!(t.const_data_ptr::<f32>(), data.as_ptr());
        assert_eq!(unsafe { *t.const_data_ptr::<f32>().add(0) }, 2.0);
        assert_eq!(unsafe { *t.const_data_ptr::<f32>().add(19) }, 0.0);
    }

    // [spec:et:sem:tensor-ptr-maker.executorch.extension.from-blob-fn/test]
    #[test]
    fn tensor_ptr_maker_test_create_tensor_using_from_blob_with_strides() {
        setup();
        let mut data: [f32; 20] = [0.0; 20];
        data[0] = 3.0;
        let tensor = from_blob_strided(
            data.as_mut_ptr() as *mut core::ffi::c_void,
            alloc::vec![2, 2, 2],
            alloc::vec![4, 2, 1],
            ScalarType::Float,
            TensorShapeDynamism::DYNAMIC_BOUND,
        );

        let t = tensor.tensor();
        assert_eq!(t.dim(), 3);
        assert_eq!(t.size(0), 2);
        assert_eq!(t.size(1), 2);
        assert_eq!(t.size(2), 2);
        assert_eq!(*t.strides().at(0), 4);
        assert_eq!(*t.strides().at(1), 2);
        assert_eq!(*t.strides().at(2), 1);
        assert_eq!(t.const_data_ptr::<f32>(), data.as_ptr());
        assert_eq!(unsafe { *t.const_data_ptr::<f32>().add(0) }, 3.0);
    }

    // [spec:et:sem:tensor-ptr-maker.executorch.extension.from-blob-fn/test]
    #[test]
    fn tensor_ptr_maker_test_create_tensor_using_from_blob_with_legal_strides() {
        setup();
        let mut data: [f32; 20] = [0.0; 20];
        data[0] = 3.0;
        let tensor = from_blob_strided(
            data.as_mut_ptr() as *mut core::ffi::c_void,
            alloc::vec![1, 2, 2],
            alloc::vec![10, 2, 1],
            ScalarType::Float,
            TensorShapeDynamism::DYNAMIC_BOUND,
        );

        let t = tensor.tensor();
        assert_eq!(t.dim(), 3);
        assert_eq!(t.size(0), 1);
        assert_eq!(t.size(1), 2);
        assert_eq!(t.size(2), 2);

        // recalculated stride[0] to 4 to meet ET's requirement while maintaining
        // the same behavior as the original tensor since size[0] == 1
        assert_eq!(*t.strides().at(0), 4);
        assert_eq!(*t.strides().at(1), 2);
        assert_eq!(*t.strides().at(2), 1);
        assert_eq!(t.const_data_ptr::<f32>(), data.as_ptr());
        assert_eq!(unsafe { *t.const_data_ptr::<f32>().add(0) }, 3.0);
    }

    // PORT-NOTE: `ET_EXPECT_DEATH` death test. `runtime_abort` -> `libc::abort()`
    // terminates the process, so `#[should_panic]` cannot catch it; ported and
    // `#[ignore]`d per the death-test convention.
    // [spec:et:sem:tensor-ptr-maker.executorch.extension.from-blob-fn/test]
    #[test]
    #[should_panic]
    #[ignore]
    fn tensor_ptr_maker_test_failed_create_tensor_using_from_blob_with_illegal_strides() {
        setup();
        let mut data: [f32; 20] = [0.0; 20];
        data[0] = 3.0;
        let _ = from_blob_strided(
            data.as_mut_ptr() as *mut core::ffi::c_void,
            alloc::vec![2, 2, 2],
            alloc::vec![10, 2, 1],
            ScalarType::Float,
            TensorShapeDynamism::DYNAMIC_BOUND,
        );
    }

    // [spec:et:sem:tensor-ptr-maker.executorch.extension.tensor-ptr-maker.make-tensor-ptr-fn/test]
    #[test]
    fn tensor_ptr_maker_test_tensor_maker_conversion_operator() {
        setup();
        let mut data: [f32; 20] = [0.0; 20];
        data[0] = 2.0;
        let tensor: TensorPtr = TensorPtr::from(
            for_blob(
                data.as_mut_ptr() as *mut core::ffi::c_void,
                alloc::vec![4, 5],
                ScalarType::Float,
            )
            .dynamism(TensorShapeDynamism::DYNAMIC_BOUND),
        );

        let t = tensor.tensor();
        assert_eq!(t.dim(), 2);
        assert_eq!(t.size(0), 4);
        assert_eq!(t.size(1), 5);
    }

    // [spec:et:sem:tensor-ptr-maker.executorch.extension.from-blob-fn/test]
    #[test]
    fn tensor_ptr_maker_test_create_tensor_with_zero_dimensions() {
        setup();
        let mut data: [f32; 1] = [2.0];
        let tensor = from_blob(
            data.as_mut_ptr() as *mut core::ffi::c_void,
            alloc::vec![],
            ScalarType::Float,
            TensorShapeDynamism::DYNAMIC_BOUND,
        );

        let t = tensor.tensor();
        assert_eq!(t.dim(), 0);
        assert_eq!(t.numel(), 1);
        assert_eq!(unsafe { *t.const_data_ptr::<f32>().add(0) }, 2.0);
    }

    // [spec:et:sem:tensor-ptr-maker.executorch.extension.tensor-ptr-maker.deleter-fn/test]
    // [spec:et:sem:tensor-ptr-maker.executorch.extension.tensor-ptr-maker.make-tensor-ptr-fn/test]
    #[test]
    fn tensor_ptr_maker_test_tensor_with_custom_data_deleter() {
        setup();
        // PORT-NOTE: C++ `new float[20]()` + `delete[]` in the deleter maps to a
        // leaked boxed slice reclaimed by the deleter via `Box::from_raw`. The
        // captured `AtomicBool` flag mirrors the `&deleter_called` capture.
        static DELETER_CALLED: AtomicBool = AtomicBool::new(false);
        DELETER_CALLED.store(false, Ordering::SeqCst);
        let data: *mut [f32] = Box::into_raw(alloc::vec![0.0f32; 20].into_boxed_slice());
        let data_ptr = data as *mut f32;
        let tensor = for_blob(
            data_ptr as *mut core::ffi::c_void,
            alloc::vec![4, 5],
            ScalarType::Float,
        )
        .deleter(alloc::boxed::Box::new(
            move |ptr: *mut core::ffi::c_void| {
                DELETER_CALLED.store(true, Ordering::SeqCst);
                unsafe {
                    drop(Box::from_raw(core::slice::from_raw_parts_mut(
                        ptr as *mut f32,
                        20,
                    )));
                }
            },
        ))
        .make_tensor_ptr();

        drop(tensor);
        assert!(DELETER_CALLED.load(Ordering::SeqCst));
    }

    // PORT-NOTE: `TensorManagesMovedVector` pins that `for_blob(...).deleter([
    // moved_data = std::move(data), ...])` captures the moved-from
    // `std::vector`, leaving the caller's `data` empty while the tensor keeps
    // pointing at `data_ptr`. Rust's closure move-capture reproduces the capture
    // and the empty-caller semantics: `data` is moved into the closure, so it can
    // no longer be observed by the caller. We verify the tensor still aliases the
    // captured buffer and that the deleter fires on reset.
    // [spec:et:sem:tensor-ptr-maker.executorch.extension.tensor-ptr-maker.deleter-fn/test]
    #[test]
    fn tensor_ptr_maker_test_tensor_manages_moved_vector() {
        setup();
        static DELETER_CALLED: AtomicBool = AtomicBool::new(false);
        DELETER_CALLED.store(false, Ordering::SeqCst);
        let data: Vec<f32> = alloc::vec![3.0f32; 20];
        let data_ptr = data.as_ptr() as *mut f32;
        let tensor = for_blob(
            data_ptr as *mut core::ffi::c_void,
            alloc::vec![4, 5],
            ScalarType::Float,
        )
        .deleter(alloc::boxed::Box::new(move |_| {
            let _moved_data = &data; // keeps the moved vector alive in the closure
            DELETER_CALLED.store(true, Ordering::SeqCst);
        }))
        .make_tensor_ptr();

        assert_eq!(tensor.tensor().data_ptr::<f32>(), data_ptr);

        drop(tensor);
        assert!(DELETER_CALLED.load(Ordering::SeqCst));
    }

    // PORT-NOTE: `TensorDeleterReleasesCapturedSharedPtr` uses a
    // `std::shared_ptr<float[]>` captured by the deleter and checks `use_count()`
    // transitions 2 -> 1 on reset. Rust's analog is an `alloc::sync::Arc<[f32]>`;
    // `Arc::strong_count` mirrors `use_count`.
    // [spec:et:sem:tensor-ptr-maker.executorch.extension.tensor-ptr-maker.deleter-fn/test]
    #[test]
    fn tensor_ptr_maker_test_tensor_deleter_releases_captured_shared_ptr() {
        setup();
        static DELETER_CALLED: AtomicBool = AtomicBool::new(false);
        DELETER_CALLED.store(false, Ordering::SeqCst);
        let data_ptr: alloc::sync::Arc<[f32]> = alloc::sync::Arc::from(alloc::vec![0.0f32; 10]);
        let raw = data_ptr.as_ptr() as *mut f32;
        let captured = data_ptr.clone();
        let tensor = from_blob_deleter(
            raw as *mut core::ffi::c_void,
            alloc::vec![4, 5],
            ScalarType::Float,
            alloc::boxed::Box::new(move |_| {
                let _keep = &captured;
                DELETER_CALLED.store(true, Ordering::SeqCst);
            }),
            TensorShapeDynamism::DYNAMIC_BOUND,
        );

        assert_eq!(alloc::sync::Arc::strong_count(&data_ptr), 2);

        drop(tensor);
        assert!(DELETER_CALLED.load(Ordering::SeqCst));
        assert_eq!(alloc::sync::Arc::strong_count(&data_ptr), 1);
    }

    // [spec:et:sem:tensor-ptr-maker.executorch.extension.empty-fn/test]
    // empty -> empty_strided:
    // [spec:et:sem:tensor-ptr-maker.executorch.extension.empty-strided-fn/test]
    #[test]
    fn tensor_ptr_maker_test_create_empty() {
        setup();
        let tensor = empty(
            alloc::vec![4, 5],
            ScalarType::Float,
            TensorShapeDynamism::DYNAMIC_BOUND,
        );
        assert_eq!(tensor.tensor().dim(), 2);
        assert_eq!(tensor.tensor().size(0), 4);
        assert_eq!(tensor.tensor().size(1), 5);
        assert_eq!(tensor.tensor().scalar_type(), ScalarType::Float);

        let tensor2 = empty(
            alloc::vec![4, 5],
            ScalarType::Int,
            TensorShapeDynamism::DYNAMIC_BOUND,
        );
        assert_eq!(tensor2.tensor().dim(), 2);
        assert_eq!(tensor2.tensor().size(0), 4);
        assert_eq!(tensor2.tensor().size(1), 5);
        assert_eq!(tensor2.tensor().scalar_type(), ScalarType::Int);

        let tensor3 = empty(
            alloc::vec![4, 5],
            ScalarType::Long,
            TensorShapeDynamism::DYNAMIC_BOUND,
        );
        assert_eq!(tensor3.tensor().dim(), 2);
        assert_eq!(tensor3.tensor().size(0), 4);
        assert_eq!(tensor3.tensor().size(1), 5);
        assert_eq!(tensor3.tensor().scalar_type(), ScalarType::Long);

        let tensor4 = empty(
            alloc::vec![4, 5],
            ScalarType::Double,
            TensorShapeDynamism::DYNAMIC_BOUND,
        );
        assert_eq!(tensor4.tensor().dim(), 2);
        assert_eq!(tensor4.tensor().size(0), 4);
        assert_eq!(tensor4.tensor().size(1), 5);
        assert_eq!(tensor4.tensor().scalar_type(), ScalarType::Double);
    }

    // [spec:et:sem:tensor-ptr-maker.executorch.extension.full-fn/test]
    // full -> full_strided; the Float/Half/BFloat16 fills drive the
    // floating-point ExtractScalar overload (integral-scalar-into-float path):
    // [spec:et:sem:tensor-ptr-maker.executorch.extension.full-strided-fn/test]
    // [spec:et:sem:tensor-ptr-maker.executorch.extension.extract-scalar-fn/test]
    #[test]
    fn tensor_ptr_maker_test_create_full() {
        setup();
        let tensor = full(
            alloc::vec![4, 5],
            Scalar::from_i64(7),
            ScalarType::Float,
            TensorShapeDynamism::DYNAMIC_BOUND,
        );
        assert_eq!(tensor.tensor().dim(), 2);
        assert_eq!(tensor.tensor().size(0), 4);
        assert_eq!(tensor.tensor().size(1), 5);
        assert_eq!(tensor.tensor().scalar_type(), ScalarType::Float);
        assert_eq!(
            unsafe { *tensor.tensor().const_data_ptr::<f32>().add(0) },
            7.0
        );

        let tensor2 = full(
            alloc::vec![4, 5],
            Scalar::from_i64(3),
            ScalarType::Int,
            TensorShapeDynamism::DYNAMIC_BOUND,
        );
        assert_eq!(tensor2.tensor().dim(), 2);
        assert_eq!(tensor2.tensor().size(0), 4);
        assert_eq!(tensor2.tensor().size(1), 5);
        assert_eq!(tensor2.tensor().scalar_type(), ScalarType::Int);
        assert_eq!(
            unsafe { *tensor2.tensor().const_data_ptr::<i32>().add(0) },
            3
        );

        let tensor3 = full(
            alloc::vec![4, 5],
            Scalar::from_i64(9),
            ScalarType::Long,
            TensorShapeDynamism::DYNAMIC_BOUND,
        );
        assert_eq!(tensor3.tensor().dim(), 2);
        assert_eq!(tensor3.tensor().size(0), 4);
        assert_eq!(tensor3.tensor().size(1), 5);
        assert_eq!(tensor3.tensor().scalar_type(), ScalarType::Long);
        assert_eq!(
            unsafe { *tensor3.tensor().const_data_ptr::<i64>().add(0) },
            9
        );

        let tensor4 = full(
            alloc::vec![4, 5],
            Scalar::from_i64(11),
            ScalarType::Double,
            TensorShapeDynamism::DYNAMIC_BOUND,
        );
        assert_eq!(tensor4.tensor().dim(), 2);
        assert_eq!(tensor4.tensor().size(0), 4);
        assert_eq!(tensor4.tensor().size(1), 5);
        assert_eq!(tensor4.tensor().scalar_type(), ScalarType::Double);
        assert_eq!(
            unsafe { *tensor4.tensor().const_data_ptr::<f64>().add(0) },
            11.0
        );

        let tensor5 = full(
            alloc::vec![4, 5],
            Scalar::from_i64(13),
            ScalarType::Half,
            TensorShapeDynamism::DYNAMIC_BOUND,
        );
        assert_eq!(tensor5.tensor().dim(), 2);
        assert_eq!(tensor5.tensor().size(0), 4);
        assert_eq!(tensor5.tensor().size(1), 5);
        assert_eq!(tensor5.tensor().scalar_type(), ScalarType::Half);
        assert_eq!(
            unsafe { *tensor5.tensor().const_data_ptr::<Half>().add(0) }.to_f32(),
            13.0
        );

        let tensor6 = full(
            alloc::vec![4, 5],
            Scalar::from_i64(15),
            ScalarType::BFloat16,
            TensorShapeDynamism::DYNAMIC_BOUND,
        );
        assert_eq!(tensor6.tensor().dim(), 2);
        assert_eq!(tensor6.tensor().size(0), 4);
        assert_eq!(tensor6.tensor().size(1), 5);
        assert_eq!(tensor6.tensor().scalar_type(), ScalarType::BFloat16);
        assert_eq!(
            unsafe { *tensor6.tensor().const_data_ptr::<BFloat16>().add(0) }.to_f32(),
            15.0
        );
    }

    // [spec:et:sem:tensor-ptr-maker.executorch.extension.scalar-tensor-fn/test]
    #[test]
    fn tensor_ptr_maker_test_create_scalar() {
        setup();
        let tensor = scalar_tensor(Scalar::from_double(3.14f32 as f64), ScalarType::Float);

        assert_eq!(tensor.tensor().dim(), 0);
        assert_eq!(tensor.tensor().numel(), 1);
        assert_eq!(tensor.tensor().scalar_type(), ScalarType::Float);
        assert_eq!(
            unsafe { *tensor.tensor().const_data_ptr::<f32>().add(0) },
            3.14f32
        );

        let tensor2 = scalar_tensor(Scalar::from_i64(5), ScalarType::Int);
        assert_eq!(tensor2.tensor().dim(), 0);
        assert_eq!(tensor2.tensor().numel(), 1);
        assert_eq!(tensor2.tensor().scalar_type(), ScalarType::Int);
        assert_eq!(
            unsafe { *tensor2.tensor().const_data_ptr::<i32>().add(0) },
            5
        );

        let tensor3 = scalar_tensor(Scalar::from_double(7.0), ScalarType::Double);
        assert_eq!(tensor3.tensor().dim(), 0);
        assert_eq!(tensor3.tensor().numel(), 1);
        assert_eq!(tensor3.tensor().scalar_type(), ScalarType::Double);
        assert_eq!(
            unsafe { *tensor3.tensor().const_data_ptr::<f64>().add(0) },
            7.0
        );
    }

    // [spec:et:sem:tensor-ptr-maker.executorch.extension.ones-fn/test]
    #[test]
    fn tensor_ptr_maker_test_create_ones() {
        setup();
        let tensor = ones(
            alloc::vec![4, 5],
            ScalarType::Float,
            TensorShapeDynamism::DYNAMIC_BOUND,
        );
        assert_eq!(tensor.tensor().dim(), 2);
        assert_eq!(tensor.tensor().size(0), 4);
        assert_eq!(tensor.tensor().size(1), 5);
        assert_eq!(tensor.tensor().scalar_type(), ScalarType::Float);
        assert_eq!(
            unsafe { *tensor.tensor().const_data_ptr::<f32>().add(0) },
            1.0
        );

        let tensor2 = ones(
            alloc::vec![4, 5],
            ScalarType::Int,
            TensorShapeDynamism::DYNAMIC_BOUND,
        );
        assert_eq!(tensor2.tensor().dim(), 2);
        assert_eq!(tensor2.tensor().size(0), 4);
        assert_eq!(tensor2.tensor().size(1), 5);
        assert_eq!(tensor2.tensor().scalar_type(), ScalarType::Int);
        assert_eq!(
            unsafe { *tensor2.tensor().const_data_ptr::<i32>().add(0) },
            1
        );

        let tensor3 = ones(
            alloc::vec![4, 5],
            ScalarType::Long,
            TensorShapeDynamism::DYNAMIC_BOUND,
        );
        assert_eq!(tensor3.tensor().dim(), 2);
        assert_eq!(tensor3.tensor().size(0), 4);
        assert_eq!(tensor3.tensor().size(1), 5);
        assert_eq!(tensor3.tensor().scalar_type(), ScalarType::Long);
        assert_eq!(
            unsafe { *tensor3.tensor().const_data_ptr::<i64>().add(0) },
            1
        );

        let tensor4 = ones(
            alloc::vec![4, 5],
            ScalarType::Double,
            TensorShapeDynamism::DYNAMIC_BOUND,
        );
        assert_eq!(tensor4.tensor().dim(), 2);
        assert_eq!(tensor4.tensor().size(0), 4);
        assert_eq!(tensor4.tensor().size(1), 5);
        assert_eq!(tensor4.tensor().scalar_type(), ScalarType::Double);
        assert_eq!(
            unsafe { *tensor4.tensor().const_data_ptr::<f64>().add(0) },
            1.0
        );
    }

    // [spec:et:sem:tensor-ptr-maker.executorch.extension.zeros-fn/test]
    #[test]
    fn tensor_ptr_maker_test_create_zeros() {
        setup();
        let tensor = zeros(
            alloc::vec![4, 5],
            ScalarType::Float,
            TensorShapeDynamism::DYNAMIC_BOUND,
        );
        assert_eq!(tensor.tensor().dim(), 2);
        assert_eq!(tensor.tensor().size(0), 4);
        assert_eq!(tensor.tensor().size(1), 5);
        assert_eq!(tensor.tensor().scalar_type(), ScalarType::Float);
        assert_eq!(
            unsafe { *tensor.tensor().const_data_ptr::<f32>().add(0) },
            0.0
        );

        let tensor2 = zeros(
            alloc::vec![4, 5],
            ScalarType::Int,
            TensorShapeDynamism::DYNAMIC_BOUND,
        );
        assert_eq!(tensor2.tensor().dim(), 2);
        assert_eq!(tensor2.tensor().size(0), 4);
        assert_eq!(tensor2.tensor().size(1), 5);
        assert_eq!(tensor2.tensor().scalar_type(), ScalarType::Int);
        assert_eq!(
            unsafe { *tensor2.tensor().const_data_ptr::<i32>().add(0) },
            0
        );

        let tensor3 = zeros(
            alloc::vec![4, 5],
            ScalarType::Long,
            TensorShapeDynamism::DYNAMIC_BOUND,
        );
        assert_eq!(tensor3.tensor().dim(), 2);
        assert_eq!(tensor3.tensor().size(0), 4);
        assert_eq!(tensor3.tensor().size(1), 5);
        assert_eq!(tensor3.tensor().scalar_type(), ScalarType::Long);
        assert_eq!(
            unsafe { *tensor3.tensor().const_data_ptr::<i64>().add(0) },
            0
        );

        let tensor4 = zeros(
            alloc::vec![4, 5],
            ScalarType::Double,
            TensorShapeDynamism::DYNAMIC_BOUND,
        );
        assert_eq!(tensor4.tensor().dim(), 2);
        assert_eq!(tensor4.tensor().size(0), 4);
        assert_eq!(tensor4.tensor().size(1), 5);
        assert_eq!(tensor4.tensor().scalar_type(), ScalarType::Double);
        assert_eq!(
            unsafe { *tensor4.tensor().const_data_ptr::<f64>().add(0) },
            0.0
        );
    }

    // [spec:et:sem:tensor-ptr-maker.executorch.extension.rand-fn/test]
    // [spec:et:sem:tensor-ptr-maker.executorch.extension.rand-strided-fn/test]
    // rand_strided -> random_strided (fills the buffer from the distribution;
    // the in-range assertions below fail if random_strided is wrong):
    // [spec:et:sem:tensor-ptr-maker.executorch.extension.random-strided-fn/test]
    #[test]
    fn tensor_ptr_maker_test_create_rand_tensor() {
        setup();
        let tensor = rand(
            alloc::vec![4, 5],
            ScalarType::Float,
            TensorShapeDynamism::DYNAMIC_BOUND,
        );

        assert_eq!(tensor.tensor().dim(), 2);
        assert_eq!(tensor.tensor().size(0), 4);
        assert_eq!(tensor.tensor().size(1), 5);
        assert_eq!(tensor.tensor().scalar_type(), ScalarType::Float);

        let t = tensor.tensor();
        for i in 0..t.numel() {
            let val = unsafe { *t.const_data_ptr::<f32>().offset(i as isize) };
            assert!(val >= 0.0f32);
            assert!(val < 1.0f32);
        }
    }

    // [spec:et:sem:tensor-ptr-maker.executorch.extension.rand-fn/test]
    #[test]
    fn tensor_ptr_maker_test_create_rand_tensor_with_int_type() {
        setup();
        let tensor = rand(
            alloc::vec![4, 5],
            ScalarType::Int,
            TensorShapeDynamism::DYNAMIC_BOUND,
        );

        assert_eq!(tensor.tensor().dim(), 2);
        assert_eq!(tensor.tensor().size(0), 4);
        assert_eq!(tensor.tensor().size(1), 5);
        assert_eq!(tensor.tensor().scalar_type(), ScalarType::Int);

        let t = tensor.tensor();
        for i in 0..t.numel() {
            let val = unsafe { *t.const_data_ptr::<i32>().offset(i as isize) };
            assert_eq!(val, 0);
        }
    }

    // [spec:et:sem:tensor-ptr-maker.executorch.extension.rand-fn/test]
    #[test]
    fn tensor_ptr_maker_test_create_rand_tensor_with_double_type() {
        setup();
        let tensor = rand(
            alloc::vec![4, 5],
            ScalarType::Double,
            TensorShapeDynamism::DYNAMIC_BOUND,
        );

        assert_eq!(tensor.tensor().dim(), 2);
        assert_eq!(tensor.tensor().size(0), 4);
        assert_eq!(tensor.tensor().size(1), 5);
        assert_eq!(tensor.tensor().scalar_type(), ScalarType::Double);

        let t = tensor.tensor();
        for i in 0..t.numel() {
            let val = unsafe { *t.const_data_ptr::<f64>().offset(i as isize) };
            assert!(val >= 0.0);
            assert!(val < 1.0);
        }
    }

    // [spec:et:sem:tensor-ptr-maker.executorch.extension.rand-fn/test]
    #[test]
    fn tensor_ptr_maker_test_create_rand_tensor_with_half_type() {
        setup();
        let tensor = rand(
            alloc::vec![4, 5],
            ScalarType::Half,
            TensorShapeDynamism::DYNAMIC_BOUND,
        );

        assert_eq!(tensor.tensor().dim(), 2);
        assert_eq!(tensor.tensor().size(0), 4);
        assert_eq!(tensor.tensor().size(1), 5);
        assert_eq!(tensor.tensor().scalar_type(), ScalarType::Half);

        let t = tensor.tensor();
        for i in 0..t.numel() {
            let val = unsafe { *t.const_data_ptr::<Half>().offset(i as isize) }.to_f32();
            assert!(val >= 0.0);
            assert!(val < 1.0);
        }
    }

    // [spec:et:sem:tensor-ptr-maker.executorch.extension.rand-fn/test]
    #[test]
    fn tensor_ptr_maker_test_create_rand_tensor_with_bfloat_type() {
        setup();
        let tensor = rand(
            alloc::vec![4, 5],
            ScalarType::BFloat16,
            TensorShapeDynamism::DYNAMIC_BOUND,
        );

        assert_eq!(tensor.tensor().dim(), 2);
        assert_eq!(tensor.tensor().size(0), 4);
        assert_eq!(tensor.tensor().size(1), 5);
        assert_eq!(tensor.tensor().scalar_type(), ScalarType::BFloat16);

        let t = tensor.tensor();
        for i in 0..t.numel() {
            let val = unsafe { *t.const_data_ptr::<BFloat16>().offset(i as isize) }.to_f32();
            assert!(val >= 0.0);
            assert!(val < 1.0);
        }
    }

    // [spec:et:sem:tensor-ptr-maker.executorch.extension.randn-fn/test]
    // [spec:et:sem:tensor-ptr-maker.executorch.extension.randn-strided-fn/test]
    #[test]
    fn tensor_ptr_maker_test_create_randn_tensor() {
        setup();
        let tensor = randn(
            alloc::vec![100, 100],
            ScalarType::Float,
            TensorShapeDynamism::DYNAMIC_BOUND,
        );

        assert_eq!(tensor.tensor().dim(), 2);
        assert_eq!(tensor.tensor().size(0), 100);
        assert_eq!(tensor.tensor().size(1), 100);
        assert_eq!(tensor.tensor().scalar_type(), ScalarType::Float);

        let t = tensor.tensor();
        let mut sum = 0.0f32;
        for i in 0..t.numel() {
            sum += unsafe { *t.const_data_ptr::<f32>().offset(i as isize) };
        }
        let average = sum / t.numel() as f32;
        assert!((average - 0.0f32).abs() < 1.0f32);
    }

    // [spec:et:sem:tensor-ptr-maker.executorch.extension.randn-fn/test]
    #[test]
    fn tensor_ptr_maker_test_create_randn_tensor_with_double_type() {
        setup();
        let tensor = randn(
            alloc::vec![100, 100],
            ScalarType::Double,
            TensorShapeDynamism::DYNAMIC_BOUND,
        );

        assert_eq!(tensor.tensor().dim(), 2);
        assert_eq!(tensor.tensor().size(0), 100);
        assert_eq!(tensor.tensor().size(1), 100);
        assert_eq!(tensor.tensor().scalar_type(), ScalarType::Double);

        let t = tensor.tensor();
        let mut sum = 0.0f64;
        for i in 0..t.numel() {
            sum += unsafe { *t.const_data_ptr::<f64>().offset(i as isize) };
        }
        let average = sum / t.numel() as f64;
        assert!((average - 0.0).abs() < 1.0);
    }

    // [spec:et:sem:tensor-ptr-maker.executorch.extension.randint-fn/test]
    // [spec:et:sem:tensor-ptr-maker.executorch.extension.randint-strided-fn/test]
    #[test]
    fn tensor_ptr_maker_test_create_rand_int_tensor_with_int_type() {
        setup();
        let tensor = randint(
            10,
            20,
            alloc::vec![4, 5],
            ScalarType::Int,
            TensorShapeDynamism::DYNAMIC_BOUND,
        );

        assert_eq!(tensor.tensor().dim(), 2);
        assert_eq!(tensor.tensor().size(0), 4);
        assert_eq!(tensor.tensor().size(1), 5);
        assert_eq!(tensor.tensor().scalar_type(), ScalarType::Int);

        let t = tensor.tensor();
        for i in 0..t.numel() {
            let val = unsafe { *t.const_data_ptr::<i32>().offset(i as isize) };
            assert!(val >= 10);
            assert!(val < 20);
        }
    }

    // [spec:et:sem:tensor-ptr-maker.executorch.extension.randint-fn/test]
    #[test]
    fn tensor_ptr_maker_test_create_rand_int_tensor_with_long_type() {
        setup();
        let tensor = randint(
            10,
            20,
            alloc::vec![4, 5],
            ScalarType::Long,
            TensorShapeDynamism::DYNAMIC_BOUND,
        );

        assert_eq!(tensor.tensor().dim(), 2);
        assert_eq!(tensor.tensor().size(0), 4);
        assert_eq!(tensor.tensor().size(1), 5);
        assert_eq!(tensor.tensor().scalar_type(), ScalarType::Long);

        let t = tensor.tensor();
        for i in 0..t.numel() {
            let val = unsafe { *t.const_data_ptr::<i64>().offset(i as isize) };
            assert!(val >= 10);
            assert!(val < 20);
        }
    }

    // PORT-NOTE: the C++ `CreateRandnTensorWithIntType` body actually calls
    // `rand(...ScalarType::Int)` (not `randn`), matching the literal source; the
    // integer `rand` path yields all zeros.
    // [spec:et:sem:tensor-ptr-maker.executorch.extension.rand-fn/test]
    #[test]
    fn tensor_ptr_maker_test_create_randn_tensor_with_int_type() {
        setup();
        let tensor = rand(
            alloc::vec![4, 5],
            ScalarType::Int,
            TensorShapeDynamism::DYNAMIC_BOUND,
        );

        assert_eq!(tensor.tensor().dim(), 2);
        assert_eq!(tensor.tensor().size(0), 4);
        assert_eq!(tensor.tensor().size(1), 5);
        assert_eq!(tensor.tensor().scalar_type(), ScalarType::Int);

        let t = tensor.tensor();
        for i in 0..t.numel() {
            let val = unsafe { *t.const_data_ptr::<i32>().offset(i as isize) };
            assert_eq!(val, 0);
        }
    }

    // PORT-NOTE: no upstream C++ test exercised the `TensorPtrMaker::type` fluent
    // setter or the `*_like` free functions directly; these focused tests pin the
    // sem-rule behavior (`type_` override, and the `Undefined`-inherits-`other`'s
    // scalar type + layout-preservation contract of the `*_like` helpers).
    // [spec:et:sem:tensor-ptr-maker.executorch.extension.tensor-ptr-maker.type-fn/test]
    #[test]
    fn tensor_ptr_maker_test_type_setter() {
        setup();
        let mut data: [i32; 20] = [0; 20];
        data[0] = 42;
        // for_blob defaults to the passed type (Float); .type_ overrides it.
        let tensor = for_blob(
            data.as_mut_ptr() as *mut core::ffi::c_void,
            alloc::vec![4, 5],
            ScalarType::Float,
        )
        .type_(ScalarType::Int)
        .make_tensor_ptr();

        let t = tensor.tensor();
        assert_eq!(t.scalar_type(), ScalarType::Int);
        assert_eq!(unsafe { *t.const_data_ptr::<i32>().add(0) }, 42);
    }

    // [spec:et:sem:tensor-ptr-maker.executorch.extension.empty-like-fn/test]
    #[test]
    fn tensor_ptr_maker_test_empty_like() {
        setup();
        let other = full(
            alloc::vec![2, 3],
            Scalar::from_i64(9),
            ScalarType::Int,
            TensorShapeDynamism::DYNAMIC_BOUND,
        );
        let tensor = empty_like(
            &other,
            ScalarType::Undefined,
            TensorShapeDynamism::DYNAMIC_BOUND,
        );
        let t = tensor.tensor();
        // Undefined -> inherits other's scalar type and sizes/strides.
        assert_eq!(t.scalar_type(), ScalarType::Int);
        assert_eq!(t.dim(), 2);
        assert_eq!(t.size(0), 2);
        assert_eq!(t.size(1), 3);
        assert_eq!(*t.strides().at(0), 3);
        assert_eq!(*t.strides().at(1), 1);
        // Explicit type override is honored.
        let tensor2 = empty_like(
            &other,
            ScalarType::Float,
            TensorShapeDynamism::DYNAMIC_BOUND,
        );
        assert_eq!(tensor2.tensor().scalar_type(), ScalarType::Float);
    }

    // [spec:et:sem:tensor-ptr-maker.executorch.extension.full-like-fn/test]
    #[test]
    fn tensor_ptr_maker_test_full_like() {
        setup();
        let other = zeros(
            alloc::vec![2, 3],
            ScalarType::Float,
            TensorShapeDynamism::DYNAMIC_BOUND,
        );
        let tensor = full_like(
            &other,
            Scalar::from_i64(7),
            ScalarType::Undefined,
            TensorShapeDynamism::DYNAMIC_BOUND,
        );
        let t = tensor.tensor();
        assert_eq!(t.scalar_type(), ScalarType::Float);
        assert_eq!(t.size(0), 2);
        assert_eq!(t.size(1), 3);
        for i in 0..t.numel() {
            assert_eq!(
                unsafe { *t.const_data_ptr::<f32>().offset(i as isize) },
                7.0
            );
        }
    }

    // [spec:et:sem:tensor-ptr-maker.executorch.extension.ones-like-fn/test]
    #[test]
    fn tensor_ptr_maker_test_ones_like() {
        setup();
        let other = zeros(
            alloc::vec![2, 3],
            ScalarType::Int,
            TensorShapeDynamism::DYNAMIC_BOUND,
        );
        let tensor = ones_like(
            &other,
            ScalarType::Undefined,
            TensorShapeDynamism::DYNAMIC_BOUND,
        );
        let t = tensor.tensor();
        assert_eq!(t.scalar_type(), ScalarType::Int);
        assert_eq!(t.numel(), 6);
        for i in 0..t.numel() {
            assert_eq!(unsafe { *t.const_data_ptr::<i32>().offset(i as isize) }, 1);
        }
    }

    // [spec:et:sem:tensor-ptr-maker.executorch.extension.zeros-like-fn/test]
    #[test]
    fn tensor_ptr_maker_test_zeros_like() {
        setup();
        let other = ones(
            alloc::vec![2, 3],
            ScalarType::Float,
            TensorShapeDynamism::DYNAMIC_BOUND,
        );
        let tensor = zeros_like(
            &other,
            ScalarType::Undefined,
            TensorShapeDynamism::DYNAMIC_BOUND,
        );
        let t = tensor.tensor();
        assert_eq!(t.scalar_type(), ScalarType::Float);
        assert_eq!(t.numel(), 6);
        for i in 0..t.numel() {
            assert_eq!(
                unsafe { *t.const_data_ptr::<f32>().offset(i as isize) },
                0.0
            );
        }
    }

    // [spec:et:sem:tensor-ptr-maker.executorch.extension.rand-like-fn/test]
    #[test]
    fn tensor_ptr_maker_test_rand_like() {
        setup();
        let other = zeros(
            alloc::vec![4, 5],
            ScalarType::Float,
            TensorShapeDynamism::DYNAMIC_BOUND,
        );
        let tensor = rand_like(
            &other,
            ScalarType::Undefined,
            TensorShapeDynamism::DYNAMIC_BOUND,
        );
        let t = tensor.tensor();
        assert_eq!(t.scalar_type(), ScalarType::Float);
        assert_eq!(t.size(0), 4);
        assert_eq!(t.size(1), 5);
        for i in 0..t.numel() {
            let val = unsafe { *t.const_data_ptr::<f32>().offset(i as isize) };
            assert!(val >= 0.0f32);
            assert!(val < 1.0f32);
        }
    }

    // [spec:et:sem:tensor-ptr-maker.executorch.extension.randn-like-fn/test]
    #[test]
    fn tensor_ptr_maker_test_randn_like() {
        setup();
        let other = zeros(
            alloc::vec![100, 100],
            ScalarType::Float,
            TensorShapeDynamism::DYNAMIC_BOUND,
        );
        let tensor = randn_like(
            &other,
            ScalarType::Undefined,
            TensorShapeDynamism::DYNAMIC_BOUND,
        );
        let t = tensor.tensor();
        assert_eq!(t.scalar_type(), ScalarType::Float);
        assert_eq!(t.numel(), 10000);
        let mut sum = 0.0f32;
        for i in 0..t.numel() {
            sum += unsafe { *t.const_data_ptr::<f32>().offset(i as isize) };
        }
        let average = sum / t.numel() as f32;
        assert!((average - 0.0f32).abs() < 1.0f32);
    }

    // [spec:et:sem:tensor-ptr-maker.executorch.extension.randint-like-fn/test]
    #[test]
    fn tensor_ptr_maker_test_randint_like() {
        setup();
        let other = zeros(
            alloc::vec![4, 5],
            ScalarType::Int,
            TensorShapeDynamism::DYNAMIC_BOUND,
        );
        let tensor = randint_like(
            &other,
            10,
            20,
            ScalarType::Undefined,
            TensorShapeDynamism::DYNAMIC_BOUND,
        );
        let t = tensor.tensor();
        assert_eq!(t.scalar_type(), ScalarType::Int);
        assert_eq!(t.size(0), 4);
        assert_eq!(t.size(1), 5);
        for i in 0..t.numel() {
            let val = unsafe { *t.const_data_ptr::<i32>().offset(i as isize) };
            assert!(val >= 10);
            assert!(val < 20);
        }
    }
}
