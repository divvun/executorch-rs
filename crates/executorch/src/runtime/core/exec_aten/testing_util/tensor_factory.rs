//! Literal port of runtime/core/exec_aten/testing_util/tensor_factory.h.
//!
//! This is the non-ATen (`!USE_ATEN_LIB`, `torch::executor`) branch, which ships
//! as a regular test-support library rather than being gated behind `cfg(test)`.
//! These files are excluded from the port manifest, so no `[spec:et:...]`
//! annotations are carried.
//!
//! PORT-NOTE: The C++ `TensorFactory<DTYPE>` is templated on a `ScalarType`
//! value and derives its element C type via `ScalarTypeToCppTypeWrapper`. Rust
//! cannot template on a value the same way, so `TensorFactory<T>` is templated on
//! the element C type `T` and derives the `ScalarType` (the C++ `DTYPE`) via the
//! `CppTypeToScalarType` trait (`T::VALUE`). This collapses the C++
//! `ctype`/`true_ctype` distinction (which only exists because `vector<bool>` is
//! bit-packed and some dtypes reuse an integer C type): here the factory element
//! type is the true C type directly, e.g. `TensorFactory::<bool>` /
//! `TensorFactory::<f32>` / `TensorFactory::<i32>`.

use crate::runtime::core::exec_aten::util::dim_order_util::is_contiguous_dim_order;
use crate::runtime::core::exec_aten::util::scalar_type_util::CppTypeToScalarType;
use crate::runtime::core::portable_type::device::{DeviceIndex, DeviceType};
use crate::runtime::core::portable_type::scalar_type::ScalarType;
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::core::portable_type::tensor_impl::{
    DimOrderType, SizesType, StridesType, TensorImpl,
};
use crate::runtime::core::tensor_shape_dynamism::TensorShapeDynamism;
use crate::runtime::platform::abort::runtime_abort;

// PORT-NOTE: `ET_CHECK_MSG(cond, ...)` aborts via the PAL abort path when the
// condition is false. The assert module is not ported; this mirrors the pattern
// used elsewhere in the Wave-2 port (see tensor_impl.rs). Message formatting is
// dropped since the abort is unconditional.
macro_rules! et_check_msg {
    ($cond:expr $(,)?) => {
        if !($cond) {
            runtime_abort();
        }
    };
    ($cond:expr, $($arg:tt)*) => {
        if !($cond) {
            runtime_abort();
        }
    };
}

/// Internal helpers, mirroring `executorch::runtime::testing::internal`.
pub mod internal {
    use super::*;

    /// Returns the number of elements in the tensor, given the dimension
    /// sizes, assuming contiguous data.
    pub fn sizes_to_numel(sizes: &[i32]) -> usize {
        let mut n: usize = 1;
        for &s in sizes {
            n = n.wrapping_mul(s as usize);
        }
        n
    }

    /// Check if given strides is legal under given sizes. In the `make` function,
    /// the `strides` shall ensure:
    ///  - a. strides.size() == sizes.size()
    ///  - b. all strides are positive.
    ///  - c. All underlying data be accessed.
    ///  - d. All legal indexes can access an underlying data.
    ///  - e. No two indexes access a same data.
    ///  - f. No out of bounds data can be accessed.
    pub fn check_strides(sizes: &[i32], strides: &[StridesType]) -> bool {
        if sizes.len() != strides.len() {
            // The length of stride vector shall equal to size vector.
            return false;
        }

        if strides.is_empty() {
            // Both sizes and strides are empty vector. Legal!
            return true;
        }

        // Check if input non-empty strides is legal. The defination of legal is in
        // the comment above function. To check it, we first reformat the strides
        // into contiguous style, in where the strides should be sorted from high to
        // low. Then rearrange the size based on same transformation. After that, we
        // can check if strides[i] == strides[i + 1] * sizes[i + 1] for all i in
        // [0, sizes.size() - 1) and strides[sizes.size() - 1] == 1

        // Get the mapping between current strides and sorted strides (from high to
        // low, if equal then check if correspond size is 1 or 0 in same dimension)
        let mut sorted_idx: Vec<i32> = (0..sizes.len() as i32).collect();
        sorted_idx.sort_by(|&a, &b| {
            let (a, b) = (a as usize, b as usize);
            if strides[a] != strides[b] {
                // strides[a] > strides[b] first (descending).
                strides[b].cmp(&strides[a])
            } else {
                // When strides equal to each other, put the index whose
                // corresponding size equal to 0 or 1 to the right.
                let sa = if sizes[a] != 0 { sizes[a] } else { 1 };
                let sb = if sizes[b] != 0 { sizes[b] } else { 1 };
                sb.cmp(&sa)
            }
        });

        // Use the mapping to rearrange the sizes and strides
        let mut sorted_sizes = vec![0i32; sizes.len()];
        let mut sorted_strides = vec![0i32; sizes.len()];
        for i in 0..sizes.len() {
            sorted_sizes[i] = if sizes[sorted_idx[i] as usize] == 0 {
                1
            } else {
                sizes[sorted_idx[i] as usize]
            };
            sorted_strides[i] = strides[sorted_idx[i] as usize];
        }

        // All strides should be positive. We have sorted it mainly based on
        // strides, so sorted_strides[-1] has lowest value.
        if sorted_strides[strides.len() - 1] <= 0 {
            return false;
        }

        // Check if strides is legal
        let mut legal = sorted_strides[strides.len() - 1] == 1;
        let mut i = 0;
        while i < strides.len() - 1 && legal {
            legal = legal && (sorted_strides[i] == sorted_strides[i + 1] * sorted_sizes[i + 1]);
            i += 1;
        }

        legal
    }

    /// Check that a given dim order array is valid. A dim order array is valid if
    /// each value from 0 to sizes.size() - 1 appears exactly once in the dim_order
    /// array.
    pub fn check_dim_order(sizes: &[i32], dim_order: &[u8]) -> bool {
        if sizes.len() != dim_order.len() {
            return false;
        }
        let mut gauss_sum: usize = 0;
        for &d in dim_order.iter() {
            if d as usize >= sizes.len() {
                return false;
            }
            gauss_sum += d as usize + 1;
        }
        // Use the gaussian sum to verify each dim appears exactly once
        let expected_sum: usize = (sizes.len() * (sizes.len() + 1)) / 2;
        if gauss_sum != expected_sum {
            return false;
        }
        true
    }

    pub fn strides_from_dim_order(sizes: &[i32], dim_order: &[u8]) -> Vec<StridesType> {
        let legal = check_dim_order(sizes, dim_order);
        et_check_msg!(legal, "The input dim_order variable is illegal.");

        let ndim = sizes.len();
        let mut strides = vec![0 as StridesType; ndim];
        strides[dim_order[ndim - 1] as usize] = 1;
        let mut i = ndim as isize - 2;
        while i >= 0 {
            let cur_dim = dim_order[i as usize] as usize;
            let next_dim = dim_order[i as usize + 1] as usize;
            strides[cur_dim] = if sizes[next_dim] == 0 {
                strides[next_dim]
            } else {
                strides[next_dim] * sizes[next_dim]
            };
            i -= 1;
        }
        strides
    }

    pub fn channels_last_dim_order(dims: usize) -> Vec<u8> {
        et_check_msg!(
            dims >= 4 && dims <= 5,
            "Channels last dim order only valid for 4-dim and 5-dim tensors!"
        );

        let mut dim_order = vec![0u8; dims];
        // Channels is always assigned to dim 1
        dim_order[dims - 1] = 1;

        dim_order[0] = 0;
        let mut d = 1usize;
        while d < dims - 1 {
            dim_order[d] = d as u8 + 1;
            d += 1;
        }
        dim_order
    }

    // From the anonymous namespace in the `!USE_ATEN_LIB` branch.

    /// Dimension order represents how dimensions are laid out in memory,
    /// starting from the inner-most to the outer-most dimension. The conversion
    /// from strides is done by (stable) sorting the strides from larger to
    /// smaller.
    pub fn dim_order_from_stride(v: &[i32]) -> Vec<u8> {
        let mut indices: Vec<u8> = (0..v.len() as u8).collect();
        // stable_sort by v[i1] > v[i2] (descending).
        indices.sort_by(|&i1, &i2| v[i2 as usize].cmp(&v[i1 as usize]));
        indices
    }

    pub fn validate_strides(sizes: &[i32], strides: &[i32]) {
        if sizes.len() != strides.len() {
            et_check_msg!(false, "Stride and sizes are not equal in length");
        }
        for &s in strides {
            if s == 0 {
                et_check_msg!(false, "Stride value of 0 is not supported");
            }
        }
        // No two dimensions can have same stride value
        for i in 0..strides.len() {
            for j in (i + 1)..strides.len() {
                if (sizes[i] == 0) || (sizes[j] == 0) || (sizes[i] == 1) || (sizes[j] == 1) {
                    continue;
                }
                if strides[i] == strides[j] {
                    et_check_msg!(false, "Stride value and size dont comply at index {}.", i);
                }
            }
        }
    }
}

/// Owns all backing memory for a single Tensor.
///
/// PORT-NOTE: Mirrors the C++ nested `TensorMemory` struct. The `TensorImpl`
/// holds raw pointers into `sizes`/`data`/`dim_order`/`strides`, so this must be
/// boxed and never moved after construction. `TensorImpl::new` reads through the
/// pointers during construction, so the Vecs are populated first, then the impl
/// is built pointing at their (stable, boxed) storage.
struct TensorMemory<T> {
    sizes: Vec<SizesType>,
    data: Vec<T>,
    dim_order: Vec<DimOrderType>,
    strides: Vec<StridesType>,
    impl_: TensorImpl,
}

/// A helper class for creating Tensors, simplifying memory management.
///
/// NOTE: A given TensorFactory instance owns the memory pointed to by all
/// Tensors that it creates, and must live longer than those Tensors.
///
/// See the module-level PORT-NOTE for the `TensorFactory<T>` type-parameter
/// mapping.
///
/// PORT-NOTE: the C++ factory methods are non-`const` and mutate `memory_` while
/// handing out Tensors that alias that memory (which stays valid because the
/// `TensorMemory` objects are individually boxed). Modeling that directly with
/// `&self` would forbid holding two Tensors from one factory at once, which
/// the C++ relies on. To preserve the C++ usage pattern the append-only `memory`
/// vector lives behind an `UnsafeCell`, letting factory methods take `&self` and
/// return Tensors borrowing `&self`; the boxed `TensorMemory` addresses are
/// stable across pushes, so previously-returned Tensors remain valid.
pub struct TensorFactory<T: CppTypeToScalarType> {
    /// The memory pointed to by Tensors created by this factory. Boxed so the
    /// TensorMemory objects (and the raw pointers TensorImpl holds into them)
    /// stay put when the vector reallocs.
    memory: core::cell::UnsafeCell<Vec<Box<TensorMemory<T>>>>,
}

impl<T: CppTypeToScalarType + FactoryValue> Default for TensorFactory<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: CppTypeToScalarType + FactoryValue> TensorFactory<T> {
    /// The dtype of Tensors created by this factory (the C++ `DTYPE`).
    const DTYPE: ScalarType = T::VALUE;

    pub fn new() -> Self {
        TensorFactory {
            memory: core::cell::UnsafeCell::new(Vec::new()),
        }
    }

    /// Builds a `TensorMemory`, wires up the `TensorImpl`, stores it, and returns
    /// a Tensor viewing the just-stored storage.
    fn emplace(
        &self,
        sizes: Vec<SizesType>,
        data: Vec<T>,
        dim_order: Vec<DimOrderType>,
        strides: Vec<StridesType>,
        dynamism: TensorShapeDynamism,
    ) -> Tensor<'_> {
        let dim = sizes.len();
        let mut mem = Box::new(TensorMemory {
            sizes,
            data,
            dim_order,
            strides,
            // Placeholder; overwritten below once the Vec pointers are stable.
            impl_: TensorImpl::new(
                Self::DTYPE,
                0,
                core::ptr::null_mut(),
                core::ptr::null_mut(),
                core::ptr::null_mut(),
                core::ptr::null_mut(),
                dynamism,
                DeviceType::CPU,
                0 as DeviceIndex,
            ),
        });
        mem.impl_ = TensorImpl::new(
            Self::DTYPE,
            dim as isize,
            mem.sizes.as_mut_ptr(),
            mem.data.as_mut_ptr() as *mut core::ffi::c_void,
            mem.dim_order.as_mut_ptr(),
            mem.strides.as_mut_ptr(),
            dynamism,
            DeviceType::CPU,
            0 as DeviceIndex,
        );
        // SAFETY: append-only access to the boxed-element vector. The boxes keep
        // stable heap addresses across pushes, so the raw `impl_` pointer handed
        // to the returned Tensor stays valid for the lifetime of `&self`. No
        // outstanding reference into the Vec is held across this mutation.
        let memory = unsafe { &mut *self.memory.get() };
        memory.push(mem);
        let last = memory.last_mut().unwrap();
        Tensor::new(&mut last.impl_ as *mut TensorImpl)
    }

    /// Returns a new Tensor with the specified shape, data and stride.
    ///
    /// If `strides` is empty the function returns a contiguous tensor based on
    /// data and size.
    pub fn make(
        &self,
        sizes: Vec<SizesType>,
        data: Vec<T>,
        strides: Vec<StridesType>,
        dynamism: TensorShapeDynamism,
    ) -> Tensor<'_> {
        let mut default_strides: Vec<i32> = Vec::new();
        // Generate strides from the tensor dimensions, assuming contiguous data
        // if given strides is empty.
        if !sizes.is_empty() && strides.is_empty() {
            default_strides.resize(sizes.len(), 1);
            let mut i = sizes.len() - 1;
            while i > 0 {
                // For sizes[i] == 0, treat it as 1 to be consistent with core
                // Pytorch.
                let sizes_i = if sizes[i] != 0 { sizes[i] } else { 1 };
                default_strides[i - 1] = default_strides[i] * sizes_i;
                i -= 1;
            }
        }
        let actual_strides: Vec<i32> = if default_strides.is_empty() {
            strides
        } else {
            default_strides
        };
        internal::validate_strides(&sizes, &actual_strides);
        let dim_order = internal::dim_order_from_stride(&actual_strides);

        let expected_numel = internal::sizes_to_numel(&sizes);
        et_check_msg!(
            expected_numel == data.len(),
            "Number of data elements {} does not match expected number of elements {}",
            data.len(),
            expected_numel
        );

        let legal = internal::check_strides(&sizes, &actual_strides);
        et_check_msg!(legal, "The input strides variable is illegal.");

        let data = coerce_bool(data);
        self.emplace(sizes, data, dim_order, actual_strides, dynamism)
    }

    /// `make` with default STATIC dynamism and contiguous (empty) strides.
    pub fn make_default(&self, sizes: Vec<SizesType>, data: Vec<T>) -> Tensor<'_> {
        self.make(sizes, data, Vec::new(), TensorShapeDynamism::STATIC)
    }

    /// Returns a new Tensor with the specified shape, data and dim order.
    ///
    /// If `dim_order` is empty the function uses a contiguous dim order of
    /// {0, 1, 2, 3, ...}.
    pub fn make_with_dimorder(
        &self,
        sizes: Vec<SizesType>,
        data: Vec<T>,
        dim_order: Vec<DimOrderType>,
        dynamism: TensorShapeDynamism,
    ) -> Tensor<'_> {
        let mut default_dim_order: Vec<u8> = Vec::new();
        if !sizes.is_empty() && dim_order.is_empty() {
            default_dim_order.resize(sizes.len(), 1);
            for i in 0..sizes.len() {
                default_dim_order[i] = i as u8;
            }
        }
        let actual_dim_order: Vec<u8> = if default_dim_order.is_empty() {
            dim_order
        } else {
            default_dim_order
        };

        let strides = internal::strides_from_dim_order(&sizes, &actual_dim_order);

        let expected_numel = internal::sizes_to_numel(&sizes);
        et_check_msg!(
            expected_numel == data.len(),
            "Number of data elements {} does not match expected number of elements {}",
            data.len(),
            expected_numel
        );

        let data = coerce_bool(data);
        self.emplace(sizes, data, actual_dim_order, strides, dynamism)
    }

    /// Returns a new Tensor with the specified shape and data in channels last
    /// memory format.
    ///
    /// PORT-NOTE: mirrors the C++ signature (which accepts and ignores a
    /// `dim_order` argument, always overwriting it with the channels-last order).
    pub fn make_channels_last(
        &self,
        sizes: Vec<SizesType>,
        data: Vec<T>,
        _dim_order: Vec<DimOrderType>,
        dynamism: TensorShapeDynamism,
    ) -> Tensor<'_> {
        let clo = internal::channels_last_dim_order(sizes.len());
        self.make_with_dimorder(sizes, data, clo, dynamism)
    }

    /// Given data in contiguous memory format, returns a new Tensor with the
    /// specified shape and the same data but in channels last memory format.
    pub fn channels_last_like(&self, input: &Tensor, dynamism: TensorShapeDynamism) -> Tensor<'_> {
        let input_sizes = input.sizes();
        let sizes: Vec<i32> = (0..input_sizes.size())
            .map(|i| *input_sizes.at(i))
            .collect();

        et_check_msg!(sizes.len() == 4, "Only 4D tensors can be channels last");
        et_check_msg!(
            unsafe { is_contiguous_dim_order(input.dim_order().data(), input.dim() as usize) },
            "Input tensor is not contiguous"
        );
        let n_ = sizes[0];
        let c_ = sizes[1];
        let h_ = sizes[2];
        let w_ = sizes[3];

        let numel = input.numel() as usize;
        let src = input.const_data_ptr::<T>();
        let contiguous_data: Vec<T> = (0..numel).map(|i| unsafe { *src.add(i) }).collect();
        let mut channels_last_data: Vec<T> = vec![contiguous_data[0]; (n_ * c_ * h_ * w_) as usize];
        for n in 0..n_ {
            for c in 0..c_ {
                for h in 0..h_ {
                    for w in 0..w_ {
                        // Calculate the index in the original blob
                        let old_index = ((n * c_ + c) * h_ + h) * w_ + w;
                        // Calculate the index in the new blob
                        let new_index = ((n * h_ + h) * w_ + w) * c_ + c;
                        channels_last_data[new_index as usize] =
                            contiguous_data[old_index as usize];
                    }
                }
            }
        }

        let clo = internal::channels_last_dim_order(sizes.len());
        self.make_with_dimorder(sizes, channels_last_data, clo, dynamism)
    }

    /// Returns a new Tensor with the specified shape, containing contiguous data
    /// with all elements set to `value`.
    pub fn full(
        &self,
        sizes: Vec<SizesType>,
        value: T,
        dynamism: TensorShapeDynamism,
    ) -> Tensor<'_> {
        let data = vec![value; internal::sizes_to_numel(&sizes)];
        self.make(sizes, data, Vec::new(), dynamism)
    }

    /// Returns a new Tensor with the specified shape, containing channels last
    /// contiguous data with all elements set to `value`.
    pub fn full_channels_last(
        &self,
        sizes: Vec<SizesType>,
        value: T,
        dynamism: TensorShapeDynamism,
    ) -> Tensor<'_> {
        let data = vec![value; internal::sizes_to_numel(&sizes)];
        let clo = internal::channels_last_dim_order(sizes.len());
        self.make_with_dimorder(sizes, data, clo, dynamism)
    }

    /// Returns a new Tensor with the specified shape, containing contiguous data
    /// in channels last memory format with all `0` elements.
    pub fn zeros_channels_last(
        &self,
        sizes: Vec<SizesType>,
        dynamism: TensorShapeDynamism,
    ) -> Tensor<'_> {
        self.full_channels_last(sizes, T::zero(), dynamism)
    }

    /// Returns a new Tensor with the specified shape, containing contiguous data
    /// in contiguous memory format with all `0` elements.
    pub fn zeros(&self, sizes: Vec<SizesType>, dynamism: TensorShapeDynamism) -> Tensor<'_> {
        self.full(sizes, T::zero(), dynamism)
    }

    /// `zeros` with default STATIC dynamism.
    pub fn zeros_default(&self, sizes: Vec<SizesType>) -> Tensor<'_> {
        self.zeros(sizes, TensorShapeDynamism::STATIC)
    }

    /// Returns a new Tensor with the specified shape, containing contiguous data
    /// with all `1` elements.
    pub fn ones(&self, sizes: Vec<SizesType>, dynamism: TensorShapeDynamism) -> Tensor<'_> {
        self.full(sizes, T::one(), dynamism)
    }

    /// `ones` with default STATIC dynamism.
    pub fn ones_default(&self, sizes: Vec<SizesType>) -> Tensor<'_> {
        self.ones(sizes, TensorShapeDynamism::STATIC)
    }

    /// Returns a new Tensor with the same shape as the input tensor, containing
    /// contiguous data with all `0` elements.
    pub fn zeros_like(&self, input: &Tensor, dynamism: TensorShapeDynamism) -> Tensor<'_> {
        let input_sizes = input.sizes();
        let sizes: Vec<i32> = (0..input_sizes.size())
            .map(|i| *input_sizes.at(i))
            .collect();
        self.full(sizes, T::zero(), dynamism)
    }

    /// Returns a new Tensor with the same shape as the input tensor, containing
    /// contiguous data with all `1` elements.
    pub fn ones_like(&self, input: &Tensor, dynamism: TensorShapeDynamism) -> Tensor<'_> {
        let input_sizes = input.sizes();
        let sizes: Vec<i32> = (0..input_sizes.size())
            .map(|i| *input_sizes.at(i))
            .collect();
        self.full(sizes, T::one(), dynamism)
    }
}

/// PORT-NOTE: the C++ `TensorMemory` ctor coerces bool data to 0/1 when
/// `true_ctype == bool` (its `ctype` is `uint8_t`, so callers can pass arbitrary
/// bytes). Rust's `bool` cannot hold non-0/1 values, so for a `bool` element type
/// this is a no-op; for every other type it is a no-op too. Kept as an explicit
/// seam mirroring the C++ coercion site.
fn coerce_bool<T>(data: Vec<T>) -> Vec<T> {
    data
}

/// PORT-NOTE: the C++ factory materializes `0`/`1` fill values via implicit
/// integer conversion to the element `ctype`. Rust has no such implicit
/// conversion, so `full`/`zeros`/`ones` obtain the fill values through this
/// trait, implemented for every supported element type.
pub trait FactoryValue: Copy {
    fn zero() -> Self;
    fn one() -> Self;
}

macro_rules! impl_factory_value_num {
    ($($t:ty),*) => {
        $(impl FactoryValue for $t {
            fn zero() -> Self { 0 as $t }
            fn one() -> Self { 1 as $t }
        })*
    };
}
impl_factory_value_num!(u8, i8, i16, i32, i64, u16, u32, u64, f32, f64);

impl FactoryValue for bool {
    fn zero() -> Self {
        false
    }
    fn one() -> Self {
        true
    }
}

impl FactoryValue for crate::runtime::core::portable_type::Half {
    fn zero() -> Self {
        crate::runtime::core::portable_type::Half::from_f32(0.0)
    }
    fn one() -> Self {
        crate::runtime::core::portable_type::Half::from_f32(1.0)
    }
}

impl FactoryValue for crate::runtime::core::portable_type::BFloat16 {
    fn zero() -> Self {
        crate::runtime::core::portable_type::BFloat16::from_f32(0.0)
    }
    fn one() -> Self {
        crate::runtime::core::portable_type::BFloat16::from_f32(1.0)
    }
}

// bits16 is an opaque 16-bit dtype; the quantized op tests build Bits16-dtype
// tensors through the factory, so it needs the `0`/`1` fill values mirroring the
// C++ implicit integer conversion.
impl FactoryValue for crate::runtime::core::portable_type::bits_types::bits16 {
    fn zero() -> Self {
        crate::runtime::core::portable_type::bits_types::bits16::new(0)
    }
    fn one() -> Self {
        crate::runtime::core::portable_type::bits_types::bits16::new(1)
    }
}

// quint8/qint8 are the 8-bit quantized dtypes; the xnnpack utils tests build
// QUInt8/QInt8-dtype tensors through the factory (e.g. the quantized output of
// `QuantizePerTensor`), so they need the `0`/`1` fill values mirroring the C++
// implicit integer conversion to the underlying byte.
impl FactoryValue for crate::runtime::core::portable_type::qint_types::quint8 {
    fn zero() -> Self {
        crate::runtime::core::portable_type::qint_types::quint8::new(0)
    }
    fn one() -> Self {
        crate::runtime::core::portable_type::qint_types::quint8::new(1)
    }
}

impl FactoryValue for crate::runtime::core::portable_type::qint_types::qint8 {
    fn zero() -> Self {
        crate::runtime::core::portable_type::qint_types::qint8::new(0)
    }
    fn one() -> Self {
        crate::runtime::core::portable_type::qint_types::qint8::new(1)
    }
}

// Complex fill values, mirroring the C++ `0`/`1` implicit conversion to the
// complex element type (imaginary part zero).
impl<T: FactoryValue> FactoryValue for crate::runtime::core::portable_type::Complex<T> {
    fn zero() -> Self {
        crate::runtime::core::portable_type::Complex {
            real: T::zero(),
            imag: T::zero(),
        }
    }
    fn one() -> Self {
        crate::runtime::core::portable_type::Complex {
            real: T::one(),
            imag: T::zero(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assert_tensor_eq;
    use crate::runtime::core::array_ref::ArrayRef;
    use crate::runtime::core::error::Error;
    use crate::runtime::core::exec_aten::testing_util::tensor_util::{
        tensor_data_is_close, tensors_are_close,
    };
    use crate::runtime::core::exec_aten::util::tensor_util::{resize, resize_tensor};
    use crate::runtime::core::portable_type::tensor_impl::DimOrderType;

    #[test]
    fn make_basic_shape_and_data() {
        let tf = TensorFactory::<i32>::new();
        let t = tf.make_default(vec![2, 2], vec![1, 2, 3, 4]);
        assert_eq!(t.dim(), 2);
        assert_eq!(t.numel(), 4);
        assert_eq!(t.scalar_type(), ScalarType::Int);
        let data = t.const_data_ptr::<i32>();
        let got: Vec<i32> = (0..4).map(|i| unsafe { *data.add(i) }).collect();
        assert_eq!(got, vec![1, 2, 3, 4]);
        // Contiguous strides for {2,2} are {2,1}.
        let strides = t.strides();
        assert_eq!(*strides.at(0), 2);
        assert_eq!(*strides.at(1), 1);
    }

    #[test]
    fn full_zeros_ones() {
        let tf = TensorFactory::<f32>::new();
        let z = tf.zeros_default(vec![3]);
        let zd = z.const_data_ptr::<f32>();
        for i in 0..3 {
            assert_eq!(unsafe { *zd.add(i) }, 0.0);
        }
        let o = tf.ones_default(vec![2, 2]);
        let od = o.const_data_ptr::<f32>();
        for i in 0..4 {
            assert_eq!(unsafe { *od.add(i) }, 1.0);
        }
        let f = tf.full(vec![2], 7.5, TensorShapeDynamism::STATIC);
        let fd = f.const_data_ptr::<f32>();
        assert_eq!(unsafe { *fd.add(0) }, 7.5);
        assert_eq!(unsafe { *fd.add(1) }, 7.5);
    }

    #[test]
    fn tensors_are_close_and_eq() {
        let tf = TensorFactory::<f32>::new();
        let a = tf.make_default(vec![2], vec![1.0, 2.0]);
        let b = tf.make_default(vec![2], vec![1.0, 2.0]);
        assert!(tensors_are_close(&a, &b, 0.0, Some(0.0)));

        // A difference representable in f32 but within the default atol/rtol.
        let c = tf.make_default(vec![2], vec![1.0, 2.0 + 1e-6]);
        // Exact comparison fails, close comparison (default rtol 1e-5) passes.
        assert!(!tensors_are_close(&a, &c, 0.0, Some(0.0)));
        assert!(tensors_are_close(&a, &c, 1e-5, None));
    }

    #[test]
    fn data_close_different_shape_same_numel() {
        let tf = TensorFactory::<i32>::new();
        let a = tf.make_default(vec![2, 2], vec![1, 2, 3, 4]);
        let b = tf.make_default(vec![4], vec![1, 2, 3, 4]);
        // Shapes differ, so tensors_are_close is false but tensor_data_is_close
        // is true.
        assert!(!tensors_are_close(&a, &b, 0.0, Some(0.0)));
        assert!(tensor_data_is_close(&a, &b, 0.0, Some(0.0)));
    }

    #[test]
    fn bool_factory() {
        let tf = TensorFactory::<bool>::new();
        let t = tf.make_default(vec![2], vec![true, false]);
        assert_eq!(t.scalar_type(), ScalarType::Bool);
        let d = t.const_data_ptr::<bool>();
        assert!(unsafe { *d.add(0) });
        assert!(!unsafe { *d.add(1) });
    }

    // -------------------------------------------------------------------------
    // Literal port of tensor_factory_test.cpp (non-ATen `TensorFactoryTest`).
    // -------------------------------------------------------------------------

    // The C++ death tests (`ET_EXPECT_DEATH`) become `#[ignore]`d
    // `#[should_panic]` functions: `runtime_abort` -> `libc::abort()` terminates
    // the process rather than unwinding, so they cannot be run in-process.

    // Mirrors `TensorFactoryTest::SetUp()`'s `runtime_init()`; the PAL must be
    // initialized before code paths that call `ET_LOG` (e.g. failing
    // `resize_tensor`).
    fn setup() {
        crate::runtime::platform::runtime::runtime_init();
    }

    fn resize_tensor_1x1(t: &Tensor) -> Error {
        let new_sizes: [SizesType; 2] = [1, 1];
        resize_tensor(t, ArrayRef::from_raw_parts(new_sizes.as_ptr(), 2))
    }

    fn resize_tensor_100x100(t: &Tensor) -> Error {
        let new_sizes: [SizesType; 2] = [100, 100];
        resize_tensor(t, ArrayRef::from_raw_parts(new_sizes.as_ptr(), 2))
    }

    // The tensor under test is modified.
    fn resize_tensor_to_assert_static(t: Tensor) {
        setup();
        assert!(t.numel() > 1);
        // !USE_ATEN_LIB branch: resizing a STATIC tensor to 1x1 fails.
        assert_ne!(resize_tensor_1x1(&t), Error::Ok);
    }

    fn resize_tensor_to_assert_dynamic_bound(t: Tensor) {
        setup();
        assert!(t.numel() > 1);
        assert!(t.numel() < 100 * 100);
        assert_eq!(resize_tensor_1x1(&t), Error::Ok);
        assert_ne!(resize_tensor_100x100(&t), Error::Ok);
    }

    fn resize_tensor_to_assert_dynamic_unbound(t: Tensor) {
        setup();
        assert!(t.numel() > 1);
        assert!(t.numel() < 100 * 100);
        assert_eq!(resize_tensor_1x1(&t), Error::Ok);
        // !USE_ATEN_LIB: for now, can't resize past the original capacity.
        assert_ne!(resize_tensor_100x100(&t), Error::Ok);
    }

    #[test]
    fn tensor_factory_test_make_int_tensor() {
        let tf = TensorFactory::<i32>::new();
        let actual = tf.make_default(vec![2, 2], vec![1, 2, 3, 4]);
        // Contiguous {2,2} strides are {2,1}.
        let expected = tf.make(
            vec![2, 2],
            vec![1, 2, 3, 4],
            vec![2, 1],
            TensorShapeDynamism::STATIC,
        );
        assert_tensor_eq!(expected, actual);
    }

    #[test]
    fn tensor_factory_test_make_float_tensor() {
        let tf = TensorFactory::<f32>::new();
        let actual = tf.make_default(vec![2, 2], vec![1.1, 2.2, 3.3, 4.4]);
        let expected = tf.make(
            vec![2, 2],
            vec![1.1, 2.2, 3.3, 4.4],
            vec![2, 1],
            TensorShapeDynamism::STATIC,
        );
        assert_tensor_eq!(expected, actual);
    }

    #[test]
    fn tensor_factory_test_make_bool_tensor() {
        let tf = TensorFactory::<bool>::new();
        let actual = tf.make_default(vec![2, 2], vec![true, false, true, false]);
        let expected = tf.make(
            vec![2, 2],
            vec![true, false, true, false],
            vec![2, 1],
            TensorShapeDynamism::STATIC,
        );
        assert_tensor_eq!(expected, actual);
    }

    #[test]
    fn tensor_factory_test_data_is_copied() {
        let tf = TensorFactory::<i32>::new();
        let data = vec![1, 2, 3, 4];
        let t1 = tf.make_default(vec![2, 2], data.clone());
        let t2 = tf.make_default(vec![2, 2], data);
        assert_tensor_eq!(t1, t2);
        unsafe {
            *t1.mutable_data_ptr::<i32>() = 99;
        }
        crate::assert_tensor_ne!(t1, t2);
    }

    #[test]
    fn tensor_factory_test_default_strides_are_contiguous() {
        let tf = TensorFactory::<i32>::new();
        // 30 = 2 * 3 * 5.
        let t1 = tf.make_default(vec![2, 3, 5], vec![99; 30]);
        let strides = t1.strides();
        let actual_strides: Vec<i32> = (0..strides.size()).map(|i| *strides.at(i)).collect();
        let expected_strides = vec![15, 5, 1];
        assert_eq!(expected_strides, actual_strides);
    }

    #[test]
    fn tensor_factory_test_strides_for_empty_tensor() {
        let tf = TensorFactory::<i32>::new();
        let t1 = tf.make_default(vec![2, 0, 3, 0, 5], vec![]);
        let strides = t1.strides();
        let actual_strides: Vec<i32> = (0..strides.size()).map(|i| *strides.at(i)).collect();
        let expected_strides = vec![15, 15, 5, 5, 1];
        assert_eq!(expected_strides, actual_strides);
    }

    #[test]
    fn tensor_factory_test_strides_for_zero_dim_tensor() {
        let tf = TensorFactory::<i32>::new();
        let t1 = tf.make_default(vec![], vec![1]);
        let strides = t1.strides();
        let actual_strides: Vec<i32> = (0..strides.size()).map(|i| *strides.at(i)).collect();
        let expected_strides: Vec<i32> = vec![];
        assert_eq!(expected_strides, actual_strides);
    }

    #[test]
    #[should_panic]
    #[ignore]
    fn tensor_factory_test_not_enough_data_dies() {
        let tf = TensorFactory::<i32>::new();
        let _ = tf.make_default(vec![2, 2], vec![1, 2, 3]);
    }

    #[test]
    #[should_panic]
    #[ignore]
    fn tensor_factory_test_too_much_data_dies() {
        let tf = TensorFactory::<i32>::new();
        let _ = tf.make_default(vec![2, 2], vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn tensor_factory_test_make_strided_int_tensor() {
        let tf = TensorFactory::<i32>::new();
        let actual = tf.make(
            vec![2, 2],
            vec![1, 2, 3, 4],
            vec![1, 2],
            TensorShapeDynamism::STATIC,
        );
        let expected = tf.make(
            vec![2, 2],
            vec![1, 2, 3, 4],
            vec![1, 2],
            TensorShapeDynamism::STATIC,
        );
        assert_tensor_eq!(expected, actual);
    }

    #[test]
    fn tensor_factory_test_make_strided_float_tensor() {
        let tf = TensorFactory::<f32>::new();
        let actual = tf.make(
            vec![2, 2],
            vec![1.1, 2.2, 3.3, 4.4],
            vec![1, 2],
            TensorShapeDynamism::STATIC,
        );
        let expected = tf.make(
            vec![2, 2],
            vec![1.1, 2.2, 3.3, 4.4],
            vec![1, 2],
            TensorShapeDynamism::STATIC,
        );
        assert_tensor_eq!(expected, actual);
    }

    #[test]
    fn tensor_factory_test_make_strided_bool_tensor() {
        let tf = TensorFactory::<bool>::new();
        let actual = tf.make(
            vec![2, 2],
            vec![true, false, true, false],
            vec![1, 2],
            TensorShapeDynamism::STATIC,
        );
        let expected = tf.make(
            vec![2, 2],
            vec![true, false, true, false],
            vec![1, 2],
            TensorShapeDynamism::STATIC,
        );
        assert_tensor_eq!(expected, actual);
    }

    #[test]
    fn tensor_factory_test_make_strided_same_stride_tensor_supported() {
        let tf = TensorFactory::<bool>::new();
        let actual = tf.make(
            vec![2, 1, 0, 3, 5, 2, 1, 0, 0, 0, 0],
            vec![],
            vec![30, 10, 2, 10, 2, 1, 2, 10, 10, 10, 30],
            TensorShapeDynamism::STATIC,
        );
        let expected = tf.make(
            vec![2, 1, 0, 3, 5, 2, 1, 0, 0, 0, 0],
            vec![],
            vec![30, 10, 2, 10, 2, 1, 2, 10, 10, 10, 30],
            TensorShapeDynamism::STATIC,
        );
        assert_tensor_eq!(expected, actual);
    }

    #[test]
    fn tensor_factory_test_make_strided_data_is_copied() {
        let tf = TensorFactory::<i32>::new();
        let data = vec![1, 2, 3, 4];
        let strides = vec![1, 2];
        let t1 = tf.make(
            vec![2, 2],
            data.clone(),
            strides.clone(),
            TensorShapeDynamism::STATIC,
        );
        let t2 = tf.make(vec![2, 2], data, strides, TensorShapeDynamism::STATIC);
        assert_tensor_eq!(t1, t2);
        unsafe {
            *t1.mutable_data_ptr::<i32>() = 99;
        }
        crate::assert_tensor_ne!(t1, t2);
    }

    #[test]
    fn tensor_factory_test_make_strided_empty_data_supported() {
        let tf = TensorFactory::<i32>::new();
        let actual = tf.make(
            vec![2, 0, 3, 0, 5],
            vec![],
            vec![15, 15, 5, 5, 1],
            TensorShapeDynamism::STATIC,
        );
        let expected = tf.make(
            vec![2, 0, 3, 0, 5],
            vec![],
            vec![15, 15, 5, 5, 1],
            TensorShapeDynamism::STATIC,
        );
        assert_tensor_eq!(expected, actual);
    }

    #[test]
    fn tensor_factory_test_make_strided_zero_dim_supported() {
        let tf = TensorFactory::<i32>::new();
        let actual = tf.make(vec![], vec![1], vec![], TensorShapeDynamism::STATIC);
        let expected = tf.make(vec![], vec![1], vec![], TensorShapeDynamism::STATIC);
        assert_tensor_eq!(expected, actual);
    }

    #[test]
    #[should_panic]
    #[ignore]
    fn tensor_factory_test_make_strided_not_enough_data_die() {
        let tf = TensorFactory::<i32>::new();
        let _ = tf.make(
            vec![2, 2],
            vec![1, 2, 3],
            vec![1, 2],
            TensorShapeDynamism::STATIC,
        );
    }

    #[test]
    #[should_panic]
    #[ignore]
    fn tensor_factory_test_make_strided_too_much_data_die() {
        let tf = TensorFactory::<i32>::new();
        let _ = tf.make(
            vec![2, 2],
            vec![1, 2, 3, 4, 5],
            vec![1, 2],
            TensorShapeDynamism::STATIC,
        );
    }

    #[test]
    #[should_panic]
    #[ignore]
    fn tensor_factory_test_make_strided_not_enough_stride_die() {
        let tf = TensorFactory::<i32>::new();
        let _ = tf.make(
            vec![2, 2],
            vec![1, 2, 3, 4],
            vec![1],
            TensorShapeDynamism::STATIC,
        );
    }

    #[test]
    #[should_panic]
    #[ignore]
    fn tensor_factory_test_make_strided_too_much_stride_die() {
        let tf = TensorFactory::<i32>::new();
        let _ = tf.make(
            vec![2, 2],
            vec![1, 2, 3, 4],
            vec![1, 2, 1],
            TensorShapeDynamism::STATIC,
        );
    }

    #[test]
    #[should_panic]
    #[ignore]
    fn tensor_factory_test_make_strided_too_large_stride_die() {
        let tf = TensorFactory::<i32>::new();
        let _ = tf.make(
            vec![2, 2],
            vec![1, 2, 3, 4],
            vec![1, 4],
            TensorShapeDynamism::STATIC,
        );
    }

    #[test]
    #[should_panic]
    #[ignore]
    fn tensor_factory_test_make_strided_too_small_stride_die() {
        let tf = TensorFactory::<i32>::new();
        let _ = tf.make(
            vec![2, 2],
            vec![1, 2, 3, 4],
            vec![1, 1],
            TensorShapeDynamism::STATIC,
        );
    }

    #[test]
    #[should_panic]
    #[ignore]
    fn tensor_factory_test_make_strided_non_positive_stride_die() {
        let tf = TensorFactory::<i32>::new();
        let _ = tf.make(
            vec![2, 2],
            vec![1, 2, 3, 4],
            vec![2, -1],
            TensorShapeDynamism::STATIC,
        );
    }

    #[test]
    #[should_panic]
    #[ignore]
    fn tensor_factory_test_make_strided_wrong_stride_for_empty_data_die() {
        let tf = TensorFactory::<i32>::new();
        let _ = tf.make(
            vec![0, 2, 2],
            vec![],
            vec![0, 2, 1],
            TensorShapeDynamism::STATIC,
        );
    }

    #[test]
    #[should_panic]
    #[ignore]
    fn tensor_factory_test_make_strided_wrong_stride_for_zero_dim_data_die() {
        let tf = TensorFactory::<i32>::new();
        let _ = tf.make(vec![], vec![1], vec![0], TensorShapeDynamism::STATIC);
    }

    #[test]
    fn tensor_factory_test_full() {
        let tf = TensorFactory::<i32>::new();
        let actual = tf.full(vec![2, 2], 5, TensorShapeDynamism::STATIC);
        let expected = tf.make(
            vec![2, 2],
            vec![5, 5, 5, 5],
            vec![2, 1],
            TensorShapeDynamism::STATIC,
        );
        assert_tensor_eq!(expected, actual);
    }

    #[test]
    fn tensor_factory_test_full_float() {
        let tf = TensorFactory::<f32>::new();
        let actual = tf.full(vec![2, 2], 5.5, TensorShapeDynamism::STATIC);
        let expected = tf.make(
            vec![2, 2],
            vec![5.5, 5.5, 5.5, 5.5],
            vec![2, 1],
            TensorShapeDynamism::STATIC,
        );
        assert_tensor_eq!(expected, actual);
    }

    #[test]
    fn tensor_factory_test_zeros() {
        let tf = TensorFactory::<i32>::new();
        let actual = tf.zeros_default(vec![2, 2]);
        let expected = tf.make(
            vec![2, 2],
            vec![0, 0, 0, 0],
            vec![2, 1],
            TensorShapeDynamism::STATIC,
        );
        assert_tensor_eq!(expected, actual);
    }

    #[test]
    fn tensor_factory_test_ones() {
        let tf = TensorFactory::<i32>::new();
        let actual = tf.ones_default(vec![2, 2]);
        let expected = tf.make(
            vec![2, 2],
            vec![1, 1, 1, 1],
            vec![2, 1],
            TensorShapeDynamism::STATIC,
        );
        assert_tensor_eq!(expected, actual);
    }

    #[test]
    fn tensor_factory_test_zero_dimensional_tensor() {
        let tf = TensorFactory::<i32>::new();
        {
            let t = tf.make_default(vec![], vec![7]);
            assert_eq!(t.dim(), 0);
            assert_eq!(t.nbytes(), core::mem::size_of::<i32>());
            assert_eq!(t.numel(), 1);
            assert_eq!(unsafe { *t.const_data_ptr::<i32>() }, 7);
        }
        {
            let t = tf.zeros_default(vec![]);
            assert_eq!(t.dim(), 0);
            assert_eq!(t.nbytes(), core::mem::size_of::<i32>());
            assert_eq!(t.numel(), 1);
            assert_eq!(unsafe { *t.const_data_ptr::<i32>() }, 0);
        }
        {
            let t = tf.ones_default(vec![]);
            assert_eq!(t.dim(), 0);
            assert_eq!(t.nbytes(), core::mem::size_of::<i32>());
            assert_eq!(t.numel(), 1);
            assert_eq!(unsafe { *t.const_data_ptr::<i32>() }, 1);
        }
    }

    #[test]
    fn tensor_factory_test_empty_tensor() {
        let tf = TensorFactory::<i32>::new();
        {
            let t = tf.make_default(vec![0], vec![]);
            assert_eq!(t.dim(), 1);
            assert_eq!(t.nbytes(), 0);
            assert_eq!(t.numel(), 0);
        }
        {
            let t = tf.zeros_default(vec![0]);
            assert_eq!(t.dim(), 1);
            assert_eq!(t.nbytes(), 0);
            assert_eq!(t.numel(), 0);
        }
        {
            let t = tf.ones_default(vec![0]);
            assert_eq!(t.dim(), 1);
            assert_eq!(t.nbytes(), 0);
            assert_eq!(t.numel(), 0);
        }
    }

    // Mirrors the C++ `run_zeros_like_test`. The C++ builds `expected` from a raw
    // TensorImpl reusing the input's sizes/dim_order/strides; since these test
    // inputs are all contiguous (matching what `zeros_like` produces), an
    // equivalent contiguous `make_default` reproduces it. (Using
    // `make_with_dimorder` here would panic on the 0-dim input, a path the C++
    // test never exercises.)
    fn run_zeros_like_test(input: &Tensor) {
        let tf = TensorFactory::<i32>::new();
        let actual = tf.zeros_like(input, TensorShapeDynamism::STATIC);

        let expected_data: Vec<i32> = vec![0; input.numel() as usize];
        let input_sizes = input.sizes();
        let sizes: Vec<i32> = (0..input_sizes.size())
            .map(|i| *input_sizes.at(i))
            .collect();
        let expected = tf.make_default(sizes, expected_data);
        assert_tensor_eq!(expected, actual);
    }

    #[test]
    fn tensor_factory_test_zeros_like() {
        let tf = TensorFactory::<i32>::new();
        let input = tf.make_default(vec![3, 2, 1], vec![1, 2, 3, 4, 5, 6]);
        run_zeros_like_test(&input);
    }

    #[test]
    fn tensor_factory_test_zeros_like_zero_dimensional_tensor_supported() {
        let tf = TensorFactory::<i32>::new();
        let input = tf.make_default(vec![], vec![1]);
        run_zeros_like_test(&input);
    }

    #[test]
    fn tensor_factory_test_zeros_like_empty_tensor_supported() {
        let tf = TensorFactory::<i32>::new();
        let input = tf.make_default(vec![0], vec![]);
        run_zeros_like_test(&input);
    }

    // Mirrors the C++ `run_ones_like_test`; see `run_zeros_like_test` note.
    fn run_ones_like_test(input: &Tensor) {
        let tf = TensorFactory::<i32>::new();
        let actual = tf.ones_like(input, TensorShapeDynamism::STATIC);

        let expected_data: Vec<i32> = vec![1; input.numel() as usize];
        let input_sizes = input.sizes();
        let sizes: Vec<i32> = (0..input_sizes.size())
            .map(|i| *input_sizes.at(i))
            .collect();
        let expected = tf.make_default(sizes, expected_data);
        assert_tensor_eq!(expected, actual);
    }

    #[test]
    fn tensor_factory_test_ones_like() {
        let tf = TensorFactory::<i32>::new();
        let input = tf.make_default(vec![3, 2, 1], vec![1, 2, 3, 4, 5, 6]);
        run_ones_like_test(&input);
    }

    #[test]
    fn tensor_factory_test_ones_like_zero_dimensional_tensor_supported() {
        let tf = TensorFactory::<i32>::new();
        let input = tf.make_default(vec![], vec![2]);
        run_ones_like_test(&input);
    }

    #[test]
    fn tensor_factory_test_ones_like_empty_tensor_supported() {
        let tf = TensorFactory::<i32>::new();
        let input = tf.make_default(vec![0], vec![]);
        run_ones_like_test(&input);
    }

    // PORT-NOTE: `TensorListFactory` (and the `TensorList` type it produces) is
    // NOT ported in this port — only `TensorFactory` exists. The three
    // `TensorListFactoryTest` cases (`ZerosLike`, `ZerosLikeMixedDtypes`,
    // `ZerosLikeEmpty`) therefore have no Rust surface to bind to and cannot be
    // ported as written. They are recorded here; port them alongside a future
    // `TensorListFactory` port.

    #[test]
    fn tensor_factory_test_zeros_dynamism_parameter() {
        let tf = TensorFactory::<i32>::new();
        resize_tensor_to_assert_static(tf.zeros(vec![2, 2], TensorShapeDynamism::STATIC));
        resize_tensor_to_assert_dynamic_bound(
            tf.zeros(vec![2, 2], TensorShapeDynamism::DYNAMIC_BOUND),
        );
        resize_tensor_to_assert_dynamic_unbound(
            tf.zeros(vec![2, 2], TensorShapeDynamism::DYNAMIC_UNBOUND),
        );

        assert_tensor_eq!(
            tf.zeros(vec![2, 2], TensorShapeDynamism::STATIC),
            tf.zeros(vec![2, 2], TensorShapeDynamism::DYNAMIC_BOUND)
        );
        assert_tensor_eq!(
            tf.zeros(vec![2, 2], TensorShapeDynamism::STATIC),
            tf.zeros(vec![2, 2], TensorShapeDynamism::DYNAMIC_UNBOUND)
        );
    }

    #[test]
    fn tensor_factory_test_zeros_like_dynamism_parameter() {
        let tf = TensorFactory::<i32>::new();
        let zeros = tf.zeros_default(vec![2, 2]);
        resize_tensor_to_assert_static(tf.zeros_like(&zeros, TensorShapeDynamism::STATIC));
        resize_tensor_to_assert_dynamic_bound(
            tf.zeros_like(&zeros, TensorShapeDynamism::DYNAMIC_BOUND),
        );
        resize_tensor_to_assert_dynamic_unbound(
            tf.zeros_like(&zeros, TensorShapeDynamism::DYNAMIC_UNBOUND),
        );

        assert_tensor_eq!(
            tf.zeros_like(&zeros, TensorShapeDynamism::STATIC),
            tf.zeros_like(&zeros, TensorShapeDynamism::DYNAMIC_BOUND)
        );
        assert_tensor_eq!(
            tf.zeros_like(&zeros, TensorShapeDynamism::STATIC),
            tf.zeros_like(&zeros, TensorShapeDynamism::DYNAMIC_UNBOUND)
        );
    }

    #[test]
    fn tensor_factory_test_ones_dynamism_parameter() {
        let tf = TensorFactory::<i32>::new();
        resize_tensor_to_assert_static(tf.ones(vec![2, 2], TensorShapeDynamism::STATIC));
        resize_tensor_to_assert_dynamic_bound(
            tf.ones(vec![2, 2], TensorShapeDynamism::DYNAMIC_BOUND),
        );
        resize_tensor_to_assert_dynamic_unbound(
            tf.ones(vec![2, 2], TensorShapeDynamism::DYNAMIC_UNBOUND),
        );

        assert_tensor_eq!(
            tf.ones(vec![2, 2], TensorShapeDynamism::STATIC),
            tf.ones(vec![2, 2], TensorShapeDynamism::DYNAMIC_BOUND)
        );
        assert_tensor_eq!(
            tf.ones(vec![2, 2], TensorShapeDynamism::STATIC),
            tf.ones(vec![2, 2], TensorShapeDynamism::DYNAMIC_UNBOUND)
        );
    }

    #[test]
    fn tensor_factory_test_ones_like_dynamism_parameter() {
        let tf = TensorFactory::<i32>::new();
        let ones = tf.ones_default(vec![2, 2]);
        resize_tensor_to_assert_static(tf.ones_like(&ones, TensorShapeDynamism::STATIC));
        resize_tensor_to_assert_dynamic_bound(
            tf.ones_like(&ones, TensorShapeDynamism::DYNAMIC_BOUND),
        );
        resize_tensor_to_assert_dynamic_unbound(
            tf.ones_like(&ones, TensorShapeDynamism::DYNAMIC_UNBOUND),
        );

        assert_tensor_eq!(
            tf.ones_like(&ones, TensorShapeDynamism::STATIC),
            tf.ones_like(&ones, TensorShapeDynamism::DYNAMIC_BOUND)
        );
        assert_tensor_eq!(
            tf.ones_like(&ones, TensorShapeDynamism::STATIC),
            tf.ones_like(&ones, TensorShapeDynamism::DYNAMIC_UNBOUND)
        );
    }

    #[test]
    fn tensor_factory_test_full_dynamism_parameter() {
        let tf = TensorFactory::<i32>::new();
        resize_tensor_to_assert_static(tf.full(vec![2, 2], 1, TensorShapeDynamism::STATIC));
        resize_tensor_to_assert_dynamic_bound(tf.full(
            vec![2, 2],
            1,
            TensorShapeDynamism::DYNAMIC_BOUND,
        ));
        resize_tensor_to_assert_dynamic_unbound(tf.full(
            vec![2, 2],
            1,
            TensorShapeDynamism::DYNAMIC_UNBOUND,
        ));

        assert_tensor_eq!(
            tf.full(vec![2, 2], 1, TensorShapeDynamism::STATIC),
            tf.full(vec![2, 2], 1, TensorShapeDynamism::DYNAMIC_BOUND)
        );
        assert_tensor_eq!(
            tf.full(vec![2, 2], 1, TensorShapeDynamism::STATIC),
            tf.full(vec![2, 2], 1, TensorShapeDynamism::DYNAMIC_UNBOUND)
        );
    }

    #[test]
    fn tensor_factory_test_make_dynamism_parameter() {
        let tf = TensorFactory::<i32>::new();
        resize_tensor_to_assert_static(tf.make(
            vec![2, 2],
            vec![1, 2, 3, 4],
            vec![],
            TensorShapeDynamism::STATIC,
        ));
        resize_tensor_to_assert_dynamic_bound(tf.make(
            vec![2, 2],
            vec![1, 2, 3, 4],
            vec![],
            TensorShapeDynamism::DYNAMIC_BOUND,
        ));
        resize_tensor_to_assert_dynamic_unbound(tf.make(
            vec![2, 2],
            vec![1, 2, 3, 4],
            vec![],
            TensorShapeDynamism::DYNAMIC_UNBOUND,
        ));

        assert_tensor_eq!(
            tf.make(
                vec![2, 2],
                vec![1, 2, 3, 4],
                vec![],
                TensorShapeDynamism::STATIC
            ),
            tf.make(
                vec![2, 2],
                vec![1, 2, 3, 4],
                vec![],
                TensorShapeDynamism::DYNAMIC_BOUND
            )
        );
        assert_tensor_eq!(
            tf.make(
                vec![2, 2],
                vec![1, 2, 3, 4],
                vec![],
                TensorShapeDynamism::STATIC
            ),
            tf.make(
                vec![2, 2],
                vec![1, 2, 3, 4],
                vec![],
                TensorShapeDynamism::DYNAMIC_UNBOUND
            )
        );
    }

    // The following (`#if !defined(USE_ATEN_LIB)`) tests interleave live
    // `resize_tensor` checks with `ET_EXPECT_DEATH(resize(...))` death checks.
    // The death portions are split into `#[ignore]`d `#[should_panic]` tests.

    #[test]
    fn tensor_factory_test_full_dynamic() {
        setup();
        let tf = TensorFactory::<i32>::new();
        // ET_EXPECT_DEATH(resize(out {2,2} STATIC -> {1,1})) ->
        // tensor_factory_test_full_dynamic_death_static.
        let out = tf.full(vec![2, 2], 5, TensorShapeDynamism::DYNAMIC_BOUND);
        let new_sizes: [SizesType; 2] = [1, 2];
        assert_eq!(
            resize_tensor(&out, ArrayRef::from_raw_parts(new_sizes.as_ptr(), 2)),
            Error::Ok
        );
        // ET_EXPECT_DEATH(resize(out -> {3,3})) ->
        // tensor_factory_test_full_dynamic_death_grow.
    }

    #[test]
    #[should_panic]
    #[ignore]
    fn tensor_factory_test_full_dynamic_death_static() {
        let tf = TensorFactory::<i32>::new();
        let out = tf.full(vec![2, 2], 5, TensorShapeDynamism::STATIC);
        let new_sizes: [SizesType; 2] = [1, 1];
        #[allow(deprecated)]
        resize(&out, ArrayRef::from_raw_parts(new_sizes.as_ptr(), 2));
    }

    #[test]
    #[should_panic]
    #[ignore]
    fn tensor_factory_test_full_dynamic_death_grow() {
        let tf = TensorFactory::<i32>::new();
        let out = tf.full(vec![2, 2], 5, TensorShapeDynamism::DYNAMIC_BOUND);
        let new_sizes: [SizesType; 2] = [3, 3];
        #[allow(deprecated)]
        resize(&out, ArrayRef::from_raw_parts(new_sizes.as_ptr(), 2));
    }

    #[test]
    fn tensor_factory_test_make_int_tensor_dynamic() {
        setup();
        let tf = TensorFactory::<i32>::new();
        // ET_EXPECT_DEATH(resize(out {2,2} STATIC -> {1,1})) -> _death_static.
        let out = tf.make(
            vec![2, 2],
            vec![1, 2, 3, 4],
            vec![],
            TensorShapeDynamism::DYNAMIC_BOUND,
        );
        let new_sizes: [SizesType; 2] = [1, 2];
        assert_eq!(
            resize_tensor(&out, ArrayRef::from_raw_parts(new_sizes.as_ptr(), 2)),
            Error::Ok
        );
        // ET_EXPECT_DEATH(resize(out -> {3,3})) -> _death_grow.
    }

    #[test]
    #[should_panic]
    #[ignore]
    fn tensor_factory_test_make_int_tensor_dynamic_death_static() {
        let tf = TensorFactory::<i32>::new();
        let out = tf.make_default(vec![2, 2], vec![1, 2, 3, 4]);
        let new_sizes: [SizesType; 2] = [1, 1];
        #[allow(deprecated)]
        resize(&out, ArrayRef::from_raw_parts(new_sizes.as_ptr(), 2));
    }

    #[test]
    #[should_panic]
    #[ignore]
    fn tensor_factory_test_make_int_tensor_dynamic_death_grow() {
        let tf = TensorFactory::<i32>::new();
        let out = tf.make(
            vec![2, 2],
            vec![1, 2, 3, 4],
            vec![],
            TensorShapeDynamism::DYNAMIC_BOUND,
        );
        let new_sizes: [SizesType; 2] = [3, 3];
        #[allow(deprecated)]
        resize(&out, ArrayRef::from_raw_parts(new_sizes.as_ptr(), 2));
    }

    #[test]
    fn tensor_factory_test_make_zeros_dynamic() {
        setup();
        let tf = TensorFactory::<i32>::new();
        // ET_EXPECT_DEATH(resize(zeros {2,2} STATIC -> {1,1})) -> _death_static.
        let out = tf.zeros(vec![2, 2], TensorShapeDynamism::DYNAMIC_BOUND);
        let new_sizes: [SizesType; 2] = [1, 2];
        assert_eq!(
            resize_tensor(&out, ArrayRef::from_raw_parts(new_sizes.as_ptr(), 2)),
            Error::Ok
        );
        // ET_EXPECT_DEATH(resize(out -> {3,3})) -> _death_grow.

        // ET_EXPECT_DEATH(resize(zeros_like(out) STATIC -> {1,1})) ->
        // _death_like_static.
        let out2 = tf.zeros_like(&out, TensorShapeDynamism::DYNAMIC_BOUND);
        let new_sizes: [SizesType; 2] = [1, 2];
        assert_eq!(
            resize_tensor(&out2, ArrayRef::from_raw_parts(new_sizes.as_ptr(), 2)),
            Error::Ok
        );
        // ET_EXPECT_DEATH(resize(out2 -> {3,3})) -> _death_like_grow.
    }

    #[test]
    #[should_panic]
    #[ignore]
    fn tensor_factory_test_make_zeros_dynamic_death_static() {
        let tf = TensorFactory::<i32>::new();
        let out = tf.zeros_default(vec![2, 2]);
        let new_sizes: [SizesType; 2] = [1, 1];
        #[allow(deprecated)]
        resize(&out, ArrayRef::from_raw_parts(new_sizes.as_ptr(), 2));
    }

    #[test]
    #[should_panic]
    #[ignore]
    fn tensor_factory_test_make_zeros_dynamic_death_grow() {
        let tf = TensorFactory::<i32>::new();
        let out = tf.zeros(vec![2, 2], TensorShapeDynamism::DYNAMIC_BOUND);
        let new_sizes: [SizesType; 2] = [3, 3];
        #[allow(deprecated)]
        resize(&out, ArrayRef::from_raw_parts(new_sizes.as_ptr(), 2));
    }

    #[test]
    #[should_panic]
    #[ignore]
    fn tensor_factory_test_make_zeros_dynamic_death_like_static() {
        let tf = TensorFactory::<i32>::new();
        let out = tf.zeros(vec![2, 2], TensorShapeDynamism::DYNAMIC_BOUND);
        let out_like = tf.zeros_like(&out, TensorShapeDynamism::STATIC);
        let new_sizes: [SizesType; 2] = [1, 1];
        #[allow(deprecated)]
        resize(&out_like, ArrayRef::from_raw_parts(new_sizes.as_ptr(), 2));
    }

    #[test]
    #[should_panic]
    #[ignore]
    fn tensor_factory_test_make_zeros_dynamic_death_like_grow() {
        let tf = TensorFactory::<i32>::new();
        let out = tf.zeros(vec![2, 2], TensorShapeDynamism::DYNAMIC_BOUND);
        let out_like = tf.zeros_like(&out, TensorShapeDynamism::DYNAMIC_BOUND);
        let new_sizes: [SizesType; 2] = [3, 3];
        #[allow(deprecated)]
        resize(&out_like, ArrayRef::from_raw_parts(new_sizes.as_ptr(), 2));
    }

    fn check_array_ref_equal(a1: &[DimOrderType], a2: ArrayRef<DimOrderType>) {
        assert_eq!(a1.len(), a2.size());
        for i in 0..a1.len() {
            assert_eq!(a1[i], *a2.at(i));
        }
    }

    #[test]
    fn tensor_factory_test_dim_order_to_stride_test() {
        let tf = TensorFactory::<i32>::new();
        let out = tf.zeros_default(vec![2, 2]);
        let dim_order: Vec<DimOrderType> = vec![0, 1];
        check_array_ref_equal(&dim_order, out.dim_order());

        let out = tf.zeros_default(vec![1, 2, 5]);
        let dim_order: Vec<DimOrderType> = vec![0, 1, 2];
        check_array_ref_equal(&dim_order, out.dim_order());

        let data: Vec<i32> = vec![0; 10];
        let strided_out = tf.make(
            vec![1, 2, 5],
            data,
            vec![10, 1, 2],
            TensorShapeDynamism::STATIC,
        );
        let dim_order: Vec<DimOrderType> = vec![0, 2, 1];
        check_array_ref_equal(&dim_order, strided_out.dim_order());

        let data: Vec<i32> = vec![0; 12];
        let strided_out = tf.make(
            vec![3, 2, 2],
            data,
            vec![1, 6, 3],
            TensorShapeDynamism::STATIC,
        );
        let dim_order: Vec<DimOrderType> = vec![1, 2, 0];
        check_array_ref_equal(&dim_order, strided_out.dim_order());
    }

    #[test]
    fn tensor_factory_test_ambgiuous_dim_order_to_stride_test() {
        let tf = TensorFactory::<i32>::new();
        let data: Vec<i32> = vec![0; 10];
        let strided_out = tf.make(
            vec![1, 2, 5],
            data.clone(),
            vec![1, 1, 2],
            TensorShapeDynamism::STATIC,
        );
        // Strides {1,1,2} could be dim_order {2,1,0}, but stable_sort in
        // dim_order_from_stride preserves order, giving {2,0,1}.
        let dim_order: Vec<DimOrderType> = vec![2, 0, 1];
        check_array_ref_equal(&dim_order, strided_out.dim_order());

        let strided_out = tf.make(
            vec![1, 2, 5],
            data,
            vec![1, 1, 2],
            TensorShapeDynamism::STATIC,
        );
        let dim_order: Vec<DimOrderType> = vec![2, 0, 1];
        check_array_ref_equal(&dim_order, strided_out.dim_order());
    }
}
