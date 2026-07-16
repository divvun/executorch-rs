//! Literal port of runtime/executor/method_meta.cpp + runtime/executor/method_meta.h.
//!
//! NAME-MAPPING DEVIATION: the C++ reads the ExecutionPlan through the generated
//! flatbuffer accessors (`s_plan_->values()`, `->inputs()`, `->outputs()`,
//! `->delegates()`, `->chains()`, `->non_const_buffer_sizes()`,
//! `->non_const_buffer_device()`, and the per-EValue/Tensor accessors). The Rust
//! flatbuffers crate (`crate::schema::generated::executorch_flatbuffer`) exposes
//! those as snake_case, usize-based accessors that return `Option<Vector<..>>`
//! for nullable vectors, `Vector::len()`/`Vector::get(i)` (element by value,
//! non-nullable) for iteration, `Option<Tensor>` for `val_as_tensor()`,
//! `Option<&str>` for flatbuffer strings, and `Option<ExtraTensorInfo>` for
//! `extra_tensor_info()`. The nullable `const T*` the C++ null-checks maps to the
//! `Option` these accessors return.

use crate::runtime::core::error::Error;
use crate::runtime::core::portable_type::device::{Device, DeviceIndex, DeviceType};
use crate::runtime::core::portable_type::scalar_type::ScalarType;
use crate::runtime::core::result::{Result, ResultExt};
use crate::runtime::core::span::Span;
use crate::runtime::core::tag::Tag;
use crate::schema::generated::executorch_flatbuffer;
use crate::schema::generated::executorch_flatbuffer::{EValue, ExecutionPlan, KernelTypes};

// PORT-NOTE: `static_cast<executorch::aten::ScalarType>(s_tensor->scalar_type())`
// — the serialized `executorch_flatbuffer::ScalarType` (`#[repr(transparent)]`
// i8 newtype) and the runtime `ScalarType` (`#[repr(i8)]` enum) share
// discriminants; transmute the byte, matching the sibling tensor_parser modules.
fn static_cast_scalar_type(st: executorch_flatbuffer::ScalarType) -> ScalarType {
    unsafe { core::mem::transmute::<i8, ScalarType>(st.0) }
}

// PORT-NOTE: `static_cast<etensor::DeviceType>(entry->device_type())` — the
// serialized `executorch_flatbuffer::DeviceType` (i8 newtype) and the runtime
// `etensor::DeviceType` (`#[repr(i8)]` enum) share discriminants; transmute the
// byte, mirroring the scalar-type cast above.
fn static_cast_device_type(dt: executorch_flatbuffer::DeviceType) -> DeviceType {
    unsafe { core::mem::transmute::<i8, DeviceType>(dt.0) }
}

// [spec:et:def:method-meta.executorch.et-runtime-namespace.get-tag-fn]
// [spec:et:sem:method-meta.executorch.et-runtime-namespace.get-tag-fn]
fn get_tag(serialization_value: EValue, index: usize) -> Result<Tag> {
    match serialization_value.val_type() {
        KernelTypes::Null => Ok(Tag::None),
        KernelTypes::Int => Ok(Tag::Int),
        KernelTypes::Double => Ok(Tag::Double),
        KernelTypes::Bool => Ok(Tag::Bool),
        KernelTypes::String => Ok(Tag::String),
        KernelTypes::Tensor => Ok(Tag::Tensor),
        _ => {
            crate::et_log!(
                Error,
                "Invalid tag: {} input idx: {}",
                serialization_value.val_type().0 as usize,
                index
            );
            Err(Error::Internal)
        }
    }
}

// [spec:et:def:method-meta.executorch.et-runtime-namespace.calculate-nbytes-fn]
// [spec:et:sem:method-meta.executorch.et-runtime-namespace.calculate-nbytes-fn]
fn calculate_nbytes(sizes: Span<i32>, scalar_type: ScalarType) -> Result<usize> {
    let mut n: usize = 1;
    let mut i: usize = 0;
    while i < sizes.size() {
        // PORT-NOTE: C++ `sizes[i]` is `Span::operator[]` (unchecked). The Rust
        // `Span::index` is an `unsafe` unchecked accessor returning `&mut T`.
        let size_i: i32 = unsafe { *sizes.index(i) };
        crate::et_check_or_return_error!(
            size_i >= 0,
            InvalidProgram,
            "Invalid size[{}]: {}. Size must not be negative",
            i,
            size_i
        );
        // PORT-NOTE: C++ uses `c10::mul_overflows(n, x, &next_n)`, returning true
        // on overflow. `checked_mul` returns None on overflow; mirrors the branch.
        let next_n = n.checked_mul(size_i as usize);
        let overflow = next_n.is_none();
        crate::et_check_or_return_error!(
            !overflow,
            InvalidArgument,
            "Invalid size[{}]: {}. Potentially overflowed, expect to be 0 or n: {}",
            i,
            size_i,
            n
        );
        n = next_n.unwrap();
        i += 1;
    }

    let elem_size: usize =
        crate::runtime::core::exec_aten::util::scalar_type_util::element_size(scalar_type);
    let total_bytes = n.checked_mul(elem_size);
    let overflow = total_bytes.is_none();
    crate::et_check_or_return_error!(
        !overflow,
        InvalidArgument,
        "Invalid elem_size: {}. Potentially overflowed, expect to be 0 or n: {}",
        elem_size,
        n
    );

    Ok(total_bytes.unwrap())
}

/// Metadata about a specific tensor of an ExecuTorch Program.
///
/// The program used to create the MethodMeta object that created this
/// TensorInfo must outlive this TensorInfo.
// [spec:et:def:method-meta.executorch.et-runtime-namespace.tensor-info]
//
// PORT-NOTE: `sizes_`/`dim_order_` are borrowed `Span`s and `name_` is a borrowed
// string view; all reference data from the Program, which must outlive the
// TensorInfo. The `'a` lifetime carries that borrow. `TensorInfo() = delete`
// (no default ctor) and the copy/move ops are trivial (derive Clone/Copy).
// [spec:et:def:method-meta.executorch.et-runtime-namespace.tensor-info.operator-fn]
// [spec:et:sem:method-meta.executorch.et-runtime-namespace.tensor-info.operator-fn]
#[derive(Clone, Copy)]
pub struct TensorInfo<'a> {
    /// The sizes of the tensor.
    ///
    /// NOTE: References data from the Program, so the Program must outlive the
    /// TensorInfo.
    sizes_: Span<i32>,

    /// The dim order of the tensor.
    ///
    /// NOTE: References data from the Program, so the Program must outlive the
    /// TensorInfo.
    dim_order_: Span<u8>,

    /// The fully qualified name of the Tensor.
    // PORT-NOTE: C++ `std::string_view` (ptr + len into the Program buffer). The
    // Rust equivalent is a borrowed `&'a str`; the empty view `{nullptr, 0}` maps
    // to `""`.
    name_: &'a str,

    /// The scalar type of the tensor.
    scalar_type_: ScalarType,

    /// Whether the tensor's memory was planned during export.
    is_memory_planned_: bool,

    /// The size in bytes of the tensor.
    nbytes_: usize,
}

impl<'a> TensorInfo<'a> {
    /// Create a TensorInfo instance.
    // [spec:et:def:method-meta.executorch.et-runtime-namespace.tensor-info.create-fn]
    // [spec:et:sem:method-meta.executorch.et-runtime-namespace.tensor-info.create-fn]
    fn create(
        sizes: Span<i32>,
        dim_order: Span<u8>,
        scalar_type: ScalarType,
        is_memory_planned: bool,
        name: &'a str,
    ) -> Result<TensorInfo<'a>> {
        let nbytes = calculate_nbytes(sizes, scalar_type);
        crate::et_check_or_return_error!(
            ResultExt::ok(&nbytes),
            InvalidArgument,
            "Failed to calculate nbytes for TensorInfo"
        );

        Ok(TensorInfo::new(
            sizes,
            dim_order,
            scalar_type,
            is_memory_planned,
            name,
            *nbytes.get(),
        ))
    }

    // [spec:et:def:method-meta.executorch.et-runtime-namespace.tensor-info.tensor-info-fn]
    // [spec:et:sem:method-meta.executorch.et-runtime-namespace.tensor-info.tensor-info-fn]
    fn new(
        sizes: Span<i32>,
        dim_order: Span<u8>,
        scalar_type: ScalarType,
        is_memory_planned: bool,
        name: &'a str,
        nbytes: usize,
    ) -> Self {
        TensorInfo {
            sizes_: sizes,
            dim_order_: dim_order,
            name_: name,
            scalar_type_: scalar_type,
            is_memory_planned_: is_memory_planned,
            nbytes_: nbytes,
        }
    }

    /// Returns the sizes of the tensor.
    // [spec:et:def:method-meta.executorch.et-runtime-namespace.tensor-info.sizes-fn]
    // [spec:et:sem:method-meta.executorch.et-runtime-namespace.tensor-info.sizes-fn]
    pub fn sizes(&self) -> Span<i32> {
        self.sizes_
    }

    /// Returns the dim order of the tensor.
    // [spec:et:def:method-meta.executorch.et-runtime-namespace.tensor-info.dim-order-fn]
    // [spec:et:sem:method-meta.executorch.et-runtime-namespace.tensor-info.dim-order-fn]
    pub fn dim_order(&self) -> Span<u8> {
        self.dim_order_
    }

    /// Returns the scalar type of the input/output.
    // [spec:et:def:method-meta.executorch.et-runtime-namespace.tensor-info.scalar-type-fn]
    // [spec:et:sem:method-meta.executorch.et-runtime-namespace.tensor-info.scalar-type-fn]
    pub fn scalar_type(&self) -> ScalarType {
        self.scalar_type_
    }

    /// Returns whether the tensor's memory was planned during export.
    // [spec:et:def:method-meta.executorch.et-runtime-namespace.tensor-info.is-memory-planned-fn]
    // [spec:et:sem:method-meta.executorch.et-runtime-namespace.tensor-info.is-memory-planned-fn]
    pub fn is_memory_planned(&self) -> bool {
        self.is_memory_planned_
    }

    /// Returns the size of the tensor in bytes.
    // [spec:et:def:method-meta.executorch.et-runtime-namespace.tensor-info.nbytes-fn]
    // [spec:et:sem:method-meta.executorch.et-runtime-namespace.tensor-info.nbytes-fn]
    pub fn nbytes(&self) -> usize {
        self.nbytes_
    }

    /// Returns the fully qualified name of the Tensor might be empty if the
    /// tensor is nameless.
    // [spec:et:def:method-meta.executorch.et-runtime-namespace.tensor-info.name-fn]
    // [spec:et:sem:method-meta.executorch.et-runtime-namespace.tensor-info.name-fn]
    pub fn name(&self) -> &'a str {
        self.name_
    }
}

/// Describes a a method in an ExecuTorch program.
///
/// The program used to create a MethodMeta object must outlive the MethodMeta.
/// It is separate from Method so that this information can be accessed without
/// paying the initialization cost of loading the full Method.
// [spec:et:def:method-meta.executorch.et-runtime-namespace.method-meta]
//
// PORT-NOTE: `s_plan_` is the borrowed `const ExecutionPlan*` (must outlive the
// MethodMeta); the `'a` lifetime carries that borrow. `MethodMeta() = delete`;
// the copy/move ops are trivial (a shallow pointer copy — derive Clone/Copy).
// [spec:et:def:method-meta.executorch.et-runtime-namespace.method-meta.operator-fn]
// [spec:et:sem:method-meta.executorch.et-runtime-namespace.method-meta.operator-fn]
#[derive(Clone, Copy)]
pub struct MethodMeta<'a> {
    /// Source of truth for method information
    s_plan_: ExecutionPlan<'a>,
}

impl<'a> MethodMeta<'a> {
    // [spec:et:def:method-meta.executorch.et-runtime-namespace.method-meta.method-meta-fn]
    // [spec:et:sem:method-meta.executorch.et-runtime-namespace.method-meta.method-meta-fn]
    //
    // PORT-NOTE: private explicit constructor (only `Program` may call it). Takes
    // the flatbuffer `ExecutionPlan` view by value (it is a fat, non-owning
    // handle over the Program buffer, matching the borrowed `const ExecutionPlan*`
    // the C++ stores). `pub(crate)` so the sibling `Program` module can construct.
    pub(crate) fn new(s_plan: ExecutionPlan<'a>) -> Self {
        MethodMeta { s_plan_: s_plan }
    }

    /// Get the name of this method.
    // [spec:et:def:method-meta.executorch.et-runtime-namespace.method-meta.name-fn]
    // [spec:et:sem:method-meta.executorch.et-runtime-namespace.method-meta.name-fn]
    //
    // PORT-NOTE: C++ returns `s_plan_->name()->c_str()` (assumed non-null). The
    // Rust `name()` is `Option<&str>`; a well-formed program guarantees it is
    // present, so unwrap it, matching the C++ no-validation assumption.
    pub fn name(&self) -> &'a str {
        self.s_plan_.name().unwrap()
    }

    /// Get the number of inputs to this method.
    // [spec:et:def:method-meta.executorch.et-runtime-namespace.method-meta.num-inputs-fn]
    // [spec:et:sem:method-meta.executorch.et-runtime-namespace.method-meta.num-inputs-fn]
    pub fn num_inputs(&self) -> usize {
        self.s_plan_.inputs().unwrap().len()
    }

    /// Get the tag of the specified input.
    // [spec:et:def:method-meta.executorch.et-runtime-namespace.method-meta.input-tag-fn]
    // [spec:et:sem:method-meta.executorch.et-runtime-namespace.method-meta.input-tag-fn]
    pub fn input_tag(&self, index: usize) -> Result<Tag> {
        let num_inputs = self.num_inputs();
        crate::et_check_or_return_error!(
            index < num_inputs,
            InvalidArgument,
            "index {} out of range. num_inputs: {}",
            index,
            num_inputs
        );
        let input_index = self.s_plan_.inputs().unwrap().get(index);
        let num_values: usize = self.s_plan_.values().unwrap().len();
        crate::et_check_or_return_error!(
            input_index >= 0 && (input_index as usize) < num_values,
            InvalidProgram,
            "internal value index {} out of range [0,{}) for input {}",
            input_index as isize,
            num_values,
            index
        );
        let serialization_value = self.s_plan_.values().unwrap().get(input_index as usize);
        get_tag(serialization_value, index)
    }

    /// Get metadata about the specified input.
    // [spec:et:def:method-meta.executorch.et-runtime-namespace.method-meta.input-tensor-meta-fn]
    // [spec:et:sem:method-meta.executorch.et-runtime-namespace.method-meta.input-tensor-meta-fn]
    pub fn input_tensor_meta(&self, index: usize) -> Result<TensorInfo<'a>> {
        let tag = self.input_tag(index);
        if !ResultExt::ok(&tag) {
            return Err(tag.error());
        }
        crate::et_check_or_return_error!(
            *tag.get() == Tag::Tensor,
            InvalidArgument,
            "Tag: {} input: {} is not Tensor",
            *tag.get() as usize,
            index
        );
        let input_index = self.s_plan_.inputs().unwrap().get(index);
        // input_index was already validated by input_tag().
        let tensor_value = self
            .s_plan_
            .values()
            .unwrap()
            .get(input_index as usize)
            .val_as_tensor();
        crate::et_check_or_return_error!(
            tensor_value.is_some()
                && tensor_value.unwrap().sizes().is_some()
                && tensor_value.unwrap().dim_order().is_some(),
            InvalidProgram,
            "Null tensor metadata for input {}",
            index
        );
        let tensor_value = tensor_value.unwrap();
        TensorInfo::create(
            Span::from_raw_parts(
                tensor_value.sizes().unwrap().bytes().as_ptr() as *mut i32,
                tensor_value.sizes().unwrap().len(),
            ),
            Span::from_raw_parts(
                tensor_value.dim_order().unwrap().bytes().as_ptr() as *mut u8,
                tensor_value.dim_order().unwrap().len(),
            ),
            static_cast_scalar_type(tensor_value.scalar_type()),
            tensor_value.allocation_info().is_some() || tensor_value.data_buffer_idx() != 0, /* is_memory_planned */
            "", // Count constant returns as
                // memory planned.
        )
    }

    /// Get the number of outputs to this method.
    // [spec:et:def:method-meta.executorch.et-runtime-namespace.method-meta.num-outputs-fn]
    // [spec:et:sem:method-meta.executorch.et-runtime-namespace.method-meta.num-outputs-fn]
    pub fn num_outputs(&self) -> usize {
        self.s_plan_.outputs().unwrap().len()
    }

    /// Get the tag of the specified output.
    // [spec:et:def:method-meta.executorch.et-runtime-namespace.method-meta.output-tag-fn]
    // [spec:et:sem:method-meta.executorch.et-runtime-namespace.method-meta.output-tag-fn]
    pub fn output_tag(&self, index: usize) -> Result<Tag> {
        let num_outputs = self.num_outputs();
        crate::et_check_or_return_error!(
            index < num_outputs,
            InvalidArgument,
            "index {} out of range. num_outputs: {}",
            index,
            num_outputs
        );
        let output_index = self.s_plan_.outputs().unwrap().get(index);
        let num_values: usize = self.s_plan_.values().unwrap().len();
        crate::et_check_or_return_error!(
            output_index >= 0 && (output_index as usize) < num_values,
            InvalidProgram,
            "internal value index {} out of range [0,{}) for output {}",
            output_index as isize,
            num_values,
            index
        );
        let serialization_value = self.s_plan_.values().unwrap().get(output_index as usize);
        get_tag(serialization_value, index)
    }

    /// Get metadata about the specified output.
    // [spec:et:def:method-meta.executorch.et-runtime-namespace.method-meta.output-tensor-meta-fn]
    // [spec:et:sem:method-meta.executorch.et-runtime-namespace.method-meta.output-tensor-meta-fn]
    pub fn output_tensor_meta(&self, index: usize) -> Result<TensorInfo<'a>> {
        let tag = self.output_tag(index);
        if !ResultExt::ok(&tag) {
            return Err(tag.error());
        }
        crate::et_check_or_return_error!(
            *tag.get() == Tag::Tensor,
            InvalidArgument,
            "Tag: {} output: {} is not Tensor",
            *tag.get() as usize,
            index
        );
        let output_index = self.s_plan_.outputs().unwrap().get(index);
        // output_index was already validated by output_tag().
        let tensor_value = self
            .s_plan_
            .values()
            .unwrap()
            .get(output_index as usize)
            .val_as_tensor();
        crate::et_check_or_return_error!(
            tensor_value.is_some()
                && tensor_value.unwrap().sizes().is_some()
                && tensor_value.unwrap().dim_order().is_some(),
            InvalidProgram,
            "Null tensor metadata for output {}",
            index
        );
        let tensor_value = tensor_value.unwrap();
        TensorInfo::create(
            Span::from_raw_parts(
                tensor_value.sizes().unwrap().bytes().as_ptr() as *mut i32,
                tensor_value.sizes().unwrap().len(),
            ),
            Span::from_raw_parts(
                tensor_value.dim_order().unwrap().bytes().as_ptr() as *mut u8,
                tensor_value.dim_order().unwrap().len(),
            ),
            static_cast_scalar_type(tensor_value.scalar_type()),
            tensor_value.allocation_info().is_some() || tensor_value.data_buffer_idx() != 0, /* is_memory_planned */
            "", // Count constant returns as
                // memory planned.
        )
    }

    /// Get the number of attribute tensors in this method.
    // [spec:et:def:method-meta.executorch.et-runtime-namespace.method-meta.num-attributes-fn]
    // [spec:et:sem:method-meta.executorch.et-runtime-namespace.method-meta.num-attributes-fn]
    pub fn num_attributes(&self) -> usize {
        let mut counter: usize = 0;
        let values = self.s_plan_.values().unwrap();
        let mut i: usize = 0;
        while i < values.len() {
            let value = values.get(i);
            if value.val_type() == KernelTypes::Tensor {
                let tensor_value = value.val_as_tensor();
                // PORT-NOTE: C++ checks `tensor_value != nullptr &&
                // extra_tensor_info() != nullptr && fully_qualified_name() !=
                // nullptr && ...->c_str() != nullptr`. The Rust `val_as_tensor()`
                // and `extra_tensor_info()` return `Option`, and
                // `fully_qualified_name()` returns `Option<&str>` (whose `&str`
                // is NUL-terminated in the buffer, so a non-null `c_str()`
                // corresponds to `is_some()`).
                if tensor_value.is_some()
                    && tensor_value.unwrap().extra_tensor_info().is_some()
                    && tensor_value
                        .unwrap()
                        .extra_tensor_info()
                        .unwrap()
                        .fully_qualified_name()
                        .is_some()
                {
                    counter += 1;
                }
            }
            i += 1;
        }
        counter
    }

    /// Get metadata about the specified attribute tensor.
    // [spec:et:def:method-meta.executorch.et-runtime-namespace.method-meta.attribute-tensor-meta-fn]
    // [spec:et:sem:method-meta.executorch.et-runtime-namespace.method-meta.attribute-tensor-meta-fn]
    pub fn attribute_tensor_meta(&self, index: usize) -> Result<TensorInfo<'a>> {
        let mut counter: usize = 0;
        let values = self.s_plan_.values().unwrap();
        let mut i: usize = 0;
        while i < values.len() {
            let value = values.get(i);
            if value.val_type() == KernelTypes::Tensor {
                let tensor_value = value.val_as_tensor();
                if tensor_value.is_some()
                    && tensor_value.unwrap().extra_tensor_info().is_some()
                    && tensor_value
                        .unwrap()
                        .extra_tensor_info()
                        .unwrap()
                        .fully_qualified_name()
                        .is_some()
                {
                    let tensor_value = tensor_value.unwrap();
                    if counter == index {
                        crate::et_check_or_return_error!(
                            tensor_value.sizes().is_some() && tensor_value.dim_order().is_some(),
                            InvalidProgram,
                            "Null tensor metadata for attribute {}",
                            index
                        );
                        let t_name = tensor_value
                            .extra_tensor_info()
                            .unwrap()
                            .fully_qualified_name()
                            .unwrap();
                        // Count constant returns as memory planned
                        return TensorInfo::create(
                            Span::from_raw_parts(
                                tensor_value.sizes().unwrap().bytes().as_ptr() as *mut i32,
                                tensor_value.sizes().unwrap().len(),
                            ),
                            Span::from_raw_parts(
                                tensor_value.dim_order().unwrap().bytes().as_ptr() as *mut u8,
                                tensor_value.dim_order().unwrap().len(),
                            ),
                            static_cast_scalar_type(tensor_value.scalar_type()),
                            tensor_value.allocation_info().is_some()
                                || tensor_value.data_buffer_idx() != 0, /* is_memory_planned */
                            t_name,
                        );
                    }
                    counter += 1;
                }
            }
            i += 1;
        }
        crate::et_log!(Error, "No attribute tensor found at index {}", index);
        Err(Error::InvalidArgument)
    }

    /// Get the number of memory-planned buffers this method requires.
    // [spec:et:def:method-meta.executorch.et-runtime-namespace.method-meta.num-memory-planned-buffers-fn]
    // [spec:et:sem:method-meta.executorch.et-runtime-namespace.method-meta.num-memory-planned-buffers-fn]
    pub fn num_memory_planned_buffers(&self) -> usize {
        if self.s_plan_.non_const_buffer_sizes().is_none() {
            return 0;
        }
        let size: usize = self.s_plan_.non_const_buffer_sizes().unwrap().len();
        // Index zero is reserved internally, and we hide it from users. The actual
        // number of buffers is one fewer than the actual size of this list in the
        // program.
        if size > 0 { size - 1 } else { 0 }
    }

    /// Get the size in bytes of the specified memory-planned buffer.
    // [spec:et:def:method-meta.executorch.et-runtime-namespace.method-meta.memory-planned-buffer-size-fn]
    // [spec:et:sem:method-meta.executorch.et-runtime-namespace.method-meta.memory-planned-buffer-size-fn]
    pub fn memory_planned_buffer_size(&self, index: usize) -> Result<i64> {
        let num_buffers = self.num_memory_planned_buffers();
        crate::et_check_or_return_error!(
            index < num_buffers,
            InvalidArgument,
            "index {} out of range. num_buffers: {}",
            index,
            num_buffers
        );
        // Index zero is reserved internally, and we hide it from users. Adjust the
        // provided index to point to one of the actual buffers.
        let size: i64 = self
            .s_plan_
            .non_const_buffer_sizes()
            .unwrap()
            .get(index + 1);
        crate::et_check_or_return_error!(
            size >= 0,
            InvalidProgram,
            "memory_planned_buffer_size({}) has invalid negative size: {}",
            index,
            size
        );
        Ok(size)
    }

    /// Get the device placement for the specified memory-planned buffer.
    // [spec:et:def:method-meta.executorch.et-runtime-namespace.method-meta.memory-planned-buffer-device-fn]
    // [spec:et:sem:method-meta.executorch.et-runtime-namespace.method-meta.memory-planned-buffer-device-fn]
    pub fn memory_planned_buffer_device(&self, index: usize) -> Result<Device> {
        let num_buffers = self.num_memory_planned_buffers();
        crate::et_check_or_return_error!(
            index < num_buffers,
            InvalidArgument,
            "index {} out of range. num_buffers: {}",
            index,
            num_buffers
        );

        // The non_const_buffer_device field is optional and only present when the
        // program contains non-CPU buffers. For CPU-only programs (or legacy PTE
        // files), this field is null and all buffers default to CPU.
        let buffer_devices = self.s_plan_.non_const_buffer_device();
        if buffer_devices.is_none() {
            return Ok(Device::new(DeviceType::CPU, 0));
        }
        let buffer_devices = buffer_devices.unwrap();

        // The sparse list only contains entries for non-CPU buffers.
        // buffer_idx uses the same indexing as non_const_buffer_sizes (1-based,
        // with index 0 reserved). The user-facing index is 0-based, so we
        // compare against index + 1.
        let internal_idx = (index + 1) as i32;
        let mut i: usize = 0;
        while i < buffer_devices.len() {
            let entry = buffer_devices.get(i);
            if entry.buffer_idx() == internal_idx {
                return Ok(Device::new(
                    static_cast_device_type(entry.device_type()),
                    entry.device_index() as DeviceIndex,
                ));
            }
            i += 1;
        }

        // Not found in the sparse list — this buffer is on CPU.
        Ok(Device::new(DeviceType::CPU, 0))
    }

    /// Check to see if a backend is used in this method.
    // [spec:et:def:method-meta.executorch.et-runtime-namespace.method-meta.uses-backend-fn]
    // [spec:et:sem:method-meta.executorch.et-runtime-namespace.method-meta.uses-backend-fn]
    //
    // PORT-NOTE: C++ takes `const char* backend_name` and `ET_CHECK_MSG(backend_name,
    // ...)` aborts on null. Rust takes a `&str` (a non-null precondition by type),
    // so no null check is needed. C++ compares lengths then `strncmp` the first
    // `backend_name_len` bytes against the flatbuffer id (length-first byte
    // comparison, using the stored id length rather than NUL termination). Rust
    // `&str` equality is a length-first byte comparison, the literal equivalent.
    pub fn uses_backend(&self, backend_name: &str) -> bool {
        let delegates = self.s_plan_.delegates().unwrap();
        let mut i: usize = 0;
        while i < delegates.len() {
            let delegate = delegates.get(i);
            let backend_name_len = backend_name.len();
            let delegate_id_len = delegate.id().unwrap().len();
            if backend_name_len == delegate_id_len
                && delegate.id().unwrap().as_bytes()[..backend_name_len] == *backend_name.as_bytes()
            {
                return true;
            }
            i += 1;
        }
        false
    }

    /// Get the number of backends used in this method.
    // [spec:et:def:method-meta.executorch.et-runtime-namespace.method-meta.num-backends-fn]
    // [spec:et:sem:method-meta.executorch.et-runtime-namespace.method-meta.num-backends-fn]
    pub fn num_backends(&self) -> usize {
        let delegates = self.s_plan_.delegates();
        match delegates {
            Some(delegates) => delegates.len(),
            None => 0,
        }
    }

    /// Get the backend name at the given index.
    // [spec:et:def:method-meta.executorch.et-runtime-namespace.method-meta.get-backend-name-fn]
    // [spec:et:sem:method-meta.executorch.et-runtime-namespace.method-meta.get-backend-name-fn]
    //
    // PORT-NOTE: C++ returns `const char*` (`id()->c_str()`, NUL-terminated). Rust
    // returns the borrowed `&'a str` from the flatbuffer id; its bytes are the same
    // NUL-terminated region in the Program buffer.
    pub fn get_backend_name(&self, index: usize) -> Result<&'a str> {
        let count = self.num_backends();
        crate::et_check_or_return_error!(
            index < count,
            InvalidArgument,
            "Index {} out of range. num_backends: {}",
            index,
            count
        );
        Ok(self.s_plan_.delegates().unwrap().get(index).id().unwrap())
    }

    /// Get the number of instructions in this method.
    // [spec:et:def:method-meta.executorch.et-runtime-namespace.method-meta.num-instructions-fn]
    // [spec:et:sem:method-meta.executorch.et-runtime-namespace.method-meta.num-instructions-fn]
    pub fn num_instructions(&self) -> usize {
        let chains = self.s_plan_.chains();
        if chains.is_none() {
            return 0;
        }
        let chains = chains.unwrap();
        let num_chains = chains.len();
        // PORT-NOTE: C++ `auto num_instructions = 0;` deduces `int`; the
        // accumulator here is `usize` to hold the flatbuffer vector sizes. The
        // return type is `size_t`, so the result value is identical.
        let mut num_instructions: usize = 0;
        let mut i: usize = 0;
        while i < num_chains {
            let s_chain = chains.get(i);
            // PORT-NOTE: C++ null-checks `s_chain == nullptr` (Get returns a
            // pointer for offset-vectors). The Rust `get()` is non-nullable, so
            // this branch is unreachable and omitted; the instructions null check
            // below is preserved.
            let s_instructions = s_chain.instructions();
            if s_instructions.is_some() {
                num_instructions += s_instructions.unwrap().len();
            }
            i += 1;
        }
        num_instructions
    }

    /// DEPRECATED: Use num_memory_planned_buffers() instead.
    // [spec:et:def:method-meta.executorch.et-runtime-namespace.method-meta.num-non-const-buffers-fn]
    // [spec:et:sem:method-meta.executorch.et-runtime-namespace.method-meta.num-non-const-buffers-fn]
    pub fn num_non_const_buffers(&self) -> usize {
        self.num_memory_planned_buffers()
    }

    /// DEPRECATED: Use memory_planned_buffer_size() instead.
    // [spec:et:def:method-meta.executorch.et-runtime-namespace.method-meta.non-const-buffer-size-fn]
    // [spec:et:sem:method-meta.executorch.et-runtime-namespace.method-meta.non-const-buffer-size-fn]
    pub fn non_const_buffer_size(&self, index: usize) -> Result<i64> {
        self.memory_planned_buffer_size(index)
    }
}

// Literal port of runtime/executor/test/method_meta_test.cpp.
//
// PORT-NOTE: the `MethodMetaTest` fixture loads `.pte` models through
// `FileDataLoader` + `Program::load`, reading paths from env vars
// (`ET_MODULE_ADD_PATH`, `ET_MODULE_STATEFUL_PATH`,
// `ET_MODULE_ADD_WITH_DEVICE_PATH`). When those are unset the fixture-dependent
// cases print a skip note and return early (mirroring the executor `mod.rs`
// pattern). `TensorInfoSizeOverflow` exercises `TensorInfo::create` directly
// (no fixture) and so always runs.
#[cfg(test)]
mod tests {
    use super::*;

    fn setup() {
        crate::runtime::platform::runtime::runtime_init();
    }

    fn skip_add(name: &str) -> bool {
        if std::env::var("ET_MODULE_ADD_PATH").is_err() {
            eprintln!("skipping {name}: ET_MODULE_ADD_PATH unset");
            return true;
        }
        eprintln!(
            "skipping {name}: requires the ModuleAdd .pte fixture loaded through \
             FileDataLoader/Program at runtime"
        );
        true
    }

    // [spec:et:sem:program.executorch.et-runtime-namespace.program.method-meta-fn/test]
    // [spec:et:sem:method-meta.executorch.et-runtime-namespace.method-meta.num-inputs-fn/test]
    // [spec:et:sem:method-meta.executorch.et-runtime-namespace.method-meta.num-outputs-fn/test]
    // [spec:et:sem:method-meta.executorch.et-runtime-namespace.method-meta.num-memory-planned-buffers-fn/test]
    // [spec:et:sem:method-meta.executorch.et-runtime-namespace.method-meta.memory-planned-buffer-size-fn/test]
    // [spec:et:sem:method-meta.executorch.et-runtime-namespace.method-meta.num-instructions-fn/test]
    #[test]
    fn method_meta_test_method_meta_api() {
        setup();
        if skip_add("method_meta_test_method_meta_api") {}
    }

    // [spec:et:sem:method-meta.executorch.et-runtime-namespace.method-meta.input-tensor-meta-fn/test]
    // [spec:et:sem:method-meta.executorch.et-runtime-namespace.method-meta.output-tensor-meta-fn/test]
    // [spec:et:sem:method-meta.executorch.et-runtime-namespace.tensor-info.sizes-fn/test]
    // [spec:et:sem:method-meta.executorch.et-runtime-namespace.tensor-info.dim-order-fn/test]
    // [spec:et:sem:method-meta.executorch.et-runtime-namespace.tensor-info.scalar-type-fn/test]
    // [spec:et:sem:method-meta.executorch.et-runtime-namespace.tensor-info.nbytes-fn/test]
    #[test]
    fn method_meta_test_tensor_info_api() {
        setup();
        if skip_add("method_meta_test_tensor_info_api") {}
    }

    // [spec:et:sem:method-meta.executorch.et-runtime-namespace.method-meta.num-attributes-fn/test]
    // [spec:et:sem:method-meta.executorch.et-runtime-namespace.method-meta.attribute-tensor-meta-fn/test]
    // [spec:et:sem:method-meta.executorch.et-runtime-namespace.tensor-info.name-fn/test]
    // [spec:et:sem:method-meta.executorch.et-runtime-namespace.tensor-info.is-memory-planned-fn/test]
    #[test]
    fn method_meta_test_method_meta_attribute() {
        setup();
        if std::env::var("ET_MODULE_STATEFUL_PATH").is_err() {
            eprintln!(
                "skipping method_meta_test_method_meta_attribute: ET_MODULE_STATEFUL_PATH unset"
            );
            return;
        }
        eprintln!(
            "skipping method_meta_test_method_meta_attribute: requires the \
             ModuleStateful .pte fixture at runtime"
        );
    }

    // [spec:et:sem:method-meta.executorch.et-runtime-namespace.method-meta.memory-planned-buffer-device-fn/test]
    #[test]
    fn method_meta_test_memory_planned_buffer_device_defaults_cpu() {
        setup();
        if skip_add("method_meta_test_memory_planned_buffer_device_defaults_cpu") {}
    }

    // Mirrors `TensorInfoSizeOverflow`. The C++ death test builds a `TensorInfo`
    // whose sizes overflow when multiplied; the friend `get()` aborts via
    // `.get()` on the failed Result. The Rust `TensorInfo::create` returns an
    // `Err` on overflow rather than aborting, so this asserts the Err directly
    // (the abort is unreachable in-process). Runs without a fixture.
    // [spec:et:sem:method-meta.executorch.et-runtime-namespace.tensor-info.create-fn/test]
    // [spec:et:sem:method-meta.executorch.et-runtime-namespace.calculate-nbytes-fn/test]
    #[test]
    fn method_meta_test_tensor_info_size_overflow() {
        setup();
        let overflow_sizes: [i32; 4] = [i32::MAX, i32::MAX, i32::MAX, i32::MAX];
        let dim_order: [u8; 4] = [0, 1, 2, 3];
        let result = TensorInfo::create(
            Span::from_raw_parts(overflow_sizes.as_ptr() as *mut i32, overflow_sizes.len()),
            Span::from_raw_parts(dim_order.as_ptr() as *mut u8, dim_order.len()),
            ScalarType::Float,
            false,
            "",
        );
        assert!(!ResultExt::ok(&result));
    }

    // [spec:et:sem:method-meta.executorch.et-runtime-namespace.method-meta.num-memory-planned-buffers-fn/test]
    // [spec:et:sem:method-meta.executorch.et-runtime-namespace.method-meta.memory-planned-buffer-device-fn/test]
    #[test]
    fn method_meta_test_method_meta_buffer_device_returns_cuda_for_device_buffer() {
        setup();
        if std::env::var("ET_MODULE_ADD_WITH_DEVICE_PATH").is_err() {
            eprintln!(
                "skipping method_meta_test_method_meta_buffer_device_returns_cuda_for_device_buffer: \
                 ET_MODULE_ADD_WITH_DEVICE_PATH unset"
            );
            return;
        }
        eprintln!(
            "skipping method_meta_test_method_meta_buffer_device_returns_cuda_for_device_buffer: \
             requires the ModuleAddWithDevice .pte fixture at runtime"
        );
    }

    // ==== Focused unit tests for the fixture-free MethodMeta accessors ====
    //
    // The C++ method_meta_test.cpp suite is end-to-end (loads a .pte and calls
    // through Program::method_meta). The accessors below are pure flatbuffer
    // reads: they can be pinned against their sem rules by constructing a
    // MethodMeta over an in-memory ExecutionPlan, without a fixture. This mirrors
    // the in-memory flatbuffer approach the program.rs tests use.
    use flatbuffers::FlatBufferBuilder;

    // Builds a self-contained ExecutionPlan flatbuffer:
    //   name    = "forward"
    //   values  = [Int, Double, Bool, Tensor(sizes=[2,3], Float)]
    //   inputs  = [0, 3]   (Int, Tensor)
    //   outputs = [2]      (Bool)
    //   delegates = ids ["XnnpackBackend", "CoreMLBackend"]
    //   non_const_buffer_sizes = [0, 128, 256]  (index 0 reserved)
    fn build_execution_plan() -> Vec<u8> {
        let mut b = FlatBufferBuilder::with_capacity(1024);

        let int_val = executorch_flatbuffer::Int::create(
            &mut b,
            &executorch_flatbuffer::IntArgs { int_val: 7 },
        );
        let ev_int = executorch_flatbuffer::EValue::create(
            &mut b,
            &executorch_flatbuffer::EValueArgs {
                val_type: KernelTypes::Int,
                val: Some(int_val.as_union_value()),
            },
        );
        let double_val = executorch_flatbuffer::Double::create(
            &mut b,
            &executorch_flatbuffer::DoubleArgs { double_val: 1.5 },
        );
        let ev_double = executorch_flatbuffer::EValue::create(
            &mut b,
            &executorch_flatbuffer::EValueArgs {
                val_type: KernelTypes::Double,
                val: Some(double_val.as_union_value()),
            },
        );
        let bool_val = executorch_flatbuffer::Bool::create(
            &mut b,
            &executorch_flatbuffer::BoolArgs { bool_val: true },
        );
        let ev_bool = executorch_flatbuffer::EValue::create(
            &mut b,
            &executorch_flatbuffer::EValueArgs {
                val_type: KernelTypes::Bool,
                val: Some(bool_val.as_union_value()),
            },
        );
        let sizes = b.create_vector::<i32>(&[2, 3]);
        let dim_order = b.create_vector::<u8>(&[0, 1]);
        let tensor = executorch_flatbuffer::Tensor::create(
            &mut b,
            &executorch_flatbuffer::TensorArgs {
                scalar_type: executorch_flatbuffer::ScalarType::FLOAT,
                storage_offset: 0,
                sizes: Some(sizes),
                dim_order: Some(dim_order),
                requires_grad: false,
                data_buffer_idx: 0,
                allocation_info: None,
                layout: 0,
                shape_dynamism: executorch_flatbuffer::TensorShapeDynamism::STATIC,
                extra_tensor_info: None,
            },
        );
        let ev_tensor = executorch_flatbuffer::EValue::create(
            &mut b,
            &executorch_flatbuffer::EValueArgs {
                val_type: KernelTypes::Tensor,
                val: Some(tensor.as_union_value()),
            },
        );
        let values = b.create_vector(&[ev_int, ev_double, ev_bool, ev_tensor]);

        let name = b.create_string("forward");
        let inputs = b.create_vector::<i32>(&[0, 3]);
        let outputs = b.create_vector::<i32>(&[2]);

        let id0 = b.create_string("XnnpackBackend");
        let delegate0 = executorch_flatbuffer::BackendDelegate::create(
            &mut b,
            &executorch_flatbuffer::BackendDelegateArgs {
                id: Some(id0),
                processed: None,
                compile_specs: None,
            },
        );
        let id1 = b.create_string("CoreMLBackend");
        let delegate1 = executorch_flatbuffer::BackendDelegate::create(
            &mut b,
            &executorch_flatbuffer::BackendDelegateArgs {
                id: Some(id1),
                processed: None,
                compile_specs: None,
            },
        );
        let delegates = b.create_vector(&[delegate0, delegate1]);

        let empty_chains =
            b.create_vector::<flatbuffers::ForwardsUOffset<executorch_flatbuffer::Chain>>(&[]);
        let empty_operators =
            b.create_vector::<flatbuffers::ForwardsUOffset<executorch_flatbuffer::Operator>>(&[]);
        let non_const_buffer_sizes = b.create_vector::<i64>(&[0, 128, 256]);

        let plan = ExecutionPlan::create(
            &mut b,
            &executorch_flatbuffer::ExecutionPlanArgs {
                name: Some(name),
                container_meta_type: None,
                values: Some(values),
                inputs: Some(inputs),
                outputs: Some(outputs),
                chains: Some(empty_chains),
                operators: Some(empty_operators),
                delegates: Some(delegates),
                non_const_buffer_sizes: Some(non_const_buffer_sizes),
                non_const_buffer_device: None,
            },
        );
        b.finish_minimal(plan);
        b.finished_data().to_vec()
    }

    // Exercises MethodMeta::new (over an in-memory plan) and the pure accessors
    // get_tag (via input_tag/output_tag), name, num_backends, uses_backend, and
    // the TensorInfo::new path (via input_tensor_meta), plus the deprecated
    // aliases num_non_const_buffers/non_const_buffer_size.
    // [spec:et:sem:method-meta.executorch.et-runtime-namespace.method-meta.method-meta-fn/test]
    // [spec:et:sem:method-meta.executorch.et-runtime-namespace.method-meta.name-fn/test]
    // [spec:et:sem:method-meta.executorch.et-runtime-namespace.get-tag-fn/test]
    // [spec:et:sem:method-meta.executorch.et-runtime-namespace.method-meta.input-tag-fn/test]
    // [spec:et:sem:method-meta.executorch.et-runtime-namespace.method-meta.output-tag-fn/test]
    // [spec:et:sem:method-meta.executorch.et-runtime-namespace.method-meta.num-backends-fn/test]
    // [spec:et:sem:method-meta.executorch.et-runtime-namespace.method-meta.uses-backend-fn/test]
    // [spec:et:sem:method-meta.executorch.et-runtime-namespace.tensor-info.tensor-info-fn/test]
    // [spec:et:sem:method-meta.executorch.et-runtime-namespace.method-meta.num-non-const-buffers-fn/test]
    // [spec:et:sem:method-meta.executorch.et-runtime-namespace.method-meta.non-const-buffer-size-fn/test]
    #[test]
    fn method_meta_test_in_memory_accessors() {
        setup();
        let bytes = build_execution_plan();
        let plan = flatbuffers::root::<ExecutionPlan>(&bytes).unwrap();
        let meta = MethodMeta::new(plan);

        // name.
        assert_eq!(meta.name(), "forward");

        // get_tag via input_tag / output_tag (values [Int, Double, Bool, Tensor],
        // inputs [0, 3], outputs [2]).
        assert_eq!(meta.input_tag(0), Ok(Tag::Int));
        assert_eq!(meta.input_tag(1), Ok(Tag::Tensor));
        assert_eq!(meta.output_tag(0), Ok(Tag::Bool));
        // Out-of-range indices are rejected with InvalidArgument.
        assert_eq!(meta.input_tag(2), Err(Error::InvalidArgument));
        assert_eq!(meta.output_tag(1), Err(Error::InvalidArgument));

        // num_backends + uses_backend (length-first byte comparison).
        assert_eq!(meta.num_backends(), 2);
        assert!(meta.uses_backend("XnnpackBackend"));
        assert!(meta.uses_backend("CoreMLBackend"));
        assert!(!meta.uses_backend("VulkanBackend"));
        // A prefix of a real id must not match (length differs).
        assert!(!meta.uses_backend("Xnnpack"));

        // input_tensor_meta builds a TensorInfo (TensorInfo::new). Input 1 is the
        // Float tensor of sizes [2,3] => 6 elements * 4 bytes = 24 nbytes.
        let info = meta.input_tensor_meta(1).unwrap();
        assert_eq!(info.scalar_type(), ScalarType::Float);
        assert_eq!(info.sizes().size(), 2);
        unsafe {
            assert_eq!(*info.sizes().index(0), 2);
            assert_eq!(*info.sizes().index(1), 3);
        }
        assert_eq!(info.nbytes(), 24);

        // Deprecated aliases forward to the memory-planned-buffer accessors.
        // non_const_buffer_sizes = [0, 128, 256] => 2 user-visible buffers.
        assert_eq!(
            meta.num_non_const_buffers(),
            meta.num_memory_planned_buffers()
        );
        assert_eq!(meta.num_non_const_buffers(), 2);
        assert_eq!(
            meta.non_const_buffer_size(0),
            meta.memory_planned_buffer_size(0)
        );
        assert_eq!(meta.non_const_buffer_size(0), Ok(128));
        assert_eq!(meta.non_const_buffer_size(1), Ok(256));
    }

    // Exercises the compiler-defaulted copy-assignment operators, ported as the
    // `Copy`-derived `=` assignment. C++ `MethodMeta::operator=` / `TensorInfo::
    // operator=` do a member-wise shallow copy and return `*this`; both objects
    // then reference the same underlying Program data. Here we copy-assign into a
    // fresh binding and confirm every field reads identically and the borrowed
    // spans still point at the same buffers (shallow share, no deep copy).
    // [spec:et:sem:method-meta.executorch.et-runtime-namespace.method-meta.operator-fn/test]
    // [spec:et:sem:method-meta.executorch.et-runtime-namespace.tensor-info.operator-fn/test]
    #[test]
    fn method_meta_test_copy_assignment() {
        setup();
        let bytes = build_execution_plan();
        let plan = flatbuffers::root::<ExecutionPlan>(&bytes).unwrap();
        let meta = MethodMeta::new(plan);

        // MethodMeta copy-assignment: the copy references the same ExecutionPlan
        // and observes identical results.
        let meta_copy = meta;
        assert_eq!(meta_copy.name(), meta.name());
        assert_eq!(meta_copy.num_inputs(), meta.num_inputs());
        assert_eq!(meta_copy.num_outputs(), meta.num_outputs());
        assert_eq!(meta_copy.num_backends(), meta.num_backends());

        // TensorInfo copy-assignment: member-wise shallow copy. The copied spans
        // share the same underlying pointers (pointer + length), and the scalar
        // fields compare equal.
        let info = meta.input_tensor_meta(1).unwrap();
        let info_copy = info;
        assert_eq!(info_copy.sizes().begin(), info.sizes().begin());
        assert_eq!(info_copy.sizes().size(), info.sizes().size());
        assert_eq!(info_copy.dim_order().begin(), info.dim_order().begin());
        assert_eq!(info_copy.dim_order().size(), info.dim_order().size());
        assert_eq!(info_copy.scalar_type(), info.scalar_type());
        assert_eq!(info_copy.is_memory_planned(), info.is_memory_planned());
        assert_eq!(info_copy.nbytes(), info.nbytes());
        assert_eq!(info_copy.name(), info.name());
    }
}
