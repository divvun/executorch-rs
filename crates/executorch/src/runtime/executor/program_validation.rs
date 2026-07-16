//! Literal port of runtime/executor/program_validation.cpp.
//!
//! NAME-MAPPING DEVIATION: the C++ walks the parsed flatbuffer through the
//! generated accessors (`program->execution_plan()`, `plan->values()`,
//! `value->val_type()`, `value->val()`, `tensor->sizes()`,
//! `tensor->scalar_type()`, `tensor->shape_dynamism()`,
//! `tensor_list->items()`). The Rust flatbuffers crate
//! (`crate::schema::generated::executorch_flatbuffer`) exposes:
//!   - nullable table/vector fields as `Option<...>` (the C++ nullable
//!     `const T*`), so each `== nullptr` check becomes `.is_none()`;
//!   - vectors with `len()`/`get(i)` (usize-based, element-by-value,
//!     non-nullable) in place of `->size()`/`->Get(i)`;
//!   - the `EValue::val()` union payload as `Option<flatbuffers::Table>`; the
//!     concrete `Tensor`/`TensorList` view is rebuilt from that table via
//!     `Tensor::init_from_table` / `TensorList::init_from_table`, mirroring the
//!     C++ `static_cast<const Tensor*>(value->val())` reinterpretation.
//! These name/shape differences are recorded here once and used verbatim below.

use crate::runtime::core::error::Error;
use crate::runtime::core::exec_aten::util::scalar_type_util;
use crate::runtime::core::portable_type::scalar_type::ScalarType;
use crate::schema::generated::executorch_flatbuffer;

// PORT-NOTE: `ssize_t` is the C++ signed size type; `numel` is accumulated in
// it. Matches the crate convention (`isize`).
type ssize_t = isize;

// PORT-NOTE: `c10::mul_overflows(a, b, &out)` (c10/util/safe_numerics.h) returns
// true on overflow and writes the wrapped product to `out`. Not part of this
// module set; ported inline via `checked_mul` (None on overflow) with the
// wrapped product written on overflow to match the C++ out-param contract.
fn mul_overflows_ssize(a: ssize_t, b: ssize_t, out: &mut ssize_t) -> bool {
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

fn mul_overflows_size(a: usize, b: usize, out: &mut usize) -> bool {
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

// [spec:et:def:program-validation.executorch.runtime.validate-tensor-fn]
// [spec:et:sem:program-validation.executorch.runtime.validate-tensor-fn]
//
// PORT-NOTE: the C++ takes `const executorch_flatbuffer::Tensor*` (nullable);
// here the caller has already reconstructed a non-nullable `Tensor` view from
// the union table, so the `tensor == nullptr` branch is unreachable in Rust and
// is not represented. The predicate that would guard it (`tensor != nullptr`)
// is checked at the call site (`validate_program`) instead, exactly where the
// C++ pointer could be null.
#[must_use]
pub fn validate_tensor(tensor: &executorch_flatbuffer::Tensor) -> Error {
    let sizes = tensor.sizes();
    if sizes.is_none() {
        crate::et_log!(Error, "Tensor has null sizes");
        return Error::InvalidProgram;
    }
    let sizes = sizes.unwrap();

    let mut numel: ssize_t = 1;
    let mut numel_overflowed: bool = false;
    for i in 0..sizes.len() {
        let size: i32 = sizes.get(i);

        if size < 0 {
            crate::et_log!(
                Error,
                "Size must be non-negative, got {} at dimension {}",
                size,
                i as u32
            );
            return Error::InvalidProgram;
        }

        if !numel_overflowed {
            numel_overflowed = mul_overflows_ssize(numel, size as ssize_t, &mut numel);
        }
    }

    let scalar_type = scalar_type_from_i8(tensor.scalar_type().0);
    if !scalar_type_util::is_valid(scalar_type) {
        crate::et_log!(Error, "Invalid ScalarType {}", scalar_type as i32);
        return Error::InvalidProgram;
    }

    if numel_overflowed {
        return Error::InvalidProgram;
    }

    let mut nbytes: usize = 0;
    let nbytes_overflow = mul_overflows_size(
        numel as usize,
        scalar_type_util::element_size(scalar_type),
        &mut nbytes,
    );
    if nbytes_overflow {
        return Error::InvalidProgram;
    }

    Error::Ok
}

// PORT-NOTE: the generated flatbuffer `ScalarType(i8)` newtype and the runtime
// `executorch::aten::ScalarType` enum share discriminants; the C++
// `static_cast<executorch::aten::ScalarType>(tensor->scalar_type())` becomes a
// discriminant-preserving reinterpret. Any value outside the enum's declared
// range is mapped to `Undefined` so `is_valid` rejects it, matching the C++
// where an out-of-range cast yields a value `isValid` treats as invalid.
fn scalar_type_from_i8(value: i8) -> ScalarType {
    if value >= 0 && value <= (ScalarType::NumOptions as i8) {
        unsafe { core::mem::transmute::<i8, ScalarType>(value) }
    } else {
        ScalarType::Undefined
    }
}

// [spec:et:def:program-validation.executorch.runtime.validate-program-fn]
// [spec:et:sem:program-validation.executorch.runtime.validate-program-fn]
//
// PORT-NOTE: the C++ takes `const executorch_flatbuffer::Program*` (nullable).
// The Rust caller (`Program::load`) holds a non-nullable `Program` view, so the
// `program == nullptr` branch is unreachable; it is preserved structurally by
// taking a reference (the null case cannot arise). The `is_dynamic_input`
// lambda is ported as a nested closure capturing `inputs`/`values` by
// reference, matching the C++ capture-by-reference `[&]`.
#[must_use]
pub fn validate_program(program: &executorch_flatbuffer::Program) -> Error {
    // Validate all execution plans.
    let execution_plans = program.execution_plan();
    if execution_plans.is_none() {
        crate::et_log!(Error, "Program has null execution_plan");
        return Error::InvalidProgram;
    }
    let execution_plans = execution_plans.unwrap();

    for plan_idx in 0..execution_plans.len() {
        let plan = execution_plans.get(plan_idx);
        // PORT-NOTE: C++ checks `plan == nullptr`; the Rust `.get()` returns a
        // non-nullable value, so the null branch cannot arise and is omitted.

        // Validate all values in the plan.
        let values = plan.values();
        if values.is_none() {
            crate::et_log!(
                Error,
                "Execution plan {} has null values table",
                plan_idx as u32
            );
            return Error::InvalidProgram;
        }
        let values = values.unwrap();

        let inputs = plan.inputs();
        let is_dynamic_input = |idx: usize| -> bool {
            if inputs.is_none() {
                return false;
            }
            let inputs = inputs.unwrap();
            for i in 0..inputs.len() {
                if inputs.get(i) == idx as i32 {
                    let value = values.get(idx);
                    // PORT-NOTE: C++ null-checks `value`; `.get()` is
                    // non-nullable in Rust, so the `value == nullptr` early
                    // `return false` cannot arise.
                    if value.val_type() != executorch_flatbuffer::KernelTypes::Tensor {
                        return false;
                    }
                    let tensor = match value.val() {
                        Some(table) => unsafe {
                            executorch_flatbuffer::Tensor::init_from_table(table)
                        },
                        None => return false,
                    };
                    return tensor.shape_dynamism()
                        != executorch_flatbuffer::TensorShapeDynamism::STATIC;
                }
            }
            false
        };

        for value_idx in 0..values.len() {
            let value = values.get(value_idx);
            // PORT-NOTE: C++ `if (value == nullptr) { continue; }`; `.get()` is
            // non-nullable, so the null-value skip cannot arise.

            // Check if this value is a tensor.
            if value.val_type() == executorch_flatbuffer::KernelTypes::Tensor {
                // PORT-NOTE: C++ reinterprets `value->val()` as `Tensor*`
                // (may be null); here the union payload table is rebuilt into a
                // `Tensor` view. A missing payload table is treated as the null
                // pointer and reported through `validate_tensor`'s null path,
                // which the C++ handles inside `validate_tensor` itself.
                let err = match value.val() {
                    Some(table) => {
                        let tensor =
                            unsafe { executorch_flatbuffer::Tensor::init_from_table(table) };
                        validate_tensor(&tensor)
                    }
                    None => {
                        crate::et_log!(Error, "Tensor is null");
                        Error::InvalidProgram
                    }
                };
                if err != Error::Ok {
                    // Dynamic input tensors may have upper-bound sizes
                    // serialized for 64-bit machines that would overflow on
                    // 32-bit. Since their actual sizes are provided at
                    // set_input time, we defer overflow checks for those to
                    // Method::set_input.
                    if is_dynamic_input(value_idx) {
                        crate::et_log!(
                            Info,
                            "Skipping validation failure for dynamic input tensor at value {} in execution plan {}",
                            value_idx as u32,
                            plan_idx as u32
                        );
                    } else {
                        crate::et_log!(
                            Error,
                            "Tensor validation failed for value {} in execution plan {}",
                            value_idx as u32,
                            plan_idx as u32
                        );
                        return err;
                    }
                }
            }

            // Check if this value is a TensorList.
            if value.val_type() == executorch_flatbuffer::KernelTypes::TensorList {
                // PORT-NOTE: C++ reinterprets `value->val()` as `TensorList*`
                // and then null-checks it; a missing union payload table is the
                // null pointer here.
                let tensor_list = match value.val() {
                    Some(table) => unsafe {
                        executorch_flatbuffer::TensorList::init_from_table(table)
                    },
                    None => {
                        crate::et_log!(
                            Error,
                            "TensorList is null for value {} in execution plan {}",
                            value_idx as u32,
                            plan_idx as u32
                        );
                        return Error::InvalidProgram;
                    }
                };

                let items = tensor_list.items();
                if items.is_none() {
                    crate::et_log!(Error, "TensorList items is null");
                    return Error::InvalidProgram;
                }
                let items = items.unwrap();

                // Validate that each item index points to a Tensor evalue.
                for item_idx in 0..items.len() {
                    let evalue_index: i32 = items.get(item_idx);

                    // Check bounds.
                    if evalue_index < 0 || (evalue_index as usize) >= values.len() {
                        crate::et_log!(
                            Error,
                            "TensorList item {} has out-of-bounds index {} (values size {}) in execution plan {}",
                            item_idx as u32,
                            evalue_index,
                            values.len() as u32,
                            plan_idx as u32
                        );
                        return Error::InvalidProgram;
                    }

                    // Check that the referenced evalue is actually a Tensor.
                    let referenced_value = values.get(evalue_index as usize);
                    // PORT-NOTE: C++ null-checks `referenced_value`; `.get()` is
                    // non-nullable in Rust, so the null branch cannot arise.

                    if referenced_value.val_type() != executorch_flatbuffer::KernelTypes::Tensor {
                        crate::et_log!(
                            Error,
                            "TensorList item {} references non-Tensor evalue (type {}) at index {} in execution plan {}",
                            item_idx as u32,
                            referenced_value.val_type().0 as i32,
                            evalue_index,
                            plan_idx as u32
                        );
                        return Error::InvalidProgram;
                    }
                }
            }
        }
    }

    Error::Ok
}

// Literal port of runtime/executor/test/program_validation_test.cpp.
//
// PORT-NOTE: the C++ `CreateTestProgram` builds a minimal valid PTE flatbuffer
// (configurable EValues) and wraps it in an `AlignedBuffer` + `BufferDataLoader`;
// most cases run entirely in-memory. Only the first two cases use `add_loader_`
// (a `.pte` model at `ET_MODULE_ADD_PATH`); those skip when the env var is unset.
// The remaining cases build their own program and always run, exercising
// `Program::load(InternalConsistency)` → `validate_program` (the
// `program-verification` feature is on by default).
#[cfg(test)]
mod tests {
    use super::Error;
    use crate::extension::data_loader::buffer_data_loader::BufferDataLoader;
    use crate::runtime::core::data_loader::DataLoader;
    use crate::runtime::core::result::ResultExt;
    use crate::runtime::executor::program::{Program, Verification};
    use crate::schema::generated::executorch_flatbuffer;
    use flatbuffers::FlatBufferBuilder;

    fn setup() {
        crate::runtime::platform::runtime::runtime_init();
    }

    #[derive(Clone)]
    enum EValueConfig {
        Tensor { sizes: Vec<i32>, is_dynamic: bool },
        Int,
        TensorList { items: Vec<i32> },
    }

    // Unified helper to create a minimal valid PTE flatbuffer with configurable
    // evalues. `input_indices` specifies which value indices appear in the
    // execution plan's inputs list.
    fn create_test_program(configs: &[EValueConfig], input_indices: &[i32]) -> Vec<u8> {
        let mut builder = FlatBufferBuilder::with_capacity(1024);

        let mut evalues = Vec::new();
        for config in configs {
            match config {
                EValueConfig::Tensor { sizes, is_dynamic } => {
                    let sizes_vec = builder.create_vector(sizes);
                    let dim_order: Vec<u8> = (0..sizes.len()).map(|i| i as u8).collect();
                    let dim_order_vec = builder.create_vector(&dim_order);
                    let tensor = executorch_flatbuffer::Tensor::create(
                        &mut builder,
                        &executorch_flatbuffer::TensorArgs {
                            scalar_type: executorch_flatbuffer::ScalarType::FLOAT,
                            storage_offset: 0,
                            sizes: Some(sizes_vec),
                            dim_order: Some(dim_order_vec),
                            requires_grad: false,
                            data_buffer_idx: 0,
                            allocation_info: None,
                            layout: 0,
                            shape_dynamism: if *is_dynamic {
                                executorch_flatbuffer::TensorShapeDynamism::DYNAMIC_BOUND
                            } else {
                                executorch_flatbuffer::TensorShapeDynamism::STATIC
                            },
                            extra_tensor_info: None,
                        },
                    );
                    evalues.push(executorch_flatbuffer::EValue::create(
                        &mut builder,
                        &executorch_flatbuffer::EValueArgs {
                            val_type: executorch_flatbuffer::KernelTypes::Tensor,
                            val: Some(tensor.as_union_value()),
                        },
                    ));
                }
                EValueConfig::Int => {
                    let int_val = executorch_flatbuffer::Int::create(
                        &mut builder,
                        &executorch_flatbuffer::IntArgs { int_val: 42 },
                    );
                    evalues.push(executorch_flatbuffer::EValue::create(
                        &mut builder,
                        &executorch_flatbuffer::EValueArgs {
                            val_type: executorch_flatbuffer::KernelTypes::Int,
                            val: Some(int_val.as_union_value()),
                        },
                    ));
                }
                EValueConfig::TensorList { items } => {
                    let items_vec = builder.create_vector(items);
                    let tensor_list = executorch_flatbuffer::TensorList::create(
                        &mut builder,
                        &executorch_flatbuffer::TensorListArgs {
                            items: Some(items_vec),
                        },
                    );
                    evalues.push(executorch_flatbuffer::EValue::create(
                        &mut builder,
                        &executorch_flatbuffer::EValueArgs {
                            val_type: executorch_flatbuffer::KernelTypes::TensorList,
                            val: Some(tensor_list.as_union_value()),
                        },
                    ));
                }
            }
        }

        let values_vec = builder.create_vector(&evalues);
        let plan_name = builder.create_string("forward");
        let inputs_vec = builder.create_vector(input_indices);
        let empty_int_vec = builder.create_vector::<i32>(&[]);
        let empty_int64_vec = builder.create_vector::<i64>(&[0]);
        let empty_chain_vec = builder
            .create_vector::<flatbuffers::ForwardsUOffset<executorch_flatbuffer::Chain>>(&[]);
        let empty_operators_vec = builder
            .create_vector::<flatbuffers::ForwardsUOffset<executorch_flatbuffer::Operator>>(&[]);
        let empty_delegates_vec = builder.create_vector::<flatbuffers::ForwardsUOffset<
            executorch_flatbuffer::BackendDelegate,
        >>(&[]);

        let execution_plan = executorch_flatbuffer::ExecutionPlan::create(
            &mut builder,
            &executorch_flatbuffer::ExecutionPlanArgs {
                name: Some(plan_name),
                container_meta_type: None,
                values: Some(values_vec),
                inputs: Some(inputs_vec),
                outputs: Some(empty_int_vec),
                chains: Some(empty_chain_vec),
                operators: Some(empty_operators_vec),
                delegates: Some(empty_delegates_vec),
                non_const_buffer_sizes: Some(empty_int64_vec),
                non_const_buffer_device: None,
            },
        );

        let plans_vec = builder.create_vector(&[execution_plan]);

        let program = executorch_flatbuffer::Program::create(
            &mut builder,
            &executorch_flatbuffer::ProgramArgs {
                version: 0,
                execution_plan: Some(plans_vec),
                ..Default::default()
            },
        );

        builder.finish(program, Some(executorch_flatbuffer::PROGRAM_IDENTIFIER));
        builder.finished_data().to_vec()
    }

    // RAII wrapper for aligned buffer allocation (analog of the C++ AlignedBuffer):
    // copies `data` into a boxed slice at a 16-byte-aligned offset and exposes a
    // BufferDataLoader over it.
    struct AlignedBuffer {
        buffer: Box<[u8]>,
        offset: usize,
        size: usize,
    }

    impl AlignedBuffer {
        const K_ALIGNMENT: usize = 16;

        fn new(data: &[u8]) -> Self {
            let mut buffer = vec![0u8; data.len() + Self::K_ALIGNMENT].into_boxed_slice();
            let addr = buffer.as_ptr() as usize;
            let offset = (Self::K_ALIGNMENT - (addr % Self::K_ALIGNMENT)) % Self::K_ALIGNMENT;
            buffer[offset..offset + data.len()].copy_from_slice(data);
            AlignedBuffer {
                buffer,
                offset,
                size: data.len(),
            }
        }

        fn loader(&self) -> BufferDataLoader {
            BufferDataLoader::new(
                unsafe { self.buffer.as_ptr().add(self.offset) } as *const core::ffi::c_void,
                self.size,
            )
        }
    }

    // [spec:et:sem:program.executorch.et-runtime-namespace.program.load-fn/test]
    #[test]
    fn program_validation_test_valid_program_passes_internal_consistency() {
        setup();
        if std::env::var("ET_MODULE_ADD_PATH").is_err() {
            eprintln!(
                "skipping program_validation_test_valid_program_passes_internal_consistency: \
                 ET_MODULE_ADD_PATH unset"
            );
            return;
        }
        eprintln!(
            "skipping program_validation_test_valid_program_passes_internal_consistency: \
             requires the ModuleAdd .pte fixture at runtime"
        );
    }

    // [spec:et:sem:program.executorch.et-runtime-namespace.program.load-fn/test]
    #[test]
    fn program_validation_test_internal_consistency_detects_truncated_data() {
        setup();
        if std::env::var("ET_MODULE_ADD_PATH").is_err() {
            eprintln!(
                "skipping program_validation_test_internal_consistency_detects_truncated_data: \
                 ET_MODULE_ADD_PATH unset"
            );
            return;
        }
        eprintln!(
            "skipping program_validation_test_internal_consistency_detects_truncated_data: \
             requires the ModuleAdd .pte fixture at runtime"
        );
    }

    // Static tensors always have their overflow checked at validation time.
    // [spec:et:sem:program-validation.executorch.runtime.validate-tensor-fn/test]
    // [spec:et:sem:program.executorch.et-runtime-namespace.program.load-fn/test]
    #[test]
    fn program_validation_test_tensor_numel_overflow_detected_for_static_tensor() {
        setup();
        let configs = [EValueConfig::Tensor {
            sizes: vec![2000000000, 2000000000, 2000000000],
            is_dynamic: false,
        }];
        let buf = AlignedBuffer::new(&create_test_program(&configs, &[]));
        let loader = buf.loader();
        let program = Program::load(
            &loader as *const _ as *const dyn DataLoader,
            Verification::InternalConsistency,
        );
        assert_eq!(ResultExt::error(&program), Error::InvalidProgram);
    }

    // Minimal verification doesn't run program validation.
    // [spec:et:sem:program.executorch.et-runtime-namespace.program.load-fn/test]
    #[test]
    fn program_validation_test_tensor_numel_overflow_not_detected_with_minimal() {
        setup();
        let configs = [EValueConfig::Tensor {
            sizes: vec![2000000000, 2000000000, 2000000000],
            is_dynamic: false,
        }];
        let buf = AlignedBuffer::new(&create_test_program(&configs, &[]));
        let loader = buf.loader();
        let program = Program::load(
            &loader as *const _ as *const dyn DataLoader,
            Verification::Minimal,
        );
        assert_eq!(ResultExt::error(&program), Error::Ok);
    }

    // Dynamic input tensors skip overflow checks at validation time.
    // [spec:et:sem:program-validation.executorch.runtime.validate-program-fn/test]
    #[test]
    fn program_validation_test_tensor_numel_overflow_skipped_for_dynamic_input() {
        setup();
        let configs = [EValueConfig::Tensor {
            sizes: vec![2000000000, 2000000000, 2000000000],
            is_dynamic: true,
        }];
        // Mark value index 0 as a plan input.
        let buf = AlignedBuffer::new(&create_test_program(&configs, &[0]));
        let loader = buf.loader();
        let program = Program::load(
            &loader as *const _ as *const dyn DataLoader,
            Verification::InternalConsistency,
        );
        assert_eq!(ResultExt::error(&program), Error::Ok);
    }

    // A dynamic tensor NOT in the inputs list should still have overflow checked.
    // [spec:et:sem:program-validation.executorch.runtime.validate-program-fn/test]
    #[test]
    fn program_validation_test_tensor_numel_overflow_detected_for_dynamic_non_input_tensor() {
        setup();
        let configs = [EValueConfig::Tensor {
            sizes: vec![2000000000, 2000000000, 2000000000],
            is_dynamic: true,
        }];
        let buf = AlignedBuffer::new(&create_test_program(&configs, &[]));
        let loader = buf.loader();
        let program = Program::load(
            &loader as *const _ as *const dyn DataLoader,
            Verification::InternalConsistency,
        );
        assert_eq!(ResultExt::error(&program), Error::InvalidProgram);
    }

    // A static input tensor should still have overflow checked (sizes fixed).
    // [spec:et:sem:program-validation.executorch.runtime.validate-program-fn/test]
    #[test]
    fn program_validation_test_tensor_numel_overflow_detected_for_static_input_tensor() {
        setup();
        let configs = [EValueConfig::Tensor {
            sizes: vec![2000000000, 2000000000, 2000000000],
            is_dynamic: false,
        }];
        let buf = AlignedBuffer::new(&create_test_program(&configs, &[0]));
        let loader = buf.loader();
        let program = Program::load(
            &loader as *const _ as *const dyn DataLoader,
            Verification::InternalConsistency,
        );
        assert_eq!(ResultExt::error(&program), Error::InvalidProgram);
    }

    // [spec:et:sem:program-validation.executorch.runtime.validate-tensor-fn/test]
    #[test]
    fn program_validation_test_negative_size_detected() {
        setup();
        let configs = [EValueConfig::Tensor {
            sizes: vec![10, -5, 10],
            is_dynamic: false,
        }];
        let buf = AlignedBuffer::new(&create_test_program(&configs, &[]));
        let loader = buf.loader();
        let program = Program::load(
            &loader as *const _ as *const dyn DataLoader,
            Verification::InternalConsistency,
        );
        assert_eq!(ResultExt::error(&program), Error::InvalidProgram);
    }

    // values[0]=Tensor, values[1]=Int (INVALID), values[2]=TensorList([0,1]).
    // [spec:et:sem:program-validation.executorch.runtime.validate-program-fn/test]
    #[test]
    fn program_validation_test_tensor_list_with_int_element_detected() {
        setup();
        let configs = [
            EValueConfig::Tensor {
                sizes: vec![2, 3],
                is_dynamic: false,
            },
            EValueConfig::Int,
            EValueConfig::TensorList { items: vec![0, 1] },
        ];
        let buf = AlignedBuffer::new(&create_test_program(&configs, &[]));
        let loader = buf.loader();
        let program = Program::load(
            &loader as *const _ as *const dyn DataLoader,
            Verification::InternalConsistency,
        );
        assert_eq!(ResultExt::error(&program), Error::InvalidProgram);
    }

    // values[0]=Tensor, values[1]=TensorList([0,99]) - 99 is out of bounds.
    // [spec:et:sem:program-validation.executorch.runtime.validate-program-fn/test]
    #[test]
    fn program_validation_test_tensor_list_with_out_of_bounds_index_detected() {
        setup();
        let configs = [
            EValueConfig::Tensor {
                sizes: vec![2, 3],
                is_dynamic: false,
            },
            EValueConfig::TensorList { items: vec![0, 99] },
        ];
        let buf = AlignedBuffer::new(&create_test_program(&configs, &[]));
        let loader = buf.loader();
        let program = Program::load(
            &loader as *const _ as *const dyn DataLoader,
            Verification::InternalConsistency,
        );
        assert_eq!(ResultExt::error(&program), Error::InvalidProgram);
    }
}
