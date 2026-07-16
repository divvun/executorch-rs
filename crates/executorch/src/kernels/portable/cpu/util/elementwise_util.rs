//! Literal port of kernels/portable/cpu/util/elementwise_util.h.
//!
//! PORT-NOTE: the C++ apply driver chooses, at compile time, between a
//! dtype-specialized fast path (`dtype_specialized_elementwise_fn_impl`, whose
//! `CTYPE_OUT` is `ScalarTypeToCppType<out_specialized_scalar_type>::type`, a
//! type computed from a runtime-derived `SupportedTensorDtypes` value) and the
//! generic path. Stable Rust cannot map a value-level `ScalarType` to a
//! compile-time element type without specialization / `generic_const_exprs`, so
//! `apply_elementwise_fn` here always takes the generic path. The generic and
//! fast paths are documented to produce identical numeric results, so behavior
//! is preserved; the fast path and `can_use_vectorized` are ported as standalone
//! items for completeness but are not wired into automatic path selection. The
//! `at::vec` SIMD sub-path (only present with PyTorch headers) is treated as
//! absent, matching the non-`ET_USE_PYTORCH_HEADERS` build.

use crate::kernels::portable::cpu::selective_build::should_include_kernel_dtype;
use crate::kernels::portable::cpu::util::broadcast_indexes_range::{
    BroadcastIndexesRange, sizes_match_ignoring_leading_1s,
};
use crate::kernels::portable::cpu::util::dtype_util::SupportedTensorDtypes;
use crate::kernels::portable::cpu::util::dtype_util::internal::{
    ComputeCast, LoadToComputeFn, StaticCast, check_tensor_dtype, get_load_to_compute_fn,
    get_store_compute_to_tensor_fn, specialized_output_scalar_type,
};
use crate::runtime::core::exec_aten::util::scalar_type_util::CppTypeToScalarType;
use crate::runtime::core::portable_type::scalar_type::ScalarType;
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::core::portable_type::{BFloat16, Half};
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;
use crate::runtime::kernel::thread_parallel_interface::{internal::GRAIN_SIZE, parallel_for};

pub mod internal {
    use super::*;

    /// Causes these utility functions to make sure to respect Tensor strides;
    /// normally, this is not strictly necessary because ExecuTorch Tensors are
    /// contiguous.
    // [spec:et:def:elementwise-util.torch.executor.native.utils.internal.support-noncontiguous-input-tensors]
    // [spec:et:def:elementwise-util.torch.executor.native.utils.internal.support-noncontiguous-input-tensors.support-noncontiguous-input-tensors-fn]
    // [spec:et:sem:elementwise-util.torch.executor.native.utils.internal.support-noncontiguous-input-tensors.support-noncontiguous-input-tensors-fn]
    //
    // PORT-NOTE: modeled as a unit marker; call sites pass
    // `SupportNoncontiguousInputTensors` to select the stride-respecting variant.
    pub struct SupportNoncontiguousInputTensors;

    // PORT-NOTE: an input is a `(tensor, dtypes)` pair, mirroring the C++
    // `std::pair<const Tensor*, SupportedTensorDtypes>`.
    pub type InputPair<'a> = (&'a Tensor<'a>, SupportedTensorDtypes);

    // [spec:et:def:elementwise-util.torch.executor.native.utils.internal.can-use-vectorized-fn]
    // [spec:et:sem:elementwise-util.torch.executor.native.utils.internal.can-use-vectorized-fn]
    //
    // PORT-NOTE: When PyTorch headers are absent the vectorized path does not
    // exist and this predicate is effectively false. This port targets scalar
    // semantics, so it is a constant `false`.
    pub const fn can_use_vectorized() -> bool {
        false
    }

    // [spec:et:def:elementwise-util.torch.executor.native.utils.internal.dtype-specialized-elementwise-fn-impl-fn]
    // [spec:et:sem:elementwise-util.torch.executor.native.utils.internal.dtype-specialized-elementwise-fn-impl-fn]
    //
    // PORT-NOTE: fast path — every input is read directly as `CTYPE_COMPUTE`,
    // results written directly as `CTYPE_OUT`, with no per-element conversion.
    // Only the generic (path B) is retained since the vectorized path (A) is
    // compiled only with PyTorch headers. Provided for completeness; see module
    // doc for why it is not auto-selected.
    pub fn dtype_specialized_elementwise_fn_impl<CTYPE_COMPUTE, CTYPE_OUT, Op>(
        compute_fun: &Op,
        _ctx: &mut KernelRuntimeContext,
        out: &Tensor,
        inputs: &[InputPair],
        support_noncontiguous_tensors: bool,
    ) where
        CTYPE_COMPUTE: Copy + CppTypeToScalarType,
        CTYPE_OUT: Copy + StaticCast<CTYPE_COMPUTE>,
        Op: Fn(&[CTYPE_COMPUTE]) -> CTYPE_COMPUTE,
    {
        let k_num_inputs = inputs.len();
        // All inputs must be of type CTYPE_COMPUTE.
        debug_assert!(
            inputs
                .iter()
                .all(|p| p.0.scalar_type() == CTYPE_COMPUTE::VALUE)
        );

        let inputs_data_ptrs: alloc::vec::Vec<*const CTYPE_COMPUTE> = inputs
            .iter()
            .map(|p| p.0.const_data_ptr::<CTYPE_COMPUTE>())
            .collect();
        let data_out = out.mutable_data_ptr::<CTYPE_OUT>();
        let input_tensors: alloc::vec::Vec<&Tensor> = inputs.iter().map(|p| p.0).collect();

        parallel_for(0, out.numel() as i64, GRAIN_SIZE, &|begin, end| {
            let range = build_range(out, &input_tensors, support_noncontiguous_tensors);
            let mut begin_it = range.begin();
            begin_it.add_assign(begin as isize);
            while begin_it.output_index() < end as isize {
                let mut loaded_inputs: alloc::vec::Vec<CTYPE_COMPUTE> =
                    alloc::vec::Vec::with_capacity(k_num_inputs);
                for idx in 0..k_num_inputs {
                    loaded_inputs
                        .push(unsafe { *inputs_data_ptrs[idx].offset(begin_it.at(idx + 1)) });
                }
                unsafe {
                    *data_out.offset(begin_it.at(0)) =
                        CTYPE_OUT::static_cast(compute_fun(&loaded_inputs));
                }
                begin_it.increment();
            }
        });
    }

    // [spec:et:def:elementwise-util.torch.executor.native.utils.internal.validate-elementwise-fn-inputs-fn]
    // [spec:et:sem:elementwise-util.torch.executor.native.utils.internal.validate-elementwise-fn-inputs-fn]
    pub fn validate_elementwise_fn_inputs<CTYPE_COMPUTE: CppTypeToScalarType>(
        ctx: &mut KernelRuntimeContext,
        out: &Tensor,
        out_dtypes: SupportedTensorDtypes,
        inputs: &[InputPair],
    ) -> bool {
        let compute_type = CTYPE_COMPUTE::VALUE;
        let all_inputs_ok = inputs
            .iter()
            .all(|input| check_tensor_dtype(input.0, input.1, compute_type));
        crate::et_kernel_check!(
            ctx,
            all_inputs_ok && check_tensor_dtype(out, out_dtypes, compute_type),
            InvalidArgument,
            false
        );

        true
    }

    // [spec:et:def:elementwise-util.torch.executor.native.utils.internal.apply-elementwise-fn-generic-impl-fn]
    // [spec:et:sem:elementwise-util.torch.executor.native.utils.internal.apply-elementwise-fn-generic-impl-fn]
    pub fn apply_elementwise_fn_generic_impl<CTYPE_COMPUTE, Op>(
        compute_fun: &Op,
        ctx: &mut KernelRuntimeContext,
        out: &Tensor,
        out_dtypes: SupportedTensorDtypes,
        inputs: &[InputPair],
        support_noncontiguous_tensors: bool,
    ) where
        CTYPE_COMPUTE: ComputeCast + CppTypeToScalarType + StaticCast<CTYPE_COMPUTE> + 'static,
        Op: Fn(&[CTYPE_COMPUTE]) -> CTYPE_COMPUTE,
        u8: StaticCast<CTYPE_COMPUTE>,
        i8: StaticCast<CTYPE_COMPUTE>,
        i16: StaticCast<CTYPE_COMPUTE>,
        i32: StaticCast<CTYPE_COMPUTE>,
        i64: StaticCast<CTYPE_COMPUTE>,
        f32: StaticCast<CTYPE_COMPUTE>,
        f64: StaticCast<CTYPE_COMPUTE>,
        bool: StaticCast<CTYPE_COMPUTE>,
        Half: StaticCast<CTYPE_COMPUTE>,
        BFloat16: StaticCast<CTYPE_COMPUTE>,
    {
        let k_num_inputs = inputs.len();

        struct InputInfo<CTYPE_COMPUTE: 'static> {
            load_to_compute: LoadToComputeFn<CTYPE_COMPUTE>,
            data_ptr: *const u8,
            element_size: isize,
        }
        let inputs_info: alloc::vec::Vec<InputInfo<CTYPE_COMPUTE>> = inputs
            .iter()
            .map(|p| InputInfo {
                load_to_compute: get_load_to_compute_fn::<CTYPE_COMPUTE>(ctx, p.0, p.1, ""),
                data_ptr: p.0.const_data_ptr::<u8>(),
                element_size: p.0.element_size(),
            })
            .collect();

        let store_compute_to_out =
            get_store_compute_to_tensor_fn::<CTYPE_COMPUTE>(ctx, out, out_dtypes, "");
        let data_out = out.mutable_data_ptr::<u8>();
        let out_element_size = out.element_size();
        let input_tensors: alloc::vec::Vec<&Tensor> = inputs.iter().map(|p| p.0).collect();

        parallel_for(0, out.numel() as i64, GRAIN_SIZE, &|begin, end| {
            let range = build_range(out, &input_tensors, support_noncontiguous_tensors);
            let mut begin_it = range.begin();
            begin_it.add_assign(begin as isize);
            while begin_it.output_index() < end as isize {
                let mut loaded_inputs: alloc::vec::Vec<CTYPE_COMPUTE> =
                    alloc::vec::Vec::with_capacity(k_num_inputs);
                for idx in 0..k_num_inputs {
                    let input_info = &inputs_info[idx];
                    loaded_inputs.push((input_info.load_to_compute)(unsafe {
                        input_info
                            .data_ptr
                            .offset(begin_it.at(idx + 1) * input_info.element_size)
                            as *const core::ffi::c_void
                    }));
                }
                let result = compute_fun(&loaded_inputs);
                store_compute_to_out(result, unsafe {
                    data_out.offset(begin_it.at(0) * out_element_size) as *mut core::ffi::c_void
                });
                begin_it.increment();
            }
        });
    }

    // [spec:et:def:elementwise-util.torch.executor.native.utils.internal.apply-elementwise-fn-runtime-out-dtypes-fn]
    // [spec:et:sem:elementwise-util.torch.executor.native.utils.internal.apply-elementwise-fn-runtime-out-dtypes-fn]
    pub fn apply_elementwise_fn_runtime_out_dtypes<CTYPE_COMPUTE, Op>(
        compute_fun: &Op,
        ctx: &mut KernelRuntimeContext,
        out: &Tensor,
        out_dtypes: SupportedTensorDtypes,
        inputs: &[InputPair],
    ) where
        CTYPE_COMPUTE: ComputeCast + CppTypeToScalarType + StaticCast<CTYPE_COMPUTE> + 'static,
        Op: Fn(&[CTYPE_COMPUTE]) -> CTYPE_COMPUTE,
        u8: StaticCast<CTYPE_COMPUTE>,
        i8: StaticCast<CTYPE_COMPUTE>,
        i16: StaticCast<CTYPE_COMPUTE>,
        i32: StaticCast<CTYPE_COMPUTE>,
        i64: StaticCast<CTYPE_COMPUTE>,
        f32: StaticCast<CTYPE_COMPUTE>,
        f64: StaticCast<CTYPE_COMPUTE>,
        bool: StaticCast<CTYPE_COMPUTE>,
        Half: StaticCast<CTYPE_COMPUTE>,
        BFloat16: StaticCast<CTYPE_COMPUTE>,
    {
        let inputs_valid =
            validate_elementwise_fn_inputs::<CTYPE_COMPUTE>(ctx, out, out_dtypes, inputs);
        if !inputs_valid {
            return;
        }

        apply_elementwise_fn_generic_impl::<CTYPE_COMPUTE, Op>(
            compute_fun,
            ctx,
            out,
            out_dtypes,
            inputs,
            /*support_noncontiguous_tensors*/ false,
        );
    }

    // [spec:et:def:elementwise-util.torch.executor.native.utils.internal.apply-elementwise-fn-fn]
    // [spec:et:sem:elementwise-util.torch.executor.native.utils.internal.apply-elementwise-fn-fn]
    //
    // PORT-NOTE: `out_dtypes` is a runtime argument here (see module doc); the
    // compile-time fast-path selection collapses to the generic path, which is
    // numerically identical.
    pub fn apply_elementwise_fn<CTYPE_COMPUTE, Op>(
        compute_fun: &Op,
        ctx: &mut KernelRuntimeContext,
        out: &Tensor,
        out_dtypes: SupportedTensorDtypes,
        support_noncontiguous_tensors: bool,
        inputs: &[InputPair],
    ) where
        CTYPE_COMPUTE: ComputeCast + CppTypeToScalarType + StaticCast<CTYPE_COMPUTE> + 'static,
        Op: Fn(&[CTYPE_COMPUTE]) -> CTYPE_COMPUTE,
        u8: StaticCast<CTYPE_COMPUTE>,
        i8: StaticCast<CTYPE_COMPUTE>,
        i16: StaticCast<CTYPE_COMPUTE>,
        i32: StaticCast<CTYPE_COMPUTE>,
        i64: StaticCast<CTYPE_COMPUTE>,
        f32: StaticCast<CTYPE_COMPUTE>,
        f64: StaticCast<CTYPE_COMPUTE>,
        bool: StaticCast<CTYPE_COMPUTE>,
        Half: StaticCast<CTYPE_COMPUTE>,
        BFloat16: StaticCast<CTYPE_COMPUTE>,
    {
        let inputs_valid =
            validate_elementwise_fn_inputs::<CTYPE_COMPUTE>(ctx, out, out_dtypes, inputs);
        if !inputs_valid {
            return;
        }

        // PORT-NOTE: the C++ compile-time fast path (matching all-input and out
        // dtypes to their specialized ScalarTypes) is not selectable in stable
        // Rust; `should_include_kernel_dtype` / `specialized_output_scalar_type`
        // are still evaluated to keep the gate observable, but the generic path
        // runs unconditionally with identical numeric results.
        let out_specialized_scalar_type =
            specialized_output_scalar_type::<CTYPE_COMPUTE>(out_dtypes);
        let _ = should_include_kernel_dtype("", out_specialized_scalar_type);

        apply_elementwise_fn_generic_impl::<CTYPE_COMPUTE, Op>(
            compute_fun,
            ctx,
            out,
            out_dtypes,
            inputs,
            support_noncontiguous_tensors,
        );
    }

    // PORT-NOTE: helper mirroring the C++
    // `BroadcastIndexesRange<kNumInputs, support>(out, (*inputs.first)...)`
    // construction. `NT = inputs.len() + 1`. Only the arities used by the
    // elementwise wrappers (1, 2, 3 inputs) are instantiated.
    fn build_range<'a>(
        out: &'a Tensor<'a>,
        inputs: &[&'a Tensor<'a>],
        support_noncontiguous_tensors: bool,
    ) -> RangeAny<'a> {
        match inputs.len() {
            1 => RangeAny::N2(BroadcastIndexesRange::<2>::new_with_support(
                out,
                inputs,
                support_noncontiguous_tensors,
            )),
            2 => RangeAny::N3(BroadcastIndexesRange::<3>::new_with_support(
                out,
                inputs,
                support_noncontiguous_tensors,
            )),
            3 => RangeAny::N4(BroadcastIndexesRange::<4>::new_with_support(
                out,
                inputs,
                support_noncontiguous_tensors,
            )),
            _ => crate::runtime::platform::abort::runtime_abort(),
        }
    }

    // PORT-NOTE: the const-generic `BroadcastIndexesRange<NT>` cannot be a single
    // runtime value across arities, so a small enum wraps the 2/3/4-tensor cases
    // used by the unary/binary/ternary wrappers, dispatching `begin()`/iteration
    // uniformly.
    enum RangeAny<'a> {
        N2(BroadcastIndexesRange<'a, 2>),
        N3(BroadcastIndexesRange<'a, 3>),
        N4(BroadcastIndexesRange<'a, 4>),
    }

    pub struct IterAny<'a> {
        inner: IterAnyInner<'a>,
    }
    enum IterAnyInner<'a> {
        N2(
            crate::kernels::portable::cpu::util::broadcast_indexes_range::BroadcastIndexesIterator<
                'a,
                2,
            >,
        ),
        N3(
            crate::kernels::portable::cpu::util::broadcast_indexes_range::BroadcastIndexesIterator<
                'a,
                3,
            >,
        ),
        N4(
            crate::kernels::portable::cpu::util::broadcast_indexes_range::BroadcastIndexesIterator<
                'a,
                4,
            >,
        ),
    }

    impl<'a> RangeAny<'a> {
        fn begin(&self) -> IterAny<'a> {
            IterAny {
                inner: match self {
                    RangeAny::N2(r) => IterAnyInner::N2(r.begin()),
                    RangeAny::N3(r) => IterAnyInner::N3(r.begin()),
                    RangeAny::N4(r) => IterAnyInner::N4(r.begin()),
                },
            }
        }
    }

    impl<'a> IterAny<'a> {
        pub fn add_assign(&mut self, n: isize) {
            match &mut self.inner {
                IterAnyInner::N2(it) => it.add_assign(n),
                IterAnyInner::N3(it) => it.add_assign(n),
                IterAnyInner::N4(it) => it.add_assign(n),
            }
        }
        pub fn increment(&mut self) {
            match &mut self.inner {
                IterAnyInner::N2(it) => it.increment(),
                IterAnyInner::N3(it) => it.increment(),
                IterAnyInner::N4(it) => it.increment(),
            }
        }
        pub fn output_index(&self) -> isize {
            match &self.inner {
                IterAnyInner::N2(it) => it.output_index(),
                IterAnyInner::N3(it) => it.output_index(),
                IterAnyInner::N4(it) => it.output_index(),
            }
        }
        // Returns `indexes[i]` (0 = output, i+1 = input i).
        pub fn at(&self, i: usize) -> isize {
            match &self.inner {
                IterAnyInner::N2(it) => it.deref()[i],
                IterAnyInner::N3(it) => it.deref()[i],
                IterAnyInner::N4(it) => it.deref()[i],
            }
        }
    }
}

// DEPRECATED: prefer the variant with out_dtypes in the template argument.
pub fn apply_unitensor_elementwise_fn_deprecated<CTYPE_COMPUTE, Op>(
    compute_fun: Op,
    ctx: &mut KernelRuntimeContext,
    a: &Tensor,
    a_dtypes: SupportedTensorDtypes,
    out: &Tensor,
    out_dtypes: SupportedTensorDtypes,
) where
    CTYPE_COMPUTE: ComputeCast + CppTypeToScalarType + StaticCast<CTYPE_COMPUTE> + 'static,
    Op: Fn(&[CTYPE_COMPUTE]) -> CTYPE_COMPUTE,
    u8: StaticCast<CTYPE_COMPUTE>,
    i8: StaticCast<CTYPE_COMPUTE>,
    i16: StaticCast<CTYPE_COMPUTE>,
    i32: StaticCast<CTYPE_COMPUTE>,
    i64: StaticCast<CTYPE_COMPUTE>,
    f32: StaticCast<CTYPE_COMPUTE>,
    f64: StaticCast<CTYPE_COMPUTE>,
    bool: StaticCast<CTYPE_COMPUTE>,
    Half: StaticCast<CTYPE_COMPUTE>,
    BFloat16: StaticCast<CTYPE_COMPUTE>,
{
    internal::apply_elementwise_fn_runtime_out_dtypes::<CTYPE_COMPUTE, Op>(
        &compute_fun,
        ctx,
        out,
        out_dtypes,
        &[(a, a_dtypes)],
    );
}

/// Useful for unary elementwise operators.
// [spec:et:def:elementwise-util.torch.executor.native.utils.internal.apply-unitensor-elementwise-fn-fn]
// [spec:et:sem:elementwise-util.torch.executor.native.utils.internal.apply-unitensor-elementwise-fn-fn]
//
// PORT-NOTE: `support_noncontiguous` = whether the trailing
// `SupportNoncontiguousInputTensors` tag was supplied by the caller.
pub fn apply_unitensor_elementwise_fn<CTYPE_COMPUTE, Op>(
    compute_fun: Op,
    ctx: &mut KernelRuntimeContext,
    a: &Tensor,
    a_dtypes: SupportedTensorDtypes,
    out: &Tensor,
    out_dtypes: SupportedTensorDtypes,
    support_noncontiguous: bool,
) where
    CTYPE_COMPUTE: ComputeCast + CppTypeToScalarType + StaticCast<CTYPE_COMPUTE> + 'static,
    Op: Fn(&[CTYPE_COMPUTE]) -> CTYPE_COMPUTE,
    u8: StaticCast<CTYPE_COMPUTE>,
    i8: StaticCast<CTYPE_COMPUTE>,
    i16: StaticCast<CTYPE_COMPUTE>,
    i32: StaticCast<CTYPE_COMPUTE>,
    i64: StaticCast<CTYPE_COMPUTE>,
    f32: StaticCast<CTYPE_COMPUTE>,
    f64: StaticCast<CTYPE_COMPUTE>,
    bool: StaticCast<CTYPE_COMPUTE>,
    Half: StaticCast<CTYPE_COMPUTE>,
    BFloat16: StaticCast<CTYPE_COMPUTE>,
{
    internal::apply_elementwise_fn::<CTYPE_COMPUTE, Op>(
        &compute_fun,
        ctx,
        out,
        out_dtypes,
        support_noncontiguous,
        &[(a, a_dtypes)],
    );
}

// DEPRECATED: prefer the variant with out_dtypes in the template argument list.
pub fn apply_bitensor_elementwise_fn_deprecated<CTYPE_COMPUTE, Op>(
    compute_fun: Op,
    ctx: &mut KernelRuntimeContext,
    a: &Tensor,
    a_dtypes: SupportedTensorDtypes,
    b: &Tensor,
    b_dtypes: SupportedTensorDtypes,
    out: &Tensor,
    out_dtypes: SupportedTensorDtypes,
) where
    CTYPE_COMPUTE: ComputeCast + CppTypeToScalarType + StaticCast<CTYPE_COMPUTE> + 'static,
    Op: Fn(&[CTYPE_COMPUTE]) -> CTYPE_COMPUTE,
    u8: StaticCast<CTYPE_COMPUTE>,
    i8: StaticCast<CTYPE_COMPUTE>,
    i16: StaticCast<CTYPE_COMPUTE>,
    i32: StaticCast<CTYPE_COMPUTE>,
    i64: StaticCast<CTYPE_COMPUTE>,
    f32: StaticCast<CTYPE_COMPUTE>,
    f64: StaticCast<CTYPE_COMPUTE>,
    bool: StaticCast<CTYPE_COMPUTE>,
    Half: StaticCast<CTYPE_COMPUTE>,
    BFloat16: StaticCast<CTYPE_COMPUTE>,
{
    internal::apply_elementwise_fn_runtime_out_dtypes::<CTYPE_COMPUTE, Op>(
        &compute_fun,
        ctx,
        out,
        out_dtypes,
        &[(a, a_dtypes), (b, b_dtypes)],
    );
}

/// Useful for bi-tensor elementwise operators.
// [spec:et:def:elementwise-util.torch.executor.native.utils.internal.apply-bitensor-elementwise-fn-fn]
// [spec:et:sem:elementwise-util.torch.executor.native.utils.internal.apply-bitensor-elementwise-fn-fn]
pub fn apply_bitensor_elementwise_fn<CTYPE_COMPUTE, Op>(
    compute_fun: Op,
    ctx: &mut KernelRuntimeContext,
    a: &Tensor,
    a_dtypes: SupportedTensorDtypes,
    b: &Tensor,
    b_dtypes: SupportedTensorDtypes,
    out: &Tensor,
    out_dtypes: SupportedTensorDtypes,
    support_noncontiguous: bool,
) where
    CTYPE_COMPUTE: ComputeCast + CppTypeToScalarType + StaticCast<CTYPE_COMPUTE> + 'static,
    Op: Fn(&[CTYPE_COMPUTE]) -> CTYPE_COMPUTE,
    u8: StaticCast<CTYPE_COMPUTE>,
    i8: StaticCast<CTYPE_COMPUTE>,
    i16: StaticCast<CTYPE_COMPUTE>,
    i32: StaticCast<CTYPE_COMPUTE>,
    i64: StaticCast<CTYPE_COMPUTE>,
    f32: StaticCast<CTYPE_COMPUTE>,
    f64: StaticCast<CTYPE_COMPUTE>,
    bool: StaticCast<CTYPE_COMPUTE>,
    Half: StaticCast<CTYPE_COMPUTE>,
    BFloat16: StaticCast<CTYPE_COMPUTE>,
{
    internal::apply_elementwise_fn::<CTYPE_COMPUTE, Op>(
        &compute_fun,
        ctx,
        out,
        out_dtypes,
        support_noncontiguous,
        &[(a, a_dtypes), (b, b_dtypes)],
    );
}

// DEPRECATED: prefer the variant with out_dtypes in the template argument list.
pub fn apply_tritensor_elementwise_fn_deprecated<CTYPE_COMPUTE, Op>(
    compute_fun: Op,
    ctx: &mut KernelRuntimeContext,
    a: &Tensor,
    a_dtypes: SupportedTensorDtypes,
    b: &Tensor,
    b_dtypes: SupportedTensorDtypes,
    c: &Tensor,
    c_dtypes: SupportedTensorDtypes,
    out: &Tensor,
    out_dtypes: SupportedTensorDtypes,
) where
    CTYPE_COMPUTE: ComputeCast + CppTypeToScalarType + StaticCast<CTYPE_COMPUTE> + 'static,
    Op: Fn(&[CTYPE_COMPUTE]) -> CTYPE_COMPUTE,
    u8: StaticCast<CTYPE_COMPUTE>,
    i8: StaticCast<CTYPE_COMPUTE>,
    i16: StaticCast<CTYPE_COMPUTE>,
    i32: StaticCast<CTYPE_COMPUTE>,
    i64: StaticCast<CTYPE_COMPUTE>,
    f32: StaticCast<CTYPE_COMPUTE>,
    f64: StaticCast<CTYPE_COMPUTE>,
    bool: StaticCast<CTYPE_COMPUTE>,
    Half: StaticCast<CTYPE_COMPUTE>,
    BFloat16: StaticCast<CTYPE_COMPUTE>,
{
    internal::apply_elementwise_fn_runtime_out_dtypes::<CTYPE_COMPUTE, Op>(
        &compute_fun,
        ctx,
        out,
        out_dtypes,
        &[(a, a_dtypes), (b, b_dtypes), (c, c_dtypes)],
    );
}

/// Useful for tri-tensor elementwise operators.
// [spec:et:def:elementwise-util.torch.executor.native.utils.internal.apply-tritensor-elementwise-fn-fn]
// [spec:et:sem:elementwise-util.torch.executor.native.utils.internal.apply-tritensor-elementwise-fn-fn]
pub fn apply_tritensor_elementwise_fn<CTYPE_COMPUTE, Op>(
    compute_fun: Op,
    ctx: &mut KernelRuntimeContext,
    a: &Tensor,
    a_dtypes: SupportedTensorDtypes,
    b: &Tensor,
    b_dtypes: SupportedTensorDtypes,
    c: &Tensor,
    c_dtypes: SupportedTensorDtypes,
    out: &Tensor,
    out_dtypes: SupportedTensorDtypes,
    support_noncontiguous: bool,
) where
    CTYPE_COMPUTE: ComputeCast + CppTypeToScalarType + StaticCast<CTYPE_COMPUTE> + 'static,
    Op: Fn(&[CTYPE_COMPUTE]) -> CTYPE_COMPUTE,
    u8: StaticCast<CTYPE_COMPUTE>,
    i8: StaticCast<CTYPE_COMPUTE>,
    i16: StaticCast<CTYPE_COMPUTE>,
    i32: StaticCast<CTYPE_COMPUTE>,
    i64: StaticCast<CTYPE_COMPUTE>,
    f32: StaticCast<CTYPE_COMPUTE>,
    f64: StaticCast<CTYPE_COMPUTE>,
    bool: StaticCast<CTYPE_COMPUTE>,
    Half: StaticCast<CTYPE_COMPUTE>,
    BFloat16: StaticCast<CTYPE_COMPUTE>,
{
    internal::apply_elementwise_fn::<CTYPE_COMPUTE, Op>(
        &compute_fun,
        ctx,
        out,
        out_dtypes,
        support_noncontiguous,
        &[(a, a_dtypes), (b, b_dtypes), (c, c_dtypes)],
    );
}

// [spec:et:def:elementwise-util.torch.executor.native.utils.internal.get-compute-type-fn]
// [spec:et:sem:elementwise-util.torch.executor.native.utils.internal.get-compute-type-fn]
pub fn get_compute_type(common_type: &mut ScalarType) -> ScalarType {
    let mut compute_type = *common_type;
    if *common_type == ScalarType::Half || *common_type == ScalarType::BFloat16 {
        compute_type = ScalarType::Float;
    }
    compute_type
}

// PORT-NOTE: exported so op files can reference the SFINAE contract helper used
// by the vectorized fast path (see [NOTE: Generic lambdas] in the C++ source).
#[allow(unused_imports)]
use sizes_match_ignoring_leading_1s as _sizes_match_ignoring_leading_1s;

extern crate alloc;

#[cfg(test)]
mod tests {
    use super::internal::{
        SupportNoncontiguousInputTensors, apply_elementwise_fn, apply_elementwise_fn_generic_impl,
        apply_elementwise_fn_runtime_out_dtypes, can_use_vectorized,
        dtype_specialized_elementwise_fn_impl, validate_elementwise_fn_inputs,
    };
    use super::{
        apply_bitensor_elementwise_fn, apply_tritensor_elementwise_fn,
        apply_unitensor_elementwise_fn, get_compute_type,
    };
    use crate::kernels::portable::cpu::util::dtype_util::SupportedTensorDtypes as D;
    use crate::runtime::core::error::Error;
    use crate::runtime::core::exec_aten::testing_util::tensor_factory::TensorFactory;
    use crate::runtime::core::memory_allocator::{MemoryAllocator, MemoryAllocatorBase};
    use crate::runtime::core::portable_type::scalar_type::ScalarType;
    use crate::runtime::core::portable_type::tensor::Tensor;
    use crate::runtime::core::tensor_shape_dynamism::TensorShapeDynamism;
    use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

    fn context() -> KernelRuntimeContext<'static> {
        crate::runtime::platform::runtime::runtime_init();
        KernelRuntimeContext::new(
            crate::extension::module::module::null_event_tracer(),
            core::ptr::null_mut::<MemoryAllocator>() as *mut dyn MemoryAllocatorBase,
        )
    }

    fn f32t(tf: &TensorFactory<f32>, data: Vec<f32>) -> Tensor<'_> {
        let n = data.len() as i32;
        tf.make(vec![n], data, vec![], TensorShapeDynamism::STATIC)
    }

    fn out_of(tf: &TensorFactory<f32>, n: usize) -> Tensor<'_> {
        tf.make(
            vec![n as i32],
            vec![0.0; n],
            vec![],
            TensorShapeDynamism::STATIC,
        )
    }

    fn read(t: &Tensor) -> Vec<f32> {
        let ptr = t.const_data_ptr::<f32>();
        (0..t.numel())
            .map(|i| unsafe { *ptr.add(i as usize) })
            .collect()
    }

    // get_compute_type widens Half/BFloat16 to Float, passes others through.
    // [spec:et:sem:elementwise-util.torch.executor.native.utils.internal.get-compute-type-fn/test]
    #[test]
    fn elementwise_get_compute_type() {
        let mut half = ScalarType::Half;
        assert_eq!(get_compute_type(&mut half), ScalarType::Float);
        assert_eq!(half, ScalarType::Half); // does not mutate the argument

        let mut bf = ScalarType::BFloat16;
        assert_eq!(get_compute_type(&mut bf), ScalarType::Float);

        let mut i = ScalarType::Int;
        assert_eq!(get_compute_type(&mut i), ScalarType::Int);

        let mut f = ScalarType::Float;
        assert_eq!(get_compute_type(&mut f), ScalarType::Float);

        let mut d = ScalarType::Double;
        assert_eq!(get_compute_type(&mut d), ScalarType::Double);
    }

    // can_use_vectorized is a constant false in the non-PyTorch-header port.
    // [spec:et:sem:elementwise-util.torch.executor.native.utils.internal.can-use-vectorized-fn/test]
    #[test]
    fn elementwise_can_use_vectorized_is_false() {
        assert!(!can_use_vectorized());
    }

    // The SupportNoncontiguousInputTensors tag constructs to an empty value.
    // [spec:et:sem:elementwise-util.torch.executor.native.utils.internal.support-noncontiguous-input-tensors.support-noncontiguous-input-tensors-fn/test]
    #[test]
    fn elementwise_support_noncontiguous_tag_constructs() {
        let _tag = SupportNoncontiguousInputTensors;
    }

    // validate_elementwise_fn_inputs: passes when all dtypes fit, fails ctx
    // with InvalidArgument otherwise.
    // [spec:et:sem:elementwise-util.torch.executor.native.utils.internal.validate-elementwise-fn-inputs-fn/test]
    #[test]
    fn elementwise_validate_inputs() {
        let tf = TensorFactory::<f32>::new();
        let a = f32t(&tf, vec![1.0]);
        let out = out_of(&tf, 1);

        let mut ctx = context();
        let ok = validate_elementwise_fn_inputs::<f32>(
            &mut ctx,
            &out,
            D::REALHBBF16,
            &[(&a, D::REALHBBF16)],
        );
        assert!(ok);
        assert_eq!(ctx.failure_state(), Error::Ok);

        // f32 input but declared BOOL -> check_tensor_dtype fails -> false + ctx err
        let mut ctx2 = context();
        let bad =
            validate_elementwise_fn_inputs::<f32>(&mut ctx2, &out, D::REALHBBF16, &[(&a, D::BOOL)]);
        assert!(!bad);
        assert_eq!(ctx2.failure_state(), Error::InvalidArgument);
    }

    // apply_unitensor_elementwise_fn runs the full load/compute/store pipeline
    // for one input, honoring broadcasting-free contiguous iteration.
    // [spec:et:sem:elementwise-util.torch.executor.native.utils.internal.apply-unitensor-elementwise-fn-fn/test]
    // also exercises apply_elementwise_fn + apply_elementwise_fn_generic_impl.
    #[test]
    fn elementwise_apply_unitensor() {
        let tf = TensorFactory::<f32>::new();
        let a = f32t(&tf, vec![1.0, 2.0, 3.0]);
        let out = out_of(&tf, 3);
        let mut ctx = context();
        apply_unitensor_elementwise_fn::<f32, _>(
            |v: &[f32]| v[0] * 2.0,
            &mut ctx,
            &a,
            D::REALHBBF16,
            &out,
            D::REALHBBF16,
            false,
        );
        assert_eq!(ctx.failure_state(), Error::Ok);
        assert_eq!(read(&out), vec![2.0, 4.0, 6.0]);
    }

    // apply_bitensor_elementwise_fn: two inputs, elementwise combine.
    // [spec:et:sem:elementwise-util.torch.executor.native.utils.internal.apply-bitensor-elementwise-fn-fn/test]
    #[test]
    fn elementwise_apply_bitensor() {
        let tf = TensorFactory::<f32>::new();
        let a = f32t(&tf, vec![1.0, 2.0, 3.0]);
        let b = f32t(&tf, vec![10.0, 20.0, 30.0]);
        let out = out_of(&tf, 3);
        let mut ctx = context();
        apply_bitensor_elementwise_fn::<f32, _>(
            |v: &[f32]| v[0] + v[1],
            &mut ctx,
            &a,
            D::REALHBBF16,
            &b,
            D::REALHBBF16,
            &out,
            D::REALHBBF16,
            false,
        );
        assert_eq!(ctx.failure_state(), Error::Ok);
        assert_eq!(read(&out), vec![11.0, 22.0, 33.0]);
    }

    // apply_bitensor with broadcasting: scalar b broadcasts across a.
    #[test]
    fn elementwise_apply_bitensor_broadcast() {
        let tf = TensorFactory::<f32>::new();
        let a = f32t(&tf, vec![1.0, 2.0, 3.0]);
        let b = f32t(&tf, vec![5.0]);
        let out = out_of(&tf, 3);
        let mut ctx = context();
        apply_bitensor_elementwise_fn::<f32, _>(
            |v: &[f32]| v[0] + v[1],
            &mut ctx,
            &a,
            D::REALHBBF16,
            &b,
            D::REALHBBF16,
            &out,
            D::REALHBBF16,
            false,
        );
        assert_eq!(read(&out), vec![6.0, 7.0, 8.0]);
    }

    // apply_tritensor_elementwise_fn: three inputs.
    // [spec:et:sem:elementwise-util.torch.executor.native.utils.internal.apply-tritensor-elementwise-fn-fn/test]
    #[test]
    fn elementwise_apply_tritensor() {
        let tf = TensorFactory::<f32>::new();
        let a = f32t(&tf, vec![1.0, 2.0]);
        let b = f32t(&tf, vec![3.0, 4.0]);
        let c = f32t(&tf, vec![5.0, 6.0]);
        let out = out_of(&tf, 2);
        let mut ctx = context();
        apply_tritensor_elementwise_fn::<f32, _>(
            |v: &[f32]| v[0] + v[1] * v[2],
            &mut ctx,
            &a,
            D::REALHBBF16,
            &b,
            D::REALHBBF16,
            &c,
            D::REALHBBF16,
            &out,
            D::REALHBBF16,
            false,
        );
        assert_eq!(ctx.failure_state(), Error::Ok);
        assert_eq!(read(&out), vec![1.0 + 3.0 * 5.0, 2.0 + 4.0 * 6.0]);
    }

    // apply_elementwise_fn returns early (writes nothing) when validation fails.
    // [spec:et:sem:elementwise-util.torch.executor.native.utils.internal.apply-elementwise-fn-fn/test]
    #[test]
    fn elementwise_apply_elementwise_fn_validation_short_circuits() {
        let tf = TensorFactory::<f32>::new();
        let a = f32t(&tf, vec![1.0, 2.0]);
        let out = out_of(&tf, 2);
        let mut ctx = context();
        // declare the f32 input as BOOL -> validation fails, out stays zero
        apply_elementwise_fn::<f32, _>(
            &(|v: &[f32]| v[0] + 100.0),
            &mut ctx,
            &out,
            D::REALHBBF16,
            false,
            &[(&a, D::BOOL)],
        );
        assert_eq!(ctx.failure_state(), Error::InvalidArgument);
        assert_eq!(read(&out), vec![0.0, 0.0]);

        // valid call writes results
        let mut ctx2 = context();
        apply_elementwise_fn::<f32, _>(
            &(|v: &[f32]| v[0] + 100.0),
            &mut ctx2,
            &out,
            D::REALHBBF16,
            false,
            &[(&a, D::REALHBBF16)],
        );
        assert_eq!(read(&out), vec![101.0, 102.0]);
    }

    // apply_elementwise_fn_generic_impl: the core loop, called directly.
    // [spec:et:sem:elementwise-util.torch.executor.native.utils.internal.apply-elementwise-fn-generic-impl-fn/test]
    #[test]
    fn elementwise_apply_generic_impl() {
        let tf = TensorFactory::<f32>::new();
        let a = f32t(&tf, vec![2.0, 4.0, 6.0]);
        let out = out_of(&tf, 3);
        let mut ctx = context();
        apply_elementwise_fn_generic_impl::<f32, _>(
            &(|v: &[f32]| v[0] / 2.0),
            &mut ctx,
            &out,
            D::REALHBBF16,
            &[(&a, D::REALHBBF16)],
            false,
        );
        assert_eq!(read(&out), vec![1.0, 2.0, 3.0]);
    }

    // apply_elementwise_fn_runtime_out_dtypes: deprecated variant, contiguous.
    // [spec:et:sem:elementwise-util.torch.executor.native.utils.internal.apply-elementwise-fn-runtime-out-dtypes-fn/test]
    #[test]
    fn elementwise_apply_runtime_out_dtypes() {
        let tf = TensorFactory::<f32>::new();
        let a = f32t(&tf, vec![1.0, 2.0]);
        let b = f32t(&tf, vec![3.0, 4.0]);
        let out = out_of(&tf, 2);
        let mut ctx = context();
        apply_elementwise_fn_runtime_out_dtypes::<f32, _>(
            &(|v: &[f32]| v[0] - v[1]),
            &mut ctx,
            &out,
            D::REALHBBF16,
            &[(&a, D::REALHBBF16), (&b, D::REALHBBF16)],
        );
        assert_eq!(ctx.failure_state(), Error::Ok);
        assert_eq!(read(&out), vec![-2.0, -2.0]);

        // validation failure short-circuits, out untouched
        let out2 = out_of(&tf, 2);
        let mut ctx2 = context();
        apply_elementwise_fn_runtime_out_dtypes::<f32, _>(
            &(|v: &[f32]| v[0] - v[1]),
            &mut ctx2,
            &out2,
            D::REALHBBF16,
            &[(&a, D::BOOL), (&b, D::REALHBBF16)],
        );
        assert_eq!(ctx2.failure_state(), Error::InvalidArgument);
        assert_eq!(read(&out2), vec![0.0, 0.0]);
    }

    // dtype_specialized_elementwise_fn_impl: fast path reading inputs directly
    // as CTYPE_COMPUTE and writing directly as CTYPE_OUT (no conversion).
    // [spec:et:sem:elementwise-util.torch.executor.native.utils.internal.dtype-specialized-elementwise-fn-impl-fn/test]
    #[test]
    fn elementwise_dtype_specialized_impl() {
        let tf = TensorFactory::<f32>::new();
        let a = f32t(&tf, vec![1.0, 2.0, 3.0]);
        let b = f32t(&tf, vec![4.0, 5.0, 6.0]);
        let out = out_of(&tf, 3);
        let mut ctx = context();
        dtype_specialized_elementwise_fn_impl::<f32, f32, _>(
            &(|v: &[f32]| v[0] * v[1]),
            &mut ctx,
            &out,
            &[(&a, D::REALHBBF16), (&b, D::REALHBBF16)],
            false,
        );
        assert_eq!(read(&out), vec![4.0, 10.0, 18.0]);
    }
}
