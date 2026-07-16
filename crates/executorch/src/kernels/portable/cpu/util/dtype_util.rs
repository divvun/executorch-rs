//! Literal port of kernels/portable/cpu/util/dtype_util.cpp + kernels/portable/cpu/util/dtype_util.h.

use crate::runtime::core::exec_aten::util::tensor_util::{
    tensor_is_floating_type, tensor_is_integral_type, tensor_is_realhbbf16_type,
    tensor_is_realhbf16_type, tensor_is_type1, tensor_is_type2,
};
use crate::runtime::core::portable_type::scalar_type::ScalarType;
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

// [spec:et:def:dtype-util.torch.executor.native.utils.supported-tensor-dtypes]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum SupportedTensorDtypes {
    REALHBBF16,
    REALHBF16,
    FLOATHBF16,
    INTB,
    BOOL,
    BOOL_OR_BYTE,
    // DEPRECATED: not likely to be correct; use SAME_AS_COMMON.
    SAME_AS_COMPUTE,
    SAME_AS_COMMON,
}

pub mod internal {
    use super::*;

    // PORT-NOTE: models C++ builtin `static_cast<To>(From)` for the ET scalar C
    // types. Half/BFloat16 have no direct integer conversions in C++ (they only
    // construct from / convert to float), so those routes go through `f32`,
    // mirroring `static_cast<Half>(int)` == `Half(static_cast<float>(int))`.
    pub trait StaticCast<From> {
        fn static_cast(v: From) -> Self;
    }

    macro_rules! impl_static_cast_as {
        ($to:ty, $from:ty) => {
            impl StaticCast<$from> for $to {
                #[inline]
                fn static_cast(v: $from) -> Self {
                    v as $to
                }
            }
        };
    }

    // Primitive numeric <-> primitive numeric (`as`-cast, matches static_cast).
    macro_rules! impl_static_cast_prim_row {
        ($to:ty) => {
            impl_static_cast_as!($to, u8);
            impl_static_cast_as!($to, i8);
            impl_static_cast_as!($to, i16);
            impl_static_cast_as!($to, i32);
            impl_static_cast_as!($to, i64);
            impl_static_cast_as!($to, f32);
            impl_static_cast_as!($to, f64);
        };
    }
    impl_static_cast_prim_row!(u8);
    impl_static_cast_prim_row!(i8);
    impl_static_cast_prim_row!(i16);
    impl_static_cast_prim_row!(i32);
    impl_static_cast_prim_row!(i64);
    impl_static_cast_prim_row!(f32);
    impl_static_cast_prim_row!(f64);

    // bool source: nonzero -> true / zero -> false.
    macro_rules! impl_static_cast_from_bool {
        ($to:ty) => {
            impl StaticCast<bool> for $to {
                #[inline]
                fn static_cast(v: bool) -> Self {
                    (v as u8) as $to
                }
            }
        };
    }
    impl_static_cast_from_bool!(u8);
    impl_static_cast_from_bool!(i8);
    impl_static_cast_from_bool!(i16);
    impl_static_cast_from_bool!(i32);
    impl_static_cast_from_bool!(i64);
    impl_static_cast_from_bool!(f32);
    impl_static_cast_from_bool!(f64);
    impl StaticCast<bool> for bool {
        #[inline]
        fn static_cast(v: bool) -> Self {
            v
        }
    }

    // bool destination: nonzero -> true.
    macro_rules! impl_static_cast_to_bool {
        ($from:ty) => {
            impl StaticCast<$from> for bool {
                #[inline]
                fn static_cast(v: $from) -> Self {
                    v != (0 as $from)
                }
            }
        };
    }
    impl_static_cast_to_bool!(u8);
    impl_static_cast_to_bool!(i8);
    impl_static_cast_to_bool!(i16);
    impl_static_cast_to_bool!(i32);
    impl_static_cast_to_bool!(i64);
    impl_static_cast_to_bool!(f32);
    impl_static_cast_to_bool!(f64);

    use crate::runtime::core::portable_type::{BFloat16, Half};

    // Half / BFloat16 conversions route through f32 (mirrors C++ where these
    // types only have float constructors / conversions).
    macro_rules! impl_static_cast_half_like {
        ($half:ty) => {
            // primitive -> half via f32
            impl StaticCast<u8> for $half {
                #[inline]
                fn static_cast(v: u8) -> Self {
                    <$half>::from_f32(v as f32)
                }
            }
            impl StaticCast<i8> for $half {
                #[inline]
                fn static_cast(v: i8) -> Self {
                    <$half>::from_f32(v as f32)
                }
            }
            impl StaticCast<i16> for $half {
                #[inline]
                fn static_cast(v: i16) -> Self {
                    <$half>::from_f32(v as f32)
                }
            }
            impl StaticCast<i32> for $half {
                #[inline]
                fn static_cast(v: i32) -> Self {
                    <$half>::from_f32(v as f32)
                }
            }
            impl StaticCast<i64> for $half {
                #[inline]
                fn static_cast(v: i64) -> Self {
                    <$half>::from_f32(v as f32)
                }
            }
            impl StaticCast<f32> for $half {
                #[inline]
                fn static_cast(v: f32) -> Self {
                    <$half>::from_f32(v)
                }
            }
            impl StaticCast<f64> for $half {
                #[inline]
                fn static_cast(v: f64) -> Self {
                    <$half>::from_f64(v)
                }
            }
            impl StaticCast<bool> for $half {
                #[inline]
                fn static_cast(v: bool) -> Self {
                    <$half>::from_f32((v as u8) as f32)
                }
            }
            // half -> primitive via f32
            impl StaticCast<$half> for u8 {
                #[inline]
                fn static_cast(v: $half) -> Self {
                    v.to_f32() as u8
                }
            }
            impl StaticCast<$half> for i8 {
                #[inline]
                fn static_cast(v: $half) -> Self {
                    v.to_f32() as i8
                }
            }
            impl StaticCast<$half> for i16 {
                #[inline]
                fn static_cast(v: $half) -> Self {
                    v.to_f32() as i16
                }
            }
            impl StaticCast<$half> for i32 {
                #[inline]
                fn static_cast(v: $half) -> Self {
                    v.to_f32() as i32
                }
            }
            impl StaticCast<$half> for i64 {
                #[inline]
                fn static_cast(v: $half) -> Self {
                    v.to_f32() as i64
                }
            }
            impl StaticCast<$half> for f32 {
                #[inline]
                fn static_cast(v: $half) -> Self {
                    v.to_f32()
                }
            }
            impl StaticCast<$half> for f64 {
                #[inline]
                fn static_cast(v: $half) -> Self {
                    v.to_f64()
                }
            }
            impl StaticCast<$half> for bool {
                #[inline]
                fn static_cast(v: $half) -> Self {
                    v.to_f32() != 0.0
                }
            }
        };
    }
    impl_static_cast_half_like!(Half);
    impl_static_cast_half_like!(BFloat16);

    // Half <-> BFloat16 and identity, via f32.
    impl StaticCast<Half> for Half {
        #[inline]
        fn static_cast(v: Half) -> Self {
            v
        }
    }
    impl StaticCast<BFloat16> for BFloat16 {
        #[inline]
        fn static_cast(v: BFloat16) -> Self {
            v
        }
    }
    impl StaticCast<BFloat16> for Half {
        #[inline]
        fn static_cast(v: BFloat16) -> Self {
            Half::from_f32(v.to_f32())
        }
    }
    impl StaticCast<Half> for BFloat16 {
        #[inline]
        fn static_cast(v: Half) -> Self {
            BFloat16::from_f32(v.to_f32())
        }
    }

    // [spec:et:def:dtype-util.torch.executor.native.utils.internal.load-and-convert-fn]
    // [spec:et:sem:dtype-util.torch.executor.native.utils.internal.load-and-convert-fn]
    pub fn load_and_convert<To, From>(from_ptr: *const core::ffi::c_void) -> To
    where
        To: StaticCast<From>,
        From: Copy,
    {
        To::static_cast(unsafe { *(from_ptr as *const From) })
    }

    // [spec:et:def:dtype-util.torch.executor.native.utils.internal.convert-and-store-fn]
    // [spec:et:sem:dtype-util.torch.executor.native.utils.internal.convert-and-store-fn]
    pub fn convert_and_store<To, From>(f: From, dst: *mut core::ffi::c_void)
    where
        To: StaticCast<From>,
    {
        unsafe {
            *(dst as *mut To) = To::static_cast(f);
        }
    }

    pub type LoadToComputeFn<CTYPE_COMPUTE> = fn(*const core::ffi::c_void) -> CTYPE_COMPUTE;
    pub type StoreComputeToTensorFn<CTYPE_COMPUTE> = fn(CTYPE_COMPUTE, *mut core::ffi::c_void);

    // PORT-NOTE: the C++ getters are templated on `<CTYPE_COMPUTE, op_name>` and
    // build a fn-pointer table via ET_SWITCH. The Rust equivalent needs
    // `CTYPE_COMPUTE: StaticCast<TENSOR_CTYPE>` for every branch's element type,
    // which we bound with the `ComputeCast` alias below (one bound covering all
    // dtypes the switch can pick).
    pub trait ComputeCast:
        StaticCast<u8>
        + StaticCast<i8>
        + StaticCast<i16>
        + StaticCast<i32>
        + StaticCast<i64>
        + StaticCast<f32>
        + StaticCast<f64>
        + StaticCast<bool>
        + StaticCast<Half>
        + StaticCast<BFloat16>
        + StaticCast<Self>
        + Copy
    {
    }
    impl<T> ComputeCast for T where
        T: StaticCast<u8>
            + StaticCast<i8>
            + StaticCast<i16>
            + StaticCast<i32>
            + StaticCast<i64>
            + StaticCast<f32>
            + StaticCast<f64>
            + StaticCast<bool>
            + StaticCast<Half>
            + StaticCast<BFloat16>
            + StaticCast<T>
            + Copy
    {
    }

    // [spec:et:def:dtype-util.torch.executor.native.utils.internal.get-load-to-compute-fn-realhbbf16-fn]
    // [spec:et:sem:dtype-util.torch.executor.native.utils.internal.get-load-to-compute-fn-realhbbf16-fn]
    pub fn get_load_to_compute_fn_realhbbf16<CTYPE_COMPUTE: ComputeCast + 'static>(
        context: &mut KernelRuntimeContext,
        t: &Tensor,
        op_name: &str,
    ) -> LoadToComputeFn<CTYPE_COMPUTE> {
        let mut result: LoadToComputeFn<CTYPE_COMPUTE> = null_load();
        crate::et_switch_realhbbf16_types!(t.scalar_type(), context, op_name, TENSOR_CTYPE, {
            result = load_and_convert::<CTYPE_COMPUTE, TENSOR_CTYPE>;
        });
        result
    }

    // [spec:et:def:dtype-util.torch.executor.native.utils.internal.get-load-to-compute-fn-realhbf16-fn]
    // [spec:et:sem:dtype-util.torch.executor.native.utils.internal.get-load-to-compute-fn-realhbf16-fn]
    pub fn get_load_to_compute_fn_realhbf16<CTYPE_COMPUTE: ComputeCast + 'static>(
        context: &mut KernelRuntimeContext,
        t: &Tensor,
        op_name: &str,
    ) -> LoadToComputeFn<CTYPE_COMPUTE> {
        let mut result: LoadToComputeFn<CTYPE_COMPUTE> = null_load();
        crate::et_switch_realhbf16_types!(t.scalar_type(), context, op_name, TENSOR_CTYPE, {
            result = load_and_convert::<CTYPE_COMPUTE, TENSOR_CTYPE>;
        });
        result
    }

    // [spec:et:def:dtype-util.torch.executor.native.utils.internal.get-load-to-compute-fn-floathbf16-fn]
    // [spec:et:sem:dtype-util.torch.executor.native.utils.internal.get-load-to-compute-fn-floathbf16-fn]
    pub fn get_load_to_compute_fn_floathbf16<CTYPE_COMPUTE: ComputeCast + 'static>(
        context: &mut KernelRuntimeContext,
        t: &Tensor,
        op_name: &str,
    ) -> LoadToComputeFn<CTYPE_COMPUTE> {
        let mut result: LoadToComputeFn<CTYPE_COMPUTE> = null_load();
        crate::et_switch_floathbf16_types!(t.scalar_type(), context, op_name, TENSOR_CTYPE, {
            result = load_and_convert::<CTYPE_COMPUTE, TENSOR_CTYPE>;
        });
        result
    }

    // [spec:et:def:dtype-util.torch.executor.native.utils.internal.get-load-to-compute-fn-intb-fn]
    // [spec:et:sem:dtype-util.torch.executor.native.utils.internal.get-load-to-compute-fn-intb-fn]
    pub fn get_load_to_compute_fn_intb<CTYPE_COMPUTE: ComputeCast + 'static>(
        context: &mut KernelRuntimeContext,
        t: &Tensor,
        op_name: &str,
    ) -> LoadToComputeFn<CTYPE_COMPUTE> {
        let mut result: LoadToComputeFn<CTYPE_COMPUTE> = null_load();
        crate::et_switch_int_types_and!(Bool, t.scalar_type(), context, op_name, TENSOR_CTYPE, {
            result = load_and_convert::<CTYPE_COMPUTE, TENSOR_CTYPE>;
        });
        result
    }

    // [spec:et:def:dtype-util.torch.executor.native.utils.internal.get-load-to-compute-fn-bool-fn]
    // [spec:et:sem:dtype-util.torch.executor.native.utils.internal.get-load-to-compute-fn-bool-fn]
    pub fn get_load_to_compute_fn_bool<CTYPE_COMPUTE: ComputeCast + 'static>(
        context: &mut KernelRuntimeContext,
        t: &Tensor,
        op_name: &str,
    ) -> LoadToComputeFn<CTYPE_COMPUTE> {
        let mut result: LoadToComputeFn<CTYPE_COMPUTE> = null_load();
        if t.scalar_type() != ScalarType::Bool {
            context.fail(crate::runtime::core::error::Error::InvalidArgument);
            crate::et_log!(
                Error,
                "Unhandled dtype {} for {}",
                crate::runtime::core::exec_aten::util::scalar_type_util::to_string(t.scalar_type()),
                op_name
            );
        } else {
            result = load_and_convert::<CTYPE_COMPUTE, bool>;
        }
        result
    }

    // [spec:et:def:dtype-util.torch.executor.native.utils.internal.get-load-to-compute-fn-bool-or-byte-fn]
    // [spec:et:sem:dtype-util.torch.executor.native.utils.internal.get-load-to-compute-fn-bool-or-byte-fn]
    pub fn get_load_to_compute_fn_bool_or_byte<CTYPE_COMPUTE: ComputeCast + 'static>(
        context: &mut KernelRuntimeContext,
        t: &Tensor,
        op_name: &str,
    ) -> LoadToComputeFn<CTYPE_COMPUTE> {
        let mut result: LoadToComputeFn<CTYPE_COMPUTE> = null_load();
        crate::et_switch_two_types!(
            Bool,
            Byte,
            t.scalar_type(),
            context,
            op_name,
            TENSOR_CTYPE,
            {
                result = load_and_convert::<CTYPE_COMPUTE, TENSOR_CTYPE>;
            }
        );
        result
    }

    // [spec:et:def:dtype-util.torch.executor.native.utils.internal.get-load-to-compute-fn-same-as-compute-fn]
    // [spec:et:sem:dtype-util.torch.executor.native.utils.internal.get-load-to-compute-fn-same-as-compute-fn]
    pub fn get_load_to_compute_fn_same_as_compute<
        CTYPE_COMPUTE: ComputeCast
            + crate::runtime::core::exec_aten::util::scalar_type_util::CppTypeToScalarType
            + 'static,
    >(
        context: &mut KernelRuntimeContext,
        t: &Tensor,
        op_name: &str,
    ) -> LoadToComputeFn<CTYPE_COMPUTE> {
        let mut result: LoadToComputeFn<CTYPE_COMPUTE> = null_load();
        let common_scalar_type = CTYPE_COMPUTE::VALUE;
        if t.scalar_type() != common_scalar_type {
            context.fail(crate::runtime::core::error::Error::InvalidArgument);
            crate::et_log!(
                Error,
                "Unhandled dtype {} for {}",
                crate::runtime::core::exec_aten::util::scalar_type_util::to_string(t.scalar_type()),
                op_name
            );
        } else {
            result = load_and_convert::<CTYPE_COMPUTE, CTYPE_COMPUTE>;
        }
        result
    }

    // [spec:et:def:dtype-util.torch.executor.native.utils.internal.get-load-to-compute-fn-same-as-common-fn]
    // [spec:et:sem:dtype-util.torch.executor.native.utils.internal.get-load-to-compute-fn-same-as-common-fn]
    //
    // PORT-NOTE: C++ SFINAE-splits on `CTYPE_COMPUTE == float`. Rust cannot
    // dispatch on the concrete type param without specialization; instead we
    // check `CTYPE_COMPUTE::VALUE == Float` at runtime and mirror both overloads.
    // The Float branch dispatches over {Float, Half, BFloat16}; the non-Float
    // branch delegates to same_as_compute.
    pub fn get_load_to_compute_fn_same_as_common<
        CTYPE_COMPUTE: ComputeCast
            + crate::runtime::core::exec_aten::util::scalar_type_util::CppTypeToScalarType
            + 'static,
    >(
        context: &mut KernelRuntimeContext,
        t: &Tensor,
        op_name: &str,
    ) -> LoadToComputeFn<CTYPE_COMPUTE> {
        if CTYPE_COMPUTE::VALUE == ScalarType::Float {
            let mut result: LoadToComputeFn<CTYPE_COMPUTE> = null_load();
            crate::et_switch_three_types!(
                Float,
                Half,
                BFloat16,
                t.scalar_type(),
                context,
                op_name,
                T,
                {
                    result = load_and_convert::<CTYPE_COMPUTE, T>;
                }
            );
            result
        } else {
            get_load_to_compute_fn_same_as_compute::<CTYPE_COMPUTE>(context, t, op_name)
        }
    }

    // [spec:et:def:dtype-util.torch.executor.native.utils.internal.get-store-compute-to-tensor-fn-realhbbf16-fn]
    // [spec:et:sem:dtype-util.torch.executor.native.utils.internal.get-store-compute-to-tensor-fn-realhbbf16-fn]
    pub fn get_store_compute_to_tensor_fn_realhbbf16<CTYPE_COMPUTE: Copy + 'static>(
        context: &mut KernelRuntimeContext,
        t: &Tensor,
        op_name: &str,
    ) -> StoreComputeToTensorFn<CTYPE_COMPUTE>
    where
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
        let mut result: StoreComputeToTensorFn<CTYPE_COMPUTE> = null_store();
        crate::et_switch_realhbbf16_types!(t.scalar_type(), context, op_name, TENSOR_CTYPE, {
            result = convert_and_store::<TENSOR_CTYPE, CTYPE_COMPUTE>;
        });
        result
    }

    // [spec:et:def:dtype-util.torch.executor.native.utils.internal.get-store-compute-to-tensor-fn-realhbf16-fn]
    // [spec:et:sem:dtype-util.torch.executor.native.utils.internal.get-store-compute-to-tensor-fn-realhbf16-fn]
    pub fn get_store_compute_to_tensor_fn_realhbf16<CTYPE_COMPUTE: Copy + 'static>(
        context: &mut KernelRuntimeContext,
        t: &Tensor,
        op_name: &str,
    ) -> StoreComputeToTensorFn<CTYPE_COMPUTE>
    where
        u8: StaticCast<CTYPE_COMPUTE>,
        i8: StaticCast<CTYPE_COMPUTE>,
        i16: StaticCast<CTYPE_COMPUTE>,
        i32: StaticCast<CTYPE_COMPUTE>,
        i64: StaticCast<CTYPE_COMPUTE>,
        f32: StaticCast<CTYPE_COMPUTE>,
        f64: StaticCast<CTYPE_COMPUTE>,
        Half: StaticCast<CTYPE_COMPUTE>,
        BFloat16: StaticCast<CTYPE_COMPUTE>,
    {
        let mut result: StoreComputeToTensorFn<CTYPE_COMPUTE> = null_store();
        crate::et_switch_realhbf16_types!(t.scalar_type(), context, op_name, TENSOR_CTYPE, {
            result = convert_and_store::<TENSOR_CTYPE, CTYPE_COMPUTE>;
        });
        result
    }

    // [spec:et:def:dtype-util.torch.executor.native.utils.internal.get-store-compute-to-tensor-fn-floathbf16-fn]
    // [spec:et:sem:dtype-util.torch.executor.native.utils.internal.get-store-compute-to-tensor-fn-floathbf16-fn]
    pub fn get_store_compute_to_tensor_fn_floathbf16<CTYPE_COMPUTE: Copy + 'static>(
        context: &mut KernelRuntimeContext,
        t: &Tensor,
        op_name: &str,
    ) -> StoreComputeToTensorFn<CTYPE_COMPUTE>
    where
        f32: StaticCast<CTYPE_COMPUTE>,
        f64: StaticCast<CTYPE_COMPUTE>,
        Half: StaticCast<CTYPE_COMPUTE>,
        BFloat16: StaticCast<CTYPE_COMPUTE>,
    {
        let mut result: StoreComputeToTensorFn<CTYPE_COMPUTE> = null_store();
        crate::et_switch_floathbf16_types!(t.scalar_type(), context, op_name, TENSOR_CTYPE, {
            result = convert_and_store::<TENSOR_CTYPE, CTYPE_COMPUTE>;
        });
        result
    }

    // [spec:et:def:dtype-util.torch.executor.native.utils.internal.get-store-compute-to-tensor-fn-intb-fn]
    // [spec:et:sem:dtype-util.torch.executor.native.utils.internal.get-store-compute-to-tensor-fn-intb-fn]
    pub fn get_store_compute_to_tensor_fn_intb<CTYPE_COMPUTE: Copy + 'static>(
        context: &mut KernelRuntimeContext,
        t: &Tensor,
        op_name: &str,
    ) -> StoreComputeToTensorFn<CTYPE_COMPUTE>
    where
        u8: StaticCast<CTYPE_COMPUTE>,
        i8: StaticCast<CTYPE_COMPUTE>,
        i16: StaticCast<CTYPE_COMPUTE>,
        i32: StaticCast<CTYPE_COMPUTE>,
        i64: StaticCast<CTYPE_COMPUTE>,
        bool: StaticCast<CTYPE_COMPUTE>,
    {
        let mut result: StoreComputeToTensorFn<CTYPE_COMPUTE> = null_store();
        crate::et_switch_int_types_and!(Bool, t.scalar_type(), context, op_name, TENSOR_CTYPE, {
            result = convert_and_store::<TENSOR_CTYPE, CTYPE_COMPUTE>;
        });
        result
    }

    // [spec:et:def:dtype-util.torch.executor.native.utils.internal.get-store-compute-to-tensor-fn-bool-fn]
    // [spec:et:sem:dtype-util.torch.executor.native.utils.internal.get-store-compute-to-tensor-fn-bool-fn]
    pub fn get_store_compute_to_tensor_fn_bool<CTYPE_COMPUTE: Copy + 'static>(
        context: &mut KernelRuntimeContext,
        t: &Tensor,
        op_name: &str,
    ) -> StoreComputeToTensorFn<CTYPE_COMPUTE>
    where
        bool: StaticCast<CTYPE_COMPUTE>,
    {
        let mut result: StoreComputeToTensorFn<CTYPE_COMPUTE> = null_store();
        if t.scalar_type() != ScalarType::Bool {
            context.fail(crate::runtime::core::error::Error::InvalidArgument);
            crate::et_log!(
                Error,
                "Unhandled dtype {} for {}",
                crate::runtime::core::exec_aten::util::scalar_type_util::to_string(t.scalar_type()),
                op_name
            );
        } else {
            result = convert_and_store::<bool, CTYPE_COMPUTE>;
        }
        result
    }

    // [spec:et:def:dtype-util.torch.executor.native.utils.internal.get-store-compute-to-tensor-fn-bool-or-byte-fn]
    // [spec:et:sem:dtype-util.torch.executor.native.utils.internal.get-store-compute-to-tensor-fn-bool-or-byte-fn]
    pub fn get_store_compute_to_tensor_fn_bool_or_byte<CTYPE_COMPUTE: Copy + 'static>(
        context: &mut KernelRuntimeContext,
        t: &Tensor,
        op_name: &str,
    ) -> StoreComputeToTensorFn<CTYPE_COMPUTE>
    where
        bool: StaticCast<CTYPE_COMPUTE>,
        u8: StaticCast<CTYPE_COMPUTE>,
    {
        let mut result: StoreComputeToTensorFn<CTYPE_COMPUTE> = null_store();
        crate::et_switch_two_types!(
            Bool,
            Byte,
            t.scalar_type(),
            context,
            op_name,
            TENSOR_CTYPE,
            {
                result = convert_and_store::<TENSOR_CTYPE, CTYPE_COMPUTE>;
            }
        );
        result
    }

    // [spec:et:def:dtype-util.torch.executor.native.utils.internal.get-store-compute-to-tensor-fn-same-as-compute-fn]
    // [spec:et:sem:dtype-util.torch.executor.native.utils.internal.get-store-compute-to-tensor-fn-same-as-compute-fn]
    pub fn get_store_compute_to_tensor_fn_same_as_compute<
        CTYPE_COMPUTE: Copy
            + StaticCast<CTYPE_COMPUTE>
            + crate::runtime::core::exec_aten::util::scalar_type_util::CppTypeToScalarType
            + 'static,
    >(
        context: &mut KernelRuntimeContext,
        t: &Tensor,
        op_name: &str,
    ) -> StoreComputeToTensorFn<CTYPE_COMPUTE> {
        let mut result: StoreComputeToTensorFn<CTYPE_COMPUTE> = null_store();
        let common_scalar_type = CTYPE_COMPUTE::VALUE;
        if t.scalar_type() != common_scalar_type {
            context.fail(crate::runtime::core::error::Error::InvalidArgument);
            crate::et_log!(
                Error,
                "Unhandled dtype {} for {}",
                crate::runtime::core::exec_aten::util::scalar_type_util::to_string(t.scalar_type()),
                op_name
            );
        } else {
            result = convert_and_store::<CTYPE_COMPUTE, CTYPE_COMPUTE>;
        }
        result
    }

    // [spec:et:def:dtype-util.torch.executor.native.utils.internal.get-store-compute-to-tensor-fn-same-as-common-fn]
    // [spec:et:sem:dtype-util.torch.executor.native.utils.internal.get-store-compute-to-tensor-fn-same-as-common-fn]
    //
    // PORT-NOTE: as with the load path, C++ SFINAE-splits on `CTYPE_COMPUTE ==
    // float`; mirrored here with a runtime `VALUE == Float` check.
    pub fn get_store_compute_to_tensor_fn_same_as_common<
        CTYPE_COMPUTE: Copy
            + StaticCast<CTYPE_COMPUTE>
            + crate::runtime::core::exec_aten::util::scalar_type_util::CppTypeToScalarType
            + 'static,
    >(
        context: &mut KernelRuntimeContext,
        t: &Tensor,
        op_name: &str,
    ) -> StoreComputeToTensorFn<CTYPE_COMPUTE>
    where
        f32: StaticCast<CTYPE_COMPUTE>,
        Half: StaticCast<CTYPE_COMPUTE>,
        BFloat16: StaticCast<CTYPE_COMPUTE>,
    {
        if CTYPE_COMPUTE::VALUE == ScalarType::Float {
            let mut result: StoreComputeToTensorFn<CTYPE_COMPUTE> = null_store();
            crate::et_switch_three_types!(
                Float,
                Half,
                BFloat16,
                t.scalar_type(),
                context,
                op_name,
                CTYPE,
                {
                    result = convert_and_store::<CTYPE, CTYPE_COMPUTE>;
                }
            );
            result
        } else {
            get_store_compute_to_tensor_fn_same_as_compute::<CTYPE_COMPUTE>(context, t, op_name)
        }
    }

    // PORT-NOTE: the C++ initializes `result = nullptr` (a null function
    // pointer). Rust function pointers are non-nullable, so a null loader/storer
    // is modeled as a sentinel that panics if ever invoked; callers detect the
    // failure via `ctx.failure_state()` exactly as C++ callers check
    // `ctx.fail()` was invoked, so the sentinel is never actually called on the
    // failure path.
    fn null_load_impl<CTYPE_COMPUTE>(_p: *const core::ffi::c_void) -> CTYPE_COMPUTE {
        crate::runtime::platform::abort::runtime_abort();
    }
    fn null_store_impl<CTYPE_COMPUTE>(_v: CTYPE_COMPUTE, _p: *mut core::ffi::c_void) {
        crate::runtime::platform::abort::runtime_abort();
    }
    pub fn null_load<CTYPE_COMPUTE>() -> LoadToComputeFn<CTYPE_COMPUTE> {
        null_load_impl::<CTYPE_COMPUTE>
    }
    pub fn null_store<CTYPE_COMPUTE>() -> StoreComputeToTensorFn<CTYPE_COMPUTE> {
        null_store_impl::<CTYPE_COMPUTE>
    }

    // [spec:et:def:dtype-util.torch.executor.native.utils.internal.get-load-to-compute-fn-impl-fn]
    // [spec:et:sem:dtype-util.torch.executor.native.utils.internal.get-load-to-compute-fn-impl-fn]
    pub fn get_load_to_compute_fn_impl<
        CTYPE_COMPUTE: ComputeCast
            + crate::runtime::core::exec_aten::util::scalar_type_util::CppTypeToScalarType
            + 'static,
    >(
        context: &mut KernelRuntimeContext,
        t: &Tensor,
        dtypes: SupportedTensorDtypes,
        op_name: &str,
    ) -> LoadToComputeFn<CTYPE_COMPUTE> {
        match dtypes {
            SupportedTensorDtypes::REALHBBF16 => {
                get_load_to_compute_fn_realhbbf16::<CTYPE_COMPUTE>(context, t, op_name)
            }
            SupportedTensorDtypes::REALHBF16 => {
                get_load_to_compute_fn_realhbf16::<CTYPE_COMPUTE>(context, t, op_name)
            }
            SupportedTensorDtypes::FLOATHBF16 => {
                get_load_to_compute_fn_realhbf16::<CTYPE_COMPUTE>(context, t, op_name)
            }
            SupportedTensorDtypes::INTB => {
                get_load_to_compute_fn_intb::<CTYPE_COMPUTE>(context, t, op_name)
            }
            SupportedTensorDtypes::BOOL => {
                get_load_to_compute_fn_bool::<CTYPE_COMPUTE>(context, t, op_name)
            }
            SupportedTensorDtypes::BOOL_OR_BYTE => {
                get_load_to_compute_fn_bool_or_byte::<CTYPE_COMPUTE>(context, t, op_name)
            }
            SupportedTensorDtypes::SAME_AS_COMPUTE => {
                get_load_to_compute_fn_same_as_compute::<CTYPE_COMPUTE>(context, t, op_name)
            }
            SupportedTensorDtypes::SAME_AS_COMMON => {
                get_load_to_compute_fn_same_as_common::<CTYPE_COMPUTE>(context, t, op_name)
            }
        }
    }

    // NOTE: applying the #ifdef EXECUTORCH_SELECTIVE_BUILD_DTYPE technique used
    // for get_load_to_compute_fn in this path was a size regression rather than
    // an improvement.
    // [spec:et:def:dtype-util.torch.executor.native.utils.internal.get-store-compute-to-tensor-fn-fn]
    // [spec:et:sem:dtype-util.torch.executor.native.utils.internal.get-store-compute-to-tensor-fn-fn]
    pub fn get_store_compute_to_tensor_fn<
        CTYPE_COMPUTE: ComputeCast
            + StaticCast<CTYPE_COMPUTE>
            + crate::runtime::core::exec_aten::util::scalar_type_util::CppTypeToScalarType
            + 'static,
    >(
        context: &mut KernelRuntimeContext,
        t: &Tensor,
        dtypes: SupportedTensorDtypes,
        op_name: &str,
    ) -> StoreComputeToTensorFn<CTYPE_COMPUTE>
    where
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
        match dtypes {
            SupportedTensorDtypes::REALHBBF16 => {
                get_store_compute_to_tensor_fn_realhbbf16::<CTYPE_COMPUTE>(context, t, op_name)
            }
            SupportedTensorDtypes::REALHBF16 => {
                get_store_compute_to_tensor_fn_realhbf16::<CTYPE_COMPUTE>(context, t, op_name)
            }
            SupportedTensorDtypes::FLOATHBF16 => {
                get_store_compute_to_tensor_fn_floathbf16::<CTYPE_COMPUTE>(context, t, op_name)
            }
            SupportedTensorDtypes::INTB => {
                get_store_compute_to_tensor_fn_intb::<CTYPE_COMPUTE>(context, t, op_name)
            }
            SupportedTensorDtypes::BOOL => {
                get_store_compute_to_tensor_fn_bool::<CTYPE_COMPUTE>(context, t, op_name)
            }
            SupportedTensorDtypes::BOOL_OR_BYTE => {
                get_store_compute_to_tensor_fn_bool_or_byte::<CTYPE_COMPUTE>(context, t, op_name)
            }
            SupportedTensorDtypes::SAME_AS_COMPUTE => {
                get_store_compute_to_tensor_fn_same_as_compute::<CTYPE_COMPUTE>(context, t, op_name)
            }
            SupportedTensorDtypes::SAME_AS_COMMON => {
                get_store_compute_to_tensor_fn_same_as_common::<CTYPE_COMPUTE>(context, t, op_name)
            }
        }
    }

    // PORT-NOTE: with no selective build, the C++ forwards to the impl using the
    // shared `kGenericElementwiseOpName`. This port passes that constant name.
    pub const K_GENERIC_ELEMENTWISE_OP_NAME: &str = "generic_elementwise_op";

    // [spec:et:def:dtype-util.torch.executor.native.utils.internal.get-load-to-compute-fn-fn]
    // [spec:et:sem:dtype-util.torch.executor.native.utils.internal.get-load-to-compute-fn-fn]
    pub fn get_load_to_compute_fn<
        CTYPE_COMPUTE: ComputeCast
            + crate::runtime::core::exec_aten::util::scalar_type_util::CppTypeToScalarType
            + 'static,
    >(
        context: &mut KernelRuntimeContext,
        t: &Tensor,
        dtypes: SupportedTensorDtypes,
        _op_name: &str,
    ) -> LoadToComputeFn<CTYPE_COMPUTE> {
        // NOTE: Selective build relies on the operator name being passed here.
        // When it's *not* active, using the same operator name everywhere saves
        // on size because we don't require a new template instantiation for
        // every operator.
        get_load_to_compute_fn_impl::<CTYPE_COMPUTE>(
            context,
            t,
            dtypes,
            K_GENERIC_ELEMENTWISE_OP_NAME,
        )
    }

    // [spec:et:def:dtype-util.torch.executor.native.utils.internal.check-tensor-dtype-fn]
    // [spec:et:sem:dtype-util.torch.executor.native.utils.internal.check-tensor-dtype-fn]
    pub fn check_tensor_dtype(
        t: &Tensor,
        dtypes: SupportedTensorDtypes,
        compute_type: ScalarType,
    ) -> bool {
        match dtypes {
            SupportedTensorDtypes::REALHBBF16 => tensor_is_realhbbf16_type(t),
            SupportedTensorDtypes::REALHBF16 => tensor_is_realhbf16_type(t),
            SupportedTensorDtypes::FLOATHBF16 => tensor_is_floating_type(t),
            SupportedTensorDtypes::INTB => tensor_is_integral_type(t, true),
            SupportedTensorDtypes::BOOL => tensor_is_type1(t, ScalarType::Bool),
            SupportedTensorDtypes::BOOL_OR_BYTE => {
                tensor_is_type2(t, ScalarType::Bool, ScalarType::Byte)
            }
            SupportedTensorDtypes::SAME_AS_COMPUTE => tensor_is_type1(t, compute_type),
            SupportedTensorDtypes::SAME_AS_COMMON => {
                if compute_type == ScalarType::Float {
                    tensor_is_type(t, ScalarType::Float, ScalarType::Half, ScalarType::BFloat16)
                } else {
                    tensor_is_type1(t, compute_type)
                }
            }
        }
    }

    use crate::runtime::core::exec_aten::util::tensor_util::tensor_is_type;

    /// Return the one output type we are willing to emit specialized code to
    /// handle, given a compute type of CTYPE_COMPUTE and supported output types
    /// of out_dtypes.
    // [spec:et:def:dtype-util.torch.executor.native.utils.internal.specialized-output-scalar-type-fn]
    // [spec:et:sem:dtype-util.torch.executor.native.utils.internal.specialized-output-scalar-type-fn]
    pub fn specialized_output_scalar_type<
        CTYPE_COMPUTE: crate::runtime::core::exec_aten::util::scalar_type_util::CppTypeToScalarType,
    >(
        out_dtypes: SupportedTensorDtypes,
    ) -> ScalarType {
        match out_dtypes {
            SupportedTensorDtypes::BOOL => ScalarType::Bool,
            SupportedTensorDtypes::BOOL_OR_BYTE => ScalarType::Bool,
            SupportedTensorDtypes::REALHBBF16
            | SupportedTensorDtypes::REALHBF16
            | SupportedTensorDtypes::FLOATHBF16
            | SupportedTensorDtypes::INTB
            | SupportedTensorDtypes::SAME_AS_COMPUTE
            | SupportedTensorDtypes::SAME_AS_COMMON => CTYPE_COMPUTE::VALUE,
        }
    }
}

// Re-export the compute-cast helpers used across the elementwise machinery.
pub use internal::{
    ComputeCast, LoadToComputeFn, StaticCast, StoreComputeToTensorFn, check_tensor_dtype,
    get_load_to_compute_fn, get_store_compute_to_tensor_fn, specialized_output_scalar_type,
};
pub use internal::{
    convert_and_store as _convert_and_store, load_and_convert as _load_and_convert,
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::core::exec_aten::testing_util::tensor_factory::TensorFactory;
    use crate::runtime::core::portable_type::{BFloat16, Half};

    // check_tensor_dtype is a pure predicate over t.scalar_type() + the
    // SupportedTensorDtypes category + compute_type. There is no dedicated C++
    // test (it is only ever reached transitively via the elementwise machinery),
    // so this pins its per-category truth table against the sem rule.
    // [spec:et:sem:dtype-util.torch.executor.native.utils.internal.check-tensor-dtype-fn/test]
    #[test]
    fn check_tensor_dtype_categories() {
        crate::runtime::platform::runtime::runtime_init();
        let tf_byte = TensorFactory::<u8>::new();
        let tf_int = TensorFactory::<i32>::new();
        let tf_float = TensorFactory::<f32>::new();
        let tf_bool = TensorFactory::<bool>::new();
        let tf_half = TensorFactory::<Half>::new();
        let tf_bf16 = TensorFactory::<BFloat16>::new();
        let byte = tf_byte.zeros_default(vec![1]);
        let int = tf_int.zeros_default(vec![1]);
        let float = tf_float.zeros_default(vec![1]);
        let boolean = tf_bool.zeros_default(vec![1]);
        let half = tf_half.zeros_default(vec![1]);
        let bf16 = tf_bf16.zeros_default(vec![1]);

        use SupportedTensorDtypes as D;
        let c = |t, d, ct| check_tensor_dtype(t, d, ct);
        // compute_type is only consulted by SAME_AS_COMPUTE / SAME_AS_COMMON;
        // use a neutral value elsewhere.
        let any = ScalarType::Int;

        // REALHBBF16: reals + Half + BFloat16 + Bool.
        assert!(c(&byte, D::REALHBBF16, any));
        assert!(c(&int, D::REALHBBF16, any));
        assert!(c(&float, D::REALHBBF16, any));
        assert!(c(&half, D::REALHBBF16, any));
        assert!(c(&bf16, D::REALHBBF16, any));
        assert!(c(&boolean, D::REALHBBF16, any));

        // REALHBF16: reals + Half + BFloat16, no Bool.
        assert!(c(&int, D::REALHBF16, any));
        assert!(c(&half, D::REALHBF16, any));
        assert!(c(&bf16, D::REALHBF16, any));
        assert!(!c(&boolean, D::REALHBF16, any));

        // FLOATHBF16: floating types only.
        assert!(c(&float, D::FLOATHBF16, any));
        assert!(c(&half, D::FLOATHBF16, any));
        assert!(c(&bf16, D::FLOATHBF16, any));
        assert!(!c(&int, D::FLOATHBF16, any));
        assert!(!c(&boolean, D::FLOATHBF16, any));

        // INTB: integral types including Bool, no float.
        assert!(c(&byte, D::INTB, any));
        assert!(c(&int, D::INTB, any));
        assert!(c(&boolean, D::INTB, any));
        assert!(!c(&float, D::INTB, any));
        assert!(!c(&half, D::INTB, any));

        // BOOL: exactly Bool.
        assert!(c(&boolean, D::BOOL, any));
        assert!(!c(&byte, D::BOOL, any));

        // BOOL_OR_BYTE: Bool or Byte.
        assert!(c(&boolean, D::BOOL_OR_BYTE, any));
        assert!(c(&byte, D::BOOL_OR_BYTE, any));
        assert!(!c(&int, D::BOOL_OR_BYTE, any));

        // SAME_AS_COMPUTE: dtype must equal compute_type exactly.
        assert!(c(&int, D::SAME_AS_COMPUTE, ScalarType::Int));
        assert!(!c(&int, D::SAME_AS_COMPUTE, ScalarType::Float));
        assert!(c(&float, D::SAME_AS_COMPUTE, ScalarType::Float));

        // SAME_AS_COMMON: compute_type Float => {Float, Half, BFloat16};
        // otherwise equals compute_type.
        assert!(c(&float, D::SAME_AS_COMMON, ScalarType::Float));
        assert!(c(&half, D::SAME_AS_COMMON, ScalarType::Float));
        assert!(c(&bf16, D::SAME_AS_COMMON, ScalarType::Float));
        assert!(!c(&int, D::SAME_AS_COMMON, ScalarType::Float));
        assert!(c(&int, D::SAME_AS_COMMON, ScalarType::Int));
        assert!(!c(&float, D::SAME_AS_COMMON, ScalarType::Int));
    }

    use crate::runtime::core::error::Error;
    use crate::runtime::core::memory_allocator::{MemoryAllocator, MemoryAllocatorBase};
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

    fn scalar_tensor_f32(tf: &TensorFactory<f32>, v: f32) -> Tensor<'_> {
        tf.make(vec![1], vec![v], vec![], TensorShapeDynamism::STATIC)
    }

    // load_and_convert / convert_and_store are the pure per-element engines
    // underneath every load/store getter; pin the C++ static_cast semantics.
    // [spec:et:sem:dtype-util.torch.executor.native.utils.internal.load-and-convert-fn/test]
    // [spec:et:sem:dtype-util.torch.executor.native.utils.internal.convert-and-store-fn/test]
    #[test]
    fn dtype_util_load_and_convert_and_store_static_cast_rules() {
        use internal::{convert_and_store, load_and_convert};
        // int -> float promotion
        let i: i32 = -7;
        let v: f32 = load_and_convert::<f32, i32>(&i as *const i32 as *const core::ffi::c_void);
        assert_eq!(v, -7.0f32);

        // float -> int truncates toward zero (C++ static_cast)
        let f: f64 = -3.9;
        let n: i32 = load_and_convert::<i32, f64>(&f as *const f64 as *const core::ffi::c_void);
        assert_eq!(n, -3);

        // nonzero -> bool true, zero -> bool false
        let a: i32 = 5;
        let b: i32 = 0;
        assert!(load_and_convert::<bool, i32>(
            &a as *const i32 as *const core::ffi::c_void
        ));
        assert!(!load_and_convert::<bool, i32>(
            &b as *const i32 as *const core::ffi::c_void
        ));

        // Half source promotes through f32
        let h = Half::from_f32(2.5);
        let hv: f32 = load_and_convert::<f32, Half>(&h as *const Half as *const core::ffi::c_void);
        assert_eq!(hv, 2.5f32);

        // convert_and_store is the inverse: cast compute value into the tensor slot
        let mut dst_i: i32 = 0;
        convert_and_store::<i32, f32>(4.8f32, &mut dst_i as *mut i32 as *mut core::ffi::c_void);
        assert_eq!(dst_i, 4);

        let mut dst_bool: bool = false;
        convert_and_store::<bool, i32>(9, &mut dst_bool as *mut bool as *mut core::ffi::c_void);
        assert!(dst_bool);

        let mut dst_bf: BFloat16 = BFloat16::from_f32(0.0);
        convert_and_store::<BFloat16, f32>(
            1.0f32,
            &mut dst_bf as *mut BFloat16 as *mut core::ffi::c_void,
        );
        assert_eq!(dst_bf.to_f32(), 1.0f32);
    }

    // The REALHBBF16 load getter accepts a real/Half/Bool/BFloat16 tensor and
    // returns a loader that promotes its element to the compute type.
    // [spec:et:sem:dtype-util.torch.executor.native.utils.internal.get-load-to-compute-fn-realhbbf16-fn/test]
    #[test]
    fn dtype_util_get_load_to_compute_fn_realhbbf16() {
        let tf = TensorFactory::<f32>::new();
        let t = scalar_tensor_f32(&tf, 3.25);
        let mut ctx = context();
        let load = internal::get_load_to_compute_fn_realhbbf16::<f32>(&mut ctx, &t, "op");
        assert_eq!(ctx.failure_state(), Error::Ok);
        let got = load(t.const_data_ptr::<core::ffi::c_void>());
        assert_eq!(got, 3.25f32);
    }

    // REALHBF16 rejects Bool (the "no Bool" set); failure path fails the ctx.
    // [spec:et:sem:dtype-util.torch.executor.native.utils.internal.get-load-to-compute-fn-realhbf16-fn/test]
    #[test]
    fn dtype_util_get_load_to_compute_fn_realhbf16_rejects_bool() {
        let tf = TensorFactory::<f32>::new();
        let t = scalar_tensor_f32(&tf, 1.0);
        let mut ctx = context();
        let load = internal::get_load_to_compute_fn_realhbf16::<f32>(&mut ctx, &t, "op");
        assert_eq!(ctx.failure_state(), Error::Ok);
        assert_eq!(load(t.const_data_ptr::<core::ffi::c_void>()), 1.0f32);

        let tfb = TensorFactory::<bool>::new();
        let tb = tfb.make(vec![1], vec![true], vec![], TensorShapeDynamism::STATIC);
        let mut ctx2 = context();
        let _ = internal::get_load_to_compute_fn_realhbf16::<f32>(&mut ctx2, &tb, "op");
        assert_eq!(ctx2.failure_state(), Error::InvalidArgument);
    }

    // FLOATHBF16 load getter accepts floating tensors.
    // [spec:et:sem:dtype-util.torch.executor.native.utils.internal.get-load-to-compute-fn-floathbf16-fn/test]
    #[test]
    fn dtype_util_get_load_to_compute_fn_floathbf16() {
        let tf = TensorFactory::<f32>::new();
        let t = scalar_tensor_f32(&tf, 7.5);
        let mut ctx = context();
        let load = internal::get_load_to_compute_fn_floathbf16::<f32>(&mut ctx, &t, "op");
        assert_eq!(ctx.failure_state(), Error::Ok);
        assert_eq!(load(t.const_data_ptr::<core::ffi::c_void>()), 7.5f32);
    }

    // INTB load getter accepts int-plus-Bool; rejects float.
    // [spec:et:sem:dtype-util.torch.executor.native.utils.internal.get-load-to-compute-fn-intb-fn/test]
    #[test]
    fn dtype_util_get_load_to_compute_fn_intb() {
        let tf = TensorFactory::<i32>::new();
        let t = tf.make(vec![1], vec![42], vec![], TensorShapeDynamism::STATIC);
        let mut ctx = context();
        let load = internal::get_load_to_compute_fn_intb::<f32>(&mut ctx, &t, "op");
        assert_eq!(ctx.failure_state(), Error::Ok);
        assert_eq!(load(t.const_data_ptr::<core::ffi::c_void>()), 42.0f32);

        let tff = TensorFactory::<f32>::new();
        let tfl = scalar_tensor_f32(&tff, 1.0);
        let mut ctx2 = context();
        let _ = internal::get_load_to_compute_fn_intb::<f32>(&mut ctx2, &tfl, "op");
        assert_eq!(ctx2.failure_state(), Error::InvalidArgument);
    }

    // BOOL load getter accepts only Bool; anything else fails the ctx.
    // [spec:et:sem:dtype-util.torch.executor.native.utils.internal.get-load-to-compute-fn-bool-fn/test]
    #[test]
    fn dtype_util_get_load_to_compute_fn_bool() {
        let tfb = TensorFactory::<bool>::new();
        let tb = tfb.make(vec![1], vec![true], vec![], TensorShapeDynamism::STATIC);
        let mut ctx = context();
        let load = internal::get_load_to_compute_fn_bool::<f32>(&mut ctx, &tb, "op");
        assert_eq!(ctx.failure_state(), Error::Ok);
        assert_eq!(load(tb.const_data_ptr::<core::ffi::c_void>()), 1.0f32);

        let tff = TensorFactory::<f32>::new();
        let tf = scalar_tensor_f32(&tff, 1.0);
        let mut ctx2 = context();
        let _ = internal::get_load_to_compute_fn_bool::<f32>(&mut ctx2, &tf, "op");
        assert_eq!(ctx2.failure_state(), Error::InvalidArgument);
    }

    // BOOL_OR_BYTE load getter accepts Bool and Byte; rejects others.
    // [spec:et:sem:dtype-util.torch.executor.native.utils.internal.get-load-to-compute-fn-bool-or-byte-fn/test]
    #[test]
    fn dtype_util_get_load_to_compute_fn_bool_or_byte() {
        let tfu = TensorFactory::<u8>::new();
        let tu = tfu.make(vec![1], vec![200], vec![], TensorShapeDynamism::STATIC);
        let mut ctx = context();
        let load = internal::get_load_to_compute_fn_bool_or_byte::<i32>(&mut ctx, &tu, "op");
        assert_eq!(ctx.failure_state(), Error::Ok);
        assert_eq!(load(tu.const_data_ptr::<core::ffi::c_void>()), 200i32);

        let tff = TensorFactory::<f32>::new();
        let tf = scalar_tensor_f32(&tff, 1.0);
        let mut ctx2 = context();
        let _ = internal::get_load_to_compute_fn_bool_or_byte::<i32>(&mut ctx2, &tf, "op");
        assert_eq!(ctx2.failure_state(), Error::InvalidArgument);
    }

    // SAME_AS_COMPUTE accepts only the tensor whose dtype equals compute type.
    // [spec:et:sem:dtype-util.torch.executor.native.utils.internal.get-load-to-compute-fn-same-as-compute-fn/test]
    #[test]
    fn dtype_util_get_load_to_compute_fn_same_as_compute() {
        let tf = TensorFactory::<f32>::new();
        let t = scalar_tensor_f32(&tf, 5.0);
        let mut ctx = context();
        let load = internal::get_load_to_compute_fn_same_as_compute::<f32>(&mut ctx, &t, "op");
        assert_eq!(ctx.failure_state(), Error::Ok);
        assert_eq!(load(t.const_data_ptr::<core::ffi::c_void>()), 5.0f32);

        // compute type i32 but tensor is f32 -> reject
        let mut ctx2 = context();
        let _ = internal::get_load_to_compute_fn_same_as_compute::<i32>(&mut ctx2, &t, "op");
        assert_eq!(ctx2.failure_state(), Error::InvalidArgument);
    }

    // SAME_AS_COMMON: for float compute, Half/BFloat16 inputs are also accepted.
    // [spec:et:sem:dtype-util.torch.executor.native.utils.internal.get-load-to-compute-fn-same-as-common-fn/test]
    #[test]
    fn dtype_util_get_load_to_compute_fn_same_as_common() {
        let tfh = TensorFactory::<Half>::new();
        let th = tfh.make(
            vec![1],
            vec![Half::from_f32(2.0)],
            vec![],
            TensorShapeDynamism::STATIC,
        );
        let mut ctx = context();
        // float compute accepts Half
        let load = internal::get_load_to_compute_fn_same_as_common::<f32>(&mut ctx, &th, "op");
        assert_eq!(ctx.failure_state(), Error::Ok);
        assert_eq!(load(th.const_data_ptr::<core::ffi::c_void>()), 2.0f32);

        // non-float compute (i32) delegates to same_as_compute: Half rejected
        let mut ctx2 = context();
        let _ = internal::get_load_to_compute_fn_same_as_common::<i32>(&mut ctx2, &th, "op");
        assert_eq!(ctx2.failure_state(), Error::InvalidArgument);
    }

    // Store getters: verify the returned storer casts compute -> tensor element.
    // [spec:et:sem:dtype-util.torch.executor.native.utils.internal.get-store-compute-to-tensor-fn-realhbbf16-fn/test]
    #[test]
    fn dtype_util_get_store_compute_to_tensor_fn_realhbbf16() {
        let tf = TensorFactory::<i32>::new();
        let t = tf.make(vec![1], vec![0], vec![], TensorShapeDynamism::STATIC);
        let mut ctx = context();
        let store = internal::get_store_compute_to_tensor_fn_realhbbf16::<f32>(&mut ctx, &t, "op");
        assert_eq!(ctx.failure_state(), Error::Ok);
        store(6.9f32, t.mutable_data_ptr::<core::ffi::c_void>());
        assert_eq!(unsafe { *t.const_data_ptr::<i32>() }, 6); // truncates toward zero
    }

    // [spec:et:sem:dtype-util.torch.executor.native.utils.internal.get-store-compute-to-tensor-fn-realhbf16-fn/test]
    #[test]
    fn dtype_util_get_store_compute_to_tensor_fn_realhbf16_rejects_bool() {
        let tff = TensorFactory::<f32>::new();
        let t = scalar_tensor_f32(&tff, 0.0);
        let mut ctx = context();
        let store = internal::get_store_compute_to_tensor_fn_realhbf16::<f32>(&mut ctx, &t, "op");
        assert_eq!(ctx.failure_state(), Error::Ok);
        store(3.5f32, t.mutable_data_ptr::<core::ffi::c_void>());
        assert_eq!(unsafe { *t.const_data_ptr::<f32>() }, 3.5f32);

        let tfb = TensorFactory::<bool>::new();
        let tb = tfb.make(vec![1], vec![false], vec![], TensorShapeDynamism::STATIC);
        let mut ctx2 = context();
        let _ = internal::get_store_compute_to_tensor_fn_realhbf16::<f32>(&mut ctx2, &tb, "op");
        assert_eq!(ctx2.failure_state(), Error::InvalidArgument);
    }

    // [spec:et:sem:dtype-util.torch.executor.native.utils.internal.get-store-compute-to-tensor-fn-floathbf16-fn/test]
    #[test]
    fn dtype_util_get_store_compute_to_tensor_fn_floathbf16() {
        let tff = TensorFactory::<f32>::new();
        let t = scalar_tensor_f32(&tff, 0.0);
        let mut ctx = context();
        let store = internal::get_store_compute_to_tensor_fn_floathbf16::<f32>(&mut ctx, &t, "op");
        assert_eq!(ctx.failure_state(), Error::Ok);
        store(-2.0f32, t.mutable_data_ptr::<core::ffi::c_void>());
        assert_eq!(unsafe { *t.const_data_ptr::<f32>() }, -2.0f32);

        // integral tensor rejected
        let tfi = TensorFactory::<i32>::new();
        let ti = tfi.make(vec![1], vec![0], vec![], TensorShapeDynamism::STATIC);
        let mut ctx2 = context();
        let _ = internal::get_store_compute_to_tensor_fn_floathbf16::<f32>(&mut ctx2, &ti, "op");
        assert_eq!(ctx2.failure_state(), Error::InvalidArgument);
    }

    // [spec:et:sem:dtype-util.torch.executor.native.utils.internal.get-store-compute-to-tensor-fn-intb-fn/test]
    #[test]
    fn dtype_util_get_store_compute_to_tensor_fn_intb() {
        let tfi = TensorFactory::<i32>::new();
        let t = tfi.make(vec![1], vec![0], vec![], TensorShapeDynamism::STATIC);
        let mut ctx = context();
        let store = internal::get_store_compute_to_tensor_fn_intb::<f32>(&mut ctx, &t, "op");
        assert_eq!(ctx.failure_state(), Error::Ok);
        store(-3.7f32, t.mutable_data_ptr::<core::ffi::c_void>());
        assert_eq!(unsafe { *t.const_data_ptr::<i32>() }, -3); // truncate toward zero

        let tff = TensorFactory::<f32>::new();
        let tf = scalar_tensor_f32(&tff, 0.0);
        let mut ctx2 = context();
        let _ = internal::get_store_compute_to_tensor_fn_intb::<f32>(&mut ctx2, &tf, "op");
        assert_eq!(ctx2.failure_state(), Error::InvalidArgument);
    }

    // [spec:et:sem:dtype-util.torch.executor.native.utils.internal.get-store-compute-to-tensor-fn-bool-fn/test]
    #[test]
    fn dtype_util_get_store_compute_to_tensor_fn_bool() {
        let tfb = TensorFactory::<bool>::new();
        let t = tfb.make(vec![1], vec![false], vec![], TensorShapeDynamism::STATIC);
        let mut ctx = context();
        let store = internal::get_store_compute_to_tensor_fn_bool::<f32>(&mut ctx, &t, "op");
        assert_eq!(ctx.failure_state(), Error::Ok);
        store(9.0f32, t.mutable_data_ptr::<core::ffi::c_void>());
        assert!(unsafe { *t.const_data_ptr::<bool>() });
        store(0.0f32, t.mutable_data_ptr::<core::ffi::c_void>());
        assert!(!unsafe { *t.const_data_ptr::<bool>() });

        let tff = TensorFactory::<f32>::new();
        let tf = scalar_tensor_f32(&tff, 0.0);
        let mut ctx2 = context();
        let _ = internal::get_store_compute_to_tensor_fn_bool::<f32>(&mut ctx2, &tf, "op");
        assert_eq!(ctx2.failure_state(), Error::InvalidArgument);
    }

    // [spec:et:sem:dtype-util.torch.executor.native.utils.internal.get-store-compute-to-tensor-fn-bool-or-byte-fn/test]
    #[test]
    fn dtype_util_get_store_compute_to_tensor_fn_bool_or_byte() {
        let tfu = TensorFactory::<u8>::new();
        let t = tfu.make(vec![1], vec![0], vec![], TensorShapeDynamism::STATIC);
        let mut ctx = context();
        let store =
            internal::get_store_compute_to_tensor_fn_bool_or_byte::<i32>(&mut ctx, &t, "op");
        assert_eq!(ctx.failure_state(), Error::Ok);
        store(130i32, t.mutable_data_ptr::<core::ffi::c_void>());
        assert_eq!(unsafe { *t.const_data_ptr::<u8>() }, 130u8);

        let tff = TensorFactory::<f32>::new();
        let tf = scalar_tensor_f32(&tff, 0.0);
        let mut ctx2 = context();
        let _ = internal::get_store_compute_to_tensor_fn_bool_or_byte::<i32>(&mut ctx2, &tf, "op");
        assert_eq!(ctx2.failure_state(), Error::InvalidArgument);
    }

    // [spec:et:sem:dtype-util.torch.executor.native.utils.internal.get-store-compute-to-tensor-fn-same-as-compute-fn/test]
    #[test]
    fn dtype_util_get_store_compute_to_tensor_fn_same_as_compute() {
        let tff = TensorFactory::<f32>::new();
        let t = scalar_tensor_f32(&tff, 0.0);
        let mut ctx = context();
        let store =
            internal::get_store_compute_to_tensor_fn_same_as_compute::<f32>(&mut ctx, &t, "op");
        assert_eq!(ctx.failure_state(), Error::Ok);
        store(1.25f32, t.mutable_data_ptr::<core::ffi::c_void>());
        assert_eq!(unsafe { *t.const_data_ptr::<f32>() }, 1.25f32);

        let mut ctx2 = context();
        let _ =
            internal::get_store_compute_to_tensor_fn_same_as_compute::<i32>(&mut ctx2, &t, "op");
        assert_eq!(ctx2.failure_state(), Error::InvalidArgument);
    }

    // [spec:et:sem:dtype-util.torch.executor.native.utils.internal.get-store-compute-to-tensor-fn-same-as-common-fn/test]
    #[test]
    fn dtype_util_get_store_compute_to_tensor_fn_same_as_common() {
        // float compute accepts Half output
        let tfh = TensorFactory::<Half>::new();
        let th = tfh.make(
            vec![1],
            vec![Half::from_f32(0.0)],
            vec![],
            TensorShapeDynamism::STATIC,
        );
        let mut ctx = context();
        let store =
            internal::get_store_compute_to_tensor_fn_same_as_common::<f32>(&mut ctx, &th, "op");
        assert_eq!(ctx.failure_state(), Error::Ok);
        store(4.0f32, th.mutable_data_ptr::<core::ffi::c_void>());
        assert_eq!(unsafe { (*th.const_data_ptr::<Half>()).to_f32() }, 4.0f32);

        // non-float compute delegates to same_as_compute: Half rejected
        let mut ctx2 = context();
        let _ =
            internal::get_store_compute_to_tensor_fn_same_as_common::<i32>(&mut ctx2, &th, "op");
        assert_eq!(ctx2.failure_state(), Error::InvalidArgument);
    }

    // impl dispatcher: FLOATHBF16 on load reuses the REALHBF16 getter, so an
    // integral tensor is accepted on load (the documented load/store asymmetry).
    // [spec:et:sem:dtype-util.torch.executor.native.utils.internal.get-load-to-compute-fn-impl-fn/test]
    #[test]
    fn dtype_util_get_load_to_compute_fn_impl_floathbf16_reuses_realhbf16() {
        let tfi = TensorFactory::<i32>::new();
        let ti = tfi.make(vec![1], vec![11], vec![], TensorShapeDynamism::STATIC);
        let mut ctx = context();
        let load = internal::get_load_to_compute_fn_impl::<f32>(
            &mut ctx,
            &ti,
            SupportedTensorDtypes::FLOATHBF16,
            "op",
        );
        // load path accepts the integral tensor (wider set) and does not fail
        assert_eq!(ctx.failure_state(), Error::Ok);
        assert_eq!(load(ti.const_data_ptr::<core::ffi::c_void>()), 11.0f32);
    }

    // public get_load_to_compute_fn forwards to the impl with the shared op name.
    // [spec:et:sem:dtype-util.torch.executor.native.utils.internal.get-load-to-compute-fn-fn/test]
    #[test]
    fn dtype_util_get_load_to_compute_fn_public() {
        let tf = TensorFactory::<f32>::new();
        let t = scalar_tensor_f32(&tf, 8.0);
        let mut ctx = context();
        let load = internal::get_load_to_compute_fn::<f32>(
            &mut ctx,
            &t,
            SupportedTensorDtypes::REALHBBF16,
            "caller",
        );
        assert_eq!(ctx.failure_state(), Error::Ok);
        assert_eq!(load(t.const_data_ptr::<core::ffi::c_void>()), 8.0f32);
    }

    // store dispatcher: FLOATHBF16 uses the TRUE floathbf16 getter (rejects
    // integral), unlike the load dispatcher.
    // [spec:et:sem:dtype-util.torch.executor.native.utils.internal.get-store-compute-to-tensor-fn-fn/test]
    #[test]
    fn dtype_util_get_store_compute_to_tensor_fn_dispatch_floathbf16_true() {
        let tff = TensorFactory::<f32>::new();
        let t = scalar_tensor_f32(&tff, 0.0);
        let mut ctx = context();
        let store = internal::get_store_compute_to_tensor_fn::<f32>(
            &mut ctx,
            &t,
            SupportedTensorDtypes::FLOATHBF16,
            "op",
        );
        assert_eq!(ctx.failure_state(), Error::Ok);
        store(2.5f32, t.mutable_data_ptr::<core::ffi::c_void>());
        assert_eq!(unsafe { *t.const_data_ptr::<f32>() }, 2.5f32);

        // integral output rejected on store (true FLOATHBF16 getter)
        let tfi = TensorFactory::<i32>::new();
        let ti = tfi.make(vec![1], vec![0], vec![], TensorShapeDynamism::STATIC);
        let mut ctx2 = context();
        let _ = internal::get_store_compute_to_tensor_fn::<f32>(
            &mut ctx2,
            &ti,
            SupportedTensorDtypes::FLOATHBF16,
            "op",
        );
        assert_eq!(ctx2.failure_state(), Error::InvalidArgument);
    }

    // specialized_output_scalar_type is a pure constexpr mapping.
    // [spec:et:sem:dtype-util.torch.executor.native.utils.internal.specialized-output-scalar-type-fn/test]
    #[test]
    fn dtype_util_specialized_output_scalar_type() {
        use internal::specialized_output_scalar_type;
        assert_eq!(
            specialized_output_scalar_type::<f32>(SupportedTensorDtypes::BOOL),
            ScalarType::Bool
        );
        assert_eq!(
            specialized_output_scalar_type::<f32>(SupportedTensorDtypes::BOOL_OR_BYTE),
            ScalarType::Bool
        );
        assert_eq!(
            specialized_output_scalar_type::<f32>(SupportedTensorDtypes::REALHBBF16),
            ScalarType::Float
        );
        assert_eq!(
            specialized_output_scalar_type::<i32>(SupportedTensorDtypes::SAME_AS_COMMON),
            ScalarType::Int
        );
        assert_eq!(
            specialized_output_scalar_type::<i64>(SupportedTensorDtypes::INTB),
            ScalarType::Long
        );
    }
}
