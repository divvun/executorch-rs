//! Literal port of kernels/portable/cpu/util/matmul_ops_util.cpp + kernels/portable/cpu/util/matmul_ops_util.h.

use crate::runtime::core::exec_aten::util::tensor_util::{
    tensor_is_rank, tensors_have_same_dtype, tensors_have_same_size_at_dims,
};
use crate::runtime::core::portable_type::scalar::Scalar;
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::core::portable_type::tensor_impl::{SizesType, ssize_t};

// PORT-NOTE: the crate-level `et_check_or_return_false!` (runtime/core/error.rs)
// drops all caller-supplied format arguments after the leading literal. To keep
// the ported check messages literal (mirroring the C++ `ET_LOG_AND_RETURN_IF_FALSE`
// which expands to `ET_CHECK_OR_RETURN_FALSE(cond, "")`), this module defines
// its own `et_log_and_return_if_false!` as tensor_util.rs does.
macro_rules! et_log_and_return_if_false {
    ($cond:expr) => {{
        if !($cond) {
            $crate::et_log!(Error, "Check failed ({}): ", ::core::stringify!($cond));
            return false;
        }
    }};
}

// [spec:et:def:matmul-ops-util.torch.executor.check-addmm-args-fn]
// [spec:et:sem:matmul-ops-util.torch.executor.check-addmm-args-fn]
pub fn check_addmm_args(
    in_: &Tensor,
    mat1: &Tensor,
    mat2: &Tensor,
    _beta: &Scalar,
    _alpha: &Scalar,
    out: &Tensor,
) -> bool {
    et_log_and_return_if_false!(tensor_is_rank(mat1, 2));
    et_log_and_return_if_false!(tensor_is_rank(mat2, 2));
    et_log_and_return_if_false!(tensor_is_rank(out, 2));

    et_log_and_return_if_false!(tensors_have_same_dtype(in_, mat1, mat2));
    et_log_and_return_if_false!(tensors_have_same_dtype2(in_, out));

    et_log_and_return_if_false!(tensors_have_same_size_at_dims(mat1, 1, mat2, 0));

    true
}

// [spec:et:def:matmul-ops-util.torch.executor.check-bmm-args-fn]
// [spec:et:sem:matmul-ops-util.torch.executor.check-bmm-args-fn]
pub fn check_bmm_args(in_: &Tensor, mat2: &Tensor, out: &Tensor) -> bool {
    et_log_and_return_if_false!(tensor_is_rank(in_, 3));
    et_log_and_return_if_false!(tensor_is_rank(mat2, 3));
    et_log_and_return_if_false!(tensor_is_rank(out, 3));

    et_log_and_return_if_false!(tensors_have_same_dtype(in_, mat2, out));

    et_log_and_return_if_false!(tensors_have_same_size_at_dims(in_, 0, mat2, 0));
    et_log_and_return_if_false!(tensors_have_same_size_at_dims(in_, 2, mat2, 1));

    true
}

// [spec:et:def:matmul-ops-util.torch.executor.get-bmm-out-target-size-fn]
// [spec:et:sem:matmul-ops-util.torch.executor.get-bmm-out-target-size-fn]
///
/// # Safety
/// `out_sizes` must point to at least 3 valid `SizesType` elements and
/// `out_ndim` to a valid `usize`.
pub unsafe fn get_bmm_out_target_size(
    mat1: &Tensor,
    mat2: &Tensor,
    out_sizes: *mut SizesType,
    out_ndim: *mut usize,
) {
    unsafe {
        *out_ndim = 3;
        *out_sizes.add(0) = mat1.size(0) as SizesType;
        *out_sizes.add(1) = mat1.size(1) as SizesType;
        *out_sizes.add(2) = mat2.size(2) as SizesType;
    }
}

// [spec:et:def:matmul-ops-util.torch.executor.check-mm-args-fn]
// [spec:et:sem:matmul-ops-util.torch.executor.check-mm-args-fn]
pub fn check_mm_args(in_: &Tensor, mat2: &Tensor, out: &Tensor) -> bool {
    et_log_and_return_if_false!(tensor_is_rank(in_, 2));
    et_log_and_return_if_false!(tensor_is_rank(mat2, 2));
    et_log_and_return_if_false!(tensor_is_rank(out, 2));

    et_log_and_return_if_false!(tensors_have_same_dtype(in_, mat2, out));

    et_log_and_return_if_false!(tensors_have_same_size_at_dims(in_, 1, mat2, 0));

    true
}

// [spec:et:def:matmul-ops-util.torch.executor.check-linear-args-fn]
// [spec:et:sem:matmul-ops-util.torch.executor.check-linear-args-fn]
pub fn check_linear_args(in_: &Tensor, mat2: &Tensor, out: &Tensor) -> bool {
    et_log_and_return_if_false!(in_.dim() == out.dim());
    et_log_and_return_if_false!(in_.dim() >= 2);
    et_log_and_return_if_false!(tensor_is_rank(mat2, 2));

    et_log_and_return_if_false!(tensors_have_same_dtype(in_, mat2, out));

    et_log_and_return_if_false!(tensors_have_same_size_at_dims(
        in_,
        (in_.dim() - 1) as usize,
        mat2,
        1
    ));

    true
}

// [spec:et:def:matmul-ops-util.torch.executor.get-mm-out-target-size-fn]
// [spec:et:sem:matmul-ops-util.torch.executor.get-mm-out-target-size-fn]
///
/// # Safety
/// `out_sizes` must point to at least 2 valid `SizesType` elements and
/// `out_ndim` to a valid `usize`.
pub unsafe fn get_mm_out_target_size(
    mat1: &Tensor,
    mat2: &Tensor,
    out_sizes: *mut SizesType,
    out_ndim: *mut usize,
) {
    unsafe {
        *out_ndim = 2;
        *out_sizes.add(0) = mat1.size(0) as SizesType;
        *out_sizes.add(1) = mat2.size(1) as SizesType;
    }
}

// [spec:et:def:matmul-ops-util.torch.executor.get-linear-out-target-size-fn]
// [spec:et:sem:matmul-ops-util.torch.executor.get-linear-out-target-size-fn]
///
/// # Safety
/// `out_sizes` must point to at least `mat1.dim()` valid `SizesType` elements
/// and `out_ndim` to a valid `usize`.
pub unsafe fn get_linear_out_target_size(
    mat1: &Tensor,
    mat2: &Tensor,
    out_sizes: *mut SizesType,
    out_ndim: *mut usize,
) {
    unsafe {
        *out_ndim = mat1.dim() as usize;
        let mut ii: ssize_t = 0;
        while ii < mat1.dim() - 1 {
            *out_sizes.add(ii as usize) = *mat1.sizes().at(ii as usize);
            ii += 1;
        }
        *out_sizes.add((mat1.dim() - 1) as usize) = mat2.size(0) as SizesType;
    }
}

// PORT-NOTE: `tensors_have_same_dtype2` (two-tensor overload) lives in
// tensor_util.rs; imported below alongside the three-tensor `tensors_have_same_dtype`.
use crate::runtime::core::exec_aten::util::tensor_util::tensors_have_same_dtype2;

pub mod internal {
    use super::*;

    // PORT-NOTE: C++ `bmm_out_impl<CTYPE>` is templated on the element type. The
    // element-type operations used are: `static_cast<CTYPE>(0.0)`, `+=`, and
    // `CTYPE * CTYPE`. Modeled via the local `BmmCtype` trait (as tensor_util.rs
    // models `static_cast` narrowing via a local trait), implemented for the
    // real dtypes bmm supports.
    pub trait BmmCtype: Copy {
        fn zero() -> Self;
        fn add_assign_mul(&mut self, a: Self, b: Self);
    }

    macro_rules! impl_bmm_ctype {
        ($($t:ty),*) => {$(
            impl BmmCtype for $t {
                fn zero() -> Self { 0 as $t }
                fn add_assign_mul(&mut self, a: Self, b: Self) { *self += a * b; }
            }
        )*};
    }
    impl_bmm_ctype!(u8, i8, i16, i32, i64, f32, f64);

    impl BmmCtype for crate::runtime::core::portable_type::Half {
        fn zero() -> Self {
            crate::runtime::core::portable_type::Half::from_f32(0.0)
        }
        fn add_assign_mul(&mut self, a: Self, b: Self) {
            *self += a * b;
        }
    }
    impl BmmCtype for crate::runtime::core::portable_type::BFloat16 {
        fn zero() -> Self {
            crate::runtime::core::portable_type::BFloat16::from_f32(0.0)
        }
        fn add_assign_mul(&mut self, a: Self, b: Self) {
            *self += a * b;
        }
    }

    // [spec:et:def:matmul-ops-util.torch.executor.internal.bmm-out-impl-fn]
    // [spec:et:sem:matmul-ops-util.torch.executor.internal.bmm-out-impl-fn]
    pub fn bmm_out_impl<CTYPE: BmmCtype>(in_: &Tensor, mat2: &Tensor, out: &Tensor) {
        let in_data: *const CTYPE = in_.const_data_ptr::<CTYPE>();
        let mat2_data: *const CTYPE = mat2.const_data_ptr::<CTYPE>();
        let out_data: *mut CTYPE = out.mutable_data_ptr::<CTYPE>();

        let batch_size: i64 = in_.size(0) as i64;
        let m: i64 = in_.size(1) as i64;
        let n: i64 = in_.size(2) as i64;
        let p: i64 = mat2.size(2) as i64;

        let mut b: i32 = 0;
        while (b as i64) < batch_size {
            let in_data_offset: *const CTYPE = unsafe { in_data.add((b as i64 * m * n) as usize) };
            let mat2_data_offset: *const CTYPE =
                unsafe { mat2_data.add((b as i64 * n * p) as usize) };
            let out_data_offset: *mut CTYPE = unsafe { out_data.add((b as i64 * m * p) as usize) };

            for i in 0..m {
                for j in 0..p {
                    let mut sum: CTYPE = CTYPE::zero();
                    for k in 0..n {
                        sum.add_assign_mul(
                            unsafe { *in_data_offset.add((i * n + k) as usize) },
                            unsafe { *mat2_data_offset.add((k * p + j) as usize) },
                        );
                    }
                    unsafe {
                        *out_data_offset.add((i * p + j) as usize) = sum;
                    }
                }
            }
            b += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::core::exec_aten::testing_util::tensor_factory::TensorFactory;
    use crate::runtime::core::exec_aten::util::tensor_util::K_TENSOR_DIMENSION_LIMIT;

    // PORT-NOTE: `check_linear_args` / `get_linear_out_target_size` have no call
    // site in the ported runtime (no op_linear ported), so no op test transitively
    // covers them. They are pure tensor-metadata helpers; these focused tests pin
    // their C++ semantics directly (matmul_ops_util.cpp).

    // [spec:et:sem:matmul-ops-util.torch.executor.check-linear-args-fn/test]
    #[test]
    fn matmul_ops_util_test_check_linear_args() {
        crate::runtime::platform::platform::pal_init();
        let tf = TensorFactory::<f32>::new();
        // in: [B, K], mat2 (weight): [N, K] (rank 2), out: [B, N]
        let in_ = tf.zeros_default(vec![3, 4]);
        let mat2 = tf.zeros_default(vec![5, 4]);
        let out = tf.zeros_default(vec![3, 5]);
        assert!(check_linear_args(&in_, &mat2, &out));

        // out.dim() must equal in.dim()
        let bad_out = tf.zeros_default(vec![3, 5, 1]);
        assert!(!check_linear_args(&in_, &mat2, &bad_out));

        // in.dim() must be >= 2
        let in_1d = tf.zeros_default(vec![4]);
        let out_1d = tf.zeros_default(vec![5]);
        assert!(!check_linear_args(&in_1d, &mat2, &out_1d));

        // mat2 must be rank 2
        let mat2_3d = tf.zeros_default(vec![5, 4, 1]);
        assert!(!check_linear_args(&in_, &mat2_3d, &out));

        // in.size(in.dim()-1) must equal mat2.size(1)
        let mat2_wrong = tf.zeros_default(vec![5, 7]);
        assert!(!check_linear_args(&in_, &mat2_wrong, &out));
    }

    // [spec:et:sem:matmul-ops-util.torch.executor.get-linear-out-target-size-fn/test]
    #[test]
    fn matmul_ops_util_test_get_linear_out_target_size() {
        let tf = TensorFactory::<f32>::new();
        // mat1: [2, 3, 4], mat2 (weight): [5, 4] -> out ndim = 3, sizes = [2, 3, 5]
        // (last dim from mat2.size(0), leading dims copied from mat1)
        let mat1 = tf.zeros_default(vec![2, 3, 4]);
        let mat2 = tf.zeros_default(vec![5, 4]);

        let mut out_sizes: [SizesType; K_TENSOR_DIMENSION_LIMIT] = [0; K_TENSOR_DIMENSION_LIMIT];
        let mut out_ndim: usize = 0;
        unsafe {
            get_linear_out_target_size(&mat1, &mat2, out_sizes.as_mut_ptr(), &mut out_ndim);
        }
        assert_eq!(out_ndim, 3);
        assert_eq!(&out_sizes[0..3], &[2, 3, 5]);
    }
}
