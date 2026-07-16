//! Literal port of kernels/portable/cpu/op_randn.cpp.

use crate::runtime::core::array_ref::IntArrayRef;
use crate::runtime::core::error::Error;
use crate::runtime::core::exec_aten::util::tensor_util::resize_tensor;
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

// PORT-NOTE: C++ `Tensor& out` / returned `Tensor&` become `&'a Tensor` (interior
// mutation through the non-owning handle's raw pointer).
//
// PORT-NOTE (nondeterminism): the C++ seeds `std::mt19937` from
// `std::random_device` on every call, so results are nondeterministic and not
// reproducible. As in op_rand.rs, the mt19937 engine and the
// `normal_distribution` are ported inline here (per-translation-unit `<random>`
// dependency, one module per source file, no new shared module). The algorithm
// (mt19937 + normal_distribution N(0,1)) and the fresh-per-call random_device
// seeding are mirrored; the exact bit sequence of a specific standard-library
// build is not reproduced.

// [spec:et:def:op-randn.torch.executor.native.randn-out-fn]
// [spec:et:sem:op-randn.torch.executor.native.randn-out-fn]
pub fn randn_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    sizes: IntArrayRef,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    // (void)ctx;

    let mut rng = Mt19937::new(random_device());
    let mut dist = NormalDistribution::new(0.0, 1.0);

    // Resize for dynamic shape
    crate::et_kernel_check_msg!(
        ctx,
        resize_tensor(out, sizes) == Error::Ok,
        InvalidArgument,
        out,
        "Failed to resize output tensor."
    );

    crate::et_switch_floathbf16_types!(out.scalar_type(), ctx, "randn.out", CTYPE, {
        let data_out = out.mutable_data_ptr::<CTYPE>();
        for i in 0..out.numel() {
            unsafe {
                *data_out.add(i as usize) = <CTYPE as FromF64>::from_f64(dist.sample(&mut rng));
            }
        }
    });

    out
}

// PORT-NOTE: C++ `static_cast<CTYPE>(dist(rng))` narrows the drawn `double` to
// the FLOATHBF16 output ctype (Float, Double, Half, BFloat16). Local `FromF64`
// reproduces the per-ctype `static_cast`.
trait FromF64 {
    fn from_f64(v: f64) -> Self;
}
impl FromF64 for f32 {
    fn from_f64(v: f64) -> Self {
        v as f32
    }
}
impl FromF64 for f64 {
    fn from_f64(v: f64) -> Self {
        v
    }
}
impl FromF64 for crate::runtime::core::portable_type::Half {
    fn from_f64(v: f64) -> Self {
        crate::runtime::core::portable_type::Half::from_f64(v)
    }
}
impl FromF64 for crate::runtime::core::portable_type::BFloat16 {
    fn from_f64(v: f64) -> Self {
        crate::runtime::core::portable_type::BFloat16::from_f64(v)
    }
}

// PORT-NOTE: inline `std::mt19937` — the standard 32-bit Mersenne Twister with
// the canonical parameters. Mirrors `std::mt19937 rng(seed)` and `rng()`.
struct Mt19937 {
    mt: [u32; 624],
    index: usize,
}

impl Mt19937 {
    const N: usize = 624;
    const M: usize = 397;
    const MATRIX_A: u32 = 0x9908_b0df;
    const UPPER_MASK: u32 = 0x8000_0000;
    const LOWER_MASK: u32 = 0x7fff_ffff;

    fn new(seed: u32) -> Self {
        let mut mt = [0u32; Self::N];
        mt[0] = seed;
        for i in 1..Self::N {
            mt[i] = (1_812_433_253u32)
                .wrapping_mul(mt[i - 1] ^ (mt[i - 1] >> 30))
                .wrapping_add(i as u32);
        }
        Mt19937 { mt, index: Self::N }
    }

    fn generate(&mut self) {
        for i in 0..Self::N {
            let y =
                (self.mt[i] & Self::UPPER_MASK) | (self.mt[(i + 1) % Self::N] & Self::LOWER_MASK);
            let mut next = self.mt[(i + Self::M) % Self::N] ^ (y >> 1);
            if y & 1 != 0 {
                next ^= Self::MATRIX_A;
            }
            self.mt[i] = next;
        }
        self.index = 0;
    }

    fn next_u32(&mut self) -> u32 {
        if self.index >= Self::N {
            self.generate();
        }
        let mut y = self.mt[self.index];
        self.index += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^= y >> 18;
        y
    }
}

// PORT-NOTE: `std::random_device` stand-in — draws OS-seeded entropy from the
// standard library's `RandomState`, giving fresh non-reproducible seeds per
// call (mirrors `std::random_device()()`).
fn random_device() -> u32 {
    use std::hash::{BuildHasher, Hasher};
    let mut h = std::collections::hash_map::RandomState::new().build_hasher();
    h.write_usize(&h as *const _ as usize);
    h.finish() as u32
}

fn generate_canonical(rng: &mut Mt19937) -> f64 {
    let hi = (rng.next_u32() >> 5) as u64; // 27 bits
    let lo = (rng.next_u32() >> 6) as u64; // 26 bits
    ((hi * 67_108_864u64 + lo) as f64) * (1.0 / 9_007_199_254_740_992.0)
}

// PORT-NOTE: inline `std::normal_distribution<double>` (mean, stddev). Uses the
// Marsaglia polar method with a cached second variate, as libstdc++ does; each
// standard-normal draw is scaled/shifted to N(mean, stddev).
struct NormalDistribution {
    mean: f64,
    stddev: f64,
    saved: Option<f64>,
}

impl NormalDistribution {
    fn new(mean: f64, stddev: f64) -> Self {
        NormalDistribution {
            mean,
            stddev,
            saved: None,
        }
    }

    fn sample(&mut self, rng: &mut Mt19937) -> f64 {
        let ret;
        if let Some(saved) = self.saved.take() {
            ret = saved;
        } else {
            let mut x: f64;
            let mut y: f64;
            let mut r2: f64;
            loop {
                x = 2.0 * generate_canonical(rng) - 1.0;
                y = 2.0 * generate_canonical(rng) - 1.0;
                r2 = x * x + y * y;
                if !(r2 > 1.0 || r2 == 0.0) {
                    break;
                }
            }
            let mult = (-2.0 * r2.ln() / r2).sqrt();
            self.saved = Some(x * mult);
            ret = y * mult;
        }
        self.mean + self.stddev * ret
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::core::array_ref::ArrayRef;
    use crate::runtime::core::exec_aten::testing_util::tensor_factory::{
        FactoryValue, TensorFactory,
    };
    use crate::runtime::core::exec_aten::util::scalar_type_util::CppTypeToScalarType;
    use crate::runtime::core::memory_allocator::{MemoryAllocator, MemoryAllocatorBase};
    use crate::runtime::core::portable_type::{BFloat16, Half};

    fn context() -> KernelRuntimeContext<'static> {
        crate::runtime::platform::platform::pal_init();
        KernelRuntimeContext::new(
            crate::extension::module::module::null_event_tracer(),
            core::ptr::null_mut::<MemoryAllocator>() as *mut dyn MemoryAllocatorBase,
        )
    }

    fn op_randn_out(sizes: IntArrayRef, out: &Tensor) {
        let mut ctx = context();
        randn_out(&mut ctx, sizes, out);
    }

    trait ToF64: Copy {
        fn to_f64(self) -> f64;
    }
    impl ToF64 for f32 {
        fn to_f64(self) -> f64 {
            self as f64
        }
    }
    impl ToF64 for f64 {
        fn to_f64(self) -> f64 {
            self
        }
    }
    impl ToF64 for Half {
        fn to_f64(self) -> f64 {
            self.to_f32() as f64
        }
    }
    impl ToF64 for BFloat16 {
        fn to_f64(self) -> f64 {
            self.to_f32() as f64
        }
    }

    fn test_randn<T>(sizes: &[i64])
    where
        T: CppTypeToScalarType + FactoryValue + ToF64,
    {
        let tf = TensorFactory::<T>::new();

        // Tensor factory wants int32 scales, op kernel wants int64.
        let sizes_i32: Vec<i32> = sizes.iter().map(|&s| s as i32).collect();
        let out = tf.zeros_default(sizes_i32);

        op_randn_out(ArrayRef::from_raw_parts(sizes.as_ptr(), sizes.len()), &out);

        // Check mean and standard deviation. To avoid flaky CI, test pretty
        // loosely.
        let out_data = out.const_data_ptr::<T>();
        let numel = out.numel() as usize;
        let mut acc = 0.0f64;
        for i in 0..numel {
            acc += unsafe { *out_data.add(i) }.to_f64();
        }
        let mean = acc / numel as f64;
        let mut var_acc = 0.0f64;
        for i in 0..numel {
            let n = unsafe { *out_data.add(i) }.to_f64();
            var_acc += (n - mean).powi(2);
        }
        let var = var_acc / numel as f64;
        let stdev = var.sqrt();

        assert!(mean.abs() <= 5.0 / (numel as f64).sqrt());
        assert!((stdev - 1.0).abs() <= 0.1);
        assert!(stdev > 0.0);
    }

    // [spec:et:sem:op-randn.torch.executor.native.randn-out-fn/test]
    #[test]
    fn op_randn_test_smoke_test() {
        let sizes = vec![2i64, 3, 4, 128];
        // ET_FORALL_FLOATHBF16_TYPES: Float,Double,Half,BFloat16.
        test_randn::<f32>(&sizes);
        test_randn::<f64>(&sizes);
        test_randn::<Half>(&sizes);
        test_randn::<BFloat16>(&sizes);
    }

    // [spec:et:sem:op-randn.torch.executor.native.randn-out-fn/test]
    #[test]
    fn op_randn_test_rank() {
        let mut sizes = vec![1024i64];
        for i in 0..4i64 {
            sizes.push(i + 1);
            test_randn::<f32>(&sizes);
        }
    }
}
