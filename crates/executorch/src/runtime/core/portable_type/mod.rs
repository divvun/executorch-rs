pub mod bits_types;
pub mod device;
pub mod qint_types;
pub mod scalar;
pub mod scalar_type;
pub mod tensor;
pub mod tensor_impl;
pub mod tensor_options;

// The vendored c10 Half/BFloat16 headers are excluded from this port; map them
// to the `half` crate types, matching PORTING.md's type table.
pub use half::bf16 as BFloat16;
pub use half::f16 as Half;

// complex<T> (portable_type/complex.h -> c10::complex) placeholder for the
// ComplexHalf/ComplexFloat/ComplexDouble ScalarType indices. The c10 complex
// type is not part of this module set; this minimal `re`/`im` struct stands in
// for the header set the ScalarType enum implies (no arithmetic ported).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Complex<T> {
    pub real: T,
    pub imag: T,
}

pub type ComplexHalf = Complex<Half>;
pub type ComplexFloat = Complex<f32>;
pub type ComplexDouble = Complex<f64>;

// PORT-NOTE: bfloat16.h / half.h are excluded from this port and mapped to the
// `half` crate's `bf16` / `f16` (see the `pub use` above). Their C++ tests
// (bfloat16_test.cpp, half_test.cpp) have no dedicated Rust module, so the
// ported suites live here on the re-export. There are no `et:sem` rules for the
// crate stand-ins, so no `/test` facets are attached (per "never annotate
// aspirationally"); c10-specific internals that do not map are called out
// per-test below.
#[cfg(test)]
mod bfloat16_tests {
    use super::BFloat16;

    // C++ test-local helper: assemble a float from sign/exponent/fraction bits.
    fn float_from_bytes(sign: u32, exponent: u32, fraction: u32) -> f32 {
        let mut bytes: u32 = 0;
        bytes |= sign;
        bytes <<= 8;
        bytes |= exponent;
        bytes <<= 23;
        bytes |= fraction;
        f32::from_bits(bytes)
    }

    // C++ test-local helper: opposite of f32_from_bits (truncate low 16 bits).
    fn bits_from_f32(src: f32) -> u16 {
        (src.to_bits() >> 16) as u16
    }

    // PORT-NOTE: c10 `internal::f32_from_bits(x)` reconstructs a float from the
    // upper-16-bit bfloat pattern; maps to `BFloat16::from_bits(x).to_f32()`.
    fn f32_from_bits(x: u16) -> f32 {
        BFloat16::from_bits(x).to_f32()
    }

    // PORT-NOTE: c10 `internal::round_to_nearest_even(f)` rounds a float to the
    // nearest-even bfloat bit pattern; `half`'s `bf16::from_f32` performs the
    // same round-to-nearest-even, so it maps to
    // `BFloat16::from_f32(f).to_bits()`.
    fn round_to_nearest_even(f: f32) -> u16 {
        BFloat16::from_f32(f).to_bits()
    }

    #[test]
    fn bfloat16_conversion_float_to_bfloat16_and_back() {
        let mut input = [0.0f32; 100];
        for i in 0..100 {
            input[i] = i as f32 + 1.25;
        }

        let mut bfloats = [0u16; 100];
        let mut out = [0.0f32; 100];

        for i in 0..100 {
            bfloats[i] = bits_from_f32(input[i]);
            out[i] = f32_from_bits(bfloats[i]);

            // The relative error should be less than 1/(2^7) since BFloat16
            // has 7 bits mantissa.
            assert!((out[i] - input[i]).abs() / input[i] <= 1.0 / 128.0);
        }
    }

    #[test]
    fn bfloat16_conversion_float_to_bfloat16_rne_and_back() {
        let mut input = [0.0f32; 100];
        for i in 0..100 {
            input[i] = i as f32 + 1.25;
        }

        let mut bfloats = [0u16; 100];
        let mut out = [0.0f32; 100];

        for i in 0..100 {
            bfloats[i] = round_to_nearest_even(input[i]);
            out[i] = f32_from_bits(bfloats[i]);

            assert!((out[i] - input[i]).abs() / input[i] <= 1.0 / 128.0);
        }
    }

    #[test]
    fn bfloat16_conversion_nan() {
        let in_nan = float_from_bytes(0, 0xFF, 0x7FFFFF);
        assert!(in_nan.is_nan());

        let a = BFloat16::from_f32(in_nan);
        let out = f32_from_bits(a.to_bits());

        assert!(out.is_nan());
    }

    #[test]
    fn bfloat16_conversion_inf() {
        let in_inf = float_from_bytes(0, 0xFF, 0);
        assert!(in_inf.is_infinite());

        let a = BFloat16::from_f32(in_inf);
        let out = f32_from_bits(a.to_bits());

        assert!(out.is_infinite());
    }

    // Mirrors gtest `EXPECT_FLOAT_EQ`: equal within 4 ULPs (not exact). The
    // input denorm_min flushes to 0 in bfloat16, so `out` is 0.0 while `input`
    // is the smallest subnormal (bit pattern 0x00000001) — 1 ULP from 0.0, which
    // `EXPECT_FLOAT_EQ` treats as equal.
    fn float_almost_equals(a: f32, b: f32) -> bool {
        const K_MAX_ULPS: i32 = 4;
        // gtest FloatingPoint: sign-and-magnitude → biased (monotonic) ordering.
        let sam_to_biased = |bits: u32| -> u32 {
            const SIGN_BIT: u32 = 1u32 << 31;
            if (SIGN_BIT & bits) != 0 {
                !bits + 1
            } else {
                SIGN_BIT | bits
            }
        };
        if a.is_nan() || b.is_nan() {
            return false;
        }
        let ba = sam_to_biased(a.to_bits());
        let bb = sam_to_biased(b.to_bits());
        let dist = if ba >= bb { ba - bb } else { bb - ba };
        dist <= K_MAX_ULPS as u32
    }

    #[test]
    fn bfloat16_conversion_smallest_denormal() {
        // The smallest non-zero subnormal float.
        let input = f32::from_bits(1);
        let a = BFloat16::from_f32(input);
        let out = f32_from_bits(a.to_bits());

        assert!(float_almost_equals(input, out));
    }

    #[test]
    fn bfloat16_math_addition() {
        // input bits: 0 | 10000000 | 1001000... = 3.125
        let input = float_from_bytes(0, 0, 0x40480000);
        // expected bits: 0 | 10000001 | 1001000... = 6.25
        let expected = float_from_bytes(0, 0, 0x40c80000);

        let mut b = BFloat16::from_bits(bits_from_f32(input));
        b = b + b;

        let res = f32_from_bits(b.to_bits());
        assert_eq!(res, expected);
    }

    #[test]
    fn bfloat16_math_subtraction() {
        // input bits: 0 | 10000001 | 1110100... = 7.625
        let input = float_from_bytes(0, 0, 0x40f40000);
        // expected bits: 0 | 10000000 | 0101000... = 2.625
        let expected = float_from_bytes(0, 0, 0x40280000);

        let mut b = BFloat16::from_bits(bits_from_f32(input));
        // PORT-NOTE: C++ `b = b - 5` uses BFloat16 - int, which promotes the
        // BFloat16 to float, subtracts, and converts back. `half`'s `bf16` has no
        // mixed `bf16 - i32` operator; the promotion is expressed explicitly here
        // via `from_f32(b.to_f32() - 5.0)`, matching the c10 semantics.
        b = BFloat16::from_f32(b.to_f32() - 5.0);

        let res = f32_from_bits(b.to_bits());
        assert_eq!(res, expected);
    }

    // PORT-NOTE: `std::nextafter(BFloat16, BFloat16)` is a c10-specific overload
    // (bfloat16-math.h). `half`'s `bf16` provides no `nextafter`, so this death-
    // free bit-equality suite over zero/-zero has no crate mapping. Ported for
    // completeness and `#[ignore]`d; the assertion (nextafter preserves the sign
    // of zero when from == to) is documented but cannot run against the stand-in.
    #[test]
    #[ignore]
    fn bfloat16_math_next_after_zero() {
        let zero = BFloat16::from_f32(0.0);
        let neg_zero = BFloat16::from_f32(-0.0);

        let check_nextafter = |from: BFloat16, _to: BFloat16, expected: BFloat16| {
            // nextafter(from, to) for from == to returns from unchanged.
            let actual = from;
            assert_eq!(actual.to_bits() ^ expected.to_bits(), 0u16);
        };
        check_nextafter(zero, zero, zero);
        check_nextafter(zero, neg_zero, neg_zero);
        check_nextafter(neg_zero, zero, zero);
        check_nextafter(neg_zero, neg_zero, neg_zero);
    }

    fn binary_to_float(bytes: u32) -> f32 {
        f32::from_bits(bytes)
    }

    // INSTANTIATE_TEST_SUITE_P(BFloat16TestInstantiation, ...): the parameterized
    // RNE cases, ported as a table-driven loop.
    #[test]
    fn bfloat16_test_rne_test() {
        let params: [(u32, u16); 5] = [
            (0x3F848000, 0x3F84),
            (0x3F848010, 0x3F85),
            (0x3F850000, 0x3F85),
            (0x3F858000, 0x3F86),
            (0x3FFF8000, 0x4000),
        ];
        for (input, rne) in params {
            let value = binary_to_float(input);
            let rounded = round_to_nearest_even(value);
            assert_eq!(rne, rounded);
        }
    }
}

#[cfg(test)]
mod half_tests {
    use super::Half;

    // According to the half-precision precision limitations, the max precision
    // error for a half in [2^n, 2^(n+1)] is 2^(n-10).
    fn tolerance_float16(f: f32) -> f32 {
        2f32.powi((f.abs().log2() as i32) - 10)
    }

    fn close_enough_float16(a: f32, b: f32) -> bool {
        (a - b).abs() <= tolerance_float16(a.abs().max(b.abs()))
    }

    // PORT-NOTE: c10 `Half` implicitly promotes to `float` for arithmetic with
    // `Half`/`float`/`double`/`int32_t`/`int64_t`. `half`'s `f16` only implements
    // `f16 op f16`, so each `ah <op> X` below is expressed as float arithmetic on
    // the promoted operands, exactly the value c10 computes (Half → float). The
    // compound-assignment tests store the result back via `Half::from_f32`.

    /// Arithmetic with Halfs

    #[test]
    fn half_test_arithmetic_half_add() {
        let af: f32 = 104.35;
        let mut ah = Half::from_f32(af);
        let bf: f32 = 72.5;
        let bh = Half::from_f32(bf);
        assert!(close_enough_float16((ah + bh).to_f32(), af + bf));
        ah = Half::from_f32(ah.to_f32() + bh.to_f32());
        let af = af + bf;
        assert!(close_enough_float16(ah.to_f32(), af));
    }

    #[test]
    fn half_test_arithmetic_half_sub() {
        let af: f32 = 31.4;
        let mut ah = Half::from_f32(af);
        let bf: f32 = 20.5;
        let bh = Half::from_f32(bf);
        assert!(close_enough_float16((ah - bh).to_f32(), af - bf));
        ah = Half::from_f32(ah.to_f32() - bh.to_f32());
        let af = af - bf;
        assert!(close_enough_float16(ah.to_f32(), af));
    }

    #[test]
    fn half_test_arithmetic_half_mul() {
        let af: f32 = 85.5;
        let mut ah = Half::from_f32(af);
        let bf: f32 = 17.5;
        let bh = Half::from_f32(bf);
        assert!(close_enough_float16((ah * bh).to_f32(), af * bf));
        ah = Half::from_f32(ah.to_f32() * bh.to_f32());
        let af = af * bf;
        assert!(close_enough_float16(ah.to_f32(), af));
    }

    #[test]
    fn half_test_arithmetic_half_div() {
        let af: f32 = 96.9;
        let mut ah = Half::from_f32(af);
        let bf: f32 = 12.5;
        let bh = Half::from_f32(bf);
        assert!(close_enough_float16((ah / bh).to_f32(), af / bf));
        ah = Half::from_f32(ah.to_f32() / bh.to_f32());
        let af = af / bf;
        assert!(close_enough_float16(ah.to_f32(), af));
    }

    /// Arithmetic with floats

    #[test]
    fn half_test_arithmetic_float_add() {
        let af: f32 = 104.35;
        let ah = Half::from_f32(af);
        let b: f32 = 72.5;
        assert!(close_enough_float16(ah.to_f32() + b, af + b));
        assert!(close_enough_float16(b + ah.to_f32(), b + af));
    }

    #[test]
    fn half_test_arithmetic_float_sub() {
        let af: f32 = 31.4;
        let ah = Half::from_f32(af);
        let b: f32 = 20.5;
        assert!(close_enough_float16(ah.to_f32() - b, af - b));
        assert!(close_enough_float16(b - ah.to_f32(), b - af));
    }

    #[test]
    fn half_test_arithmetic_float_mul() {
        let af: f32 = 85.5;
        let ah = Half::from_f32(af);
        let b: f32 = 17.5;
        assert!(close_enough_float16(ah.to_f32() * b, af * b));
        assert!(close_enough_float16(b * ah.to_f32(), b * af));
    }

    #[test]
    fn half_test_arithmetic_float_div() {
        let af: f32 = 96.9;
        let ah = Half::from_f32(af);
        let b: f32 = 12.5;
        assert!(close_enough_float16(ah.to_f32() / b, af / b));
        assert!(close_enough_float16(b / ah.to_f32(), b / af));
    }

    /// Arithmetic with doubles

    #[test]
    fn half_test_arithmetic_double_add() {
        let af: f32 = 104.35;
        let ah = Half::from_f32(af);
        let b: f64 = 72.5;
        assert!(close_enough_float16(
            (ah.to_f32() as f64 + b) as f32,
            (af as f64 + b) as f32
        ));
        assert!(close_enough_float16(
            (b + ah.to_f32() as f64) as f32,
            (b + af as f64) as f32
        ));
    }

    #[test]
    fn half_test_arithmetic_double_sub() {
        let af: f32 = 31.4;
        let ah = Half::from_f32(af);
        let b: f64 = 20.5;
        assert!(close_enough_float16(
            (ah.to_f32() as f64 - b) as f32,
            (af as f64 - b) as f32
        ));
        assert!(close_enough_float16(
            (b - ah.to_f32() as f64) as f32,
            (b - af as f64) as f32
        ));
    }

    #[test]
    fn half_test_arithmetic_double_mul() {
        let af: f32 = 85.5;
        let ah = Half::from_f32(af);
        let b: f64 = 17.5;
        assert!(close_enough_float16(
            (ah.to_f32() as f64 * b) as f32,
            (af as f64 * b) as f32
        ));
        assert!(close_enough_float16(
            (b * ah.to_f32() as f64) as f32,
            (b * af as f64) as f32
        ));
    }

    #[test]
    fn half_test_arithmetic_double_div() {
        let af: f32 = 96.9;
        let ah = Half::from_f32(af);
        let b: f64 = 12.5;
        assert!(close_enough_float16(
            (ah.to_f32() as f64 / b) as f32,
            (af as f64 / b) as f32
        ));
        assert!(close_enough_float16(
            (b / ah.to_f32() as f64) as f32,
            (b / af as f64) as f32
        ));
    }

    /// Arithmetic with ints

    #[test]
    fn half_test_arithmetic_int32_add() {
        let af: f32 = 104.35;
        let ah = Half::from_f32(af);
        let b: i32 = 72;
        assert!(close_enough_float16(ah.to_f32() + b as f32, af + b as f32));
        assert!(close_enough_float16(b as f32 + ah.to_f32(), b as f32 + af));
    }

    #[test]
    fn half_test_arithmetic_int32_sub() {
        let af: f32 = 31.4;
        let ah = Half::from_f32(af);
        let b: i32 = 20;
        assert!(close_enough_float16(ah.to_f32() - b as f32, af - b as f32));
        assert!(close_enough_float16(b as f32 - ah.to_f32(), b as f32 - af));
    }

    #[test]
    fn half_test_arithmetic_int32_mul() {
        let af: f32 = 85.5;
        let ah = Half::from_f32(af);
        let b: i32 = 17;
        assert!(close_enough_float16(ah.to_f32() * b as f32, af * b as f32));
        assert!(close_enough_float16(b as f32 * ah.to_f32(), b as f32 * af));
    }

    #[test]
    fn half_test_arithmetic_int32_div() {
        let af: f32 = 96.9;
        let ah = Half::from_f32(af);
        let b: i32 = 12;
        assert!(close_enough_float16(ah.to_f32() / b as f32, af / b as f32));
        assert!(close_enough_float16(b as f32 / ah.to_f32(), b as f32 / af));
    }

    /// Arithmetic with int64_t

    #[test]
    fn half_test_arithmetic_int64_add() {
        let af: f32 = 104.35;
        let ah = Half::from_f32(af);
        let b: i64 = 72;
        assert!(close_enough_float16(ah.to_f32() + b as f32, af + b as f32));
        assert!(close_enough_float16(b as f32 + ah.to_f32(), b as f32 + af));
    }

    #[test]
    fn half_test_arithmetic_int64_sub() {
        let af: f32 = 31.4;
        let ah = Half::from_f32(af);
        let b: i64 = 20;
        assert!(close_enough_float16(ah.to_f32() - b as f32, af - b as f32));
        assert!(close_enough_float16(b as f32 - ah.to_f32(), b as f32 - af));
    }

    #[test]
    fn half_test_arithmetic_int64_mul() {
        let af: f32 = 85.5;
        let ah = Half::from_f32(af);
        let b: i64 = 17;
        assert!(close_enough_float16(ah.to_f32() * b as f32, af * b as f32));
        assert!(close_enough_float16(b as f32 * ah.to_f32(), b as f32 * af));
    }

    #[test]
    fn half_test_arithmetic_int64_div() {
        let af: f32 = 96.9;
        let ah = Half::from_f32(af);
        let b: i64 = 12;
        assert!(close_enough_float16(ah.to_f32() / b as f32, af / b as f32));
        assert!(close_enough_float16(b as f32 / ah.to_f32(), b as f32 / af));
    }
}
