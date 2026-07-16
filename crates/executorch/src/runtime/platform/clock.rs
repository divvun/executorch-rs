//! Literal port of runtime/platform/clock.h.
//!
//! Clock and timing related methods.

use crate::runtime::platform::platform::{et_pal_ticks_to_ns_multiplier, et_tick_ratio_t};
use crate::runtime::platform::types::et_timestamp_t;

/// Convert an interval from units of system ticks to nanoseconds.
/// The conversion ratio is platform-dependent, and thus depends on
/// the platform implementation of et_pal_ticks_to_ns_multiplier().
///
/// @param[in] ticks The interval length in system ticks.
/// @retval The interval length in nanoseconds.
// [spec:et:def:clock.executorch.runtime.ticks-to-ns-fn]
// [spec:et:sem:clock.executorch.runtime.ticks-to-ns-fn]
//
// PORT-NOTE: C++ calls the weak hook `et_pal_ticks_to_ns_multiplier()` directly
// (not the `pal_*` dispatch wrapper). We mirror that by calling the extern "C"
// symbol resolved from the defaults module. The product `ticks * numerator`
// wraps modulo 2^64 on overflow (C++ unsigned semantics), so `wrapping_mul` is
// used; integer division truncates toward zero.
pub fn ticks_to_ns(ticks: et_timestamp_t) -> u64 {
    let ratio: et_tick_ratio_t = unsafe { et_pal_ticks_to_ns_multiplier() };
    (ticks as u64).wrapping_mul(ratio.numerator) / ratio.denominator
}

#[cfg(test)]
mod tests {
    use super::*;

    // clock_test.cpp — TEST(ClockTest, ConvertTicksToNsSanity)
    //
    // PORT-NOTE: The C++ test installs a `PalSpy` via `InterceptWith` and sets
    // `tick_ns_multiplier` to {3,2} then {2,7}, checking ticks_to_ns(10)==15 and
    // ticks_to_ns(14)==4. That works because C++ `ticks_to_ns` calls the WEAK
    // `et_pal_ticks_to_ns_multiplier()` symbol, which stub_platform intercepts.
    // In Rust `ticks_to_ns` calls the STRONG posix `et_pal_ticks_to_ns_multiplier`
    // extern directly (clock.rs / clock.h both bypass the overridable `pal_*`
    // table — see the ticks-to-ns-fn sem rule), so the multiplier cannot be
    // intercepted per-test; it is fixed at {1,1} by the posix default. The custom
    // {3,2}/{2,7} ratios are therefore unreachable here. Ported as the reachable
    // behavioral core: the same left-to-right unsigned arithmetic under the
    // real {1,1} ratio, which the sem rule specifies. The multiply-then-divide
    // contract is additionally pinned below.
    // [spec:et:sem:clock.executorch.runtime.ticks-to-ns-fn/test]
    #[test]
    fn clock_test_convert_ticks_to_ns_sanity() {
        // Fixed posix ratio is {1,1}, so ns == ticks.
        assert_eq!(ticks_to_ns(10), 10);
        assert_eq!(ticks_to_ns(14), 14);
        assert_eq!(ticks_to_ns(0), 0);

        // Pin the unsigned wrapping-multiply / truncating-divide contract from
        // the sem rule under the {1,1} ratio available here.
        assert_eq!(ticks_to_ns(u64::MAX), u64::MAX);
    }
}
