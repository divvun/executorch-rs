//! Literal port of backends/xnnpack/runtime/XNNWeightsCacheManager.cpp +
//! backends/xnnpack/runtime/XNNWeightsCacheManager.h.
//!
//! PORT-NOTE: Depends on the XNNPACK C API (via `XNNWeightsCache`), so the whole
//! module is gated behind the `xnnpack` feature.
#![cfg(feature = "xnnpack")]

use super::XNNWeightsCache::XNNWeightsCache;
use crate::runtime::core::error::Error;
use crate::runtime::core::result::Result;

use std::collections::HashMap;
use std::sync::{Arc, Mutex, Weak};

/// One `XNNWeightsCache` per cache file path. Mirrors `XNNWorkspaceManager`'s
/// PerModel pattern with `weak_ptr` so instances live as long as the executors
/// owning them.
// [spec:et:def:xnn-weights-cache-manager.executorch.backends.xnnpack.xnn-weights-cache-manager]
//
// PORT-NOTE: `weak_ptr<XNNWeightsCache>` → `std::sync::Weak<XNNWeightsCache>`,
// `shared_ptr` → `Arc`. The C++ has two separate mutexes (`meta_mutex_` and
// `empty_path_mutex_`); both are kept as distinct `std::sync::Mutex`es guarding
// their respective maps/slots to preserve the lock-separation rationale. The
// deleted copy/move-assign is the Rust default (non-`Clone`, owned in place), so
// its markers collapse onto this struct.
// [spec:et:def:xnn-weights-cache-manager.executorch.backends.xnnpack.xnn-weights-cache-manager.operator-fn]
// [spec:et:sem:xnn-weights-cache-manager.executorch.backends.xnnpack.xnn-weights-cache-manager.operator-fn]
pub struct XNNWeightsCacheManager {
    caches_: Mutex<HashMap<String, Weak<XNNWeightsCache>>>,
    empty_path_cache_: Mutex<Weak<XNNWeightsCache>>,
}

impl XNNWeightsCacheManager {
    // [spec:et:def:xnn-weights-cache-manager.executorch.backends.xnnpack.xnn-weights-cache-manager.xnn-weights-cache-manager-fn]
    // [spec:et:sem:xnn-weights-cache-manager.executorch.backends.xnnpack.xnn-weights-cache-manager.xnn-weights-cache-manager-fn]
    pub fn new() -> Self {
        XNNWeightsCacheManager {
            caches_: Mutex::new(HashMap::new()),
            empty_path_cache_: Mutex::new(Weak::new()),
        }
    }

    /// Shared `XNNWeightsCache` for `cache_file_path`.
    // [spec:et:def:xnn-weights-cache-manager.executorch.backends.xnnpack.xnn-weights-cache-manager.get-or-create-fn]
    // [spec:et:sem:xnn-weights-cache-manager.executorch.backends.xnnpack.xnn-weights-cache-manager.get-or-create-fn]
    pub fn get_or_create(&self, cache_file_path: &str) -> Result<Arc<XNNWeightsCache>> {
        // Empty path → one shared heap-only instance.
        if cache_file_path.is_empty() {
            let mut empty = self.empty_path_cache_.lock().unwrap();
            if let Some(live) = empty.upgrade() {
                return Ok(live);
            }
            let mut cache = XNNWeightsCache::new();
            cache.wire_provider();
            let cache = Arc::new(cache);
            *empty = Arc::downgrade(&cache);
            return Ok(cache);
        }

        let mut caches = self.caches_.lock().unwrap();
        if let Some(weak) = caches.get(cache_file_path) {
            if let Some(live) = weak.upgrade() {
                return Ok(live);
            }
            caches.remove(cache_file_path);
        }

        let mut cache = XNNWeightsCache::new();
        cache.wire_provider();
        // Set path before publishing into the map so concurrent callers observe
        // a fully initialized instance.
        cache.set_packed_cache_path(cache_file_path);
        let cache = Arc::new(cache);
        caches.insert(cache_file_path.to_string(), Arc::downgrade(&cache));
        Ok(cache)
    }

    /// Walk live caches and call `save_packed_index()` on each.
    // [spec:et:def:xnn-weights-cache-manager.executorch.backends.xnnpack.xnn-weights-cache-manager.save-all-fn]
    // [spec:et:sem:xnn-weights-cache-manager.executorch.backends.xnnpack.xnn-weights-cache-manager.save-all-fn]
    //
    // PORT-NOTE: `save_packed_index` needs `&mut self` on the cache, but the
    // manager only holds `Arc<XNNWeightsCache>` (shared). The C++ obtains
    // exclusive access purely via the per-instance `mutex()` while aliasing
    // through the shared_ptr. To mirror that here we take the cache's mutex
    // guard (matching the lock order) and mutate through a raw pointer, which is
    // the same aliasing contract the C++ relies on.
    pub fn save_all(&self) -> Error {
        // Snapshot live shared_ptrs under caches_ lock, then release it before
        // the per-instance save.
        let mut live: Vec<Arc<XNNWeightsCache>> = Vec::new();
        {
            let mut caches = self.caches_.lock().unwrap();
            live.reserve(caches.len());
            let mut expired: Vec<String> = Vec::new();
            for (path, weak) in caches.iter() {
                if let Some(cache) = weak.upgrade() {
                    live.push(cache);
                } else {
                    expired.push(path.clone());
                }
            }
            for path in expired {
                caches.remove(&path);
            }
        }

        let mut first_err = Error::Ok;
        for cache in live.iter() {
            let _lock = cache.mutex().lock().unwrap();
            // SAFETY: exclusive access is established by holding the cache's
            // per-instance mutex, mirroring the C++ caller-owned locking
            // contract; no other thread mutates the cache while this guard is
            // held.
            let cache_mut = unsafe { &mut *(Arc::as_ptr(cache) as *mut XNNWeightsCache) };
            let err = cache_mut.save_packed_index();
            if err != Error::Ok && first_err == Error::Ok {
                first_err = err;
            }
        }
        first_err
    }

    /// Test-only: count of live (non-expired) entries.
    // [spec:et:def:xnn-weights-cache-manager.executorch.backends.xnnpack.xnn-weights-cache-manager.live-count-fn]
    // [spec:et:sem:xnn-weights-cache-manager.executorch.backends.xnnpack.xnn-weights-cache-manager.live-count-fn]
    pub fn live_count(&self) -> usize {
        let caches = self.caches_.lock().unwrap();
        let mut count = 0;
        for (_path, weak) in caches.iter() {
            if weak.strong_count() != 0 {
                count += 1;
            }
        }
        count
    }
}

// Literal port of
// backends/xnnpack/test/runtime/test_xnn_weights_cache_manager.cpp.
//
// PORT-NOTE: LINK GAP. `XNNWeightsCache` (constructed by `get_or_create`) is
// itself gated on the XNNPACK C API — its `xnn_weights_cache_provider` struct
// and the C trampolines it wires up reference `extern "C"` XNNPACK symbols that
// nothing links yet. These tests compile under `--features xnnpack` but cannot
// link/run until XNNPACK is wired into the build. Left as plain `#[test]`s per
// the Wave-3 xnnpack-gated convention; the default `cargo test` build does not
// enable the feature.
//
// The threading tests (`ConcurrentSamePathSameInstance`,
// `ConcurrentDifferentPathsIndependent`) are ported faithfully with
// `std::thread`.
#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    // Mirrors `XNNWeightsCacheManagerTest::SetUp()`: PAL init + fresh manager.
    fn setup() -> XNNWeightsCacheManager {
        crate::runtime::platform::runtime::runtime_init();
        XNNWeightsCacheManager::new()
    }

    // Raw address of the underlying XNNWeightsCache, mirroring C++
    // `shared_ptr::get()`.
    fn addr(cache: &Arc<XNNWeightsCache>) -> *const XNNWeightsCache {
        Arc::as_ptr(cache)
    }

    // --- Core dedup semantics ---

    // setup() constructs a fresh manager via the ctor.
    // [spec:et:sem:xnn-weights-cache-manager.executorch.backends.xnnpack.xnn-weights-cache-manager.xnn-weights-cache-manager-fn/test]
    // [spec:et:sem:xnn-weights-cache-manager.executorch.backends.xnnpack.xnn-weights-cache-manager.get-or-create-fn/test]
    #[test]
    fn same_path_returns_same_instance() {
        let manager = setup();
        let a = manager.get_or_create("/tmp/test_cache_same.bin").unwrap();
        let b = manager.get_or_create("/tmp/test_cache_same.bin").unwrap();
        assert_eq!(
            addr(&a),
            addr(&b),
            "same path must return the same shared instance"
        );
    }

    // [spec:et:sem:xnn-weights-cache-manager.executorch.backends.xnnpack.xnn-weights-cache-manager.get-or-create-fn/test]
    #[test]
    fn different_paths_return_different_instances() {
        let manager = setup();
        let a = manager.get_or_create("/tmp/test_cache_a.bin").unwrap();
        let b = manager.get_or_create("/tmp/test_cache_b.bin").unwrap();
        assert_ne!(
            addr(&a),
            addr(&b),
            "different paths must return independent instances"
        );
    }

    // [spec:et:sem:xnn-weights-cache-manager.executorch.backends.xnnpack.xnn-weights-cache-manager.get-or-create-fn/test]
    // [spec:et:sem:xnn-weights-cache-manager.executorch.backends.xnnpack.xnn-weights-cache-manager.live-count-fn/test]
    #[test]
    fn empty_path_shared_across_callers() {
        let manager = setup();
        let a = manager.get_or_create("").unwrap();
        let b = manager.get_or_create("").unwrap();
        // Empty-path sharing keeps XNNPACK's name-based dedup working across PTEs.
        assert_eq!(addr(&a), addr(&b));
        assert_eq!(
            manager.live_count(),
            0,
            "empty-path sharing is kept off the path-keyed map"
        );
    }

    // [spec:et:sem:xnn-weights-cache-manager.executorch.backends.xnnpack.xnn-weights-cache-manager.get-or-create-fn/test]
    #[test]
    fn empty_path_recreated_after_all_refs_drop() {
        let manager = setup();
        let first_addr: *const XNNWeightsCache;
        {
            let a = manager.get_or_create("").unwrap();
            first_addr = addr(&a);
        }
        // All Arcs dropped → Weak expires → next call gets a fresh instance.
        let b = manager.get_or_create("").unwrap();
        assert_ne!(addr(&b), first_addr);
    }

    // [spec:et:sem:xnn-weights-cache-manager.executorch.backends.xnnpack.xnn-weights-cache-manager.get-or-create-fn/test]
    // [spec:et:sem:xnn-weights-cache-manager.executorch.backends.xnnpack.xnn-weights-cache-manager.live-count-fn/test]
    #[test]
    fn empty_path_does_not_share_with_mmap_path() {
        let manager = setup();
        let empty = manager.get_or_create("").unwrap();
        let mmap = manager
            .get_or_create("/tmp/test_cache_isolation.bin")
            .unwrap();
        // Empty-path cache stays separate from any mmap-path cache.
        assert_ne!(addr(&empty), addr(&mmap));
        assert_eq!(
            manager.live_count(),
            1,
            "only the mmap-path call registers in the path-keyed map"
        );
    }

    // --- weak_ptr cleanup ---

    // [spec:et:sem:xnn-weights-cache-manager.executorch.backends.xnnpack.xnn-weights-cache-manager.live-count-fn/test]
    #[test]
    fn expired_entry_does_not_leak() {
        let manager = setup();
        {
            let _a = manager.get_or_create("/tmp/test_cache_expire.bin").unwrap();
            assert_eq!(manager.live_count(), 1);
        }
        // Arc dropped → Weak in map is now expired.
        assert_eq!(manager.live_count(), 0);
    }

    // [spec:et:sem:xnn-weights-cache-manager.executorch.backends.xnnpack.xnn-weights-cache-manager.get-or-create-fn/test]
    // [spec:et:sem:xnn-weights-cache-manager.executorch.backends.xnnpack.xnn-weights-cache-manager.live-count-fn/test]
    #[test]
    fn expired_entry_recreated_on_next_call() {
        let manager = setup();
        let first_addr: *const XNNWeightsCache;
        {
            let a = manager
                .get_or_create("/tmp/test_cache_recreate.bin")
                .unwrap();
            first_addr = addr(&a);
        }
        // Address re-use allowed but not required; only guarantee is a usable
        // instance.
        let b = manager
            .get_or_create("/tmp/test_cache_recreate.bin")
            .unwrap();
        assert_eq!(manager.live_count(), 1);
        let _ = first_addr;
    }

    // --- Concurrent same-path returns the same instance ---

    // [spec:et:sem:xnn-weights-cache-manager.executorch.backends.xnnpack.xnn-weights-cache-manager.get-or-create-fn/test]
    // [spec:et:sem:xnn-weights-cache-manager.executorch.backends.xnnpack.xnn-weights-cache-manager.live-count-fn/test]
    #[test]
    fn concurrent_same_path_same_instance() {
        const K_THREADS: usize = 16;
        let manager = Arc::new(setup());
        let ready = Arc::new(AtomicUsize::new(0));
        let mut threads = Vec::with_capacity(K_THREADS);
        for _ in 0..K_THREADS {
            let manager = Arc::clone(&manager);
            let ready = Arc::clone(&ready);
            threads.push(std::thread::spawn(move || {
                // Spin to maximize the chance of true concurrent entry.
                ready.fetch_add(1, Ordering::AcqRel);
                while ready.load(Ordering::Acquire) < K_THREADS {
                    std::thread::yield_now();
                }
                let r = manager.get_or_create("/tmp/test_cache_race.bin").unwrap();
                r
            }));
        }
        let results: Vec<Arc<XNNWeightsCache>> =
            threads.into_iter().map(|t| t.join().unwrap()).collect();
        // All threads must hold the exact same instance pointer.
        for i in 1..K_THREADS {
            assert_eq!(
                addr(&results[0]),
                addr(&results[i]),
                "thread {} got a different instance",
                i
            );
        }
        assert_eq!(manager.live_count(), 1);
    }

    // [spec:et:sem:xnn-weights-cache-manager.executorch.backends.xnnpack.xnn-weights-cache-manager.get-or-create-fn/test]
    // [spec:et:sem:xnn-weights-cache-manager.executorch.backends.xnnpack.xnn-weights-cache-manager.live-count-fn/test]
    #[test]
    fn concurrent_different_paths_independent() {
        const K_THREADS: usize = 8;
        let manager = Arc::new(setup());
        let mut threads = Vec::with_capacity(K_THREADS);
        for i in 0..K_THREADS {
            let manager = Arc::clone(&manager);
            threads.push(std::thread::spawn(move || {
                let path = format!("/tmp/test_cache_diff_{}.bin", i);
                manager.get_or_create(&path).unwrap()
            }));
        }
        let results: Vec<Arc<XNNWeightsCache>> =
            threads.into_iter().map(|t| t.join().unwrap()).collect();
        for i in 0..K_THREADS {
            for j in (i + 1)..K_THREADS {
                assert_ne!(addr(&results[i]), addr(&results[j]));
            }
        }
        assert_eq!(manager.live_count(), K_THREADS);
    }

    // --- save_all walks live caches ---

    // [spec:et:sem:xnn-weights-cache-manager.executorch.backends.xnnpack.xnn-weights-cache-manager.save-all-fn/test]
    #[test]
    fn save_all_no_live_instances_is_ok() {
        let manager = setup();
        assert_eq!(manager.save_all(), Error::Ok);
    }

    // [spec:et:sem:xnn-weights-cache-manager.executorch.backends.xnnpack.xnn-weights-cache-manager.save-all-fn/test]
    // [spec:et:sem:xnn-weights-cache-manager.executorch.backends.xnnpack.xnn-weights-cache-manager.live-count-fn/test]
    #[test]
    fn save_all_walks_live_caches() {
        let manager = setup();
        let _a = manager.get_or_create("/tmp/test_cache_save_a.bin").unwrap();
        let _b = manager.get_or_create("/tmp/test_cache_save_b.bin").unwrap();
        assert_eq!(manager.live_count(), 2);
        // Neither has been through initialize_for_runtime, so save_packed_index
        // short-circuits on fd<0 and returns Ok.
        assert_eq!(manager.save_all(), Error::Ok);
    }

    // [spec:et:sem:xnn-weights-cache-manager.executorch.backends.xnnpack.xnn-weights-cache-manager.save-all-fn/test]
    // [spec:et:sem:xnn-weights-cache-manager.executorch.backends.xnnpack.xnn-weights-cache-manager.live-count-fn/test]
    #[test]
    fn save_all_skips_expired_entries() {
        let manager = setup();
        {
            let _a = manager
                .get_or_create("/tmp/test_cache_save_expired.bin")
                .unwrap();
        }
        // The entry's Weak is now expired. save_all must not crash; erases it.
        assert_eq!(manager.save_all(), Error::Ok);
        assert_eq!(manager.live_count(), 0);
    }

    // --- Path is set on the instance before publishing ---

    // [spec:et:sem:xnn-weights-cache-manager.executorch.backends.xnnpack.xnn-weights-cache-manager.get-or-create-fn/test]
    // [spec:et:sem:xnn-weights-cache-manager.executorch.backends.xnnpack.xnn-weights-cache-manager.live-count-fn/test]
    #[test]
    fn non_empty_path_registers_in_map() {
        let manager = setup();
        let _a = manager
            .get_or_create("/tmp/test_cache_register.bin")
            .unwrap();
        assert_eq!(manager.live_count(), 1);
    }

    // The deleted copy-assign (`operator=(const XNNWeightsCacheManager&) =
    // delete`) collapses onto the non-`Clone` struct in Rust: the manager (and
    // its per-path dedup map) is never duplicated, only shared. The probe
    // resolves the inherent `is_clone` (true) only when the type implements
    // `Clone`, so this fails to hold if `Clone` is ever derived. The aliasing
    // check then shows a shared handle observes registrations made through
    // another — one map, not a copied one.
    // [spec:et:sem:xnn-weights-cache-manager.executorch.backends.xnnpack.xnn-weights-cache-manager.operator-fn/test]
    #[test]
    fn manager_is_not_copyable() {
        struct Probe<T>(core::marker::PhantomData<T>);
        trait NotClone {
            fn is_clone(&self) -> bool {
                false
            }
        }
        impl<T> NotClone for Probe<T> {}
        impl<T: Clone> Probe<T> {
            #[allow(dead_code)]
            fn is_clone(&self) -> bool {
                true
            }
        }
        assert!(!Probe::<XNNWeightsCacheManager>(core::marker::PhantomData).is_clone());

        let manager = Arc::new(setup());
        let alias = Arc::clone(&manager);
        let _cache = manager.get_or_create("/tmp/test_cache_alias.bin").unwrap();
        assert_eq!(alias.live_count(), 1);
    }
}
