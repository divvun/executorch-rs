# backends/xnnpack/runtime/XNNWeightsCacheManager.cpp, backends/xnnpack/runtime/XNNWeightsCacheManager.h

> [spec:et:def:xnn-weights-cache-manager.executorch.backends.xnnpack.xnn-weights-cache-manager]
> class XNNWeightsCacheManager {
>   mutable std::mutex meta_mutex_;
>   std::unordered_map<std::string, std::weak_ptr<delegate::XNNWeightsCache>> caches_;
>   mutable std::mutex empty_path_mutex_;
>   std::weak_ptr<delegate::XNNWeightsCache> empty_path_cache_;
> }

> [spec:et:def:xnn-weights-cache-manager.executorch.backends.xnnpack.xnn-weights-cache-manager.get-or-create-fn]
> Result<std::shared_ptr<delegate::XNNWeightsCache>>

> [spec:et:sem:xnn-weights-cache-manager.executorch.backends.xnnpack.xnn-weights-cache-manager.get-or-create-fn]
> Returns a shared `XNNWeightsCache` instance keyed by `cache_file_path`, creating one on first use. Never returns null on success. Two distinct keying regimes:
>
> Empty-path case (`cache_file_path.empty()` is true): acquire `empty_path_mutex_` (scoped lock held for the whole branch). Attempt to promote the stored `empty_path_cache_` weak_ptr via `.lock()`. If it yields a live shared_ptr, return it — this is the single shared heap-only instance reused by all empty-path callers (no file path is set on it, so its `reserve_space` always uses the heap path per `[spec:et:sem:xnn-weights-cache.executorch.backends.xnnpack.delegate.xnn-weights-cache.reserve-space-fn]`). Otherwise construct a new `XNNWeightsCache` via make_shared (do NOT set any packed cache path on it), store a weak reference into `empty_path_cache_`, and return the shared_ptr.
>
> Non-empty-path case: acquire `meta_mutex_` (scoped lock for the rest of the function). Look up `cache_file_path` in the `caches_` map. If found, attempt `.lock()` on the stored weak_ptr; if it yields a live instance, return it. If the entry exists but its weak_ptr is expired (lock returned null), erase that stale map entry and fall through to creation. Construct a new `XNNWeightsCache` via make_shared, call `set_packed_cache_path(cache_file_path)` on it BEFORE publishing it into the map (so concurrent callers can never observe a half-initialized instance with no path — path assignment must precede map insertion) per `[spec:et:sem:xnn-weights-cache.executorch.backends.xnnpack.delegate.xnn-weights-cache.set-packed-cache-path-fn]`, insert a weak reference into `caches_[cache_file_path]`, and return the shared_ptr.
>
> The manager holds only weak_ptrs; instances stay alive exactly as long as some external owner holds a strong ref. Lock order is `meta_mutex_` (or `empty_path_mutex_`) first, then per-instance `XNNWeightsCache::mutex()`; never take these in reverse. The empty-path slot uses a separate mutex to avoid string-hashing/contention with mmap-path callers.

> [spec:et:def:xnn-weights-cache-manager.executorch.backends.xnnpack.xnn-weights-cache-manager.live-count-fn]
> size_t XNNWeightsCacheManager::live_count() const

> [spec:et:sem:xnn-weights-cache-manager.executorch.backends.xnnpack.xnn-weights-cache-manager.live-count-fn]
> Test-only accessor returning the number of non-expired cache entries in `caches_`. Acquire `meta_mutex_` (scoped lock; the method is const and `meta_mutex_` is declared mutable). Initialize a counter to 0. Iterate every entry in the `caches_` map and increment the counter for each whose weak_ptr `.expired()` returns false. Return the count. This counts only the non-empty-path map; the separate `empty_path_cache_` slot is not included. This method does not erase expired entries (unlike `save_all`).

> [spec:et:def:xnn-weights-cache-manager.executorch.backends.xnnpack.xnn-weights-cache-manager.save-all-fn]
> Error XNNWeightsCacheManager::save_all()

> [spec:et:sem:xnn-weights-cache-manager.executorch.backends.xnnpack.xnn-weights-cache-manager.save-all-fn]
> Walks all live caches and calls `save_packed_index()` on each. Returns `Error::Ok` if all succeed, otherwise the first non-Ok error encountered (keeps going past failures so one failing cache does not strand the others).
>
> Step 1 (snapshot under `meta_mutex_`): acquire `meta_mutex_` (scoped lock for this block only). Build a local vector of strong shared_ptrs. Iterate `caches_`: for each entry, attempt `.lock()` on the weak_ptr; if it yields a live shared_ptr, move it into the vector and advance the iterator; if it is expired (lock null), erase that map entry (opportunistic cleanup of dead weak_ptrs) using the erase-returns-next-iterator idiom. Release `meta_mutex_` at end of block — this honors the lock order (`meta_mutex_` before per-instance mutex) and lets `get_or_create` on unrelated paths proceed during the per-instance save walk.
>
> Step 2 (per-instance save): initialize `first_err = Error::Ok`. For each shared_ptr in the snapshot vector, acquire that instance's `XNNWeightsCache::mutex()` (a lock_guard held for the single call), call `save_packed_index()` per `[spec:et:sem:xnn-weights-cache.executorch.backends.xnnpack.delegate.xnn-weights-cache.save-packed-index-fn]`, and if it returns non-Ok and `first_err` is still `Error::Ok`, record it into `first_err`. After the loop, return `first_err`. The empty-path heap-only cache in `empty_path_cache_` is intentionally not saved (it has no file path, so its `save_packed_index` would be a no-op anyway).

> [spec:et:def:xnn-weights-cache-manager.executorch.backends.xnnpack.xnn-weights-cache-manager.xnn-weights-cache-manager-fn]
> XNNWeightsCacheManager() = default

> [spec:et:sem:xnn-weights-cache-manager.executorch.backends.xnnpack.xnn-weights-cache-manager.xnn-weights-cache-manager-fn]
> Compiler-defaulted default constructor (`= default`). Default-initializes all members: `meta_mutex_` and `empty_path_mutex_` to fresh unlocked mutexes, `caches_` to an empty map, and `empty_path_cache_` to an empty (null) weak_ptr. No custom logic. In Rust, this corresponds to a `Default`/`new` that starts with empty maps and no cached instances.

> [spec:et:def:xnn-weights-cache-manager.executorch.backends.xnnpack.xnn-weights-cache-manager.operator-fn]
> XNNWeightsCacheManager& operator=(const XNNWeightsCacheManager&) = delete

> [spec:et:sem:xnn-weights-cache-manager.executorch.backends.xnnpack.xnn-weights-cache-manager.operator-fn]
> Deleted copy-assignment operator (`= delete`). The manager is non-copyable and non-movable (copy/move constructors and move-assignment are also deleted) because it owns per-path mutexes and a registry of weak_ptrs that must not be duplicated or relocated. Any attempt to copy-assign is a compile-time error. In Rust this is the natural default: do not derive `Clone` and do not implement copy/move for this type.

