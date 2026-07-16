# backends/xnnpack/runtime/XNNWeightsCache.cpp, backends/xnnpack/runtime/XNNWeightsCache.h

> [spec:et:def:xnn-weights-cache.executorch.backends.xnnpack.delegate.append-le-fn]
> static void append_le(std::vector<uint8_t>& buf, T value)

> [spec:et:sem:xnn-weights-cache.executorch.backends.xnnpack.delegate.append-le-fn]
> Function template. Appends the raw in-memory bytes of `value` (a `T` such as u32 or u64) to the end of the byte buffer `buf`. Takes the address of `value`, reinterprets it as a `const uint8_t*`, and inserts `sizeof(T)` bytes starting at that address onto `buf.end()`. Because the host platforms in scope (the file-backed cache path is compiled only on non-Windows targets) are little-endian, this writes the value in little-endian byte order; a portable Rust re-implementation must explicitly serialize `value` as little-endian bytes (e.g. `value.to_le_bytes()`) and extend the buffer with them, regardless of host endianness. Used to build the on-disk trailer/index; must round-trip exactly with `[spec:et:sem:xnn-weights-cache.executorch.backends.xnnpack.delegate.read-le-fn]`.

> [spec:et:def:xnn-weights-cache.executorch.backends.xnnpack.delegate.open-locked-fn]
> static int open_locked(const std::string& path, int flags)

> [spec:et:sem:xnn-weights-cache.executorch.backends.xnnpack.delegate.open-locked-fn]
> POSIX-only helper (compiled only on non-Windows). Opens the file at `path` with the given open `flags` and mode `0600` (rw for owner only). If `open` fails (returns fd < 0), logs an Error `"open(<path>) failed (errno=<errno>)"` and returns -1. Otherwise attempts a non-blocking advisory exclusive lock via `flock(fd, LOCK_EX | LOCK_NB)`. If the flock fails (nonzero return — e.g. another process already holds a conflicting lock), logs an Error `"flock(<path>) failed (errno=<errno>)"`, closes the fd, and returns -1. On success returns the open, exclusively-locked fd. The lock is advisory and per-open-file-description; it is released when the fd is closed. Callers treat a -1 return as "skip the mmap/file-backed path for this init and fall back to heap." A Rust port should acquire an exclusive, non-blocking advisory file lock (e.g. flock on unix) and mirror this fd-or-error contract.

> [spec:et:def:xnn-weights-cache.executorch.backends.xnnpack.delegate.packed-data-meta]
> struct PackedDataMeta {
>   size_t offset{};
>   size_t data_size{0};
>   size_t ref_count{};
>   bool in_current_runtime{};
>   bool from_load{false};
>   uint32_t seed{0};
> }

> [spec:et:def:xnn-weights-cache.executorch.backends.xnnpack.delegate.read-le-fn]
> static T read_le(const uint8_t* src)

> [spec:et:sem:xnn-weights-cache.executorch.backends.xnnpack.delegate.read-le-fn]
> Function template. Reads a value of type `T` (e.g. u32 or u64) from the raw byte pointer `src`. Declares an uninitialized `T`, `memcpy`s `sizeof(T)` bytes from `src` into it, and returns it. This is the inverse of `[spec:et:sem:xnn-weights-cache.executorch.backends.xnnpack.delegate.append-le-fn]`: the bytes at `src` are interpreted in host byte order, which for the in-scope little-endian non-Windows targets means little-endian. A portable Rust re-implementation must read the `sizeof(T)` bytes at `src` and decode them as little-endian (e.g. `T::from_le_bytes(...)`). The caller is responsible for ensuring `src` points at `sizeof(T)` valid, in-bounds bytes (bounds are checked by the caller in `load_packed_cache` before each read).

> [spec:et:def:xnn-weights-cache.executorch.backends.xnnpack.delegate.xnn-weights-cache]
> class XNNWeightsCache {
>   static const size_t kPackedAllocationAlignment = 64;
>   static constexpr uint32_t kCacheMagic = 0x58505743;
>   static constexpr uint32_t kCacheVersion = 2;
>   MemoryAllocator* runtime_allocator_;
>   const NamedDataMap* named_data_map_;
>   std::unordered_map<const void*, std::string> unpacked_data_to_name_;
>   std::unordered_map<std::string, PackedDataMeta> name_to_packed_data_metadata_;
>   std::vector<void*> packed_data_ptrs_;
>   std::unordered_map<void*, std::string> packed_pointer_to_container_;
>   std::vector<FreeableBuffer> unpacked_data_;
>   xnn_weights_cache_provider weights_cache_;
>   bool is_finalized_;
>   std::string packed_cache_path_;
>   int packed_file_fd_{-1};
>   size_t packed_file_used_{0};
>   std::unordered_map<void*, size_t> ptr_to_file_offset_;
>   struct MmapRegion { void* addr; size_t size; };
>   std::vector<MmapRegion> mmap_regions_;
>   size_t mmap_regions_synced_{0};
>   size_t mmap_regions_at_last_save_{0};
>   std::unordered_map<void*, size_t> file_ptr_to_region_index_;
>   std::mutex instance_mutex_;
> }

> [spec:et:def:xnn-weights-cache.executorch.backends.xnnpack.delegate.xnn-weights-cache.delete-cache-fn]
> enum xnn_status XNNWeightsCache::delete_cache(XNNWeightsCache* context)

> [spec:et:sem:xnn-weights-cache.executorch.backends.xnnpack.delegate.xnn-weights-cache.delete-cache-fn]
> Static callback wired into the `xnn_weights_cache_provider.delete_cache` slot. Unconditionally returns `xnn_status_success` and does nothing else — this cache's lifetime is managed by the owning `XNNWeightsCache`/`XNNWeightsCacheManager` (weak_ptr registry), not by XNNPACK, so XNNPACK's request to delete the cache is a no-op. `context` is ignored. A Rust port implements this as a callback that returns success without freeing anything.

> [spec:et:def:xnn-weights-cache.executorch.backends.xnnpack.delegate.xnn-weights-cache.delete-packed-data-fn]
> Error XNNWeightsCache::delete_packed_data( const std::vector<std::string>& packed_data_names)

> [spec:et:sem:xnn-weights-cache.executorch.backends.xnnpack.delegate.xnn-weights-cache.delete-packed-data-fn]
> Decrements the reference count of each named packed-data entry and frees entries whose count reaches zero. Called when a runtime that used these packed weights is torn down. Precondition: the cache must be finalized.
>
> Guard: if `is_finalized_` is false, log Error `"delete_packed_data called before finalize_for_runtime"` and return `Error::InvalidArgument` (no state mutated).
>
> For each `name` in `packed_data_names`, in order: look it up in `name_to_packed_data_metadata_`. If absent, log Error `"delete_packed_data: '<name>' not found"` and return `Error::InvalidArgument` immediately (entries processed before this one keep their already-applied decrements — no rollback). Otherwise pre-decrement `ref_count`; if the resulting value is still > 0, skip to the next name (entry stays live for other runtimes). If it reached 0:
>   - If the entry's `from_load` flag is true (its bytes are persisted in the on-disk cache file), do NOT free it: set `in_current_runtime = false` and continue to the next name. Keeping it avoids forcing the next init to re-pack and re-append the packed bytes (hundreds of MB) to the file each cycle.
>   - Otherwise (heap- or fresh-mmap-backed, not persisted): call `release_entry` on the pointer at `packed_data_ptrs_[entry.offset]` per `[spec:et:sem:xnn-weights-cache.executorch.backends.xnnpack.delegate.xnn-weights-cache.release-entry-fn]`, then null out `packed_data_ptrs_[entry.offset]` (so existing offsets held by other entries stay valid — slots are never removed, only nulled), then erase the entry from `name_to_packed_data_metadata_`.
>
> After processing all names: if `name_to_packed_data_metadata_` is now empty (last entry gone), call `full_unload` per `[spec:et:sem:xnn-weights-cache.executorch.backends.xnnpack.delegate.xnn-weights-cache.full-unload-fn]` to drop all in-memory state and close the fd; the on-disk file is deliberately left intact so a later process can `load_packed_cache` and skip re-packing (if the trailer was corrupted by a post-save reserve_space, that load falls through to fresh-write — same net effect as truncating). Return `Error::Ok`.
>
> Note the pre-decrement semantics: an entry with `ref_count == 0` at entry (should not normally happen post-finalize) would wrap around to a huge value under `--` on an unsigned `size_t`; the code assumes counts are positive when this is called.

> [spec:et:def:xnn-weights-cache.executorch.backends.xnnpack.delegate.xnn-weights-cache.finalize-for-runtime-fn]
> Result<std::vector<std::string>> XNNWeightsCache::finalize_for_runtime()

> [spec:et:sem:xnn-weights-cache.executorch.backends.xnnpack.delegate.xnn-weights-cache.finalize-for-runtime-fn]
> Called once after `xnn_create_runtime` has packed all weights for the runtime being built. Releases unpacked source data, bumps ref counts on the packed entries this runtime uses, flushes new file-backed regions to disk, and optionally saves the on-disk index. Returns `Result<std::vector<std::string>>` = the list of packed-data names used by this runtime (never an error in this implementation).
>
> Steps:
> 1. Set `is_finalized_ = true`.
> 2. Free all unpacked source buffers: for each `FreeableBuffer` in `unpacked_data_`, call its `Free()`; then clear `unpacked_data_` and `unpacked_data_to_name_`. The packed weights are all XNNPACK needs from here on, so the unpacked constant data is no longer retained.
> 3. Build `packed_data_names` (empty vector). Iterate `name_to_packed_data_metadata_` in unspecified (hash) order; for every entry whose `in_current_runtime` is true: increment its `ref_count`, set `in_current_runtime = false` (reset for the next runtime build), and append its name to `packed_data_names`.
> 4. (Non-Windows only) Synchronous flush of newly added mmap regions: if `mmap_regions_.size() > mmap_regions_synced_`, then for each region index `i` from `mmap_regions_synced_` up to (not including) `mmap_regions_.size()`, if that region's `addr` is non-null call `msync(addr, size, MS_SYNC)` (blocks until dirty pages are written to disk and marked clean). Set `mmap_regions_synced_ = mmap_regions_.size()`. Log Info with the count of newly synced regions, total region count, and `packed_file_used_ / (1024*1024)` MB.
> 5. Auto-save: if `packed_cache_path_` is non-empty AND `mmap_regions_.size() > mmap_regions_at_last_save_` (this compile session appended new file-backed regions), call `save_packed_index()` per `[spec:et:sem:xnn-weights-cache.executorch.backends.xnnpack.delegate.xnn-weights-cache.save-packed-index-fn]` (return value ignored).
> 6. Return `packed_data_names`.

> [spec:et:def:xnn-weights-cache.executorch.backends.xnnpack.delegate.xnn-weights-cache.full-unload-fn]
> void XNNWeightsCache::full_unload()

> [spec:et:sem:xnn-weights-cache.executorch.backends.xnnpack.delegate.xnn-weights-cache.full-unload-fn]
> Tears down all in-memory file-backed state and closes the write fd; the on-disk cache file itself is left untouched. Called by `delete_packed_data` when the last packed entry is removed. Entire body is non-Windows-only (no-op on Windows). Steps:
> 1. For each region in `mmap_regions_`: if its `addr` is non-null and not `MAP_FAILED`, `munmap(addr, size)`, then set `addr = nullptr` and `size = 0`.
> 2. Clear `mmap_regions_`; set `mmap_regions_synced_ = 0`.
> 3. Clear `packed_data_ptrs_`, `ptr_to_file_offset_`, and `file_ptr_to_region_index_`.
> 4. If `packed_file_fd_ >= 0`, `close` it and set `packed_file_fd_ = -1` (releases the advisory lock).
>
> Note this does NOT reset `packed_file_used_`, `mmap_regions_at_last_save_`, `packed_cache_path_`, `name_to_packed_data_metadata_`, or `packed_pointer_to_container_` — the caller (`delete_packed_data`) has already emptied the metadata map, and the path is retained so a fresh init can reopen/reload. Heap-backed containers in `packed_pointer_to_container_` are freed separately by their entries' `release_entry` calls before this runs.

> [spec:et:def:xnn-weights-cache.executorch.backends.xnnpack.delegate.xnn-weights-cache.get-fn]
> inline xnn_weights_cache_t get()

> [spec:et:sem:xnn-weights-cache.executorch.backends.xnnpack.delegate.xnn-weights-cache.get-fn]
> Inline accessor. Returns the address of the internal `weights_cache_` member (`xnn_weights_cache_provider`) cast to `xnn_weights_cache_t`, i.e. the provider struct XNNPACK uses to invoke this cache's callbacks (`look_up`, `reserve_space`, `look_up_or_insert`, `is_finalized`, `offset_to_addr`, `delete_cache`) whose `context` field points back at this instance (set up in the constructor per `[spec:et:sem:xnn-weights-cache.executorch.backends.xnnpack.delegate.xnn-weights-cache.xnn-weights-cache-fn]`). The returned pointer aliases into this object and is only valid while the `XNNWeightsCache` is alive.

> [spec:et:def:xnn-weights-cache.executorch.backends.xnnpack.delegate.xnn-weights-cache.get-num-unpacked-data-fn]
> inline size_t get_num_unpacked_data()

> [spec:et:sem:xnn-weights-cache.executorch.backends.xnnpack.delegate.xnn-weights-cache.get-num-unpacked-data-fn]
> Inline accessor. Returns `unpacked_data_.size()` — the current number of unpacked `FreeableBuffer`s held (i.e. constant tensors loaded via `load_unpacked_data` and not yet freed). After `finalize_for_runtime` this is 0 because finalization frees and clears `unpacked_data_`.

> [spec:et:def:xnn-weights-cache.executorch.backends.xnnpack.delegate.xnn-weights-cache.get-packed-data-names-fn]
> inline std::vector<std::string> get_packed_data_names()

> [spec:et:sem:xnn-weights-cache.executorch.backends.xnnpack.delegate.xnn-weights-cache.get-packed-data-names-fn]
> Inline accessor. Returns a `std::vector<std::string>` of all packed-data entry names. Reserves capacity `name_to_packed_data_metadata_.size()`, then iterates that map and pushes each entry's key (`pair.first`, the packed name) into the vector. Order is the map's unspecified (hash) iteration order — callers must not depend on ordering. Returns names for all metadata entries regardless of `ref_count` or `in_current_runtime`.

> [spec:et:def:xnn-weights-cache.executorch.backends.xnnpack.delegate.xnn-weights-cache.get-unpacked-data-names-fn]
> inline std::vector<std::string> get_unpacked_data_names()

> [spec:et:sem:xnn-weights-cache.executorch.backends.xnnpack.delegate.xnn-weights-cache.get-unpacked-data-names-fn]
> Inline accessor. Returns a `std::vector<std::string>` of the names of all currently-held unpacked data. Reserves capacity `unpacked_data_to_name_.size()`, then iterates that map (keyed by unpacked data pointer) and pushes each entry's value (`pair.second`, the name) into the vector. Order is the map's unspecified (hash) iteration order. After `finalize_for_runtime` this is empty because finalization clears `unpacked_data_to_name_`.

> [spec:et:def:xnn-weights-cache.executorch.backends.xnnpack.delegate.xnn-weights-cache.initialize-for-runtime-fn]
> Error XNNWeightsCache::initialize_for_runtime( MemoryAllocator* runtime_allocator, const NamedDataMap* named_data_map)

> [spec:et:sem:xnn-weights-cache.executorch.backends.xnnpack.delegate.xnn-weights-cache.initialize-for-runtime-fn]
> Prepares the cache for the next `xnn_create_runtime`. Stores the runtime allocator and named-data-map handles and, if a file-backed cache path is configured, ensures the packed cache file is loaded (or freshly opened) and the write fd is live. Always returns `Error::Ok` in the observable success paths (file errors degrade gracefully to a heap fallback rather than failing).
>
> Steps:
> 1. `runtime_allocator_ = runtime_allocator`; `named_data_map_ = named_data_map`; `is_finalized_ = false`. (These three assignments are the entire body on Windows; everything below is non-Windows-only.)
> 2. If `packed_cache_path_` is empty (heap-only mode) OR the write fd is already open (`packed_file_fd_ >= 0`), return `Error::Ok` — nothing more to do.
> 3. Case "entries already in memory": if `name_to_packed_data_metadata_` is non-empty (from a prior `load_packed_cache` or a prior fresh-write session in this process), just reopen the write fd via `open_locked(packed_cache_path_, O_RDWR)` per `[spec:et:sem:xnn-weights-cache.executorch.backends.xnnpack.delegate.open-locked-fn]` (assign its result to `packed_file_fd_`, which may be -1 on failure → later reserve_space falls back to heap) and return `Error::Ok`. Gating on metadata emptiness (rather than a separate flag) deliberately avoids re-entering `load_packed_cache` and double-mmapping the file after a fresh-write→save→re-init cycle.
> 4. Case "no in-memory entries": call `load_packed_cache()` per `[spec:et:sem:xnn-weights-cache.executorch.backends.xnnpack.delegate.xnn-weights-cache.load-packed-cache-fn]`. If it returns true (a valid saved trailer was loaded), log Info `"Loaded packed weight cache: <path> (<n> entries)"`, open the write fd via `open_locked(path, O_RDWR)` (assign to `packed_file_fd_`), and return `Error::Ok`.
> 5. Case "fresh write" (load failed / no valid file): open via `open_locked(path, O_RDWR | O_CREAT)` — note O_TRUNC is deliberately NOT passed so a concurrent holder's existing mmap stays valid. If the open returns < 0, set `packed_file_fd_` to that (-1) and return `Error::Ok` (heap fallback). Otherwise, now that the exclusive lock is held, explicitly `ftruncate(packed_file_fd_, 0)` to empty the file. If ftruncate fails, log Error `"ftruncate(0) failed for <path> (errno=...); heap fallback this init"`, close the fd, set `packed_file_fd_ = -1`, and return `Error::Ok`. On success call `reset_for_fresh_write()` per `[spec:et:sem:xnn-weights-cache.executorch.backends.xnnpack.delegate.xnn-weights-cache.reset-for-fresh-write-fn]` and log Info `"Opened packed weight file for writing: <path>"`.
> 6. Return `Error::Ok`.

> [spec:et:def:xnn-weights-cache.executorch.backends.xnnpack.delegate.xnn-weights-cache.is-finalized-fn]
> bool XNNWeightsCache::is_finalized(XNNWeightsCache* context)

> [spec:et:sem:xnn-weights-cache.executorch.backends.xnnpack.delegate.xnn-weights-cache.is-finalized-fn]
> Static callback wired into the `xnn_weights_cache_provider.is_finalized` slot. Returns `context->is_finalized_` — true once `finalize_for_runtime` has run for this cache, false before. XNNPACK queries this to decide whether the cache still accepts insertions (a non-finalized cache lets `look_up_or_insert` add new packed entries).

> [spec:et:def:xnn-weights-cache.executorch.backends.xnnpack.delegate.xnn-weights-cache.load-packed-cache-fn]
> bool XNNWeightsCache::load_packed_cache()

> [spec:et:sem:xnn-weights-cache.executorch.backends.xnnpack.delegate.xnn-weights-cache.load-packed-cache-fn]
> Attempts to load a previously-saved packed weight cache file at `packed_cache_path_` by mmapping it read-only and parsing its trailer/index. Returns true if a valid cache was loaded (populating `packed_data_ptrs_`, `ptr_to_file_offset_`, `name_to_packed_data_metadata_`, and one entry in `mmap_regions_`); returns false on any failure/corruption, leaving no partial state. On Windows the whole function is a no-op returning false.
>
> On-disk layout (all integers little-endian per `[spec:et:sem:xnn-weights-cache.executorch.backends.xnnpack.delegate.read-le-fn]`): packed data bytes at the front, then an index region of per-entry records, then a fixed 20-byte footer at the very end. Footer = `[index_start:u64][entry_count:u32][magic:u32][version:u32]`. Each index record = `[name_len:u32][name bytes: name_len][file_offset:u64][data_size:u64][seed:u32]` (so 4 + name_len + 20 bytes).
>
> Steps (non-Windows):
> 1. `open(packed_cache_path_, O_RDONLY)`; if fd < 0 return false.
> 2. Take a non-blocking shared advisory lock: `flock(fd, LOCK_SH | LOCK_NB)`; if it fails (a concurrent writer holds LOCK_EX), close fd and return false.
> 3. Determine size via `lseek(fd, 0, SEEK_END)` (deliberately not fstat, to avoid Apple's Required-Reason-API file-timestamp category). If the result `end_off < 20` (too small to even hold a footer), close fd and return false. Let `file_size = end_off`.
> 4. Read the 20-byte footer with `pread(fd, footer, 20, file_size - 20)`; if it doesn't read exactly 20 bytes, close fd and return false. Decode `index_start` (u64 @0), `entry_count` (u32 @8), `magic` (u32 @12), `version` (u32 @16).
> 5. Validate: if `magic != kCacheMagic` (0x58505743) OR `version != kCacheVersion` (2) OR `index_start >= file_size - 20`, close fd and return false. (Rejecting version!=2 outright means old v1 files, which lack per-entry seeds, are not loaded.) Let `index_region_end = file_size - 20`.
> 6. `mmap(nullptr, file_size, PROT_READ, MAP_SHARED, fd, 0)`; then close fd immediately (the mapping keeps the file alive). If mmap returned MAP_FAILED, return false. Push `{map, file_size}` onto `mmap_regions_`.
> 7. Set `cursor = map + index_start` and `end = map + index_region_end`. Loop `i` from 0 while `i < entry_count` AND `cursor + 4 <= end`:
>    - Read `name_len` (u32) at cursor; advance cursor by 4.
>    - Bounds check the rest of the record: if `cursor + name_len + 20 > end`, the trailer is truncated/inconsistent with `entry_count` → treat as corrupt: log Error, `munmap(map, file_size)`, pop the region just pushed, clear `name_to_packed_data_metadata_`, `packed_data_ptrs_`, and `ptr_to_file_offset_`, and return false (full rollback — never accept a partial cache, or the next save would drop the rest).
>    - Read `name` = `name_len` bytes at cursor as a string; advance cursor by `name_len`. Read `file_offset` (u64), advance 8. Read `data_size` (u64), advance 8. Read `seed` (u32), advance 4.
>    - Bounds check the payload: if `file_offset >= index_start` OR `data_size > index_start - file_offset` (the entry's bytes don't lie entirely within the packed-data region `[0, index_start)`), log Error and perform the same full rollback (munmap, pop region, clear the three containers) and return false.
>    - Otherwise register the entry: `ptr_index = packed_data_ptrs_.size()`; `entry_ptr = map + file_offset`; push `entry_ptr` onto `packed_data_ptrs_`; set `ptr_to_file_offset_[entry_ptr] = file_offset` (so a later save can rewrite a trailer covering loaded + new entries). Build a `PackedDataMeta` with `offset = ptr_index`, `data_size = data_size`, `ref_count = 0`, `in_current_runtime = false`, `from_load = true`, `seed = seed`, and store it as `name_to_packed_data_metadata_[name]`.
> 8. After the loop: set `packed_file_used_ = file_size` (new packs are appended AFTER the existing trailer, never at `index_start`, so the loaded trailer stays intact for cross-process reuse). Set `mmap_regions_at_last_save_ = mmap_regions_.size()` so `save_packed_index` short-circuits until new packs actually arrive. Return true.
>
> Note the loop condition `i < entry_count && cursor + 4 <= end` can exit early if the region is shorter than `entry_count` records without hitting the truncation branch on that final partial header; entries read so far are then accepted. The interior `cursor + name_len + 20 > end` check guards each full record after its 4-byte length prefix.

> [spec:et:def:xnn-weights-cache.executorch.backends.xnnpack.delegate.xnn-weights-cache.load-unpacked-data-fn]
> Result<const uint8_t*> XNNWeightsCache::load_unpacked_data( const std::string& name)

> [spec:et:sem:xnn-weights-cache.executorch.backends.xnnpack.delegate.xnn-weights-cache.load-unpacked-data-fn]
> Loads a named constant (unpacked) tensor from the `NamedDataMap` and registers it so it can later serve as a cache key. Returns `Result<const uint8_t*>` = a pointer to the raw unpacked bytes.
>
> Steps:
> 1. Call `named_data_map_->get_data(name.c_str())`. If the result is not ok, log Error `"Failed to load constant data for key <name>"` and return `Error::InvalidExternalData`.
> 2. Take `data_pointer` = the buffer's `.data()` cast to `const uint8_t*`.
> 3. Move the returned `FreeableBuffer` into `unpacked_data_` (the cache now owns it and will free it at `finalize_for_runtime`).
> 4. Record `unpacked_data_to_name_[data_pointer] = name` so later `look_up`/`look_up_or_insert` can map this pointer (the XNNPACK cache key's kernel/bias pointer) back to its name.
> 5. Return `data_pointer`.
>
> This pointer is what gets handed to XNNPACK's `define_tensor` APIs and reappears as `cache_key->kernel` / `cache_key->bias` during packing.

> [spec:et:def:xnn-weights-cache.executorch.backends.xnnpack.delegate.xnn-weights-cache.look-up-fn]
> size_t XNNWeightsCache::look_up( XNNWeightsCache* context, const xnn_weights_cache_look_up_key* cache_key)

> [spec:et:sem:xnn-weights-cache.executorch.backends.xnnpack.delegate.xnn-weights-cache.look-up-fn]
> Static callback wired into `xnn_weights_cache_provider.look_up`. Given a cache key (unpacked kernel pointer, optional unpacked bias pointer, and a ukernel seed), returns the offset of the matching packed entry, or `SIZE_MAX` on a miss.
>
> Steps:
> 1. Let `unpacked_weights_ptr = cache_key->kernel` and `unpacked_bias_ptr = cache_key->bias`.
> 2. Look up `unpacked_weights_ptr` in `unpacked_data_to_name_`. If absent, return `SIZE_MAX` (the weights were never registered via `load_unpacked_data`).
> 3. Start `weight_bias_name` = the weights' registered name.
> 4. If `unpacked_bias_ptr` is non-null, look it up in `unpacked_data_to_name_`; if found, append the bias's name to `weight_bias_name` (string concatenation, weights-name then bias-name). This composite name is the packed-entry key. (If the bias pointer is non-null but unregistered, it contributes nothing — key is just the weights name.)
> 5. Look up `weight_bias_name` in `name_to_packed_data_metadata_`. If absent, return `SIZE_MAX`.
> 6. Seed check (XNNPACK upgrade detection): if the stored entry's `seed != cache_key->seed`, the ukernel implementation changed; log Info about the seed mismatch and return `SIZE_MAX` so `look_up_or_insert` re-packs with the current ukernel rather than serving stale bytes.
> 7. On a match: set the entry's `in_current_runtime = true` and return its `offset`.

> [spec:et:def:xnn-weights-cache.executorch.backends.xnnpack.delegate.xnn-weights-cache.look-up-or-insert-fn]
> size_t XNNWeightsCache::look_up_or_insert( XNNWeightsCache* context, const xnn_weights_cache_look_up_key* cache_key, void* ptr, size_t size)

> [spec:et:sem:xnn-weights-cache.executorch.backends.xnnpack.delegate.xnn-weights-cache.look-up-or-insert-fn]
> Static callback wired into `xnn_weights_cache_provider.look_up_or_insert`. XNNPACK calls this after packing weights into buffer `ptr` (size `size`) to either confirm a cache hit or register the newly packed bytes. Returns the entry offset.
>
> Steps:
> 1. `offset = look_up(context, cache_key)` per `[spec:et:sem:xnn-weights-cache.executorch.backends.xnnpack.delegate.xnn-weights-cache.look-up-fn]`.
> 2. If `ptr == nullptr` (XNNPACK signals a pure lookup after a hit — nothing was packed, nothing to validate), return `offset` as-is (may be SIZE_MAX or a valid offset).
> 3. If `offset != SIZE_MAX` (name hit): fetch `saved_ptr = offset_to_addr(context, offset)`. If `saved_ptr` is non-null AND `memcmp(ptr, saved_ptr, size) == 0` (the previously-cached packed bytes are byte-identical), return `offset` (true hit). Otherwise the cached bytes differ despite the name matching (stale cache) → return `SIZE_MAX` (forces XNNPACK to treat it as a miss/insert).
> 4. Otherwise (offset == SIZE_MAX, a real miss) — insert. Compute `next_offset = packed_data_ptrs_.size()`. Look up `cache_key->kernel` in `unpacked_data_to_name_`:
>    - If found: build `weight_bias_name` from the weights name, and if `cache_key->bias` is non-null and registered, append the bias name (same composite-key rule as `look_up`). Construct a `PackedDataMeta` with `offset = next_offset`, `data_size = size`, `ref_count = 0` (ref counts are only bumped at `finalize_for_runtime`), `in_current_runtime = true`, `seed = cache_key->seed` (and `from_load` defaults false), and store it as `name_to_packed_data_metadata_[weight_bias_name]`.
>    - If not found (weights not registered with a name): log Info warning that unnamed weight/bias create new cache entries and may affect performance; do NOT add a metadata entry (this packed buffer is tracked only by its slot in `packed_data_ptrs_`, so it can never be re-looked-up by name).
> 5. In both insert sub-cases push `ptr` onto `packed_data_ptrs_` (so `packed_data_ptrs_[next_offset] == ptr`) and return `next_offset`.
>
> Note the memcmp path assumes the caller-provided `size` matches the stored allocation; when it does not exactly correspond to `saved_ptr`'s size the comparison still runs over `size` bytes.

> [spec:et:def:xnn-weights-cache.executorch.backends.xnnpack.delegate.xnn-weights-cache.mmap-region]
> struct MmapRegion {
>   void* addr;
>   size_t size;
> }

> [spec:et:def:xnn-weights-cache.executorch.backends.xnnpack.delegate.xnn-weights-cache.offset-to-addr-fn]
> void* XNNWeightsCache::offset_to_addr(XNNWeightsCache* context, size_t offset)

> [spec:et:sem:xnn-weights-cache.executorch.backends.xnnpack.delegate.xnn-weights-cache.offset-to-addr-fn]
> Static callback wired into `xnn_weights_cache_provider.offset_to_addr`. Translates a packed-data offset (as returned by `look_up`/`look_up_or_insert`) back into the actual pointer to the packed bytes: returns `context->packed_data_ptrs_[offset]`. No bounds checking — `offset` is assumed to be a valid index the cache itself previously handed out; an out-of-range offset is undefined behavior in C++ (a Rust port should index the vec, which panics on out-of-range). The returned pointer may be null if the slot was released via `delete_packed_data`.

> [spec:et:def:xnn-weights-cache.executorch.backends.xnnpack.delegate.xnn-weights-cache.release-entry-fn]
> void XNNWeightsCache::release_entry(void* packed_data_ptr)

> [spec:et:sem:xnn-weights-cache.executorch.backends.xnnpack.delegate.xnn-weights-cache.release-entry-fn]
> Frees the backing storage for a single packed entry identified by its pointer `packed_data_ptr`. Called from `delete_packed_data` when an entry's ref_count hits 0 and it is not a persisted `from_load` entry.
> 1. Erase `packed_data_ptr` from `packed_pointer_to_container_` — if it was a heap-backed allocation, this drops the owning `std::string` container and frees the heap memory (a Rust port drops the owning `Vec<u8>`/box).
> 2. (Non-Windows) If `packed_data_ptr` maps to a per-entry file-backed region in `file_ptr_to_region_index_`: fetch that `MmapRegion`; if its `addr` is non-null and not `MAP_FAILED`, `munmap(addr, size)` and set `addr = nullptr`, `size = 0` (releasing that region's virtual memory in place, leaving the `mmap_regions_` vector length unchanged so other region indices stay valid). Then erase `packed_data_ptr` from `file_ptr_to_region_index_`.
> The caller nulls the corresponding `packed_data_ptrs_[offset]` slot after this returns so existing offsets held by other entries remain valid. If `packed_data_ptr` is neither heap- nor file-backed (e.g. a loaded shared-map entry), both lookups miss and this is effectively a no-op.

> [spec:et:def:xnn-weights-cache.executorch.backends.xnnpack.delegate.xnn-weights-cache.reserve-space-fn]
> void* XNNWeightsCache::reserve_space(XNNWeightsCache* context, size_t n)

> [spec:et:sem:xnn-weights-cache.executorch.backends.xnnpack.delegate.xnn-weights-cache.reserve-space-fn]
> Static callback wired into `xnn_weights_cache_provider.reserve_space`. Returns a pointer to at least `n` bytes of scratch space (aligned to `kPackedAllocationAlignment` = 64) into which XNNPACK will write packed weights. Uses the file-backed mmap path when a cache file is open, else falls back to heap.
>
> File-backed path (non-Windows, only if `packed_file_fd_ >= 0`):
> 1. `page_size = sysconf(_SC_PAGESIZE)`.
> 2. Round the current `packed_file_used_` UP to a page boundary → `file_offset = (packed_file_used_ + page_size - 1) & ~(page_size - 1)`. Round `n` up to a page multiple → `map_size = (n + page_size - 1) & ~(page_size - 1)`.
> 3. `ftruncate(packed_file_fd_, file_offset + map_size)` to grow the file to cover the new region. On failure: log Error, close the fd, set `packed_file_fd_ = -1`, and fall back to `reserve_space_heap(n)` per `[spec:et:sem:xnn-weights-cache.executorch.backends.xnnpack.delegate.xnn-weights-cache.reserve-space-heap-fn]`.
> 4. `mmap(nullptr, map_size, PROT_READ|PROT_WRITE, MAP_SHARED, packed_file_fd_, file_offset)`. On MAP_FAILED: log Error, close fd, set `packed_file_fd_ = -1`, fall back to `reserve_space_heap(n)`.
> 5. mmap results are page-aligned (>= 4 KiB) which trivially satisfies the 64-byte alignment requirement (a debug-only assertion checks `ptr % 64 == 0`).
> 6. Advance `packed_file_used_ = file_offset + map_size`. Record `file_ptr_to_region_index_[ptr] = mmap_regions_.size()`, push `{ptr, map_size}` onto `mmap_regions_`, and record `ptr_to_file_offset_[ptr] = file_offset` (so `save_packed_index` can serialize this entry's on-disk offset). Return `ptr`.
>
> Heap path: if no file fd (or the file path failed above), return `reserve_space_heap(n)`. On Windows, always the heap path.
>
> Each call produces a fresh, page-aligned region; regions are per-allocation and never coalesced.

> [spec:et:def:xnn-weights-cache.executorch.backends.xnnpack.delegate.xnn-weights-cache.reserve-space-heap-fn]
> void* XNNWeightsCache::reserve_space_heap(size_t n)

> [spec:et:sem:xnn-weights-cache.executorch.backends.xnnpack.delegate.xnn-weights-cache.reserve-space-heap-fn]
> Heap-backed allocation of at least `n` bytes aligned to `kPackedAllocationAlignment` (64). Returns the aligned pointer, or `nullptr` on OOM.
> 1. Over-allocate a `std::string` container of `raw_allocation_size = n + kPackedAllocationAlignment - 1` bytes (resize) so there is room to align within it.
> 2. Compute the aligned address inside the container via `std::align(64, n, container.data(), raw_allocation_size)` — this returns the first 64-byte-aligned address within the buffer that still has `n` bytes of room (and mutates the size-in/out param). Assert (ET_CHECK_MSG) the result is non-null ("Memory alignment failed.").
> 3. Move the container into `packed_pointer_to_container_[aligned_space]` so the backing storage stays owned/alive keyed by the aligned pointer (this is how `release_entry` later frees it). Return `aligned_space`.
> 4. If the allocation throws `std::bad_alloc`, catch it, log Error `"XNN weight cache failed to allocate <n> bytes: <what>."`, and return `nullptr`.
> A Rust port allocates an over-sized buffer, finds the aligned sub-pointer, stores the owning allocation keyed by that pointer, and returns null on allocation failure.

> [spec:et:def:xnn-weights-cache.executorch.backends.xnnpack.delegate.xnn-weights-cache.reset-for-fresh-write-fn]
> void XNNWeightsCache::reset_for_fresh_write()

> [spec:et:sem:xnn-weights-cache.executorch.backends.xnnpack.delegate.xnn-weights-cache.reset-for-fresh-write-fn]
> Non-Windows-only helper. Drops in-memory state that referenced the now-truncated cache file (called after `initialize_for_runtime` ftruncates the file to 0 for a fresh write), while preserving heap-backed entries so their offsets remain valid.
> 1. For each region in `mmap_regions_`: if `addr` is non-null and not `MAP_FAILED`, `munmap(addr, size)`. Then clear `mmap_regions_` and set `mmap_regions_synced_ = 0`.
> 2. Set `packed_file_used_ = 0`; clear `ptr_to_file_offset_` and `file_ptr_to_region_index_`.
> 3. Prune `name_to_packed_data_metadata_`: iterate its entries; for each, determine `is_heap_backed` by checking whether `entry.offset < packed_data_ptrs_.size()` AND `packed_data_ptrs_[entry.offset]` is non-null AND that pointer is present in `packed_pointer_to_container_`. Keep (advance past) heap-backed entries; erase all others (i.e. file-backed entries whose bytes were just truncated away). Use erase-returns-next-iterator to walk safely.
>
> Rationale captured in comments: heap-backed entries live in `packed_pointer_to_container_` and keep their `packed_data_ptrs_` slots, so existing offsets don't shift; only the file-mmap-backed metadata is invalidated by the truncation. `packed_data_ptrs_` itself is NOT cleared here (slots are retained to preserve offsets).

> [spec:et:def:xnn-weights-cache.executorch.backends.xnnpack.delegate.xnn-weights-cache.save-packed-index-fn]
> Error XNNWeightsCache::save_packed_index()

> [spec:et:sem:xnn-weights-cache.executorch.backends.xnnpack.delegate.xnn-weights-cache.save-packed-index-fn]
> Serializes the current packed-entry index as a trailer appended to the cache file so future processes can `load_packed_cache` and skip re-packing. Returns `Error::Ok`, `Error::Internal` (file write failure), and is a no-op returning `Error::Ok` on Windows or when no file fd is open. Trailer/record layout is the inverse of `[spec:et:sem:xnn-weights-cache.executorch.backends.xnnpack.delegate.xnn-weights-cache.load-packed-cache-fn]`, all integers little-endian via `[spec:et:sem:xnn-weights-cache.executorch.backends.xnnpack.delegate.append-le-fn]`.
>
> Steps (non-Windows):
> 1. If `packed_file_fd_ < 0`, return `Error::Ok`.
> 2. No-op guard: if `mmap_regions_.size() == mmap_regions_at_last_save_` AND `mmap_regions_at_last_save_ > 0` (nothing new since last successful save), return `Error::Ok` without touching the file (avoids bumping mtime and making the file appear modified every load).
> 3. `index_start = packed_file_used_` (the offset where the index region begins — just past the last packed region). Build a byte buffer `buf` and `entry_count = 0`.
> 4. For each `(name, meta)` in `name_to_packed_data_metadata_` (hash order): find `ptr = packed_data_ptrs_[meta.offset]` in `ptr_to_file_offset_`. If not present (heap-backed or released entry — not on disk), skip it. Otherwise increment `entry_count` and append one record: `name.size()` as u32, then the raw `name` bytes, then the file offset (`it->second`) as u64, then `meta.data_size` as u64, then `meta.seed` as u32.
> 5. Append the footer to `buf`: `index_start` as u64, `entry_count` as u32, `kCacheMagic` (0x58505743) as u32, `kCacheVersion` (2) as u32. Footer is exactly 20 bytes.
> 6. `ftruncate(packed_file_fd_, index_start + buf.size())`; on failure log Error and return `Error::Internal`.
> 7. `pwrite(packed_file_fd_, buf.data(), buf.size(), index_start)`; if the return != `buf.size()` log Error and return `Error::Internal`.
> 8. `fsync(packed_file_fd_)`; on failure log Error but CONTINUE (durability is best-effort; data is at least in page cache).
> 9. Log Info with `entry_count`, `index_start`, and `file_bytes = index_start + buf.size()`.
> 10. Promote freshly-packed entries to persisted: for each `(name, meta)` in `name_to_packed_data_metadata_`, if `meta.from_load` is false AND `packed_data_ptrs_[meta.offset]` is in `ptr_to_file_offset_`, set `meta.from_load = true` (so `delete_packed_data` preserves it across unload/reload).
> 11. Set `mmap_regions_at_last_save_ = mmap_regions_.size()`. Advance `packed_file_used_ = index_start + buf.size()` (PAST the trailer just written) so the next `reserve_space` appends after it rather than overwriting the trailer; the old trailer becomes orphan dead bytes and the next save writes a new trailer at the new file end (trailer is always kept at file end for cross-process reuse).
> 12. Keep `packed_file_fd_` OPEN (subsequent `finalize_for_runtime` auto-saves need a live fd). Return `Error::Ok`.

> [spec:et:def:xnn-weights-cache.executorch.backends.xnnpack.delegate.xnn-weights-cache.set-packed-cache-path-fn]
> void XNNWeightsCache::set_packed_cache_path(const std::string& path)

> [spec:et:sem:xnn-weights-cache.executorch.backends.xnnpack.delegate.xnn-weights-cache.set-packed-cache-path-fn]
> Stores `path` into `packed_cache_path_` (a plain string copy). This switches the cache into file-backed mode: once set (non-empty), `initialize_for_runtime` will open/load the file and `reserve_space` will allocate from a MAP_SHARED file instead of the heap. Contract (enforced by the manager, not this method): call once, before any other method, and never change it afterward — two instances sharing the same path would corrupt each other on truncation (SIGBUS), which `XNNWeightsCacheManager` prevents via per-path dedup. Setting an empty string leaves the cache in heap-only mode.

> [spec:et:def:xnn-weights-cache.executorch.backends.xnnpack.delegate.xnn-weights-cache.xnn-weights-cache-fn]
> XNNWeightsCache::XNNWeightsCache()

> [spec:et:sem:xnn-weights-cache.executorch.backends.xnnpack.delegate.xnn-weights-cache.xnn-weights-cache-fn]
> Constructor. Initializes the embedded `xnn_weights_cache_provider` struct (`weights_cache_`) so XNNPACK can drive this cache through C function pointers:
> 1. `weights_cache_.context = this` — the back-pointer every static callback receives and casts to `XNNWeightsCache*`.
> 2. Assign each provider function-pointer slot to the matching static member, cast to the C signature XNNPACK expects: `look_up` → `XNNWeightsCache::look_up`; `reserve_space` → `reserve_space`; `look_up_or_insert` → `look_up_or_insert`; `is_finalized` → `is_finalized`; `offset_to_addr` → `offset_to_addr`; `delete_cache` → `delete_cache`.
> All other members are default-initialized by their in-class initializers: `packed_file_fd_ = -1`, `packed_file_used_ = 0`, `mmap_regions_synced_ = 0`, `mmap_regions_at_last_save_ = 0`, empty maps/vectors, empty `packed_cache_path_`, and `is_finalized_` left as an ordinary bool set false only later by `initialize_for_runtime` (not initialized in the constructor — a Rust port should initialize it to false explicitly). The instance owns OS resources (fd, mmap regions) freed in the destructor and is non-copyable/non-movable.

> [spec:et:def:xnn-weights-cache.executorch.backends.xnnpack.delegate.xnn-weights-cache.mutex-fn]
> std::mutex& mutex() noexcept

> [spec:et:sem:xnn-weights-cache.executorch.backends.xnnpack.delegate.xnn-weights-cache.mutex-fn]
> Inline accessor returning a reference to the per-instance `instance_mutex_` (noexcept). The cache has NO internal synchronization; callers must hold this mutex around every public method call and around every XNNPACK callback that touches the cache during `xnn_create_runtime`. The class itself never locks it — it is entirely caller-owned. Lock order relative to other subsystems is `XNNWeightsCacheManager::meta_mutex_` first, then this mutex (see the manager). In a Rust port, this synchronization would typically be expressed by wrapping the cache in a `Mutex`/lock held by callers rather than an exposed accessor.

> [spec:et:def:xnn-weights-cache.executorch.backends.xnnpack.delegate.xnn-weights-cache.operator-fn]
> XNNWeightsCache& operator=(const XNNWeightsCache&) = delete

> [spec:et:sem:xnn-weights-cache.executorch.backends.xnnpack.delegate.xnn-weights-cache.operator-fn]
> Deleted copy-assignment operator (`= delete`). `XNNWeightsCache` owns OS resources — a file descriptor and mmap regions — and its `weights_cache_.context` back-pointer must keep pointing at the same object, so copying or moving would double-free/dangle. The copy constructor, move constructor, and move-assignment are likewise deleted; the type is non-copyable and non-movable. Any copy-assignment is a compile-time error. A Rust port keeps this type behind a stable address (e.g. `Arc`/pinned) and does not implement `Clone`/move for it.

