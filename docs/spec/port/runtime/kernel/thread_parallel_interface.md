# runtime/kernel/thread_parallel_interface.h

> [spec:et:def:thread-parallel-interface.executorch.extension.get-thread-num-fn]
> inline int64_t get_thread_num()

> [spec:et:sem:thread-parallel-interface.executorch.extension.get-thread-num-fn]
> This is the no-threadpool build variant (compiled when `ET_USE_THREADPOOL`
> is not defined). It takes no arguments and unconditionally returns the
> constant `0` (the index of the only, main thread). It has no side effects.
> When `ET_USE_THREADPOOL` is defined instead, `get_thread_num` is an
> out-of-line function backed by the threadpool that returns the calling
> worker's thread index; that variant is not specified by this rule.

> [spec:et:def:thread-parallel-interface.executorch.extension.internal.parallel-for-no-threadpool-fn]
> inline bool parallel_for_no_threadpool( const int64_t begin, const int64_t end, const int64_t grain_size, const Func& f)

> [spec:et:sem:thread-parallel-interface.executorch.extension.internal.parallel-for-no-threadpool-fn]
> Runs the callback `f` over the half-open work-item range `[begin, end)`
> synchronously on the calling thread (no actual parallelism). `f` has
> signature `void f(int64_t chunk_begin, int64_t chunk_end)` and is expected
> to process the half-open sub-range `[chunk_begin, chunk_end)`.
>
> Argument validation, in order:
> - `ET_CHECK_OR_RETURN_FALSE(begin >= 0 && end >= 0 && end >= begin, ...)`:
>   if `begin` is negative, or `end` is negative, or `end < begin`, log the
>   failed condition with `begin` and `end` and return `false` immediately
>   (do not call `f`).
> - `ET_CHECK_OR_RETURN_FALSE(grain_size > 0, ...)`: if `grain_size <= 0`,
>   log and return `false` immediately. Note `grain_size` is otherwise
>   ignored in this variant — there is no chunking, the whole range is one
>   unit of work.
>
> Behavior after validation depends on build mode:
> - Debug builds (`NDEBUG` not defined): deliberately invoke `f` on each
>   single-element chunk in reverse order to catch callers that wrongly
>   assume in-order, whole-range execution. For each `i` in `[begin, end)`,
>   let `offset = i - begin` and `idx = end - offset - 1`, then call
>   `f(idx, idx + 1)`. This visits `idx` values `end-1, end-2, ..., begin`,
>   each as a one-element chunk `[idx, idx+1)`.
> - Release builds (`NDEBUG` defined): call `f(begin, end)` exactly once,
>   passing the entire range as a single chunk.
>
> If `begin == end` the range is empty: no call to `f` is made in either
> build mode, and the function returns `true`.
>
> Returns `true` after the work is dispatched. A conforming Rust port may
> collapse the two build modes into a single call over the whole range; the
> reverse single-element iteration is purely a debug assertion aid and is
> not part of the observable contract.

> [spec:et:def:thread-parallel-interface.executorch.extension.parallel-for-fn]
> bool parallel_for( const int64_t begin, const int64_t end, const int64_t grain_size, const Func& func)

> [spec:et:sem:thread-parallel-interface.executorch.extension.parallel-for-fn]
> This rule covers the no-threadpool build variant (compiled when
> `ET_USE_THREADPOOL` is not defined). It is a thin template forwarder: it
> calls `internal::parallel_for_no_threadpool(begin, end, grain_size, func)`
> and returns its result verbatim. All argument validation, the empty-range
> behavior, and the synchronous single-thread execution are exactly as
> described in
> `[spec:et:sem:thread-parallel-interface.executorch.extension.internal.parallel-for-no-threadpool-fn]`.
>
> When `ET_USE_THREADPOOL` is defined instead, `parallel_for` is an
> out-of-line function that partitions `[begin, end)` into chunks of at least
> `grain_size` work items and dispatches them across the threadpool's worker
> threads, invoking `f(chunk_begin, chunk_end)` per chunk (possibly
> concurrently and in unspecified order), returning `true` on success; that
> threadpool variant is not specified by this rule. In both variants the
> callback has signature `void(int64_t, int64_t)` and thread-local state is
> not propagated to workers, so a mutated capture must be synchronized by the
> caller.

> [spec:et:def:thread-parallel-interface.executorch.extension.set-thread-num-fn]
> inline void set_thread_num(int64_t thread_num)

> [spec:et:sem:thread-parallel-interface.executorch.extension.set-thread-num-fn]
> This is the no-threadpool build variant (compiled when `ET_USE_THREADPOOL`
> is not defined). Setting a thread number is meaningless without threading
> support, so it ignores its `thread_num` argument entirely and does nothing
> functional. In debug builds it fires `ET_DCHECK_MSG(false, "cannot
> set_thread_num without threading support!")`, which aborts (assertion
> failure) if debug checks are enabled; in release builds the check is
> compiled out and the function is a no-op. It returns nothing (`void`).
> When `ET_USE_THREADPOOL` is defined instead, `set_thread_num` is an
> out-of-line function that records the calling worker's thread index; that
> variant is not specified by this rule.

