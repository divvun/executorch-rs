# kernels/portable/cpu/op_rand.cpp

> [spec:et:def:op-rand.torch.executor.native.rand-out-fn]
> Tensor&

> [spec:et:sem:op-rand.torch.executor.native.rand-out-fn]
> Fills `out` (resized to `sizes`) with independent samples from the uniform distribution over [0, 1); returns `out`.
>
> Steps:
> 1. Construct an RNG: `std::mt19937 gen` seeded from `std::random_device`, and `std::uniform_real_distribution<double> dist(0.0, 1.0)`. (Freshly seeded per call → nondeterministic; exact values not reproducible.)
> 2. ET_KERNEL_CHECK_MSG: `resize_tensor(out, sizes)` == Ok (message "Failed to resize output tensor."); on failure set `Error::InvalidArgument` on `ctx`, return `out` unchanged.
> 3. Dispatch on `out.scalar_type()` over FLOATHBF16 (Float, Double, Half, BFloat16); unsupported dtype → InvalidArgument, return `out`.
> 4. For each flat index `i` in [0,out.numel()): draw `dist(gen)` (a double in [0,1)) and store `out_data[i] = static_cast<CTYPE>(dist(gen))`.
> 5. Return `out`.

