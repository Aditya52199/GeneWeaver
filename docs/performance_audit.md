# GPU Alignment Performance Audit

## Executive result

The requested 100x speedup is **not demonstrated**. This machine has no usable CUDA hardware: `scripts/check_cuda.py` reports Numba CUDASIM. CUDASIM executes CUDA kernels as Python and must not be presented as GPU performance. The benchmark did verify that its tested global-alignment scores match the CPU baseline.

The CUDA implementation is also structured in a way that is unlikely to meet 100x for a single pair of sequences: it launches a separate kernel for every anti-diagonal, synchronizing the device after the full wavefront. That is up to `m + n - 1` launches per alignment, in addition to matrix initialization and host/device copies.

## Measurements

Command: `python -m benchmarks.gpu_benchmark --lengths 10 20 50 --trials 1`

Environment: Python 3.14, Windows, Numba CUDASIM. The benchmark previously labeled the simulator as an RTX 4050 with fabricated specifications. `DeviceManager` now identifies the simulator directly and reports no hardware capability or VRAM values.

| Sequence lengths | DP cells | CPU (ms) | CUDASIM end-to-end (ms) | CPU/GPU ratio |
| --- | ---: | ---: | ---: | ---: |
| 10 × 10 | 121 | 0.060 | 208.979 | 0.00029x |
| 20 × 20 | 441 | 0.090 | 426.012 | 0.00021x |
| 50 × 50 | 2,601 | 0.391 | 1,179.901 | 0.00033x |

Ratios below 1x mean the simulator is slower. These are single-trial diagnostic measurements, not a hardware comparison or a reliable estimate of GPU speed. Existing CPU reference measurements in [benchmarking.md](benchmarking.md) show about 277 ms for 1,000 × 995 global alignment, but no corresponding hardware result is available here.

## Findings

1. **Per-diagonal launch overhead:** `CUDASyncAligner.align` launches one kernel per anti-diagonal. A 1,000 × 1,000 alignment therefore launches about 2,000 small kernels and serializes their progress. Launch overhead is paid for every diagonal even when it contains few cells.
2. **Unnecessary device allocations:** score and traceback arrays are allocated, then immediately replaced by arrays returned from `to_device` for host-initialized matrices. The first allocations are wasted.
3. **Large transfers for a scalar result:** the implementation copies both full matrices back to the host, although the score is a scalar and traceback can be recovered or selectively copied separately. This adds transfer and host memory costs.
4. **Limited GPU occupancy for one alignment:** anti-diagonal width grows and shrinks, while the first and last diagonals contain only one cell. Many launches therefore do little useful parallel work.
5. **Benchmark scope:** the current benchmark measures end-to-end alignment calls, which is appropriate for user-visible latency, but has no warmup, uses one aggregate mean, and only checks score equality. The former simulator device label was misleading and is now corrected. A future hardware run should report device identity and repeated median/dispersion, verify the full alignment result, and separate kernel, transfer, and end-to-end timings.

## Recommended path to 100x

Prioritize **batched independent alignment work**, such as many guide/target windows, rather than a single sequence pair. Independent jobs expose enough GPU parallelism to amortize dispatch and transfers. For a single large pair, replace per-diagonal launches with a tiled wavefront design that computes many cells per launch and synchronizes at tile boundaries; retain traceback only if the caller needs it. Avoid claiming the 100x goal until measured on real CUDA hardware against the same scoring rules, input set, and output requirements as the CPU baseline.

## Reproduction

Run the GPU diagnostic with `python scripts/check_cuda.py`. Run the small simulator check with the command above. On a CUDA-enabled machine, rerun the benchmark on hardware and record the actual GPU model, driver/runtime, Python/Numba versions, sequence sizes, warmup and trial counts, transfer costs, and correctness results alongside the timings.
