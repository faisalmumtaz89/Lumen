# Lumen CUPTI injection profiler

Out-of-process per-kernel profiler for the Lumen CUDA decode path. Loaded by the
CUDA driver via `CUDA_INJECTION64_PATH`, so the Lumen binary needs no rebuild, no
CUPTI link dependency, and no source change.

> **Status: written, not yet built or run.** It has been syntax-checked against a
> stub header on macOS only. Nothing here has executed against real CUPTI or a
> real GPU. Build it in the container first (step 1) and treat the first run as a
> smoke test, not as data.

## Why this exists alongside `LUMEN_CUDA_PROFILE`

The two instruments answer different questions and neither subsumes the other.

| | `LUMEN_CUDA_PROFILE` (in-process) | this library (CUPTI) |
|---|---|---|
| Granularity | source region ("phase") | individual kernel |
| Measures | GPU-timeline span of a region | queued / submitted / start / end per kernel |
| Sees true busy time | no — a span includes idle inside the region | yes — `sum(end - start)` |
| Sees submission stalls | only in aggregate, as a residual | per kernel, directly |
| Attributes to source | yes, phases are named regions | no, only kernel names |
| Overhead | **unmeasured.** Adds 198 CUDA event records/token at level 1, 634 at level 2, against ~400 existing kernel launches/token | **unmeasured.** CUPTI buffering only; no forced syncs |
| Build impact | in the Rust crate | none |

Neither overhead figure has been measured. Both must be established by an A/B
(flag off vs on, same weights, same prompt) before any absolute number from
either instrument is quoted. Level 1 vs level 2 disagreeing is itself the
overhead measurement for the depth-1 brackets.

Use the phase profiler to find *which* region owns the time. Use this to find out
whether that region is actually device-bound or just badly fed.

`nsys` would normally cover this, but it cannot finalize inside the Modal
container (`No GPU associated to the given UUID`). CUPTI works because it runs
in-process and never resolves a device UUID out of band.

## The measurement that matters

With `cuptiActivityEnableLatencyTimestamps(1)`, each kernel record carries four
timestamps rather than two:

```
queued ──────► submitted ──────► start ──────► end
       submit           launch          device
       latency          latency         busy
```

Per token:

* `sum(end - start)` — **true device busy time**. This is the number event
  brackets cannot give you.
* `max(end) - min(queued)` — the token's full span.
* the difference — **idle**: the GPU had nothing to run.
* `start[i] - end[i-1]` over consecutive kernels — **per-gap starvation**, which
  localizes *which* launch the host was late for.

A decode path that is launch-bound rather than bandwidth-bound shows up as small
`end - start` values separated by large gaps. That is the distinction the phase
profiler's `gpu_unattributed` residual can only hint at.

## 1. Build (in the container)

Requires the CUPTI headers and library, present in `nvidia/cuda:*-devel` images
under `$CUDA_HOME/extras/CUPTI`.

```sh
cd /path/to/lumen/tools/cupti-inject
make
```

The Makefile probes which `CUpti_ActivityKernel*` struct version the toolkit
provides (the record type is versioned and gains fields with each CUDA release)
and requires that it expose all four latency-timestamp fields. If the probe
cannot run it warns and defaults to `CUpti_ActivityKernel9` (CUDA 12.x).

Override if it picks wrong:

```sh
make KERNEL_STRUCT=CUpti_ActivityKernel8
make print-struct          # show what was probed, build nothing
```

If CUPTI lives elsewhere:

```sh
make CUDA_HOME=/usr/local/cuda-12.2 CUPTI_DIR=/opt/cupti
```

## 2. Run

```sh
export CUDA_INJECTION64_PATH=$PWD/liblumen_cupti_inject.so
export LUMEN_CUPTI_CSV=/tmp/lumen-9bq4-decode.csv

lumen run qwen3.5-9b:q4_0 "..." 2>&1 | tee /tmp/run.log
```

`CUDA_INJECTION64_PATH` must be an **absolute** path. The driver loads the
library before CUDA initializes, which is why activity kinds can be enabled in
time to catch the very first launch.

### Environment

| variable | default | meaning |
|---|---|---|
| `CUDA_INJECTION64_PATH` | — | absolute path to `liblumen_cupti_inject.so`. Required. |
| `LUMEN_CUPTI_CSV` | `/tmp/lumen-cupti.csv` | output path. Falls back to stderr if unopenable. |
| `LUMEN_CUPTI_MEMOPS` | unset | `1` also records memcpy/memset (catches the per-layer `attn_proj -> x_gpu` commit and the argmax D2H). |
| `LUMEN_CUPTI_QUIET` | unset | `1` suppresses the `[CUPTI]` banner. |

### Permissions

CUPTI activity tracing needs GPU performance counters on some drivers. If
enabling fails with `CUPTI_ERROR_INSUFFICIENT_PRIVILEGES`, either run with
`--cap-add=SYS_ADMIN` or set
`NVreg_RestrictProfilingToAdminUsers=0` on the host module. Kernel *activity*
tracing (what this tool uses) is usually unrestricted, unlike counter
collection — try without the capability first.

## 3. Do NOT combine the two profilers in one run

Run them separately. `LUMEN_CUDA_PROFILE` records ~200-630 extra CUDA events per
token; those show up in CUPTI's timeline and inflate the gaps you are trying to
measure. Keep `LUMEN_CUDA_PROFILE` unset for CUPTI runs.

Also keep `LUMEN_XCHK`, `LUMEN_MOE_PROBE`, and
`LUMEN_CUDA_GDN_SUBSTAGE_TIMING` unset: each injects blocking device-to-host
copies or forced syncs into the decode path and will dominate the result.

## 4. Output schema

One row per kernel (and per memcpy/memset when `LUMEN_CUPTI_MEMOPS=1`):

```
kind,name,device,stream,correlation,queued_ns,submitted_ns,start_ns,end_ns,
grid_x,grid_y,grid_z,block_x,block_y,block_z,dyn_smem,static_smem
```

* Timestamps are raw CUPTI nanoseconds on one monotonic device timeline, so rows
  are directly differenceable across records.
* `queued_ns` / `submitted_ns` are `0` when latency timestamps could not be
  enabled. They are emitted verbatim rather than pre-differenced so that "not
  measured" stays distinguishable from a genuine zero interval. **If those
  columns are all zero, the run measured nothing this tool exists for** — check
  the `[CUPTI] WARN` lines.
* For memcpy/memset rows the `bytes` value occupies the `grid_x` column and the
  copy kind occupies `grid_y`; the remaining geometry columns are zero.
* Sorting by `start_ns` gives execution order; sorting by `queued_ns` gives
  submission order. **The two differing is itself a finding.**

### Dropped records

If CUPTI's buffers overflow, the library prints

```
[CUPTI] WARN <n> activity records were DROPPED -- the CSV is incomplete
```

A truncated CSV summed as if whole reads as a *faster* run. Always check for
this line before trusting a total.

## 5. Analysis

`summarize.py` (same directory, stdlib only) does the standard reductions:

```sh
python3 summarize.py /tmp/lumen-9bq4-decode.csv
python3 summarize.py /tmp/lumen-9bq4-decode.csv --top 25
python3 summarize.py /tmp/lumen-9bq4-decode.csv --gaps
```

It reports, per kernel name: call count, total device busy time, mean/median
busy, and share of total busy; plus the whole-run busy-vs-span decomposition
and, with `--gaps`, the largest inter-kernel starvation gaps.

## 6. Known limitations

* Kernel names are whatever NVRTC produced. Lumen's kernels are `extern "C"`, so
  they should appear unmangled; C++ mangled names are not demangled here.
* There is no token boundary in the CSV. The decode loop does not emit a marker,
  so per-token slicing has to be inferred — the `argmax_f32` kernel fires exactly
  once per token and is the natural delimiter. `summarize.py --gaps` uses it.
* CUPTI's own buffering perturbs timing somewhat. It is far lighter than forcing
  a sync per region, but it is not free and has not been quantified here.
* `CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL` is used deliberately rather than
  `CUPTI_ACTIVITY_KIND_KERNEL`: the latter serializes kernel execution to time
  it, which would destroy the overlap this tool exists to measure.
