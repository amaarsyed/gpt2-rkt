# Phase 9: packed Q/K/V and worker measurements

Phase 9 starts from merged phase 8, `fda38fb`. The measured candidate is
`8ed5dc6`. Rejected head kernels remain in history but are not in that candidate.

## Results

The comparison does not establish a meaningful throughput gain. Greedy
generation measured 27.85 versus 28.03 tokens/s; sampling measured 27.20 versus
27.24 tokens/s. The differences are small relative to the observed variation.
The earlier 5–15% estimate depended on successful head or worker changes;
neither produced a gain worth retaining.

| Workload | Phase 8 median | Phase 9 median | Phase 9 time change |
| --- | ---: | ---: | ---: |
| Load | 3241.730 ms | 3091.165 ms | -4.64% |
| Prefill 16 | 234.159 ms | 232.087 ms | -0.88% |
| Prefill 128 | 1013.341 ms | 986.489 ms | -2.65% |
| Prefill 512 | 4563.970 ms | 4573.892 ms | +0.22% |
| Prefill 1000 | 11527.865 ms | 11462.071 ms | -0.57% |
| Greedy 128 tokens | 4596.740 ms | 4567.006 ms | -0.65% |
| Sampled 128 tokens | 4705.441 ms | 4698.893 ms | -0.14% |
| Tied head | 7.467 ms | 7.495 ms | +0.38% |
| Matmul 768×768×768 | 29.767 ms | 30.702 ms | +3.14% |

Negative time changes are faster. No loader or square-matrix kernel change was
retained; their timing differences are not evidence of an optimization there.
One baseline head sample took 35.005 ms while the other two took 7.283 and
7.467 ms. This is another reason not to read the small medians as firm gains.

The retained copy reduction lowers timed allocation. Greedy generation falls
from 626,025,536 to 597,274,112 bytes (4.59%). The 1000-token prefill falls from
1,551,391,120 to 1,333,450,608 bytes (14.05%). These are cumulative allocations,
not retained memory. The phase keeps this measured reduction and unchanged
answers without claiming the estimated speedup.

All 54 requested samples completed and token IDs matched between trees and
runs. There are 53 distinct PIDs: the OS reused one after an earlier child
exited. The driver starts and waits for a separate child for every sample;
PID uniqueness alone is not its isolation mechanism.

Raw [samples](../bench/results/phase9-windows-2026-11-17/samples.jsonl),
[summary](../bench/results/phase9-windows-2026-11-17/summary.json), and
[comparison](../bench/results/phase9-windows-2026-11-17/comparison.md) are retained.

## Changes

The cached block reads queries from the packed projection and writes keys and
values straight into their cache rows. It no longer allocates three tensors
from `qkv-split` on that path. The uncached reference path still uses the split.

`attend` accepts `#:query-width` for leading query columns in a wider stored
row. Its default is the stored row width, preserving existing calls. Output
rows use the logical width, independently of the packed input stride. Cache
shapes and capacity are checked before writing.

The NN module adds optional `current-decode-workers` and
`current-prefill-workers` parameters. `#f` keeps the host count; a positive
budget is capped at that count. These control matrix products, not attention
heads. The worker sweep selected the existing 12-worker default on this host.

The scalar output-head loop is retained. Four head experiments did not produce
a gain worth keeping. Their source commits, timings, allocation checks, and
decisions are in the [experiment log](PHASE9-EXPERIMENTS.md).

## Validation and method

The local full suite passed 124 tests, including the real GPT-2 eager checks.
Tests cover short and empty dot tails, worker budgets, packed query strides,
cached prefixes, untouched cache tails, and interleaved requests. Precision and
existing reference tolerances are unchanged.

The final comparison alternates baseline/candidate, candidate/baseline, then
baseline/candidate. Every sample gets a fresh process, one warmup, and one timed
call. Medians come from three samples. This is Windows, Racket CS 9.3, Ryzen 5
8645HS, with 12 logical processors. Power, temperature, and background activity
are not controlled.

Greedy and sampled generation include the four-token prompt prefill. Sampling
uses temperature 0.8, top-k 40, seed 1. Token counts and IDs must match across
trees and runs. Allocation counts cover the timed call; process peaks include
loading and are not OS working-set readings. The tied head is measured as a
transposed product.

## Reproduce

With the pinned assets fetched, run from the repository root in PowerShell:

```powershell
git archive --format=zip --output=models/phase9-baseline.zip fda38fb
Expand-Archive models/phase9-baseline.zip models/phase9-baseline
./bench/phase8.ps1 -Phase phase9 -BaselineRef fda38fb `
  -BaselineSource models/phase9-baseline -OutDir bench/results/my-phase9-run
```

The driver retains its historical filename and defaults. Use a new output
directory. Add `-Smoke` for a tiny-model compatibility check, not a speed result.
Remove the extracted snapshot after the run. Worker-sweep reproduction uses
`bench/sweep-workers.ps1` with a snapshot of `5456731` and its resolved full SHA.
