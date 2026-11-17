# Phase 8 experiments

Trials on Windows, Racket CS 9.3, Ryzen 5 8645HS, 12 reported processors.
Kernel rows use one warmup and seven timed calls. These early trials ran in
sequence; they are screening measurements, not the final alternating comparison.

| Trial | Reason to try it | Observation | Decision | Commit or evidence |
| --- | --- | --- | --- | --- |
| Statistical profile of 128-token generation | Find the expensive calls before changing kernels | The profiler reported zero stack samples. It timed the workload but did not identify hot functions | Do not infer CPU attribution from this profile. Measure projection, MLP, tied-head, and generation workloads directly | [Profile output](../bench/results/phase8-experiments-2026-11-17/profile.txt); kernel driver `3f1600e` |
| Four-column register accumulators, with a four-row variant for prefill | Avoid reading and writing the output buffer at every inner-loop step; reuse each weight across rows | The 128×768×3072 median rose from 23.065 to 49.522 ms. The single-row 768×3072 product rose from 0.444 to 0.748 ms. Thirty targeted tests passed | Discard. Correct arithmetic did not compensate for the slower MLP. Narrow column walks and register pressure are possible causes, not measured conclusions | [Baseline](../bench/results/phase8-experiments-2026-11-17/baseline.json), [register trial](../bench/results/phase8-experiments-2026-11-17/registers.json); uncommitted prototype |
| One future block per processor instead of four | Reduce setup and synchronization for the many small decode products | A 128-token generation trial reached 27.39 tok/s versus 21.77 in the baseline profile run. Kernel results were mixed: the tied head was not faster in the screening sample | Keep for the final alternating comparison. Select on inference results, not one microbenchmark | `1dec7c9`; [kernel samples](../bench/results/phase8-experiments-2026-11-17/scheduler.json) |
| Four-row weight reuse with output-buffer accumulation | Retain contiguous weight reads while sharing each weight among four rows, without sixteen live loop accumulators | With one block per processor, 128×768×3072 ranged from 20.119 to 34.245 ms, median 28.646. The scheduling-only median was 22.349 ms | Discard. A few fast samples did not justify the slower median and extra code | [Four-row trial](../bench/results/phase8-experiments-2026-11-17/four-rows.json), [scheduling control](../bench/results/phase8-experiments-2026-11-17/scheduler.json); uncommitted prototype |
| Visible causal attention and one reusable score row per head | Stop calculating masked scores and multiplying by their zero probabilities. Avoid a square score tensor and per-head query copies | Existing eager tests passed. A 1000-token prefill trial took 11.161 s; the earlier phase 7 median was 15.464 s on this machine | Keep, then compare against a fresh baseline. The earlier run alone does not isolate the revision effect | `d0b976d`; final prefill and allocation measurements |
| Size the generation cache to the request | Avoid zeroing a full 1024-row cache for a short generation | Existing context and cached-forward tests passed | Keep with the attention workspace. Measure allocation and startup as well as throughput | `d0b976d` |
| Reuse fresh product buffers for GELU and residual additions | Remove intermediate buffers without turning caller inputs or returned traces into scratch space | Forty-one targeted tests passed, including an input/earlier-output ownership check | Keep for final measurement. Do not claim a separate timing gain without an isolated comparison | `3857dd8` |
| Bounded top-k heap | Retain forty IDs without sorting all 50,257 logits; preserve lower IDs on ties | Twenty targeted tests passed, including tied and descending logits | Keep. The final sampling workload uses temperature 0.8, top-k 40, seed 1; greedy timing does not measure this change | `6f64173` |

The discarded prototypes were removed before the request to preserve failing
states as commits. Their source was not committed or retained; the raw timings
above remain, but the exact prototypes cannot be reproduced from a commit.
The register output contained one stray `#<procedure:+>` line. Its JSON payload
is preserved separately; that line was removed from the saved JSON, not from
the timing samples.

## Limits and next decisions

The final run compares the full set of retained changes with `1ad7035`. It does
not assign a percentage gain to each change. Timed allocation counts distinguish
attention scratch savings from the process peak left by loading the weights.
The actual tied output head is measured separately from the biased row product.

No packed-weight format, reduced precision, native backend, or persistent
worker pool was implemented. The experiments did not establish a need for those
larger changes. Full tensor-buffer recycling was narrowed to request-owned
attention scores and fresh product buffers, where ownership is explicit.

Implementation and test mistakes are recorded in [BUGLOG.md](BUGLOG.md). Earlier
fixes are marked retrospective; later caught mistakes are preserved before the
fix, as requested. The final PR must end on the corrected, tested revision.
