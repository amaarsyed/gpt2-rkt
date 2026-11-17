# Recorded benchmarks

Recorded 2026-11-17. Windows runs use Racket CS 9.3, Ryzen 5 8645HS, 12 logical processors. Times are medians in ms; generation includes the four-token prompt prefill. Compare within each pair: separate runs have different machine conditions.

## Phase 6 → 7

| Workload | Phase 6 | Phase 7 |
| --- | ---: | ---: |
| Load | 3.908 s | 3.244 s |
| Prefill 16 | 223.684 ms | 232.051 ms |
| Prefill 128 | 1299.934 ms | 1336.835 ms |
| Prefill 512 | 6507.507 ms | 6288.714 ms |
| Prefill 1000 | 14299.251 ms | 15463.771 ms |
| Generate 128 | 6355.449 ms | 5835.142 ms |
| Matmul 768×768×768 | 31.930 ms | 30.538 ms |
| Matmul 1×768×50257 | 7.088 ms | 7.812 ms |

Greedy: 20.14 → 21.94 tok/s. Sequential runs, one warmup and three timed calls per workload; matching token IDs. The row-shaped matmul is biased Conv1D, not the tied head.

## Phase 8 comparison

Baseline `1ad7035` → candidate `e8af302`. Alternating AB/BA/AB, one warmup and one timed call per fresh process, three samples per tree; matching token IDs.

| Workload | Baseline ms | Phase 8 ms | Allocated MB, baseline → candidate |
| --- | ---: | ---: | ---: |
| load 0 | 3167.608 | 2998.507 | — |
| prefill 16 | 219.288 | 228.170 | 195.607 → 27.602 |
| prefill 128 | 1176.652 | 1027.874 | 503.399 → 194.194 |
| prefill 512 | 6135.564 | 5030.384 | 1837.611 → 779.014 |
| prefill 1000 | 15183.579 | 10928.493 | 4572.572 → 1551.351 |
| decode 128 | 5838.217 | 4583.654 | 1793.805 → 625.925 |
| sample 128 | 6824.062 | 4748.245 | 2348.517 → 782.969 |
| head 1 | 7.771 | 7.512 | 0.812 → 0.801 |
| square 768 | 30.012 | 30.282 | 5.717 → 5.508 |

[Raw samples](../bench/results/phase8-windows-2026-11-17/samples.jsonl) · [Summary, including process peaks](../bench/results/phase8-windows-2026-11-17/summary.json) · [Method and interpretation](PHASE8.md)

## Phase 9 comparison

Baseline `fda38fb` → candidate `8ed5dc6`. Alternating AB/BA/AB, one warmup and one timed call per fresh process, three samples per tree; matching token IDs.

| Workload | Baseline ms | Phase 9 ms | Allocated MB, baseline → candidate |
| --- | ---: | ---: | ---: |
| load 0 | 3241.730 | 3091.165 | — |
| prefill 16 | 234.159 | 232.087 | 27.592 → 24.218 |
| prefill 128 | 1013.341 | 986.489 | 194.244 → 165.914 |
| prefill 512 | 4563.970 | 4573.892 | 779.010 → 667.730 |
| prefill 1000 | 11527.865 | 11462.071 | 1551.391 → 1333.451 |
| decode 128 | 4596.740 | 4567.006 | 626.026 → 597.274 |
| sample 128 | 4705.441 | 4698.893 | 782.924 → 752.092 |
| head 1 | 7.467 | 7.495 | 0.801 → 0.798 |
| square 768 | 29.767 | 30.702 | 5.505 → 6.096 |

[Raw samples](../bench/results/phase9-windows-2026-11-17/samples.jsonl) · [Summary, including process peaks](../bench/results/phase9-windows-2026-11-17/summary.json) · [Method and interpretation](PHASE9.md)

Phase 8: greedy 21.92 → 27.93 tok/s; sampled 18.76 → 26.96. Phase 9: greedy 27.85 → 28.03; sampled 27.20 → 27.24. Phase 9 establishes lower allocation, with no meaningful throughput gain. Allocated MB means cumulative timed allocation (1 MB = 1,000,000 bytes). Process peaks include loading; neither counter is OS working-set memory.

## Phase 8 kernel trials

Screening medians in ms, one warmup and seven timed calls; sequential trials.

| Shape | baseline | registers | scheduler | four-rows |
| --- | ---: | ---: | ---: | ---: |
| 1×768×2304 | 0.295 | 0.287 | 0.221 | 0.286 |
| 1×768×3072 | 0.444 | 0.748 | 0.436 | 0.416 |
| 1×3072×768 | 0.756 | 0.728 | 0.675 | 0.672 |
| 128×768×3072 | 23.065 | 49.522 | 22.349 | 28.646 |
| 768×768×768 | 32.117 | 29.486 | 35.252 | 31.494 |
| 1×768×50257 (tied head) | 7.246 | 7.088 | 8.445 | 7.864 |

[Hypotheses, decisions, commits, and raw samples](PHASE8-EXPERIMENTS.md). Retained scheduler changes; rejected register and four-row kernels.

## Phase 9 kernel trials

Screening medians in ms, one warmup and seven timed calls; sequential trials.

| Shape | baseline-kernels | ordered-unroll-kernels | four-output-kernels |
| --- | ---: | ---: | ---: |
| 1×768×2304 | 0.235 | 0.255 | 0.257 |
| 1×768×3072 | 0.405 | 0.387 | 0.419 |
| 1×3072×768 | 0.641 | 0.681 | 0.663 |
| 128×768×3072 | 21.993 | 25.066 | 22.854 |
| 768×768×768 | 31.243 | 38.187 | 33.882 |
| 1×768×50257 (tied head) | 6.755 | 17.832 | 7.427 |

[Hypotheses, decisions, commits, and raw samples](PHASE9-EXPERIMENTS.md). Rejected both head rewrites.

## Phase 9 head follow-ups

| Trial | Median ms | Timed allocation |
| --- | ---: | --- |
| ordered-unroll-head | 14.611 | 155.839–156.105 MB |
| inline-head | 8.281 | 0.697–1.049 MB |
| split-tail-head | 15.463 | 156.307–156.626 MB |

All three rejected: no measured speed gain. Unrolled allocation suggests lost flonum unboxing; compiler cause remains unverified.

## Phase 9 workers

| Matrix workers | Greedy 128 tokens, ms | Prefill 128 tokens, ms | Tied head, ms |
| ---: | ---: | ---: | ---: |
| 1 | 11036.114 | 5424.326 | 36.498 |
| 2 | 10863.069 | 4360.699 | 18.688 |
| 4 | 6797.511 | 2375.533 | 10.185 |
| 6 | 5105.980 | 1664.859 | 8.913 |
| 12 | 4509.203 | 1150.675 | 6.722 |

Ascending worker counts, one fresh process per setting, one warmup and three timed calls. Snapshot `5456731`; optional budgets retained, default unchanged. [Raw results](../bench/results/phase9-workers-2026-11-17/summary.json).

## Cloud VM reference

Intel Xeon, four logical processors, Racket CS 9.3, revision `1474696`. Separate host; do not treat it as a phase-to-phase comparison or an M5 Pro measurement.

| configuration | median | peak bytes | pid |
| --- | --- | ---: | ---: |
| load | 5.0367 s | 2,248,894,968 | 22444 |
| prefill 16 | 488.9253 ms | 2,248,844,344 | 22445 |
| prefill 128 | 1837.0529 ms | 2,248,863,000 | 22473 |
| prefill 512 | 8394.3052 ms | 2,248,857,768 | 22481 |
| prefill 1000 | 20660.1813 ms | 2,248,871,536 | 22494 |
| decode 128 (incl. 4-token prefill) | 9.3366 tok/s | 2,248,894,184 | 22505 |
| matmul 768×768×768 | 14.6842 GFLOP/s | 116,141,152 | 22534 |
| matmul 1×768×50257 | 6.0236 GFLOP/s | 413,302,272 | 22539 |

Resident after load and collection: 1,085,651,056 bytes. [Full reference and raw results](../bench/results/vm-reference-2026-11-17.md).

Power, temperature, and background activity were uncontrolled. Three-sample comparisons do not establish statistical significance. Experiment logs also retain one-off screening observations and the profiler run, which collected zero stack samples.
