# Benchmarks

Recorded on 2026-11-17. Compare numbers only inside one table. A laptop and a cloud VM are not the same measurement, and an earlier revision is not the current sources. Nothing here is an Apple M5 Pro run. The repository does not contain one.

Power, temperature, and other programs on the machine were not controlled. Three samples do not establish a statistical result. Where a section says the token ids matched, they matched across the runs in that section.

Allocated megabytes are cumulative bytes allocated during the timed call, with 1 MB = 1,000,000 bytes. They are not memory still in use. Process peaks include loading the weights. Neither counter is the operating system's working set.

## This tree, Windows laptop

AMD Ryzen 5 8645HS, 12 logical processors, Racket CS 9.3. The run alternated this tree with the previous one, AB then BA then AB. Each sample is a fresh process, one warmup, one timed call. The median is three samples. Token ids matched.

The right-hand column is `8ed5dc6`. Comments in `src/` changed after that commit. The engine did not. The left-hand column is `fda38fb`, the same machine before packed queries were written straight into the cache. Generation includes the four-token prompt `Once upon a time`. Sampling is temperature 0.8, top-k 40, seed 1.

| Workload | Previous, ms | This tree, ms | Allocated MB |
| --- | ---: | ---: | ---: |
| load | 3241.730 | 3091.165 | n/a |
| prefill 16 | 234.159 | 232.087 | 27.592 to 24.218 |
| prefill 128 | 1013.341 | 986.489 | 194.244 to 165.914 |
| prefill 512 | 4563.970 | 4573.892 | 779.010 to 667.730 |
| prefill 1000 | 11527.865 | 11462.071 | 1551.391 to 1333.451 |
| decode 128 | 4596.740 | 4567.006 | 626.026 to 597.274 |
| sample 128 | 4705.441 | 4698.893 | 782.924 to 752.092 |
| tied head | 7.467 | 7.495 | 0.801 to 0.798 |
| matmul 768×768×768 | 29.767 | 30.702 | 5.505 to 6.096 |

Greedy throughput is 27.85 tok/s on the left and 28.03 tok/s on the right. Sampled is 27.20 and 27.24. The time differences are small next to the variation in the samples. One tied-head sample on the left took 35.005 ms while the other two took 7.283 and 7.467 ms. What this revision did keep is the lower allocation: greedy generation went from 626,025,536 bytes to 597,274,112, and the 1000-token prefill from 1,551,391,120 to 1,333,450,608.

Raw samples are in [`bench/results/phase9-windows-2026-11-17/`](../bench/results/phase9-windows-2026-11-17/). The directory name is historical. The numbers are this comparison.

## The cache and attention revision, same laptop

`1ad7035` on the left, `e8af302` on the right. Same alternating method, same prompt, same sampling. This is the revision that switched to one future per processor, scored only visible keys, sized the cache to the request, reused product buffers, and replaced the full top-k sort with a heap.

| Workload | Before, ms | After, ms | Allocated MB |
| --- | ---: | ---: | ---: |
| load | 3167.608 | 2998.507 | n/a |
| prefill 16 | 219.288 | 228.170 | 195.607 to 27.602 |
| prefill 128 | 1176.652 | 1027.874 | 503.399 to 194.194 |
| prefill 512 | 6135.564 | 5030.384 | 1837.611 to 779.014 |
| prefill 1000 | 15183.579 | 10928.493 | 4572.572 to 1551.351 |
| decode 128 | 5838.217 | 4583.654 | 1793.805 to 625.925 |
| sample 128 | 6824.062 | 4748.245 | 2348.517 to 782.969 |
| tied head | 7.771 | 7.512 | 0.812 to 0.801 |
| matmul 768×768×768 | 30.012 | 30.282 | 5.717 to 5.508 |

Greedy went from 21.92 tok/s to 27.93. Sampled went from 18.76 to 26.96. The 16-token prefill got a little slower. The square matmul did not move. Load times moved, and nothing in the loader was changed to explain that, so treat the load row as noise. Greedy allocation fell from 1,793,805,440 bytes to 625,925,216. The 1000-token prefill fell from 4,572,571,728 to 1,551,351,184. Those are cumulative allocations, not the resident set, and the large peak from loading the weights is still there.

Raw samples are in [`bench/results/phase8-windows-2026-11-17/`](../bench/results/phase8-windows-2026-11-17/).

## Two earlier revisions, same laptop

`4e0631f` then `0f32e9b`, run one after the other rather than alternated. Same pinned weights, one warmup and three timed calls, each configuration in its own process. Token ids matched. The later of the two reads each attention head from the cache in place instead of copying the prefix out.

| Workload | `4e0631f` | `0f32e9b` |
| --- | ---: | ---: |
| Load | 3.908 s | 3.244 s |
| Prefill 16 | 223.684 ms | 232.051 ms |
| Prefill 128 | 1299.934 ms | 1336.835 ms |
| Prefill 512 | 6507.507 ms | 6288.714 ms |
| Prefill 1000 | 14299.251 ms | 15463.771 ms |
| Generate 128 | 6355.449 ms | 5835.142 ms |
| Matmul 768×768×768 | 31.930 ms | 30.538 ms |
| Matmul 1×768×50257 | 7.088 ms | 7.812 ms |

Greedy throughput is 20.14 tok/s, then 21.94. The 1000-token prefill is slower on the right. Loading ranged from 3.428 s to 4.700 s on the left and 3.065 s to 3.338 s on the right, so the load median is not a clean result. The row-shaped matmul is the biased Conv1D kernel, not the tied head.

| Memory | `4e0631f` | `0f32e9b` |
| --- | ---: | ---: |
| Resident after load and a collection | 1,092,109,688 bytes | 1,092,135,064 bytes |
| Generation process peak, including load | 2,255,331,536 bytes | 2,255,368,432 bytes |

Raw results are in [`bench/results/phase6-vs-phase7-windows-2026-11-17/`](../bench/results/phase6-vs-phase7-windows-2026-11-17/).

## Kernel trials

Screening medians, one warmup and seven timed calls, trials run in sequence on the same laptop. These picked what to keep. They are not the alternating comparisons above.

The first round. "One block" is the scheduler that stayed: one future block per processor. The register and four-row kernels were discarded.

| Shape | baseline | registers | one block | four rows |
| --- | ---: | ---: | ---: | ---: |
| 1×768×2304 | 0.295 | 0.287 | 0.221 | 0.286 |
| 1×768×3072 | 0.444 | 0.748 | 0.436 | 0.416 |
| 1×3072×768 | 0.756 | 0.728 | 0.675 | 0.672 |
| 128×768×3072 | 23.065 | 49.522 | 22.349 | 28.646 |
| 768×768×768 | 32.117 | 29.486 | 35.252 | 31.494 |
| 1×768×50257, tied head | 7.246 | 7.088 | 8.445 | 7.864 |

The second round tried to speed up the tied head. Both rewrites were discarded. The scalar loop stayed.

| Shape | scalar | ordered unroll | four outputs |
| --- | ---: | ---: | ---: |
| 1×768×2304 | 0.235 | 0.255 | 0.257 |
| 1×768×3072 | 0.405 | 0.387 | 0.419 |
| 1×3072×768 | 0.641 | 0.681 | 0.663 |
| 128×768×3072 | 21.993 | 25.066 | 22.854 |
| 768×768×768 | 31.243 | 38.187 | 33.882 |
| 1×768×50257, tied head | 6.755 | 17.832 | 7.427 |

Follow-ups on the tied head alone, also discarded:

| Trial | Median ms | Timed allocation |
| --- | ---: | --- |
| ordered unroll | 14.611 | 155.839 to 156.105 MB |
| forced inline | 8.281 | 0.697 to 1.049 MB |
| unroll with the tail split out | 15.463 | 156.307 to 156.626 MB |

The extra allocation on the unrolled variants was measured. It looks like lost flonum unboxing. Nothing in the compiler output was checked to confirm that.

## Worker counts

Same laptop, snapshot `5456731`. One fresh process per setting, settings in ascending order, one warmup and three timed calls. Only matrix products use the budget. Attention heads are scheduled on their own. Twelve was fastest for all three workloads, which is the host count, so the default stayed there.

| Matrix workers | Greedy 128, ms | Prefill 128, ms | Tied head, ms |
| ---: | ---: | ---: | ---: |
| 1 | 11036.114 | 5424.326 | 36.498 |
| 2 | 10863.069 | 4360.699 | 18.688 |
| 4 | 6797.511 | 2375.533 | 10.185 |
| 6 | 5105.980 | 1664.859 | 8.913 |
| 12 | 4509.203 | 1150.675 | 6.722 |

Raw results are in [`bench/results/phase9-workers-2026-11-17/`](../bench/results/phase9-workers-2026-11-17/).

## Cloud VM, earlier revision

Intel Xeon, four logical processors, Racket CS 9.3. The tree is `3e6a0ef`. The result file records git `14746964b28ee2deb1c387745a45618207fd2fcd`, which is that same tree, from before the cache and attention changes. Do not line it up against the laptop tables.

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

Resident after load and a collection: 1,085,651,056 bytes. The writeup and the raw JSON are in [`bench/results/vm-reference-2026-11-17.md`](../bench/results/vm-reference-2026-11-17.md).

## Running it yourself

```sh
sh bench/run.sh
```

That checks the pinned files, then records load, prefill, generation, the two matrix shapes, and memory under `bench/results/`. Each workload is a fresh process, one warmup, three timed calls. Allow five minutes or more. Prefill at 1,000 tokens is the long step. Leave the machine idle, and on power if it has a battery. `sh bench/run.sh --runs 5` changes the run count.

The Windows comparisons were driven from PowerShell with `bench/phase8.ps1` and `bench/compare-phases.rkt`. How to repeat them, and why each trial was kept or dropped, is in [Speed notes](dev/speed.md).

A statistical profile of a 128-token generation on the laptop returned zero stack samples. It timed the run and did not say where the time went. The kernel timings above were used instead.
