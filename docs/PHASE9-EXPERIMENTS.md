# Phase 9 experiments

Screening on Windows, Racket CS 9.3, Ryzen 5 8645HS, 12 logical processors.
The baseline is phase 8 at `fda38fb`. Kernel screening uses one warmup and seven
timed calls. These are sequential trials; final validation alternates trees.

| Trial | Reason | Observation | Decision | Commits and evidence |
| --- | --- | --- | --- | --- |
| Unroll four inner-loop elements in order | Reduce loop overhead without changing the sum order | Head median rose from 6.755 to 17.832 ms in the kernel trial. A later allocation check measured 14.611 ms and about 156 MB allocated per call | Reject. Correct output was not enough to justify the regression | Trial `b0bccd6`, restored in `a23a911`; [kernels](../bench/results/phase9-experiments-2026-11-17/ordered-unroll-kernels.json), [allocation check](../bench/results/phase9-experiments-2026-11-17/ordered-unroll-head.json) |
| Share each input load among four output rows | Read the input once per four products, with four separate ordered sums | Head median was 7.427 ms, versus the 6.755 ms screening baseline | Reject. No measured gain justified the extra kernel | Trial `8ea9f9e`, removed in `9f50897`; [samples](../bench/results/phase9-experiments-2026-11-17/four-output-kernels.json) |
| Force inline expansion of the scalar dot | Remove the procedure boundary while retaining the original loop | Head median was 8.281 ms. Allocation stayed below roughly 1.1 MB per call, but a 34.244 ms outlier also appeared | Reject. No reliable gain in this screening run | Trial `4974769`, removed in `82eab45`; [samples](../bench/results/phase9-experiments-2026-11-17/inline-head.json) |
| Put the unrolled loop and scalar tail in separate loops | Test whether the original unroll's tail/control flow caused its allocation overhead | Head median was 15.463 ms and allocation remained about 156 MB per call | Reject. Separating the tail did not remove the measured allocation | Trial `5ef4d6b`, removed in `d8f1bab`; [samples](../bench/results/phase9-experiments-2026-11-17/split-tail-head.json) |
| Sweep 1, 2, 4, 6, and 12 matrix workers | Find separate decode and prefill budgets instead of assuming all logical processors are best | Twelve was fastest for all three workloads below | Keep the default host count. Retain optional NN-module budgets so the experiment can be repeated elsewhere | Frozen trial tree `5456731`; [raw samples and summary](../bench/results/phase9-workers-2026-11-17/summary.json) |
| Read packed queries and copy K/V straight to cache | Avoid splitting the projection into three temporary tensors, then copying two again | Thirty-eight targeted tests passed, including packed query stride, cached-prefix equality, unused cache tails, and interleaved requests | Keep for final measurement against phase 8 | `06f0942`; `tests/attention.rkt`, `tests/block.rkt`, `tests/model.rkt` |

The scalar [baseline kernel samples](../bench/results/phase9-experiments-2026-11-17/baseline-kernels.json)
are retained. All rejected source variants are committed and can be restored
for reproduction. Their removal commits do not erase the trial history.

The unrolled variants' extra allocation is measured, not inferred from timing.
It is consistent with lost flonum unboxing, but no compiler analysis established
the exact cause. The four-output and inlining trials were screening decisions;
their small timing differences do not establish a general slowdown on all hosts.

## Worker sweep

Each setting ran in its own process, with one warmup and three timed calls.
Settings ran in ascending order, so this is screening rather than an alternating
causal comparison. Only matrix products use the budget; attention heads have
their own scheduling. The head workload is the actual transposed tied product.

| Matrix workers | Greedy 128 tokens, ms | Prefill 128 tokens, ms | Tied head, ms |
| ---: | ---: | ---: | ---: |
| 1 | 11036.114 | 5424.326 | 36.498 |
| 2 | 10863.069 | 4360.699 | 18.688 |
| 4 | 6797.511 | 2375.533 | 10.185 |
| 6 | 5105.980 | 1664.859 | 8.913 |
| 12 | 4509.203 | 1150.675 | 6.722 |

The budgets are optional `current-decode-workers` and `current-prefill-workers`
parameters in `src/nn.rkt`. `#f` uses the host count. Positive budgets are clamped
to that count. They do not change arithmetic or the generation configuration.

The missing test import and benchmark-label collision are recorded in
[BUGLOG.md](BUGLOG.md), with their pre-fix commits. The tiny-model smoke run
checked three baseline and three phase 9 samples for each configuration before
the full comparison started. It is not a GPT-2 performance result.
