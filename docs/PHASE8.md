# Phase 8: faster inference

The baseline is `1ad7035`, the tree after cleanup and the phase 6/7 comparison.
The measured candidate is `e8af302`. Both use the same pinned GPT-2 Small files
on Windows, Racket CS 9.3, AMD Ryzen 5 8645HS, with 12 reported processors.

## Changes

- Matrix products use one future block per processor instead of four.
- Attention computes only visible causal entries. Large prefills run heads in
  parallel; decode stays serial. There is no dense square score matrix.
- Each generation owns its score workspace and reuses it across queries and
  layers. Key/value buffers fit the prompt plus the requested new tokens.
- GELU and residual additions reuse the product buffer they just allocated.
  Inputs and returned traces are not scratch storage.
- Top-k uses a bounded heap and retains the same lower-ID tie ordering.

The model architecture, public entry points, weight format, arithmetic
precision, and reference tolerances stay the same. No native math backend is
required.

## Measurement

`bench/phase8.ps1` alternates baseline/candidate, candidate/baseline, then
baseline/candidate for each configuration. Every sample has a fresh process,
one warmup, and one timed call. The table reports the median of three samples.
This reduces ordering bias but does not control power, temperature, or other
applications on the machine.

Generation includes the four-token prompt prefill. Sampling uses temperature
0.8, top-k 40, and seed 1. The driver checks all generated IDs across runs and
trees. The head case measures `matmul-transposed-b`, not the biased Conv1D
kernel. Timed allocation counts exclude model loading. Process peaks include
loading and are not OS working-set readings.

The statistical profiler returned zero stack samples in this environment. It
does not establish where CPU time went. Targeted projection/head measurements
and end-to-end timings were used instead. Register-heavy and four-row matrix
kernel trials were discarded: their measured prefill results did not justify
keeping more code.

The [experiment log](PHASE8-EXPERIMENTS.md) records the hypotheses, raw screening
measurements, rejected trials, and reasons for each decision. Implementation
and test failures are recorded in [BUGLOG.md](BUGLOG.md).

## Reproduce

From the repository root in PowerShell, with pinned assets already fetched:

```powershell
git archive --format=zip --output=models/phase8-baseline.zip 1ad7035
Expand-Archive models/phase8-baseline.zip models/phase8-baseline
./bench/phase8.ps1 -OutDir bench/results/my-phase8-run
```

Use a fresh output directory. Raw samples are appended to `samples.jsonl` as
they finish. The summary and comparison are written only after every token-ID
check succeeds. The script labels this Windows host and date explicitly;
update those labels for a different host or day.

Remove the extracted baseline after the run. It is an ignored snapshot, not a
worktree. `bench/kernels.rkt --source PATH` separately measures the selected
tree's projection, MLP, square, and tied-head kernels.
