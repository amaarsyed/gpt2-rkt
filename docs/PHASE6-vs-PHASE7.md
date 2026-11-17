# Phase 6 and phase 7

This comparison uses the merge that closed each milestone, on the same Windows
machine. It does not compare the old cloud VM result with this machine, and it
does not include the later cleanup commits in either measured engine.

| Engine | Revision | Milestone |
| --- | --- | --- |
| Phase 6 | `4e0631f4bb12ebd9e11182174de6ec808195568d` | GPT-2 Small, PR #8 |
| Phase 7 | `0f32e9b23836e07dd18c7d83c00caf84bb2c1e0b` | Benchmark and release prep, PR #9 |

## Results, 2026-11-17

Phase 7 improves 128-token generation in this run, but not every workload.
Generation takes 6.355 s at phase 6 and 5.835 s at phase 7: 20.14 versus 21.94
tokens/s, an 8.9% throughput increase. Those rates include the initial prefill.
The 1000-token prefill is 8.1% slower at phase 7.

| Configuration | Phase 6 median | Phase 7 median | Phase 7 time change |
| --- | ---: | ---: | ---: |
| Load | 3.908 s | 3.244 s | -17.0% |
| Prefill 16 | 223.684 ms | 232.051 ms | +3.7% |
| Prefill 128 | 1299.934 ms | 1336.835 ms | +2.8% |
| Prefill 512 | 6507.507 ms | 6288.714 ms | -3.4% |
| Prefill 1000 | 14299.251 ms | 15463.771 ms | +8.1% |
| Generate 128 | 6355.449 ms | 5835.142 ms | -8.2% |
| Matmul 768×768×768 | 31.930 ms | 30.538 ms | -4.4% |
| Matmul 1×768×50257 | 7.088 ms | 7.812 ms | +10.2% |

A negative time change is faster. Each row is the median of three timed calls,
not the fastest sample. All four prefill token IDs and all 128 generation IDs
match between phases; each timed call also matches its own warmup.

| Memory counter | Phase 6 | Phase 7 |
| --- | ---: | ---: |
| Resident after load and collection | 1,092,109,688 bytes | 1,092,135,064 bytes |
| Generation process peak, including load | 2,255,331,536 bytes | 2,255,368,432 bytes |

Resident size and process peak are essentially unchanged. The peak includes the
large loading allocation, so this counter does not establish how much transient
allocation phase 7 saves during attention.

Loading varied from 3.428 to 4.700 s in phase 6 and 3.065 to 3.338 s in phase 7.
The row-shaped matrix samples also overlap: 6.724–7.705 ms versus 6.753–7.962 ms.
Do not treat the median changes as statistically established speedups or assign
them to one code change. A follow-up that alternates phases would help separate
revision effects from machine conditions.

The raw results are in
[`phase6-windows-2026-11-17.json`](../bench/results/phase6-vs-phase7-windows-2026-11-17/phase6-windows-2026-11-17.json)
and
[`phase7-windows-2026-11-17.json`](../bench/results/phase6-vs-phase7-windows-2026-11-17/phase7-windows-2026-11-17.json).
The [generated comparison](../bench/results/phase6-vs-phase7-windows-2026-11-17/phase6-vs-phase7-2026-11-17.md)
includes a peak-memory column for every configuration.

## Method

- AMD Ryzen 5 8645HS, Windows x86-64, 12 processors reported to Racket.
- Racket CS 9.3. CPU execution; no GPU or native math library.
- The same pinned GPT-2 Small files, verified against `tools/gpt2-assets.sha256`.
  The weight-file SHA-256 is
  `248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707`.
- A common driver, `bench/compare-phases.rkt` at `f8e414d`, calls each snapshot's
  original loader, tokenizer, generator, and matrix functions. Neither snapshot
  is patched or given the other phase's inference code.
- One discarded warmup, then three timed calls per configuration. The median is
  reported. A fresh child process is used for each configuration.
- Phase 6 runs first, then phase 7. The timed samples are retained in the JSON
  files. This is one sequential comparison, not an alternating statistical trial.

Prefill repeats the token IDs of `Once upon a time` to exactly 16, 128, 512, or
1000 tokens, then calls the generator for one new token. These times include
cache allocation and the output head. They are not attention-only timings.

Decode starts with that four-token prompt and generates 128 tokens. The time
includes prompt prefill and cache allocation, but excludes model loading,
tokenization, and terminal output. The driver uses both phases' original EOS
behavior, rejects an early stop, and checks that every timed result agrees with
the warmup and with the other phase's token IDs.

The matrix cases are the biased product `x * W + b`, with ones for `x` and `W`
and zeros for `b`, at `768×768×768` and `1×768×50257`. The second is a row-shaped
Conv1D kernel workload, not the tied output-head function `matmul-transposed-b`.

Load time is measured around `load-model`. Resident bytes are read after a full
collection with the loaded model still live; that collection is outside the
load timer. Peak bytes are `current-memory-use 'peak` for the entire child,
including loading. Neither counter is an OS working-set or physical-RAM reading.

## What changed in phase 7

Phase 6 already preallocated key/value buffers and used parallel matrix kernels.
Phase 7 avoids copying the filled key and value columns into a fresh tensor for
each attention head. It adds matrix kernels that read those columns by offset,
and applies the causal mask in place. It also removes the legacy append-cache
path and makes an advanced cache handle unusable.

The generation benchmark still uses the full position-table cache in both
phases. Phase 7's smaller cache for `next-token-logits` is not exercised here.
The hashing change is also outside the timed sections: the common driver checks
asset digests once before either phase starts.

A phase-to-phase difference measures the complete set of changes, not one
optimization in isolation. Matrix results and timing spread should be read
alongside generation results. Background activity, power mode, temperature, and
clock speed were not controlled or recorded.

## Reproduce

Run from the repository root in PowerShell. The snapshots and ZIP files stay
under the ignored `models/` directory. The benchmark does not switch branches.

```powershell
$racketExe = 'C:\Program Files\Racket\Racket.exe'
$env:GPT2_FETCH_WEIGHTS = '1'
& 'C:\Program Files\Git\usr\bin\sh.exe' tools/fetch-gpt2-vocab.sh

New-Item -ItemType Directory -Force models/phase-bench | Out-Null
git archive --format=zip --output=models/phase-bench/phase6.zip 4e0631f
git archive --format=zip --output=models/phase-bench/phase7.zip 0f32e9b
Expand-Archive models/phase-bench/phase6.zip models/phase-bench/phase6
Expand-Archive models/phase-bench/phase7.zip models/phase-bench/phase7

& $racketExe bench/compare-phases.rkt `
  --out-dir bench/results/my-phase-comparison `
  --date 2026-11-17 `
  --cpu 'AMD Ryzen 5 8645HS'
```

Use a new output directory for another run. The driver refuses to overwrite
existing measurements. Replace the date and CPU label for another host or day.
Each phase's summary JSON includes all samples, output IDs, PIDs, source SHA,
driver SHA, and runtime metadata. Separate row JSON files preserve completed
configurations if a long run is interrupted.

For a quick compatibility check, add `--smoke --model-dir tests/tiny-gpt2` and
use another output directory. That uses tiny weights and is not a GPT-2 speed
measurement. Run the driver's checks with `raco test bench/compare-phases.rkt`.

Remove the two extracted snapshot directories when finished. They are temporary
inputs, not worktrees. Keeping historical Racket sources under `models/` would
make a later `raco test .` discover those copies too. The original ZIPs can stay
cached, and the committed reports do not depend on the snapshots remaining.
