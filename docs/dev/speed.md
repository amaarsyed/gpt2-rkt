# Speed notes

These are the measurements behind [Benchmarks](../BENCHMARKS.md), taken on 2026-11-17. Unless a section says otherwise, the machine is a Windows laptop: AMD Ryzen 5 8645HS, 12 logical processors, Racket CS 9.3, CPU only.

The order below is the order the runs were taken. Each one compares two trees of this repository. The current sources are the last of them, plus comments.

## Reading the cache in place

`1fcbb84` already preallocated the key and value buffers and used parallel matrix kernels. `7261f14` stops copying each head's filled prefix into a fresh tensor. It reads those columns by offset and applies the causal mask in place. An advanced cache handle becomes unusable, so a second call cannot overwrite the rows.

The two trees were run one after the other, not alternated. Generation of 128 tokens went from 6355.449 ms to 5835.142 ms, which is 20.14 tok/s to 21.94, including the prompt prefill. The 1000-token prefill went the other way, 14299.251 ms to 15463.771 ms. Resident memory after load and a collection was 1,092,109,688 bytes, then 1,092,135,064. The generation peak, including load, was 2,255,331,536 bytes, then 2,255,368,432. That peak is dominated by loading, so it does not show whether the copy removal saved transient allocation.

A common driver, `bench/compare-phases.rkt` at `cd77acb`, calls each snapshot's own loader, tokenizer, generator, and matrix functions. Prefill repeats the token ids of `Once upon a time` out to 16, 128, 512, or 1000 tokens and then asks for one new token. Decode starts from the four-token prompt and generates 128 tokens. The timed section includes prompt prefill and cache allocation. It excludes model loading, tokenization, and terminal output. The matrix cases are `x * W + b` with ones for `x` and `W` and zeros for `b`. The `1×768×50257` case is that biased kernel, not `matmul-transposed-b`.

To repeat it, from PowerShell at the repository root, with the pinned weights fetched:

```powershell
$racketExe = 'C:\Program Files\Racket\Racket.exe'
New-Item -ItemType Directory -Force models/phase-bench | Out-Null
git archive --format=zip --output=models/phase-bench/a.zip 1fcbb84
git archive --format=zip --output=models/phase-bench/b.zip 7261f14
Expand-Archive models/phase-bench/a.zip models/phase-bench/a
Expand-Archive models/phase-bench/b.zip models/phase-bench/b
& $racketExe bench/compare-phases.rkt `
  --out-dir bench/results/my-comparison `
  --date 2026-11-17 `
  --cpu 'AMD Ryzen 5 8645HS'
```

Use a new output directory. The driver refuses to overwrite. Point `$racketExe` at your own install. Delete the extracted snapshots when you are done. A later `raco test .` will pick up Racket sources left under `models/`.

## One future per processor, and a smaller cache

The baseline is `7d507ff`. The candidate is `0d8bcaf`. Five changes went in together, so the comparison does not assign a percentage to each one:

- Matrix products use one future block per processor instead of four.
- Attention scores only keys the query can see, and keeps one reusable score row per head. Prefill can run heads in parallel. Decode stays serial.
- The generation cache fits the prompt plus the requested new tokens.
- GELU and the residual add reuse the product buffer they just allocated. Inputs and returned traces are not scratch.
- Top-k uses a bounded heap. Ties still keep the lower id.

The alternating run is the table in Benchmarks. Greedy generation went from 21.92 tok/s to 27.93, sampled from 18.76 to 26.96, and the 1000-token prefill from 15183.579 ms to 10928.493 ms. The 16-token prefill went from 219.288 ms to 228.170 ms. The square matmul was unchanged for any practical purpose. Greedy allocation fell from 1,793,805,440 bytes to 625,925,216.

Before that run, a few kernels were screened and two were thrown out. Register accumulators, meant to keep weights in locals, made the `128×768×3072` product slower: 23.065 ms to 49.522 ms. A four-row variant that reused each weight across rows landed at 28.646 ms, against 22.349 ms for the scheduling change alone. Both were discarded. The one-block scheduler stayed, on the strength of a generation trial at 27.39 tok/s against 21.77 in an earlier profile run, and then on the alternating comparison. A single 1000-token prefill trial took 11.161 s against a 15.464 s median from the previous revision on this machine. That trial is why the visible-key attention was kept long enough to measure properly. It is not itself the result.

The statistical profiler returned zero stack samples. It does not say which function was hot.

`bench/phase8.ps1` alternates the trees. From PowerShell, with the weights fetched:

```powershell
git archive --format=zip --output=models/phase8-baseline.zip 7d507ff
Expand-Archive models/phase8-baseline.zip models/phase8-baseline
./bench/phase8.ps1 -OutDir bench/results/my-run
```

Use a fresh output directory. The summary is written only after every token-id check succeeds. Remove the extracted baseline afterwards. The script writes `AMD Ryzen 5 8645HS` and `2026-11-17` into that summary as fixed strings, so change them in the script if the machine or the day is different. `bench/kernels.rkt --source PATH` times one tree's projection, MLP, square, and tied-head kernels on their own.

## Packed queries, and the head kernels that lost

`9c7411c` is the baseline. `c73c2df` is the candidate, and it is the engine in this tree. The cached block reads queries from the packed projection and writes keys and values straight into the cache, instead of allocating three tensors and copying two of them again. The uncached path still splits.

The alternating comparison does not show a meaningful speedup. Greedy went from 27.85 tok/s to 28.03, sampled from 27.20 to 27.24. Allocation did drop: greedy generation from 626,025,536 bytes to 597,274,112, and the 1000-token prefill from 1,551,391,120 to 1,333,450,608. One baseline head sample took 35.005 ms while the other two took 7.283 and 7.467 ms, which is another reason not to treat a few tenths of a millisecond as a result. All 54 samples finished with matching token ids. The operating system reused one process id, so the log has 53 distinct pids. The driver still starts and waits for a separate child every time.

Four attempts to speed up the tied head were measured and removed:

| Attempt | What happened | Median |
| --- | --- | ---: |
| Unroll four inner-loop elements, same sum order | Head time and allocation both rose. The allocation check was about 156 MB per call. | 17.832 ms in the kernel screen, 14.611 ms in the follow-up |
| Share each input across four output rows | No gain against the 6.755 ms scalar screen. | 7.427 ms |
| Force the scalar dot to inline | No reliable gain. One sample was 34.244 ms. Allocation stayed under about 1.1 MB. | 8.281 ms |
| Split the unrolled tail into its own loop | The allocation stayed at about 156 MB. Separating the tail did not fix it. | 15.463 ms |

The scalar loop is the one that stayed. The extra allocation is consistent with the flonum unboxing being lost. That cause was not confirmed from the compiler.

A worker sweep at `9cc1371` tried 1, 2, 4, 6, and 12 matrix workers. Twelve, the host count, was fastest for greedy generation (4509.203 ms), a 128-token prefill (1150.675 ms), and the tied head (6.722 ms). The default stayed "use every processor." `current-decode-workers` and `current-prefill-workers` remain so the sweep can be repeated on another machine. They cap matrix products only.

Repeat the comparison with the same driver, pointed at this pair:

```powershell
git archive --format=zip --output=models/baseline.zip 9c7411c
Expand-Archive models/baseline.zip models/baseline
./bench/phase8.ps1 -Phase later -BaselineRef 9c7411c `
  -BaselineSource models/baseline -OutDir bench/results/my-run
```

The script file is `bench/phase8.ps1`. `-Phase` is only the label it writes for the candidate, and its default baseline is `7d507ff`, so pass both flags for this pair. `-Smoke` runs the tiny model as a compatibility check, not a speed result. The worker sweep is `bench/sweep-workers.ps1` against a snapshot of `9cc1371`.

## What was not tried

No packed-weight format, no reduced precision, no native backend, and no persistent worker pool. The runs did not make a case for those. Buffer reuse stayed where the ownership is obvious: the request's score rows, and a product buffer that the next pass consumes immediately.
