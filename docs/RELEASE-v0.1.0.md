# gpt2-rkt 0.1.0

Pure-Racket inference for GPT-2 Small (124M) on Racket CS 9.3. This note describes the tree as it stands: the model that matches the eager reference, and the cache and kernel work that followed it. Both are 0.1.0. `info.rkt` says the same version.

Published numbers for a machine come from `sh bench/run.sh` on that machine. The figures below are the two runs that are checked in. Neither is an Apple M5 Pro measurement. The repository does not contain one.

## Pinned assets

`openai-community/gpt2` at `607a30d783dfa663caf39e06633721c8d4cfcd7e`.

Digests are in `tools/gpt2-assets.sha256`. `sh tools/fetch-gpt2-vocab.sh` checks them. `GPT2_FETCH_WEIGHTS=0` skips `model.safetensors`. The unit CI job uses that flag. A separate CI job downloads the weights and runs `tests/gpt2-small.rkt`.

## Agreement with the reference

The fp32 tolerance is `|a - e| <= 1e-3 + 1e-4 * |e|` against the eager fixture. The largest stage error across the four prompts is:

| Stage | max abs vs fp32 | max abs vs float64 |
| --- | ---: | ---: |
| embed | 2.14e-7 | 0 |
| block 0 | 1.75e-4 | ~1e-14 |
| block 5 | 2.65e-4 | ~1e-14 |
| block 11 | 1.09e-3 | ~1e-13 |
| ln_f | 1.31e-4 | ~1e-13 |
| top logits | 6.94e-5 | ~1e-13 |

Block 11 at `1.09e-3` is the fp32 reference. A float64 last row of every channel matches the Racket result at about `1e-13`, inside the extra tolerance `1e-6 + 1e-8 * |e|`.

Greedy continuation of `Once upon a time`, 16 tokens:

```
, the world was a place of great beauty and great danger. The world was
```

## This tree, Windows laptop

AMD Ryzen 5 8645HS, 12 logical processors, Racket CS 9.3, measured at `c73c2df`. Comments in the sources changed after that commit. The engine did not. The right-hand column of the comparison in [Benchmarks](BENCHMARKS.md) is this run. Medians of three samples:

| Workload | Median |
| --- | ---: |
| Load | 3091.165 ms |
| Prefill 128 | 986.489 ms |
| Prefill 1000 | 11462.071 ms |
| Greedy, 128 tokens | 28.03 tok/s (4567.006 ms) |
| Sampled, 128 tokens | 27.24 tok/s (4698.893 ms) |

Generation includes the four-token prompt. Sampling is temperature 0.8, top-k 40, seed 1. The 28.03 tok/s rate is 128 divided by 4.567006 seconds.

## Cloud VM, earlier revision

Intel Xeon, 4 logical processors, Racket CS 9.3. This is the tree at `844d4a3`, which is before the cache and attention changes above. The result file records git `14746964b28ee2deb1c387745a45618207fd2fcd`. That commit is the same tree. `hw.perflevel0.physicalcpu` and `hw.ncpu` were unavailable on that host.

The prompt is `Once upon a time`. Decoding is greedy. One warmup is discarded. The table is the median of three runs. Each row is a separate process, so the peak is that configuration's high-water mark from `current-memory-use`. Load, prefill, and decode peaks include loading the weights, and that peak still includes the weight-file buffer. The resident size after a collection in the load measurement is 1,085,651,056 bytes. The first 16 decoded ids match the greedy fixture, and the run produced 128 ids.

| configuration | median | peak bytes |
| --- | --- | ---: |
| load | 5.0367 s | 2,248,894,968 |
| prefill 16 | 488.9253 ms | 2,248,844,344 |
| prefill 128 | 1837.0529 ms | 2,248,863,000 |
| prefill 512 | 8394.3052 ms | 2,248,857,768 |
| prefill 1000 | 20660.1813 ms | 2,248,871,536 |
| decode 128 (incl. 4-token prefill) | 9.3366 tok/s | 2,248,894,184 |
| matmul 768×768×768 | 14.6842 GFLOP/s | 116,141,152 |
| matmul 1×768×50257 | 6.0236 GFLOP/s | 413,302,272 |

The committed file is `bench/results/vm-reference-2026-11-17.md`. Do not read 9.3366 tok/s as the speed of the current sources, and do not read it as a laptop number. The two machines are not a before-and-after.

## Known limitations

- Arithmetic is Racket flonums. There is no BLAS and no C extension.
- Decoding is greedy, or temperature with an optional top-k. There is no beam search. Top-k keeps a heap of `k` ids and does not sort the row.
- The context is `min(1024, n_positions)`.
- A partial UTF-8 sequence is emitted as one U+FFFD per unfinished byte, not one per maximal subpart.
- The engine does not train. Dropout is off.
- `attn.bias` and `attn.masked_bias` are skipped. The causal mask is computed.
- Key and value buffers are mutated in place. A cache handle that has been advanced cannot be passed again.
- `load-model` checks names and shapes. The digest check is the fetch script and `check-file-sha256`.
- The speed and memory numbers in this note are the Windows laptop run of this tree and the cloud VM run of `844d4a3`. There is no M5 Pro result in the repository.
