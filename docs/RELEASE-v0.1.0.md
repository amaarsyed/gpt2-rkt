# gpt2-rkt 0.1.0

Pure-Racket inference for GPT-2 Small (124M) on Racket CS 9.3. This note
describes the tree that is ready to release. It does not create a git
tag or a GitHub release. The owner merges the stack first, then runs
`sh bench/run.sh` on the M5 Pro. The numbers below are from the cloud
VM, and the checked-in `vm-reference` result is labeled as that.

## Pinned assets

`openai-community/gpt2` at `607a30d783dfa663caf39e06633721c8d4cfcd7e`.

Digests are in `tools/gpt2-assets.sha256`. `sh tools/fetch-gpt2-vocab.sh`
checks them. `GPT2_FETCH_WEIGHTS=0` skips `model.safetensors`. The unit
CI job uses that flag. A separate CI job downloads the weights and runs
`tests/gpt2-small.rkt`.

## Verification

The plan bar is `|a−e| ≤ 1e-3 + 1e-4·|e|` against the fp32 eager
fixture. The largest stage error across the four prompts is:

| Stage | max abs vs fp32 | max abs vs float64 |
| --- | ---: | ---: |
| embed | 2.14e-7 | 0 |
| block 0 | 1.75e-4 | ~1e-14 |
| block 5 | 2.65e-4 | ~1e-14 |
| block 11 | 1.09e-3 | ~1e-13 |
| ln_f | 1.31e-4 | ~1e-13 |
| top logits | 6.94e-5 | ~1e-13 |

Block 11 at 1.09e-3 is the fp32 reference. A float64 last row of every
channel matches the Racket result at about 1e-13, inside the extra bar
`1e-6 + 1e-8·|e|`.

Greedy continuation of `Once upon a time`, 16 tokens:

`, the world was a place of great beauty and great danger. The world was`

On this VM the benchmark median is the table below. A separate 16-token
generation, which includes the prompt, was about 6.3 to 7.2 tokens per
second. Neither figure is the M5 Pro number.

## Reference benchmark

`sh bench/run.sh` records the host, the Racket version, `processor-count`,
and the git SHA. The prompt is `Once upon a time` and the seed is 1.
Decoding is greedy. One warmup is discarded. The table is the median of
three runs. Each row is a separate process, so `current-memory-use 'peak`
is that configuration's high-water mark. For a model process that peak
still includes the weight-file buffer. The resident size after a
collection is 1,085,651,056 bytes. The load time is 5.04 s.

The committed file is `bench/results/vm-reference-2026-11-17.md`, at git
`14746964b28ee2deb1c387745a45618207fd2fcd`. It is a cloud VM run (Intel
Xeon, 4 processors, Racket CS 9.3), not an M5 Pro. `hw.perflevel0.physicalcpu`
and `hw.ncpu` are unavailable here. The first 16 decoded ids match the
greedy fixture, and the run produced 128 ids.

| configuration | median | peak bytes |
| --- | --- | ---: |
| load | 5.0367 s | 2,248,894,968 |
| prefill 16 | 488.9253 ms | 2,248,844,344 |
| prefill 128 | 1837.0529 ms | 2,248,863,000 |
| prefill 512 | 8394.3052 ms | 2,248,857,768 |
| prefill 1000 | 20660.1813 ms | 2,248,871,536 |
| decode 128 | 9.3366 tok/s | 2,248,894,184 |
| matmul 768×768×768 | 14.6842 GFLOP/s | 116,141,152 |
| matmul 1×768×50257 | 6.0236 GFLOP/s | 413,302,272 |

## Known limitations

- Arithmetic is Racket flonums. There is no BLAS and no C extension.
- Decoding is greedy, or temperature with optional top-k. There is no beam search. Top-k sorts the row.
- The context is `min(1024, n_positions)`.
- A partial UTF-8 sequence is emitted as one U+FFFD per unfinished byte, not one per maximal subpart.
- The engine does not train. Dropout is off.
- `attn.bias` and `attn.masked_bias` are skipped. The causal mask is computed.
- Key and value buffers are mutated in place. A cache handle that has been advanced cannot be passed again.
- `load-model` checks names and shapes. The digest check is the fetch script and `check-file-sha256`.
- The published speed and memory numbers for this release are the owner's M5 Pro run. The VM file is only a reference.
