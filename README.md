# gpt2-rkt

Pure-Racket inference for GPT-2 Small (124M). Numeric work uses `flvector` and `racket/flonum`. On more than one processor, a matrix multiply of at least 100000 flops is split across rows when it has 32 or more rows, and across output columns otherwise. That still covers a single-token logits multiply. A smaller multiply, including the per-head attention products, stays in one thread. Each split uses four times as many blocks as `processor-count`. Model weights are not in git.

This is the 0.1.0 line. `generate` parses a request and checks the model directory. Weight loading is not implemented yet, so a missing directory is a missing-assets error, and a directory that already has the expected files stops with "model loading is not implemented".

Tested with Racket CS 9.3. CI installs that version and runs `raco test`. Python is not required.

## Commands

Fetch the BPE files (about 1.5 MB; weights stay out of the tree), then run the tests:

```sh
sh tools/fetch-gpt2-vocab.sh
raco test .
```

Show CLI help:

```sh
racket main.rkt --help
racket main.rkt generate --help
```

Generate. This exits non-zero until model assets are installed:

```sh
racket main.rkt generate \
  --model-dir models/gpt2 \
  --prompt "Once upon a time" \
  --max-new-tokens 16
```

Decoding is greedy unless `--temperature` is greater than 0. `--top-k` and `--seed` require sampling. A flag's value is the next argument, even when that text starts with `-`. `-h` and `--help` are help only in flag position. Range checks live on `generate-config`; the parser only decodes arguments.

```sh
racket main.rkt generate \
  --model-dir models/gpt2 \
  --prompt "Once upon a time" \
  --max-new-tokens 16 \
  --temperature 0.8 \
  --top-k 40 \
  --seed 1
```

Benchmark harness. This prints the Racket version, VM, and machine, then the reports. The prompt report is still a placeholder `flvector` dot product: load time, first-token latency, steady-state tokens per second, and `(current-memory-use 'peak)` for a short prompt length (16) and a longer one (128). Each length runs in its own process, because `'peak` is a process-lifetime high-water mark. The matmul report warms up, then prints the median of five runs for a 768×768 Conv1D multiply and for a 1×768 · 768×50257 logits multiply: milliseconds, GFLOP/s, bytes allocated, and `vector-set-performance-stats!` counters:

```sh
racket bench/harness.rkt
```

## Numeric kernels

`src/tensor.rkt` is a row-major tensor on one `flvector`, with explicit dimensions. `src/nn.rkt` is the GPT-2 arithmetic:

- `matmul` is Hugging Face Conv1D: `y[n, out] = x[n, in] · W[in, out] + b[out]`. The inner loop is i-k-j so each step reads a contiguous weight row. A product of at least 100000 flops with thirty-two or more rows is split across rows. A shorter product of that size is split across columns. Smaller products stay serial. Both splits use four blocks per core.
- `matmul-transposed-b` is `h[n, in] · Wᵀ` with `W` stored as `[out, in]`, for tied-embedding logits. It does not allocate a transpose.
- `tensor-add` is the residual add.
- `softmax` is row-wise on the last axis. It subtracts the row max, and an all-masked row (every value `-inf.0`) is zeros.
- `layer-norm` uses biased variance (divide by `n`), epsilon `1e-5`, and a gain and bias.
- `gelu` is the GPT-2 tanh approximation.
- `slice-rows`, `slice-cols`, `split-last`, and `tensor-reshape` cut and regroup axes for later `q`, `k`, `v`, and heads. Slices copy. Reshape shares the `flvector`.

## Causal attention

`gpt2-attention` is one GPT-2 block's attention. `c_attn` is a Conv1D to `[n, 3*d]`, split on the last axis into `q`, `k`, and `v`. Head `h` is columns `[h*d-head, (h+1)*d-head)`, not an interleaved stride. Scores are dotted per head, scaled by `1/sqrt(d-head)` (so `1/8` when the head is 64), and a key strictly after the query is `-inf.0` before the stable softmax. The heads are merged and passed through `c_proj`.

`k` and `v` may be longer than `q`. Query row `i` is at position `rows(k) - rows(q) + i`, which is the same alignment as a past-key prefix: the new queries see the cached keys and not a later query in the same chunk. Milestone 5 can append key and value rows and call `attend` unchanged. That is the cache seam; the cache itself is not stored here.

## Tokenizer and embeddings

`encode` and `decode` are GPT-2 byte-level BPE. The byte map is reversible. The pre-tokenizer uses Unicode whitespace, not the ASCII-only class, and contractions are case-sensitive. `<|endoftext|>` is token 50256. Matching that literal in a prompt is tiktoken and HF behaviour, so the text `<|endoftext|>` encodes as 50256. `token-embed` and `position-embed` look up rows of `wte` and `wpe`. A sequence longer than 1024 positions is an error.

The vocab and merges are pinned to `openai-community/gpt2` at `607a30d783dfa663caf39e06633721c8d4cfcd7e`. Their sha256 digests live in `tools/gpt2-assets.sha256`. `tools/fetch-gpt2-vocab.sh` checks them with `sha256sum`, or `shasum -a 256` when that is missing, and caches the files under `models/gpt2/`.

## Layout

- `main.rkt` — CLI entry and public API
- `src/` — implementation modules
- `tests/` — RackUnit tests
- `bench/` — benchmark harness
- `tools/fixtures/` — Python fixture generator only, not used by tests or CI
- `docs/BUGLOG.md` — bug log

## Model directory

`generate` expects these files in `--model-dir`:

- `config.json`
- `vocab.json`
- `merges.txt`
- `model.safetensors`

A download script with a sha256 check will land in a later milestone. Do not commit weights.
