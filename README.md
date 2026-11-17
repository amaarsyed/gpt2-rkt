# gpt2-rkt

Pure-Racket inference for GPT-2 Small (124M). Numeric work uses `flvector` and `racket/flonum`. On more than one processor, a matrix multiply of at least 100000 flops is split across rows when it has 32 or more rows, and across output columns otherwise. That still covers a single-token logits multiply. A smaller multiply, including the per-head attention products, stays in one thread. Each split uses four times as many blocks as `processor-count`. Model weights are not in git.

This is the 0.1.0 line. `generate` loads a GPT-2 directory, encodes the prompt, and writes each decoded token as it is chosen. A missing file names the asset and `sh tools/fetch-gpt2-vocab.sh`. The fetch script and the digest check reject a corrupt file; `load-model` checks shapes and keys, not the digest. `model.safetensors` is little-endian F32, read in place from the header offsets. Names may be bare (`wte.weight`) or prefixed with `transformer.`. The `attn.bias` and `attn.masked_bias` buffers are ignored. A `lm_head.weight` that matches `wte` is accepted and unused. The tiny model uses the same reader.

Tested with Racket CS 9.3. CI installs that version and runs `raco test`. Python is not required.

## Commands

Fetch the pinned tokenizer, `config.json`, and `model.safetensors` (the weights stay out of git), then run the tests:

```sh
sh tools/fetch-gpt2-vocab.sh
raco test .
```

Show CLI help:

```sh
racket main.rkt --help
racket main.rkt generate --help
```

Generate from a model directory:

```sh
racket main.rkt generate \
  --model-dir models/gpt2 \
  --prompt "Once upon a time" \
  --max-new-tokens 16
```

Decoding is greedy unless `--temperature` is greater than 0. A temperature of 0 or less is an error. `--top-k` and `--seed` require sampling. A flag's value is the next argument, even when that text starts with `-`. `-h` and `--help` are help only in flag position. Range checks live on `generate-config`; the parser only decodes arguments.

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

`k` and `v` may be longer than `q`. Query row `i` is at position `rows(k) - rows(q) + i`, which is the same alignment as a past-key prefix: the new queries see the cached keys and not a later query in the same chunk. `attend` takes `#:kv-rows` so a buffer allocated to `n_positions` is not treated as a full sequence. The model writes each new key and value row into that buffer. A head reads its columns from the filled prefix with a column offset, so the prefix is not copied on every token.

## Transformer block

`gpt2-block` is pre-LN. Let `h = x + attn(ln_1(x))`. The block output is `h + mlp(ln_2(h))`. Both norms use biased variance and epsilon `1e-5`. The MLP is `c_fc` from the model width to the inner width, the tanh GELU, and `c_proj` back. For GPT-2 small those matrices are `[768, 3072]` and `[3072, 768]`. `gpt2-blocks` folds a list of blocks. The final norm is the same `layer-norm`.

## Inference

`load-model` reads `config.json`, the tokenizer files, and little-endian F32 `model.safetensors`. It checks every tensor shape, names a missing or unexpected key, and records the load time and the memory after a collection. `tests/tiny-gpt2` is a 2-layer eager model in that format. `racket main.rkt generate --model-dir tests/tiny-gpt2 --prompt hi --max-new-tokens 4` prints its greedy continuation. `next-token-logits` and generation score only the last hidden row, and both use the cached forward pass. `model-generate` scores the prompt once, then one new token at a time. Generation allocates each layer's key and value buffer to `n_positions`. `next-token-logits` allocates only the prompt length. The buffers are updated in place, and a cache handle that has already been advanced cannot be passed again. It stops at `eos_token_id` (50256 when the config omits it) or at the requested count. The id list includes that eos token; the printed text does not. A UTF-8 character split across tokens stays buffered until the next token completes it, or until generation finishes, and the finished piece is written as soon as the token is chosen. Omit temperature for greedy decoding. A temperature of 0 or less is an error. A positive temperature subtracts the row max before dividing. `top-k` keeps the largest scores before the softmax. A seed is an integer from 0 through 2147483647 and uses a private generator. The prompt plus the new tokens must fit in `min(1024, n_positions)`. `position-embed` takes `#:start` so a cached token uses its absolute position. After the text, `generate` writes `load-seconds`, `load-bytes`, and `tokens-per-second` to the error port.

## Tokenizer and embeddings

`encode` and `decode` are GPT-2 byte-level BPE. The byte map is reversible. The pre-tokenizer uses Unicode whitespace, not the ASCII-only class, and contractions are case-sensitive. `<|endoftext|>` is token 50256. Matching that literal in a prompt is tiktoken and HF behaviour, so the text `<|endoftext|>` encodes as 50256. `token-embed` and `position-embed` look up rows of `wte` and `wpe`. A sequence longer than 1024 positions is an error.

The vocab, merges, `config.json`, and `model.safetensors` are pinned to `openai-community/gpt2` at `607a30d783dfa663caf39e06633721c8d4cfcd7e`. Their sha256 digests live in `tools/gpt2-assets.sha256`. `tools/fetch-gpt2-vocab.sh` checks every name in that file with `sha256sum`, or `shasum -a 256` when that is missing, and caches the files under `models/gpt2/`.

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

`sh tools/fetch-gpt2-vocab.sh` downloads them and checks the digests. A missing directory, a missing file, or a bad digest names the file and that command. Do not commit weights.
