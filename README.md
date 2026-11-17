# gpt2-rkt

Pure-Racket inference for GPT-2 Small (124M). Numeric work uses `flvector` and `racket/flonum`. Large matrix products split across rows or output columns, with one block per processor. Attention computes only the visible causal prefix. Prefill can run heads in parallel; a decode query stays serial. Model weights are not in git.

This is the 0.1.0 line. `generate` loads a GPT-2 directory, encodes the prompt, and writes each decoded token as it is chosen. A missing file names the asset and `sh tools/fetch-gpt2-vocab.sh`. The fetch script and the digest check reject a corrupt file; `load-model` checks shapes and keys, not the digest. `model.safetensors` is little-endian F32, read in place from the header offsets. Names may be bare (`wte.weight`) or prefixed with `transformer.`. The `attn.bias` and `attn.masked_bias` buffers are ignored. A `lm_head.weight` that matches `wte` is accepted and unused. The tiny model uses the same reader.

Tested with Racket CS 9.3. CI installs that version and runs `raco test`. Python is not required. The version of this tree is 0.1.0.

## Quickstart

Fetch the pinned files, run the tests, then generate. The weights stay out of git.

```sh
sh tools/fetch-gpt2-vocab.sh
raco test .
racket main.rkt generate \
  --model-dir models/gpt2 \
  --prompt "Once upon a time" \
  --max-new-tokens 16
```

The same call from Racket. `#f` leaves temperature, top-k, and seed unset, which is greedy:

```racket
#lang racket/base
(require "main.rkt")

(define result
  (generate (generate-config "models/gpt2" "Once upon a time" 16 #f #f #f)))
(displayln (generation-text result))
```

On the M5 Pro, install the full Racket CS 9.3 and put `racket` and `raco` on PATH. `raco test` comes from the full distribution; a minimal Racket answers `raco test .` with `Unrecognized command: test`. Then:

```sh
sh bench/run.sh
```

That checks the pinned files first, then writes `bench/results/<machine>-<date>.json` and a markdown table. Expect about 5 minutes or more. Prefill at 1000 tokens is the longest step. Keep the machine plugged in and idle, and commit both generated files. Do not pass `--not-m5-pro`. The checked-in `vm-reference` file is a cloud VM run, not an M5 Pro.

## Commands

Fetch the pinned tokenizer, `config.json`, and `model.safetensors` (the weights stay out of git), then run the tests. `GPT2_FETCH_WEIGHTS=0` skips the weight file. CI's unit job does that, and a separate job downloads the weights and runs the real-model test:

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

`sh bench/run.sh` is the model benchmark: load time, prefill at 16, 128, 512, and 1000 tokens, greedy decode of 128 tokens, and the 768² and 1×768·768×50257 matmuls. Each configuration is its own process. It warms up once and reports the median of `--runs` (default 3). `racket bench/harness.rkt` is the CI smoke: a placeholder dot product at lengths 16 and 128, plus those two matmuls as a median of five runs.

## Numeric kernels

The [phase 6 and phase 7 comparison](docs/PHASE6-vs-PHASE7.md) measures the two
milestone merges on the same Windows machine with one shared driver. It records
load, prefill, 128-token generation, matrix kernels, and memory, with raw samples
and matching greedy token IDs. These results are separate from the VM reference.

[Phase 8](docs/PHASE8.md) records the speed changes and an alternating comparison
against the tree before those changes, including timed allocation counts.

[Phase 9](docs/PHASE9.md) removes cached-path Q/K/V intermediates and records the
head-kernel experiments and worker sweep. Its worker defaults remain unchanged.

`src/tensor.rkt` is a row-major tensor on one `flvector`, with explicit dimensions. `src/nn.rkt` is the GPT-2 arithmetic:

- `matmul` is Hugging Face Conv1D: `y[n, out] = x[n, in] · W[in, out] + b[out]`. The inner loop is i-k-j so each step reads a contiguous weight row. A product of at least 100000 flops with thirty-two or more rows is split across rows. A shorter product of that size is split across columns. Smaller products stay serial. Both splits use one block per processor.
- `matmul-transposed-b` is `h[n, in] · Wᵀ` with `W` stored as `[out, in]`, for tied-embedding logits. It does not allocate a transpose.
- `tensor-add` is the residual add.
- `softmax` is row-wise on the last axis. It subtracts the row max, and an all-masked row (every value `-inf.0`) is zeros.
- `layer-norm` uses biased variance (divide by `n`), epsilon `1e-5`, and a gain and bias.
- `gelu` is the GPT-2 tanh approximation.
- `slice-rows`, `slice-cols`, `split-last`, and `tensor-reshape` cut and regroup axes for later `q`, `k`, `v`, and heads. Slices copy. Reshape shares the `flvector`.

## Causal attention

`gpt2-attention` is one GPT-2 block's attention. `c_attn` is a Conv1D to `[n, 3*d]`, split on the last axis into `q`, `k`, and `v`. Head `h` is columns `[h*d-head, (h+1)*d-head)`, not an interleaved stride. Scores are dotted per head, scaled by `1/sqrt(d-head)` (so `1/8` when the head is 64), and a key strictly after the query is `-inf.0` before the stable softmax. The heads are merged and passed through `c_proj`.

`k` and `v` may be longer than `q`. Query row `i` is at position `rows(k) - rows(q) + i`, which is the same alignment as a past-key prefix: the new queries see the cached keys and not a later query in the same chunk. `attend` takes `#:kv-rows` so a buffer allocated to `n_positions` is not treated as a full sequence. The model writes each new key and value row into that buffer. A head reads its columns from the filled prefix with a column offset, so the prefix is not copied on every token.

Each query computes and normalizes only its visible scores. A request-owned
workspace reuses one score row per head across queries and layers. Large
prefills give each head a future, with disjoint scratch and output columns.
GELU and residual additions reuse fresh product buffers; caller inputs and
returned trace tensors remain independent.

The cached block reads queries from the leading columns of the packed Q/K/V
projection and writes keys and values directly into the cache. `attend` uses
`#:query-width` to distinguish that logical width from the stored row stride.

## Transformer block

`gpt2-block` is pre-LN. Let `h = x + attn(ln_1(x))`. The block output is `h + mlp(ln_2(h))`. Both norms use biased variance and epsilon `1e-5`. The MLP is `c_fc` from the model width to the inner width, the tanh GELU, and `c_proj` back. For GPT-2 small those matrices are `[768, 3072]` and `[3072, 768]`. `gpt2-blocks` folds a list of blocks. The final norm is the same `layer-norm`.

## Inference

`load-model` reads `config.json`, the tokenizer files, and little-endian F32 `model.safetensors`. It checks every tensor shape, names a missing or unexpected key, and records the load time and `current-memory-use` without collecting. The benchmark collects before it records the resident size. `tests/tiny-gpt2` is a 2-layer eager model in that format. `racket main.rkt generate --model-dir tests/tiny-gpt2 --prompt hi --max-new-tokens 4` prints its greedy continuation. `next-token-logits` and generation score only the last hidden row, and both use the cached forward pass. `model-generate` scores the prompt once, then one new token at a time. Generation sizes each layer's key and value buffers to the prompt plus the requested new tokens. `next-token-logits` allocates only the prompt length. The buffers are updated in place, and a cache handle that has already been advanced cannot be passed again. It stops at `eos_token_id` (50256 when the config omits it) or at the requested count. The id list includes that eos token; the printed text does not. A UTF-8 character split across tokens stays buffered until the next token completes it, or until generation finishes, and the finished piece is written as soon as the token is chosen. Omit temperature for greedy decoding. A temperature of 0 or less is an error. A positive temperature subtracts the row max before dividing. `top-k` keeps the largest scores before the softmax. A seed is an integer from 0 through 2147483647 and uses a private generator. The prompt plus the new tokens must fit in `min(1024, n_positions)`. `position-embed` takes `#:start` so a cached token uses its absolute position. After the text, `generate` writes `load-seconds`, `load-bytes`, and `tokens-per-second` to the error port.

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
