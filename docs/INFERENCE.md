# Inference reference

## Tensors and kernels

`src/tensor.rkt` stores row-major tensors in one flvector with explicit
dimensions. Slices copy; reshape shares storage. Mutating the data also changes
reshaped views.

`src/nn.rkt` implements:

- `matmul`: Hugging Face Conv1D, `x[n,in] · W[in,out] + b[out]`, with contiguous weight-row reads.
- `matmul-transposed-b`: `h[n,in] · W[out,in]ᵀ` for the tied output head, without a transpose allocation.
- Stable row softmax, biased-variance layer norm (`epsilon = 1e-5`), tanh GELU, and residual addition.

All-masked softmax rows produce zeros. Matrix products below the 100000 work
threshold stay serial. Larger products split across rows at 32 rows or more,
otherwise across output columns. Each worker gets one block.
`current-decode-workers` and `current-prefill-workers` accept `#f` for the host
count or a positive integer capped at that count. These budgets control matrix
products; attention schedules its heads separately.

## Attention and blocks

`c_attn` projects to `[n,3*d]`. Head `h` reads contiguous columns
`[h*d-head,(h+1)*d-head)`. Scores scale by `1/sqrt(d-head)` and include only
visible causal keys. Prefill can run heads in parallel; decode stays serial.

Each request owns a reusable score row per head and preallocated K/V buffers.
`#:kv-rows` limits reads to the filled prefix. Query row `i` has absolute
position `kv-rows - query-rows + i`. Cached blocks read Q from the packed
projection and copy K/V straight into cache. `#:query-width` separates the
logical query width from its stored row stride.

Blocks use pre-LN: `h = x + attn(ln_1(x))`, then `h + mlp(ln_2(h))`.
The MLP projects 768 → 3072, applies tanh GELU, and projects back to 768.
Fresh product buffers also hold GELU and residual results; inputs and returned
traces retain their values.

## Generation

The model adds token and position embeddings, runs the blocks and final norm,
then computes logits with tied token weights. Generation scores only the last
hidden row, prefills once, and processes one new token per step. Cache capacity
fits the prompt plus requested output. `next-token-logits` allocates only the
prompt length. Advancing a cache handle invalidates that handle.

Sampling uses a stable softmax and a bounded top-k heap; ties retain lower IDs.
Seeds use a private generator. EOS defaults to 50256. Returned IDs include EOS;
printed text omits it. Streaming buffers incomplete UTF-8 until completion or
the end of generation. CLI flag values consume the next argument, including
text beginning with `-`; help flags apply only in flag position.

## Tokenizer and model files

The tokenizer uses GPT-2 byte-level BPE, Unicode whitespace, case-sensitive
contractions, and literal `<|endoftext|>` recognition as token 50256.

Model directories contain `config.json`, `vocab.json`, `merges.txt`, and
`model.safetensors`. Assets are pinned to `openai-community/gpt2` revision
`607a30d783dfa663caf39e06633721c8d4cfcd7e`; digests live in
`tools/gpt2-assets.sha256`. The fetch script verifies digests. `load-model`
checks keys and shapes, accepts bare or `transformer.` names, skips attention
mask buffers, and accepts an identical tied `lm_head.weight`.

The safetensors reader decodes little-endian F32 from header offsets, including
unaligned payloads. Arithmetic uses Racket flonums. `load-model` records memory
without collecting; the benchmark collects before measuring resident size.
